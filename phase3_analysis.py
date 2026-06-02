import os
import glob
import re
import csv
import urllib.parse
import polars as pl
import pandas as pd
import multiprocessing

DATA_DIR = os.getenv("DATA_DIR", "data")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
TARGET_TITLES_PARQUET = [os.path.join(DATA_DIR, "wikidata_computer_science.parquet")]
PARQUET_PEOPLE_DIR = os.path.join(DATA_DIR, "wikidata_parquet_batches")
PHASE2_PARQUET_DIR = os.getenv("PHASE2_PARQUET_DIR", "data")
TRANSLATION_TABLE = os.getenv("TRANSLATION_TABLE", os.path.join(DATA_DIR, "translation_table.csv"))

POSITIVE_WORDS = {"good", "great", "excellent", "amazing", "beautiful", "best", "brilliant", "success", "innovative", "won", "award", "praise", "popular"}
NEGATIVE_WORDS = {"bad", "worst", "terrible", "awful", "failed", "died", "decline", "poor", "criticized", "controversy", "scandal", "lost", "death"}

# Globals for workers
people_meta = {}
target_people = set()

def load_people_metadata():
    parquet_files = glob.glob(os.path.join(PARQUET_PEOPLE_DIR, "*.parquet"))
    meta = {}
    for pf in parquet_files:
        df = pd.read_parquet(pf, columns=['enwiki', 'gender'])
        df = df.dropna(subset=['enwiki'])
        for _, row in df.iterrows():
            url = row['enwiki']
            gender = row['gender']
            if pd.isna(url) or not isinstance(url, str):
                continue
            title_part = url.split("/wiki/")[-1]
            title = urllib.parse.unquote(title_part).replace("_", " ")
            gender_label = "unknown"
            if gender == "http://www.wikidata.org/entity/Q6581097":
                gender_label = "male"
            elif gender == "http://www.wikidata.org/entity/Q6581072":
                gender_label = "female"
            meta[title] = {"gender": gender_label}
    return meta

def init_worker():
    global people_meta
    global target_people
    print(f"[{os.getpid()}] Worker loading people meta...")
    people_meta = load_people_metadata()
    target_people = set(people_meta.keys())
    print(f"[{os.getpid()}] Worker ready.")

def get_sentences_with_citations(text, target_titles):
    citations = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        links = re.findall(r'\[\[([^\|\]]+)(?:\|[^\]]+)?\]\]', sentence)
        for link in links:
            link_clean = link.split('#')[0].strip()
            if link_clean in target_titles:
                citations.append((link_clean, sentence))
    return citations

def heuristic_sentiment(text):
    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))
    pos_count = len(words.intersection(POSITIVE_WORDS))
    neg_count = len(words.intersection(NEGATIVE_WORDS))
    if pos_count > neg_count:
        return "POSITIVE"
    elif neg_count > pos_count:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def process_page_batch(batch_tuple):
    batch_idx, batch_pages = batch_tuple
    global people_meta
    global target_people
    
    citation_records = []
    
    for page in batch_pages:
        disk_path = page["disk_path"]
        if not os.path.exists(disk_path):
            continue
        try:
            with open(disk_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        except Exception:
            continue
            
        cites = get_sentences_with_citations(raw_text, target_people)
        unique_cited_in_page = set()
        for cite_title, sentence in cites:
            if cite_title not in unique_cited_in_page:
                unique_cited_in_page.add(cite_title)
            
            citation_records.append({
                "source_page": page["title"],
                "cited_person": cite_title,
                "cited_gender": people_meta[cite_title]["gender"],
                "sentence": sentence,
                "sentiment": heuristic_sentiment(sentence)
            })
            
    if citation_records:
        df_cites = pl.DataFrame(citation_records)
        out_file = os.path.join(OUTPUT_DIR, f"phase3_cites_part_{batch_idx}.parquet")
        df_cites.write_parquet(out_file)
        
    print(f"[{os.getpid()}] Finished Phase 3 batch {batch_idx}")
    return batch_idx

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    pages = []
    if os.path.exists(TRANSLATION_TABLE):
        with open(TRANSLATION_TABLE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pages.append(row)
                
    batch_size = 5000
    batches = []
    for i in range(0, len(pages), batch_size):
        batches.append((i // batch_size, pages[i:i + batch_size]))
        
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"Starting Phase 3 multiprocessing pool for {len(batches)} batches...")
    
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker) as pool:
        pool.map(process_page_batch, batches)
        
    # Merge Phase 3 citation parquets
    cites_files = glob.glob(os.path.join(OUTPUT_DIR, "phase3_cites_part_*.parquet"))
    if cites_files:
        df_cites = pl.concat([pl.read_parquet(f) for f in cites_files])
        unique_cites = df_cites.unique(subset=["source_page", "cited_person"])
        unique_cites.write_parquet(os.path.join(OUTPUT_DIR, "phase3_citations.parquet"))
        for f in cites_files:
            os.remove(f)
        
    # Join sections
    p2_files = glob.glob(os.path.join(PHASE2_PARQUET_DIR, "phase2_parsed_data_part_*.parquet"))
    if p2_files:
        print("Loading people metadata in main process for joining...")
        pmeta = load_people_metadata()
        all_sec_stats = []
        gender_df = pl.DataFrame([{"page_title": k, "gender": v["gender"]} for k,v in pmeta.items()])
        
        for p2_file in p2_files:
            df_sec = pl.read_parquet(p2_file)
            df_sec = df_sec.join(gender_df, on="page_title", how="left")
            sec_stats = df_sec.filter(pl.col("classification").is_in(["biography", "work"]))
            all_sec_stats.append(sec_stats)
            
        if all_sec_stats:
            combined_sec_stats = pl.concat(all_sec_stats)
            combined_sec_stats.write_parquet(os.path.join(OUTPUT_DIR, "phase3_sections.parquet"))
            
    print("Phase 3 complete!")

if __name__ == "__main__":
    main()
