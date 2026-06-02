import os
import csv
import re
import polars as pl
import multiprocessing

DATA_DIR = os.getenv("DATA_DIR", "data")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data")
RAW_PAGES_DIR = os.path.join(DATA_DIR, "extracted_raw_pages")
OUTPUT_PARQUET_DIR = OUTPUT_DIR
TRANSLATION_TABLE = os.path.join(DATA_DIR, "translation_table.csv")

def count_words(text):
    return len(re.findall(r'\b\w+\b', text))

def count_letters(text):
    return len([c for c in text if c.isalpha()])

def count_sentences(text):
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])

def parse_wikitext(text):
    sections = []
    current_section = "Lead"
    current_level = 1
    current_content = []
    
    for line in text.split('\n'):
        match = re.match(r'^(={2,6})\s*(.+?)\s*\1$', line.strip())
        if match:
            sections.append({
                "section": current_section,
                "level": current_level,
                "content": "\n".join(current_content).strip()
            })
            current_level = len(match.group(1))
            current_section = match.group(2)
            current_content = []
        else:
            current_content.append(line)
            
    sections.append({
        "section": current_section,
        "level": current_level,
        "content": "\n".join(current_content).strip()
    })
    
    return [s for s in sections if s["content"]]

def heuristic_classify(text):
    text_lower = text.lower()
    if any(w in text_lower for w in ["born", "early life", "career", "personal life", "family"]):
        return "biography"
    if any(w in text_lower for w in ["work", "publication", "album", "film", "book", "discography"]):
        return "work"
    if any(w in text_lower for w in ["history", "background", "century", "war", "dynasty"]):
        return "history"
    if any(w in text_lower for w in ["geography", "location", "climate", "demographics", "city"]):
        return "geography"
    if any(w in text_lower for w in ["science", "research", "theory", "discovery", "mathematics"]):
        return "science"
    return "other"

def process_page_batch(batch_tuple):
    batch_idx, batch_pages = batch_tuple
    parsed_batch = []
    
    for page in batch_pages:
        if not os.path.exists(page["disk_path"]):
            continue
        try:
            with open(page["disk_path"], "r", encoding="utf-8") as f:
                raw_text = f.read()
        except Exception:
            continue
            
        sections = parse_wikitext(raw_text)
        
        for sec in sections:
            content = sec["content"]
            words_cnt = count_words(content)
            
            label = "other"
            if words_cnt >= 5:
                # Fast heuristic classification
                label = heuristic_classify(f"Section: {sec['section']}. {content[:300]}")
            
            parsed_batch.append({
                "page_id": page["page_id"],
                "page_title": page["title"],
                "section_title": sec["section"],
                "section_level": sec["level"],
                "content": content,
                "classification": label,
                "words": words_cnt,
                "letters": count_letters(content),
                "sentences": count_sentences(content)
            })
            
    if parsed_batch:
        df = pl.DataFrame(parsed_batch)
        out_file = os.path.join(OUTPUT_PARQUET_DIR, f"phase2_parsed_data_part_{batch_idx}.parquet")
        df.write_parquet(out_file)
        
    print(f"[{os.getpid()}] Finished batch {batch_idx}")
    return batch_idx

def main():
    if not os.path.exists(TRANSLATION_TABLE):
        print("Translation table not found.")
        return
        
    pages = []
    with open(TRANSLATION_TABLE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pages.append(row)
            
    print(f"Loaded {len(pages)} pages to process. Processing ALL pages!")
    os.makedirs(OUTPUT_PARQUET_DIR, exist_ok=True)
    
    batch_size = 5000
    batches = []
    for i in range(0, len(pages), batch_size):
        batches.append((i // batch_size, pages[i:i + batch_size]))
        
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"Starting multiprocessing pool with {num_workers} workers for {len(batches)} batches using fast heuristics...")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(process_page_batch, batches)

    print("Phase 2 complete!")

if __name__ == "__main__":
    main()
