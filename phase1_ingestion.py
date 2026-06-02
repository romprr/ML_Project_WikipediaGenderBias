import os
import bz2
import glob
import csv
import urllib.parse
import xml.etree.ElementTree as ET
import pandas as pd
import multiprocessing
import fileinput

DATA_DIR = os.getenv("DATA_DIR", "data")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(DATA_DIR, "extracted_raw_pages"))
PARQUET_PEOPLE_DIR = os.path.join(DATA_DIR, "wikidata_parquet_batches")
PARQUET_CS_DIR = DATA_DIR  # fallback
DUMPS_DIR = os.path.join(DATA_DIR, "wikipedia_text_dumps")
TRANSLATION_TABLE_PATH = os.path.join(DATA_DIR, "translation_table.csv")

def get_target_titles():
    target_titles = set()
    parquet_files = glob.glob(os.path.join(PARQUET_PEOPLE_DIR, "*.parquet")) + \
                    glob.glob(os.path.join(PARQUET_CS_DIR, "*.parquet"))
    for pf in parquet_files:
        df = pd.read_parquet(pf, columns=['enwiki'])
        df = df.dropna(subset=['enwiki'])
        for url in df['enwiki']:
            if pd.isna(url) or not isinstance(url, str):
                continue
            title_part = url.split("/wiki/")[-1]
            title = urllib.parse.unquote(title_part).replace("_", " ")
            target_titles.add(title)
    return frozenset(target_titles)

def sanitize_filename(title):
    return title.replace("/", "_").replace("\0", "")

def process_single_dump(args):
    dump_file, target_titles = args
    print(f"[{os.getpid()}] Processing dump: {dump_file}")
    
    local_translations = []
    
    try:
        with bz2.open(dump_file, "rt", encoding="utf-8") as file:
            context = ET.iterparse(file, events=("end",))
            for event, elem in context:
                tag = elem.tag.split('}')[-1]
                if tag == "page":
                    ns_elem = elem.find(".//{*}ns")
                    if ns_elem is not None and ns_elem.text == "0":
                        title_elem = elem.find(".//{*}title")
                        title = title_elem.text if title_elem is not None else "Unknown"
                        if title in target_titles:
                            id_elem = elem.find(".//{*}id")
                            page_id = id_elem.text if id_elem is not None else "Unknown"
                            text_elem = elem.find(".//{*}text")
                            raw_text = text_elem.text if text_elem is not None and text_elem.text else ""
                            
                            has_redirect_tag = elem.find(".//{*}redirect") is not None
                            has_redirect_text = raw_text.strip().lower().startswith("#redirect")
                            
                            if not has_redirect_tag and not has_redirect_text:
                                safe_title = sanitize_filename(title)
                                output_filename = os.path.join(OUTPUT_DIR, f"{page_id}_{safe_title}.txt")
                                with open(output_filename, "w", encoding="utf-8") as out_file:
                                    out_file.write(raw_text)
                                local_translations.append([page_id, title, output_filename])
                    elem.clear()
    except Exception as e:
        print(f"[{os.getpid()}] Warning: Error reading {dump_file} (may be partially processed): {e}")
    
    # Save partial translations to a temporary CSV
    tmp_csv = f"{dump_file}.tmp.csv"
    with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(local_translations)
        
    print(f"[{os.getpid()}] Finished {dump_file}")
    return tmp_csv

def main():
    print("Loading target titles from parquet...")
    targets = get_target_titles()
    print(f"Found {len(targets)} target titles.")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    dump_files = glob.glob(os.path.join(DUMPS_DIR, "*.xml.bz2"))
    
    # We use multiprocessing to process dumps in parallel
    pool_args = [(df, targets) for df in dump_files]
    print(f"Starting multiprocessing pool with {len(dump_files)} workers...")
    
    with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), len(dump_files))) as pool:
        tmp_csvs = pool.map(process_single_dump, pool_args)
        
    print("Consolidating translation table...")
    with open(TRANSLATION_TABLE_PATH, "w", newline="", encoding="utf-8") as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(["page_id", "title", "disk_path"])
        for tmp_csv in tmp_csvs:
            if os.path.exists(tmp_csv):
                with open(tmp_csv, "r", encoding="utf-8") as in_csv:
                    for line in in_csv:
                        out_csv.write(line)
                os.remove(tmp_csv)
                
    print("Phase 1 complete.")

if __name__ == "__main__":
    main()
