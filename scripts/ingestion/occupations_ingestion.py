import requests
import time
import os
import glob
import polars as pl
from io import BytesIO

ENDPOINT = "https://qlever.cs.uni-freiburg.de/api/wikidata"
BATCH_SIZE = 500000
# Changed the output directory to reflect occupations
OUTPUT_DIR = "data/wikidata_occupations_parquet_batches"

# The updated Occupations Query
# Note: All SPARQL curly braces are doubled ({{ }}) so Python's .format() doesn't crash.
QUERY_TEMPLATE = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?occupation ?name_en WHERE {{
  # Explicit UNIONs prevent QLever from doing a full database table scan
  {{
    {{ ?occupation wdt:P279* wd:Q82594 . }}          # computer scientist
    UNION {{ ?occupation wdt:P279* wd:Q170790 . }}   # mathematician
    UNION {{ ?occupation wdt:P279* wd:Q5482740 . }}  # programmer
    UNION {{ ?occupation wdt:P279* wd:Q8093228 . }}  # software engineer
    UNION {{ ?occupation wdt:P279* wd:Q1650130 . }}  # statistician
    UNION {{ ?occupation wdt:P279* wd:Q14562208 . }} # data scientist
    UNION {{ ?occupation wdt:P279* wd:Q20966453 . }} # artificial intelligence researcher
    UNION {{ ?occupation wdt:P279* wd:Q104868516 . }}# computing occupation
    UNION {{ ?occupation wdt:P279* wd:Q11357211 . }} # web developer
    UNION {{ ?occupation wdt:P279* wd:Q17596041 . }} # computer engineer
    UNION {{ ?occupation wdt:P279* wd:Q5266851 . }}  # database administrator
    UNION {{ ?occupation wdt:P279* wd:Q3275037 . }}  # computer technician
  }}

  # Block Wikidata data-entry errors (where humans are accidentally marked as subclasses of jobs)
  MINUS {{ ?occupation wdt:P31 wd:Q5 . }}

  # Fetch the English label explicitly
  OPTIONAL {{
    ?occupation rdfs:label ?name_en .
    FILTER(LANG(?name_en) = "en")
  }}
}}
ORDER BY ?occupation
LIMIT {limit}
OFFSET {offset}
"""

def get_starting_offset():
    """Finds the highest existing batch number in the folder to resume."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        return 0
        
    # Updated file pattern
    existing_files = glob.glob(os.path.join(OUTPUT_DIR, "wikidata_occupations_batch_*.parquet"))
    
    if not existing_files:
        return 0
        
    offsets = []
    for filepath in existing_files:
        filename = os.path.basename(filepath)
        try:
            # Updated string replacement
            offset_str = filename.replace("wikidata_occupations_batch_", "").replace(".parquet", "")
            offsets.append(int(offset_str))
        except ValueError:
            continue
            
    if not offsets:
        return 0
        
    return max(offsets) + BATCH_SIZE

def download_all_occupations():
    offset = get_starting_offset()
    
    print(f"Starting/Resuming occupations extraction from QLever at OFFSET {offset}...")
    
    while True:
        print(f"Fetching batch: OFFSET {offset} to {offset + BATCH_SIZE}...")
        query = QUERY_TEMPLATE.format(limit=BATCH_SIZE, offset=offset)
        
        try:
            response = requests.post(
                ENDPOINT,
                data={"query": query},
                headers={"Accept": "text/csv"},
                timeout=120 
            )
            
            if response.status_code != 200:
                print(f"Server error {response.status_code}: {response.text}")
                print("Waiting 10 seconds and retrying...")
                time.sleep(10)
                continue 
                
            df = pl.read_csv(BytesIO(response.content))
            
            if df.is_empty():
                print("No more occupations data found. Extraction complete!")
                break
                
            # Updated output filename
            filename = os.path.join(OUTPUT_DIR, f"wikidata_occupations_batch_{offset}.parquet")
            df.write_parquet(filename, compression="zstd")
            
            print(f"Successfully saved {df.height} rows to {filename}")
            
            offset += BATCH_SIZE
            time.sleep(2)
            
        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")
            print("Waiting 10 seconds and retrying...")
            time.sleep(10)
        except pl.exceptions.NoDataError:
            print("Received empty data from server, stopping.")
            break

if __name__ == "__main__":
    download_all_occupations()