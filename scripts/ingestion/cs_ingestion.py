import requests
import time
import os
import glob
import polars as pl
from io import BytesIO

ENDPOINT = "https://qlever.cs.uni-freiburg.de/api/wikidata"
BATCH_SIZE = 500000
# Changed the output directory so it doesn't mix with your people data
OUTPUT_DIR = "data/wikidata_cs_parquet_batches"

# The updated CS Query
QUERY_TEMPLATE = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT 
  ?item 
  (SAMPLE(?raw_name_en) AS ?name_en)
  (SAMPLE(?raw_name_fr) AS ?name_fr)
  (SAMPLE(?raw_enwiki) AS ?enwiki)
  (SAMPLE(?raw_frwiki) AS ?frwiki)
WHERE {{
  
  # --- 1. INNER QUERY: Find the CS topics and apply the limit FIRST ---
  {{
    SELECT DISTINCT ?item ?raw_enwiki ?raw_frwiki WHERE {{
      # Find anything that is a subclass of (P279) OR part of (P361) Computer Science (Q21198)
      # The asterisk (*) means "traverse all the way down the tree"
      ?item (wdt:P279* | wdt:P361*) wd:Q21198 .
      
      OPTIONAL {{
        ?raw_enwiki schema:about ?item ;
                    schema:isPartOf <https://en.wikipedia.org/> .
      }}
      OPTIONAL {{
        ?raw_frwiki schema:about ?item ;
                        schema:isPartOf <https://fr.wikipedia.org/> .
      }}
      
      FILTER (BOUND(?raw_enwiki) || BOUND(?raw_frwiki))
    }}
    ORDER BY ?item
    LIMIT {limit} OFFSET {offset}
  }}
  
  # --- 2. OUTER QUERY: Fetch names ONLY for the topics found above ---
  OPTIONAL {{ 
    ?item rdfs:label ?raw_name_en . 
    FILTER(LANG(?raw_name_en) = "en") 
  }}
  OPTIONAL {{ 
    ?item rdfs:label ?raw_name_fr . 
    FILTER(LANG(?raw_name_fr) = "fr") 
  }}
}}
GROUP BY ?item
ORDER BY ?item
"""

def get_starting_offset():
    """Finds the highest existing batch number in the folder to resume."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        return 0
        
    # Updated file pattern
    existing_files = glob.glob(os.path.join(OUTPUT_DIR, "wikidata_cs_batch_*.parquet"))
    
    if not existing_files:
        return 0
        
    offsets = []
    for filepath in existing_files:
        filename = os.path.basename(filepath)
        try:
            # Updated string replacement
            offset_str = filename.replace("wikidata_cs_batch_", "").replace(".parquet", "")
            offsets.append(int(offset_str))
        except ValueError:
            continue
            
    if not offsets:
        return 0
        
    return max(offsets) + BATCH_SIZE

def download_all_cs_topics():
    offset = get_starting_offset()
    
    print(f"Starting/Resuming CS topics extraction from QLever at OFFSET {offset}...")
    
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
                print("No more CS data found. Extraction complete!")
                break
                
            # Updated output filename
            filename = os.path.join(OUTPUT_DIR, f"wikidata_cs_batch_{offset}.parquet")
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
    download_all_cs_topics()