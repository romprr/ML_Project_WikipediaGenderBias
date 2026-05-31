import requests
import time
import os
import glob
import polars as pl
from io import BytesIO

ENDPOINT = "https://qlever.cs.uni-freiburg.de/api/wikidata"
BATCH_SIZE = 500000
OUTPUT_DIR = "data/wikidata_parquet_batches"

QUERY_TEMPLATE = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT 
  ?person 
  (SAMPLE(?raw_name_en) AS ?name_en)
  (SAMPLE(?raw_name_fr) AS ?name_fr)
  (SAMPLE(?raw_gender) AS ?gender)
  (SAMPLE(?raw_birthdate) AS ?birthdate)
  (SAMPLE(?raw_birthplace) AS ?birthplace)
  (SAMPLE(?raw_deathdate) AS ?deathdate)
  (SAMPLE(?raw_deathplace) AS ?deathplace)
  (GROUP_CONCAT(DISTINCT ?raw_citizenship; separator="|") AS ?citizenships)
  (GROUP_CONCAT(DISTINCT ?raw_occupation; separator="|") AS ?occupations)
  (SAMPLE(?raw_enwiki) AS ?enwiki)
  (SAMPLE(?raw_frwiki) AS ?frwiki)
WHERE {{
  
  # --- 1. INNER QUERY: Find the people and apply the limit FIRST ---
  {{
    SELECT ?person ?raw_enwiki ?raw_frwiki WHERE {{
      ?person wdt:P31 wd:Q5 .
      
      OPTIONAL {{
        ?raw_enwiki schema:about ?person ;
                    schema:isPartOf <https://en.wikipedia.org/> .
      }}
      OPTIONAL {{
        ?raw_frwiki schema:about ?person ;
                        schema:isPartOf <https://fr.wikipedia.org/> .
      }}
      
      FILTER (BOUND(?raw_enwiki) || BOUND(?raw_frwiki))
    }}
    ORDER BY ?person
    LIMIT {limit} OFFSET {offset}
  }}
  
  # --- 2. OUTER QUERY: Fetch metadata ONLY for the people found above ---
  
  # Get the English and French names
  OPTIONAL {{ 
    ?person rdfs:label ?raw_name_en . 
    FILTER(LANG(?raw_name_en) = "en") 
  }}
  OPTIONAL {{ 
    ?person rdfs:label ?raw_name_fr . 
    FILTER(LANG(?raw_name_fr) = "fr") 
  }}
  
  OPTIONAL {{ ?person wdt:P21 ?raw_gender . }}
  OPTIONAL {{ ?person wdt:P569 ?raw_birthdate . }}
  OPTIONAL {{ ?person wdt:P19 ?raw_birthplace . }}
  OPTIONAL {{ ?person wdt:P570 ?raw_deathdate . }}
  OPTIONAL {{ ?person wdt:P20 ?raw_deathplace . }}
  OPTIONAL {{ ?person wdt:P27 ?raw_citizenship . }}
  OPTIONAL {{ ?person wdt:P106 ?raw_occupation . }}
}}
GROUP BY ?person
ORDER BY ?person
"""

def get_starting_offset():
    """Finds the highest existing batch number in the folder to resume."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        return 0
        
    # Find all parquet files in our output directory
    existing_files = glob.glob(os.path.join(OUTPUT_DIR, "wikidata_people_batch_*.parquet"))
    
    if not existing_files:
        return 0
        
    offsets = []
    for filepath in existing_files:
        filename = os.path.basename(filepath)
        try:
            offset_str = filename.replace("wikidata_people_batch_", "").replace(".parquet", "")
            offsets.append(int(offset_str))
        except ValueError:
            continue
            
    if not offsets:
        return 0
        
    # The next batch to fetch is the highest completed offset + BATCH_SIZE
    return max(offsets) + BATCH_SIZE

def download_all_people():
    offset = get_starting_offset()
    
    print(f"Starting/Resuming data extraction from QLever at OFFSET {offset}...")
    
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
                
            # Polars reads bytes much faster than strings. 
            # We wrap the response content in BytesIO for immediate parsing.
            df = pl.read_csv(BytesIO(response.content))
            
            # If the dataframe is empty, we've hit the end of the database
            if df.is_empty():
                print("No more data found. Extraction complete!")
                break
                
            # Save this specific batch as a Parquet file
            filename = os.path.join(OUTPUT_DIR, f"wikidata_people_batch_{offset}.parquet")
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
    download_all_people()