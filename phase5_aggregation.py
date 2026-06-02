import os
import glob
import pandas as pd
import polars as pl
import urllib.parse
from datetime import datetime

DATA_DIR = os.getenv("DATA_DIR", "data")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
PARQUET_PEOPLE_DIR = os.path.join(DATA_DIR, "wikidata_parquet_batches")
PARQUET_CS_OCCS_DIR = os.path.join(DATA_DIR, "wikidata_occupations_parquet_batches")

def load_cs_occupations():
    files = glob.glob(os.path.join(PARQUET_CS_OCCS_DIR, "*.parquet"))
    cs_occs = set()
    for f in files:
        df = pd.read_parquet(f, columns=['name_en'])
        cs_occs.update(df['name_en'].dropna().str.lower().tolist())
    return cs_occs

def get_best_occupation(occ_str, cs_occs):
    if pd.isna(occ_str) or not occ_str:
        return "unknown"
    occs = [o.strip().lower() for o in str(occ_str).split("|")]
    for o in occs:
        if o in cs_occs:
            return o
    return occs[0]

def load_people_metadata():
    print("Loading people metadata for Phase 5...")
    cs_occs = load_cs_occupations()
    parquet_files = glob.glob(os.path.join(PARQUET_PEOPLE_DIR, "*.parquet"))
    records = []
    
    for pf in parquet_files:
        df = pd.read_parquet(pf, columns=['enwiki', 'gender', 'occupation_labels', 'birthdate', 'deathdate'])
        df = df.dropna(subset=['enwiki'])
        for _, row in df.iterrows():
            url = row['enwiki']
            if pd.isna(url) or not isinstance(url, str):
                continue
            title_part = url.split("/wiki/")[-1]
            title = urllib.parse.unquote(title_part).replace("_", " ")
            
            birth_year = None
            if not pd.isna(row['birthdate']):
                try:
                    birth_year = int(str(row['birthdate'])[:4])
                except:
                    pass
                    
            death_year = None
            if not pd.isna(row['deathdate']):
                try:
                    death_year = int(str(row['deathdate'])[:4])
                except:
                    pass
                    
            is_dead = death_year is not None
            
            gender_label = "unknown"
            if row['gender'] == "http://www.wikidata.org/entity/Q6581097":
                gender_label = "male"
            elif row['gender'] == "http://www.wikidata.org/entity/Q6581072":
                gender_label = "female"
            
            records.append({
                "person_title": title,
                "gender": gender_label,
                "occupation": get_best_occupation(row['occupation_labels'], cs_occs),
                "birth_year": birth_year,
                "death_year": death_year,
                "is_dead": is_dead
            })
    return pl.DataFrame(records).unique(subset=["person_title"])

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    meta_df = load_people_metadata()
    meta_df.write_parquet(os.path.join(OUTPUT_DIR, "viz_metadata.parquet"))
    
    # Enrich Phase 3 citations
    p3_cites = os.path.join(OUTPUT_DIR, "phase3_citations.parquet")
    if os.path.exists(p3_cites):
        # Use streaming to avoid OOM
        viz_cites = pl.scan_parquet(p3_cites).join(meta_df.lazy(), left_on="cited_person", right_on="person_title", how="left")
        viz_cites.collect(engine="streaming").write_parquet(os.path.join(OUTPUT_DIR, "viz_phase3_citations.parquet"))
        print("Exported viz_phase3_citations.parquet")
        
    # Enrich Phase 4 Domain Ratios
    p4_domains = os.path.join(OUTPUT_DIR, "phase4_domain_ratios.parquet")
    if os.path.exists(p4_domains):
        viz_domains = pl.scan_parquet(p4_domains).join(meta_df.lazy(), left_on="page_title", right_on="person_title", how="left")
        viz_domains.collect(engine="streaming").write_parquet(os.path.join(OUTPUT_DIR, "viz_phase4_domain_ratios.parquet"))
        print("Exported viz_phase4_domain_ratios.parquet")
        
    print("Phase 5 aggregation complete. Ready for visualization.")

if __name__ == "__main__":
    main()
