import os
import glob
import pandas as pd
import polars as pl
import urllib.parse

DATA_DIR = os.getenv("DATA_DIR", "data")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
PARQUET_PEOPLE_DIR = os.path.join(DATA_DIR, "wikidata_parquet_batches")
PARQUET_CS_OCCS_DIR = os.path.join(DATA_DIR, "wikidata_occupations_parquet_batches")
PHASE2_PARQUET_DIR = os.getenv("PHASE2_PARQUET_DIR", "data")
PHASE3_CITES_PARQUET = os.path.join(OUTPUT_DIR, "phase3_citations.parquet")

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
    # Prefer CS occupations
    for o in occs:
        if o in cs_occs:
            return o
    return occs[0] # Fallback to first

def load_people_metadata():
    cs_occs = load_cs_occupations()
    parquet_files = glob.glob(os.path.join(PARQUET_PEOPLE_DIR, "*.parquet"))
    people_meta = {}
    
    for pf in parquet_files:
        df = pd.read_parquet(pf, columns=['enwiki', 'gender', 'occupation_labels', 'birthdate', 'deathdate'])
        df = df.dropna(subset=['enwiki'])
        for _, row in df.iterrows():
            url = str(row['enwiki'])
            if pd.isna(url):
                continue
            title_part = url.split("/wiki/")[-1]
            title = urllib.parse.unquote(title_part).replace("_", " ")
            people_meta[title] = {
                "gender": str(row['gender']).split('/')[-1] if not pd.isna(row['gender']) else "unknown",
                "occupations": get_best_occupation(row['occupation_labels'], cs_occs),
                "birthdate": str(row['birthdate']).split('T')[0] if not pd.isna(row['birthdate']) else None,
                "deathdate": str(row['deathdate']).split('T')[0] if not pd.isna(row['deathdate']) else None
            }
    return people_meta

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading metadata...")
    people_meta = load_people_metadata()
    
    if os.path.exists(PHASE3_CITES_PARQUET):
        print("Processing Phase 3 citations lazily...")
        occ_df = pl.DataFrame([{"cited_person": k, "occupation": v["occupations"], "gender": v["gender"]} for k,v in people_meta.items()]).lazy()
        
        df_cites = pl.scan_parquet(PHASE3_CITES_PARQUET)
        
        unique_cites = df_cites.join(occ_df, on="cited_person", how="left").unique(subset=["source_page", "cited_person"])
        
        occ_impact = unique_cites.group_by("occupation").len().sort("len", descending=True).collect(engine="streaming")
        occ_impact.write_parquet(os.path.join(OUTPUT_DIR, "phase4_occupation_impact.parquet"))
        
        occ_gender_impact = unique_cites.group_by(["occupation", "cited_gender"]).len().sort("len", descending=True).collect(engine="streaming")
        occ_gender_impact.write_parquet(os.path.join(OUTPUT_DIR, "phase4_occupation_gender_impact.parquet"))
        print("Exported Phase 4 impact metrics.")
    
    p2_files = glob.glob(os.path.join(PHASE2_PARQUET_DIR, "phase2_parsed_data_part_*.parquet"))
    if p2_files:
        print(f"Running Native Polars Analysis on {len(p2_files)} Phase 2 parts...")
        
        domain_expr = (
            pl.when(pl.col("content").str.to_lowercase().str.contains(r"software|algorithm|computer|programming|data|internet"))
            .then(pl.lit("Computer Science"))
            .when(pl.col("content").str.to_lowercase().str.contains(r"election|government|policy|law|president|minister"))
            .then(pl.lit("Politics"))
            .when(pl.col("content").str.to_lowercase().str.contains(r"company|market|revenue|business|ceo|finance"))
            .then(pl.lit("Business"))
            .when(pl.col("content").str.to_lowercase().str.contains(r"born|life|career|died|biography"))
            .then(pl.lit("General Biography"))
            .otherwise(pl.lit("Other"))
        )
        
        df = pl.scan_parquet(p2_files)
        df = df.with_columns(domain_classification=domain_expr)
        
        # Group and aggregate words lazily, then collect
        final_domain_counts = df.group_by(["page_title", "domain_classification"]).agg(pl.col("words").sum().alias("total_words")).collect(engine="streaming")
        final_domain_counts.write_parquet(os.path.join(OUTPUT_DIR, "phase4_domain_ratios.parquet"))
        print("Saved domain ratios.")
            
    print("Phase 4 complete!")

if __name__ == "__main__":
    main()
