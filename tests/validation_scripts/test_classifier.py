import os
import random
import polars as pl

OUTPUT_PARQUET = "data/phase2_parsed_data.parquet"

def test_classifier(num_samples=5):
    import glob
    p2_files = glob.glob("data/phase2_parsed_data_part_*.parquet")
    if not p2_files:
        print("Data files not found. Please run phase 2 script.")
        return
        
    print(f"Loading data from Phase 2 parts...")
    dfs = [pl.read_parquet(f) for f in p2_files]
    df = pl.concat(dfs)
    
    total_sections = df.height
    print(f"Total sections available: {total_sections}")
    
    if total_sections == 0:
        print("No sections to sample.")
        return
        
    samples = df.sample(n=min(num_samples, total_sections))
    
    for i, row in enumerate(samples.iter_rows(named=True)):
        print("="*50)
        print(f"Sample {i+1}")
        print(f"Page Title: {row['page_title']}")
        print(f"Section Title: {row['section_title']}")
        print(f"Predicted Label: ** {row['classification']} **")
        print("-"*50)
        content_preview = row['content'][:500] + ("..." if len(row['content']) > 500 else "")
        print(f"Content Preview:\n{content_preview}")
        print("="*50)
        print()

if __name__ == "__main__":
    test_classifier(10)
