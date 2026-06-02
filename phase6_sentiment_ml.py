import polars as pl
import os
import glob
import random
import re
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

def count_words(text):
    return len(text.split())

def get_sentences_with_citations(text, target_person=None):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    results = []
    
    for sent in sentences:
        matches = re.finditer(r'\[\[(.*?)\]\]', sent)
        for match in matches:
            content = match.group(1)
            cited = content.split('|')[0].strip()
            if target_person and cited != target_person:
                continue
                
            if "#" not in cited and ":" not in cited:
                results.append((cited, sent))
    return results

def main():
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
    DATA_DIR = os.getenv("DATA_DIR", "data")
    
    print("Loading metadata...")
    patched_path = os.path.join(OUTPUT_DIR, "viz_metadata_patched.parquet")
    meta_path = os.path.join(OUTPUT_DIR, "viz_metadata.parquet")
    df_meta = pl.read_parquet(patched_path) if os.path.exists(patched_path) else pl.read_parquet(meta_path)
    
    print("Finding CS Top 40 people...")
    cs_topics = set()
    cs_files = glob.glob(os.path.join(DATA_DIR, "wikidata_cs_parquet_batches", "*.parquet"))
    cs_files.extend(glob.glob(os.path.join(DATA_DIR, "wikidata_computer_science.parquet")))
    for batch in cs_files:
        df = pl.read_parquet(batch)
        if "enwiki" in df.columns:
            urls = df["enwiki"].drop_nulls().to_list()
            titles = [url.split("/wiki/")[-1].replace("_", " ") for url in urls]
            cs_topics.update(titles)
            
    cites_path = os.path.join(OUTPUT_DIR, "phase3_citations.parquet")
    lazy_cites = pl.scan_parquet(cites_path)
    
    cs_cited = lazy_cites.filter(pl.col("source_page").is_in(list(cs_topics))).select("cited_person").unique().collect(engine="streaming")["cited_person"].to_list()
    df_meta = df_meta.with_columns(pl.col("person_title").is_in(cs_cited).alias("cited_in_cs_topic"))
    df_cs_people = df_meta.filter(pl.col("cited_in_cs_topic") == True)
    
    cs_cites = lazy_cites.join(df_cs_people.lazy().select(["person_title", "gender"]), left_on="cited_person", right_on="person_title", how="inner")
    cite_counts = cs_cites.group_by(["cited_person", "gender"]).len().sort("len", descending=True).collect(engine="streaming")
    
    top_men = cite_counts.filter(pl.col("gender") == "male").head(100)['cited_person'].to_list()
    top_women = cite_counts.filter(pl.col("gender") == "female").head(100)['cited_person'].to_list()
    target_people = set(top_men + top_women)
    
    print(f"Extracting ALL citations for the {len(target_people)} top individuals...")
    
    target_cites_pd = cs_cites.filter(pl.col("cited_person").is_in(list(target_people))).select(["source_page", "cited_person", "sentence", "sentiment"]).collect(engine="streaming").to_pandas()
    sampled_cites_list = []
    for person, df_group in target_cites_pd.groupby('cited_person'):
        sampled_cites_list.append(df_group) # Process ALL citations instead of sampling
    import pandas as pd
    if not sampled_cites_list:
        print("No citations found to sample for sentiment analysis!")
        results = []
    else:
        sampled_cites = pd.concat(sampled_cites_list, ignore_index=True)
                    
        print("Loading ML models for Sentences...")
        sent_classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", truncation=True, max_length=512, device=-1)
        event_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", truncation=True, max_length=512, device=-1)
        
        label_map = {"LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE"}
        
        results = []
        
        total = len(sampled_cites)
        print(f"Running ML Sentiment & Event Quality on {total} citations...")
        
        for idx, row in sampled_cites.iterrows():
            src = row['source_page']
            target = row['cited_person']
            sentence = row['sentence']
            
            # Author Tone (RoBERTa)
            res_tone = sent_classifier(sentence)
            sentiment_raw = res_tone[0]["label"]
            tone = label_map.get(sentiment_raw, sentiment_raw)
            
            # Event Quality (SST-2)
            res_event = event_classifier(sentence)
            event_raw = res_event[0]["label"]
            
            results.append({
                "source_page": src,
                "cited_entity": target,
                "sentence": sentence,
                "sentiment": tone,
                "event_quality": event_raw
            })
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{total} citations...")
                
    if results:
        new_df = pl.DataFrame(results)
    else:
        new_df = pl.DataFrame({"source_page": pl.Series(dtype=pl.Utf8), "cited_entity": pl.Series(dtype=pl.Utf8), "sentence": pl.Series(dtype=pl.Utf8), "sentiment": pl.Series(dtype=pl.Utf8), "event_quality": pl.Series(dtype=pl.Utf8)})

    existing_path = os.path.join(OUTPUT_DIR, "top200_ml_sentiment_analysis.parquet")
    if os.path.exists(existing_path):
        existing_df = pl.read_parquet(existing_path)
        existing_df = existing_df.filter(~pl.col("cited_entity").is_in(list(target_people)))
        combined_df = pl.concat([existing_df, new_df], how="diagonal")
        combined_df.write_parquet(existing_path)
        print(f"Appended {len(new_df)} new sentiment records to {existing_path}!")
    else:
        new_df.write_parquet(existing_path)
        print("Created new dataset!")

if __name__ == "__main__":
    main()
