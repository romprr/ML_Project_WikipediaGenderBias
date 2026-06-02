from transformers import pipeline
import polars as pl
import os
import warnings
warnings.filterwarnings("ignore")

def test_robustness():
    classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=-1)
    
    label_map = {
        "LABEL_0": "NEGATIVE",
        "LABEL_1": "NEUTRAL",
        "LABEL_2": "POSITIVE"
    }
    
    test_cases = [
        # Gender bias parity testing (Model should score identically regardless of pronouns)
        ("Torvalds has been criticized for his harsh behavior towards developers on the Linux kernel mailing list.", "NEGATIVE"),
        ("Torvalds has been criticized for her harsh behavior towards developers on the Linux kernel mailing list.", "NEGATIVE"),
        
        ("He is widely regarded as one of the most brilliant mathematicians of his generation.", "POSITIVE"),
        ("She is widely regarded as one of the most brilliant mathematicians of her generation.", "POSITIVE"),
        
        # Complex / Mixed (Wikipedia reality check)
        ("Stallman resigned from MIT and the FSF in 2019 following controversial comments he made.", "NEGATIVE"),
        
        # CS specific domain terms from Wikipedia
        ("Hamilton's software approach was so successful that no software bugs were ever known to have occurred.", "POSITIVE"),
        
        # Neutral Wikipedia Facts
        ("Knuth is the author of the multi-volume work The Art of Computer Programming.", "NEUTRAL"),
        ("Lovelace was the only legitimate child of poet Lord Byron and his wife Lady Byron.", "NEUTRAL")
    ]
    
    print("Running Robustness Tests on cardiffnlp/twitter-roberta-base-sentiment:\n" + "-"*60)
    passed = 0
    for text, expected in test_cases:
        res = classifier(text)[0]
        actual = label_map[res["label"]]
        status = "PASS" if actual == expected else "FAIL"
        if actual == expected: passed += 1
        print(f"{status} Text: '{text}'\n   Expected: {expected} | Got: {actual} (Confidence: {res['score']:.4f})\n")
    
    print(f"Robustness Score: {passed}/{len(test_cases)}\n")

def check_saved_results():
    if not os.path.exists("output/top200_ml_sentiment_analysis.parquet"):
        print("No saved sentiment analysis found.")
        return
        
    df_sent = pl.read_parquet("output/top200_ml_sentiment_analysis.parquet")
    print("-" * 60 + f"\nTotal Saved Sentences in output/top200_ml_sentiment_analysis.parquet: {len(df_sent)}")
    
    meta_path = "output/viz_metadata_patched.parquet" if os.path.exists("output/viz_metadata_patched.parquet") else "output/viz_metadata.parquet"
    df_meta = pl.read_parquet(meta_path)
    
    joined = df_sent.join(df_meta.select(["person_title", "gender"]), left_on="cited_entity", right_on="person_title", how="inner")
    
    counts = joined.group_by("gender").len().to_pandas()
    print("Saved Sentiment Analysis Results by Cited Gender:")
    for _, row in counts.iterrows():
        print(f"- {row['gender'].capitalize()}: {row['len']} sentences evaluated")

if __name__ == "__main__":
    test_robustness()
    check_saved_results()
