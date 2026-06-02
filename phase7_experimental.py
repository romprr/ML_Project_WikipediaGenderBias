import os
import glob
import polars as pl
import re
from collections import Counter
import time
import warnings
warnings.filterwarnings("ignore")

import spacy
from gensim.models import Word2Vec

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
PHASE2_DIR = os.environ.get("PHASE2_PARQUET_DIR", "data")

def get_cs_targets():
    """Returns a dictionary of {page_title: gender} for CS people"""
    meta_path = os.path.join(OUTPUT_DIR, "viz_metadata_patched.parquet")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(OUTPUT_DIR, "viz_metadata.parquet")

    df_meta = pl.read_parquet(meta_path)

    cs_pattern = r"(?i)(computer|software|programmer|informatician|cryptographer|hacker|developer|data scientist|machine learning|artificial intelligence|cybersecurity|algorithm|systems engineer|network engineer)"

    df_cs = df_meta.filter(
        pl.col("occupation").str.contains(cs_pattern) &
        pl.col("gender").is_in(["male", "female"])
    )

    print(f"CS filter matched {len(df_cs)} people "
          f"({len(df_cs.filter(pl.col('gender') == 'male'))} men, "
          f"{len(df_cs.filter(pl.col('gender') == 'female'))} women)")
    n_male   = len(df_cs.filter(pl.col("gender") == "male"))
    n_female = len(df_cs.filter(pl.col("gender") == "female"))
    title_gender = dict(zip(df_cs["person_title"].to_list(), df_cs["gender"].to_list()))
    return title_gender, n_male, n_female

def clean_wikitext(text):
    """Strip basic wiki markup"""
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'\{\{[^}]+\}\}', '', text)
    text = re.sub(r"'{2,}", '', text)
    return text

def main():
    print("Starting Phase 7: Advanced NLP & Computational Linguistics")
    print("Loading spaCy model (en_core_web_sm)...")
    # disable unused components for speed
    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    nlp.max_length = 2_000_000

    print("Loading CS targets...")
    targets, n_male, n_female = get_cs_targets()
    print(f"Found {len(targets)} CS biographies (normalizing per 1,000 people).\n")

    parquet_files = sorted(glob.glob(os.path.join(PHASE2_DIR, "phase2_parsed_data_part_*.parquet")))
    print(f"Scanning {len(parquet_files)} text chunk files...")

    parquet_files = sorted(glob.glob(os.path.join(PHASE2_DIR, "phase2_parsed_data_part_*.parquet")))
    print(f"Scanning {len(parquet_files)} text chunk files...")

    male_adjs   = Counter()
    female_adjs = Counter()
    male_orgs   = Counter()
    female_orgs = Counter()
    corpus_sentences = []

    start = time.time()
    total_chunks = len(parquet_files)

    for idx, chunk_path in enumerate(parquet_files):
        if idx % 50 == 0:
            elapsed = time.time() - start
            print(f"  Processing chunk {idx}/{total_chunks} ... ({elapsed:.0f}s elapsed)")

        try:
            df = pl.read_parquet(chunk_path)
        except Exception:
            continue

        # Collect rows belonging to our CS targets
        rows = [(row["page_title"], row.get("content", "") or "")
                for row in df.iter_rows(named=True)
                if row.get("page_title") in targets]

        if not rows:
            continue

        for title, raw_text in rows:
            gender = targets[title]
            text = clean_wikitext(raw_text)
            if not text.strip():
                continue

            doc = nlp(text)
            pronouns = (["he", "his", "him"] if gender == "male"
                        else ["she", "her", "hers"])

            #  Word2Vec corpus 
            for sent in doc.sents:
                tokens = [t.text.lower() for t in sent if t.is_alpha and len(t.text) > 1]
                if tokens:
                    corpus_sentences.append(tokens)

            #  NER 
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    if gender == "male":
                        male_orgs[ent.text] += 1
                    else:
                        female_orgs[ent.text] += 1

            #  Adjective proximity 
            for token in doc:
                if token.pos_ == "ADJ":
                    w_start = max(0, token.i - 5)
                    w_end   = min(len(doc), token.i + 6)
                    ctx = [t.text.lower() for t in doc[w_start:w_end]]
                    if any(p in ctx for p in pronouns):
                        if gender == "male":
                            male_adjs[token.text.lower()] += 1
                        else:
                            female_adjs[token.text.lower()] += 1

    print(f"\nText processing done in {time.time() - start:.1f}s\n")

    print("=" * 55)
    print("1. Adjective Proximity Results (Top 15 per gender)")
    print("=" * 55)
    print("Male top adjectives  :", [w for w, _ in male_adjs.most_common(15)])
    print("Female top adjectives:", [w for w, _ in female_adjs.most_common(15)])

    print()
    print("=" * 55)
    print("2. Named Entity Recognition - Top 10 ORGs per gender")
    print("=" * 55)
    print("Male top organizations  :", [w for w, _ in male_orgs.most_common(10)])
    print("Female top organizations:", [w for w, _ in female_orgs.most_common(10)])

    print()
    print("=" * 55)
    print("3. Word2Vec Language Bias Scoring")
    print("=" * 55)
    print(f"Training Word2Vec on {len(corpus_sentences):,} sentences...")
    w2v_start = time.time()
    
    # Adjust min_count for tiny test datasets to prevent vocabulary build errors
    min_count = 1 if len(corpus_sentences) < 100 else 10
    
    model = Word2Vec(
        sentences=corpus_sentences,
        vector_size=100,
        window=5,
        min_count=min_count,
        workers=max(1, os.cpu_count() - 1),
        epochs=5
    )
    print(f"Word2Vec trained in {time.time() - w2v_start:.1f}s\n")

    male_pronouns   = ["he", "him"]
    female_pronouns = ["she", "her"]

    test_words = [
        "genius", "leader", "pioneer", "visionary", "brilliant",
        "assistant", "secretary", "emotional", "supportive",
        "logical", "innovative", "aggressive", "dedicated"
    ]

    print(f"{'Word':<15} {'Male sim':>9} {'Female sim':>11} {'Bias':>16}")
    print("-" * 55)
    for word in test_words:
        if word not in model.wv:
            print(f"'{word}' not in vocabulary, skipping.")
            continue
        m_sims = [model.wv.similarity(word, p) for p in male_pronouns   if p in model.wv]
        f_sims = [model.wv.similarity(word, p) for p in female_pronouns if p in model.wv]
        if not m_sims or not f_sims:
            continue
        m_score = sum(m_sims) / len(m_sims)
        f_score = sum(f_sims) / len(f_sims)
        diff    = m_score - f_score
        bias    = "Male skewed" if diff > 0.005 else ("Female skewed" if diff < -0.005 else "Neutral")
        print(f"{word:<15} {m_score:>9.4f} {f_score:>11.4f} {bias:>16}")

    # Save results to parquet for Streamlit
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Adjectives - normalize to % of people in each group who have this adjective nearby
    adj_rows = []
    all_adj_words = set(list(dict(male_adjs.most_common(30)).keys()) + list(dict(female_adjs.most_common(30)).keys()))
    for word in all_adj_words:
        adj_rows.append({
            "word": word,
            "male_count":   male_adjs.get(word, 0),
            "female_count": female_adjs.get(word, 0),
            "male_pct":   round(male_adjs.get(word, 0)   / n_male   * 100, 3) if n_male > 0 else 0,
            "female_pct": round(female_adjs.get(word, 0) / n_female * 100, 3) if n_female > 0 else 0,
        })
    pl.DataFrame(adj_rows).write_parquet(os.path.join(OUTPUT_DIR, "phase7_adjectives.parquet"))

    # Organizations - normalize to % of people in each group
    org_rows = []
    all_org_words = set(list(dict(male_orgs.most_common(20)).keys()) + list(dict(female_orgs.most_common(20)).keys()))
    for org in all_org_words:
        org_rows.append({
            "org": org,
            "male_count":   male_orgs.get(org, 0),
            "female_count": female_orgs.get(org, 0),
            "male_pct":   round(male_orgs.get(org, 0)   / n_male   * 100, 3) if n_male > 0 else 0,
            "female_pct": round(female_orgs.get(org, 0) / n_female * 100, 3) if n_female > 0 else 0,
        })
    pl.DataFrame(org_rows).write_parquet(os.path.join(OUTPUT_DIR, "phase7_organizations.parquet"))

    # Word2Vec bias
    bias_rows = []
    for word in test_words:
        if word not in model.wv:
            continue
        m_sims = [model.wv.similarity(word, p) for p in male_pronouns   if p in model.wv]
        f_sims = [model.wv.similarity(word, p) for p in female_pronouns if p in model.wv]
        if not m_sims or not f_sims:
            continue
        m_score = sum(m_sims) / len(m_sims)
        f_score = sum(f_sims) / len(f_sims)
        diff    = m_score - f_score
        bias    = "Male skewed" if diff > 0.005 else ("Female skewed" if diff < -0.005 else "Neutral")
        bias_rows.append({"word": word, "male_sim": round(m_score, 4), "female_sim": round(f_score, 4), "diff": round(diff, 4), "bias": bias})
    pl.DataFrame(bias_rows).write_parquet(os.path.join(OUTPUT_DIR, "phase7_word2vec_bias.parquet"))

    print("Results saved to output/phase7_*.parquet")
    print(f"\nPhase 7 complete! Total time: {time.time() - start:.1f}s")

if __name__ == "__main__":
    main()
