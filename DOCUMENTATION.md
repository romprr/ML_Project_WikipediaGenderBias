# Architecture & Code Documentation

This document provides a technical overview of the project's architecture, key functions, and overall data flow.

##  Architecture Overview

The pipeline is intentionally modular. It processes large Wikipedia dumps incrementally using PyArrow/Polars to handle out-of-memory tabular data efficiently. 

The process is split into 7 phases (Phase 0 to Phase 6) plus a visualization interface.

---

##  Project Structure

- `phase*.py`: The sequential core pipeline scripts.
- `streamlit_app.py`: The user-facing frontend dashboard.
- `tests/`: Contains `test_pipeline.py` for end-to-end integration testing and validation scripts.
- `scripts/ingestion/`: Contains the Phase 0 SPARQL Wikidata download scripts.

---

##  Core Pipeline Scripts

### 1. `phase0_ingestion.py`
A wrapper script that sequentially executes:
- `scripts/ingestion/people_ingestion.py`: Uses `SPARQLWrapper` to download biographical metadata (birthdates, gender) for all humans on Wikidata.
- `scripts/ingestion/cs_ingestion.py`: Uses SPARQL to find all Computer Science related pages/topics on Wikidata.
- `scripts/ingestion/occupations_ingestion.py`: Downloads all possible occupations mapped to Wikidata Q-IDs.

### 2. `phase1_ingestion.py`
**Purpose**: Parse massive compressed Wikipedia `.xml.bz2` dumps.
**Key Functions**:
- `process_single_dump(args)`: Uses `xml.etree.ElementTree.iterparse` to memory-efficiently stream XML data. Extracts Wikipedia text, cleans markup, and saves plain text locally.
- Builds `translation_table.csv`, mapping Wikipedia Page IDs to plain english URL titles.

### 3. `phase2_segmentation.py`
**Purpose**: Segment raw Wikipedia text into categorized sections.
**Key Functions**:
- `parse_wikitext(text)`: Uses Regex to slice Wikipedia text into headers (e.g. `== Early Life ==`).
- `heuristic_classify(text)`: A fast keyword-based categorizer that labels sections as `biography`, `work`, `history`, `geography`, `science`, or `other`.
- Uses `multiprocessing.Pool` to process chunks of text across CPU cores, writing out chunked `.parquet` files.

### 4. `phase3_analysis.py`
**Purpose**: Build the citation network.
**Key Functions**:
- `get_sentences_with_citations(text, target_titles)`: Scans a string of text for Wikipedia internal link markup `[[Link]]`. If the `Link` matches a target person, it extracts the surrounding sentence.
- `init_worker()`: Multiprocessing initializer that ensures worker processes load the global dictionary of target people into memory.

### 5. `phase4_domain.py`
**Purpose**: Domain-level aggregations and metric generation.
**Key Functions**:
- `get_best_occupation(occ_str, target_occs)`: Finds the most relevant occupation for a person by parsing their multi-label occupation string against a target list.
- Calculates absolute domain ratios (how often men vs. women are cited overall) using `polars` for extreme tabular speed.

### 6. `phase5_aggregation.py`
**Purpose**: Final data preparation.
**Key Functions**:
- Joins the metadata from Wikidata with the citation metrics and classification counts generated in previous phases.
- Outputs a single, lightweight `viz_metadata.parquet` file that the dashboard can load instantly.

### 7. `phase6_sentiment_ml.py`
**Purpose**: Deep NLP Sentiment Analysis on citations.
**Key Functions**:
- Identifies the Top 100 most cited men and Top 100 most cited women in the Computer Science domain.
- Loads Hugging Face `transformers` pipelines.
- **Author Tone (RoBERTa)**: Analyzes the inherent sentiment of the sentence citing the individual (`cardiffnlp/twitter-roberta-base-sentiment`).
- **Event Quality (SST-2)**: Analyzes whether the event being described in the sentence was generally positive or negative (`distilbert-base-uncased-finetuned-sst-2-english`).

### 8. `phase7_experimental.py`
**Purpose**: Advanced NLP Bias Analysis via Computational Linguistics.
**Key Functions**:
- **spaCy NLP pipeline**: Implements Part-of-Speech (POS) tagging to extract adjectives near gendered pronouns, and Named Entity Recognition (NER) to map organizational affiliations.
- **Word2Vec Embeddings**: Trains a localized `Gensim` neural word embedding model strictly on the CS biography corpus.
- Computes mathematical cosine similarity vectors to objectively measure latent linguistic framing relative to gender.

---

##  Frontend: `streamlit_app.py`

The frontend uses `Streamlit` with a dark-mode, neon-accented, glassmorphic CSS design scheme for a premium presentation.

**Key Components**:
- **Data Loading**: Heavily utilizes `@st.cache_data` to ensure the `.parquet` datasets are only loaded into RAM once, allowing the web app to feel snappy.
- **Dashboard Tabs**:
  - `Global Demographics`: Analyzes Wikipedia as a whole.
  - `CS Demographics`: Filters the dataset down purely to individuals whose primary profession matches a strict pure-CS regex filter.
  - `Citation Network`: Uses `plotly` to graph the gender distribution of incoming citations.
  - `Sentiment & Rank`: Displays the ML-computed Author Tone and Event Quality metrics for the Top 10 individuals.
  - `Language Bias (Phase 7)`: Visualizes normalized adjective/organization frequencies and Word2Vec cosine similarity skew.
