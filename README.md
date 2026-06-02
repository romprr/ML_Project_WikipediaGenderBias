# Wikipedia Gender Bias in Computer Science (ML Project)
Author: Romain Poirrier

This project analyzes gender bias on Wikipedia, focusing specifically on the domain of Computer Science. It downloads Wikidata metadata and Wikipedia text dumps, extracts citations and sections, and uses Natural Language Processing to quantify how often men and women are cited, in what context, and with what sentiment.

---

##  Setup & Installation

You can set up this project using the standard `venv` or the ultra-fast `uv` package manager.

### Option 1: Using `uv` (Recommended)
`uv` is an extremely fast Python package manager written in Rust.

1. Install `uv` if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Create and activate a virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate    # On Linux/macOS
   .venv\Scripts\activate       # On Windows
   ```
3. Install the dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

### Option 2: Using standard `venv`
1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
2. Activate it:
   ```bash
   source .venv/bin/activate    # On Linux/macOS
   .venv\Scripts\activate       # On Windows
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

###  Important: Install PyTorch
The `requirements.txt` installs `torch`, but depending on your machine (Nvidia GPU, AMD GPU, Apple Silicon, or CPU only), you should install the specific version of PyTorch tailored to your hardware for hardware acceleration.
Please visit the **[PyTorch Getting Started Page](https://pytorch.org/get-started/locally/)** to get the exact install command for your machine. 

*(Example for CUDA 11.8 on Windows/Linux: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`)*

---

## 🧪 Running the Tests

The project includes an end-to-end integration test suite that generates dummy data and runs the entire pipeline to ensure everything works flawlessly on your machine.

1. Generate the test data fixtures:
   ```bash
   python tests/create_test_fixtures.py
   ```
2. Run the integration tests using `pytest`:
   ```bash
   PYTHONPATH=. pytest tests/ -v -s
   ```
   *(If all tests pass, your environment is perfectly configured!)*

---

##  Running the Pipeline

The data processing pipeline is broken into sequential phases. You should run them in order.

> **Note on Data:** I strongly recommend using the pre-downloaded Wikidata & Wikidumps file available [here](https://drive.google.com/drive/folders/19RtHnD9J-Rd_eUnKZxF2IEIOcgDAiv53?usp=drive_link) and placing them in a `data/` folder, as downloading raw dumps from Wikipedia can take hours. If you use the provided data, you can skip **Phase 0**.

### Phase 0: Data Ingestion (SPARQL)
Downloads real-time metadata from Wikidata.
```bash
python phase0_ingestion.py
```

### Phase 1: XML Dump Parsing
Parses the massive Wikipedia XML bz2 dumps into raw text files.
```bash
python phase1_ingestion.py
```

### Phase 2: Section Segmentation
Splits the raw text files into categorized sections (Biography, Work, etc.) and sentences.
```bash
python phase2_segmentation.py
```

### Phase 3: Citation Extraction
Scans sentences to find wiki-links and extracts cross-citations between individuals.
```bash
python phase3_analysis.py
```

### Phase 4: Domain Metrics
Computes broad Wikipedia occupation metrics and citation imbalances.
```bash
python phase4_domain.py
```

### Phase 5: Aggregation
Merges the outputs of all previous phases into `viz_metadata.parquet` for fast visualization.
```bash
python phase5_aggregation.py
```

### Phase 6: Sentiment & ML Analysis
Runs heavy Transformer models (RoBERTa & SST-2) to score Author Tone and Event Quality on the Top 200 cited individuals in Computer Science.
```bash
python phase6_sentiment_ml.py
```

### Phase 7: Advanced NLP Bias Analysis
Trains a custom Word2Vec neural embedding model on the CS corpus and uses `spaCy` to extract Named Entities (Organizations) and Parts-of-Speech (Adjectives) near gendered pronouns, quantifying latent linguistic bias.
```bash
python phase7_experimental.py
```

---

##  Launching the Dashboard

Once the pipeline has completed, you can visualize the results using the interactive Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```
This will open a rich, modern, dynamic web interface in your browser displaying the gender demographics, citation networks, and sentiment analyses.
