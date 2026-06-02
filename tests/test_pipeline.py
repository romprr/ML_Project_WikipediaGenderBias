import os
import pytest
import polars as pl


# Set environment variables so the pipeline uses the fixture paths
os.environ["DATA_DIR"] = "tests/fixtures/data"
os.environ["OUTPUT_DIR"] = "tests/fixtures/output"
# For phase 3 & 4
os.environ["PHASE2_PARQUET_DIR"] = "tests/fixtures/output"
os.environ["TRANSLATION_TABLE"] = "tests/fixtures/output/translation_table.csv"

# Important: Only import pipeline modules AFTER setting env vars
import phase2_segmentation
import phase3_analysis
import phase4_domain
import phase5_aggregation


def test_full_pipeline():
    """
    Runs an end-to-end integration test of the data pipeline using the fake test dataset.
    """
    # 0. Run Phase 1 (Ingestion from fake BZ2)
    import phase1_ingestion
    phase1_ingestion.main()
    
    assert os.path.exists("tests/fixtures/output/1_Test Man.txt")
    assert os.path.exists("tests/fixtures/output/translation_table.csv")

    # 1. Run Phase 2 (Segmentation)
    # Output should go to tests/fixtures/output/phase2_parsed_data_part_*.parquet
    phase2_segmentation.OUTPUT_PARQUET_DIR = "tests/fixtures/output"
    phase2_segmentation.main()
    
    assert os.path.exists("tests/fixtures/output/phase2_parsed_data_part_0.parquet")
    
    # 2. Run Phase 3 (Analysis / Citation Extraction)
    phase3_analysis.main()
    
    assert os.path.exists("tests/fixtures/output/phase3_citations.parquet")
    df_cites = pl.read_parquet("tests/fixtures/output/phase3_citations.parquet")
    # Test Woman cited Test Man
    assert df_cites.filter(pl.col("cited_person") == "Test Man").height > 0
    
    # 3. Run Phase 4 (Domain Ratios)
    phase4_domain.main()
    
    assert os.path.exists("tests/fixtures/output/phase4_domain_ratios.parquet")
    assert os.path.exists("tests/fixtures/output/phase4_occupation_impact.parquet")
    
    # 4. Run Phase 5 (Aggregation)
    phase5_aggregation.main()
    
    assert os.path.exists("tests/fixtures/output/viz_metadata.parquet")
    assert os.path.exists("tests/fixtures/output/viz_phase3_citations.parquet")
    
    # 5. Run Phase 6 (ML Sentiment)
    import phase6_sentiment_ml
    phase6_sentiment_ml.main()
    
    assert os.path.exists("tests/fixtures/output/top200_ml_sentiment_analysis.parquet")
    
    # 6. Run Phase 7 (NLP & Word2Vec Bias)
    import phase7_experimental
    phase7_experimental.main()
    
    assert os.path.exists("tests/fixtures/output/phase7_adjectives.parquet")
    assert os.path.exists("tests/fixtures/output/phase7_organizations.parquet")
    assert os.path.exists("tests/fixtures/output/phase7_word2vec_bias.parquet")
    
    print("Full pipeline integration test passed successfully!")

