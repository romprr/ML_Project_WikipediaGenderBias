import os
import polars as pl
import pandas as pd

def create_fixtures():
    os.makedirs("tests/fixtures/data/wikidata_parquet_batches", exist_ok=True)
    os.makedirs("tests/fixtures/data/wikipedia_text_dumps", exist_ok=True)
    os.makedirs("tests/fixtures/data/extracted_raw_pages", exist_ok=True)
    os.makedirs("tests/fixtures/output", exist_ok=True)
    
    import bz2
    xml_content = """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/">
      <page>
        <title>Test Man</title>
        <ns>0</ns>
        <id>1</id>
        <revision>
          <text>{{Infobox person
| name = Test Man
| birth_date = 1950
| occupation = Programmer
}}
Test Man is a [[programmer]]. He invented the [[Test Algorithm]].
== Early life ==
He studied [[Computer Science|CS]].</text>
        </revision>
      </page>
      <page>
        <title>Test Woman</title>
        <ns>0</ns>
        <id>2</id>
        <revision>
          <text>{{Infobox person
| name = Test Woman
| birth_date = 1960
| profession = [[Software engineer]]
}}
Test Woman is a [[mathematician]]. She cited [[Test Man]].
== Work ==
She worked on [[software]].</text>
        </revision>
      </page>
    </mediawiki>"""
    with bz2.open("tests/fixtures/data/wikipedia_text_dumps/dump.xml.bz2", "wt", encoding="utf-8") as f:
        f.write(xml_content)
        
    # Fake people metadata
    people = pl.DataFrame({
        "person": ["Q1", "Q2", "Q3"],
        "enwiki": ["https://en.wikipedia.org/wiki/Test_Man", "https://en.wikipedia.org/wiki/Test_Woman", "https://en.wikipedia.org/wiki/Test_Unknown"],
        "gender": ["http://www.wikidata.org/entity/Q6581097", "http://www.wikidata.org/entity/Q6581072", None],
        "occupation_labels": ["computer scientist", "mathematician", None],
        "birthdate": ["1950-01-01", "1960-01-01", None],
        "deathdate": [None, None, None]
    })
    people.write_parquet("tests/fixtures/data/wikidata_parquet_batches/batch_0.parquet")
    
    # Fake Occupations
    os.makedirs("tests/fixtures/data/wikidata_occupations_parquet_batches", exist_ok=True)
    occs = pl.DataFrame({
        "occupation": ["Q1", "Q2"],
        "name_en": ["computer scientist", "mathematician"]
    })
    occs.write_parquet("tests/fixtures/data/wikidata_occupations_parquet_batches/batch_0.parquet")
    
    # Fake CS Topics
    topics = pd.DataFrame({
        "topic": ["Q10"],
        "enwiki": ["https://en.wikipedia.org/wiki/Test_Algorithm"]
    })
    topics.to_parquet("tests/fixtures/data/wikidata_computer_science.parquet")
    
    # Fake raw wikitexts
    t1 = """{{Infobox person
| name = Test Man
| birth_date = 1950
| occupation = Programmer
}}
Test Man is a [[programmer]]. He invented the [[Test Algorithm]].
== Early life ==
He studied [[Computer Science|CS]]."""
    with open("tests/fixtures/data/extracted_raw_pages/1_Test_Man.txt", "w") as f: f.write(t1)
    
    t2 = """{{Infobox person
| name = Test Woman
| birth_date = 1960
| profession = [[Software engineer]]
}}
Test Woman is a [[mathematician]]. She cited [[Test Man]].
== Work ==
She worked on [[software]]."""
    with open("tests/fixtures/data/extracted_raw_pages/2_Test_Woman.txt", "w") as f: f.write(t2)
    
    t3 = """Test Unknown is a mysterious person."""
    with open("tests/fixtures/data/extracted_raw_pages/3_Test_Unknown.txt", "w") as f: f.write(t3)
    
    print("Test fixtures created!")

if __name__ == "__main__":
    create_fixtures()
