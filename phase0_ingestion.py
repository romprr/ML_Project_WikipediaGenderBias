import subprocess
import sys
import os

def run_script(script_path):
    print(f"==================================================")
    print(f"Running Phase 0 script: {script_path}")
    print(f"==================================================")
    
    result = subprocess.run([sys.executable, script_path], check=False)
    
    if result.returncode != 0:
        print(f"\n[ERROR] Script {script_path} failed with exit code {result.returncode}.")
        sys.exit(result.returncode)
    else:
        print(f"\n[SUCCESS] Completed {script_path} successfully.\n")

def main():
    print("Starting Phase 0: Data Ingestion from Wikidata via SPARQL")
    
    scripts_to_run = [
        "scripts/ingestion/people_ingestion.py",
        "scripts/ingestion/cs_ingestion.py",
        "scripts/ingestion/occupations_ingestion.py"
    ]
    
    for script in scripts_to_run:
        if not os.path.exists(script):
            print(f"[ERROR] Could not find script at {script}")
            sys.exit(1)
        
        run_script(script)
        
    print("Phase 0 complete! All initial data has been ingested.")

if __name__ == "__main__":
    main()
