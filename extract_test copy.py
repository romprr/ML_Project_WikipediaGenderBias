import bz2
import os
import xml.etree.ElementTree as ET

# ==========================================
# CONFIGURATION VARIABLES
# ==========================================
DATA_DIR = "data/wikipedia_text_dumps"
TARGET_FILE = "enwiki-2026-05-01-p10p1134785.xml.bz2" # Change to your actual file
OUTPUT_DIR = "data/extracted_raw_pages"
PAGES_TO_EXTRACT = 10 # Set how many real pages you want to save
# ==========================================

def sanitize_filename(title):
    """
    Linux file systems hate forward slashes. 
    This removes illegal characters so 'OS/2' becomes 'OS_2'.
    """
    safe_title = title.replace("/", "_").replace("\0", "")
    return safe_title

def extract_raw_pages():
    file_path = os.path.join(DATA_DIR, TARGET_FILE)
    
    if not os.path.exists(file_path):
        print(f"❌ Could not find file: {file_path}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"📂 Reading from: {file_path}")
    print(f"💾 Saving raw wikitext to: {OUTPUT_DIR}/")
    print(f"⏳ Extracting first {PAGES_TO_EXTRACT} real articles (skipping redirects)...")
    print("=" * 50)

    articles_saved = 0

    with bz2.open(file_path, "rt", encoding="utf-8") as file:
        context = ET.iterparse(file, events=("end",))
        
        for event, elem in context:
            tag = elem.tag.split('}')[-1]
            
            if tag == "page":
                ns_elem = elem.find(".//{*}ns")
                
                # 1. Ensure it's a standard article (ns=0)
                if ns_elem is not None and ns_elem.text == "0":
                    
                    # Get the text immediately so we can check it
                    text_elem = elem.find(".//{*}text")
                    raw_text = text_elem.text if text_elem is not None and text_elem.text else ""
                    
                    # 2. THE BULLETPROOF REDIRECT CHECKS
                    # Check A: Does the XML tag exist?
                    has_redirect_tag = elem.find(".//{*}redirect") is not None
                    # Check B: Does the text start with #REDIRECT? (Catching #redirect, #Redirect, etc.)
                    has_redirect_text = raw_text.strip().lower().startswith("#redirect")
                    
                    if has_redirect_tag or has_redirect_text:
                        # It is definitely a redirect. Kill it.
                        elem.clear()
                        continue

                    # 3. If it survived the checks, it's a real page!
                    title_elem = elem.find(".//{*}title")
                    title = title_elem.text if title_elem is not None else "Unknown"
                    
                    safe_title = sanitize_filename(title)
                    output_filename = os.path.join(OUTPUT_DIR, f"{safe_title}.txt")
                    
                    with open(output_filename, "w", encoding="utf-8") as out_file:
                        out_file.write(f"TITLE: {title}\n")
                        out_file.write("=" * 50 + "\n\n")
                        out_file.write(raw_text)
                        
                    print(f"✅ Saved Raw: {output_filename}")
                    articles_saved += 1
                    
                    if articles_saved >= PAGES_TO_EXTRACT:
                        break
                
                elem.clear()

    print("=" * 50)
    print("🎉 Raw extraction complete!")

if __name__ == "__main__":
    extract_raw_pages()