import os
import re
import requests
from tqdm import tqdm

# Add any language codes you want here
LANGUAGES = ["fr", "en"]
# The correct NEW base URL
BASE_URL = "https://dumps.wikimedia.org/other/mediawiki_content_current"
OUTPUT_DIR = "data/wikipedia_text_dumps"

def get_latest_valid_directory(lang):
    """
    Scans the new mediawiki_content_current directories for the most recent
    date that has successfully generated the xml/bzip2 folder.
    """
    url = f"{BASE_URL}/{lang}wiki/"
    response = requests.get(url)
    response.raise_for_status()
    
    # The new system uses YYYY-MM-DD format
    dates = sorted(re.findall(r'href="(\d{4}-\d{2}-\d{2})/"', response.text), reverse=True)
    
    for date in dates:
        bzip2_url = f"{BASE_URL}/{lang}wiki/{date}/xml/bzip2/"
        
        # Check if the bzip2 directory exists for this date
        check_resp = requests.get(bzip2_url)
        if check_resp.status_code == 200 and ".xml.bz2" in check_resp.text:
            return date, bzip2_url
            
    raise ValueError(f"Could not find any completed bzip2 dumps for {lang}wiki")

def get_chunk_files(bzip2_url):
    """
    Scrapes the target directory for all .xml.bz2 files. 
    (English wiki will have several, French might just have one).
    """
    response = requests.get(bzip2_url)
    response.raise_for_status()
    
    # Find all hrefs that end in .xml.bz2
    files = list(set(re.findall(r'href="([^"]+\.xml\.bz2)"', response.text)))
    
    # Filter out any weird absolute paths or parent directory links just in case
    files = [f for f in files if not f.startswith('/') and not f.startswith('http')]
    return sorted(files)

def download_with_resume(url, filepath):
    """
    Downloads a file with a progress bar, safely resuming if interrupted.
    """
    headers = {}
    file_mode = "wb"
    downloaded_bytes = 0
    
    if os.path.exists(filepath):
        downloaded_bytes = os.path.getsize(filepath)
        headers["Range"] = f"bytes={downloaded_bytes}-"
        file_mode = "ab" 
        
    response = requests.get(url, headers=headers, stream=True)
    
    if response.status_code == 416:
        print(f"✓ File already fully downloaded: {os.path.basename(filepath)}")
        return
        
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0)) + downloaded_bytes
    
    with open(filepath, file_mode) as f, tqdm(
        desc=os.path.basename(filepath),
        initial=downloaded_bytes,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                size = f.write(chunk)
                bar.update(size)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for lang in LANGUAGES:
        print(f"\n{'='*50}")
        print(f"Starting {lang.upper()} Wikipedia Extraction")
        print(f"{'='*50}")
        
        try:
            # 1. Find the right directory
            date, bzip2_url = get_latest_valid_directory(lang)
            print(f"Targeting valid dump from: {date}")
            
            # 2. Get all chunked files in that directory
            files_to_download = get_chunk_files(bzip2_url)
            print(f"Found {len(files_to_download)} file(s) to download.")
            
            # 3. Download them sequentially
            for filename in files_to_download:
                file_url = f"{bzip2_url}{filename}"
                output_path = os.path.join(OUTPUT_DIR, filename)
                download_with_resume(file_url, output_path)
                
        except Exception as e:
            print(f"Error processing {lang}wiki: {e}")

if __name__ == "__main__":
    main()