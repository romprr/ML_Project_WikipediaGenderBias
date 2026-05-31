# ML project : Wikipedia from a gender point of view (Subject 5)
Author : Romain Poirrier

## Installation

First create a virtual envinronment :
```
python -m venv .venv
```

Connect to your venv :

**Linux / MacOS**
```
source .venv/bin/activate
```

**Windows**
```
.venv/Scripts/activate
```

Next, install the required libraries :
```
pip install -r requirements.txt
```

**Finally**  
You will need to install the appropriate pytorch version based on your system, for this, it is better to follow the instructions on pytorch's installation page [here](https://pytorch.org/get-started/locally/).

## Usage

I strongly recommend to use the save of the downloaded wikidata & wikidumps file available [here](https://drive.google.com/drive/folders/19RtHnD9J-Rd_eUnKZxF2IEIOcgDAiv53?usp=drive_link), instead of downloading them from wikipedia, since the download of the dumps can be very slow. 

### Download content from wikidata & wikipedia

Run the following : 
```
python content_downloader.py
python people_ingestion.py
pthon topics_ingestion.py
```

