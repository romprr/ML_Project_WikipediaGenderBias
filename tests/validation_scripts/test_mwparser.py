import mwparserfromhell
import re
import time

sample_text = """
{{Infobox person
| name = Ada Lovelace
| birth_date = {{Birth date|1815|12|10|df=yes}}
| occupation = [[Mathematician]], [[computer scientist]]
| known_for = First computer program
}}
'''Augusta Ada King, Countess of Lovelace''' (''née'' '''Byron''') was an English [[mathematician]] and writer.
She worked on [[Charles Babbage]]'s early mechanical general-purpose computer, the [[Analytical Engine]].

== Early life ==
She was the only legitimate child of poet [[Lord Byron]]. She was born on 10 December 1815.

== Work ==
Lovelace's notes are important in the early history of computers. See also [[Computer Science|CS]].
"""

def test_regex():
    start = time.perf_counter()
    # Regex Section split
    sections = re.split(r'\n==\s*(.*?)\s*==\n', sample_text)
    
    # Regex link extraction
    links = re.findall(r'\[\[([^\|\]]+)(?:\|[^\]]+)?\]\]', sample_text)
    
    # Regex infobox
    occ_match = re.search(r'\|\s*occupation\s*=\s*(.*?)\n\|', sample_text, re.IGNORECASE | re.DOTALL)
    
    print(f"REGEX Time: {(time.perf_counter()-start)*1000:.3f}ms")
    print(f"Regex Links: {links}")
    if occ_match: print(f"Regex Occ: {occ_match.group(1)}")

def test_mwparser():
    start = time.perf_counter()
    parsed = mwparserfromhell.parse(sample_text)
    
    # Sections
    sections = parsed.get_sections(include_lead=True)
    
    # Links
    links = [link.title.strip_code() for link in parsed.filter_wikilinks()]
    
    # Infobox
    templates = parsed.filter_templates()
    infoboxes = [t for t in templates if t.name.strip().lower().startswith("infobox")]
    occ = None
    if infoboxes:
        box = infoboxes[0]
        if box.has("occupation"):
            occ = box.get("occupation").value.strip_code().strip()
            
    print(f"\nMWPARSER Time: {(time.perf_counter()-start)*1000:.3f}ms")
    print(f"MWParser Links: {links}")
    print(f"MWParser Occ: {occ}")
    print(f"MWParser Sections: {[s.filter_headings()[0].title.strip() if s.filter_headings() else 'LEAD' for s in sections]}")

if __name__ == "__main__":
    test_regex()
    test_mwparser()
