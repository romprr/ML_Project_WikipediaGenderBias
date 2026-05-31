from urllib.parse import unquote
import mwparserfromhell

def normalize_title(title):
    """
    Normalizes a Wikipedia title so both URLs and Wikilinks match perfectly.
    'rené descartes' -> 'René_Descartes'
    """
    if not title: return ""
    title = unquote(title)
    title = title.replace(" ", "_")
    if len(title) > 0:
        title = title[0].upper() + title[1:]
    return title