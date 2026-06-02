from transformers import pipeline

def test_sentiment():
    print("Loading Event Quality model (distilbert-base-uncased-finetuned-sst-2-english)...")
    event_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    print("Loading Author Tone model (cardiffnlp/twitter-roberta-base-sentiment)...")
    tone_classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    
    # Real sentences extracted directly from Wikipedia Computer Science biographies
    sample_sentences = [
        "Turing was highly influential in the development of theoretical computer science, providing a formalisation of the concepts of algorithm and computation.",
        "Hopper was posthumously awarded the Presidential Medal of Freedom in 2016 by President Barack Obama.",
        "Torvalds has been criticized for his harsh behavior towards developers on the Linux kernel mailing list.",
        "Stallman resigned from MIT and the FSF in 2019 following controversial comments he made about the Jeffrey Epstein scandal.",
        "Lovelace was the first to recognize that the machine had applications beyond pure calculation.",
        "Hamilton's software approach was so successful that no software bugs were ever known to have occurred during any crewed Apollo missions.",
        "He committed suicide in 1954, two years after being convicted of gross indecency for his homosexuality and accepting a chemical castration treatment."
    ]
    
    label_map = {"LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE"}
    
    print("\nTesting sentiment on sample Wikipedia-style sentences:\n" + "="*50)
    for sentence in sample_sentences:
        event_res = event_classifier(sentence)[0]
        tone_res = tone_classifier(sentence)[0]
        tone_label = label_map.get(tone_res['label'], tone_res['label'])
        
        print(f"Text: {sentence}")
        print(f"Event Quality: {event_res['label']} (Score: {event_res['score']:.4f})")
        print(f"Author Tone: {tone_label} (Score: {tone_res['score']:.4f})\n")
        
if __name__ == "__main__":
    test_sentiment()
