from transformers import pipeline

def test_domain_classifier():
    print("Loading zero-shot classifier: cross-encoder/nli-deberta-v3-small...")
    classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-deberta-v3-small")
    
    candidate_labels = ["Computer Science", "Politics", "Business/Entrepreneurship", "General Biography"]
    
    sample_texts = [
        "She founded three different software companies and became a millionaire by age 30.",
        "He served as a senator for two terms before running for president.",
        "The algorithm uses a novel approach to graph traversal, achieving O(V+E) time complexity.",
        "He was born in London and attended the local grammar school, where he developed an early interest in mathematics.",
        "As CEO, she led the corporate restructuring that doubled the company's profit margins."
    ]
    
    print("\nTesting Domain Classification:\n" + "="*50)
    for text in sample_texts:
        result = classifier(text, candidate_labels)
        print(f"Text: {text}")
        print(f"Predicted Domain: {result['labels'][0]} (Score: {result['scores'][0]:.4f})\n")

if __name__ == "__main__":
    test_domain_classifier()
