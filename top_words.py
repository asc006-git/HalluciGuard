import pandas as pd
from collections import Counter
import re

try:
    from nltk.corpus import stopwords
    import nltk
    nltk.data.find('corpora/stopwords')
except LookupError:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords

def get_top_words(texts, n=30):
    stop_words = set(stopwords.words('english'))
    words = []
    for text in texts:
        tokens = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        words.extend([w for w in tokens if w not in stop_words])
    return Counter(words).most_common(n)

def main():
    df = pd.read_csv("halluciguard_dataset.csv")
    
    label_0_texts = df[df["label"] == 0]["text"].tolist()
    label_1_texts = df[df["label"] == 1]["text"].tolist()
    
    print("=== Top 30 Words - Label 0 (Factual) ===")
    top_0 = get_top_words(label_0_texts, 30)
    for i, (word, count) in enumerate(top_0, 1):
        print(f"  {i:2d}. {word:<20s} {count:,}")
    
    print("\n=== Top 30 Words - Label 1 (Hallucination) ===")
    top_1 = get_top_words(label_1_texts, 30)
    for i, (word, count) in enumerate(top_1, 1):
        print(f"  {i:2d}. {word:<20s} {count:,}")

if __name__ == "__main__":
    main()