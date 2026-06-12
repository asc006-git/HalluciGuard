import pandas as pd
import joblib
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def clean_text(text):
    text = str(text)
    text = re.sub(r'\bKnowledge:\s*', '', text)
    text = re.sub(r'\bQuestion:\s*', '', text)
    text = re.sub(r'\bAnswer:\s*', '', text)
    return text.strip()

def main():
    print("Loading model and vectorizer...")
    clf = joblib.load("logistic_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    print("Loading test data...")
    test_df = pd.read_csv("test.csv")
    print(f"Test samples: {len(test_df)}")

    print("Cleaning text...")
    test_df["text_clean"] = test_df["text"].apply(clean_text)

    X_test = test_df["text_clean"]
    y_test = test_df["label"]

    print("Vectorizing...")
    X_test_tfidf = vectorizer.transform(X_test)

    print("Predicting...")
    y_pred = clf.predict(X_test_tfidf)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Factual (0)', 'Hallucination (1)'],
                yticklabels=['Factual (0)', 'Hallucination (1)'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - HalluciGuard Baseline Model', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Saved: results/confusion_matrix.png")
    plt.close()

if __name__ == "__main__":
    main()