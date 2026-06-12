import pandas as pd
import joblib
import re
import numpy as np
import os
RESULTS_DIR = "results"
os.makedirs(os.path.join(RESULTS_DIR, "features"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def clean_text(text):
    text = str(text)
    text = re.sub(r'\bKnowledge:\s*', '', text)
    text = re.sub(r'\bQuestion:\s*', '', text)
    text = re.sub(r'\bAnswer:\s*', '', text)
    return text.strip()

def extract_parts(text):
    text = str(text)
    knowledge = ""
    question = ""
    answer = ""
    
    if "Knowledge:" in text and "Question:" in text and "Answer:" in text:
        k_part = text.split("Knowledge:")[1].split("Question:")[0].strip()
        q_part = text.split("Question:")[1].split("Answer:")[0].strip()
        a_part = text.split("Answer:")[1].strip()
        knowledge = k_part
        question = q_part
        answer = a_part
    
    return knowledge, question, answer

def main():
    print("Loading model and vectorizer...")
    clf = joblib.load("models/answer_only_model.pkl")
    vectorizer = joblib.load("models/answer_only_vectorizer.pkl")

    print("Loading test data...")
    test_df = pd.read_csv("data/processed/test.csv")
    print(f"Test samples: {len(test_df)}")

    print("Extracting answer component...")
    test_df["knowledge"], test_df["question"], test_df["answer"] = zip(*test_df["text"].apply(extract_parts))
    test_df["answer_clean"] = test_df["answer"].apply(clean_text)

    X_test = test_df["answer_clean"]
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
    plt.title('Confusion Matrix - Answer-Only Model', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/plots/answer_only_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Saved: results/plots/answer_only_confusion_matrix.png")
    plt.close()

if __name__ == "__main__":
    main()