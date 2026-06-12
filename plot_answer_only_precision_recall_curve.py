import pandas as pd
import joblib
import re
import numpy as np
import os
RESULTS_DIR = "results"
os.makedirs(os.path.join(RESULTS_DIR, "features"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

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

    print("Predicting probabilities...")
    y_proba = clf.predict_proba(X_test_tfidf)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color='#A23B72', lw=2.5, 
             label=f'Precision-Recall Curve (AP = {avg_precision:.4f})')
    plt.axhline(y=y_test.mean(), color='gray', lw=1.5, linestyle='--', 
                label=f'Random Classifier (AP = {y_test.mean():.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - Answer-Only Model', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/answer_only_precision_recall_curve.png', dpi=300, bbox_inches='tight')
    print(f"Saved: results/plots/answer_only_precision_recall_curve.png (AP = {avg_precision:.4f})")
    plt.close()

if __name__ == "__main__":
    main()