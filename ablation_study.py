import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def clean_text(text):
    text = str(text)
    text = re.sub(r'\bKnowledge:\s*', '', text)
    text = re.sub(r'\bQuestion:\s*', '', text)
    text = re.sub(r'\bAnswer:\s*', '', text)
    return text.strip()

def extract_parts(text):
    """Extract knowledge, question, answer from full text."""
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

def train_and_evaluate(X_train, X_test, y_train, y_test, model_name):
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_tfidf, y_train)
    
    y_pred = clf.predict(X_test_tfidf)
    y_proba = clf.predict_proba(X_test_tfidf)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "vocab_size": len(vectorizer.vocabulary_)
    }

def main():
    print("Loading data...")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    print("Extracting components...")
    train_df["knowledge"], train_df["question"], train_df["answer"] = zip(*train_df["text"].apply(extract_parts))
    test_df["knowledge"], test_df["question"], test_df["answer"] = zip(*test_df["text"].apply(extract_parts))
    
    y_train = train_df["label"]
    y_test = test_df["label"]
    
    # Model A: Answer only
    print("\n=== Model A: Answer Only ===")
    X_train_a = train_df["answer"].apply(clean_text)
    X_test_a = test_df["answer"].apply(clean_text)
    results_a = train_and_evaluate(X_train_a, X_test_a, y_train, y_test, "Answer Only")
    
    # Model B: Question + Answer
    print("\n=== Model B: Question + Answer ===")
    train_df["qa"] = train_df["question"] + " " + train_df["answer"]
    test_df["qa"] = test_df["question"] + " " + test_df["answer"]
    X_train_b = train_df["qa"].apply(clean_text)
    X_test_b = test_df["qa"].apply(clean_text)
    results_b = train_and_evaluate(X_train_b, X_test_b, y_train, y_test, "Question + Answer")
    
    # Model C: Knowledge + Question + Answer (Full)
    print("\n=== Model C: Knowledge + Question + Answer (Full) ===")
    X_train_c = train_df["text"].apply(clean_text)
    X_test_c = test_df["text"].apply(clean_text)
    results_c = train_and_evaluate(X_train_c, X_test_c, y_train, y_test, "Full (Knowledge + Q + A)")
    
    # Comparison table
    results = [results_a, results_b, results_c]
    
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(f"{'Model':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-"*80)
    for r in results:
        print(f"{r['model']:<30} {r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f} {r['roc_auc']:>10.4f}")
    print("-"*80)
    
    print("\nVocabulary sizes:")
    for r in results:
        print(f"  {r['model']}: {r['vocab_size']} features")
    
    # Best model
    best = max(results, key=lambda x: x['f1'])
    print(f"\nBest model by F1: {best['model']} (F1={best['f1']:.4f})")

if __name__ == "__main__":
    main()
