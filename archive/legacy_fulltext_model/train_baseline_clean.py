import pandas as pd
import joblib
import re
import numpy as np
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def clean_text(text):
    text = str(text)
    text = re.sub(r'\bKnowledge:\s*', '', text)
    text = re.sub(r'\bQuestion:\s*', '', text)
    text = re.sub(r'\bAnswer:\s*', '', text)
    return text.strip()

def main():
    print("Loading clean splits...")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    print("Cleaning text...")
    train_df["text_clean"] = train_df["text"].apply(clean_text)
    test_df["text_clean"] = test_df["text"].apply(clean_text)
    
    X_train = train_df["text_clean"]
    y_train = train_df["label"]
    X_test = test_df["text_clean"]
    y_test = test_df["label"]
    
    print("Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Train shape: {X_train_tfidf.shape}, Test shape: {X_test_tfidf.shape}")
    
    print("Training Logistic Regression...")
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_tfidf, y_train)
    
    print("Evaluating on test set...")
    y_pred = clf.predict(X_test_tfidf)
    y_proba = clf.predict_proba(X_test_tfidf)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n=== Test Set Results ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    print("\n=== Confusion Matrix ===")
    print(f"               Predicted 0    Predicted 1")
    print(f"Actual 0 (factual)      {cm[0,0]:5d}           {cm[0,1]:5d}")
    print(f"Actual 1 (halluc)       {cm[1,0]:5d}           {cm[1,1]:5d}")
    
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Factual (0)", "Hallucination (1)"]))
    
    print("\n=== Top 20 Features (Positive = Hallucination, Negative = Factual) ===")
    feature_names = np.array(vectorizer.get_feature_names_out())
    coef = clf.coef_[0]
    
    top_positive_idx = np.argsort(coef)[-20:][::-1]
    top_negative_idx = np.argsort(coef)[:20]
    
    print("\nTop 20 POSITIVE (predict Hallucination):")
    for i, idx in enumerate(top_positive_idx, 1):
        print(f"  {i:2d}. {feature_names[idx]:<30s} {coef[idx]:.4f}")
    
    print("\nTop 20 NEGATIVE (predict Factual):")
    for i, idx in enumerate(top_negative_idx, 1):
        print(f"  {i:2d}. {feature_names[idx]:<30s} {coef[idx]:.4f}")
    
    print("\nSaving model and vectorizer...")
    joblib.dump(clf, "logistic_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("  Saved: logistic_model.pkl")
    print("  Saved: tfidf_vectorizer.pkl")

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1])
        }
    }
    
    with open(os.path.join(RESULTS_DIR, "baseline_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {os.path.join(RESULTS_DIR, 'baseline_metrics.json')}")
    
    with open(os.path.join(RESULTS_DIR, "baseline_metrics.txt"), "w") as f:
        f.write("=== Baseline Model Metrics ===\n\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"ROC-AUC:   {roc_auc:.4f}\n\n")
        f.write("=== Confusion Matrix ===\n")
        f.write(f"               Predicted 0    Predicted 1\n")
        f.write(f"Actual 0 (factual)      {cm[0,0]:5d}           {cm[0,1]:5d}\n")
        f.write(f"Actual 1 (halluc)       {cm[1,0]:5d}           {cm[1,1]:5d}\n\n")
        f.write("=== Confusion Matrix Values ===\n")
        f.write(f"True Negatives:  {cm[0,0]}\n")
        f.write(f"False Positives: {cm[0,1]}\n")
        f.write(f"False Negatives: {cm[1,0]}\n")
        f.write(f"True Positives:  {cm[1,1]}\n")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'baseline_metrics.txt')}")

if __name__ == "__main__":
    main()