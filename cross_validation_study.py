import pandas as pd
import re
import numpy as np
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

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
    print("Loading data...")
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    print("Extracting answer component...")
    train_df["knowledge"], train_df["question"], train_df["answer"] = zip(*train_df["text"].apply(extract_parts))
    test_df["knowledge"], test_df["question"], test_df["answer"] = zip(*test_df["text"].apply(extract_parts))
    
    X = train_df["answer"].apply(clean_text)
    y = train_df["label"]
    
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_results = []
    
    print("\nStarting 5-Fold Cross Validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        X_train_tfidf = vectorizer.fit_transform(X_train_fold)
        X_val_tfidf = vectorizer.transform(X_val_fold)
        
        clf.fit(X_train_tfidf, y_train_fold)
        
        y_pred = clf.predict(X_val_tfidf)
        y_proba = clf.predict_proba(X_val_tfidf)[:, 1]
        
        acc = accuracy_score(y_val_fold, y_pred)
        prec = precision_score(y_val_fold, y_pred)
        rec = recall_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred)
        roc_auc = roc_auc_score(y_val_fold, y_proba)
        
        fold_results.append({
            "fold": fold,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc
        })
        
        print(f"  Fold {fold}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")
    
    results_df = pd.DataFrame(fold_results)
    
    print("\n" + "="*80)
    print("CROSS-VALIDATION RESULTS (Answer Only)")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    
    print("\nAggregate Statistics:")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        vals = results_df[metric]
        print(f"  {metric:12s}: Mean={vals.mean():.4f}, Std={vals.std():.4f}, Min={vals.min():.4f}, Max={vals.max():.4f}")

    summary = {}
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        vals = results_df[metric]
        summary[metric] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "cv": float(vals.std() / vals.mean()) if vals.mean() != 0 else 0.0
        }

    results_df.to_csv(os.path.join(RESULTS_DIR, "metrics/cross_validation_results.csv"), index=False)
    print("\nSaved: " + os.path.join(RESULTS_DIR, "metrics/cross_validation_results.csv"))

    with open(os.path.join(RESULTS_DIR, "metrics/cross_validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved: " + os.path.join(RESULTS_DIR, "metrics/cross_validation_summary.json"))

    with open(os.path.join(RESULTS_DIR, "metrics/cross_validation_summary.txt"), "w") as f:
        f.write("CROSS-VALIDATION SUMMARY (Answer Only)\n")
        f.write("="*50 + "\n\n")
        f.write(f"{'Metric':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'CV%':>8}\n")
        f.write("-"*50 + "\n")
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            s = summary[metric]
            f.write(f"{metric:<12} {s['mean']:>10.4f} {s['std']:>10.4f} {s['min']:>10.4f} {s['max']:>10.4f} {s['cv']*100:>7.2f}\n")
        f.write("-"*50 + "\n")
        f.write(f"\nFolds: 5 (StratifiedKFold, shuffle=True, random_state=42)\n")
        f.write(f"Model: LogisticRegression + TF-IDF (Answer Only)\n")
        f.write(f"TF-IDF: max_features=10000, ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True\n")
        f.write(f"LogReg: C=1.0, class_weight=balanced, max_iter=1000\n")
    print("Saved: " + os.path.join(RESULTS_DIR, "metrics/cross_validation_summary.txt"))

if __name__ == "__main__":
    main()