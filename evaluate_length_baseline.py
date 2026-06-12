import pandas as pd
import re
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

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
    
    train_df["answer_clean"] = train_df["answer"].apply(clean_text)
    test_df["answer_clean"] = test_df["answer"].apply(clean_text)
    
    # Extract length features
    train_df["char_count"] = train_df["answer_clean"].str.len()
    train_df["word_count"] = train_df["answer_clean"].str.split().str.len()
    test_df["char_count"] = test_df["answer_clean"].str.len()
    test_df["word_count"] = test_df["answer_clean"].str.split().str.len()
    
    X_train = train_df[["char_count", "word_count"]].values
    X_test = test_df[["char_count", "word_count"]].values
    y_train = train_df["label"].values
    y_test = test_df["label"].values
    
    print(f"\nTrain length stats:")
    print(f"  char_count: mean={train_df['char_count'].mean():.1f}, std={train_df['char_count'].std():.1f}")
    print(f"  word_count: mean={train_df['word_count'].mean():.1f}, std={train_df['word_count'].std():.1f}")
    print(f"\nTest length stats:")
    print(f"  char_count: mean={test_df['char_count'].mean():.1f}, std={test_df['char_count'].std():.1f}")
    print(f"  word_count: mean={test_df['word_count'].mean():.1f}, std={test_df['word_count'].std():.1f}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining Logistic Regression on length features...")
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)
    
    print("Evaluating...")
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print("\n" + "="*60)
    print("LENGTH-ONLY BASELINE RESULTS")
    print("="*60)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("="*60)
    print(f"\nFeature coefficients:")
    print(f"  char_count: {clf.coef_[0][0]:.4f}")
    print(f"  word_count: {clf.coef_[0][1]:.4f}")
    print(f"  intercept:  {clf.intercept_[0]:.4f}")
    
    results_df = pd.DataFrame([{
        "model": "Length-Only Logistic Regression",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc
    }])
    results_df.to_csv(os.path.join(RESULTS_DIR, "metrics/length_baseline_results.csv"), index=False)
    print("\nSaved: " + os.path.join(RESULTS_DIR, "metrics/length_baseline_results.csv"))

if __name__ == "__main__":
    main()