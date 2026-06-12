import pandas as pd
import re
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

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
    
    y_train = train_df["label"]
    y_test = test_df["label"]
    
    X_train = train_df["answer"].apply(clean_text)
    X_test = test_df["answer"].apply(clean_text)
    
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
    
    print("\n=== Test Set Results (Answer Only) ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    model_path = os.path.join(MODELS_DIR, "answer_only_model.pkl")
    vectorizer_path = os.path.join(MODELS_DIR, "answer_only_vectorizer.pkl")
    
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"\nSaved: {model_path}")
    print(f"Saved: {vectorizer_path}")

if __name__ == "__main__":
    main()