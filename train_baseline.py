import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def clean_text(text):
    text = str(text)
    text = re.sub(r'\bKnowledge:\s*', '', text)
    text = re.sub(r'\bQuestion:\s*', '', text)
    text = re.sub(r'\bAnswer:\s*', '', text)
    return text.strip()

def main():
    print("Loading dataset...")
    df = pd.read_csv("halluciguard_dataset.csv")
    
    print("Cleaning text (removing template tokens)...")
    df["text_clean"] = df["text"].apply(clean_text)
    
    X = df["text_clean"]
    y = df["label"]
    
    print("Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
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
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  Train shape: {X_train_tfidf.shape}")
    
    print("Training Logistic Regression...")
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_tfidf, y_train)
    
    print("Evaluating...")
    y_pred = clf.predict(X_test_tfidf)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n=== Results ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print("\n=== Confusion Matrix ===")
    print(f"               Predicted 0    Predicted 1")
    print(f"Actual 0 (factual)      {cm[0,0]:5d}           {cm[0,1]:5d}")
    print(f"Actual 1 (halluc)       {cm[1,0]:5d}           {cm[1,1]:5d}")
    
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Factual (0)", "Hallucination (1)"]))
    
    print("Saving model and vectorizer...")
    joblib.dump(clf, "halluciguard_logreg_model.pkl")
    joblib.dump(vectorizer, "halluciguard_tfidf_vectorizer.pkl")
    print("  Saved: halluciguard_logreg_model.pkl")
    print("  Saved: halluciguard_tfidf_vectorizer.pkl")

if __name__ == "__main__":
    main()