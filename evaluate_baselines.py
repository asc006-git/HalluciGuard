import pandas as pd
import re
import numpy as np
import os
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

def evaluate_baseline(y_true, y_pred, y_proba, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.5
    
    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc
    }

def main():
    print("Loading test data...")
    test_df = pd.read_csv("data/processed/test.csv")
    print(f"Test samples: {len(test_df)}")
    
    print("Extracting answer component...")
    test_df["knowledge"], test_df["question"], test_df["answer"] = zip(*test_df["text"].apply(extract_parts))
    test_df["answer_clean"] = test_df["answer"].apply(clean_text)
    
    y_true = test_df["label"].values
    
    class_dist = np.bincount(y_true)
    majority_class = np.argmax(class_dist)
    class_prior = class_dist / class_dist.sum()
    
    print(f"Class distribution: {class_dist}")
    print(f"Majority class: {majority_class} (prior: {class_prior[majority_class]:.4f})")
    
    np.random.seed(42)
    
    # Random Baseline: predict according to class distribution
    y_pred_random = np.random.choice([0, 1], size=len(y_true), p=class_prior)
    y_proba_random = np.tile(class_prior, (len(y_true), 1))
    random_results = evaluate_baseline(y_true, y_pred_random, y_proba_random[:, 1], "Random Baseline")
    
    # Majority Baseline: always predict majority class
    y_pred_majority = np.full(len(y_true), majority_class)
    y_proba_majority = np.zeros((len(y_true), 2))
    y_proba_majority[:, majority_class] = 1.0
    majority_results = evaluate_baseline(y_true, y_pred_majority, y_proba_majority[:, 1], "Majority Baseline")
    
    print("\n" + "="*80)
    print("BASELINE COMPARISON")
    print("="*80)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-"*80)
    for r in [random_results, majority_results]:
        print(f"{r['model']:<25} {r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f} {r['roc_auc']:>10.4f}")
    print("-"*80)
    
    results_df = pd.DataFrame([random_results, majority_results])
    results_df.to_csv(os.path.join(RESULTS_DIR, "metrics/baseline_comparison.csv"), index=False)
    print("\nSaved: " + os.path.join(RESULTS_DIR, "metrics/baseline_comparison.csv"))

if __name__ == "__main__":
    main()