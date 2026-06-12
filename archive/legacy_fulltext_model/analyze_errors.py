import pandas as pd
import joblib
import re
import numpy as np
import os

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

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
    y_proba = clf.predict_proba(X_test_tfidf)
    
    test_df["predicted"] = y_pred
    test_df["proba_0"] = y_proba[:, 0]
    test_df["proba_1"] = y_proba[:, 1]
    test_df["confidence"] = np.maximum(y_proba[:, 0], y_proba[:, 1])
    
    # False Positives: predicted 1 (hallucination), actual 0 (factual)
    fps = test_df[(test_df["predicted"] == 1) & (test_df["label"] == 0)].copy()
    fps = fps.sort_values("proba_1", ascending=False).head(50)
    
    # False Negatives: predicted 0 (factual), actual 1 (hallucination)
    fns = test_df[(test_df["predicted"] == 0) & (test_df["label"] == 1)].copy()
    fns = fns.sort_values("proba_0", ascending=False).head(50)
    
    print(f"\n=== Error Analysis ===")
    print(f"Total test samples: {len(test_df)}")
    print(f"False Positives: {len(test_df[(test_df['predicted']==1) & (test_df['label']==0)])}")
    print(f"False Negatives: {len(test_df[(test_df['predicted']==0) & (test_df['label']==1)])}")
    print(f"True Positives:  {len(test_df[(test_df['predicted']==1) & (test_df['label']==1)])}")
    print(f"True Negatives:  {len(test_df[(test_df['predicted']==0) & (test_df['label']==0)])}")
    
    fps_out = fps[["text", "label", "predicted", "proba_0", "proba_1", "confidence"]].copy()
    fps_out.columns = ["text", "true_label", "predicted_label", "prob_factual", "prob_hallucination", "confidence"]
    fps_out.to_csv("false_positives.csv", index=False)
    print(f"\nSaved top 50 false positives to false_positives.csv")
    
    fns_out = fns[["text", "label", "predicted", "proba_0", "proba_1", "confidence"]].copy()
    fns_out.columns = ["text", "true_label", "predicted_label", "prob_factual", "prob_hallucination", "confidence"]
    fns_out.to_csv("false_negatives.csv", index=False)
    print(f"Saved top 50 false negatives to false_negatives.csv")
    
    print("\n=== Top 5 False Positives (predicted hallucination, actually factual) ===")
    for i, row in fps.head(5).iterrows():
        print(f"\n  [{row['proba_1']:.3f}] {row['text'][:200]}...")
    
    print("\n=== Top 5 False Negatives (predicted factual, actually hallucination) ===")
    for i, row in fns.head(5).iterrows():
        print(f"\n  [{row['proba_0']:.3f}] {row['text'][:200]}...")
    
    print("\n=== Confidence Statistics ===")
    fp_count = len(test_df[(test_df['predicted']==1) & (test_df['label']==0)])
    fn_count = len(test_df[(test_df['predicted']==0) & (test_df['label']==1)])
    tp_count = len(test_df[(test_df['predicted']==1) & (test_df['label']==1)])
    tn_count = len(test_df[(test_df['predicted']==0) & (test_df['label']==0)])
    
    fp_avg_conf = fps['confidence'].mean() if len(fps) > 0 else 0
    fn_avg_conf = fns['confidence'].mean() if len(fns) > 0 else 0
    tp_avg_conf = test_df[(test_df['predicted']==1) & (test_df['label']==1)]['confidence'].mean() if tp_count > 0 else 0
    tn_avg_conf = test_df[(test_df['predicted']==0) & (test_df['label']==0)]['confidence'].mean() if tn_count > 0 else 0
    
    print(f"FP avg confidence: {fp_avg_conf:.3f}")
    print(f"FN avg confidence: {fn_avg_conf:.3f}")
    print(f"TP avg confidence: {tp_avg_conf:.3f}")
    print(f"TN avg confidence: {tn_avg_conf:.3f}")

    with open(os.path.join(RESULTS_DIR, "error_analysis_summary.txt"), "w") as f:
        f.write("=== Error Analysis Summary ===\n\n")
        f.write(f"Total test samples: {len(test_df)}\n")
        f.write(f"False Positives: {fp_count}\n")
        f.write(f"False Negatives: {fn_count}\n")
        f.write(f"True Positives:  {tp_count}\n")
        f.write(f"True Negatives:  {tn_count}\n\n")
        f.write("=== Confidence Statistics ===\n")
        f.write(f"FP avg confidence: {fp_avg_conf:.3f}\n")
        f.write(f"FN avg confidence: {fn_avg_conf:.3f}\n")
        f.write(f"TP avg confidence: {tp_avg_conf:.3f}\n")
        f.write(f"TN avg confidence: {tn_avg_conf:.3f}\n\n")
        f.write("=== Top 5 False Positives ===\n")
        for i, row in fps.head(5).iterrows():
            f.write(f"  [{row['proba_1']:.3f}] {row['text'][:200]}...\n\n")
        f.write("=== Top 5 False Negatives ===\n")
        for i, row in fns.head(5).iterrows():
            f.write(f"  [{row['proba_0']:.3f}] {row['text'][:200]}...\n")
    print(f"\nSaved: {os.path.join(RESULTS_DIR, 'error_analysis_summary.txt')}")

if __name__ == "__main__":
    main()