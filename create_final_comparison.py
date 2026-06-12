import pandas as pd
import json
import os

RESULTS_DIR = "results"

def main():
    models = []
    
    # Load Random and Majority baselines
    baseline_df = pd.read_csv(os.path.join(RESULTS_DIR, "metrics/baseline_comparison.csv"))
    for _, row in baseline_df.iterrows():
        models.append({
            "Model": row["model"],
            "Accuracy": row["accuracy"],
            "Precision": row["precision"],
            "Recall": row["recall"],
            "F1": row["f1"],
            "ROC-AUC": row["roc_auc"]
        })
    
    # Load Length baseline
    length_df = pd.read_csv(os.path.join(RESULTS_DIR, "metrics/length_baseline_results.csv"))
    for _, row in length_df.iterrows():
        models.append({
            "Model": row["model"],
            "Accuracy": row["accuracy"],
            "Precision": row["precision"],
            "Recall": row["recall"],
            "F1": row["f1"],
            "ROC-AUC": row["roc_auc"]
        })
    
    # Load Answer-Only Logistic Regression from ablation results (test set performance)
    ablation_df = pd.read_csv(os.path.join(RESULTS_DIR, "metrics/ablation_results.csv"))
    answer_only = ablation_df[ablation_df["model"] == "Answer Only"].iloc[0]
    models.append({
        "Model": "Answer-Only Logistic Regression (TF-IDF)",
        "Accuracy": answer_only["accuracy"],
        "Precision": answer_only["precision"],
        "Recall": answer_only["recall"],
        "F1": answer_only["f1"],
        "ROC-AUC": answer_only["roc_auc"]
    })
    
    # Load cross-validation mean for Answer-Only as well
    with open(os.path.join(RESULTS_DIR, "metrics/cross_validation_summary.json"), "r") as f:
        cv_summary = json.load(f)
    models.append({
        "Model": "Answer-Only Logistic Regression (CV Mean)",
        "Accuracy": cv_summary["accuracy"]["mean"],
        "Precision": cv_summary["precision"]["mean"],
        "Recall": cv_summary["recall"]["mean"],
        "F1": cv_summary["f1"]["mean"],
        "ROC-AUC": cv_summary["roc_auc"]["mean"]
    })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(models)
    comparison_df = comparison_df.sort_values("F1", ascending=False).reset_index(drop=True)
    comparison_df.index += 1
    
    # Save
    output_path = os.path.join(RESULTS_DIR, "metrics/model_comparison.csv")
    comparison_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    # Print ranking
    print("\n" + "="*90)
    print("FINAL MODEL COMPARISON (Sorted by F1)")
    print("="*90)
    print(f"{'Rank':<5} {'Model':<45} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-"*90)
    for idx, row in comparison_df.iterrows():
        print(f"{idx:<5} {row['Model']:<45} {row['Accuracy']:>10.4f} {row['Precision']:>10.4f} {row['Recall']:>10.4f} {row['F1']:>10.4f} {row['ROC-AUC']:>10.4f}")
    print("-"*90)

if __name__ == "__main__":
    main()