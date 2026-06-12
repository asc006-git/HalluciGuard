import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

RESULTS_DIR = "results"

def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "cross_validation_results.csv"))
    
    # Plot 1: F1 Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    x = df["fold"]
    f1_vals = df["f1"]
    
    bars = ax.bar(x, f1_vals, color="#2E86AB", edgecolor="black", linewidth=1.2, width=0.6)
    ax.axhline(y=f1_vals.mean(), color="#E74C3C", linestyle="--", linewidth=2, 
               label=f"Mean = {f1_vals.mean():.4f}")
    ax.fill_between([0.5, 5.5], f1_vals.mean() - f1_vals.std(), f1_vals.mean() + f1_vals.std(),
                    color="#E74C3C", alpha=0.15, label=f"±1 Std = {f1_vals.std():.4f}")
    
    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    
    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Cross-Validation F1 Distribution (Answer Only)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_ylim(0.90, 0.92)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cv_f1_distribution.png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'cv_f1_distribution.png')}")
    plt.close()
    
    # Plot 2: Metric Comparison
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    colors = ["#2E86AB", "#28B463", "#F39C12", "#E74C3C", "#8E44AD"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.15
    
    for i, fold in enumerate(range(1, 6)):
        fold_data = df[df["fold"] == fold]
        vals = [fold_data[m].values[0] for m in metrics]
        bars = ax.bar(x + (i - 2) * width, vals, width, 
                      label=f"Fold {fold}", color=colors[i], edgecolor="black", linewidth=0.5)
    
    means = [df[m].mean() for m in metrics]
    ax.scatter(x, means, color="black", s=80, zorder=5, marker="D", label="Mean", edgecolors="white", linewidth=1)
    
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Cross-Validation Metric Comparison Across Folds (Answer Only)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.85, 0.98)
    ax.legend(fontsize=10, ncol=3, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cv_metric_comparison.png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'cv_metric_comparison.png')}")
    plt.close()

if __name__ == "__main__":
    main()