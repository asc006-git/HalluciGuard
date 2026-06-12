import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    os.makedirs("docs/assets/eda", exist_ok=True)
    
    df = pd.read_csv("data/raw/halluciguard_dataset.csv")
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12
    
    # 1. Class Distribution Bar Chart
    fig, ax = plt.subplots()
    class_counts = df["label"].value_counts().sort_index()
    labels = ["Factual (0)", "Hallucination (1)"]
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(labels, class_counts.values, color=colors, edgecolor="black", linewidth=1.2)
    ax.set_title("Class Distribution", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel("Number of Samples", fontsize=13)
    ax.set_ylim(0, max(class_counts.values) * 1.15)
    for bar, count in zip(bars, class_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                f"{count:,}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("docs/assets/eda/class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: docs/assets/eda/class_distribution.png")
    
    # 2. Text Length Histogram
    fig, ax = plt.subplots()
    for label, color, label_name in [(0, "#2ecc71", "Factual"), (1, "#e74c3c", "Hallucination")]:
        subset = df[df["label"] == label]["text_length"]
        ax.hist(subset, bins=50, alpha=0.6, label=label_name, color=color, edgecolor="black", linewidth=0.5)
    ax.set_title("Text Length Distribution by Class", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Text Length (characters)", fontsize=13)
    ax.set_ylabel("Frequency", fontsize=13)
    ax.legend(fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("docs/assets/eda/text_length_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: docs/assets/eda/text_length_histogram.png")
    
    # 3. Word Count Histogram
    fig, ax = plt.subplots()
    for label, color, label_name in [(0, "#2ecc71", "Factual"), (1, "#e74c3c", "Hallucination")]:
        subset = df[df["label"] == label]["word_count"]
        ax.hist(subset, bins=50, alpha=0.6, label=label_name, color=color, edgecolor="black", linewidth=0.5)
    ax.set_title("Word Count Distribution by Class", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Word Count", fontsize=13)
    ax.set_ylabel("Frequency", fontsize=13)
    ax.legend(fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("docs/assets/eda/word_count_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: docs/assets/eda/word_count_histogram.png")
    
    print("\nAll plots saved to docs/assets/eda/")

if __name__ == "__main__":
    main()