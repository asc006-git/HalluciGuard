import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import os

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
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    print("Extracting answer component...")
    df["knowledge"], df["question"], df["answer"] = zip(*df["text"].apply(extract_parts))
    df["answer_clean"] = df["answer"].apply(clean_text)
    df["char_count"] = df["answer_clean"].str.len()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data = [
        df[df["label"] == 0]["char_count"],
        df[df["label"] == 1]["char_count"]
    ]
    labels = ["Factual (0)", "Hallucination (1)"]
    colors = ["#2E86AB", "#E74C3C"]
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.2)
    
    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.2)
    
    for flier in bp['fliers']:
        flier.set_marker('o')
        flier.set_markerfacecolor('gray')
        flier.set_alpha(0.5)
        flier.set_markersize(4)
    
    ax.set_ylabel("Character Length", fontsize=12)
    ax.set_title("Answer Character Length by Class (Boxplot)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Add mean markers
    means = [d.mean() for d in data]
    ax.scatter([1, 2], means, color="black", s=80, zorder=5, marker="D", 
               label="Mean", edgecolors="white", linewidth=1)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "boxplot_length_by_class.png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'boxplot_length_by_class.png')}")
    plt.close()

if __name__ == "__main__":
    main()