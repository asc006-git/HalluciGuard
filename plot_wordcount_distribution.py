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
    df["word_count"] = df["answer_clean"].str.split().str.len()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for label, color, label_name in [(0, "#2E86AB", "Factual"), (1, "#E74C3C", "Hallucination")]:
        subset = df[df["label"] == label]["word_count"]
        ax.hist(subset, bins=50, alpha=0.6, label=label_name, color=color, 
                edgecolor="black", linewidth=0.5, density=True)
    
    ax.set_xlabel("Word Count", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Answer Word Count Distribution by Class", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    mean_factual = df[df["label"] == 0]["word_count"].mean()
    mean_halluc = df[df["label"] == 1]["word_count"].mean()
    ax.axvline(mean_factual, color="#2E86AB", linestyle="--", linewidth=2, 
               label=f"Factual Mean = {mean_factual:.1f}")
    ax.axvline(mean_halluc, color="#E74C3C", linestyle="--", linewidth=2,
               label=f"Hallucination Mean = {mean_halluc:.1f}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "wordcount_distribution_by_class.png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'wordcount_distribution_by_class.png')}")
    plt.close()

if __name__ == "__main__":
    main()