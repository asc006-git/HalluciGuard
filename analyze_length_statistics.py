import pandas as pd
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
    print(f"Total samples: {len(df)}")
    
    print("Extracting answer component...")
    df["knowledge"], df["question"], df["answer"] = zip(*df["text"].apply(extract_parts))
    df["answer_clean"] = df["answer"].apply(clean_text)
    
    df["char_count"] = df["answer_clean"].str.len()
    df["word_count"] = df["answer_clean"].str.split().str.len()
    
    stats = []
    for label in [0, 1]:
        subset = df[df["label"] == label]
        label_name = "Factual" if label == 0 else "Hallucination"
        n = len(subset)
        
        char_mean = subset["char_count"].mean()
        char_median = subset["char_count"].median()
        char_std = subset["char_count"].std()
        char_min = subset["char_count"].min()
        char_max = subset["char_count"].max()
        
        word_mean = subset["word_count"].mean()
        word_median = subset["word_count"].median()
        word_std = subset["word_count"].std()
        word_min = subset["word_count"].min()
        word_max = subset["word_count"].max()
        
        stats.append({
            "label": label,
            "class": label_name,
            "n_samples": n,
            "char_mean": char_mean,
            "char_median": char_median,
            "char_std": char_std,
            "char_min": char_min,
            "char_max": char_max,
            "word_mean": word_mean,
            "word_median": word_median,
            "word_std": word_std,
            "word_min": word_min,
            "word_max": word_max
        })
        
        print(f"\n=== {label_name} (Label {label}) ===")
        print(f"  Samples: {n}")
        print(f"  Character length: mean={char_mean:.1f}, median={char_median:.1f}, std={char_std:.1f}, min={char_min}, max={char_max}")
        print(f"  Word count:       mean={word_mean:.1f}, median={word_median:.1f}, std={word_std:.1f}, min={word_min}, max={word_max}")
    
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(os.path.join(RESULTS_DIR, "features/length_statistics.csv"), index=False)
    print(f"Saved: {os.path.join(RESULTS_DIR, 'features/length_statistics.csv')}")

if __name__ == "__main__":
    main()