import pandas as pd
import numpy as np

def main():
    print("Loading dataset...")
    df = pd.read_csv("halluciguard_dataset.csv")
    
    print(f"\n=== Dataset Overview ===")
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    print(f"\n=== Class Distribution ===")
    class_dist = df["label"].value_counts().sort_index()
    print(class_dist)
    print(f"  0 (factual): {class_dist.get(0, 0)}")
    print(f"  1 (hallucination): {class_dist.get(1, 0)}")
    
    print(f"\n=== Duplicate Analysis ===")
    duplicate_count = df.duplicated(subset=["text"]).sum()
    print(f"Duplicate texts: {duplicate_count}")
    if duplicate_count > 0:
        print(f"Duplicate examples:")
        print(df[df.duplicated(subset=["text"], keep=False)].sort_values("text").head(10))
    
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    
    print(f"\n=== Text Length Statistics (characters) ===")
    for label in [0, 1]:
        subset = df[df["label"] == label]["text_length"]
        print(f"\n  Label {label} ({'factual' if label == 0 else 'hallucination'}):")
        print(f"    Mean:   {subset.mean():.1f}")
        print(f"    Median: {subset.median():.1f}")
        print(f"    Min:    {subset.min()}")
        print(f"    Max:    {subset.max()}")
        print(f"    Std:    {subset.std():.1f}")
    
    print(f"\n=== Word Count Statistics ===")
    for label in [0, 1]:
        subset = df[df["label"] == label]["word_count"]
        print(f"\n  Label {label} ({'factual' if label == 0 else 'hallucination'}):")
        print(f"    Mean:   {subset.mean():.1f}")
        print(f"    Median: {subset.median():.1f}")
        print(f"    Min:    {subset.min()}")
        print(f"    Max:    {subset.max()}")
        print(f"    Std:    {subset.std():.1f}")
    
    print(f"\n=== Extremely Short Texts (< 50 chars) ===")
    short_texts = df[df["text_length"] < 50]
    print(f"Count: {len(short_texts)}")
    if len(short_texts) > 0:
        print(short_texts[["text", "label", "text_length"]].to_string())
    
    print(f"\n=== Extremely Long Texts (> 2000 chars) ===")
    long_texts = df[df["text_length"] > 2000]
    print(f"Count: {len(long_texts)}")
    if len(long_texts) > 0:
        print(long_texts[["text", "label", "text_length"]].head(5).to_string())
    
    print(f"\n=== Overall Text Length Percentiles ===")
    print(df["text_length"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))

if __name__ == "__main__":
    main()