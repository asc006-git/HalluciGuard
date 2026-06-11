import pandas as pd
from datasets import load_dataset

def main():
    print("Loading HaluEval dataset...")
    dataset = load_dataset("pminervini/HaluEval", "qa")
    split_name = list(dataset.keys())[0]
    df = pd.DataFrame(dataset[split_name])
    
    print(f"Original rows: {len(df)}")
    
    records = []
    
    for _, row in df.iterrows():
        knowledge = str(row["knowledge"])
        question = str(row["question"])
        right_answer = str(row["right_answer"])
        hallucinated_answer = str(row["hallucinated_answer"])
        
        context = f"Knowledge: {knowledge}\nQuestion: {question}\nAnswer:"
        
        records.append({
            "text": f"{context} {right_answer}",
            "label": 0
        })
        
        records.append({
            "text": f"{context} {hallucinated_answer}",
            "label": 1
        })
    
    binary_df = pd.DataFrame(records)
    
    print(f"\n=== Binary Classification Dataset ===")
    print(f"Total samples: {len(binary_df)}")
    print(f"Class distribution:")
    print(binary_df["label"].value_counts().sort_index())
    print(f"  0 (factual): {(binary_df['label'] == 0).sum()}")
    print(f"  1 (hallucination): {(binary_df['label'] == 1).sum()}")
    
    print(f"\n=== Sample Records ===")
    pd.set_option('display.max_colwidth', 150)
    pd.set_option('display.width', 200)
    print(binary_df.head(4).to_string())
    
    output_path = "halluciguard_dataset.csv"
    binary_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    print(f"File size: {binary_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()