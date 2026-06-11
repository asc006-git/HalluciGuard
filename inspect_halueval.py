import pandas as pd
from datasets import load_dataset

def main():
    print("Downloading HaluEval dataset...")
    dataset = load_dataset("pminervini/HaluEval", "qa")
    
    print(f"Available splits: {list(dataset.keys())}")
    split_name = list(dataset.keys())[0]
    df = pd.DataFrame(dataset[split_name])
    
    print(f"\n=== Dataset Overview ===")
    print(f"Number of rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    print(f"\n=== Sample Records (first 3) ===")
    pd.set_option('display.max_colwidth', 100)
    pd.set_option('display.width', 200)
    print(df.head(3).to_string())
    
    print(f"\n=== Dataset Structure ===")
    print("This is a PAIRED dataset - each row contains:")
    print("  1. A question + knowledge context")
    print("  2. The correct answer (right_answer)")
    print("  3. A hallucinated answer (hallucinated_answer)")
    print("\nFor training, you would create binary labels by pairing:")
    print("  - (question, knowledge, right_answer) -> label=0 (factual)")
    print("  - (question, knowledge, hallucinated_answer) -> label=1 (hallucination)")
    
    print(f"\n=== Column Descriptions ===")
    column_descriptions = {
        "knowledge": "Ground truth knowledge/reference text used to answer the question",
        "question": "The question posed to the model",
        "right_answer": "The factually correct answer based on the knowledge",
        "hallucinated_answer": "An incorrect/hallucinated answer that appears plausible but contradicts the knowledge",
    }
    
    for col in df.columns:
        desc = column_descriptions.get(col, "No description available")
        print(f"  {col}: {desc}")
    
    print(f"\n=== Data Quality Checks ===")
    print(f"Missing values per column:")
    print(df.isnull().sum())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print(f"\nAnswer length stats:")
    for col in ["right_answer", "hallucinated_answer"]:
        if col in df.columns:
            lengths = df[col].str.len()
            print(f"  {col}: mean={lengths.mean():.1f}, min={lengths.min()}, max={lengths.max()}")

if __name__ == "__main__":
    main()