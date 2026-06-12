import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def main():
    print("Loading original HaluEval dataset...")
    dataset = load_dataset("pminervini/HaluEval", "qa")
    split_name = list(dataset.keys())[0]
    df_orig = pd.DataFrame(dataset[split_name])
    print(f"Original rows: {len(df_orig)}")
    
    print("\nSplitting original paired rows (80/20)...")
    df_train_orig, df_test_orig = train_test_split(
        df_orig, test_size=0.2, random_state=42
    )
    print(f"Train pairs: {len(df_train_orig)}")
    print(f"Test pairs:  {len(df_test_orig)}")
    
    def expand_to_binary(df_split, split_name):
        records = []
        for _, row in df_split.iterrows():
            knowledge = str(row["knowledge"])
            question = str(row["question"])
            right_answer = str(row["right_answer"])
            hallucinated_answer = str(row["hallucinated_answer"])
            
            context = f"Knowledge: {knowledge}\nQuestion: {question}\nAnswer:"
            
            records.append({
                "text": f"{context} {right_answer}",
                "label": 0,
                "knowledge": knowledge,
                "question": question,
                "split": split_name
            })
            records.append({
                "text": f"{context} {hallucinated_answer}",
                "label": 1,
                "knowledge": knowledge,
                "question": question,
                "split": split_name
            })
        return pd.DataFrame(records)
    
    print("\nExpanding train split...")
    df_train = expand_to_binary(df_train_orig, "train")
    print(f"Expanded train samples: {len(df_train)}")
    
    print("Expanding test split...")
    df_test = expand_to_binary(df_test_orig, "test")
    print(f"Expanded test samples: {len(df_test)}")
    
    print("\nVerifying no leakage...")
    train_pairs = set(zip(df_train["knowledge"], df_train["question"]))
    test_pairs = set(zip(df_test["knowledge"], df_test["question"]))
    overlap = train_pairs & test_pairs
    print(f"Unique train pairs: {len(train_pairs)}")
    print(f"Unique test pairs:  {len(test_pairs)}")
    print(f"Overlap: {len(overlap)} (should be 0)")
    
    if len(overlap) == 0:
        print("OK - NO LEAKAGE - splits are clean")
    else:
        print("ERROR - LEAKAGE DETECTED!")
        for pair in list(overlap)[:3]:
            print(f"  Leaked: {pair[0][:50]}... | {pair[1][:50]}...")
    
    df_train[["text", "label"]].to_csv("data/processed/train.csv", index=False)
    df_test[["text", "label"]].to_csv("data/processed/test.csv", index=False)
    print("\nSaved: data/processed/train.csv, data/processed/test.csv")
    
    print("\n=== Class Distribution ===")
    print("Train:")
    print(df_train["label"].value_counts().sort_index())
    print("Test:")
    print(df_test["label"].value_counts().sort_index())

if __name__ == "__main__":
    main()