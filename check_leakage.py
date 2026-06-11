import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def extract_knowledge_question(text):
    """Extract knowledge and question from text to identify unique pairs."""
    knowledge = ""
    question = ""
    if "Knowledge:" in text and "Question:" in text:
        parts = text.split("Question:")
        if len(parts) >= 2:
            knowledge = parts[0].replace("Knowledge:", "").strip()
            question = parts[1].split("Answer:")[0].strip()
    return f"{knowledge}|||{question}"

def main():
    print("=== Loading original HaluEval dataset ===")
    dataset = load_dataset("pminervini/HaluEval", "qa")
    split_name = list(dataset.keys())[0]
    df_orig = pd.DataFrame(dataset[split_name])
    print(f"Original rows: {len(df_orig)}")
    print(f"Columns: {list(df_orig.columns)}")
    
    print("\n=== Creating paired binary dataset (correct method) ===")
    records = []
    for _, row in df_orig.iterrows():
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
            "pair_id": len(records) // 2
        })
        records.append({
            "text": f"{context} {hallucinated_answer}",
            "label": 1,
            "knowledge": knowledge,
            "question": question,
            "pair_id": len(records) // 2
        })
    
    df_binary = pd.DataFrame(records)
    print(f"Binary dataset rows: {len(df_binary)}")
    
    print("\n=== Checking for duplicate knowledge-question pairs in original ===")
    df_orig["kq_pair"] = df_orig.apply(lambda r: f"{r['knowledge']}|||{r['question']}", axis=1)
    dup_kq = df_orig["kq_pair"].duplicated().sum()
    print(f"Duplicate knowledge-question pairs in original: {dup_kq}")
    
    print("\n=== Method 1: Split AFTER expansion (current approach - WRONG) ===")
    X = df_binary["text"]
    y = df_binary["label"]
    X_train_wrong, X_test_wrong, y_train_wrong, y_test_wrong = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    train_kq_wrong = set()
    for text in X_train_wrong:
        train_kq_wrong.add(extract_knowledge_question(text))
    
    test_kq_wrong = set()
    for text in X_test_wrong:
        test_kq_wrong.add(extract_knowledge_question(text))
    
    overlap_wrong = train_kq_wrong & test_kq_wrong
    print(f"Unique K-Q pairs in train: {len(train_kq_wrong)}")
    print(f"Unique K-Q pairs in test:  {len(test_kq_wrong)}")
    print(f"OVERLAP (LEAKAGE): {len(overlap_wrong)} pairs appear in BOTH train and test")
    
    if overlap_wrong:
        print("\nExample leaked pairs:")
        for pair in list(overlap_wrong)[:3]:
            print(f"  {pair[:100]}...")
    
    print("\n=== Method 2: Split BEFORE expansion (correct approach) ===")
    df_orig_train, df_orig_test = train_test_split(
        df_orig, test_size=0.2, random_state=42
    )
    print(f"Original train: {len(df_orig_train)}, test: {len(df_orig_test)}")
    
    train_kq_correct = set(df_orig_train["kq_pair"])
    test_kq_correct = set(df_orig_test["kq_pair"])
    overlap_correct = train_kq_correct & test_kq_correct
    print(f"Unique K-Q pairs in train: {len(train_kq_correct)}")
    print(f"Unique K-Q pairs in test:  {len(test_kq_correct)}")
    print(f"OVERLAP: {len(overlap_correct)} (should be 0)")
    
    print("\n=== Expanding after split (correct methodology) ===")
    def expand_split(df_split):
        records = []
        for _, row in df_split.iterrows():
            knowledge = str(row["knowledge"])
            question = str(row["question"])
            right_answer = str(row["right_answer"])
            hallucinated_answer = str(row["hallucinated_answer"])
            context = f"Knowledge: {knowledge}\nQuestion: {question}\nAnswer:"
            records.append({"text": f"{context} {right_answer}", "label": 0})
            records.append({"text": f"{context} {hallucinated_answer}", "label": 1})
        return pd.DataFrame(records)
    
    df_train_correct = expand_split(df_orig_train)
    df_test_correct = expand_split(df_orig_test)
    print(f"Expanded train: {len(df_train_correct)}, test: {len(df_test_correct)}")
    
    print("\n=== Summary ===")
    print(f"WRONG method (split after expansion): {len(overlap_wrong)} leaked pairs")
    print(f"CORRECT method (split before expansion): {len(overlap_correct)} leaked pairs")
    print("\nCONCLUSION: Must split at the original paired level, THEN expand to binary classification.")

if __name__ == "__main__":
    main()