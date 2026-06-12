with open('analyze_length_statistics.py', 'r') as f:
    content = f.read()

content = content.replace('pd.read_csv("train.csv")', 'pd.read_csv("data/processed/train.csv")')
content = content.replace('pd.read_csv("test.csv")', 'pd.read_csv("data/processed/test.csv")')
content = content.replace('os.path.join(RESULTS_DIR, "length_statistics.csv")', 'os.path.join(RESULTS_DIR, "features/length_statistics.csv")')

with open('analyze_length_statistics.py', 'w') as f:
    f.write(content)
print('Fixed analyze_length_statistics.py')