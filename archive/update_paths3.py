# Update evaluate_length_baseline.py
with open('evaluate_length_baseline.py', 'r') as f:
    content = f.read()

content = content.replace('pd.read_csv("train.csv")', 'pd.read_csv("data/processed/train.csv")')
content = content.replace('pd.read_csv("test.csv")', 'pd.read_csv("data/processed/test.csv")')

with open('evaluate_length_baseline.py', 'w') as f:
    f.write(content)
print('Updated evaluate_length_baseline.py')

# Update evaluate_baselines.py
with open('evaluate_baselines.py', 'r') as f:
    content = f.read()

content = content.replace('pd.read_csv("test.csv")', 'pd.read_csv("data/processed/test.csv")')

with open('evaluate_baselines.py', 'w') as f:
    f.write(content)
print('Updated evaluate_baselines.py')

# Update ablation_study.py
with open('ablation_study.py', 'r') as f:
    content = f.read()

content = content.replace('pd.read_csv("train.csv")', 'pd.read_csv("data/processed/train.csv")')
content = content.replace('pd.read_csv("test.csv")', 'pd.read_csv("data/processed/test.csv")')

with open('ablation_study.py', 'w') as f:
    f.write(content)
print('Updated ablation_study.py')