# Update train_answer_only.py
with open('train_answer_only.py', 'r') as f:
    content = f.read()

content = content.replace('pd.read_csv("train.csv")', 'pd.read_csv("data/processed/train.csv")')
content = content.replace('pd.read_csv("test.csv")', 'pd.read_csv("data/processed/test.csv")')

with open('train_answer_only.py', 'w') as f:
    f.write(content)
print('Updated train_answer_only.py')

# Update train_linear_svc.py
with open('train_linear_svc.py', 'r') as f:
    content = f.read()

content = content.replace('pd.read_csv("train.csv")', 'pd.read_csv("data/processed/train.csv")')
content = content.replace('pd.read_csv("test.csv")', 'pd.read_csv("data/processed/test.csv")')

with open('train_linear_svc.py', 'w') as f:
    f.write(content)
print('Updated train_linear_svc.py')

# Update cross_validation_study.py
with open('cross_validation_study.py', 'r') as f:
    content = f.read()

content = content.replace('pd.read_csv("train.csv")', 'pd.read_csv("data/processed/train.csv")')
content = content.replace('pd.read_csv("test.csv")', 'pd.read_csv("data/processed/test.csv")')

with open('cross_validation_study.py', 'w') as f:
    f.write(content)
print('Updated cross_validation_study.py')