with open('create_splits.py', 'r') as f:
    content = f.read()

content = content.replace('print("\\nSaved: train.csv, test.csv")', 
                         'print("\\nSaved: data/processed/train.csv, data/processed/test.csv")')

with open('create_splits.py', 'w') as f:
    f.write(content)
print('Fixed print statement')