import re

# Update create_splits.py
with open('create_splits.py', 'r') as f:
    content = f.read()

content = content.replace('to_csv("train.csv", index=False)', 'to_csv("data/processed/train.csv", index=False)')
content = content.replace('to_csv("test.csv", index=False)', 'to_csv("data/processed/test.csv", index=False)')

with open('create_splits.py', 'w') as f:
    f.write(content)
print('Updated create_splits.py')