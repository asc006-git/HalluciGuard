with open('analyze_length_statistics.py', 'r') as f:
    content = f.read()

content = content.replace(
    "print(f\"\\nSaved: {os.path.join(RESULTS_DIR, 'length_statistics.csv')}\")",
    "print(f\"Saved: {os.path.join(RESULTS_DIR, 'features/length_statistics.csv')}\")"
)

with open('analyze_length_statistics.py', 'w') as f:
    f.write(content)
print('Fixed print statement')