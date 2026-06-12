with open('evaluate_length_baseline.py', 'r') as f:
    content = f.read()

content = content.replace(
    "print(f\"\\nSaved: {os.path.join(RESULTS_DIR, 'length_baseline_results.csv')}\")",
    'print("\\nSaved: " + os.path.join(RESULTS_DIR, "metrics/length_baseline_results.csv"))'
)

with open('evaluate_length_baseline.py', 'w') as f:
    f.write(content)
print('Fixed print statement')