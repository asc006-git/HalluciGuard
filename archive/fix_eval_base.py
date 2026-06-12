with open('evaluate_baselines.py', 'r') as f:
    content = f.read()

content = content.replace(
    'os.path.join(RESULTS_DIR, "baseline_comparison.csv")',
    'os.path.join(RESULTS_DIR, "metrics/baseline_comparison.csv")'
)
content = content.replace(
    "print(f\"\\nSaved: {os.path.join(RESULTS_DIR, 'baseline_comparison.csv')}\")",
    'print("\\nSaved: " + os.path.join(RESULTS_DIR, "metrics/baseline_comparison.csv"))'
)

with open('evaluate_baselines.py', 'w') as f:
    f.write(content)
print('Fixed evaluate_baselines.py')