with open('evaluate_length_baseline.py', 'r') as f:
    content = f.read()

content = content.replace('os.path.join(RESULTS_DIR, "length_baseline_results.csv")', 'os.path.join(RESULTS_DIR, "metrics/length_baseline_results.csv")')

with open('evaluate_length_baseline.py', 'w') as f:
    f.write(content)
print('Fixed evaluate_length_baseline.py')