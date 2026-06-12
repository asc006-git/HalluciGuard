with open('ablation_study.py', 'r') as f:
    content = f.read()

content = content.replace(
    'os.path.join(RESULTS_DIR, "ablation_results.csv")',
    'os.path.join(RESULTS_DIR, "metrics/ablation_results.csv")'
)

with open('ablation_study.py', 'w') as f:
    f.write(content)
print('Fixed ablation_study.py')