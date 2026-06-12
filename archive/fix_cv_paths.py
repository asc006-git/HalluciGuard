with open('cross_validation_study.py', 'r') as f:
    content = f.read()

content = content.replace('os.path.join(RESULTS_DIR, "cross_validation_results.csv")', 'os.path.join(RESULTS_DIR, "metrics/cross_validation_results.csv")')
content = content.replace('os.path.join(RESULTS_DIR, "cross_validation_summary.json")', 'os.path.join(RESULTS_DIR, "metrics/cross_validation_summary.json")')
content = content.replace('os.path.join(RESULTS_DIR, "cross_validation_summary.txt")', 'os.path.join(RESULTS_DIR, "metrics/cross_validation_summary.txt")')

with open('cross_validation_study.py', 'w') as f:
    f.write(content)
print('Fixed cross_validation_study.py output paths')