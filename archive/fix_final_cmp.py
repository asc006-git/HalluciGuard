with open('create_final_comparison.py', 'r') as f:
    content = f.read()

content = content.replace(
    'os.path.join(RESULTS_DIR, "cross_validation_summary.json")',
    'os.path.join(RESULTS_DIR, "metrics/cross_validation_summary.json")'
)

with open('create_final_comparison.py', 'w') as f:
    f.write(content)
print('Fixed create_final_comparison.py')