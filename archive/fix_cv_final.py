with open('cross_validation_study.py', 'r') as f:
    content = f.read()

# Fix print statements - use different quote style
content = content.replace(
    'print(f"\\nSaved: {os.path.join(RESULTS_DIR, "metrics/cross_validation_results.csv")}")',
    'print("\\nSaved: " + os.path.join(RESULTS_DIR, "metrics/cross_validation_results.csv"))'
)
content = content.replace(
    'print(f"Saved: {os.path.join(RESULTS_DIR, "metrics/cross_validation_summary.json")}")',
    'print("Saved: " + os.path.join(RESULTS_DIR, "metrics/cross_validation_summary.json"))'
)
content = content.replace(
    'print(f"Saved: {os.path.join(RESULTS_DIR, "metrics/cross_validation_summary.txt")}")',
    'print("Saved: " + os.path.join(RESULTS_DIR, "metrics/cross_validation_summary.txt"))'
)

with open('cross_validation_study.py', 'w') as f:
    f.write(content)
print('Fixed print statements')