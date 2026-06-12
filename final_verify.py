import pandas as pd
import json

# Load model comparison
comp = pd.read_csv('results/metrics/model_comparison.csv')
print('=== MODEL COMPARISON ===')
print(comp.to_string(index=False))
print()

# Load CV summary
with open('results/metrics/cross_validation_summary.json', 'r') as f:
    cv = json.load(f)
print()
print('=== CROSS-VALIDATION SUMMARY ===')
for metric, vals in cv.items():
    print(metric + ': mean=' + str(vals['mean']) + ', std=' + str(vals['std']) + ', cv%=' + str(vals['cv']*100))

# Load length stats
length = pd.read_csv('results/features/length_statistics.csv')
print()
print('=== LENGTH STATISTICS ===')
print(length.to_string(index=False))