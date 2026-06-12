# Update create_final_comparison.py
with open('create_final_comparison.py', 'r') as f:
    content = f.read()

content = content.replace('pd.read_csv(os.path.join(RESULTS_DIR, "baseline_comparison.csv"))', 
                         'pd.read_csv(os.path.join(RESULTS_DIR, "metrics/baseline_comparison.csv"))')
content = content.replace('pd.read_csv(os.path.join(RESULTS_DIR, "length_baseline_results.csv"))', 
                         'pd.read_csv(os.path.join(RESULTS_DIR, "metrics/length_baseline_results.csv"))')
content = content.replace('pd.read_csv(os.path.join(RESULTS_DIR, "ablation_results.csv"))', 
                         'pd.read_csv(os.path.join(RESULTS_DIR, "metrics/ablation_results.csv"))')
content = content.replace('open(os.path.join(RESULTS_DIR, "cross_validation_summary.json"))', 
                         'open(os.path.join(RESULTS_DIR, "metrics/cross_validation_summary.json"))')
content = content.replace('comparison_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)', 
                         'comparison_df.to_csv(os.path.join(RESULTS_DIR, "metrics/model_comparison.csv"), index=False)')

with open('create_final_comparison.py', 'w') as f:
    f.write(content)
print('Updated create_final_comparison.py')

# Update predict.py - already has correct paths
with open('predict.py', 'r') as f:
    content = f.read()

# Check if paths are correct
if 'models/answer_only_model.pkl' in content and 'models/answer_only_vectorizer.pkl' in content:
    print('predict.py paths already correct')
else:
    print('WARNING: predict.py may need update')

with open('predict.py', 'w') as f:
    f.write(content)
print('Verified predict.py')