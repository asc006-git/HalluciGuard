# Update plotting scripts
plots = [
    'plot_answer_only_confusion_matrix.py',
    'plot_answer_only_roc_curve.py',
    'plot_answer_only_precision_recall_curve.py',
    'plot_answer_only_feature_importance.py',
    'plot_cv_results.py',
    'plot_length_distribution.py',
    'plot_wordcount_distribution.py',
    'plot_length_boxplot.py',
]

for script in plots:
    with open(script, 'r') as f:
        content = f.read()
    
    # Update data paths
    content = content.replace('pd.read_csv("train.csv")', 'pd.read_csv("data/processed/train.csv")')
    content = content.replace('pd.read_csv("test.csv")', 'pd.read_csv("data/processed/test.csv")')
    content = content.replace('pd.read_csv("halluciguard_dataset.csv")', 'pd.read_csv("data/raw/halluciguard_dataset.csv")')
    
    # Update model loading paths (already correct for models/ but double-check)
    content = content.replace('joblib.load("models/answer_only_model.pkl")', 'joblib.load("models/answer_only_model.pkl")')
    content = content.replace('joblib.load("models/answer_only_vectorizer.pkl")', 'joblib.load("models/answer_only_vectorizer.pkl")')
    
    # Update output paths - ensure they point to results/plots/
    content = content.replace('plt.savefig("results/', 'plt.savefig("results/plots/')
    content = content.replace('plt.savefig("results/', 'plt.savefig("results/plots/')
    
    # Update feature importance output
    content = content.replace("df_all.to_csv('results/answer_only_top_features.csv', index=False)", 
                             "df_all.to_csv(os.path.join(RESULTS_DIR, 'features/answer_only_top_features.csv'), index=False)")
    content = content.replace('df_all.to_csv("results/answer_only_top_features.csv", index=False)', 
                             'df_all.to_csv(os.path.join(RESULTS_DIR, "features/answer_only_top_features.csv"), index=False)')
    
    # Update eda_outputs paths
    content = content.replace('plt.savefig("eda_outputs/', 'plt.savefig("docs/assets/eda/')
    
    # Add RESULTS_DIR and os import if not present
    if 'RESULTS_DIR' not in content:
        content = content.replace('import matplotlib.pyplot as plt', 
                                 'import os\nRESULTS_DIR = "results"\nos.makedirs(os.path.join(RESULTS_DIR, "features"), exist_ok=True)\nos.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)\nimport matplotlib.pyplot as plt')
    
    with open(script, 'w') as f:
        f.write(content)
    print('Updated ' + script)