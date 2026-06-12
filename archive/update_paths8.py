# Update plotting scripts
plot_scripts = [
    'plot_answer_only_confusion_matrix.py',
    'plot_answer_only_roc_curve.py',
    'plot_answer_only_precision_recall_curve.py',
    'plot_answer_only_feature_importance.py',
    'plot_cv_results.py',
    'plot_length_distribution.py',
    'plot_wordcount_distribution.py',
    'plot_length_boxplot.py',
]

for script in plot_scripts:
    with open(script, 'r') as f:
        content = f.read()
    
    # Update data paths
    content = content.replace('pd.read_csv("train.csv")', 'pd.read_csv("data/processed/train.csv")')
    content = content.replace('pd.read_csv("test.csv")', 'pd.read_csv("data/processed/test.csv")')
    content = content.replace('pd.read_csv("halluciguard_dataset.csv")', 'pd.read_csv("data/raw/halluciguard_dataset.csv")')
    
    # Update model loading paths
    content = content.replace('joblib.load("models/answer_only_model.pkl")', 'joblib.load("models/answer_only_model.pkl")')
    content = content.replace('joblib.load("models/answer_only_vectorizer.pkl")', 'joblib.load("models/answer_only_vectorizer.pkl")')
    content = content.replace('joblib.load("logistic_model.pkl")', 'joblib.load("models/answer_only_model.pkl")')
    content = content.replace('joblib.load("tfidf_vectorizer.pkl")', 'joblib.load("models/answer_only_vectorizer.pkl")')
    content = content.replace('joblib.load("halluciguard_logreg_model.pkl")', 'joblib.load("models/answer_only_model.pkl")')
    content = content.replace('joblib.load("halluciguard_tfidf_vectorizer.pkl")', 'joblib.load("models/answer_only_vectorizer.pkl")')
    
    # Update output paths
    content = content.replace('plt.savefig(\'results/', 'plt.savefig(\'results/plots/')
    content = content.replace('plt.savefig("results/', 'plt.savefig("results/plots/')
    content = content.replace('df_all.to_csv(\'results/top_features.csv\'', 'df_all.to_csv(\'results/features/answer_only_top_features.csv\'')
    content = content.replace('df_all.to_csv("results/top_features.csv"', 'df_all.to_csv("results/features/answer_only_top_features.csv"')
    content = content.replace('df_all.to_csv(\'results/length_statistics.csv\'', 'df_all.to_csv(\'results/features/length_statistics.csv\'')
    content = content.replace('df_all.to_csv("results/length_statistics.csv"', 'df_all.to_csv("results/features/length_statistics.csv"')
    
    # Update eda_outputs paths
    content = content.replace('plt.savefig("eda_outputs/', 'plt.savefig("docs/assets/eda/')
    
    with open(script, 'w') as f:
        f.write(content)
    print('Updated ' + script)