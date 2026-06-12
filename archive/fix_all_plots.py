# Fix all plotting scripts to save to results/plots/ and update print statements
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
    
    # Fix print statements to match actual save paths
    content = content.replace("print(\"Saved: results/", 'print("Saved: results/plots/')
    
    with open(script, 'w') as f:
        f.write(content)
    print('Fixed ' + script)