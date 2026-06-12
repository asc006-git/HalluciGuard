## Repository Structure

```
HalluciGuard/
├── requirements.txt                    # System dependencies
├── create_splits.py                    # Pair-level clean stratified splits
├── check_leakage.py                    # Diagnostic data leakage demo
├── analyze_length_statistics.py        # Length statistics computation
├── train_answer_only.py                # Answer-Only Logistic Regression training
├── train_linear_svc.py                 # LinearSVC training on Answer-Only
├── cross_validation_study.py           # Stratified 5-Fold Cross-Validation (leakage-free)
├── evaluate_baselines.py               # Random and Majority class baseline evaluation
├── evaluate_length_baseline.py         # Length-Only baseline evaluation
├── ablation_study.py                   # Input configuration ablation study
├── create_final_comparison.py          # Final comparison table generator
├── predict.py                          # CLI and interactive inference pipeline
├── eda_plots.py                        # Statistical distributions plotting
├── plot_answer_only_confusion_matrix.py
├── plot_answer_only_roc_curve.py
├── plot_answer_only_precision_recall_curve.py
├── plot_answer_only_feature_importance.py
├── plot_cv_results.py
├── plot_length_distribution.py
├── plot_wordcount_distribution.py
├── plot_length_boxplot.py
├── data/
│   ├── raw/
│   │   └── halluciguard_dataset.csv    # Full binary dataset (20K samples)
│   └── processed/
│       ├── train.csv                   # Clean train split (16K samples)
│       └── test.csv                    # Clean test split (4K samples)
├── archive/                            # Legacy scripts and data
│   ├── legacy_fulltext_model/
│   │   ├── .py
│   │   └── .py
│   ├── create_binary_dataset.py
│   ├── eda_basic.py
│   ├── inspect_halueval.py
│   ├── top_words.py
│   └── train_baseline.py
├── models/                             # Serialized production models
│   ├── answer_only_model.pkl
│   └── answer_only_vectorizer.pkl
└── results/                            # Metrics, features, and visualizations
    ├── metrics/
    │   ├── ablation_results.csv
    │   ├── model_comparison.csv
    │   ├── cross_validation_results.csv
    │   ├── cross_validation_summary.json
    │   ├── baseline_comparison.csv
    │   └── length_baseline_results.csv
    ├── features/
    │   ├── answer_only_
    │   └── length_statistics.csv
    └── plots/
        ├── answer_only_confusion_matrix.png
        ├── answer_only_roc_curve.png
        ├── answer_only_precision_recall_curve.png
        ├── answer_only_feature_importance.png
        ├── cv_f1_distribution.png
        ├── cv_metric_comparison.png
        ├── length_distribution_by_class.png
        ├── wordcount_distribution_by_class.png
        └── boxplot_length_by_class.png
```

