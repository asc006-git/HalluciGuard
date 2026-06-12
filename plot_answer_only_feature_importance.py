import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("Loading model and vectorizer...")
    clf = joblib.load("models/answer_only_model.pkl")
    vectorizer = joblib.load("models/answer_only_vectorizer.pkl")

    feature_names = np.array(vectorizer.get_feature_names_out())
    coef = clf.coef_[0]

    top_positive_idx = np.argsort(coef)[-20:][::-1]
    top_negative_idx = np.argsort(coef)[:20]

    pos_features = feature_names[top_positive_idx]
    pos_coef = coef[top_positive_idx]
    neg_features = feature_names[top_negative_idx]
    neg_coef = coef[top_negative_idx]

    df_pos = pd.DataFrame({'feature': pos_features, 'coefficient': pos_coef, 'type': 'Hallucination'})
    df_neg = pd.DataFrame({'feature': neg_features, 'coefficient': neg_coef, 'type': 'Factual'})
    df_all = pd.concat([df_pos, df_neg], ignore_index=True)
    df_all.to_csv(os.path.join(RESULTS_DIR, 'features/answer_only_top_features.csv'), index=False)
    print("Saved: " + os.path.join(RESULTS_DIR, "features/answer_only_top_features.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    y_pos = np.arange(len(pos_features))
    ax1.barh(y_pos, pos_coef, color='#E74C3C', edgecolor='black', height=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(pos_features, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel('Coefficient Value', fontsize=12)
    ax1.set_title('Top 20 Hallucination Indicators\n(Positive Coefficients)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(pos_coef):
        ax1.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)

    y_neg = np.arange(len(neg_features))
    ax2.barh(y_neg, neg_coef, color='#2E86AB', edgecolor='black', height=0.7)
    ax2.set_yticks(y_neg)
    ax2.set_yticklabels(neg_features, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel('Coefficient Value', fontsize=12)
    ax2.set_title('Top 20 Factual Indicators\n(Negative Coefficients)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(neg_coef):
        ax2.text(v - 0.02, i, f'{v:.3f}', va='center', ha='right', fontsize=9)

    fig.suptitle('Feature Importance - Answer-Only Model', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/plots/answer_only_feature_importance.png', dpi=300, bbox_inches='tight')
    print("Saved: " + os.path.join(RESULTS_DIR, "plots/answer_only_feature_importance.png"))
    plt.close()

if __name__ == "__main__":
    main()