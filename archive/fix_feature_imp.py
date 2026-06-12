with open('plot_answer_only_feature_importance.py', 'r') as f:
    content = f.read()

content = content.replace(
    "df_all.to_csv('results/answer_only_top_features.csv', index=False)",
    "df_all.to_csv(os.path.join(RESULTS_DIR, 'features/answer_only_top_features.csv'), index=False)"
)
content = content.replace(
    'print("Saved: results/plots/plots/plots/answer_only_top_features.csv")',
    'print("Saved: " + os.path.join(RESULTS_DIR, "features/answer_only_top_features.csv"))'
)
content = content.replace(
    'print("Saved: results/plots/plots/plots/answer_only_feature_importance.png")',
    'print("Saved: " + os.path.join(RESULTS_DIR, "plots/answer_only_feature_importance.png"))'
)

# Add RESULTS_DIR and os import if not present
if 'RESULTS_DIR' not in content:
    content = content.replace('import matplotlib.pyplot as plt', 
                             'import os\nRESULTS_DIR = "results"\nos.makedirs(os.path.join(RESULTS_DIR, "features"), exist_ok=True)\nos.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)\nimport matplotlib.pyplot as plt')

with open('plot_answer_only_feature_importance.py', 'w') as f:
    f.write(content)
print('Fixed plot_answer_only_feature_importance.py')