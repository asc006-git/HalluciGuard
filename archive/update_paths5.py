# Update analyze_length_statistics.py
with open('analyze_length_statistics.py', 'r') as f:
    content = f.read()

content = content.replace('pd.read_csv("train.csv")', 'pd.read_csv("data/processed/train.csv")')
content = content.replace('pd.read_csv("test.csv")', 'pd.read_csv("data/processed/test.csv")')
content = content.replace('os.path.join(RESULTS_DIR, "length_statistics.csv")', 'os.path.join(RESULTS_DIR, "features/length_statistics.csv")')

with open('analyze_length_statistics.py', 'w') as f:
    f.write(content)
print('Updated analyze_length_statistics.py')

# Update eda_plots.py
with open('eda_plots.py', 'r') as f:
    content = f.read()

content = content.replace('pd.read_csv("halluciguard_dataset.csv")', 'pd.read_csv("data/raw/halluciguard_dataset.csv")')
content = content.replace('plt.savefig("eda_outputs/class_distribution.png"', 'plt.savefig("docs/assets/eda/class_distribution.png"')
content = content.replace('plt.savefig("eda_outputs/text_length_histogram.png"', 'plt.savefig("docs/assets/eda/text_length_histogram.png")')
content = content.replace('plt.savefig("eda_outputs/word_count_histogram.png"', 'plt.savefig("docs/assets/eda/word_count_histogram.png")')

with open('eda_plots.py', 'w') as f:
    f.write(content)
print('Updated eda_plots.py')

# Update check_leakage.py
with open('check_leakage.py', 'r') as f:
    content = f.read()

content = content.replace('pd.read_csv("halluciguard_dataset.csv")', 'pd.read_csv("data/raw/halluciguard_dataset.csv")')
content = content.replace('pd.read_csv("train.csv")', 'pd.read_csv("data/processed/train.csv")')
content = content.replace('pd.read_csv("test.csv")', 'pd.read_csv("data/processed/test.csv")')

with open('check_leakage.py', 'w') as f:
    f.write(content)
print('Updated check_leakage.py')