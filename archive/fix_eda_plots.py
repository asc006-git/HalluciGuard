with open('eda_plots.py', 'r') as f:
    content = f.read()

content = content.replace('os.makedirs("eda_outputs", exist_ok=True)', 'os.makedirs("docs/assets/eda", exist_ok=True)')
content = content.replace('print("Saved: eda_outputs/class_distribution.png")', 'print("Saved: docs/assets/eda/class_distribution.png")')
content = content.replace('print("Saved: eda_outputs/text_length_histogram.png")', 'print("Saved: docs/assets/eda/text_length_histogram.png")')
content = content.replace('print("Saved: eda_outputs/word_count_histogram.png")', 'print("Saved: docs/assets/eda/word_count_histogram.png")')
content = content.replace('print("\\nAll plots saved to eda_outputs/")', 'print("\\nAll plots saved to docs/assets/eda/")')

with open('eda_plots.py', 'w') as f:
    f.write(content)
print('Fixed eda_plots.py')