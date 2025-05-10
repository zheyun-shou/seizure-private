import matplotlib.pyplot as plt
import pandas as pd

# 读入转置后的 CSV
df = pd.read_csv("./results/relevance_values_transposed.csv", index_col=0)

plt.figure(figsize=(14, 6))

for model_name in df.index:
    plt.scatter(df.columns, df.loc[model_name], label=model_name)

plt.xlabel('EEG Channel')
plt.ylabel('Relevance Value')
plt.title('Relevance Values Across EEG Channels (Scatter Plot)')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()
