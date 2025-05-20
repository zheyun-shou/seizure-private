import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 读入转置后的 CSV
df = pd.read_csv("./results/relevance.csv", index_col=0)

# plt.figure(figsize=(14, 6))

# for model_name in df.index:
#     plt.scatter(df.columns, df.loc[model_name], color='grey')

# mean_relevance = df.mean(axis=0)
# plt.scatter(df.columns, mean_relevance, color='red', marker='X', s=100, zorder=5)

# plt.xlabel('EEG Channel')
# plt.ylabel('Relevance Value')
# plt.title('Relevance Values Across EEG Channels')
# plt.xticks(rotation=45)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.grid(True)
# plt.show()
mean_relevance = df.mean(axis=0)
std_relevance = df.std(axis=0)

# --- Create the Plot ---
plt.figure(figsize=(15, 7)) # Adjusted figure size for better readability

# Plot means with error bars for standard deviation
# The 'x' markers are red.
# `capsize` adds small caps to the error bars.
plt.errorbar(
    x=df.columns,          # EEG Channel names for the x-axis
    y=mean_relevance,      # Mean relevance values for the y-axis
    yerr=std_relevance,    # Standard deviation for the error bars
    fmt='x',               # Format for the markers ('x' for crosses)
    color='red',           # Color of the markers
    ecolor='darkgrey',     # Color of the error bars
    elinewidth=1.5,        # Thickness of the error bar lines
    capsize=5,             # Length of the error bar caps
    capthick=1.5,          # Thickness of the error bar caps
    markersize=8,          # Size of the 'x' markers
    markeredgewidth=2,     # Thickness of the 'x' marker lines
    label='Mean Relevance +/- Std Dev' # Label for the legend
)

# --- Customize the Plot ---
plt.xlabel('EEG Channel', fontsize=14)
plt.ylabel('Relevance Value', fontsize=14)
plt.title('Relevance Values Across EEG Channels (Mean +/- Std Dev)', fontsize=16)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)

# Add a legend
plt.legend(fontsize=12)

# Ensure everything fits without overlapping
plt.tight_layout()

# Add a grid for better visual structure
plt.grid(True, linestyle='-', alpha=0.7)

# --- Show or Save the Plot ---
# To save the figure, uncomment the line below:
# plt.savefig('./results/relevance_plot_with_std.png', dpi=300)
plt.show()

print("Plot generation complete.")
print("\nMean relevance per channel:\n", mean_relevance)
print("\nStandard deviation per channel:\n", std_relevance)
