
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Load datasets
df_bge_large = pd.read_csv("./dataset.csv")  # bge-large embedding
df_llama3_2 = pd.read_csv("./dataset-l-l.csv")  # llama3.2 embedding

# 1. Summary Statistics


def print_summary_stats(df, label):
    avg_score = df["score"].mean()
    max_score = df["score"].max()
    min_score = df["score"].min()
    print(f"{label} - Average Score: {avg_score:.2f}, Max Score: {max_score}, Min Score: {min_score}")
    return avg_score


print_summary_stats(df_bge_large, "BGE-Large Embedding")
print_summary_stats(df_llama3_2, "Llama3.2 Embedding")

# 2. Visualize Score Distributions
plt.hist(df_bge_large["score"], bins=10, alpha=0.5, label="BGE-Large")
plt.hist(df_llama3_2["score"], bins=10, alpha=0.5, label="Llama3.2")
plt.legend(loc="upper left")
plt.title("Score Distributions")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

# 3. Percentage of High Scores (>=4)


def high_score_percentage(df, label):
    high_scores = (df["score"] >= 4).sum()
    total_scores = len(df)
    percentage = (high_scores / total_scores) * 100
    print(f"{label} - Percentage of Scores >= 4: {percentage:.2f}%")


high_score_percentage(df_bge_large, "BGE-Large Embedding")
high_score_percentage(df_llama3_2, "Llama3.2 Embedding")

# 4. Statistical Significance Test
t_stat, p_value = ttest_ind(df_bge_large["score"], df_llama3_2["score"])
print(
    f"T-test p-value: {p_value:.4f} (p < 0.05 indicates a significant difference)")
