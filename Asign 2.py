import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Sample Data ----------------
# Simulated RNA-Seq count data (genes x samples)
# Normally, this comes from an RNA-Seq pipeline (HTSeq, featureCounts, etc.)
data = {
    "Gene": ["Gene1", "Gene2", "Gene3", "Gene4", "Gene5", "Gene6"],
    "Control_1": [50, 200, 500, 20, 5, 100],
    "Control_2": [45, 210, 480, 25, 8, 95],
    "Treatment_1": [80, 400, 450, 60, 15, 300],
    "Treatment_2": [75, 420, 470, 55, 20, 280]
}

df = pd.DataFrame(data)
df.set_index("Gene", inplace=True)

print("\n--- RNA-Seq Count Data ---")
print(df)

# ---------------- Normalization ----------------
# Normalize counts using log2 transformation
log_df = np.log2(df + 1)

print("\n--- Log2 Transformed Counts ---")
print(log_df)

# ---------------- Differential Expression ----------------
# Perform t-test between Control and Treatment for each gene
results = []
for gene in df.index:
    control_vals = df.loc[gene, ["Control_1", "Control_2"]]
    treatment_vals = df.loc[gene, ["Treatment_1", "Treatment_2"]]
    t_stat, p_val = stats.ttest_ind(treatment_vals, control_vals)
    fold_change = (treatment_vals.mean() + 1) / (control_vals.mean() + 1)  # simple fold-change
    log2_fc = np.log2(fold_change)
    results.append([gene, log2_fc, p_val])

results_df = pd.DataFrame(results, columns=["Gene", "Log2_FC", "P_Value"])
results_df["Significant"] = results_df["P_Value"] < 0.05

print("\n--- Differential Expression Results ---")
print(results_df)

# ---------------- Visualization ----------------
# Volcano Plot
plt.figure(figsize=(8,6))
sns.scatterplot(data=results_df, x="Log2_FC", y=-np.log10(results_df["P_Value"]),
                hue="Significant", palette={True: "red", False: "gray"}, s=100)
plt.axhline(-np.log10(0.05), color="blue", linestyle="--")
plt.xlabel("Log2 Fold Change")
plt.ylabel("-Log10 P-Value")
plt.title("Volcano Plot of Differential Gene Expression")
plt.show()

# Heatmap of normalized expression
plt.figure(figsize=(8,6))
sns.heatmap(log_df, cmap="viridis", annot=True)
plt.title("Heatmap of Log2 Normalized RNA-Seq Counts")
plt.show()
