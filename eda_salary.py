"""
Exploratory Data Analysis (EDA) Project - Salary Dataset
Statistical summaries, distributions, correlations, key influencing factors.
Run:  python eda_salary.py     (keep Salary_Data_Cleaned.csv in the same folder)
Needs: pip install pandas numpy matplotlib seaborn scipy
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

OUT = "output_eda"
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid")
fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k")

df = pd.read_csv("Salary_Data_Cleaned.csv")
print("Shape:", df.shape)

# ------------------------------------------------------------ 1. STATISTICAL SUMMARY
summary = df.describe().T
summary["skew"] = df.select_dtypes("number").skew()
summary["median"] = df.select_dtypes("number").median()
summary.to_csv(f"{OUT}/statistical_summary.csv")
print("\nStatistical summary:\n", summary)

cat_summary = {c: df[c].value_counts() for c in ["Gender", "Education Level"]}
for c, v in cat_summary.items():
    print(f"\n{c}:\n", v)

# ------------------------------------------------------------ 2. CORRELATIONS
corr = df[["Age", "Years of Experience", "Salary"]].corr()
corr.to_csv(f"{OUT}/correlation_matrix.csv")
print("\nCorrelation matrix:\n", corr)

edu_order = ["High School", "Bachelor's Degree", "Master's Degree", "PhD"]
edu_avg = df.groupby("Education Level")["Salary"].mean().reindex(edu_order)
gender_avg = df[df["Gender"].isin(["Male", "Female"])].groupby("Gender")["Salary"].agg(["mean", "median", "count"])
job_avg = df.groupby("Job Title")["Salary"].agg(["mean", "count"])
top_jobs = job_avg[job_avg["count"] >= 5].sort_values("mean", ascending=False).head(10)
df["Exp Band"] = pd.cut(df["Years of Experience"], [-1, 2, 5, 10, 15, 20, 40],
                        labels=["0-2", "3-5", "6-10", "11-15", "16-20", "20+"])
exp_avg = df.groupby("Exp Band", observed=True)["Salary"].mean()

# statistical test: does gender significantly affect salary (t-test)?
m = df.loc[df["Gender"] == "Male", "Salary"]
f = df.loc[df["Gender"] == "Female", "Salary"]
t_stat, p_val = stats.ttest_ind(m, f, equal_var=False)
print(f"\nGender t-test: t={t_stat:.2f}, p={p_val:.4f}")

# ANOVA: does education level significantly affect salary?
groups = [df.loc[df["Education Level"] == e, "Salary"] for e in edu_order]
f_stat, p_anova = stats.f_oneway(*groups)
print(f"Education ANOVA: F={f_stat:.2f}, p={p_anova:.6f}")

with open(f"{OUT}/statistical_tests.txt", "w") as fh:
    fh.write(f"Independent t-test (Male vs Female Salary): t = {t_stat:.3f}, p = {p_val:.4f}\n")
    fh.write("-> " + ("Statistically significant difference (p < 0.05)\n" if p_val < 0.05 else "No statistically significant difference (p >= 0.05)\n"))
    fh.write(f"\nOne-way ANOVA (Salary across Education Levels): F = {f_stat:.3f}, p = {p_anova:.6f}\n")
    fh.write("-> " + ("Statistically significant difference (p < 0.05)\n" if p_anova < 0.05 else "No statistically significant difference (p >= 0.05)\n"))

# ------------------------------------------------------------ 3. CHARTS
def save(name):
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", dpi=140); plt.close()

# 1. Distributions of numeric variables
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, f2 in zip(axes, ["Age", "Years of Experience", "Salary"], [None, None, fmt]):
    sns.histplot(df[col], bins=25, kde=True, ax=ax, color="#2563eb")
    ax.axvline(df[col].mean(), color="red", ls="--", label=f"Mean {df[col].mean():.0f}")
    ax.axvline(df[col].median(), color="green", ls="--", label=f"Median {df[col].median():.0f}")
    ax.set_title(f"{col} Distribution (skew={df[col].skew():.2f})"); ax.legend(fontsize=8)
    if f2: ax.xaxis.set_major_formatter(f2)
save("1_distributions")

# 2. Boxplots (outlier / spread check)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["Age", "Years of Experience", "Salary"]):
    sns.boxplot(y=df[col], ax=ax, color="#93c5fd")
    ax.set_title(f"{col} Spread")
save("2_boxplots")

# 3. Correlation heatmap + pairplot-style scatter
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0])
axes[0].set_title("Correlation Heatmap")
sns.scatterplot(data=df.sample(min(1500, len(df)), random_state=1), x="Years of Experience", y="Salary",
                hue="Education Level", hue_order=edu_order, alpha=0.6, ax=axes[1], palette="viridis")
axes[1].set_title("Experience vs Salary (by Education)"); axes[1].yaxis.set_major_formatter(fmt)
save("3_correlation")

# 4. Categorical breakdowns
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
sns.barplot(x=edu_avg.index, y=edu_avg.values, ax=axes[0], color="#10b981")
axes[0].set_title("Avg Salary by Education"); axes[0].yaxis.set_major_formatter(fmt)
axes[0].set_xticks(range(len(edu_avg))); axes[0].set_xticklabels([e.replace(" Degree", "") for e in edu_avg.index], rotation=20)
sns.barplot(x=exp_avg.index.astype(str), y=exp_avg.values, ax=axes[1], color="#f59e0b")
axes[1].set_title("Avg Salary by Experience Band"); axes[1].yaxis.set_major_formatter(fmt)
sns.boxplot(data=df[df["Gender"].isin(["Male", "Female"])], x="Gender", y="Salary", ax=axes[2])
axes[2].set_title(f"Salary by Gender (t-test p={p_val:.3f})"); axes[2].yaxis.set_major_formatter(fmt)
save("4_categorical_breakdown")

# 5. Top paying jobs
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(top_jobs.index, top_jobs["mean"], color="#7c3aed")
ax.set_title("Top 10 Highest-Paying Jobs (n>=5)"); ax.xaxis.set_major_formatter(fmt)
plt.gca().invert_yaxis()
save("5_top_jobs")

print("\nDone. Check the 'output_eda' folder.")
