"""
Data Cleaning & Visualization Project - Salary Dataset
Steps: Load -> Inspect -> Clean (duplicates, missing, typos, outliers) -> Visualize -> Dashboard
Run:  python salary_analysis.py      (keep Salary_Data.xlsx in the same folder)
Needs: pip install pandas openpyxl matplotlib seaborn
"""
import os
import textwrap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

INPUT = "Salary_Data.xlsx"
OUT = "output"
os.makedirs(f"{OUT}/charts", exist_ok=True)
sns.set_theme(style="whitegrid", palette="deep")

# ------------------------------------------------------------------ 1. LOAD
df = pd.read_excel(INPUT)
# first column name has a hidden BOM character (ï»¿Age) -> rename cleanly
df.columns = ["Age", "Gender", "Education Level", "Job Title", "Years of Experience", "Salary"]
raw_rows = len(df)
print("Raw shape:", df.shape)
print(df.info())
print(df.describe())

log = {"Raw rows": raw_rows}

# ------------------------------------------------------------------ 2. DUPLICATES
log["Fully empty rows"] = int(df.isnull().all(axis=1).sum())
df = df.dropna(how="all")
log["Duplicate rows removed"] = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

# ------------------------------------------------------------------ 3. MISSING VALUES
print("\nMissing values:\n", df.isnull().sum())
log["Rows dropped (missing Salary)"] = int(df["Salary"].isnull().sum())
df = df.dropna(subset=["Salary"])                       # target column -> never guess it
df["Years of Experience"] = df["Years of Experience"].fillna(df["Years of Experience"].median())
df["Age"] = df["Age"].fillna(df["Age"].median())
for col in ["Gender", "Education Level", "Job Title"]:  # categorical -> mode
    df[col] = df[col].fillna(df[col].mode()[0])

# ------------------------------------------------------------------ 4. INCONSISTENT LABELS
df["Education Level"] = df["Education Level"].replace({
    "Bachelor's": "Bachelor's Degree",
    "Master's": "Master's Degree",
    "phD": "PhD",
})
df["Job Title"] = df["Job Title"].replace({
    "Social M": "Social Media Manager",
    "Social Media Man": "Social Media Manager",
})

# ------------------------------------------------------------------ 5. OUTLIERS
def iqr_bounds(s):
    q1, q3 = s.quantile([0.25, 0.75])
    i = q3 - q1
    return q1 - 1.5 * i, q3 + 1.5 * i

before = df.copy()  # for before/after boxplot

for col in ["Age", "Years of Experience", "Salary"]:
    lo, hi = iqr_bounds(df[col])
    n = int(((df[col] < lo) | (df[col] > hi)).sum())
    log[f"IQR-flagged outliers in {col} (kept if realistic)"] = n

# Rule 1: impossible salaries (IQR lower bound is negative, so it can't catch them).
bad_salary = df["Salary"] < 1000
log["Rows removed (Salary < 1000, data-entry errors)"] = int(bad_salary.sum())
df = df[~bad_salary]

# Rule 2: impossible experience (started working before age 16).
bad_exp = df["Years of Experience"] > (df["Age"] - 16)
log["Rows removed (experience > Age - 16)"] = int(bad_exp.sum())
df = df[~bad_exp]

# Rule 3: IQR outliers in Age / Experience are real senior employees -> kept, not deleted.
df = df.reset_index(drop=True)
log["Final rows"] = len(df)
print("\nCleaning log:")
for k, v in log.items():
    print(f"  {k}: {v}")

df.to_csv(f"{OUT}/Salary_Data_Cleaned.csv", index=False)
pd.Series(log).to_csv(f"{OUT}/cleaning_log.csv", header=["count"])

# ------------------------------------------------------------------ 6. ANALYSIS
edu_order = ["High School", "Bachelor's Degree", "Master's Degree", "PhD"]
edu_avg = df.groupby("Education Level")["Salary"].mean().reindex(edu_order)
gender = df[df["Gender"].isin(["Male", "Female"])].groupby("Gender")["Salary"].mean()
job_stats = df.groupby("Job Title")["Salary"].agg(["mean", "count"])
top_jobs = job_stats[job_stats["count"] >= 5].sort_values("mean", ascending=False).head(10)
df["Exp Band"] = pd.cut(df["Years of Experience"], [-1, 2, 5, 10, 15, 20, 40],
                        labels=["0-2", "3-5", "6-10", "11-15", "16-20", "20+"])
exp_avg = df.groupby("Exp Band", observed=True)["Salary"].mean()
corr = df[["Age", "Years of Experience", "Salary"]].corr()

stats = {
    "avg_salary": df["Salary"].mean(), "median_salary": df["Salary"].median(),
    "max_salary": df["Salary"].max(), "employees": len(df),
    "corr_exp_salary": corr.loc["Years of Experience", "Salary"],
    "corr_age_salary": corr.loc["Age", "Salary"],
}
pd.Series(stats).to_csv(f"{OUT}/key_stats.csv", header=["value"])
edu_avg.to_csv(f"{OUT}/avg_salary_by_education.csv")
gender.to_csv(f"{OUT}/avg_salary_by_gender.csv")
top_jobs.to_csv(f"{OUT}/top10_jobs.csv")
exp_avg.to_csv(f"{OUT}/avg_salary_by_experience.csv")
print("\n", pd.Series(stats), "\n", edu_avg, "\n", gender, "\n", top_jobs, "\n", exp_avg)

# ------------------------------------------------------------------ 7. CHARTS
fmt = matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k")

def save(name):
    plt.tight_layout()
    plt.savefig(f"{OUT}/charts/{name}.png", dpi=150)
    plt.close()

def c_hist(ax):
    sns.histplot(df["Salary"], bins=30, kde=True, color="#2563eb", ax=ax)
    ax.axvline(df["Salary"].median(), color="red", ls="--", label=f"Median {df['Salary'].median()/1000:.0f}k")
    ax.set_title("Salary Distribution"); ax.xaxis.set_major_formatter(fmt); ax.legend()

def c_edu(ax):
    sns.barplot(x=edu_avg.index, y=edu_avg.values, ax=ax, color="#10b981")
    ax.set_title("Average Salary by Education"); ax.yaxis.set_major_formatter(fmt)
    ax.set_xlabel(""); ax.tick_params(axis="x", labelsize=8)
    ax.set_xticks(range(len(edu_avg)))
    ax.set_xticklabels([s.replace(" Degree", "") for s in edu_avg.index])

def c_scatter(ax):
    sns.scatterplot(data=df.sample(min(1500, len(df)), random_state=1), x="Years of Experience",
                    y="Salary", alpha=0.5, ax=ax, color="#7c3aed")
    sns.regplot(data=df, x="Years of Experience", y="Salary", scatter=False, ax=ax, color="red")
    ax.set_title(f"Experience vs Salary (r = {stats['corr_exp_salary']:.2f})"); ax.yaxis.set_major_formatter(fmt)

def c_jobs(ax):
    sns.barplot(x=top_jobs["mean"].values, y=top_jobs.index, ax=ax, color="#f59e0b")
    ax.set_yticks(range(len(top_jobs)))
    ax.set_yticklabels([textwrap.fill(j, 20) for j in top_jobs.index])
    ax.set_title("Top 10 Highest-Paying Jobs (n >= 5)"); ax.xaxis.set_major_formatter(fmt); ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8)

def c_heat(ax):
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation Heatmap")

def c_box_gender(ax):
    sns.boxplot(data=df[df["Gender"].isin(["Male", "Female"])], x="Gender", y="Salary", ax=ax)
    ax.set_title("Salary by Gender"); ax.yaxis.set_major_formatter(fmt)

def c_exp_band(ax):
    sns.barplot(x=exp_avg.index.astype(str), y=exp_avg.values, ax=ax, color="#ef4444")
    ax.set_title("Average Salary by Experience (years)"); ax.yaxis.set_major_formatter(fmt); ax.set_xlabel("")

def c_box_before_after(ax):
    d = pd.concat([before.assign(Stage="Before cleaning"), df.assign(Stage="After cleaning")])
    sns.boxplot(data=d, x="Stage", y="Salary", ax=ax)
    ax.set_title("Salary Outliers: Before vs After"); ax.yaxis.set_major_formatter(fmt); ax.set_xlabel("")

charts = {"1_salary_distribution": c_hist, "2_salary_by_education": c_edu, "3_experience_vs_salary": c_scatter,
          "4_top10_jobs": c_jobs, "5_correlation_heatmap": c_heat, "6_salary_by_gender": c_box_gender,
          "7_salary_by_experience_band": c_exp_band, "8_outliers_before_after": c_box_before_after}
for name, fn in charts.items():
    fig, ax = plt.subplots(figsize=(7, 4.5)); fn(ax); save(name)

# ------------------------------------------------------------------ 8. DASHBOARD
fig = plt.figure(figsize=(16, 11), facecolor="white")
fig.suptitle("Salary Insights Dashboard", fontsize=22, fontweight="bold", y=0.985)
kpis = [("Employees", f"{stats['employees']:,}"), ("Average Salary", f"${stats['avg_salary']:,.0f}"),
        ("Median Salary", f"${stats['median_salary']:,.0f}"), ("Highest Salary", f"${stats['max_salary']:,.0f}"),
        ("Exp-Salary Corr.", f"{stats['corr_exp_salary']:.2f}")]
for i, (label, val) in enumerate(kpis):
    ax = fig.add_axes([0.03 + i * 0.194, 0.865, 0.18, 0.08]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="#eff6ff", transform=ax.transAxes))
    ax.text(0.5, 0.62, val, ha="center", va="center", fontsize=17, fontweight="bold", color="#1d4ed8")
    ax.text(0.5, 0.18, label, ha="center", va="center", fontsize=10, color="#475569")
gs = fig.add_gridspec(2, 3, left=0.075, right=0.97, top=0.83, bottom=0.06, hspace=0.35, wspace=0.42)
for pos, fn in zip([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
                   [c_hist, c_edu, c_scatter, c_jobs, c_exp_band, c_heat]):
    fn(fig.add_subplot(gs[pos]))
plt.savefig(f"{OUT}/Dashboard.png", dpi=130)
plt.close()
print("\nDone. Check the 'output' folder.")
