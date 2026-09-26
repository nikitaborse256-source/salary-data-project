"""
Real-World Data Project - Retail Sales Analysis & Revenue Prediction
Domain: Retail | End-to-end: clean -> explore -> predict -> visualize -> conclude
Run:  python retail_project.py     (keep retail_sales_raw.csv in the same folder)
Needs: pip install pandas numpy matplotlib seaborn scikit-learn
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

OUT = "output_retail"
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid")
fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k")
SEED = 42

# ------------------------------------------------------------ 1. LOAD + CLEAN
df = pd.read_csv("retail_sales_raw.csv", parse_dates=["Date"])
raw_rows = len(df)
log = {"Raw rows": raw_rows}

log["Duplicate rows removed"] = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

log["Missing Units Sold (filled with median)"] = int(df["Units Sold"].isnull().sum())
df["Units Sold"] = df["Units Sold"].fillna(df["Units Sold"].median())
log["Missing Discount % (filled with 0)"] = int(df["Discount %"].isnull().sum())
df["Discount %"] = df["Discount %"].fillna(0)

# recompute Revenue consistently after filling (data integrity)
df["Revenue"] = (df["Units Sold"] * df["Unit Price"] * (1 - df["Discount %"] / 100)).round(2)

df["Month"] = df["Date"].dt.month
df["MonthName"] = df["Date"].dt.strftime("%b")
df["Quarter"] = df["Date"].dt.quarter
df["Weekday"] = df["Date"].dt.day_name()

log["Final rows"] = len(df)
df.to_csv(f"{OUT}/retail_sales_cleaned.csv", index=False)
pd.Series(log).to_csv(f"{OUT}/cleaning_log.csv", header=["count"])
print("Cleaning log:", log)

# ------------------------------------------------------------ 2. EDA
kpi = {
    "Total Revenue": df["Revenue"].sum(), "Total Orders": len(df),
    "Avg Order Value": df["Revenue"].mean(), "Total Units Sold": df["Units Sold"].sum(),
    "Avg Discount %": df["Discount %"].mean(),
}
pd.Series(kpi).to_csv(f"{OUT}/kpi_summary.csv", header=["value"])
print("\nKPIs:\n", pd.Series(kpi))

rev_by_store = df.groupby("Store")["Revenue"].sum().sort_values(ascending=False)
rev_by_cat = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
rev_by_month = df.groupby("MonthName")["Revenue"].sum().reindex(month_order)
rev_by_weekday = df.groupby("Weekday")["Revenue"].mean().reindex(
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
corr = df[["Units Sold", "Unit Price", "Discount %", "Revenue"]].corr()

rev_by_store.to_csv(f"{OUT}/revenue_by_store.csv")
rev_by_cat.to_csv(f"{OUT}/revenue_by_category.csv")

# ------------------------------------------------------------ 3. PREDICTIVE MODEL (Revenue)
X = df[["Store", "Category", "Units Sold", "Unit Price", "Discount %", "Month"]]
y = df["Revenue"]
cat_cols = ["Store", "Category"]
pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)], remainder="passthrough")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

models = {"Linear Regression": LinearRegression(),
          "Random Forest": RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)}
rows = []
for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("model", m)]).fit(X_train, y_train)
    p = pipe.predict(X_test)
    rows.append({"Model": name, "R2": r2_score(y_test, p), "MAE": mean_absolute_error(y_test, p)})
    if name == "Random Forest":
        best_pred, best_pipe = p, pipe
model_results = pd.DataFrame(rows).set_index("Model").round(3)
model_results.to_csv(f"{OUT}/model_results.csv")
print("\nModel results:\n", model_results)

# ------------------------------------------------------------ 4. CHARTS
def save(name):
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", dpi=140); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.barplot(x=rev_by_store.index, y=rev_by_store.values, ax=axes[0], color="#2563eb")
axes[0].set_title("Total Revenue by Store"); axes[0].yaxis.set_major_formatter(fmt)
sns.barplot(x=rev_by_cat.values, y=rev_by_cat.index, ax=axes[1], color="#10b981")
axes[1].set_title("Total Revenue by Category"); axes[1].xaxis.set_major_formatter(fmt)
save("1_revenue_by_store_category")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.lineplot(x=rev_by_month.index, y=rev_by_month.values, marker="o", ax=axes[0], color="#f59e0b")
axes[0].set_title("Monthly Revenue Trend (seasonal peak: Oct-Dec)"); axes[0].yaxis.set_major_formatter(fmt)
axes[0].tick_params(axis="x", rotation=30)
sns.barplot(x=rev_by_weekday.index, y=rev_by_weekday.values, ax=axes[1], color="#7c3aed")
axes[1].set_title("Avg Order Value by Weekday"); axes[1].yaxis.set_major_formatter(fmt)
axes[1].tick_params(axis="x", rotation=30)
save("2_time_trends")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0])
axes[0].set_title("Correlation Heatmap")
sns.scatterplot(data=df.sample(min(1200, len(df)), random_state=1), x="Discount %", y="Revenue",
                hue="Category", alpha=0.6, ax=axes[1])
axes[1].set_title("Discount % vs Revenue"); axes[1].yaxis.set_major_formatter(fmt)
save("3_correlation_discount")

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.scatter(y_test, best_pred, alpha=0.4, color="#2563eb")
lim = [0, max(y.max(), best_pred.max())]
ax.plot(lim, lim, "r--", label="Perfect prediction")
ax.set_title(f"Actual vs Predicted Revenue (Random Forest, R2={model_results.loc['Random Forest','R2']:.2f})")
ax.set_xlabel("Actual Revenue"); ax.set_ylabel("Predicted Revenue")
ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt); ax.legend()
save("4_actual_vs_predicted")

print("\nDone. Check the 'output_retail' folder.")
