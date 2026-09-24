"""
Predictive Modeling Using Machine Learning - Salary Dataset
Part A: Regression  -> predict Salary (Linear Regression, Decision Tree, Random Forest)
Part B: Classification -> predict High vs Low salary (confusion matrix + ROC curve)
Run:  python ml_salary_prediction.py     (keep Salary_Data_Cleaned.csv in the same folder)
Needs: pip install pandas numpy matplotlib seaborn scikit-learn
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, roc_auc_score, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.inspection import permutation_importance

OUT = "output_ml"
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid")
SEED = 42

# ------------------------------------------------------------ LOAD (cleaned data from Task 1)
df = pd.read_csv("Salary_Data_Cleaned.csv")
print("Shape:", df.shape)
X = df.drop(columns="Salary")
y = df["Salary"]
num_cols = ["Age", "Years of Experience"]
cat_cols = ["Gender", "Education Level", "Job Title"]

def make_pipe(model):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ], remainder="passthrough")          # numeric columns pass straight through
    return Pipeline([("pre", pre), ("model", model)])

# ------------------------------------------------------------ TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
print("Train:", X_train.shape, " Test:", X_test.shape)

# ------------------------------------------------------------ PART A: REGRESSION
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=SEED),
    "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
}
rows, fitted, preds = [], {}, {}
for name, m in models.items():
    pipe = make_pipe(m).fit(X_train, y_train)
    p = pipe.predict(X_test)
    cv = cross_val_score(make_pipe(m), X, y, cv=5, scoring="r2")
    rows.append({"Model": name,
                 "R2 (test)": r2_score(y_test, p),
                 "MAE": mean_absolute_error(y_test, p),
                 "RMSE": np.sqrt(mean_squared_error(y_test, p)),
                 "CV R2 (5-fold mean)": cv.mean()})
    fitted[name], preds[name] = pipe, p
results = pd.DataFrame(rows).set_index("Model").round(3)
results.to_csv(f"{OUT}/model_comparison.csv")
print("\nRegression results:\n", results)

best_name = results["R2 (test)"].idxmax()
best_pred = preds[best_name]
print("\nBest model:", best_name)

imp = permutation_importance(fitted[best_name], X_test, y_test, n_repeats=10, random_state=SEED, n_jobs=-1)
imp_s = pd.Series(imp.importances_mean, index=X.columns).sort_values()
imp_s.to_csv(f"{OUT}/feature_importance.csv", header=["importance"])
print("\nFeature importance:\n", imp_s)

# ------------------------------------------------------------ PART B: CLASSIFICATION (High vs Low salary)
threshold = y.median()
yc = (y >= threshold).astype(int)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, yc, test_size=0.2, random_state=SEED, stratify=yc)
clfs = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
}
clf_rows, clf_fit, clf_proba = [], {}, {}
for name, m in clfs.items():
    pipe = make_pipe(m).fit(Xc_train, yc_train)
    pr = pipe.predict(Xc_test); pp = pipe.predict_proba(Xc_test)[:, 1]
    clf_rows.append({"Model": name, "Accuracy": accuracy_score(yc_test, pr),
                     "Precision": precision_score(yc_test, pr), "Recall": recall_score(yc_test, pr),
                     "F1": f1_score(yc_test, pr), "AUC": roc_auc_score(yc_test, pp)})
    clf_fit[name], clf_proba[name] = pipe, pp
clf_res = pd.DataFrame(clf_rows).set_index("Model").round(3)
clf_res.to_csv(f"{OUT}/classification_results.csv")
print(f"\nClassification (High salary = Salary >= {threshold:,.0f}):\n", clf_res)
rf_pred = clf_fit["Random Forest"].predict(Xc_test)

# ------------------------------------------------------------ CHARTS
fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k")

def ch_compare(ax):
    r = results["R2 (test)"]
    sns.barplot(x=r.index, y=r.values, ax=ax, hue=r.index, palette="viridis", legend=False)
    for i, v in enumerate(r.values):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1.05); ax.set_title("Model Comparison (R2 on test data)"); ax.set_xlabel(""); ax.set_ylabel("R2 score")

def ch_actual(ax):
    ax.scatter(y_test, best_pred, alpha=0.5, color="#2563eb")
    lim = [y.min(), y.max()]
    ax.plot(lim, lim, "r--", label="Perfect prediction")
    ax.set_title(f"Actual vs Predicted ({best_name})"); ax.set_xlabel("Actual Salary"); ax.set_ylabel("Predicted Salary")
    ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt); ax.legend()

def ch_resid(ax):
    sns.histplot(y_test - best_pred, bins=30, kde=True, ax=ax, color="#7c3aed")
    ax.axvline(0, color="red", ls="--"); ax.set_title("Prediction Errors (Residuals)")
    ax.set_xlabel("Actual - Predicted"); ax.xaxis.set_major_formatter(fmt)

def ch_imp(ax):
    ax.barh(imp_s.index, imp_s.values, color="#10b981")
    ax.set_title("Feature Importance (permutation)"); ax.set_xlabel("Drop in R2 when feature is shuffled")

def ch_cm(ax):
    cm = confusion_matrix(yc_test, rf_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Low", "High"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix (Random Forest)"); ax.grid(False)

def ch_roc(ax):
    for name, pp in clf_proba.items():
        fpr, tpr, _ = roc_curve(yc_test, pp)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc_score(yc_test, pp):.2f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random guess")
    ax.set_title("ROC Curve (High vs Low salary)"); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
for ax, fn in zip(axes.ravel(), [ch_compare, ch_actual, ch_resid, ch_imp, ch_cm, ch_roc]):
    fn(ax)
fig.suptitle("Predictive Modeling Results - Salary Dataset", fontsize=20, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{OUT}/ML_Results.png", dpi=130)
plt.close()
print("\nDone. Check the 'output_ml' folder.")
