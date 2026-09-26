# Exploratory Data Analysis (EDA) - Salary Dataset

Statistical exploration of the cleaned salary dataset to find patterns, correlations and key influencing factors.

## Files
- `eda_salary.py`: complete code
- `EDA_Report.pdf`: full report with charts and findings
- `statistical_summary.csv`, `correlation_matrix.csv`, `statistical_tests.txt`
- Charts: `1_distributions.png`, `2_boxplots.png`, `3_correlation.png`, `4_categorical_breakdown.png`, `5_top_jobs.png`

## Key Findings
- Years of Experience has the strongest correlation with Salary (r = 0.82)
- Salary differs significantly across Education levels (ANOVA p < 0.0001): High School ~$45K to PhD ~$161K
- Gender shows a statistically significant (p < 0.0001) but comparatively small salary gap
- Director-level job titles are the highest paid (~$170K-200K)

## How to run
pip install pandas numpy matplotlib seaborn scipy
python eda_salary.py

Keep `Salary_Data_Cleaned.csv` in the same folder. Output is saved in the `output_eda` folder.
