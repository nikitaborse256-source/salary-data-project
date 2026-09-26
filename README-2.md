# Real-World Data Project - Retail Sales Analysis & Revenue Prediction

End-to-end retail analytics project: data cleaning, exploratory analysis, revenue prediction model, and business conclusions.

## Domain
Retail - store transactions across 5 cities (Mumbai, Pune, Delhi, Bangalore, Chennai) and 6 product categories, Jan 2024 - Dec 2025.

## Files
- `retail_project.py`: complete code (cleaning + EDA + ML model + charts)
- `retail_sales_raw.csv`: raw input dataset
- `retail_sales_cleaned.csv`: cleaned dataset
- `Retail_Report.pdf`: full report with findings and visualizations
- `kpi_summary.csv`, `revenue_by_store.csv`, `revenue_by_category.csv`, `model_results.csv`, `cleaning_log.csv`
- Charts: `1_revenue_by_store_category.png`, `2_time_trends.png`, `3_correlation_discount.png`, `4_actual_vs_predicted.png`

## Key Findings
- Mumbai is the top-performing store (~Rs 1.46 crore revenue)
- Electronics dominates category revenue (~Rs 3.5 crore), ~7x the next category
- Clear festive-season seasonality: revenue peaks Oct-Dec
- Random Forest predicts order revenue with R2 = 0.995 (MAE ~Rs 643)

## How to run
pip install pandas numpy matplotlib seaborn scikit-learn
python retail_project.py

Output is saved in the `output_retail` folder.
