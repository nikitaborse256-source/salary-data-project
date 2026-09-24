# Predictive Modeling Using Machine Learning (Salary Dataset)

Machine learning models that predict salary from age, gender, education, job title and years of experience. Uses the cleaned dataset from the Data Cleaning & Visualization project.

## Files
- `ml_salary_prediction.py`: complete code (training, evaluation, charts)
- `Salary_Data_Cleaned.csv`: cleaned input data
- `ML_Results.png`: model comparison, actual vs predicted, residuals, feature importance, confusion matrix, ROC curve
- `ML_Report.pdf`: project report

## What was done
- 80/20 train-test split, plus 5-fold cross-validation
- **Regression:** Linear Regression, Decision Tree, Random Forest
- **Classification:** High vs Low salary (median split) with Logistic Regression and Random Forest

## Results
| Model | R2 (test) | MAE |
|---|---|---|
| Linear Regression | 0.824 | $15,738 |
| Decision Tree | 0.820 | $15,735 |
| **Random Forest** | **0.893** | **$11,264** |

Classification: about 90% accuracy, AUC about 0.97 to 0.98.

## Key insights
- Years of experience is the most important predictor, then job title, age and education
- Gender has almost no effect on the prediction once experience and role are known

## How to run
pip install pandas numpy matplotlib seaborn scikit-learn
python ml_salary_prediction.py

Output is saved in the `output_ml` folder.
