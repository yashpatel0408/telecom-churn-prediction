# Telecom Customer Churn Prediction

Machine learning project to predict whether a telecom customer will leave the company.

## Dataset
Telco Customer Churn (7043 customers, 21 columns). Data is imbalanced (about 73% stay, 27% churn).

## Approach
1. EDA: contract type, tenure and monthly charges vs churn
2. Cleaning: fixed blank values in TotalCharges, dropped customerID
3. Encoding: label encoding for binary columns, one-hot encoding for multi-category columns
4. Scaling numeric columns with StandardScaler
5. Models: Logistic Regression and Random Forest (class_weight='balanced')
6. Evaluation: precision, recall, F1, ROC-AUC

## Results
- Best model: <naam daalo>
- ROC-AUC: <value daalo>
- Churn recall: <value daalo>

## Key Insights
- Month-to-month contract customers churn the most
- New customers (low tenure) churn more
- Higher monthly charges increase churn

## How to run
pip install -r requirements.txt
cd src
python train_model.py
python predict.py

OUTPUT:-
Model: Logistic Regression
              precision    recall  f1-score   support

           0       0.90      0.70      0.79      1033
           1       0.49      0.79      0.61       374

    accuracy                           0.73      1407
   macro avg       0.70      0.75      0.70      1407
weighted avg       0.79      0.73      0.74      1407

ROC-AUC: 0.8353440733857567
----------------------------------------
Model: Random Forest
              precision    recall  f1-score   support

           0       0.83      0.90      0.86      1033
           1       0.63      0.49      0.55       374

    accuracy                           0.79      1407
   macro avg       0.73      0.69      0.71      1407
weighted avg       0.78      0.79      0.78      1407

ROC-AUC: 0.8185700752183299