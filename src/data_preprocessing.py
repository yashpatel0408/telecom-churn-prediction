import pandas as pd
import numpy as np

BINARY_YES_NO_COLS = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
MULTI_COLS = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
              'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
              'Contract', 'PaymentMethod']
NUM_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    df = df.copy()
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan).astype(float)
    df = df.dropna()
    return df


def encode_data(df, drop_first=True):
    df = df.copy()
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    for col in BINARY_YES_NO_COLS:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    df = pd.get_dummies(df, columns=MULTI_COLS, drop_first=drop_first, dtype=int)
    return df