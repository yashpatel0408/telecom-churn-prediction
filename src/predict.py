import joblib
import pandas as pd

from data_preprocessing import clean_data, encode_data, NUM_COLS

MODEL_PATH = '../models/churn_model.pkl'


def predict_churn(customer):
    saved = joblib.load(MODEL_PATH)
    model = saved['model']
    scaler = saved['scaler']
    columns = saved['columns']

    df = pd.DataFrame([customer])
    df = clean_data(df)
    df = encode_data(df, drop_first=False)
    df = df.reindex(columns=columns, fill_value=0)
    df[NUM_COLS] = scaler.transform(df[NUM_COLS])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return prediction, probability


if __name__ == '__main__':
    sample_customer = {
        'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
        'tenure': 2, 'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
        'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
        'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check', 'MonthlyCharges': 80.5, 'TotalCharges': 160.0,
    }

    result, prob = predict_churn(sample_customer)
    if result == 1:
        print('Customer will CHURN, probability:', round(prob, 2))
    else:
        print('Customer will STAY, probability:', round(prob, 2))