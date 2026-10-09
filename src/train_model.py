import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from data_preprocessing import load_data, clean_data, encode_data, NUM_COLS

DATA_PATH = '../data/telecom_churn.csv'
MODEL_PATH = '../models/churn_model.pkl'


def main():
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = encode_data(df, drop_first=True)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
    X_test[NUM_COLS] = scaler.transform(X_test[NUM_COLS])

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    }

    best_name = None
    best_model = None
    best_auc = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print('Model:', name)
        print(classification_report(y_test, pred))
        print('ROC-AUC:', auc)
        print('-' * 40)
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = model

    print('Best model:', best_name)

    joblib.dump(
        {'model': best_model, 'scaler': scaler, 'columns': list(X.columns)},
        MODEL_PATH
    )
    print('Model saved at', MODEL_PATH)


if __name__ == '__main__':
    main()