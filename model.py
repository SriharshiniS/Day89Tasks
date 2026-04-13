import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

def train_model():
    df = pd.read_csv("data/data.csv")

    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Encode categorical
    le = LabelEncoder()
    df['Department'] = le.fit_transform(df['Department'])

    X = df.drop('Target', axis=1)
    y = df['Target']

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)

    # Create folders if not exist
    os.makedirs("model", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Save model artifacts
    joblib.dump(model, "model/model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(le, "model/encoder.pkl")

    # Save predictions for Power BI
    df['Prediction'] = model.predict(X_scaled)
    df.to_csv("data/output.csv", index=False)

    print("Model and output.csv saved successfully!")

if __name__ == "__main__":
    train_model()