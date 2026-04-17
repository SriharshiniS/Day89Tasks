import pandas as pd
import numpy as np
import joblib
import logging
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Logging setup
logging.basicConfig(
    filename='retrain_logs.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Paths
DATA_PATH = "data/new_data.csv"
MODEL_DIR = "model/"

def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        logging.info("Data loaded successfully")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess(df):
    try:
        # Features and target
        X = df[['Age', 'Salary', 'Department', 'Experience']]
        y = df['Target']

        # Encode categorical
        encoder = LabelEncoder()
        X['Department'] = encoder.fit_transform(X['Department'])

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        logging.info("Preprocessing completed")
        return X_scaled, y, scaler, encoder

    except Exception as e:
        logging.error(f"Preprocessing error: {e}")
        raise

def train_model(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluation
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        logging.info(f"Model trained with accuracy: {acc}")
        logging.info("\n" + classification_report(y_test, y_pred))

        print(f"✅ Model Accuracy: {acc}")

        return model

    except Exception as e:
        logging.error(f"Training error: {e}")
        raise

def save_artifacts(model, scaler, encoder):
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)

        joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
        joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
        joblib.dump(encoder, os.path.join(MODEL_DIR, "encoder.pkl"))

        logging.info("Artifacts saved successfully")
        print("✅ Model & preprocessors saved!")

    except Exception as e:
        logging.error(f"Saving error: {e}")
        raise

def main():
    df = load_data()
    X, y, scaler, encoder = preprocess(df)
    model = train_model(X, y)
    save_artifacts(model, scaler, encoder)

if __name__ == "__main__":
    main()
