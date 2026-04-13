from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import logging
import os

app = Flask(__name__)

# Logging setup (Step 18)
logging.basicConfig(
    filename='logs.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Load model safely
def load_artifacts():
    model = joblib.load("model/model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    encoder = joblib.load("model/encoder.pkl")
    return model, scaler, encoder

model, scaler, encoder = load_artifacts()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        age = int(data['Age'])
        salary = int(data['Salary'])
        dept = encoder.transform([data['Department']])[0]
        exp = int(data['Experience'])

        input_data = np.array([[age, salary, dept, exp]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        # Logging
        logging.info(f"UI Input: {input_data.tolist()}, Prediction: {prediction}")

        return render_template("index.html",
                               prediction_text=f"Prediction: {prediction}")

    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/api', methods=['POST'])
def api():
    try:
        data = request.get_json()

        input_data = np.array([[
            data['Age'],
            data['Salary'],
            encoder.transform([data['Department']])[0],
            data['Experience']
        ]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        # Logging
        logging.info(f"API Input: {data}, Prediction: {prediction}")

        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)})

# Reload model without restarting server (Step 19)
@app.route('/reload', methods=['GET'])
def reload_model():
    global model, scaler, encoder
    model, scaler, encoder = load_artifacts()
    return "Model reloaded successfully!"

if __name__ == "__main__":
    app.run(debug=True)