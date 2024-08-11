import os
import sys
from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import pandas as pd

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

TEMPLATE_DIR = os.path.join(parent_dir, 'Templates')
STATIC_DIR = os.path.join(parent_dir, 'Static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# Load the saved model and scalers
lstm_model = load_model(os.path.join(current_dir, 'lstm_model.keras'))
scaler = joblib.load(os.path.join(current_dir, 'scaler.pkl'))
target_scaler = joblib.load(os.path.join(current_dir, 'target_scaler.pkl'))
label_encoders = joblib.load(os.path.join(current_dir, 'label_encoders.pkl'))


def preprocess_input(input_data):
    numerical_columns = ['Age', 'BMI', 'MeanCycleLength', 'LengthofMenses', 'UnusualBleeding', 'MeanBleedingIntensity']
    input_values = [float(input_data.get(col, 0)) for col in numerical_columns]

    input_df = pd.DataFrame([input_values], columns=numerical_columns)

    input_scaled = scaler.transform(input_df)
    return input_scaled


def create_sequences(features, time_steps=5):
    X = np.array([features for _ in range(time_steps)])
    return X.reshape(1, time_steps, features.shape[1])


def predict_cycle_length(input_data):
    input_scaled = preprocess_input(input_data)
    print("Input Scaled: ", input_scaled)

    input_seq = create_sequences(input_scaled)
    prediction_scaled = lstm_model.predict(input_seq)
    print("Raw Model Output (Scaled): ", prediction_scaled)

    prediction_scaled = prediction_scaled.reshape(-1, 1)
    prediction = target_scaler.inverse_transform(prediction_scaled)
    print("Final Prediction (After Inverse Transform): ", prediction)

    return round(prediction[0, 0])




@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = {
            'Age': float(request.form['age']),
            'BMI': float(request.form['bmi']),
            'MeanCycleLength': float(request.form['mean_cycle_length']),
            'LengthofMenses': float(request.form['menses_length']),
            'UnusualBleeding': 1.0 if request.form['unusual_bleeding'] == 'yes' else 0.0,
            'MeanBleedingIntensity': float(request.form['bleeding_intensity']),
            'DaysSinceLast': float(request.form['days_since_last_period'])
        }

        print("Form data:", input_data)

        # Make prediction
        prediction = predict_cycle_length(input_data)

        prediction = round(prediction)

        return render_template('result.html', prediction=prediction)

    return render_template('form.html')


if __name__ == '__main__':
    app.run(debug=True)
