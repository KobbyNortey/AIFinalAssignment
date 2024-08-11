import os
import sys
from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

TEMPLATE_DIR = os.path.join(parent_dir, 'Templates')
STATIC_DIR = os.path.join(parent_dir, 'Static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# Load the saved model and scalers
lstm_model = load_model('../Models/lstm_model.keras')
scaler = joblib.load('../Models/scaler.pkl')
target_scaler = joblib.load('../Models/target_scaler.pkl')
label_encoders = joblib.load('../Models/label_encoders.pkl')


def preprocess_input(input_data):
    print("Input data:", input_data)
    numerical_columns = ['Age', 'BMI', 'MeanCycleLength', 'LengthofMenses', 'UnusualBleeding', 'MeanBleedingIntensity']
    print("Numerical columns:", numerical_columns)

    # Create a list of values in the order expected by the scaler
    input_values = []
    for column in numerical_columns:
        if column in input_data:
            input_values.append(float(input_data[column]))
        else:
            print(f"Warning: {column} not found in input data. Setting to 0.")
            input_values.append(0.0)

    print("Input values:", input_values)

    # Scale numerical features
    input_scaled = scaler.transform([input_values])

    return input_scaled


def create_sequences(features, time_steps=5):
    if features.ndim == 1:
        features = features.reshape(1, -1)
    X = np.repeat(features, time_steps, axis=0)
    return X


def predict_cycle_length(input_data):
    # Preprocess the input data
    input_scaled = preprocess_input(input_data)

    # Create sequences
    input_seq = create_sequences(input_scaled[0])  # Use [0] to get the first (and only) row

    print("Input sequence shape:", input_seq.shape)

    # Make prediction
    prediction = lstm_model.predict(np.expand_dims(input_seq, axis=0))

    print("Raw prediction shape:", prediction.shape)
    print("Raw prediction:", prediction)

    # Reshape prediction if necessary
    if prediction.ndim > 2:
        prediction = prediction.reshape(prediction.shape[0], -1)

    print("Reshaped prediction shape:", prediction.shape)

    # Inverse transform the prediction
    prediction_rescaled = target_scaler.inverse_transform(prediction)

    print("Rescaled prediction shape:", prediction_rescaled.shape)
    print("Rescaled prediction:", prediction_rescaled)

    return prediction_rescaled[0][0]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get data from form
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

        # Round prediction to nearest integer
        prediction = round(prediction)

        return render_template('result.html', prediction=prediction)

    return render_template('form.html')


if __name__ == '__main__':
    app.run(debug=True)
