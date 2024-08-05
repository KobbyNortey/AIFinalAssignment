import os
import sys
from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

# Define paths
MODEL_PATH = os.path.join(current_dir, 'period_predictor_model.joblib')
SCALER_PATH = os.path.join(current_dir, 'scaler.joblib')
FEATURE_NAMES_PATH = os.path.join(current_dir, 'feature_names.joblib')
TEMPLATE_DIR = os.path.join(parent_dir, 'Templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Load the trained model
model = joblib.load(MODEL_PATH)

# Load the scaler
scaler = joblib.load(SCALER_PATH)

# Load the feature names
feature_names = joblib.load(FEATURE_NAMES_PATH)


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Load all possible features
        all_features = joblib.load(FEATURE_NAMES_PATH)

        # Create a dictionary to hold all feature values, initialized with zeros or appropriate default values
        user_data = {feature: 0 for feature in all_features}

        # Update the dictionary with the values we actually have from the form
        user_data.update({
            'Age': float(request.form['age']),
            'BMI': float(request.form['bmi']),
            'MeanCycleLength': float(request.form['mean_cycle_length']),
            'LengthofMenses': float(request.form.get('menses_length', 5)),
            'UnusualBleeding': int(request.form.get('unusual_bleeding', 0)),
            'MeanBleedingIntensity': float(request.form.get('bleeding_intensity', 0)),
        })

        # Create a dataframe with user input
        user_df = pd.DataFrame([user_data])

        # Normalize the input data
        user_data_normalized = scaler.transform(user_df)

        # Make prediction
        prediction = model.predict(user_data_normalized)

        # The prediction is the length of the cycle, so we need to subtract the days since last period
        days_since_last_period = float(request.form['days_since_last_period'])
        days_until_next_period = max(0, prediction[0] - days_since_last_period)

        return render_template('result.html', prediction=days_until_next_period)

    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)