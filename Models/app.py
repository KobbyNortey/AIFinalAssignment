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
        # Get user input from the form
        user_data = {
            'Age': float(request.form['age']),
            'BMI': float(request.form['bmi']),
            'LastCycleLength': float(request.form['last_cycle_length']),
            'AvgCycleLength': float(request.form['avg_cycle_length'])
        }

        # Create a dataframe with user input
        user_df = pd.DataFrame([user_data])

        # Create a DataFrame with all features, filling missing ones with 0
        full_user_df = pd.DataFrame(columns=feature_names)
        for feature in feature_names:
            if feature in user_df.columns:
                full_user_df[feature] = user_df[feature]
            else:
                full_user_df[feature] = 0

        # Normalize the input data
        user_data_normalized = scaler.transform(full_user_df)

        # Make prediction
        prediction = model.predict(user_data_normalized)

        return render_template('result.html', prediction=prediction[0])

    return render_template('form.html')


if __name__ == '__main__':
    app.run(debug=True)