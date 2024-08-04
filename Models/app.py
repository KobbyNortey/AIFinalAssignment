from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('period_predictor_model.joblib')

# Load the scaler
scaler = joblib.load('scaler.joblib')


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

        # Get the feature names from the model
        feature_names = model.feature_names_in_

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