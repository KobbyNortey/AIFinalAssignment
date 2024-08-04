#%%
import pandas as pd

# Load the CSV file
file_path = '../Datasets/FedCycleData071012 (2).csv'
data = pd.read_csv(file_path)

# Display the first few rows of the DataFrame to understand its structure
print(data.head(10))

# Check for missing values
print(data.isnull().sum())

# Display the columns and their data types
print(data.dtypes)

# Basic statistical summary
print(data.describe())
#%%
threshold = 0.5

# Remove columns with majority N/A values
data_cleaned = data.loc[:, data.isnull().mean() < threshold]

# Display the columns that remain after cleaning
print("Remaining columns after removing those with majority N/A values:")
print(data_cleaned.columns)
#%%
data_cleaned
#%%
df=data_cleaned
#%%
df
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder




for column in df.columns:
    if df[column].dtype == 'object':
        df[column].fillna(data_cleaned[column].mode()[0], inplace=True)
    else:
        df[column].fillna(df[column].median(), inplace=True)

# Encode categorical variables
label_encoders = {}
for column in data_cleaned.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

#%%
# Normalize numerical features
scaler = StandardScaler()
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

#%%
# Define the target variable
target = 'LengthofCycle'  # Target variable for prediction
features = df.drop(columns=[target])
target_data = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target_data, test_size=0.2, random_state=42)




#%%

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_rmse = mean_squared_error(y_test, lr_predictions, squared=False)

# Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = mean_squared_error(y_test, rf_predictions, squared=False)

# Print evaluation metrics
print(f'Linear Regression MAE: {lr_mae}, RMSE: {lr_rmse}')
print(f'Random Forest MAE: {rf_mae}, RMSE: {rf_rmse}')
#%%

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Support Vector Regressor (SVR)
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train)
svr_predictions = svr_model.predict(X_test)
svr_mae = mean_absolute_error(y_test, svr_predictions)
svr_rmse = mean_squared_error(y_test, svr_predictions, squared=False)
print(f'SVR MAE: {svr_mae}, RMSE: {svr_rmse}')

# Gradient Boosting Regressor
gbr_model = GradientBoostingRegressor(random_state=42)
gbr_model.fit(X_train, y_train)
gbr_predictions = gbr_model.predict(X_test)
gbr_mae = mean_absolute_error(y_test, gbr_predictions)
gbr_rmse = mean_squared_error(y_test, gbr_predictions, squared=False)
print(f'Gradient Boosting Regressor MAE: {gbr_mae}, RMSE: {gbr_rmse}')


#%%
# XGBoost Regressor
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_predictions)
xgb_rmse = mean_squared_error(y_test, xgb_predictions, squared=False)
print(f'XGBoost Regressor MAE: {xgb_mae}, RMSE: {xgb_rmse}')

# K-Nearest Neighbors Regressor
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
knn_mae = mean_absolute_error(y_test, knn_predictions)
knn_rmse = mean_squared_error(y_test, knn_predictions, squared=False)
print(f'K-Nearest Neighbors Regressor MAE: {knn_mae}, RMSE: {knn_rmse}')
#%%
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of LengthofCycle
plt.figure(figsize=(10, 6))
sns.histplot(data['LengthofCycle'], kde=True)
plt.title('Histogram of Cycle Lengths')
plt.xlabel('Cycle Length')
plt.ylabel('Frequency')
plt.show()
#%%
# Correlation Matrix
plt.figure(figsize=(12, 10))
correlation_matrix = data_cleaned.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
#%%
plt.figure(figsize=(10, 6))
plt.scatter(y_test, rf_predictions, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
plt.title('Prediction vs Actual Values (Random Forest)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

#%%
import joblib

# Get feature names excluding the target variable
feature_names = [col for col in df.columns if col != 'LengthofCycle']

# Prepare the features and target
X = df[feature_names]
y = df['LengthofCycle']

# Create and fit the scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a new Random Forest model with only the input features
rf_model_for_prediction = RandomForestRegressor(random_state=42)
rf_model_for_prediction.fit(X_scaled, y)

# Save the scaler
joblib.dump(scaler, '../Models/scaler.joblib')

# Save the Random Forest model
joblib.dump(rf_model_for_prediction, '../Models/period_predictor_model.joblib')

# Save the feature names
joblib.dump(feature_names, '../Models/feature_names.joblib')

print("Models and feature names saved successfully.")
#%%
