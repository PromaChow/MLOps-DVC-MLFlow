import pandas as pd
import mlflow
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
import sys
train_data = pd.read_csv("../data/sensor_data.csv")
test_data = pd.read_csv("../data/test_sensor_data.csv")

# Load testing data from a file (replace 'test_file.csv' with your actual file name)

# Display the first few rows of the training data
print("Training Data:")
print(train_data.head())

# Display the first few rows of the testing data
print("Testing Data:")
print(test_data.head())

# Assuming you have loaded the data into Pandas DataFrames
label_encoder_machine = LabelEncoder()
label_encoder_sensor = LabelEncoder()

# Apply label encoding to training data
train_data['Machine_ID'] = label_encoder_machine.fit_transform(train_data['Machine_ID'])
train_data['Sensor_ID'] = label_encoder_sensor.fit_transform(train_data['Sensor_ID'])

# Apply label encoding to testing data
test_data['Machine_ID'] = label_encoder_machine.transform(test_data['Machine_ID'])
test_data['Sensor_ID'] = label_encoder_sensor.transform(test_data['Sensor_ID'])

# Handle missing values in training data (if any)
train_data.fillna(0, inplace=True)  # Replace NaN values with 0, you may choose a different strategy

# Separate features and target variable for training data
X_train = train_data[['Machine_ID', 'Reading']]
y_train = train_data['Sensor_ID']

# Separate features and target variable for testing data
X_test = test_data[['Machine_ID', 'Reading']]
y_test = test_data['Sensor_ID']

# Set up MLflow experiment
experiment_name = 'SensorPrediction'
mlflow.set_experiment(experiment_name)

# Initialize variables to keep track of the best model
best_accuracy = 0
best_model = None


learning_rate = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
max_depth = float(sys.argv[2]) if len(sys.argv) > 2 else 10

with mlflow.start_run():
    xgboost_model = xgb.XGBClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
    
    xgboost_model.fit(X_train, y_train)
    
    y_pred_xgb = xgboost_model.predict(X_test)
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    
    # Log parameters, metrics, and model
    mlflow.log_param('model', 'XGBoost')
    mlflow.log_metric('accuracy', accuracy_xgb)
    mlflow.sklearn.log_model(xgboost_model, 'XGBoostModel')
    print("accuracy_xgb:",accuracy_xgb)
    # Update best model if current model has higher accuracy
    if accuracy_xgb > best_accuracy:
        print(accuracy_xgb)
        best_accuracy = accuracy_xgb
        best_model = xgboost_model

# Save the best model as a pickle file
model_filename = 'best_model.pkl'
joblib.dump(best_model, model_filename)

# test = joblib.load(model_filename)

# new_data = pd.DataFrame({
#     'Machine_ID': [label_encoder_machine.transform(['Machine_1'])[0]],
#     'Reading': [105.0]
# })

# # Make predictions using Decision Tree model
# prediction_dt = test.predict(new_data[['Machine_ID', 'Reading']])
# print(f"Decision Tree Prediction: {label_encoder_sensor.inverse_transform(prediction_dt)}")

"""
# Train a Decision Tree model
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)

# Evaluate the Decision Tree model
y_pred_dt = decision_tree_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt}")

# Train an XGBoost model
xgboost_model = xgb.XGBClassifier(random_state=42)
xgboost_model.fit(X_train, y_train)

# Evaluate the XGBoost model
y_pred_xgb = xgboost_model.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb}")

# Predicting with new data
new_data = pd.DataFrame({
    'Machine_ID': [label_encoder_machine.transform(['Machine_1'])[0]],
    'Reading': [105.0]
})

# Make predictions using Decision Tree model
prediction_dt = decision_tree_model.predict(new_data[['Machine_ID', 'Reading']])
print(f"Decision Tree Prediction: {label_encoder_sensor.inverse_transform(prediction_dt)}")

# Make predictions using XGBoost model
prediction_xgb = xgboost_model.predict(new_data[['Machine_ID', 'Reading']])
print(f"XGBoost Prediction: {label_encoder_sensor.inverse_transform(prediction_xgb)}")
"""


# # client = mlflow.tracking.MlflowClient()
# # runs = client.search_runs(exp_id, "", order_by=["metrics.rmse DESC"], max_results=1)
# # best_run = runs[0]
# # print(best_run)
