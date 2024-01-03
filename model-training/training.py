from mlflow.models import infer_signature
import pickle
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


# Display the first few rows of the training data
# print("Training Data:")
# print(train_data.head())

# # Display the first few rows of the testing data
# print("Testing Data:")
# print(test_data.head())

label_encoder_machine = LabelEncoder()
label_encoder_sensor = LabelEncoder()

train_data['Machine_ID'] = label_encoder_machine.fit_transform(train_data['Machine_ID'])
train_data['Sensor_ID'] = label_encoder_sensor.fit_transform(train_data['Sensor_ID'])

test_data['Machine_ID'] = label_encoder_machine.transform(test_data['Machine_ID'])
test_data['Sensor_ID'] = label_encoder_sensor.transform(test_data['Sensor_ID'])

train_data.fillna(0, inplace=True)  

X_train = train_data[['Machine_ID', 'Reading']]
y_train = train_data['Sensor_ID']

X_test = test_data[['Machine_ID', 'Reading']]
y_test = test_data['Sensor_ID']

experiment_name = 'SensorPrediction'
mlflow.set_experiment(experiment_name)

best_accuracy = 0
best_model = None


learning_rate = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10




with mlflow.start_run():
    xgboost_model = xgb.XGBClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
    
    xgboost_model.fit(X_train, y_train)
    
    y_pred_xgb = xgboost_model.predict(X_test)
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    
    mlflow.log_param('model', 'XGBoost')
    mlflow.log_metric('accuracy', accuracy_xgb)
    mlflow.sklearn.log_model(xgboost_model, 'XGBoostModel')
    print("accuracy_xgb:",accuracy_xgb)
    signature = infer_signature(X_train, xgboost_model.predict(X_train))
    model_info = mlflow.sklearn.log_model(
        sk_model=xgboost_model,
        artifact_path="model-training",
        signature=signature,
        input_example=X_train,
        registered_model_name="xgboost-model",
    )
    if accuracy_xgb > best_accuracy:
        print(accuracy_xgb)
        best_accuracy = accuracy_xgb
        best_model = xgboost_model


with open('label_encoder_machine.pkl', 'wb') as file:
    pickle.dump(label_encoder_machine, file)

with open('label_encoder_sensor.pkl', 'wb') as file:
    pickle.dump(label_encoder_sensor, file)

