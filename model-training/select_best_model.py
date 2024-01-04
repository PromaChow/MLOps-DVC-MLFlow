import json
import mlflow
import joblib
import os

experiment_name = "SensorPrediction"
runs = mlflow.search_runs(experiment_names=[experiment_name], order_by=["start_time desc"], max_results=4)

best_accuracy = 0
best_model = None
accuracy = 0
model_name = "xgboost-model"
experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id
for index, run in runs.iterrows():
    accuracy = run['metrics.accuracy']
    if best_accuracy < accuracy:
        model_path = "mlruns/"+experiment_id+"/"+run['run_id']+"/artifacts/model-training/model.pkl"
        best_accuracy = accuracy
    print(f"Run ID: {run['run_id']}, Accuracy: {run['metrics.accuracy']}")
print(best_accuracy, model_path)

json_filename = 'accuracy.json'
try:
    with open(json_filename, 'r') as file:
        if os.stat("accuracy.json").st_size == 0:
            accuracy_history = {}
            accuracy_history['accuracy'] = best_accuracy

        else:
            accuracy_history = json.load(file)

except FileNotFoundError:
    accuracy_history = {}


else:
    if accuracy_history['accuracy'] < best_accuracy:    
        accuracy_history['accuracy'] = best_accuracy

# Save updated JSON file
with open(json_filename, 'w') as file:
    json.dump(accuracy_history, file)



model = joblib.load(model_path)
model_filename = 'best_model.pkl'
joblib.dump(model, model_filename)