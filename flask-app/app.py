import pickle
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from flask import Flask, jsonify, render_template, request
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello():

  try:
    model_path = "best_model.pkl"
    loaded_model = joblib.load(model_path)
  except FileNotFoundError:
    model_path = "../model-training/best_model.pkl"
    loaded_model = joblib.load(model_path)

  try:
    encoder_path = 'label_encoder_machine.pkl'
    with open(encoder_path, 'rb') as file:
      label_encoder_machine = pickle.load(file)
  except FileNotFoundError:
    encoder_path = "../model-training/label_encoder_machine.pkl"
    with open(encoder_path, 'rb') as file:
      label_encoder_machine = pickle.load(file)  

  prediction = None
  if request.method == 'POST':
    reading = float(request.form['reading'])
    machine_id = request.form['machine_id']
    new_data = pd.DataFrame({
      'Machine_ID': [label_encoder_machine.transform([machine_id])[0]],
      'Reading': [reading]
    })
    prediction = loaded_model.predict(new_data[['Machine_ID', 'Reading']])
    prediction = str(prediction[0])
    # print(prediction)
    return jsonify({'prediction': prediction})

  return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
  app.run(host="0.0.0.0",port=9000,debug=True)