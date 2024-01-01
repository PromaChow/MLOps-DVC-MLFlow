import pickle
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from flask import Flask, jsonify, request
app = Flask(__name__)


@app.route('/')
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
  




  new_data = pd.DataFrame({
    'Machine_ID': [label_encoder_machine.transform(['Machine_1'])[0]],
    'Reading': [105.0]
})
  prediction = loaded_model.predict(new_data[['Machine_ID', 'Reading']])


  return str(prediction[0])

# @app.route('/appointments', methods=["GET"])
# def getAppointments():
#   return jsonify(appointments)

# @app.route('/appointment/<id>', methods=["GET"])
# def getAppointment(id):
#   id = int(id) - 1
#   return jsonify(appointments[id])

if __name__ == "__main__":
  app.run(host="0.0.0.0",port=9000)