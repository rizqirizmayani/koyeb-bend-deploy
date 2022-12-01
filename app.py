from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

# App Initialization
app = Flask(__name__)

# Load the Model
scaler = pickle.load(open('scaler.pkl','rb'))
model = tf.keras.models.load_model('model.h5')

# Endpoint for Homepage
@app.route("/")
def home():
    return "<h1>It Works!</h1>"

# Endpoint for Prediction
@app.route("/predict", methods=['POST'])
def model_predict():
    args = request.json
    new_data = {
      'Temperature[C]': args.get('Temperature[C]'),
      'Humidity[%]': args.get('Humidity[%]'), 
      'TVOC[ppb]': args.get('TVOC[ppb]'), 
      'eCO2[ppm]': args.get('eCO2[ppm]'),
      'Raw H2' : args.get('Raw H2'),
      'Ethanol' : args.get('Ethanol'),
      'Pressure[hPa]' : args.get('Pressure[hPa]'),
      'PM1.0' : args.get('PM1.0'),
      'PM2.5' : args.get('PM2.5'),
      'NC0.5' : args.get('NC0.5'),
      'NC1.0' : args.get('NC1.0'),
      'NC2.5' : args.get('NC2.5'),
      'CNT' : args.get('CNT')
    }

    new_data = pd.DataFrame([new_data])
    print('New Data : ', new_data)

    #Scaling
    X = scaler.transform(new_data)

    # Predict
    y_label = ['Off','On']
    y_pred = int(np.round(model.predict(X)[0][0]))

    # Return the Response
    response = jsonify(
      result = str(y_pred), 
      label_names = y_label[y_pred])

    return response


if __name__ == "__main__":
    app.run(host=0.0.0.0, debug=True)
