from flask import Flask, jsonify, request
from flask_ngrok import run_with_ngrok
from tensorflow import keras
import json
import os

app = Flask(__name__)
run_with_ngrok(app)


current_directory = os.path.dirname(os.path.abspath(__file__))


model_json_path = os.path.join(current_directory, 'files', 'model.json')
weights_path = os.path.join(current_directory, 'files', 'weights.h5')
config_json_path = os.path.join(current_directory, 'files', 'config.json')


with open(model_json_path, 'r') as f:
    model_json = json.load(f)

model = keras.models.model_from_json(json.dumps(model_json))


model.load_weights(weights_path)


with open(config_json_path, 'r') as f:
    classes = json.load(f)

@app.route("/")
def home():
    return "<h1>Welcome</h1>"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = request.get_json()
    input_data = data['inputs']
    predictions = model.predict(input_data)
    results = []
    for prediction in predictions:
        class_index = prediction.argmax()
        predicted_class = classes[class_index]
        result = {'class': predicted_class, 'confidence': float(prediction[class_index])}
        results.append(result)
    return jsonify(results)

#app.run()

