import os

import numpy as np
from flask import Flask, render_template, request, jsonify
import json
from flask_cors import CORS, cross_origin
from io import BytesIO
from PIL import Image
import base64



from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import re


app = Flask(__name__)
CORS(app)

MODEL_PATH = './model.h5'

model = load_model(MODEL_PATH)


def predict_disease(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    return prediction


@app.route('/', methods=['GET'])
def home():
    return "Heyy wagwan yeahh"


@app.route('/predict', methods=['POST'])
def predict():
    file = request.form['image']
    filename = request.form['filename']
    file = re.sub('^data:image/.+;base64,', '', file)
    file = Image.open(BytesIO(base64.b64decode(file)))


    print(file)
    base = os.path.dirname(__file__)
    filePath = os.path.join(base, 'uploads', secure_filename(filename))
    file.save(filePath)
    prediction = predict_disease(filePath, model)
    os.remove(filePath)
    labels = ["Black pod rot", "Healthy", "Pod borer"]
    indi = prediction
    print(indi)
    indii = indi.tolist()
    max = np.amax(indii)
    pred, j = (np.where(indii == max))
    predLabel = labels[j[0]]
    print(indii)
    print(j)
    print(max)
    response = {"Label": predLabel, "confidence" : max}
    return jsonify(response)


if __name__ == '__main__':
    app.run()