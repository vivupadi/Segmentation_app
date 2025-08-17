import torch
from flask import Flask, jsonify, request
from PIL import Image
import numpy as np

import Qt

app = Flask(__name__)

#Load models
#model =


#Load image 
def preprocess(image):
    image = image.resize(224,224)
    image = np.array(image)/255
    image = np.expand_dims(image, axis= 0)
    return image
    

@app.route('/predict', methods=['POST'])

def predict():
    file = request.files['file']
    image = Image.open(file)
    image = preprocess(image)

    prediction = model.predict(image)
    return jsonify({predictions: prediction.tolist()})

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000)