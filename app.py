# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:13:32 2024

@author: S145
"""

from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

# Correct the file path for Windows
pickle_in = open(r"C:\docker_app\iris_model_pkl.pkl", "rb")
model = pickle.load(pickle_in)

@app.route('/predict', methods=["GET"])
def predict_class():
    # Ensure the parameter names are consistent
    sepal_length = float(request.args.get("sepal_length"))
    sepal_width = float(request.args.get("sepal_width"))
    petal_length = float(request.args.get("petal_length"))
    petal_width = float(request.args.get("petal_width"))
    
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return "Model prediction is: " + str(prediction[0])

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
