 
import json
import numpy as np
import os
import cv2
 
from azureml.core.model import Model
 
def init(src):
    global model
    # retrieve the path to the model file using the model name
    model_path = Model.get_model_path('yolov3')
    model = joblib.load(model_path)
def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    # make prediction
    y_hat = model.predict(data)
    return json.dumps(y_hat.tolist())
