# inference.py

import tensorflow as tf
import json
import numpy as np

def model_fn(model_dir):
    return tf.keras.models.load_model(model_dir)

def input_fn(request_body, content_type):
    if content_type == 'application/json':
        data = json.loads(request_body)
        return np.array(data['instances'])
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, content_type):
    if content_type == 'application/json':
        return json.dumps(prediction.tolist())
    raise ValueError(f"Unsupported content type: {content_type}")
