import joblib
import numpy as np
from sagemaker_inference import content_types, decoder, encoder

model_path = "/opt/ml/model/rf_model.joblib"
rf = joblib.load(model_path)

def input_fn(request_body, content_type):
    if content_type == content_types.JSON:
        payload = decoder.decode(request_body, content_type)
        arr = np.array(payload["instances"])
        return arr
    elif content_type == content_types.CSV:
        arr = np.genfromtxt(request_body.splitlines(), delimiter=",")
        return arr
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    preds = model.predict(input_data)
    probs = model.predict_proba(input_data)[:, 1].tolist()
    return {"predictions": preds.tolist(), "probabilities": probs}

def output_fn(prediction, accept):
    return encoder.encode(prediction, accept)
