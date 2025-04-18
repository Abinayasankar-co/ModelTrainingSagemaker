import joblib
import os
import pandas as pd
import json

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))

def input_fn(request_body, content_type):
    if content_type == "application/json":
        data = json.loads(request_body)
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, accept):
    if accept == "application/json":
        return json.dumps(prediction.tolist()), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
