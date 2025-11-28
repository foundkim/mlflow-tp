import mlflow

from core import config

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

def get_iris_model():
    model_uri = config.MODEL_URI
    return mlflow.pyfunc.load_model(model_uri)
