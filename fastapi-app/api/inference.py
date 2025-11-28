from fastapi import APIRouter

import schemas
from services import make_predictions

# Initialize the router
inference_router = APIRouter(prefix="/inference", tags=["Prediction", "Inference"])

# Define the prediction endpoint
@inference_router.post("/predict")
def predict(input_data: schemas.inference.InferenceIn):
    predictions = make_predictions(input_data)

    print(f"Predictions: {predictions}")

    return schemas.inference.InferenceOut(prediction=predictions)
