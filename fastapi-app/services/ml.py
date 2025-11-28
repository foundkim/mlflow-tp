import pandas as pd

from .ml_models import get_iris_model
from schemas import inference


def make_predictions(input_data: inference.InferenceIn):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data.model_dump(mode="python")])

    model = get_iris_model()
    # Make predictions
    predictions = model.predict(input_df)

    # Return the predictions
    return predictions.tolist()
