"""Evaluate a trained model using MLflow and scikit-learn."""

import argparse
import pathlib

import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score


def evaluate_model(
    data_root_folder: str,
    model_name: str = "iris_model",
    evaluation_threshold: float = 0.8,
) -> bool:
    """Evaluate given model on given dataset."""
    # Load the model
    model = mlflow.pyfunc.load_model("mlflow")

    # Load the test data
    test_df = pd.read_csv(pathlib.Path(data_root_folder).joinpath("test.csv"), sep=",")

    x_test = test_df.iloc[:, 1:-1]
    y_test = test_df.iloc[:, -1]

    # Evaluate the model
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Validation Accuracy: {accuracy}")

    if accuracy < evaluation_threshold:
        raise ValueError(
            f"Model accuracy {accuracy} is below the threshold of {evaluation_threshold}"
        )


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--data_folder", type=str)
    parser.add_argument("--model_name", type=str)

    args = parser.parse_args()

    evaluate_model(args.data_folder, args.model_name)
