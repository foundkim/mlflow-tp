import argparse
import joblib
import mlflow
import pathlib
import pandas as pd
from sklearn.metrics import accuracy_score


root_folder = pathlib.Path("__file__").resolve().parent


def evaluate_model(
    data_root_folder: str,
    evaluation_threshold: float = 0.8,
) -> bool:
    # Load the model
    model = mlflow.pyfunc.load_model("iris_model")

    # Load the test data
    test_df = pd.read_csv(
        pathlib.Path(data_root_folder).joinpath("test.csv"), sep=","
    )

    x_test = test_df.iloc[:, 1:-1]
    y_test = test_df.iloc[:, -1]

    # Evaluate the model
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Validation Accuracy: {accuracy}")

    if accuracy >= evaluation_threshold:
        return True
    return False

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "--data_folder",
        type=str,
        default=str(root_folder.joinpath("data", "processed")),
    )

    args = parser.parse_args()

    evaluate_model(args.data_folder)
