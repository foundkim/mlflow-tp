"""Prepare the data for training, validation, and testing."""

import argparse
import pathlib

import numpy as np
import pandas as pd


def prepare_data(input_data: str, output_folder: str) -> None:
    """Prepare data."""
    iris_df = pd.read_csv(input_data, sep=",")

    train, validate, test = np.split(
        iris_df.sample(frac=1, random_state=42),
        [int(0.6 * len(iris_df)), int(0.8 * len(iris_df))],
    )

    train.to_csv(pathlib.Path(output_folder).joinpath("train.csv"))
    validate.to_csv(pathlib.Path(output_folder).joinpath("validate.csv"))
    test.to_csv(pathlib.Path(output_folder).joinpath("test.csv"))


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str)
    parser.add_argument("--output_folder", type=str)

    args = parser.parse_args()

    prepare_data(args.input_data, args.output_folder)
