import argparse
import os
import json

import numpy as np
import mlflow
from sklearn.datasets import load_diabetes
from src.extract_data import imputer, fit_imputer, parse_parquet


def main():
    parser = argparse.ArgumentParser(
        description="Make dataset",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--downstream",
        type=str,
        default="/data/preprocess/",
        help="downstream directory",
    )

    args = parser.parse_args()

    downstream_directory = args.downstream

    train_output_destination = os.path.join(
        downstream_directory,
        "train",
    )

    test_output_destination = os.path.join(
        downstream_directory,
        "test",
    )

    valid_output_destination = os.path.join(
        downstream_directory,
        "valid",
    )

    os.makedirs(downstream_directory, exist_ok=True)
    os.makedirs(train_output_destination, exist_ok=True)
    os.makedirs(test_output_destination, exist_ok=True)
    os.makedirs(valid_output_destination, exist_ok=True)

    raw_data = load_diabetes(as_frame=True)
    df = raw_data["data"].merge(raw_data["target"], how="inner", left_index=True, right_index=True)
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    half_length = int(len(df) / 2)
    three_quarter_length = half_length + int(half_length / 2)
    train_df = df[:half_length].astype(np.float32)
    test_df = df[half_length:three_quarter_length].astype(np.float32)
    valid_df = df[three_quarter_length:].astype(np.float32)

    train_df, transformer = imputer(train_df)
    test_df = fit_imputer(test_df, transformer=transformer)
    valid_df = fit_imputer(valid_df, transformer=transformer)

    train_file_name = parse_parquet(train_df, train_output_destination)
    test_file_name = parse_parquet(test_df, test_output_destination)
    valid_file_name = parse_parquet(valid_df, valid_output_destination)

    meta_train = {"filename": train_file_name}
    meta_test = {"filename": test_file_name}
    meta_valid = {"filename": valid_file_name}

    meta_train_filepath = os.path.join(
        downstream_directory,
        "meta_train.json",
    )
    meta_test_filepath = os.path.join(
        downstream_directory,
        "meta_test.json",
    )
    meta_valid_path = os.path.join(
        downstream_directory,
        "meta_valid.json",
    )
    with open(meta_train_filepath, "w") as f:
        json.dump(meta_train, f)
    with open(meta_test_filepath, "w") as f:
        json.dump(meta_test, f)
    with open(meta_valid_path, "w") as f:
        json.dump(meta_valid, f)

    mlflow.log_artifacts(
        downstream_directory,
        artifact_path="downstream_directory",
    )


if __name__ == "__main__":
    main()
