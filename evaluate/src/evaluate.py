import argparse
import json
import os
import time
from typing import Dict

import mlflow
import tensorflow as tf
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from joblib import load


def evaluate(
    valid_data_directory: str,
    train_data_directory: str,
    model_file_path: str,
    model_type: str,
) -> Dict:
    valid_dataset = pd.read_parquet(
        os.path.join(valid_data_directory),
        engine="pyarrow",
    )

    X_valid, y_valid = valid_dataset.drop(columns="target", axis=1), valid_dataset["target"]

    if model_type == "skreg":
        model = load(model_file_path)

        start = time.time()
        predictions = model.predict(X_valid)
        end = time.time()

        total_time = end - start
        total_tested = len(X_valid)
        _r2_score = r2_score(y_valid, predictions)
        rmse_score = mean_squared_error(y_valid, predictions, squared=False)

        evaluation = {
            "total_tested": total_tested,
            "r2_score": _r2_score,
            "rmse_score": rmse_score,
            "total_time": total_time,
        }

        return {"evaluation": evaluation, "predictions": predictions}

    else:
        train_dataset = pd.read_parquet(
            train_data_directory,
            engine="pyarrow",
        )
        X_train = train_dataset.drop(columns="target", axis=1)

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_valid = scaler.fit_transform(X_valid)

        model = tf.keras.models.load_model(model_file_path)

        start = time.time()
        predictions = model.predict(X_valid)
        end = time.time()

        total_time = end - start
        total_tested = len(X_valid)
        _r2_score = r2_score(y_valid, predictions)
        rmse_score = mean_squared_error(y_valid, predictions, squared=False)

        evaluation = {
            "total_tested": total_tested,
            "r2_score": _r2_score,
            "rmse_score": rmse_score,
            "total_time": total_time,
        }

        return {"evaluation": evaluation, "predictions": predictions}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # parser.add_argument(
    #     "--upstream",
    #     type=str,
    #     default="./preprocess/data/preprocess/train",
    #     help="upstream directory",
    # )
    parser.add_argument(
        "--downstream",
        type=str,
        default="/data/evaluate/",
        help="downstream directory",
    )
    parser.add_argument(
        "--valid_data_directory",
        type=str,
        default="./preprocess/data/preprocess/valid",
        help="valid data directory",
    )
    parser.add_argument(
        "--train_data_directory",
        type=str,
        default="/preprocess/data/preprocess/valid/",
        help="train data direcory",
    )
    parser.add_argument(
        "--model_file_path",
        type=str,
        default="/model/",
        help="model file path",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=[
            "simplenet",
            "skreg",
        ],
        help="model type",
    )
    args = parser.parse_args()
    mlflow_experiment_id = int(os.getenv("MLFLOW_EXPERIMENT_ID", 0))

    # upstream_directory = args.upstream
    downstream_directory = args.downstream
    valid_data_directory = args.valid_data_directory
    train_data_directory = args.train_data_directory
    model_file_path = args.model_file_path
    model_type = args.model_type

    # os.makedirs(upstream_directory, exist_ok=True)
    os.makedirs(downstream_directory, exist_ok=True)

    result = evaluate(
        valid_data_directory=valid_data_directory,
        train_data_directory=train_data_directory,
        model_file_path=model_file_path,
        model_type=model_type,
    )

    log_file = os.path.join(downstream_directory, f"{mlflow_experiment_id}.json")

    with open(log_file, "w") as f:
        json.dump(log_file, f)

    mlflow.log_metric(
        "total_tested",
        result["evaluation"]["total_tested"],
    )
    mlflow.log_metric(
        "total_time",
        result["evaluation"]["total_time"],
    )
    mlflow.log_metric(
        "r2_score",
        result["evaluation"]["r2_score"],
    )
    mlflow.log_metric(
        "rmse_score",
        result["evaluation"]["rmse_score"],
    )
    mlflow.log_artifact(log_file)


if __name__ == "__main__":
    main()
