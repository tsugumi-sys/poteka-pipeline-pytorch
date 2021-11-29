import argparse
import os
from typing import Dict
import sys
import logging

import mlflow
import mlflow.tensorflow
from sklearn.metrics import mean_squared_error as mse
from src.prediction import create_prediction

sys.path.append("..")
from common.data_loader import data_loader
from common.custom_logger import CustomLogger

logging.basicConfig(
    level=logging.INFO,
)
logger = CustomLogger("Evaluate_Logger")


def evaluate(
    upstream_directory: str,
    downstream_directory: str,
    preprocess_downstream_directory: str,
    preprocess_delta: int,
) -> Dict:
    valid_data_paths = os.path.join(preprocess_downstream_directory, "meta_valid.json")

    valid_dataset = data_loader(valid_data_paths, isTrain=False)

    model = mlflow.pyfunc.load_model(upstream_directory)
    results = create_prediction(model, valid_dataset, downstream_directory, preprocess_delta)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--parent_run_name",
        type=str,
        default="defaultRun",
        help="Parent Run Name",
    )
    parser.add_argument(
        "--upstream",
        type=str,
        default="/data/train/model",
        help="upstream directory (model file directory created by mlflow.log_model)",
    )
    parser.add_argument(
        "--downstream",
        type=str,
        default="/data/evaluate/",
        help="downstream directory",
    )
    parser.add_argument(
        "--preprocess_downstream",
        type=str,
        default="/data/preprocess/",
        help="preprocess data folder for validation data.",
    )
    parser.add_argument(
        "--preprocess_delta",
        type=int,
        default=10,
        help="preprocess delta (time step) for validation data.",
    )

    args = parser.parse_args()

    mlflow.set_tag("mlflow.runName", args.parent_run_name + "_evaluation")

    mlflow_experiment_id = int(os.getenv("MLFLOW_EXPERIMENT_ID", 0))

    upstream_directory = args.upstream
    downstream_directory = args.downstream
    preprocess_downstream_directory = args.preprocess_downstream
    preprocess_delta = args.preprocess_delta

    os.makedirs(downstream_directory, exist_ok=True)

    results = evaluate(
        upstream_directory=upstream_directory,
        downstream_directory=downstream_directory,
        preprocess_downstream_directory=preprocess_downstream_directory,
        preprocess_delta=preprocess_delta,
    )

    for key, value in results.items():
        mlflow.log_metric(key, value)
        logger.info(f"Evaluation: {key}: {value}")

    mlflow.log_artifacts(
        downstream_directory,
        artifact_path="evaluations",
    )
    logger.info("Evaluation successfully ended.")


if __name__ == "__main__":
    main()
