import argparse
import os
from typing import Dict
import sys

import torch
import mlflow
from src.prediction import create_prediction

sys.path.append("..")
from common.data_loader import data_loader
from common.custom_logger import CustomLogger
from common.config import ScalingMethod
from train.src.seq_to_seq import Seq2Seq

logger = CustomLogger("Evaluate_Logger")

device = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(
    upstream_directory: str,
    downstream_directory: str,
    preprocess_downstream_directory: str,
    preprocess_delta: int,
) -> Dict:
    test_data_paths = os.path.join(preprocess_downstream_directory, "meta_test.json")

    scaling_method = ScalingMethod.MinMaxStandard.value
    debug_mode = False
    test_dataset, feature_names = data_loader(test_data_paths, scaling_method=scaling_method, isTrain=False, debug_mode=debug_mode)

    trained_model = torch.load(os.path.join(upstream_directory, "model.pth"))
    model = Seq2Seq(
        num_channels=trained_model["num_channels"],
        kernel_size=trained_model["kernel_size"],
        num_kernels=trained_model["num_kernels"],
        padding=trained_model["padding"],
        activation=trained_model["activation"],
        frame_size=trained_model["frame_size"],
        num_layers=trained_model["num_layers"],
        weights_initializer=trained_model["weights_initializer"],
    )
    model.load_state_dict(trained_model["model_state_dict"])
    model.to(device)
    model.float()

    results = create_prediction(
        model=model,
        test_dataset=test_dataset,
        downstream_directory=downstream_directory,
        preprocess_delta=preprocess_delta,
        scaling_method=scaling_method,
        feature_names=feature_names,
    )

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

    # mlflow_experiment_id = int(os.getenv("MLFLOW_EXPERIMENT_ID", 0))

    upstream_directory = args.upstream
    downstream_directory = args.downstream
    preprocess_downstream_directory = args.preprocess_downstream
    preprocess_delta = args.preprocess_delta

    os.makedirs(downstream_directory, exist_ok=True)

    logger.info(upstream_directory)
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
