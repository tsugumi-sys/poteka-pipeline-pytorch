import os
from typing import Dict
import sys
import hydra
from omegaconf import DictConfig

import torch
import mlflow

from src.prediction import create_prediction

sys.path.append("..")
from common.data_loader import data_loader
from common.custom_logger import CustomLogger
from common.utils import get_mlflow_tag_from_input_parameters, split_input_parameters_str
from train.src.seq_to_seq import Seq2Seq
from train.src.model_for_test import TestModel

logger = CustomLogger("Evaluate_Logger")

device = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(
    upstream_directory: str,
    downstream_directory: str,
    preprocess_downstream_directory: str,
    preprocess_time_step_minutes: int,
    use_dummy_data: bool,
    use_test_model: bool,
    scaling_method: str,
) -> Dict:
    test_data_paths = os.path.join(preprocess_downstream_directory, "meta_test.json")
    debug_mode = False
    test_dataset, feature_names = data_loader(
        test_data_paths, scaling_method=scaling_method, isTrain=False, debug_mode=debug_mode, use_dummy_data=use_dummy_data
    )

    trained_model = torch.load(os.path.join(upstream_directory, "model.pth"))
    if use_test_model is True:
        logger.info("... using test model ...")
        model = TestModel()
    else:
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
        preprocess_time_step_minutes=preprocess_time_step_minutes,
        scaling_method=scaling_method,
        feature_names=feature_names,
        use_dummy_data=use_dummy_data,
    )

    return results


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    input_parameters = split_input_parameters_str(cfg.input_parameters)
    mlflow.set_tag("mlflow.runName", get_mlflow_tag_from_input_parameters(input_parameters) + "_evaluate")

    # mlflow_experiment_id = int(os.getenv("MLFLOW_EXPERIMENT_ID", 0))

    upstream_directory = cfg.evaluate.model_file_dir_path
    downstream_directory = cfg.evaluate.downstream_dir_path
    preprocess_downstream_directory = cfg.evaluate.preprocess_meta_file_dir_path
    preprocess_time_step_minutes = cfg.preprocess.time_step_minutes

    os.makedirs(downstream_directory, exist_ok=True)

    logger.info(upstream_directory)
    results = evaluate(
        upstream_directory=upstream_directory,
        downstream_directory=downstream_directory,
        preprocess_downstream_directory=preprocess_downstream_directory,
        preprocess_time_step_minutes=preprocess_time_step_minutes,
        use_dummy_data=cfg.use_dummy_data,
        use_test_model=cfg.train.use_test_model,
        scaling_method=cfg.scaling_method,
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
