import logging
import os
import sys
from typing import List
import json
import argparse

from omegaconf import DictConfig
import mlflow

sys.path.append("..")
from train.src.trainer import Trainer  # noqa: E402
from train.src.utils.learning_curve_plot import learning_curve_plot  # noqa: E402
from common.config import DEVICE  # noqa: E402
from common.utils import get_mlflow_tag_from_input_parameters, split_input_parameters_str  # noqa: E402
from common.data_loader import train_data_loader  # noqa: E402
from common.omegaconf_manager import OmegaconfManager
from common.custom_logger import CustomLogger  # noqa: E402

logger = CustomLogger("Train_Logger", level=logging.INFO)


def main(hydra_cfg: DictConfig):
    input_parameters = split_input_parameters_str(hydra_cfg.input_parameters)
    upstream_directory = hydra_cfg.train.upstream_dir_path
    downstream_directory = hydra_cfg.train.downstream_dir_path
    scaling_method = hydra_cfg.scaling_method
    is_obpoint_label_data = hydra_cfg.is_obpoint_labeldata
    is_max_datasize_limit = hydra_cfg.train.is_max_datasize_limit
    use_test_model = hydra_cfg.train.use_test_model

    os.makedirs(downstream_directory, exist_ok=True)

    train_data_paths = os.path.join(upstream_directory, "meta_train.json")
    valid_data_paths = os.path.join(upstream_directory, "meta_valid.json")

    observation_point_file_path = "../common/meta-data/observation_point.json"

    train_input_tensor, train_label_tensor = train_data_loader(
        train_data_paths,
        observation_point_file_path=observation_point_file_path,
        scaling_method=scaling_method,
        isObPointLabelData=is_obpoint_label_data,
        isMaxSizeLimit=is_max_datasize_limit,
        debug_mode=False,
    )
    valid_input_tensor, valid_label_tensor = train_data_loader(
        valid_data_paths,
        observation_point_file_path=observation_point_file_path,
        scaling_method=scaling_method,
        isObPointLabelData=is_obpoint_label_data,
        isMaxSizeLimit=is_max_datasize_limit,
        debug_mode=False,
    )

    train_input_tensor, train_label_tensor = train_input_tensor.to(DEVICE), train_label_tensor.to(DEVICE)
    valid_input_tensor, valid_label_tensor = valid_input_tensor.to(DEVICE), valid_label_tensor.to(DEVICE)

    trainer = Trainer(
        input_parameters=input_parameters,
        train_input_tensor=train_input_tensor,
        train_label_tensor=train_label_tensor,
        valid_input_tensor=valid_input_tensor,
        valid_label_tensor=valid_label_tensor,
        hydra_cfg=hydra_cfg,
        checkpoints_directory=downstream_directory,
    )
    results = trainer.run()

    meta_models = {}
    for model_name, result in results.items():
        _ = learning_curve_plot(
            save_dir_path=downstream_directory,
            model_name=model_name,
            training_losses=result["training_loss"],  # type: ignore
            validation_losses=result["validation_loss"],  # type: ignore
            validation_accuracy=result["validation_accuracy"],  # type: ignore
        )
        meta_models[model_name] = {}
        meta_models[model_name]["return_sequences"] = result["return_sequences"]  # type: ignore
        meta_models[model_name]["input_parameters"] = result["input_parameters"]  # type: ignore
        meta_models[model_name]["output_parameters"] = result["output_parameters"]  # type: ignore

    with open(os.path.join(downstream_directory, "meta_models.json"), "w") as f:
        json.dump(meta_models, f)

    # Save results to mlflow
    mlflow.log_artifacts(downstream_directory)
    logger.info("Training finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train arguments")
    parser.add_argument("--hydra_file_path", type=str, help="Hydra configuration file saved in main.py.")
    args = parser.parse_args()
    omegaconf_manager = OmegaconfManager()
    hydra_cfg = omegaconf_manager.load(args.hydra_file_path)
    main(hydra_cfg)
