import logging
import os
import sys
from typing import List
import json

import hydra
from omegaconf import DictConfig
import mlflow

sys.path.append("..")
from train.src.trainer import Trainer
from train.src.learning_curve_plot import learning_curve_plot
from train.src.config import DEVICE
from common.utils import get_mlflow_tag_from_input_parameters, split_input_parameters_str
from common.data_loader import train_data_loader
from common.custom_logger import CustomLogger

logger = CustomLogger("Train_Logger", level=logging.INFO)


def start_run(
    input_parameters: List[str],
    upstream_directory: str,
    downstream_directory: str,
    scaling_method: str,
    is_max_datasize_limit: bool = False,
    use_test_model: bool = False,
):
    train_data_paths = os.path.join(upstream_directory, "meta_train.json")
    valid_data_paths = os.path.join(upstream_directory, "meta_valid.json")

    train_input_tensor, train_label_tensor = train_data_loader(
        train_data_paths,
        scaling_method=scaling_method,
        isMaxSizeLimit=is_max_datasize_limit,
        debug_mode=False,
    )
    valid_input_tensor, valid_label_tensor = train_data_loader(
        valid_data_paths,
        scaling_method=scaling_method,
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
        checkpoints_directory=downstream_directory,
        use_test_model=use_test_model,
        hydra_overrides=[f"train.use_test_model={use_test_model}", f"input_parameters={input_parameters}"],
    )
    results = trainer.run()

    meta_models = {}
    for model_name, result in results.items():
        _ = learning_curve_plot(
            save_dir_path=downstream_directory,
            model_name=model_name,
            training_losses=result["training_loss"],
            validation_losses=result["validation_loss"],
            validation_accuracy=result["validation_accuracy"],
        )
        meta_models[model_name] = {}
        meta_models[model_name]["return_sequences"] = result["return_sequences"]
        meta_models[model_name]["input_parameters"] = result["input_parameters"]
        meta_models[model_name]["output_parameters"] = result["output_parameters"]

    with open(os.path.join(downstream_directory, "meta_models.json"), "w") as f:
        json.dump(meta_models, f)

    # Save results to mlflow
    mlflow.log_artifacts(downstream_directory)
    logger.info("Training finished")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.debug_mode is True:
        logger.info("Running train in debug mode ...")

    input_parameters = split_input_parameters_str(cfg.input_parameters)
    mlflow.set_tag("mlflow.runName", get_mlflow_tag_from_input_parameters(input_parameters) + "_train & tuning")

    upstream_dir_path = cfg.train.upstream_dir_path
    downstream_dir_path = cfg.train.downstream_dir_path

    os.makedirs(downstream_dir_path, exist_ok=True)

    start_run(
        input_parameters=input_parameters,
        upstream_directory=upstream_dir_path,
        downstream_directory=downstream_dir_path,
        scaling_method=cfg.scaling_method,
        is_max_datasize_limit=cfg.train.is_max_datasize_limit,
        use_test_model=cfg.train.use_test_model,
    )


if __name__ == "__main__":
    main()
