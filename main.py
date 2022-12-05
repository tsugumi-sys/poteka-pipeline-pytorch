import json
import os
import shutil
import logging

import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from common.line_notify import send_line_notify
from common.utils import get_mlflow_tag_from_input_parameters
from common.omegaconf_manager import OmegaconfManager
from train.src.common.constants import WeightsInitializer
from train.src.utils.model_interactor import ModelName

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    if not torch.cuda.is_available():
        logger.warning("\N{no entry}: GPU is not AVAILABLE.")

    mlflow_run_name = get_mlflow_tag_from_input_parameters(cfg.input_parameters)
    mlflow_experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID", 0)
    # [NOTE]: mlflow.active_run doesnt work here.
    # override_hydra_conf = get_override_hydra_conf(mlflow_experiment_id)
    # Check root dir settings
    if not os.path.exists(cfg.project_root_dir_path):
        raise ValueError(f"Invalid project_root_dir_path setting in conf/config.yaml. The path {cfg.project_root_dir_path} does not exist.")

    # Initialize data directory. This data directory is temporaly directory for saving results.
    # These results are also saved in mlflow direcotory (./mlruns)
    if os.path.exists(os.path.join(cfg.project_root_dir_path, "data")):
        logger.warning("./data directory is automatically deleted.")
        shutil.rmtree(os.path.join(cfg.project_root_dir_path, "data"), ignore_errors=True)
    os.makedirs("./data")

    if not ModelName.is_valid(cfg.model_name):
        raise ValueError(f"Invalid Model Name {cfg.model_name}. This should be in {ModelName.all_names()}")

    if not WeightsInitializer.is_valid(cfg.weights_initializer):
        raise ValueError(f"Invalid weight initializer {cfg.weight_initializer}. This should be in {WeightsInitializer.all_names()}")

    logging_core_hydra_parameters(cfg)

    try:
        with mlflow.start_run():
            # Save whole configuration of this run.
            omegaconf_manager = OmegaconfManager()
            hydra_file_path = os.path.join(cfg.project_root_dir_path, "data", "hydra.yaml")
            omegaconf_manager.save(cfg, hydra_file_path, except_keys="secrets")

            # Save scaling method in pearent run.
            mlflow.log_param("model_name", cfg.model_name)
            mlflow.log_param("scaling_method", cfg.scaling_method)
            mlflow.set_tag("mlflow.runName", mlflow_run_name)

            # Run preprocess run in child run.
            preprocess_run = mlflow.run(
                uri="./preprocess", entry_point="preprocess", backend="local", env_manager="local", parameters={"hydra_file_path": hydra_file_path}
            )
            preprocess_run = mlflow.tracking.MlflowClient().get_run(preprocess_run.run_id)

            # Update hydra conf file
            current_dir = os.getcwd()
            preprocess_artifact_uri = os.path.join(current_dir, "mlruns/", str(mlflow_experiment_id), preprocess_run.info.run_id, "artifacts/")
            cfg = omegaconf_manager.update(cfg, {"train.upstream_dir_path": preprocess_artifact_uri})
            omegaconf_manager.save(cfg, hydra_file_path, except_keys=["secrets"])

            # Run train run in child run.
            train_run = mlflow.run(uri="./train", entry_point="train", backend="local", env_manager="local", parameters={"hydra_file_path": hydra_file_path})
            train_run = mlflow.tracking.MlflowClient().get_run(train_run.run_id)

            # Update hydra conf file
            model_file_dir_path = train_run.info.artifact_uri
            model_file_dir_path = model_file_dir_path.replace("file://", "")
            cfg = omegaconf_manager.update(
                cfg, {"evaluate.model_file_dir_path": model_file_dir_path, "evaluate.preprocess_meta_file_dir_path": preprocess_artifact_uri}
            )
            omegaconf_manager.save(cfg, hydra_file_path, except_keys=["secrets"])

            # Run evaluate run in child run.
            evaluate_run = mlflow.run(
                uri="./evaluate", entry_point="evaluate", backend="local", env_manager="local", parameters={"hydra_file_path": hydra_file_path}
            )
            evaluate_run = mlflow.tracking.MlflowClient().get_run(evaluate_run.run_id)

            mlflow.log_artifact(hydra_file_path)
        send_line_notify("[Succesfully ended]: ppoteka-pipeine-pytorch", cfg["secrets"]["line_notify_api_token"])
    except:
        send_line_notify("[Faild]: ppotela-pipeline-pytorch", cfg["secrets"]["line_notify_api_token"])


def logging_core_hydra_parameters(cfg: DictConfig):
    print("=" * 50)
    print("config.yaml core parameters.")
    print("=" * 50)
    logger.info(f"input_parameters: {cfg.input_parameters}")
    logger.info(f"model_name: {cfg.model_name}")
    logger.info(f"scaling_method: {cfg.scaling_method}")
    logger.info(f"weights_initializer: {cfg.weights_initializer}")
    logger.info(f"is_obpoint_labeldata: {cfg.is_obpoint_labeldata}")
    logger.info(f"multi_parameters_model.return_sequences: {cfg.multi_parameters_model.return_sequences}")
    logger.info(f"single_parameter_model.return_sequences: {cfg.single_parameter_model.return_sequences}")
    logger.info(f"use_dummy_data: {cfg.use_dummy_data}")

    print("=" * 50)
    print("train.train.yaml core parameters.")
    print("=" * 50)
    logger.info(f"train.epochs: {cfg.train.epochs}")
    logger.info(f"train.batch_size: {cfg.train.batch_size}")
    logger.info(f"train.train_separately: {cfg.train.train_separately}")
    logger.info(f"train.is_max_datasize_limit: {cfg.train.is_max_datasize_limit}")


if __name__ == "__main__":
    main()
