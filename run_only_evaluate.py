import logging
import os
from typing import List, Optional
import shutil

import hydra
import mlflow
import torch
from omegaconf import DictConfig

from common.line_notify import send_line_notify
from common.omegaconf_manager import OmegaconfManager

logger = logging.getLogger(__name__)


def get_artifact_path(run_id: str) -> str:
    run = mlflow.tracking.MlflowClient().get_run(run_id)
    return run.info.artifact_uri.replace("file://", "")


def get_experiment_id(run_id: str):
    run = mlflow.tracking.MlflowClient().get_run(run_id)
    return run.info.experiment_id


def check_run_ids(parent_run_id: str, child_run_ids: Optional[List[str]] = None):
    if parent_run_id is None:
        raise ValueError(
            "Set parent, preprocess and train run_id in conf/evaluate/evaluate.yaml for re-run evaluation process."
        )

    if child_run_ids is not None:
        for child_id in child_run_ids:
            if parent_run_id != mlflow.tracking.MlflowClient().get_run(child_id).data.tags["mlflow.parentRunId"]:
                raise ValueError(f"The parent run ({parent_run_id}) doesn not have the child run ({child_id})")


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    logger.info("=========== Rerun Evalaution Process ===========")
    if not torch.cuda.is_available():
        logger.warning("\N{no entry}: GPU is not AVAILABLE.")

    if os.path.exists(os.path.join(cfg.project_root_dir_path, "data")):
        logger.warning("./data directory is automatically deleted.")
        shutil.rmtree(os.path.join(cfg.project_root_dir_path, "data"), ignore_errors=True)
    os.makedirs("./data")

    parent_run_id = cfg.evaluate.re_run.parent_run_id
    check_run_ids(parent_run_id)

    mlflow.set_experiment(experiment_id=get_experiment_id(parent_run_id))

    # Copy paernt runs hydra.yaml to data direcotry
    omegaconf_manager = OmegaconfManager()
    parent_run_cfg = omegaconf_manager.load(os.path.join(get_artifact_path(parent_run_id), "hydra.yaml"))
    hydra_file_path = os.path.join(cfg.project_root_dir_path, "data", "hydra.yaml")
    omegaconf_manager.save(parent_run_cfg, hydra_file_path)

    try:
        with mlflow.start_run():
            # Save scaling method in pearent run.
            mlflow.log_param("model_name", parent_run_cfg.model_name)
            mlflow.log_param("scaling_method", parent_run_cfg.scaling_method)
            mlflow.log_param("original run id", parent_run_id)
            mlflow.set_tag("mlflow.runName", "Rerun Evaluate")

            # Run evaluate run in child run.
            _ = mlflow.run(
                uri="./evaluate",
                entry_point="evaluate",
                backend="local",
                env_manager="local",
                parameters={"hydra_file_path": hydra_file_path},
            )

            mlflow.log_artifact(hydra_file_path)
        send_line_notify("[Succesfully ended]: ppoteka-pipeine-pytorch", cfg["secrets"]["line_notify_api_token"])
    except Exception:
        send_line_notify("[Faild]: ppotela-pipeline-pytorch", cfg["secrets"]["line_notify_api_token"])


if __name__ == "__main__":
    main()
