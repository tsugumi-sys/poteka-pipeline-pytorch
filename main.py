import os

import mlflow
import hydra
from omegaconf import DictConfig

from common.utils import get_mlflow_tag_from_input_parameters
from common.mlflow_utils import get_override_hydra_conf


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    mlflow_run_name = get_mlflow_tag_from_input_parameters(cfg.input_parameters)
    mlflow_experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID", 0)
    # [NOTE]: mlflow.active_run doesnt work here.
    override_hydra_conf = get_override_hydra_conf(mlflow_experiment_id)

    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", mlflow_run_name)
        preprocess_run = mlflow.run(
            uri="./preprocess",
            entry_point="preprocess",
            backend="local",
            env_manager="local",
            parameters={
                "use_dummy_data": cfg.use_dummy_data,
                "override_hydra_conf": str(override_hydra_conf),
            },
        )
        preprocess_run = mlflow.tracking.MlflowClient().get_run(preprocess_run.run_id)

        current_dir = os.getcwd()
        preprocess_artifact_uri = os.path.join(
            current_dir,
            "mlruns/",
            str(mlflow_experiment_id),
            preprocess_run.info.run_id,
            "artifacts/",
        )

        train_run = mlflow.run(
            uri="./train",
            entry_point="train",
            backend="local",
            parameters={
                "upstream_dir_path": preprocess_artifact_uri,
                "use_dummy_data": cfg.use_dummy_data,
                "use_test_model": cfg.train.use_test_model,
                "override_hydra_conf": override_hydra_conf,
            },
            env_manager="local",
        )
        train_run = mlflow.tracking.MlflowClient().get_run(train_run.run_id)

        model_file_dir_path = train_run.info.artifact_uri
        model_file_dir_path = model_file_dir_path.replace("file://", "")
        evaluate_run = mlflow.run(
            uri="./evaluate",
            entry_point="evaluate",
            backend="local",
            env_manager="local",
            parameters={
                "preprocess_meta_file_dir_path": preprocess_artifact_uri,
                "model_file_dir_path": model_file_dir_path,
                "use_dummy_data": cfg.use_dummy_data,
                "use_test_model": cfg.train.use_test_model,
                "override_hydra_conf": override_hydra_conf,
            },
        )
        evaluate_run = mlflow.tracking.MlflowClient().get_run(evaluate_run.run_id)


if __name__ == "__main__":
    main()
