from typing import Dict, Optional

from mlflow import entities
from mlflow.tracking import MlflowClient


def get_latest_run_by_experiment_name(experiment_id: str) -> Optional[entities.Run]:
    mlflow_client = MlflowClient()
    runs = mlflow_client.search_runs(experiment_ids=[experiment_id], filter_string="attributes.status = 'RUNNING'", order_by=["attributes.start_time DESC"])
    if len(runs) == 0:
        return None
    return runs[0]


def get_override_hydra_conf(experiment_id: str) -> str:
    running_runs = get_latest_run_by_experiment_name(experiment_id=experiment_id)
    if running_runs is None:
        return ""
    run_params: Dict = running_runs.data.params
    return run_params["override_hydra_conf"] if "override_hydra_conf" in run_params else ""
