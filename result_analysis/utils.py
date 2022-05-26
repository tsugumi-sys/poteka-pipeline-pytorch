from typing import Dict, List
import re

import mlflow
from mlflow.tracking.client import MlflowClient
import pandas as pd


def get_metrics_list(mlflow_metrics_history: List[mlflow.entities.Metric]) -> Dict:
    res = {}
    for metric in mlflow_metrics_history:
        res[str(metric.step)] = metric.value
    return res


def get_results_dict(eval_runs: List[mlflow.entities.Run]) -> Dict:
    # One value metric pattern
    all_sample_rmse_pattern = "All_sample_RMSE"
    ten_minutes_prediction_rmse_pattern = "One_Hour_Prediction_RMSE"

    # Mluitple vlayues metric pattern
    # TC case
    tc_case_regex = "^TC_case_.+"
    seq_tc_case_regex = "^Sequential_TC_case_.+"

    # NOT TC case
    not_tc_case_regex = "^NOT_TC_case_.+"
    seq_not_tc_case_regex = "^Sequential_NOT_TC_case_.+"

    # R2 score
    r2_regex = "^r2_.+"

    all_sample_rmse: Dict[str, float] = {}
    ten_minutes_prediction_rmse: Dict[str, float] = {}

    tc_case_rmses: Dict[str, list] = {}
    seq_tc_case_rmses: Dict[str, list] = {}

    not_tc_case_rmses: Dict[str, list] = {}
    seq_not_tc_case_rmses: Dict[str, list] = {}

    r2_scores: Dict[str, float] = {}
    for run in eval_runs:
        metrics_key = run.data.tags["mlflow.runName"].replace("_evaluation", "")

        tc_case_rmses[metrics_key] = {}
        seq_tc_case_rmses[metrics_key] = {}

        not_tc_case_rmses[metrics_key] = {}
        seq_not_tc_case_rmses[metrics_key] = {}

        r2_scores[metrics_key] = {}

        # Get Metrics
        metrics: Dict[str, float] = run.data.metrics
        for key, val in metrics.items():
            if key == all_sample_rmse_pattern:
                all_sample_rmse[metrics_key] = val
            elif key == ten_minutes_prediction_rmse_pattern:
                ten_minutes_prediction_rmse[metrics_key] = val

            elif re.match(tc_case_regex, key) is not None:
                metrics_history = MlflowClient().get_metric_history(run.info.run_id, key)
                tc_case_rmses[metrics_key][key] = get_metrics_list(metrics_history)
            elif re.match(seq_tc_case_regex, key) is not None:
                metrics_history = MlflowClient().get_metric_history(run.info.run_id, key)
                seq_tc_case_rmses[metrics_key][key] = get_metrics_list(metrics_history)

            elif re.match(not_tc_case_regex, key) is not None:
                metrics_history = MlflowClient().get_metric_history(run.info.run_id, key)
                not_tc_case_rmses[metrics_key][key] = get_metrics_list(metrics_history)

            elif re.match(seq_not_tc_case_regex, key) is not None:
                metrics_history = MlflowClient().get_metric_history(run.info.run_id, key)
                seq_not_tc_case_rmses[metrics_key][key] = get_metrics_list(metrics_history)

            if re.match(r2_regex, key) is not None:
                r2_scores[metrics_key][key] = val

    return {
        "all_sample_rmse": all_sample_rmse,
        "ten_minutes_prediction_rmse": ten_minutes_prediction_rmse,
        "tc_case_rmses": tc_case_rmses,
        "seq_tc_case_rmses": seq_tc_case_rmses,
        "not_tc_case_rmses": not_tc_case_rmses,
        "seq_not_tc_case_rmses": seq_not_tc_case_rmses,
        "r2_scores": r2_scores,
    }


def get_metrics_history_df(metrics_histories: Dict[str, List]) -> pd.DataFrame:
    time_index_col = []
    rmse_col = []
    input_parameters_col = []
    for input_param_name, histories in metrics_histories.items():
        for case_name, result_dict in histories.items():
            time_idxs = list(result_dict.keys())
            rmse_values = list(result_dict.values())
            input_parameters = [input_param_name] * len(time_idxs)

            time_index_col += time_idxs
            rmse_col += rmse_values
            input_parameters_col += input_parameters

    df = pd.DataFrame({"time_index": time_index_col, "rmse": rmse_col, "input_parameters": input_parameters_col})
    return df


def get_r2_scores_df(results_dict: Dict) -> Dict[str, pd.DataFrame]:
    param_names_list, seq_param_names_list = [], []
    r2_scores_list, seq_r2_scores_list = [], []
    for param_name, results_dict in results_dict.items():
        for key, val in results_dict.items():
            if "Sequential" in key:
                seq_param_names_list.append(param_name)
                seq_r2_scores_list.append(val)
            else:
                param_names_list.append(param_name)
                r2_scores_list.append(val)

    df = pd.DataFrame({"input_parameters": param_names_list, "r2_score": r2_scores_list})
    seq_df = pd.DataFrame({"input_parameters": seq_param_names_list, "r2_score": seq_r2_scores_list})

    return {"one_hour": df, "sequential": seq_df}
