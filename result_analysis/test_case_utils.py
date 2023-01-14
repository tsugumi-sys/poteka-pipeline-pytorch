import os
from datetime import datetime, timedelta
from enum import Enum
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

###
# Utils for mlflow artifacts
###


class MlflowConfig:
    tracking_uri = "../mlruns"
    experiment_id = "12"
    eval_run_ids = {"rain_only": "fe8315b263294e9a856d83d4eb4ac136", "rain_temp_humid": "26257fd09cb84827a1f29064d2e02a7a"}
    eval_artifact_dir = "evaluations"
    reuse_predict_eval_dir = "model/sequential_evaluation/reuse_predict"
    update_inputs_eval_dir = "model/sequential_evaluation/update_inputs"
    metrics_csv = "predict_metrics.csv"
    result_csv = "predict_result.csv"


mlflow_client = mlflow.MlflowClient(MlflowConfig.tracking_uri)


def get_artifact_path(run_id):
    artifact_location = mlflow_client.get_experiment(MlflowConfig.experiment_id).artifact_location
    return artifact_location.replace("file://", "")


def get_eval_artifacts_dir(run_id):
    return os.path.join(get_artifact_path(run_id), run_id, "artifacts", MlflowConfig.eval_artifact_dir)


def get_metrics_df(run_id, eval_type: str = "reuse_predict"):
    eval_artifacts_path = get_eval_artifacts_dir(run_id)
    if eval_type == "reuse_predict":
        evaltype_dir = MlflowConfig.reuse_predict_eval_dir
    else:
        evaltype_dir = MlflowConfig.update_inputs_eval_dir
    return pd.read_csv(os.path.join(eval_artifacts_path, evaltype_dir, MlflowConfig.metrics_csv))


def get_result_df(run_id, eval_type: str = "reuse_predict"):
    eval_artifacts_path = get_eval_artifacts_dir(run_id)
    if eval_type == "reuse_predict":
        evaltype_dir = MlflowConfig.reuse_predict_eval_dir
    else:
        evaltype_dir = MlflowConfig.update_inputs_eval_dir
    return pd.read_csv(os.path.join(eval_artifacts_path, evaltype_dir, MlflowConfig.result_csv))


def get_max_rainfalls_per_case(run_id):
    pred_df = get_result_df(run_id)
    results = {"test_case_name": [], "predict_utc_time": [], "max_rainfall": []}
    for (case_name, pred_utc_time), df in pred_df.groupby(by=["test_case_name", "predict_utc_time"]):
        results["test_case_name"].append(case_name)
        results["predict_utc_time"].append(pred_utc_time)
        results["max_rainfall"].append(df["hour-rain"].max())
    return pd.DataFrame(results)


def get_metrics_with_maxrainfall(run_id):
    metrics_df = get_metrics_df(run_id)
    max_rainfall = get_max_rainfalls_per_case(run_id)
    metrics_df = pd.merge(metrics_df, max_rainfall, how="left", left_on=["test_case_name", "predict_utc_time"], right_on=["test_case_name", "predict_utc_time"])
    return metrics_df[["test_case_name", "predict_utc_time", "rmse", "max_rainfall"]].sort_values(by="max_rainfall", ascending=False)


###
# Utils for testcase
###
class TargetCases:
    good_case_names = ["TC_case_2020-10-12_8-0_start", "NOT_TC_case_2019-10-12_8-40_start"]
    bad_case_names = ["TC_case_2020-09-14_5-0_start", "TC_case_2020-10-12_7-0_start"]


class WeatherParams(str, Enum):
    rainfall = "rainfall"
    temperature = "temperature"
    humidity = "humidity"

    @classmethod
    def valid(self, param: str):
        if param not in WeatherParams._member_names_:
            raise ValueError(f"Use params in {WeatherParams._member_names_} instead of {param}")

    @classmethod
    def unit(self, param: str):
        WeatherParams.valid(param)

        if param == WeatherParams.rainfall:
            return "mm/h"
        elif param == WeatherParams.temperature:
            return "℃"
        else:
            return "%"

    @classmethod
    def min(self, param: str):
        WeatherParams.valid(param)

        if param == WeatherParams.rainfall:
            return 0
        elif param == WeatherParams.temperature:
            return 20
        else:
            return 30

    @classmethod
    def max(self, param: str):
        WeatherParams.valid(param)

        if param == WeatherParams.rainfall:
            return 100
        elif param == WeatherParams.temperature:
            return 40
        else:
            return 90

    @classmethod
    def get_cmap(self, param: str):
        WeatherParams.valid(param)

        if param == WeatherParams.rainfall:
            cmap_data = [
                (1.0, 1.0, 1.0),
                (0.3137255012989044, 0.8156862854957581, 0.8156862854957581),
                (0.0, 1.0, 1.0),
                (0.0, 0.8784313797950745, 0.501960813999176),
                (0.0, 0.7529411911964417, 0.0),
                (0.501960813999176, 0.8784313797950745, 0.0),
                (1.0, 1.0, 0.0),
                (1.0, 0.627451002597808, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 0.125490203499794, 0.501960813999176),
                (0.9411764740943909, 0.250980406999588, 1.0),
                (0.501960813999176, 0.125490203499794, 1.0),
            ]
            return mcolors.ListedColormap(cmap_data, "precipitation")
        elif param == WeatherParams.temperature:
            return plt.cm.inferno
        else:
            return plt.cm.Greens

    @classmethod
    def get_clevels(self, param: str):
        WeatherParams.valid(param)

        if param == WeatherParams.rainfall:
            return [0, 5, 7.5, 10, 15, 20, 30, 40, 50, 70, 100]
        elif param == WeatherParams.temperature:
            return [i for i in range(WeatherParams.min(param), WeatherParams.max(param), 1)]
        else:
            return [i for i in range(WeatherParams.min(param), WeatherParams.max(param), 2)]


def datetime_range(start: datetime, end: datetime, delta: timedelta):
    current = start
    while current <= end:
        yield current
        current += delta


def timestep_names(year: int = 2020, month: int = 1, date: int = 1, delta: int = 10) -> list[str]:
    dts = [f"{dt.hour}-{dt.minute}" for dt in datetime_range(datetime(year, month, date, 0), datetime(year, month, date, 23, 59), timedelta(minutes=delta))]
    return dts


class TestCase:
    data_root_dir = "../../poteka-sample-data/"
    datafile_fmt = "csv"
    predictfile_fmt = "parquet.gzip"

    def __init__(self, test_case_name: str, input_seq_length: int = 6, label_seq_length: int = 6, timestep_delta: int = 10, run_id=None) -> None:
        self.run_id = run_id
        if run_id is not None:
            self.mlflow_artifact_dir = get_eval_artifacts_dir(run_id)
        self.test_case_name = test_case_name
        self.input_seq_length = input_seq_length
        self.label_seq_length = label_seq_length
        self.parsed_test_case_name = self._parse_test_case_name()
        self.timestep_names = timestep_names(delta=timestep_delta)

    @property
    def date(self) -> str:
        return self.parsed_test_case_name["date"]

    @property
    def start_time(self) -> str:
        return self.parsed_test_case_name["start_time"]

    @property
    def input_times(self) -> list[str]:
        start_time_idx = self.timestep_names.index(self.start_time)
        return self.timestep_names[start_time_idx - self.input_seq_length : start_time_idx]

    @property
    def pred_times(self) -> list[str]:
        start_time_idx = self.timestep_names.index(self.start_time)
        return self.timestep_names[start_time_idx : start_time_idx + self.label_seq_length]

    @property
    def data_pathes(self) -> dict:
        paths = {}
        paths["oneday_data"] = {
            "input": [os.path.join(self.data_root_dir, "oneday_data", self.date, f"{f}.{self.datafile_fmt}") for f in self.input_times],
            "label": [os.path.join(self.data_root_dir, "oneday_data", self.date, f"{f}.{self.datafile_fmt}") for f in self.pred_times],
        }
        for param in WeatherParams._member_names_:
            paths[param] = {
                "input": [os.path.join(self.data_root_dir, param, self.date, f"{f}.{self.datafile_fmt}") for f in self.input_times],
                "label": [os.path.join(self.data_root_dir, param, self.date, f"{f}.{self.datafile_fmt}") for f in self.pred_times],
            }
            if param == WeatherParams.rainfall and self.run_id is not None:
                paths[param].update(
                    {
                        "predict": [
                            os.path.join(self.mlflow_artifact_dir, MlflowConfig.reuse_predict_eval_dir, self.test_case_name, f"{f}.{self.predictfile_fmt}")
                            for f in self.pred_times
                        ]
                    }
                )
        return paths

    @property
    def result_df(self):
        if self.run_id is None:
            raise ValueError("set run_id to get result_df")
        df = get_result_df(self.run_id)
        return df.loc[df.test_case_name == self.test_case_name]

    @property
    def metrrics_df(self):
        if self.run_id is None:
            raise ValueError("set run_id to get metrics_df")
        return get_metrics_df(self.run_id)

    def _parse_test_case_name(self) -> dict:
        split_by_underscore = self.test_case_name.split("_")
        if self._is_tc_case():
            return {"date": split_by_underscore[2], "start_time": split_by_underscore[3]}
        else:
            return {"date": split_by_underscore[3], "start_time": split_by_underscore[4]}

    def _is_tc_case(self):
        return self.test_case_name.startswith("TC")


###
# tests
###
# test_case = TestCase(TargetCases.good_case_names[0], run_id=MlflowConfig.eval_run_ids["rain_only"])
# for item in test_case.data_pathes.values():
#     for paths in item.values():
#         is_exist = [os.path.exists(p) for p in paths]
#         assert all(is_exist)
