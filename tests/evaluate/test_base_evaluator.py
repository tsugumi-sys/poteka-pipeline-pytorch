from typing import List, Dict
import unittest
from unittest.mock import MagicMock
import json
import os
import shutil

from hydra import initialize
import hydra
import torch
import numpy as np
import pandas as pd
from common.utils import timestep_csv_names

from evaluate.src.base_evaluator import BaseEvaluator
from train.src.config import DEVICE
from common.config import WEATHER_PARAMS, GridSize, PPOTEKACols


class TestBaseEvaluator(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.model = MagicMock()
        self.model_name = "test_model"
        self.input_parameter_names = ["rain", "temperature", "humidity"]
        self.output_parameter_names = ["rain", "temperature", "humidity"]
        self.downstream_directory = "./tmp"
        self.test_dataset = generate_dummy_test_dataset(self.input_parameter_names)
        self.observation_point_file_path = "./common/meta-data/observation_point.json"

    def setUp(self) -> None:
        if os.path.exists(self.downstream_directory):
            shutil.rmtree(self.downstream_directory)
        os.makedirs(self.downstream_directory, exist_ok=True)

        initialize(config_path="../../conf", version_base=None)
        self.base_evaluator = BaseEvaluator(
            self.model,
            self.model_name,
            self.test_dataset,
            self.input_parameter_names,
            self.output_parameter_names,
            self.downstream_directory,
            self.observation_point_file_path,
        )
        return super().setUp()

    def tearDown(self) -> None:
        hydra.core.global_hydra.GlobalHydra.instance().clear()  # type:ignore
        return super().tearDown()

    def test__init__(self):
        """This tests evaluate initialziation of BaseEvaluator."""
        self.assertTrue(isinstance(self.base_evaluator.results_df, pd.DataFrame))
        self.assertTrue(isinstance(self.base_evaluator.metrics_df, pd.DataFrame))

    def test_load_test_case_dataset(self):
        """This function tests that test dataset of a certain test case loaded to torch Tensor correctly."""
        for test_case_name, test_case_dataset in self.test_dataset.items():
            X_test, y_test = self.base_evaluator.load_test_case_dataset(test_case_name)

            self.assertTrue(torch.equal(X_test, test_case_dataset["input"]))
            self.assertTrue(torch.equal(y_test, test_case_dataset["label"]))

    def test_rescale_pred_tensor(self):
        """This function tests a given tensor is rescaled for a given parameter's scale."""
        tensor = (torch.rand((49, 50)) + (-0.50)) * 2  # This tensor is scaled as [-1, 1]
        rain_rescaled_tensor = self.base_evaluator.rescale_pred_tensor(tensor, target_param="rain")  # A given tensor scaled to [0, 100]
        self.assertTrue(rain_rescaled_tensor.min().item() >= 0.0)
        self.assertTrue(rain_rescaled_tensor.max().item() <= 100.0)

        temp_rescaled_tensor = self.base_evaluator.rescale_pred_tensor(tensor, target_param="temperature")
        self.assertTrue(temp_rescaled_tensor.min().item() >= 10.0)
        self.assertTrue(temp_rescaled_tensor.max().item() <= 45.0)

        humid_rescaled_tensor = self.base_evaluator.rescale_pred_tensor(tensor, target_param="humidity")
        self.assertTrue(humid_rescaled_tensor.min().item() >= 0.0)
        self.assertTrue(humid_rescaled_tensor.max().item() <= 100.0)

        wind_rescaled_tensor = self.base_evaluator.rescale_pred_tensor(tensor, target_param="u_wind")
        self.assertTrue(wind_rescaled_tensor.min().item() >= -10.0)
        self.assertTrue(wind_rescaled_tensor.max().item() <= 10.0)

    def test_rmse_from_label_df(self):
        # [NOTE] Wind direction (WD1) is not used this independently.
        target_cols = [col for col in PPOTEKACols.get_cols() if col not in ["WD1"]]
        label_df = pd.DataFrame({col: [idx] * 35 for idx, col in enumerate(target_cols)})
        for idx, col in enumerate(target_cols):
            pred_tensor = torch.ones(GridSize.HEIGHT, GridSize.WIDTH) * idx
            rmse = self.base_evaluator.rmse_from_label_df(
                pred_tensor=pred_tensor,
                label_df=label_df,
                target_param=WEATHER_PARAMS.get_param_from_ppoteka_col(col),
            )
            self.assertTrue(rmse == 0)

    def test_r2_score_from_pred_tensor(self):
        target_cols = [col for col in PPOTEKACols.get_cols() if col not in ["WD1"]]
        label_df = pd.DataFrame({col: [idx] * 35 for idx, col in enumerate(target_cols)})
        for idx, col in enumerate(target_cols):
            pred_tensor = torch.ones(GridSize.HEIGHT, GridSize.WIDTH) * idx
            r2_score = self.base_evaluator.r2_score_from_pred_tensor(
                pred_tensor=pred_tensor,
                label_df=label_df,
                target_param=WEATHER_PARAMS.get_param_from_ppoteka_col(col),
            )
            self.assertTrue(r2_score == 1.0)

    def test_r2_score_from_results_df(self):
        target_cols = [col for col in PPOTEKACols.get_cols() if col not in ["WD1"]]
        results_df = pd.DataFrame({col: [idx] * 35 for idx, col in enumerate(target_cols)})

        # NOTE: This test calcurate r2 score from all result_df.
        for idx, col in enumerate(target_cols):
            results_df["Pred_Value"] = [idx] * 35
            self.base_evaluator.results_df = results_df

            # Calcurate from all case.
            r2_score = self.base_evaluator.r2_score_from_results_df(
                output_param_name=WEATHER_PARAMS.get_param_from_ppoteka_col(col),
            )
            self.assertTrue(r2_score == 1.0)

        # NOTE: This test calculate r2 score from date queried results_df
        for idx, col in enumerate(target_cols):
            # NOTE: Creating another dataframe with a date different from a target result dataframe.
            another_date_results_df = results_df.copy()
            another_date_results_df["date"] = "2020-10-12"
            another_date_results_df["Pred_Value"] = [idx + 1] * 35

            results_df["Pred_Value"] = [idx] * 35
            results_df["date"] = "2021-1-5"

            self.base_evaluator.results_df = pd.concat([results_df, another_date_results_df], axis=0)

            # Calcurate from all case.
            r2_score = self.base_evaluator.r2_score_from_results_df(
                output_param_name=WEATHER_PARAMS.get_param_from_ppoteka_col(col),
                target_date="2021-1-5",
            )
            self.assertTrue(r2_score == 1.0)

        # NOTE: This test calculate r2 score from a result dataframe queried with data and is_tc_case flag.
        for idx, col in enumerate(target_cols):
            # NOTE: Creating another dataframe with a date different from a target result dataframe and case_type.
            another_date_results_df = results_df.copy()
            another_date_results_df["date"] = "2020-10-12"
            another_date_results_df["Pred_Value"] = [idx + 1] * 35
            another_date_results_df["case_type"] = "tc"

            results_df["Pred_Value"] = [idx] * 35
            results_df["date"] = "2021-1-5"
            results_df["case_type"] = "not_tc"

            self.base_evaluator.results_df = pd.concat([results_df, another_date_results_df], axis=0)

            # Calculate from all case
            r2_score = self.base_evaluator.r2_score_from_results_df(
                output_param_name=WEATHER_PARAMS.get_param_from_ppoteka_col(col),
                target_date="2021-1-5",
                is_tc_case=False,
            )
            self.assertTrue(r2_score == 1.0)

    def test_query_result_df(self):
        target_cols = [col for col in PPOTEKACols.get_cols() if col not in ["WD1"]]
        results_df = pd.DataFrame({col: [idx] * 35 for idx, col in enumerate(target_cols)})
        results_df["Pred_Value"] = [1] * 35
        results_df["date"] = "2020-1-5"
        results_df["case_type"] = "tc"

        another_results_df = results_df.copy()
        another_results_df["Pred_Value"] = [2] * 35
        another_results_df["date"] = "2021-3-5"
        another_results_df["case_type"] = "not_tc"

        self.base_evaluator.results_df = pd.concat([results_df, another_results_df], axis=0)

        # NOTE: test querying without args
        df = self.base_evaluator.query_result_df()
        self.assertTrue(df.equals(self.base_evaluator.results_df))

        # NOTE: test querying with date
        df = self.base_evaluator.query_result_df(target_date="2020-1-5")
        self.assertTrue(df.equals(results_df))

        # NOTE: test querying with is_tc_case flag
        df = self.base_evaluator.query_result_df(is_tc_case=False)
        self.assertTrue(df.equals(another_results_df))

        # NOTE: test querying with date amd is_tc_case flag
        df = self.base_evaluator.query_result_df(target_date="2021-3-5", is_tc_case=False)
        self.assertTrue(df.equals(another_results_df))

        # NOTE: querying is invalid and return empty dataframe.
        df = self.base_evaluator.query_result_df(target_date="2023-1-1")
        self.assertTrue(df.empty)

    def test_get_pred_df_from_tensor(self):
        pred_tensor = torch.ones((50, 50))
        pred_df = self.base_evaluator.get_pred_df_from_tensor(pred_tensor)

        with open(self.observation_point_file_path, "r") as f:
            ob_point_data = json.load(f)
        exact_pred_df = pd.DataFrame({"Pred_Value": [1.0] * 35}, dtype=np.float32, index=list(ob_point_data.keys()))
        self.assertTrue(pred_df.equals(exact_pred_df))

    def test_save_results_to_csv(self):
        self.base_evaluator.results_df = pd.DataFrame({"sample": [1]})
        self.base_evaluator.save_results_df_to_csv(save_dir_path=self.downstream_directory)

        results_df_from_csv = pd.read_csv(os.path.join(self.downstream_directory, "predict_result.csv"), index_col=0)
        self.assertTrue(self.base_evaluator.results_df.equals(results_df_from_csv))

    def test_save_metrics_to_csv(self):
        self.base_evaluator.metrics_df = pd.DataFrame({"sample": [1]})
        self.base_evaluator.save_metrics_df_to_csv(self.downstream_directory)

        metrics_df_from_csv = pd.read_csv(os.path.join(self.downstream_directory, "predict_metrics.csv"), index_col=0)
        self.assertTrue(self.base_evaluator.metrics_df.equals(metrics_df_from_csv))

    def test_scatter_plot(self):
        target_cols = [col for col in PPOTEKACols.get_cols() if col not in ["WD1"]]
        results_df = pd.DataFrame({col: [idx] * 35 for idx, col in enumerate(target_cols)})
        results_df["Pred_Value"] = [1] * 35
        results_df["date"] = "2020-1-5"
        results_df["date_time"] = "2020-1-5 800UTC start"
        results_df["case_type"] = "tc"

        another_results_df = results_df.copy()
        another_results_df["Pred_Value"] = [2] * 35
        another_results_df["date"] = "2021-3-5"
        another_results_df["date_time"] = "2021-1-5 800UTC start"
        another_results_df["case_type"] = "not_tc"

        self.base_evaluator.results_df = pd.concat([results_df, another_results_df], axis=0)
        self.base_evaluator.scatter_plot(self.downstream_directory)

        for file_name in ["all_cases.png", "2020-1-5_cases.png", "2021-3-5_cases.png"]:
            with self.subTest(file_name=file_name):
                self.assertTrue(os.path.exists(os.path.join(self.downstream_directory, file_name)))

    def test_geo_plot(self):
        test_case_name = "sample1"
        pred_tensors = {idx: torch.rand((50, 50)) for idx in range(6)}
        self.base_evaluator.hydra_cfg.use_dummy_data = True
        self.base_evaluator.geo_plot(test_case_name=test_case_name, save_dir_path=self.downstream_directory, pred_tensors=pred_tensors)

        start_utc_time = self.test_dataset[test_case_name]["start"]  # this ends with .csv
        _timestep_csv_names = timestep_csv_names(time_step_minutes=10)
        start_utc_time_idx = _timestep_csv_names.index(start_utc_time)
        predict_end_utc_time_idx = start_utc_time_idx + 6
        predict_utc_times = _timestep_csv_names[start_utc_time_idx:predict_end_utc_time_idx]
        if predict_end_utc_time_idx > len(_timestep_csv_names) - 1:
            for i in range(predict_end_utc_time_idx - len(_timestep_csv_names)):
                predict_utc_times.append(_timestep_csv_names[i])

        for predict_utc_time in predict_utc_times:
            filename = predict_utc_time.replace(".csv", ".parquet.gzip")
            with self.subTest(predict_utc_time=predict_utc_time):
                self.assertTrue(os.path.exists(os.path.join(self.downstream_directory, filename)))


def generate_dummy_test_dataset(input_parameter_names: List) -> Dict:
    """This function creates dummy test dataset."""
    dummy_tensor = torch.ones((5, len(input_parameter_names), 6, 50, 50), dtype=torch.float, device=DEVICE)
    sample1_input_tensor = dummy_tensor.clone().detach()
    sample1_label_tensor = dummy_tensor.clone().detach()
    sample2_input_tensor = dummy_tensor.clone().detach()
    sample2_label_tensor = dummy_tensor.clone().detach()
    # change value for each input parameters
    # rain -> 0, temperature -> 1, humidity -> 0.5)
    for i in range(len(input_parameter_names)):
        val = 1 / i if i > 0 else 0
        sample1_input_tensor[:, i, :, :, :] = val
        sample1_label_tensor[:, i, :, :, :] = val
        sample2_input_tensor[:, i, :, :, :] = val
        sample2_label_tensor[:, i, :, :, :] = val
    label_dfs = {}
    for i in range(sample1_input_tensor.size()[2]):
        data = {}
        for col in PPOTEKACols.get_cols():
            data[col] = np.ones((10))
            if col == "hour-rain":
                data[col] *= 0
            elif col == "RH1":
                data[col] /= 2
        label_dfs[i] = pd.DataFrame(data)

    test_dataset = {
        "sample1": {
            "date": "2022-01-01",
            "start": "23-20.csv",
            "input": sample1_input_tensor,
            "label": sample1_label_tensor,
            "label_df": label_dfs,
            "standarize_info": {"rain": {"mean": 1.0, "std": 0.1}, "temperature": {"mean": 2.0, "std": 0.2}, "humidity": {"mean": 3.0, "std": 0.3}},
        },
        "sample2": {
            "date": "2022-01-02",
            "start": "1-0.csv",
            "input": sample2_input_tensor,
            "label": sample2_label_tensor,
            "label_df": label_dfs,
            "standarize_info": {"rain": {"mean": 1.0, "std": 0.1}, "temperature": {"mean": 2.0, "std": 0.2}, "humidity": {"mean": 3.0, "std": 0.3}},
        },
    }

    return test_dataset