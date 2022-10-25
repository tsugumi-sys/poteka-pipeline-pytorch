import unittest
from unittest.mock import MagicMock
import json
import os
import shutil
import itertools
from typing import Dict

from hydra import initialize
import hydra
import torch
import numpy as np
import pandas as pd

from common.utils import timestep_csv_names
from evaluate.src.base_evaluator import BaseEvaluator
from common.config import WEATHER_PARAMS, GridSize, PPOTEKACols, ScalingMethod
from tests.evaluate.utils import generate_dummy_test_dataset
from train.src.config import DEVICE
from evaluate.src.utils import normalize_tensor
from common.interpolate_by_gpr import interpolate_by_gpr


class TestBaseEvaluator(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.model = MagicMock()
        self.model_name = "test_model"
        self.input_parameter_names = ["rain", "temperature", "humidity"]
        self.output_parameter_names = ["rain", "temperature", "humidity"]
        self.downstream_directory = "./tmp"
        self.observation_point_file_path = "./common/meta-data/observation_point.json"
        self.test_dataset = generate_dummy_test_dataset(self.input_parameter_names, self.observation_point_file_path)

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

    def test_clean_dfs(self):
        self.base_evaluator.clean_dfs()
        self.assertTrue(self.base_evaluator.results_df.equals(pd.DataFrame()))
        self.assertTrue(self.base_evaluator.metrics_df.equals(pd.DataFrame()))

    def test_load_test_case_dataset(self):
        """This function tests that test dataset of a certain test case loaded to torch Tensor correctly."""
        for test_case_name, test_case_dataset in self.test_dataset.items():
            X_test, y_test = self.base_evaluator.load_test_case_dataset(test_case_name)

            self.assertTrue(torch.equal(X_test, test_case_dataset["input"]))
            self.assertTrue(torch.equal(y_test, test_case_dataset["label"]))

    def test_rescale_pred_tensor(self):
        """This function tests a given tensor is rescaled for a given parameter's scale."""
        tensor = (torch.rand((49, 50)).to(DEVICE) + (-0.50)) * 2  # This tensor is scaled as [-1, 1]
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

    def test_add_result_df_from_pred_tensor(self):
        """This function tests result_df is correctly updated with a given pred_tensor"""
        pred_tensor = torch.ones((50, 50)).to(DEVICE)

        with open(self.observation_point_file_path, "r") as f:
            observation_infos = json.load(f)
        observation_names = list(observation_infos.keys())
        target_cols = [col for col in PPOTEKACols.get_cols() if col not in ["WD1"]]
        label_df = pd.DataFrame({col: [idx] * 35 for idx, col in enumerate(target_cols)}, index=observation_names)
        expect_result_df = label_df.copy()
        expect_result_df["Pred_Value"] = 1.0
        # Note: Pandas cast np.float64 for float.
        expect_result_df["Pred_Value"] = expect_result_df["Pred_Value"].astype(np.float32)
        test_case_name = "sample1"
        expect_result_df["test_case_name"] = test_case_name
        expect_result_df["date"] = self.test_dataset[test_case_name]["date"]
        expect_result_df["predict_utc_time"] = "23-30"
        expect_result_df["target_parameter"] = self.output_parameter_names[0]
        expect_result_df["time_step"] = 1

        # Check if result_df is empty
        self.assertTrue(self.base_evaluator.results_df.equals(pd.DataFrame()))

        self.base_evaluator.add_result_df_from_pred_tensor(
            "sample1",
            time_step=1,
            pred_tensor=pred_tensor,
            label_df=label_df,
            target_param=self.output_parameter_names[0],
        )
        self.assertTrue(self.base_evaluator.results_df.equals(expect_result_df))

    def test_add_metrics_df_from_pred_tensor(self):
        target_cols = [col for col in PPOTEKACols.get_cols() if col not in ["WD1"]]
        test_case_name = "sample1"  # This case starts 23-20.
        time_step = 1
        target_param = "rain"
        pred_tensor = torch.ones((50, 50))

        with open(self.observation_point_file_path, "r") as f:
            observation_infos = json.load(f)
        observation_names = list(observation_infos.keys())
        label_df = pd.DataFrame({col: [1] * 35 for _, col in enumerate(target_cols)}, index=observation_names)

        expect_metrics_df = pd.DataFrame(
            {
                "test_case_name": [test_case_name],
                "predict_utc_time": ["23-30"],
                "target_parameter": [target_param],
                "r2": [1.0],
                "rmse": [0.0],
            }
        )

        # Check if metrics_df is empty
        self.assertTrue(self.base_evaluator.metrics_df.equals(pd.DataFrame()))
        self.base_evaluator.add_metrics_df_from_pred_tensor(
            test_case_name,
            time_step,
            pred_tensor,
            label_df,
            target_param,
        )
        self.assertTrue(self.base_evaluator.metrics_df.equals(expect_metrics_df))

    def test_get_prediction_utc_time(self):
        test_case_name = "sample1"  # this case starts 23-20.
        time_steps = [i for i in range(6)]
        expect_utc_time = ["23-20", "23-30", "23-40", "23-50", "0-0", "0-10"]
        self.base_evaluator.hydra_cfg.preprocess.time_step_minutes = 10  # 10 minutes step.

        for i in time_steps:
            utc_time = self.base_evaluator.get_prediction_utc_time(test_case_name, i)
            self.assertTrue(utc_time, expect_utc_time[i])

    def test_rmse_from_pred_tensor(self):
        # [NOTE] Wind direction (WD1) is not used this independently.
        target_cols = [col for col in PPOTEKACols.get_cols() if col not in ["WD1"]]
        with open(self.observation_point_file_path, "r") as f:
            observation_infos = json.load(f)
        observation_names = list(observation_infos.keys())
        label_df = pd.DataFrame({col: [idx] * 35 for idx, col in enumerate(target_cols)}, index=observation_names)

        for idx, col in enumerate(target_cols):
            pred_tensor = torch.ones(GridSize.HEIGHT, GridSize.WIDTH) * idx
            rmse = self.base_evaluator.rmse_from_pred_tensor(
                pred_tensor=pred_tensor,
                label_df=label_df,
                target_param=WEATHER_PARAMS.get_param_from_ppoteka_col(col),
            )
            self.assertTrue(rmse == 0)

    def test_rmse_from_results_df(self):
        target_cols = [col for col in PPOTEKACols.get_cols() if col not in ["WD1"]]
        results_df = pd.DataFrame({col: [idx] * 35 for idx, col in enumerate(target_cols)})

        for idx, col in enumerate(target_cols):
            results_df["Pred_Value"] = idx
            self.base_evaluator.results_df = results_df

            rmse = self.base_evaluator.rmse_from_results_df(output_param_name=WEATHER_PARAMS.get_param_from_ppoteka_col(col))
            self.assertTrue(rmse == 0.0)

    def test_r2_score_from_pred_tensor(self):
        target_cols = [col for col in PPOTEKACols.get_cols() if col not in ["WD1"]]
        with open(self.observation_point_file_path, "r") as f:
            observation_infos = json.load(f)
        observation_names = list(observation_infos.keys())
        label_df = pd.DataFrame({col: [idx] * 35 for idx, col in enumerate(target_cols)}, index=observation_names)

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
        # The case if predict tensor is invalid shape.
        invalid_pred_tensor = torch.ones((1, 50, 50))
        with self.assertRaises(ValueError):
            _ = self.base_evaluator.get_pred_df_from_tensor(invalid_pred_tensor)

        # The case if predict tensor shapes is grid e.g (50, 50)
        grid_pred_tensor = torch.ones((50, 50))
        pred_df = self.base_evaluator.get_pred_df_from_tensor(grid_pred_tensor)

        with open(self.observation_point_file_path, "r") as f:
            ob_point_data = json.load(f)
        exact_pred_df = pd.DataFrame({"Pred_Value": [1.0] * 35}, dtype=np.float32, index=list(ob_point_data.keys()))
        self.assertTrue(pred_df.equals(exact_pred_df))

        # The cae if predict tensor is one dimention e.g. (ob_point_count).
        ob_point_pred_tensor = torch.ones((35))
        pred_df = self.base_evaluator.get_pred_df_from_tensor(ob_point_pred_tensor)
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
        results_df["predict_utc_time"] = "8-0"
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
        pred_tensors = torch.rand((1, 1, 6, 50, 50))
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
            with self.subTest(test_case_name=test_case_name, predict_utc_time=predict_utc_time):
                self.assertTrue(os.path.exists(os.path.join(self.downstream_directory, filename)))

    def test_update_input_tensor(self):
        scaling_methods = ScalingMethod.get_methods()
        before_input_tensors = [
            torch.zeros(1, len(self.input_parameter_names), self.base_evaluator.hydra_cfg.input_seq_length, GridSize.WIDTH, GridSize.HEIGHT).to(DEVICE),
        ]
        next_frame_tensors = [
            torch.rand((len(self.input_parameter_names), GridSize.WIDTH, GridSize.HEIGHT)).to(DEVICE),
            torch.rand((len(self.input_parameter_names), 35)).to(DEVICE),
        ]
        test_cases = itertools.product(scaling_methods, before_input_tensors, next_frame_tensors)
        before_standarized_info = {param_name: {"mean": 0, "std": 1} for param_name in self.input_parameter_names}
        for (scaling_method, before_input_tensor, next_frame_tensor) in test_cases:
            with self.subTest(
                scaling_method=scaling_method, before_input_tensor_shape=before_input_tensor.shape, next_frame_tensor_shape=next_frame_tensor.shape
            ):
                self._test_update_input_tensor(scaling_method, before_input_tensor, before_standarized_info, next_frame_tensor)

    def _test_update_input_tensor(self, scaling_method: str, before_input_tensor: torch.Tensor, before_standarized_info: Dict, next_frame_tensor: torch.Tensor):
        """This function tests SequentialEvaluator._update_input_tensor
        NOTE: For ease, before_standarized_info shoud be mean=0, std=1.

        """
        self.base_evaluator.hydra_cfg.scaling_method = scaling_method
        updated_tensor, standarized_info = self.base_evaluator.update_input_tensor(before_input_tensor, before_standarized_info, next_frame_tensor)

        if next_frame_tensor.ndim == 2:
            _next_frame_tensor = next_frame_tensor.cpu().detach().numpy().copy()
            next_frame_tensor = torch.zeros(next_frame_tensor.size()[0], GridSize.WIDTH, GridSize.HEIGHT).to(DEVICE)
            for param_dim in range(len(self.input_parameter_names)):
                next_frame_ndarray = interpolate_by_gpr(_next_frame_tensor[param_dim, ...], self.observation_point_file_path)
                next_frame_tensor[param_dim, ...] = torch.from_numpy(next_frame_ndarray).to(DEVICE)
            next_frame_tensor = normalize_tensor(next_frame_tensor, device=DEVICE)

        expect_updated_tensor = torch.cat(
            [
                before_input_tensor.clone().detach()[:, :, 1:, ...],
                torch.reshape(next_frame_tensor, (1, before_input_tensor.size(dim=1), 1, *before_input_tensor.size()[3:])),
            ],
            dim=2,
        )
        expect_standarized_info = {}
        if scaling_method != ScalingMethod.MinMax.value:
            # NOTE: mean and std are 0 and 1 each others. So restandarization is not needed here.
            for param_dim, param_name in enumerate(self.input_parameter_names):
                expect_standarized_info[param_name] = {}
                means = torch.mean(expect_updated_tensor[:, param_dim, ...])
                stds = torch.std(expect_updated_tensor[:, param_dim, ...])
                expect_updated_tensor[:, param_dim, ...] = (expect_updated_tensor[:, param_dim, ...] - means) / stds
                expect_standarized_info[param_name]["mean"] = means
                expect_standarized_info[param_name]["std"] = stds

        self.assertEqual(standarized_info, expect_standarized_info)
        self.assertTrue(torch.equal(updated_tensor, expect_updated_tensor))
