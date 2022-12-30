import unittest
from unittest.mock import MagicMock
import os
import shutil
import json

import hydra
from hydra import compose, initialize
import torch
import pandas as pd
import numpy as np
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from common.config import GridSize, MinMaxScalingValue
from common.utils import timestep_csv_names
from evaluate.src.combine_models_evaluator import CombineModelsEvaluator
from evaluate.src.utils import save_parquet
from tests.evaluate.utils import generate_dummy_test_dataset
from common.config import DEVICE


class TestCombineModelsEvaluator(unittest.TestCase):
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
        hydra_cfg = compose(config_name="config")
        self.combine_models_evaluator = CombineModelsEvaluator(
            self.model,
            self.model_name,
            self.test_dataset,
            self.input_parameter_names,
            self.output_parameter_names,
            self.downstream_directory,
            self.observation_point_file_path,
            hydra_cfg,
        )
        self.combine_models_evaluator.hydra_cfg.use_dummy_data = True
        return super().setUp()

    def tearDown(self) -> None:
        shutil.rmtree(self.downstream_directory)
        hydra.core.global_hydra.GlobalHydra.instance().clear()  # type:ignore
        return super().tearDown()

    def test_runs(self):
        model_return_values = [
            torch.zeros((1, len(self.output_parameter_names), 1, GridSize.WIDTH, GridSize.HEIGHT)).to(DEVICE),
            torch.zeros((1, len(self.output_parameter_names), 1, 35)).to(DEVICE),
        ]

        for model_return_val in model_return_values:
            with self.subTest(model_return_value_shape=model_return_val.shape):
                is_ob_point_label = True if model_return_val.ndim == 4 else False
                self.combine_models_evaluator.test_dataset = generate_dummy_test_dataset(
                    self.input_parameter_names,
                    self.observation_point_file_path,
                    self.combine_models_evaluator.hydra_cfg.input_seq_length,
                    self.combine_models_evaluator.hydra_cfg.label_seq_length,
                    is_ob_point_label,
                )

                self._test_run(model_return_val)

                # NOTE: Initialize results_df and metrics_df for next run.
                self.combine_models_evaluator.results_df = pd.DataFrame()
                self.combine_models_evaluator.metrics_df = pd.DataFrame()

    @ignore_warnings(category=ConvergenceWarning)
    def _test_run(self, model_return_value: torch.Tensor):
        self._generate_dummy_pred_files(is_grid_data=False)
        self.model.return_value = model_return_value
        # NOTE: `save_attention_maps` is tests in `test_evaluate_test_case`
        self.combine_models_evaluator.hydra_cfg.evaluate.save_attention_maps = False
        results = self.combine_models_evaluator.run()

        self.assertTrue(results[f"{self.model_name}_combine_models_r2"] == 1.0)
        self.assertTrue(results[f"{self.model_name}_combine_models_rmse"] == 0.0)

        with open(self.observation_point_file_path, "r") as f:
            ob_point_data = json.load(f)

        pred_df = pd.DataFrame(index=list(ob_point_data.keys()))
        pred_df["Pred_Value"] = 0.0  # NOTE: Pred_Value is rain and scaled to orignal scale.
        pred_df["Pred_Value"] = pred_df["Pred_Value"].astype(np.float32)
        label_seq_length = self.combine_models_evaluator.hydra_cfg.label_seq_length

        expect_result_df = pd.DataFrame()
        expect_metrics_df = pd.DataFrame()
        for test_case_name in self.test_dataset.keys():
            start_utc_time = self.test_dataset[test_case_name]["start"]  # this ends with .csv
            _timestep_csv_names = timestep_csv_names(time_step_minutes=10)
            start_utc_time_idx = _timestep_csv_names.index(start_utc_time)
            predict_end_utc_time_idx = start_utc_time_idx + 6
            predict_utc_times = _timestep_csv_names[start_utc_time_idx:predict_end_utc_time_idx]
            if predict_end_utc_time_idx > len(_timestep_csv_names) - 1:
                for i in range(predict_end_utc_time_idx - len(_timestep_csv_names)):
                    predict_utc_times.append(_timestep_csv_names[i])
            predict_utc_times = [i.replace(".csv", "") for i in predict_utc_times]

            # create expect_result_df
            for seq_idx in range(label_seq_length):
                label_df = self.test_dataset[test_case_name]["label_df"][seq_idx]
                result_df = label_df.merge(pred_df, right_index=True, left_index=True)
                result_df["test_case_name"] = test_case_name
                result_df["date"] = self.test_dataset[test_case_name]["date"]
                result_df["predict_utc_time"] = predict_utc_times[seq_idx]
                result_df["target_parameter"] = self.output_parameter_names[0]
                result_df["time_step"] = seq_idx
                expect_result_df = pd.concat([expect_result_df, result_df], axis=0)

            # create expect_metrics_df
            start_utc_time = self.test_dataset[test_case_name]["start"]  # this ends with .csv
            _timestep_csv_names = timestep_csv_names(time_step_minutes=10)
            start_utc_time_idx = _timestep_csv_names.index(start_utc_time)
            predict_end_utc_time_idx = start_utc_time_idx + 6
            predict_utc_times = _timestep_csv_names[start_utc_time_idx:predict_end_utc_time_idx]
            if predict_end_utc_time_idx > len(_timestep_csv_names) - 1:
                for i in range(predict_end_utc_time_idx - len(_timestep_csv_names)):
                    predict_utc_times.append(_timestep_csv_names[i])
            predict_utc_times = [i.replace(".csv", "") for i in predict_utc_times]
            metrics_df = pd.DataFrame(
                {
                    "test_case_name": [test_case_name] * label_seq_length,
                    "predict_utc_time": predict_utc_times,
                    "target_parameter": ["rain"] * label_seq_length,
                    "r2": [1.0] * label_seq_length,
                    "rmse": [0.0] * label_seq_length,
                }
            )
            expect_metrics_df = pd.concat([expect_metrics_df, metrics_df], axis=0)

        expect_metrics_df["r2"] = expect_metrics_df["r2"].astype(np.float64)
        expect_metrics_df["rmse"] = expect_metrics_df["rmse"].astype(np.float64)
        expect_metrics_df.index = pd.Index([0] * len(expect_metrics_df))

        self.assertTrue(self.combine_models_evaluator.results_df.equals(expect_result_df))
        self.assertTrue(self.combine_models_evaluator.metrics_df.equals(expect_metrics_df))

        self.assertTrue(os.path.exists(os.path.join(self.downstream_directory, self.model_name, "combine_models_evaluation", "timeseries_rmse_plot.png")))
        self.assertTrue(os.path.exists(os.path.join(self.downstream_directory, self.model_name, "combine_models_evaluation", "timeseries_r2_score_plot.png")))

    @ignore_warnings(category=ConvergenceWarning)
    def test_evaluate_test_case(self):
        self._test_evaluate_test_case()
        self._test_evaluate_test_case(save_attention_maps=True)

    def _test_evaluate_test_case(self, save_attention_maps: bool = False):
        self._generate_dummy_pred_files(is_grid_data=False)
        test_case_name = "sample1"
        self.model.return_value = torch.zeros((1, len(self.output_parameter_names), self.combine_models_evaluator.hydra_cfg.label_seq_length, 50, 50)).to(
            DEVICE
        )
        self.combine_models_evaluator.clean_dfs()
        self.combine_models_evaluator.hydra_cfg.evaluate.save_attention_maps = save_attention_maps
        if save_attention_maps:
            self.model.get_attention_maps.return_value = {"convlstm": torch.zeros((1))}

        self.combine_models_evaluator.evaluate_test_case(test_case_name)

        with open(self.observation_point_file_path, "r") as f:
            ob_point_data = json.load(f)

        pred_df = pd.DataFrame(index=list(ob_point_data.keys()))
        pred_df["Pred_Value"] = 0.0  # NOTE: Pred_Value is rain and scaled to orignal scale.
        pred_df["Pred_Value"] = pred_df["Pred_Value"].astype(np.float32)
        label_seq_length = self.combine_models_evaluator.hydra_cfg.label_seq_length

        expect_result_df = pd.DataFrame()

        start_utc_time = self.test_dataset[test_case_name]["start"]  # this ends with .csv
        _timestep_csv_names = timestep_csv_names(time_step_minutes=10)
        start_utc_time_idx = _timestep_csv_names.index(start_utc_time)
        predict_end_utc_time_idx = start_utc_time_idx + 6
        predict_utc_times = _timestep_csv_names[start_utc_time_idx:predict_end_utc_time_idx]
        if predict_end_utc_time_idx > len(_timestep_csv_names) - 1:
            for i in range(predict_end_utc_time_idx - len(_timestep_csv_names)):
                predict_utc_times.append(_timestep_csv_names[i])
        predict_utc_times = [i.replace(".csv", "") for i in predict_utc_times]

        for seq_idx in range(label_seq_length):
            label_df = self.test_dataset[test_case_name]["label_df"][seq_idx]
            result_df = label_df.merge(pred_df, right_index=True, left_index=True)
            result_df["test_case_name"] = test_case_name
            result_df["date"] = self.test_dataset[test_case_name]["date"]
            result_df["predict_utc_time"] = predict_utc_times[seq_idx]
            result_df["target_parameter"] = self.output_parameter_names[0]
            result_df["time_step"] = seq_idx
            expect_result_df = pd.concat([expect_result_df, result_df], axis=0)
        self.assertTrue(self.combine_models_evaluator.results_df.equals(expect_result_df))

        start_utc_time = self.test_dataset[test_case_name]["start"]  # this ends with .csv
        _timestep_csv_names = timestep_csv_names(time_step_minutes=10)
        start_utc_time_idx = _timestep_csv_names.index(start_utc_time)
        predict_end_utc_time_idx = start_utc_time_idx + 6
        predict_utc_times = _timestep_csv_names[start_utc_time_idx:predict_end_utc_time_idx]
        if predict_end_utc_time_idx > len(_timestep_csv_names) - 1:
            for i in range(predict_end_utc_time_idx - len(_timestep_csv_names)):
                predict_utc_times.append(_timestep_csv_names[i])
        predict_utc_times = [i.replace(".csv", "") for i in predict_utc_times]
        expect_metrics_df = pd.DataFrame(
            {
                "test_case_name": [test_case_name] * label_seq_length,
                "predict_utc_time": predict_utc_times,
                "target_parameter": ["rain"] * label_seq_length,
                "r2": [1.0] * label_seq_length,
                "rmse": [0.0] * label_seq_length,
            }
        )
        expect_metrics_df["r2"] = expect_metrics_df["r2"].astype(np.float64)
        expect_metrics_df["rmse"] = expect_metrics_df["rmse"].astype(np.float64)
        expect_metrics_df.index = pd.Index([0] * len(expect_metrics_df))
        self.assertTrue(self.combine_models_evaluator.metrics_df.equals(expect_metrics_df))

        expect_save_dir_path = os.path.join(self.downstream_directory, self.model_name, "combine_models_evaluation", test_case_name)
        for predict_utc_time in predict_utc_times:
            filename = predict_utc_time + ".parquet.gzip"
            with self.subTest(test_case_name=test_case_name, predict_utc_time=predict_utc_time):
                self.assertTrue(os.path.exists(os.path.join(expect_save_dir_path, filename)))

        for time_step in range(label_seq_length):
            self.assertTrue(os.path.join(expect_save_dir_path, f"attention_maps_timestep{time_step}.pt"))

    @ignore_warnings(category=ConvergenceWarning)
    def test_load_sub_models_predict_tensor(self):
        test_case_name = "sample1"

        expected_tensor = torch.zeros(
            (1, len(self.input_parameter_names), self.combine_models_evaluator.hydra_cfg.label_seq_length, GridSize.HEIGHT, GridSize.WIDTH), dtype=torch.float
        ).to(DEVICE)
        for param_idx in range(len(self.input_parameter_names)):
            # NOTE: rain is 0
            if param_idx != 0:
                expected_tensor[:, param_idx, ...] = 1 / 2 ** param_idx

        with self.subTest(prediction_data_type="grid"):
            self._generate_dummy_pred_files(is_grid_data=True)

            sub_models_predict_tensor = self.combine_models_evaluator.load_sub_models_predict_tensor(test_case_name)
            self.assertTrue(sub_models_predict_tensor.equal(expected_tensor))

        with self.subTest(prediction_data_type="ob_point"):
            self._generate_dummy_pred_files(is_grid_data=False)

            sub_models_predict_tensor = self.combine_models_evaluator.load_sub_models_predict_tensor(test_case_name)
            self.assertTrue(sub_models_predict_tensor.equal(expected_tensor))

    def _generate_dummy_pred_files(self, is_grid_data: bool = False):
        """This function generate dummy prediction data files (.parquet.gzip or .csv).

        1. This prediction files are saved after NormalEvaluator runs except for rain.
        2. Mock these dummy data files here saving to ./tmp dir. Return None (These files are loaded in load_sub_models_predict_tensor.).

        """
        for test_case_name in self.test_dataset.keys():
            for param_dim, param_name in enumerate(self.input_parameter_names):
                if param_name == "rain":
                    continue

                save_dir_path = os.path.join(self.downstream_directory, param_name, "normal_evaluation", test_case_name)
                os.makedirs(save_dir_path, exist_ok=True)
                for time_step in range(self.combine_models_evaluator.hydra_cfg.label_seq_length):
                    save_file_path = os.path.join(
                        save_dir_path, f"{self.combine_models_evaluator.get_prediction_utc_time(test_case_name, time_step)}.parquet.gzip"
                    )
                    if is_grid_data:
                        dummy_pred_ndarray = np.zeros((50, 50), dtype=np.float32)
                    else:
                        dummy_pred_ndarray = np.zeros((35,), dtype=np.float32)

                    # NOTE: rain -> 0.0, temperature(1) -> 0.5, humidity = 0.25
                    dummy_pred_ndarray[...] = 1 / 2 ** param_dim
                    min_val, max_val = MinMaxScalingValue.get_minmax_values_by_weather_param(param_name)
                    dummy_pred_ndarray = (max_val - min_val) * dummy_pred_ndarray + min_val
                    save_parquet(dummy_pred_ndarray, save_file_path, self.observation_point_file_path)
