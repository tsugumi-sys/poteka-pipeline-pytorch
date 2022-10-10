import unittest
from unittest.mock import MagicMock
import os
import shutil
import json
from typing import Dict
import itertools

import hydra
from hydra import initialize
import torch
import pandas as pd
import numpy as np
from common.config import GridSize, ScalingMethod
from common.interpolate_by_gpr import interpolate_by_gpr

from common.utils import param_date_path, timestep_csv_names
from evaluate.src.sequential_evaluator import SequentialEvaluator
from evaluate.src.utils import normalize_tensor
from tests.evaluate.utils import generate_dummy_test_dataset
from train.src.config import DEVICE


class TestSequentialEvaluator(unittest.TestCase):
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
        self.sequential_evaluator = SequentialEvaluator(
            self.model,
            self.model_name,
            self.test_dataset,
            self.input_parameter_names,
            self.output_parameter_names,
            self.downstream_directory,
            self.observation_point_file_path,
        )
        self.sequential_evaluator.hydra_cfg.use_dummy_data = True
        return super().setUp()

    def tearDown(self) -> None:
        hydra.core.global_hydra.GlobalHydra.instance().clear()  # type:ignore
        return super().tearDown()

    def test_runs(self):
        evaluate_types = ["reuse_predict", "update_inputs"]
        model_return_values = [
            torch.zeros((1, len(self.output_parameter_names), 1, GridSize.WIDTH, GridSize.HEIGHT)),
            torch.zeros((1, len(self.output_parameter_names), 1, 35)),
        ]
        test_cases = itertools.product(evaluate_types, model_return_values)

        for (e_type, model_return_val) in test_cases:
            with self.subTest(model_return_value_shape=model_return_val.shape, evaluate_type=e_type):
                is_ob_point_label = True if model_return_val.ndim == 4 else False
                self.sequential_evaluator.test_dataset = generate_dummy_test_dataset(
                    self.input_parameter_names,
                    self.observation_point_file_path,
                    self.sequential_evaluator.hydra_cfg.input_seq_length,
                    self.sequential_evaluator.hydra_cfg.label_seq_length,
                    is_ob_point_label,
                )

                self._test_run(model_return_val, e_type)

                # NOTE: Initialize results_df and metrics_df for next run.
                self.sequential_evaluator.results_df = pd.DataFrame()
                self.sequential_evaluator.metrics_df = pd.DataFrame()

    def _test_run(self, model_return_value: torch.Tensor, evaluate_type: str):
        self.model.return_value = model_return_value
        self.model.evaluate_type = evaluate_type
        results = self.sequential_evaluator.run()

        self.assertTrue(results["r2"] == 1.0)
        self.assertTrue(results["rmse"] == 0.0)

        with open(self.observation_point_file_path, "r") as f:
            ob_point_data = json.load(f)

        pred_df = pd.DataFrame(index=list(ob_point_data.keys()))
        pred_df["Pred_Value"] = 0.0  # NOTE: Pred_Value is rain and scaled to orignal scale.
        pred_df["Pred_Value"] = pred_df["Pred_Value"].astype(np.float32)
        label_seq_length = self.sequential_evaluator.hydra_cfg.label_seq_length

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

        self.assertTrue(self.sequential_evaluator.results_df.equals(expect_result_df))
        self.assertTrue(self.sequential_evaluator.metrics_df.equals(expect_metrics_df))

    def test_evaluate_test_case_reuse_predict(self):
        self._test_evaluate_test_case(evaluate_type="reuse_predict")

    def test_evaluate_test_case_update_inputs(self):
        self._test_evaluate_test_case(evaluate_type="update_inputs")

    def _test_evaluate_test_case(self, evaluate_type: str):
        self.sequential_evaluator.evaluate_type = evaluate_type

        test_case_name = "sample1"
        self.model.return_value = torch.zeros((1, len(self.output_parameter_names), self.sequential_evaluator.hydra_cfg.label_seq_length, 50, 50))
        self.sequential_evaluator.evaluate_test_case(test_case_name)

        with open(self.observation_point_file_path, "r") as f:
            ob_point_data = json.load(f)

        pred_df = pd.DataFrame(index=list(ob_point_data.keys()))
        pred_df["Pred_Value"] = 0.0  # NOTE: Pred_Value is rain and scaled to orignal scale.
        pred_df["Pred_Value"] = pred_df["Pred_Value"].astype(np.float32)
        label_seq_length = self.sequential_evaluator.hydra_cfg.label_seq_length

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
            expect_result_df = pd.concat([expect_result_df, result_df], axis=0)
        self.assertTrue(self.sequential_evaluator.results_df.equals(expect_result_df))

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
        self.assertTrue(self.sequential_evaluator.metrics_df.equals(expect_metrics_df))

        for predict_utc_time in predict_utc_times:
            filename = predict_utc_time.replace(".csv", ".parquet.gzip")
            with self.subTest(test_case_name=test_case_name, predict_utc_time=predict_utc_time):
                self.assertTrue(os.path.join(self.downstream_directory, filename))

    def test_update_input_tensor(self):
        scaling_methods = ScalingMethod.get_methods()
        before_input_tensors = [
            torch.zeros(1, len(self.input_parameter_names), self.sequential_evaluator.hydra_cfg.input_seq_length, GridSize.WIDTH, GridSize.HEIGHT),
        ]
        next_frame_tensors = [torch.rand((len(self.input_parameter_names), GridSize.WIDTH, GridSize.HEIGHT)), torch.rand((len(self.input_parameter_names), 35))]
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
        self.sequential_evaluator.hydra_cfg.scaling_method = scaling_method
        updated_tensor, standarized_info = self.sequential_evaluator._update_input_tensor(before_input_tensor, before_standarized_info, next_frame_tensor)

        if next_frame_tensor.ndim == 2:
            _next_frame_tensor = next_frame_tensor.cpu().detach().numpy().copy()
            next_frame_tensor = torch.zeros(next_frame_tensor.size()[0], GridSize.WIDTH, GridSize.HEIGHT)
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
