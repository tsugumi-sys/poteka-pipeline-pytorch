import unittest
from unittest.mock import MagicMock
import os
import shutil
import json

import hydra
from hydra import initialize
import torch
import pandas as pd
import numpy as np

from common.utils import timestep_csv_names
from evaluate.src.normal_evaluator import NormalEvaluator
from tests.evaluate.utils import generate_dummy_test_dataset
from train.src.config import DEVICE


class TestNormalEvaluator(unittest.TestCase):
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
        self.normal_evaluator = NormalEvaluator(
            self.model,
            self.model_name,
            self.test_dataset,
            self.input_parameter_names,
            self.output_parameter_names,
            self.downstream_directory,
            self.observation_point_file_path,
        )
        self.normal_evaluator.hydra_cfg.use_dummy_data = True
        return super().setUp()

    def tearDown(self) -> None:
        hydra.core.global_hydra.GlobalHydra.instance().clear()  # type:ignore
        return super().tearDown()

    def test_run_grid_return(self):
        self._test_run(torch.zeros((1, len(self.output_parameter_names), self.normal_evaluator.hydra_cfg.label_seq_length, 50, 50)).to(DEVICE))

    def test_run_obpoint_return(self):
        self._test_run(torch.zeros((1, len(self.output_parameter_names), self.normal_evaluator.hydra_cfg.label_seq_length, 35)).to(DEVICE))

    def _test_run(self, model_return_value: torch.Tensor):
        self.model.return_value = model_return_value
        results = self.normal_evaluator.run()

        self.assertTrue(results[f"{self.model_name}_normal_r2"] == 1.0)
        self.assertTrue(results[f"{self.model_name}_normal_rmse"] == 0.0)

        with open(self.observation_point_file_path, "r") as f:
            ob_point_data = json.load(f)

        pred_df = pd.DataFrame(index=list(ob_point_data.keys()))
        pred_df["Pred_Value"] = 0.0  # NOTE: Pred_Value is rain and scaled to orignal scale.
        pred_df["Pred_Value"] = pred_df["Pred_Value"].astype(np.float32)
        label_seq_length = self.normal_evaluator.hydra_cfg.label_seq_length

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

        self.assertTrue(self.normal_evaluator.results_df.equals(expect_result_df))
        self.assertTrue(self.normal_evaluator.metrics_df.equals(expect_metrics_df))

    def test_evaluate_test_case(self):
        test_case_name = "sample1"
        self.model.return_value = torch.zeros((1, len(self.output_parameter_names), self.normal_evaluator.hydra_cfg.label_seq_length, 50, 50)).to(DEVICE)
        self.normal_evaluator.evaluate_test_case(test_case_name)

        with open(self.observation_point_file_path, "r") as f:
            ob_point_data = json.load(f)

        pred_df = pd.DataFrame(index=list(ob_point_data.keys()))
        pred_df["Pred_Value"] = 0.0  # NOTE: Pred_Value is rain and scaled to orignal scale.
        pred_df["Pred_Value"] = pred_df["Pred_Value"].astype(np.float32)
        label_seq_length = self.normal_evaluator.hydra_cfg.label_seq_length

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
        self.assertTrue(self.normal_evaluator.results_df.equals(expect_result_df))

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
        self.assertTrue(self.normal_evaluator.metrics_df.equals(expect_metrics_df))

        expect_save_dir_path = os.path.join(self.downstream_directory, self.model_name, "normal_evaluation", test_case_name)
        for predict_utc_time in predict_utc_times:
            filename = predict_utc_time + ".parquet.gzip"
            with self.subTest(test_case_name=test_case_name, predict_utc_time=predict_utc_time):
                self.assertTrue(os.path.exists(os.path.join(expect_save_dir_path, filename)))
