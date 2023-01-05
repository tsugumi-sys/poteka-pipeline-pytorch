import unittest
from unittest.mock import MagicMock
from typing import Optional
import os
import shutil
import json
import itertools

from torch import nn
import hydra
from hydra import compose, initialize
import torch
import pandas as pd
import numpy as np
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from common.config import GridSize
from common.utils import timestep_csv_names
from evaluate.src.sequential_evaluator import SequentialEvaluator
from tests.evaluate.utils import generate_dummy_test_dataset
from common.config import DEVICE
from train.src.models.convlstm.seq2seq import Seq2Seq


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
        self.evaluate_type = "reuse_predict"

    def setUp(self) -> None:
        if os.path.exists(self.downstream_directory):
            shutil.rmtree(self.downstream_directory)
        os.makedirs(self.downstream_directory, exist_ok=True)

        initialize(config_path="../../conf", version_base=None)
        hydra_cfg = compose(config_name="config")
        self.sequential_evaluator = SequentialEvaluator(
            self.model,
            self.model_name,
            self.test_dataset,
            self.input_parameter_names,
            self.output_parameter_names,
            self.downstream_directory,
            self.observation_point_file_path,
            hydra_cfg=hydra_cfg,
            evaluate_type=self.evaluate_type,
        )
        self.sequential_evaluator.hydra_cfg.use_dummy_data = True
        return super().setUp()

    def tearDown(self) -> None:
        shutil.rmtree(self.downstream_directory)
        hydra.core.global_hydra.GlobalHydra.instance().clear()  # type:ignore
        return super().tearDown()

    @ignore_warnings(category=ConvergenceWarning)
    def test_runs(self):
        evaluate_types = ["reuse_predict", "update_inputs"]
        model_return_values = [
            torch.zeros((1, len(self.output_parameter_names), 1, GridSize.WIDTH, GridSize.HEIGHT)).to(DEVICE),
            torch.zeros((1, len(self.output_parameter_names), 1, 35)).to(DEVICE),
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
        self.evaluate_type = evaluate_type
        self.sequential_evaluator.evaluate_type = evaluate_type
        # NOTE: save_attention_maps tests in `evalaute_test_case` method.
        self.sequential_evaluator.hydra_cfg.evaluate.save_attention_maps = False
        results = self.sequential_evaluator.run()

        self.assertTrue(results[f"{self.model_name}_sequential_{self.evaluate_type}_r2"] == 1.0)
        self.assertTrue(results[f"{self.model_name}_sequential_{self.evaluate_type}_rmse"] == 0.0)

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
                result_df["time_step"] = seq_idx
                result_df["case_type"] = "not_tc"
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

        self.assertTrue(
            os.path.exists(
                os.path.join(
                    self.downstream_directory,
                    self.model_name,
                    "sequential_evaluation",
                    evaluate_type,
                    "timeseries_rmse_plot.png",
                )
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(
                    self.downstream_directory,
                    self.model_name,
                    "sequential_evaluation",
                    evaluate_type,
                    "timeseries_r2_score_plot.png",
                )
            )
        )

    @ignore_warnings(category=ConvergenceWarning)
    def test_evaluate_test_case_bad(self):
        test_cases = [(True, Seq2Seq(len(self.input_parameter_names), 3, 3, "same", "relu", (50, 50), 2, 6).to(DEVICE))]
        for test_case in test_cases:
            with self.assertRaises(ValueError):
                self._test_evaluate_test_case(
                    evaluate_type="reuse_predict", save_attention_maps=test_case[0], model=test_case[1]
                )
                self._test_evaluate_test_case(
                    evaluate_type="update_inputs", save_attention_maps=test_case[0], model=test_case[1]
                )

    @ignore_warnings(category=ConvergenceWarning)
    def test_evaluate_test_case_good(self):
        self._test_evaluate_test_case("reuse_predict")
        self._test_evaluate_test_case("update_inputs")
        self._test_evaluate_test_case("reuse_predict", save_attention_maps=True)

    def _test_evaluate_test_case(
        self, evaluate_type: str, save_attention_maps: bool = False, model: Optional[nn.Module] = None
    ):
        self.evaluate_type = evaluate_type
        self.sequential_evaluator.clean_dfs()
        self.sequential_evaluator.evaluate_type = evaluate_type
        self.sequential_evaluator.hydra_cfg.evaluate.save_attention_maps = save_attention_maps
        if save_attention_maps:
            self.model.get_attention_maps.return_value = {"convlstm": torch.zeros((1, 6, 50 * 50))}
        if model is not None:
            self.sequential_evaluator.model = model

        test_case_name = "sample1"
        self.model.return_value = torch.zeros(
            (1, len(self.output_parameter_names), self.sequential_evaluator.hydra_cfg.label_seq_length, 50, 50)
        ).to(DEVICE)
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
            result_df["time_step"] = seq_idx
            result_df["case_type"] = "not_tc"
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

        expect_save_dir_path = os.path.join(
            self.downstream_directory, self.model_name, "sequential_evaluation", self.evaluate_type, test_case_name
        )
        for predict_utc_time in predict_utc_times:
            filename = predict_utc_time + ".parquet.gzip"
            with self.subTest(test_case_name=test_case_name, predict_utc_time=predict_utc_time):
                self.assertTrue(os.path.exists(os.path.join(expect_save_dir_path, filename)))

        if save_attention_maps:
            # NOTE: if cartopy not installed, generating geo image is skipped. So only check directory exits.
            self.assertTrue(os.path.exists(os.path.join(expect_save_dir_path, "attention_maps", "convlstm")))
