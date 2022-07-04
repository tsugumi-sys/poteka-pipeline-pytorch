import unittest
from unittest.mock import MagicMock, patch
from typing import Dict, List
import os

from hydra import initialize
import hydra
from omegaconf import DictConfig
import torch
from common.config import MinMaxScalingValue, PPOTEKACols
import pandas as pd
import numpy as np

from evaluate.src.evaluator import Evaluator
from train.src.config import DEVICE


class TestEvaluator(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.model = MagicMock()  # You need to set different shape torch.Tensor to return_value based on the evaluate_type
        self.model_name = "test_model"
        self.input_parameters = ["rain", "temperature", "humidity"]
        self.output_parameters = ["rain", "temperature", "humidity"]
        self.downstream_directory = "./dummy_downstream_directory"
        # create dummy test dataset
        sample1_input_tensor = torch.ones((5, 3, 6, 50, 50), dtype=torch.float, device=DEVICE)
        sample1_label_tensor = torch.ones((5, 3, 6, 50, 50), dtype=torch.float, device=DEVICE)
        sample2_input_tensor = torch.ones((5, 3, 6, 50, 50), dtype=torch.float, device=DEVICE)
        sample2_label_tensor = torch.ones((5, 3, 6, 50, 50), dtype=torch.float, device=DEVICE)
        # change value for each input parameters (rain -> 0, temperature -> 1, humidity -> 0.5)
        for i in range(len(self.input_parameters)):
            val = 1 / i if i > 0 else 0
            sample1_input_tensor[:, i, :, :, :] = val
            sample1_label_tensor[:, i, :, :, :] = val
            sample2_input_tensor[:, i, :, :, :] = val
            sample2_label_tensor[:, i, :, :, :] = val
        label_dfs = {}
        for i in range(sample1_input_tensor.size()[2]):
            data = {}
            for col in PPOTEKACols.get_cols():
                min_val, max_val = MinMaxScalingValue.get_minmax_values_by_ppoteka_cols(col)
                data[col] = np.ones((10))
                if col == "hour-rain":
                    data[col] *= 0
                elif col == "RH1":
                    data[col] /= 2
            label_dfs[i] = pd.DataFrame(data)
        self.test_dataset = {
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

    def setUp(self) -> None:
        initialize(config_path="../../conf", version_base=None)
        return super().setUp()

    def tearDown(self) -> None:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        return super().tearDown()

    def test_Evaluator__initialize_hydra_conf(self):
        evaluator = Evaluator(self.model, self.model_name, self.test_dataset, self.input_parameters, self.output_parameters, self.downstream_directory)
        self.assertIsInstance(evaluator.hydra_cfg, DictConfig)

    def test_Evaluator__initialize_results_df(self):
        evaluator = Evaluator(self.model, self.model_name, self.test_dataset, self.input_parameters, self.output_parameters, self.downstream_directory)
        self.assertEqual(evaluator.results_df, None)
        evaluator._Evaluator__initialize_results_df()
        self.assertIsInstance(evaluator.results_df, pd.DataFrame)

    @patch("evaluate.src.evaluator.Evaluator._Evaluator__evaluate")
    def test_run(self, mock_Evaluator__evaluate: MagicMock):
        mock_Evaluator__evaluate.side_effect = lambda x: {"evaluate_type": x}
        evaluate_types = ["typeA", "typeB", "typeC"]
        evaluator = Evaluator(self.model, self.model_name, self.test_dataset, self.input_parameters, self.output_parameters, self.downstream_directory)
        results = evaluator.run(evaluate_types)
        self.assertIsInstance(results, Dict)
        for idx, eval_type in enumerate(evaluate_types):
            self.assertTrue(eval_type in results)
            self.assertEqual(results[eval_type], {"evaluate_type": eval_type})
            self.assertEqual(mock_Evaluator__evaluate.call_args_list[idx].args, (eval_type,))
        self.assertEqual(mock_Evaluator__evaluate.call_count, 3)

    @patch("evaluate.src.evaluator.Evaluator._Evaluator__initialize_results_df")
    @patch("evaluate.src.evaluator.Evaluator._Evaluator__eval_test_case")
    @patch("evaluate.src.evaluator.Evaluator._Evaluator__visualize_results")
    def test__evaluate(
        self, mock_Evaluator__visualize_results: MagicMock, mock_Evaluator__eval_test_case: MagicMock, mock_Evaluator_initialize_results_df: MagicMock
    ):
        _result = {"result": 111}
        _evaluate_type = "typeA"
        mock_Evaluator__visualize_results.return_value = _result
        evaluator = Evaluator(self.model, self.model_name, self.test_dataset, self.input_parameters, self.output_parameters, self.downstream_directory)
        result = evaluator._Evaluator__evaluate(evaluate_type=_evaluate_type)
        self.assertEqual(result, _result)
        self.assertEqual(mock_Evaluator_initialize_results_df.call_count, 1)
        self.assertEqual(mock_Evaluator__eval_test_case.call_count, 2)
        for i in range(2):
            self.assertEqual(mock_Evaluator__eval_test_case.call_args_list[i].kwargs, {"test_case_name": f"sample{i+1}", "evaluate_type": _evaluate_type})
        self.assertEqual(mock_Evaluator__visualize_results.call_count, 1)

    @patch("evaluate.src.evaluator.Evaluator._Evaluator__eval_combine_models")
    @patch("evaluate.src.evaluator.Evaluator._Evaluator__eval_successibely")
    @patch("evaluate.src.evaluator.Evaluator._Evaluator__eval_normal")
    @patch("mlflow.log_metric")
    @patch("os.makedirs")
    def test__eval_test_case(
        self,
        mock_os_makedirs: MagicMock,
        mock_mlflow_log_metric: MagicMock,
        mock_evaluator__eval_normal: MagicMock,
        mock_evaluator__eval_successibely: MagicMock,
        mock_evaluator__eval_combine_models: MagicMock,
    ):
        mock__eval_normal_result = {idx: idx * 2 for idx in range(6)}
        mock_evaluator__eval_normal.return_value = mock__eval_normal_result
        mock__eval_successibely_result = {idx: idx * 3 for idx in range(6)}
        mock_evaluator__eval_successibely.return_value = mock__eval_successibely_result
        mock__eval_combine_models_result = {idx: idx * 4 for idx in range(6)}
        mock_evaluator__eval_combine_models.return_value = mock__eval_combine_models_result
        mlflow_log_metric_call_count = 0
        evaluator = Evaluator(self.model, self.model_name, self.test_dataset, self.input_parameters, self.output_parameters, self.downstream_directory)
        # test of not exists evaluate_type
        with self.assertRaises(ValueError):
            evaluator._Evaluator__eval_test_case(test_case_name="sample1", evaluate_type="unko")
        # test evaluate_type normal
        test_case_name = "sample1"
        evaluate_type = "normal"
        evaluator._Evaluator__eval_test_case(test_case_name=test_case_name, evaluate_type=evaluate_type)
        save_dir_path = os.path.join(self.downstream_directory, self.model_name, evaluate_type, test_case_name)
        self.assertEqual(mock_os_makedirs.call_args.args, (save_dir_path,))
        self.assertEqual(
            mock_evaluator__eval_normal.call_args.kwargs,
            {"X_test": self.test_dataset[test_case_name]["input"], "save_results_dir_path": save_dir_path, "test_case_name": test_case_name},
        )
        for i in range(6):
            self.assertEqual(
                mock_mlflow_log_metric.call_args_list[i].kwargs,
                {"key": f"{self.model_name}-{evaluate_type}-{test_case_name}", "value": mock__eval_normal_result[i], "step": i},
            )
        mlflow_log_metric_call_count += 6
        # test evaluate_type is sequential
        test_case_name = "sample1"
        evaluate_type = "sequential"
        evaluator._Evaluator__eval_test_case(test_case_name=test_case_name, evaluate_type=evaluate_type)
        save_dir_path = os.path.join(self.downstream_directory, self.model_name, evaluate_type, test_case_name)
        self.assertEqual(mock_os_makedirs.call_args.args, (save_dir_path,))
        self.assertEqual(
            mock_evaluator__eval_successibely.call_args.kwargs,
            {
                "X_test": self.test_dataset[test_case_name]["input"],
                "y_test": self.test_dataset[test_case_name]["label"],
                "save_results_dir_path": save_dir_path,
                "test_case_name": test_case_name,
                "evaluate_type": evaluate_type,
            },
        )
        for i in range(6):
            self.assertEqual(
                mock_mlflow_log_metric.call_args_list[i + mlflow_log_metric_call_count].kwargs,
                {"key": f"{self.model_name}-{evaluate_type}-{test_case_name}", "value": mock__eval_successibely_result[i], "step": i},
            )
        # test evaluate_type is reuse_predict
        mlflow_log_metric_call_count += 6
        evaluate_type = "reuse_predict"
        evaluator._Evaluator__eval_test_case(test_case_name=test_case_name, evaluate_type=evaluate_type)
        save_dir_path = os.path.join(self.downstream_directory, self.model_name, evaluate_type, test_case_name)
        self.assertEqual(mock_os_makedirs.call_args.args, (save_dir_path,))
        self.assertEqual(
            mock_evaluator__eval_successibely.call_args.kwargs,
            {
                "X_test": self.test_dataset[test_case_name]["input"],
                "y_test": self.test_dataset[test_case_name]["label"],
                "save_results_dir_path": save_dir_path,
                "test_case_name": test_case_name,
                "evaluate_type": evaluate_type,
            },
        )
        for i in range(6):
            self.assertEqual(
                mock_mlflow_log_metric.call_args_list[i + mlflow_log_metric_call_count].kwargs,
                {"key": f"{self.model_name}-{evaluate_type}-{test_case_name}", "value": mock__eval_successibely_result[i], "step": i},
            )
        mlflow_log_metric_call_count += 6
        # test evaluate_type is combine models
        test_case_name = "sample1"
        evaluate_type = "combine_models"
        evaluator._Evaluator__eval_test_case(test_case_name=test_case_name, evaluate_type=evaluate_type)
        save_dir_path = os.path.join(self.downstream_directory, self.model_name, evaluate_type, test_case_name)
        self.assertEqual(mock_os_makedirs.call_args.args, (save_dir_path,))
        self.assertEqual(
            mock_evaluator__eval_combine_models.call_args.kwargs,
            {"X_test": self.test_dataset[test_case_name]["input"], "save_results_dir_path": save_dir_path, "test_case_name": test_case_name},
        )
        for i in range(6):
            self.assertEqual(
                mock_mlflow_log_metric.call_args_list[i + mlflow_log_metric_call_count].kwargs,
                {"key": f"{self.model_name}-{evaluate_type}-{test_case_name}", "value": mock__eval_combine_models_result[i], "step": i},
            )

    @patch("evaluate.src.evaluator.rescale_tensor")
    @patch("evaluate.src.evaluator.Evaluator._Evaluator__calc_rmse")
    @patch("evaluate.src.evaluator.save_parquet")
    @patch("evaluate.src.evaluator.save_rain_image")
    @patch("pandas.DataFrame.to_csv")
    def test__eval_normal(
        self,
        mock_pandas_to_csv: MagicMock,
        mock_save_rain_image: MagicMock,
        mock_save_parquet: MagicMock,
        mock_Evaluator__calc_rmse: MagicMock,
        mock_rescale_tensor: MagicMock,
    ):
        # setting mock
        predict_utc_times = {0: "0-20", 1: "0-30", 2: "0-40", 3: "0-50", 4: "1-0", 5: "1-10"}  # sample1 starts 23-20
        result_df = pd.DataFrame(
            {
                "isSequential": [False],
                "case_type": ["tc"],
                "date": ["xxx"],
                "date_time": ["ddd"],
                "hour-rain": [5.0],
                "Pred_Value": [4.0],
            }
        )
        mock_Evaluator__calc_rmse.return_value = (1.0, result_df)
        test_case_name = "sample1"
        self.model.return_value = self.test_dataset[test_case_name]["label"]
        # case1: use multiple parameters as input and rain as input
        evaluator = Evaluator(self.model, self.model_name, self.test_dataset, self.input_parameters, self.output_parameters, self.downstream_directory)
        evaluator.hydra_cfg.use_dummy_data = False
        evaluator.hydra_cfg.preprocess.time_step_minutes = 10
        result = evaluator._Evaluator__eval_normal(
            X_test=self.test_dataset[test_case_name]["input"], save_results_dir_path=self.downstream_directory, test_case_name=test_case_name
        )
        self.assertEqual(result, {idx: 1.0 for idx in range(6)})
        self.assertEqual(mock_Evaluator__calc_rmse.call_count, 6)
        pd.testing.assert_frame_equal(evaluator.results_df.iloc[:1], result_df)
        for i in range(6):
            # test rescale_tensor call_args
            self.assertEqual(mock_rescale_tensor.call_args_list[i].kwargs["min_value"], MinMaxScalingValue.RAIN_MIN.value)
            self.assertEqual(mock_rescale_tensor.call_args_list[i].kwargs["max_value"], MinMaxScalingValue.RAIN_MAX.value)
            # test __calc_rmse kwargs
            pd.testing.assert_frame_equal(mock_Evaluator__calc_rmse.call_args_list[i].kwargs["label_df"], self.test_dataset[test_case_name]["label_df"][i])
            self.assertEqual(mock_Evaluator__calc_rmse.call_args_list[0].kwargs["test_case_name"], test_case_name)
            self.assertEqual(mock_Evaluator__calc_rmse.call_args_list[i].kwargs["date"], self.test_dataset[test_case_name]["date"])
            self.assertEqual(mock_Evaluator__calc_rmse.call_args_list[i].kwargs["start"], self.test_dataset[test_case_name]["start"].replace(".csv", ""))
            self.assertEqual(mock_Evaluator__calc_rmse.call_args_list[i].kwargs["time_step"], i)
            # test scaled pred array
            # the scaled pred array (rain) must be all 0.
            self.assertEqual(mock_save_rain_image.call_args_list[i].args[1], os.path.join(self.downstream_directory, f"{predict_utc_times[i]}.png"))
            self.assertEqual(
                mock_pandas_to_csv.call_args_list[i].args,
                (os.path.join(self.downstream_directory, f"pred_observ_df_{predict_utc_times[i]}.csv"),),
            )
            self.assertEqual(mock_save_parquet.call_args_list[i].args[1], os.path.join(self.downstream_directory, f"{predict_utc_times[i]}.parquet.gzip"))
        # cse2: use temperature as input and label
        evaluator = Evaluator(self.model, self.model_name, self.test_dataset, ["temperature"], ["temperature"], self.downstream_directory)
        self.model.return_value = self.test_dataset[test_case_name]["label"][:, 1:2, :, :, :]
        result = evaluator._Evaluator__eval_normal(
            X_test=self.test_dataset[test_case_name]["input"][:, 1:2, :, :, :], save_results_dir_path=self.downstream_directory, test_case_name=test_case_name
        )
        for i in range(6):
            # test rescale_tensor call_args
            self.assertEqual(mock_rescale_tensor.call_args_list[i + 6].kwargs["min_value"], MinMaxScalingValue.TEMPERATURE_MIN.value)
            self.assertEqual(mock_rescale_tensor.call_args_list[i + 6].kwargs["max_value"], MinMaxScalingValue.TEMPERATURE_MAX.value)

    @patch("evaluate.src.evaluator.Evaluator._Evaluator__update_input_tensor")
    @patch("evaluate.src.evaluator.Evaluator._Evaluator__calc_rmse")
    @patch("evaluate.src.evaluator.save_parquet")
    @patch("evaluate.src.evaluator.save_rain_image")
    @patch("pandas.DataFrame.to_csv")
    def test__eval_successibely(
        self,
        mock_pandas_to_csv: MagicMock,
        mock_save_rain_image: MagicMock,
        mock_save_parquet: MagicMock,
        mock_Evaluator__calc_rmse: MagicMock,
        mock_Evaluator__update_input_tensor: MagicMock,
    ):
        # setting mock
        predict_utc_times = {0: "0-20", 1: "0-30", 2: "0-40", 3: "0-50", 4: "1-0", 5: "1-10"}  # sample1 starts 23-20
        result_df = pd.DataFrame(
            {
                "isSequential": [False],
                "case_type": ["tc"],
                "date": ["xxx"],
                "date_time": ["ddd"],
                "hour-rain": [5.0],
                "Pred_Value": [4.0],
            }
        )
        mock_Evaluator__calc_rmse.return_value = (1.0, result_df)
        test_case_name = "sample1"
        mock_Evaluator__update_input_tensor.return_value = (self.test_dataset[test_case_name]["input"], self.test_dataset[test_case_name]["standarize_info"])
        self.model.return_value = self.test_dataset[test_case_name]["label"]
        # case: use multiple parameters as input and rain as input
        evaluator = Evaluator(self.model, self.model_name, self.test_dataset, self.input_parameters, self.output_parameters, self.downstream_directory)
        evaluator.hydra_cfg.use_dummy_data = False
        evaluator.hydra_cfg.preprocess.time_step_minutes = 10
        # test when evaluate_type is sequential
        result = evaluator._Evaluator__eval_successibely(
            X_test=self.test_dataset[test_case_name]["input"],
            y_test=self.test_dataset[test_case_name]["label"],
            save_results_dir_path=self.downstream_directory,
            test_case_name=test_case_name,
            evaluate_type="sequential",
        )
        self.assertEqual(result, {idx: 1.0 for idx in range(6)})
        self.assertEqual(mock_Evaluator__calc_rmse.call_count, 6)
        pd.testing.assert_frame_equal(evaluator.results_df.iloc[:1], result_df)
        for i in range(6):
            # test __calc_rmse kwargs
            self.assertEqual(mock_Evaluator__calc_rmse.call_args_list[i].kwargs["pred_ndarray"].sum(), 0)
            pd.testing.assert_frame_equal(mock_Evaluator__calc_rmse.call_args_list[i].kwargs["label_df"], self.test_dataset[test_case_name]["label_df"][i])
            self.assertEqual(mock_Evaluator__calc_rmse.call_args_list[0].kwargs["test_case_name"], test_case_name)
            self.assertEqual(mock_Evaluator__calc_rmse.call_args_list[i].kwargs["date"], self.test_dataset[test_case_name]["date"])
            self.assertEqual(mock_Evaluator__calc_rmse.call_args_list[i].kwargs["start"], self.test_dataset[test_case_name]["start"].replace(".csv", ""))
            self.assertEqual(mock_Evaluator__calc_rmse.call_args_list[i].kwargs["time_step"], i)
            # test scaled pred array
            # the scaled pred array (rain) must be all 0.
            self.assertEqual(mock_save_rain_image.call_args_list[i].args[0].sum(), 0)
            self.assertEqual(mock_save_rain_image.call_args_list[i].args[1], os.path.join(self.downstream_directory, f"{predict_utc_times[i]}.png"))
            self.assertEqual(
                mock_pandas_to_csv.call_args_list[i].args,
                (os.path.join(self.downstream_directory, f"pred_observ_df_{predict_utc_times[i]}.csv"),),
            )
            self.assertEqual(mock_save_parquet.call_args_list[i].args[0].sum(), 0)
            self.assertEqual(mock_save_parquet.call_args_list[i].args[1], os.path.join(self.downstream_directory, f"{predict_utc_times[i]}.parquet.gzip"))
            self.assertTrue(
                torch.equal(mock_Evaluator__update_input_tensor.call_args_list[i].args[2], self.test_dataset[test_case_name]["label"][0, :, i, :, :])
            )
        self.assertTrue(torch.equal(mock_Evaluator__update_input_tensor.call_args_list[0].args[0], self.test_dataset[test_case_name]["input"]))
        self.assertEqual(mock_Evaluator__update_input_tensor.call_args_list[0].args[1], self.test_dataset[test_case_name]["standarize_info"])
        # test when evaluate_type is reuse_predict
        result = evaluator._Evaluator__eval_successibely(
            X_test=self.test_dataset[test_case_name]["input"],
            y_test=self.test_dataset[test_case_name]["label"],
            save_results_dir_path=self.downstream_directory,
            test_case_name=test_case_name,
            evaluate_type="reuse_predict",
        )
        for i in range(6):
            self.assertTrue(torch.equal(mock_Evaluator__update_input_tensor.call_args_list[i + 6].args[2], self.model.return_value[0, :, i, :, :]))
        self.assertTrue(torch.equal(mock_Evaluator__update_input_tensor.call_args_list[6].args[0], self.test_dataset[test_case_name]["input"]))

    @patch("evaluate.src.evaluator.Evaluator._Evaluator__update_input_tensor")
    @patch("evaluate.src.evaluator.Evaluator._Evaluator__calc_rmse")
    @patch("evaluate.src.evaluator.save_parquet")
    @patch("evaluate.src.evaluator.save_rain_image")
    @patch("evaluate.src.evaluator.Evaluator._Evaluator__sort_predict_data_files")
    @patch("pandas.read_parquet")
    @patch("pandas.DataFrame.to_csv")
    def test__eval_combine_models(
        self,
        mock_pandas_to_csv: MagicMock,
        mock_pandas_read_parquet: MagicMock,
        mock_Evaluator__sort_predict_data_files: MagicMock,
        mock_save_rain_image: MagicMock,
        mock_save_parquet: MagicMock,
        mock_Evaluator__calc_rmse: MagicMock,
        mock_Evaluator__update_input_tensor: MagicMock,
    ):
        # setting mock
        predict_utc_times = {0: "0-20", 1: "0-30", 2: "0-40", 3: "0-50", 4: "1-0", 5: "1-10"}  # sample1 starts 23-20
        mock_Evaluator__sort_predict_data_files.side_effect = self.__sort_predict_data_files_side_effect
        mock_pandas_read_parquet.side_effect = self.__read_parquet_side_effect
        result_df = pd.DataFrame(
            {
                "isSequential": [False],
                "case_type": ["tc"],
                "date": ["xxx"],
                "date_time": ["ddd"],
                "hour-rain": [5.0],
                "Pred_Value": [4.0],
            }
        )
        mock_Evaluator__calc_rmse.return_value = (1.0, result_df)
        test_case_name = "sample1"
        mock_Evaluator__update_input_tensor.return_value = (self.test_dataset[test_case_name]["input"], self.test_dataset[test_case_name]["standarize_info"])
        self.model.return_value = self.test_dataset[test_case_name]["label"]
        evaluator = Evaluator(self.model, self.model_name, self.test_dataset, self.input_parameters, self.output_parameters, self.downstream_directory)
        evaluator.hydra_cfg.use_dummy_data = False
        evaluator.hydra_cfg.preprocess.time_step_minutes = 10
        result = evaluator._Evaluator__eval_combine_models(
            X_test=self.test_dataset[test_case_name]["input"], save_results_dir_path=self.downstream_directory, test_case_name=test_case_name
        )
        self.assertEqual(result, {idx: 1.0 for idx in range(6)})
        self.assertEqual(mock_Evaluator__calc_rmse.call_count, 6)
        self.assertEqual(mock_Evaluator__sort_predict_data_files.call_count, 2)
        read_parquet_call_count = 0
        for param_idx, param_name in enumerate(self.input_parameters[1:]):
            result_dir_path = os.path.join(self.downstream_directory, param_name, "normal", test_case_name)
            self.assertEqual(mock_Evaluator__sort_predict_data_files.call_args_list[param_idx].args[0], result_dir_path)
            for i in range(6):
                self.assertEqual(
                    mock_pandas_read_parquet.call_args_list[i + read_parquet_call_count].args[0],
                    os.path.join(result_dir_path, f"{predict_utc_times[i]}.parquet.gzip"),
                )
            read_parquet_call_count += 6
        for i in range(6):
            # test __calc_rmse kwargs
            self.assertEqual(mock_Evaluator__calc_rmse.call_args_list[i].kwargs["pred_ndarray"].sum(), 0)
            pd.testing.assert_frame_equal(mock_Evaluator__calc_rmse.call_args_list[i].kwargs["label_df"], self.test_dataset[test_case_name]["label_df"][i])
            self.assertEqual(mock_Evaluator__calc_rmse.call_args_list[0].kwargs["test_case_name"], test_case_name)
            self.assertEqual(mock_Evaluator__calc_rmse.call_args_list[i].kwargs["date"], self.test_dataset[test_case_name]["date"])
            self.assertEqual(mock_Evaluator__calc_rmse.call_args_list[i].kwargs["start"], self.test_dataset[test_case_name]["start"].replace(".csv", ""))
            self.assertEqual(mock_Evaluator__calc_rmse.call_args_list[i].kwargs["time_step"], i)
            # test scaled pred array
            # the scaled pred array (rain) must be all 0.
            self.assertEqual(mock_save_rain_image.call_args_list[i].args[0].sum(), 0)
            self.assertEqual(mock_save_rain_image.call_args_list[i].args[1], os.path.join(self.downstream_directory, f"{predict_utc_times[i]}.png"))
            self.assertEqual(
                mock_pandas_to_csv.call_args_list[i].args,
                (os.path.join(self.downstream_directory, f"pred_observ_df_{predict_utc_times[i]}.csv"),),
            )
            self.assertEqual(mock_save_parquet.call_args_list[i].args[0].sum(), 0)
            self.assertEqual(mock_save_parquet.call_args_list[i].args[1], os.path.join(self.downstream_directory, f"{predict_utc_times[i]}.parquet.gzip"))
            self.assertTrue(
                torch.equal(mock_Evaluator__update_input_tensor.call_args_list[i].args[2], self.test_dataset[test_case_name]["label"][0, :, i, :, :])
            )
        self.assertTrue(torch.equal(mock_Evaluator__update_input_tensor.call_args_list[0].args[0], self.test_dataset[test_case_name]["input"]))
        self.assertEqual(mock_Evaluator__update_input_tensor.call_args_list[0].args[1], self.test_dataset[test_case_name]["standarize_info"])

    def __sort_predict_data_files_side_effect(self, results_dir_path: str, filename_extention: str) -> List[str]:
        # setting mock
        predict_utc_times = {0: "0-20", 1: "0-30", 2: "0-40", 3: "0-50", 4: "1-0", 5: "1-10"}  # sample1 starts 23-20
        return [os.path.join(results_dir_path, f"{utc_time}{filename_extention}") for utc_time in predict_utc_times.values()]

    def __read_parquet_side_effect(self, path: str) -> pd.DataFrame:
        df = pd.DataFrame(np.ones((50, 50), dtype=np.float32))
        if "rain" in path:
            df *= 0
        elif "temperature" in path:
            df *= 1
        elif "humidity" in path:
            df *= 0.5
        return df

    @patch("torch.mean")
    @patch("torch.std")
    @patch("evaluate.src.evaluator.validate_scaling")
    def test__update_input_tensor(self, mock_evaluate_validate_sclaing: MagicMock, mock_torch_std: MagicMock, mock_torch_mean: MagicMock):
        test_case_name = "sample1"
        mock_torch_mean.return_value = 1.0
        mock_torch_std.return_value = 0.5
        evaluator = Evaluator(self.model, self.model_name, self.test_dataset, self.input_parameters, self.output_parameters, self.downstream_directory)
        before_input_tensor = torch.zeros((1, 3, 6, 50, 50), dtype=torch.float, device=DEVICE)
        next_input_tensor = self.test_dataset[test_case_name]["label"][0, :, 0, :, :]
        before_standarized_info = self.test_dataset[test_case_name]["standarize_info"]
        # test case : scaling method is minmax
        evaluator.hydra_cfg.scaling_method = "minmax"
        output_tensor, standarize_info = evaluator._Evaluator__update_input_tensor(
            before_input_tensor=before_input_tensor.clone().detach(),
            before_standarized_info=before_standarized_info,
            next_input_tensor=next_input_tensor.clone().detach(),
        )
        self.assertEqual(mock_evaluate_validate_sclaing.call_count, 0)
        self.assertEqual(mock_torch_mean.call_count, 0)
        self.assertEqual(mock_torch_std.call_count, 0)
        self.assertEqual(standarize_info, {})
        self.assertEqual(output_tensor.shape, (1, 3, 6, 50, 50))
        self.assertTrue(torch.equal(output_tensor[:, :, :5, :, :], before_input_tensor[:, :, 1:, :, :]))
        self.assertTrue(torch.equal(output_tensor[:, :, 5:, :, :], torch.reshape(next_input_tensor, (1, 3, 1, 50, 50))))
        # test case : scaling method is standard or minmaxstandard
        evaluator.hydra_cfg.scaling_method = "standard"
        output_tensor, standarize_info = evaluator._Evaluator__update_input_tensor(
            before_input_tensor=before_input_tensor.clone().detach(),
            before_standarized_info=before_standarized_info,
            next_input_tensor=next_input_tensor.clone().detach(),
        )
        # generate correct tensor
        for param_dim, param_name in enumerate(self.input_parameters):
            means = before_standarized_info[param_name]["mean"]
            stds = before_standarized_info[param_name]["std"]
            before_input_tensor[:, param_dim, :, :, :] = before_input_tensor[:, param_dim, :, :, :] * stds + means
        updated_tensor = torch.cat((before_input_tensor[:, :, 1:, :, :], torch.reshape(next_input_tensor, (1, 3, 1, 50, 50))), dim=2)
        self.assertEqual(mock_evaluate_validate_sclaing.call_count, 1)
        self.assertEqual(mock_torch_mean.call_count, 3)
        self.assertEqual(mock_torch_std.call_count, 3)
        for param_dim in range(len(self.input_parameters)):
            # assert with after re standard tensor because arguments of mock_torch_mean and mock_torch_std referes updated_input_tensor itself.
            # So, the updated_input_tensor is changable and it changes in re standard process.
            self.assertTrue(
                torch.equal(
                    mock_torch_mean.call_args_list[param_dim].args[0],
                    (updated_tensor[:, param_dim, :, :, :] - mock_torch_mean.return_value) / mock_torch_std.return_value,
                )
            )
            self.assertTrue(
                torch.equal(
                    mock_torch_std.call_args_list[param_dim].args[0],
                    (updated_tensor[:, param_dim, :, :, :] - mock_torch_mean.return_value) / mock_torch_std.return_value,
                )
            )
        return_standarize_info = {}
        for param_dim, param_name in enumerate(self.input_parameters):
            means, stds = mock_torch_mean.return_value, mock_torch_std.return_value
            return_standarize_info[param_name] = {}
            return_standarize_info[param_name]["mean"] = means
            return_standarize_info[param_name]["std"] = stds
            updated_tensor[:, param_dim, :, :, :] = (updated_tensor[:, param_dim, :, :, :] - means) / stds
        self.assertEqual(standarize_info, return_standarize_info)
        self.assertTrue(torch.equal(output_tensor, updated_tensor))
