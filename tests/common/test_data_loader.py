import unittest
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np
import torch
from common.config import GridSize, PPOTEKACols

from common.data_loader import train_data_loader, test_data_loader


class TestDataLoader(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        # meta_data.json has two pairs of input and label (rain, temperature, humidity, u_wind, v_wind).
        self.input_parameters = ["rain", "temperature", "humidity", "u_wind", "v_wind"]
        self.input_seq_length = 6
        self.label_seq_length = 6
        self.meta_train_file_path = "./tests/common/meta_train.json"
        self.meta_test_file_path = "./tests/common/meta_test.json"
        # valid tensor
        # NOTE: tensor is still cpu in train_test_loader
        self.tensor_multiplyer = {"rain": 1, "temperature": 2, "humidity": 3, "wind": 4}
        self.rain_tensor = torch.ones((GridSize.HEIGHT, GridSize.WIDTH), dtype=torch.float) * self.tensor_multiplyer["rain"]
        self.temperature_tensor = torch.ones((GridSize.HEIGHT, GridSize.WIDTH), dtype=torch.float) * self.tensor_multiplyer["temperature"]
        self.humidity_tensor = torch.ones((GridSize.HEIGHT, GridSize.WIDTH), dtype=torch.float) * self.tensor_multiplyer["humidity"]
        self.wind_tensor = torch.ones((GridSize.HEIGHT, GridSize.WIDTH), dtype=torch.float) * self.tensor_multiplyer["wind"]

    @patch("torch.std")
    @patch("torch.mean")
    @patch("common.data_loader.load_scaled_data")
    def test_train_data_loader(self, mock_load_scaled_data: MagicMock, mock_torch_mean: MagicMock, mock_torch_std: MagicMock):
        mock_load_scaled_data.side_effect = self.__side_effect_load_scaled_data
        mock_torch_mean.return_value = 0
        mock_torch_std.return_value = 1.0
        with open(self.meta_train_file_path, "r") as f:
            meta_data = json.load(f)
        data_file_paths = meta_data["file_paths"]
        input_tensor, label_tensor = train_data_loader(path=self.meta_train_file_path, scaling_method="min_max_standard")
        for batch_idx in range(input_tensor.size(0)):
            for param_dim, param_name in enumerate(self.input_parameters):
                torch_stdmean_arg_tensor = torch.reshape(self.__get_tensor_by_param(param_name), (1, GridSize.HEIGHT, GridSize.WIDTH))
                torch_stdmean_arg_tensor = torch.cat(tuple(torch_stdmean_arg_tensor for _ in range(self.input_seq_length)), dim=0)
                with self.subTest(f"Tests of torch.std .mean at batch_idx: {batch_idx}, param_name: {param_name}"):
                    call_count = batch_idx * len(self.input_parameters) + param_dim
                    self.assertTrue(torch.equal(mock_torch_std.call_args_list[call_count].args[0], torch_stdmean_arg_tensor))
                    self.assertTrue(torch.equal(mock_torch_mean.call_args_list[call_count].args[0], torch_stdmean_arg_tensor))
                with self.subTest("Tests of input tensor and load_scaled_data args"):
                    for seq_idx in range(self.input_seq_length):
                        # [NOTE]: standarization is performed because scalinf_method is "min_max_standard"
                        standarized_tensor = (self.__get_tensor_by_param(param_name) - mock_torch_mean.return_value) / mock_torch_std.return_value
                        self.assertTrue(torch.equal(input_tensor[batch_idx, param_dim, seq_idx, :, :], standarized_tensor))
                        input_call_count = (
                            batch_idx * (len(self.input_parameters) * self.input_seq_length + len(self.input_parameters) * self.label_seq_length)
                            + param_dim * (self.input_seq_length + self.label_seq_length)  # noqa: W503
                            + seq_idx  # noqa: W503
                        )
                        self.assertEqual(
                            mock_load_scaled_data.call_args_list[input_call_count].args[0], data_file_paths[batch_idx][param_name]["input"][seq_idx]
                        )
        for batch_idx in range(label_tensor.size(0)):
            for param_dim, param_name in enumerate(self.input_parameters):
                with self.subTest("Tests of label tensor and laod_scaled_data args."):
                    for seq_idx in range(self.label_seq_length):
                        # [NOTE]: label tensor is scaled to [0, 1]
                        self.assertTrue(torch.equal(label_tensor[batch_idx, param_dim, seq_idx, :, :], self.__get_tensor_by_param(param_name)))
                        label_call_count = (
                            batch_idx * (len(self.input_parameters) * self.input_seq_length + len(self.input_parameters) * self.label_seq_length)
                            + param_dim * (self.input_seq_length + self.label_seq_length)  # noqa: W503
                            + seq_idx  # noqa: W503
                            + self.input_seq_length  # noqa: W503
                        )
                        self.assertEqual(
                            mock_load_scaled_data.call_args_list[label_call_count].args[0], data_file_paths[batch_idx][param_name]["label"][seq_idx]
                        )

    @patch("torch.std")
    @patch("torch.mean")
    @patch("common.data_loader.load_scaled_data")
    def test_test_data_loader(self, mock_load_scaled_data: MagicMock, mock_torch_mean: MagicMock, mock_torch_std: MagicMock):
        mock_load_scaled_data.side_effect = self.__side_effect_load_scaled_data
        mock_torch_mean.return_value = 0
        mock_torch_std.return_value = 1.0
        with open(self.meta_test_file_path, "r") as f:
            meta_data = json.load(f)
        data_file_paths = meta_data["file_paths"]
        output_data, features_dict = test_data_loader(path=self.meta_test_file_path, scaling_method="min_max_standard", use_dummy_data=True)
        self.assertEqual(features_dict, dict((idx, param_name) for idx, param_name in enumerate(self.input_parameters)))
        for sample_idx, sample_name in enumerate(list(data_file_paths.keys())):
            input_tensor = output_data[sample_name]["input"]
            label_tensor = output_data[sample_name]["label"]
            date = output_data[sample_name]["date"]
            label_dfs = output_data[sample_name]["label_df"]
            standarize_info = output_data[sample_name]["standarize_info"]
            # test meta info
            self.assertEqual(date, data_file_paths[sample_name]["date"])
            for df in label_dfs.values():
                self.assertIsInstance(df, pd.DataFrame)
                self.assertEqual(df.columns.tolist(), PPOTEKACols.get_cols())
            self.assertEqual(
                standarize_info, dict((param, {"mean": mock_torch_mean.return_value, "std": mock_torch_std.return_value}) for param in self.input_parameters)
            )
            for param_dim, param_name in enumerate(self.input_parameters):
                # test load input data
                torch_stdmean_arg_tensor = torch.reshape(self.__get_tensor_by_param(param_name), (1, GridSize.HEIGHT, GridSize.WIDTH))
                torch_stdmean_arg_tensor = torch.cat(tuple(torch_stdmean_arg_tensor for _ in range(self.input_seq_length)), dim=0)
                with self.subTest(f"Tests of torch.std .mean at batch_idx: {sample_idx}, param_name: {param_name}"):
                    call_count = sample_idx * len(self.input_parameters) + param_dim
                    self.assertTrue(torch.equal(mock_torch_std.call_args_list[call_count].args[0], torch_stdmean_arg_tensor))
                    self.assertTrue(torch.equal(mock_torch_mean.call_args_list[call_count].args[0], torch_stdmean_arg_tensor))
                with self.subTest("Tests of input tensor and load_scaled_data args"):
                    for seq_idx in range(self.input_seq_length):
                        # [NOTE]: standarization is performed because scalinf_method is "min_max_standard"
                        standarized_tensor = (self.__get_tensor_by_param(param_name) - mock_torch_mean.return_value) / mock_torch_std.return_value
                        self.assertTrue(torch.equal(input_tensor[0, param_dim, seq_idx, :, :], standarized_tensor))
                        input_call_count = (
                            sample_idx * (len(self.input_parameters) * self.input_seq_length + len(self.input_parameters) * self.label_seq_length)
                            + param_dim * (self.input_seq_length + self.label_seq_length)  # noqa: W503
                            + seq_idx  # noqa: W503
                        )
                        self.assertEqual(
                            mock_load_scaled_data.call_args_list[input_call_count].args[0], data_file_paths[sample_name][param_name]["input"][seq_idx]
                        )
                # test load test data
                with self.subTest("Tests of label tensor and laod_scaled_data args."):
                    for seq_idx in range(self.label_seq_length):
                        # [NOTE]: label tensor is scaled to [0, 1]
                        self.assertTrue(torch.equal(label_tensor[0, param_dim, seq_idx, :, :], self.__get_tensor_by_param(param_name)))
                        label_call_count = (
                            sample_idx * (len(self.input_parameters) * self.input_seq_length + len(self.input_parameters) * self.label_seq_length)
                            + param_dim * (self.input_seq_length + self.label_seq_length)  # noqa: W503
                            + seq_idx  # noqa: W503
                            + self.input_seq_length  # noqa: W503
                        )
                        self.assertEqual(
                            mock_load_scaled_data.call_args_list[label_call_count].args[0], data_file_paths[sample_name][param_name]["label"][seq_idx]
                        )

    def __side_effect_load_scaled_data(self, meta_train_file_path: str) -> np.ndarray:
        if "rain" in meta_train_file_path:
            return np.ones((GridSize.HEIGHT, GridSize.WIDTH), dtype=np.float32) * self.tensor_multiplyer["rain"]
        elif "temp" in meta_train_file_path:
            return np.ones((GridSize.HEIGHT, GridSize.WIDTH), dtype=np.float32) * self.tensor_multiplyer["temperature"]
        elif "humid" in meta_train_file_path:
            return np.ones((GridSize.HEIGHT, GridSize.WIDTH), dtype=np.float32) * self.tensor_multiplyer["humidity"]
        elif "wind" in meta_train_file_path:
            return np.ones((GridSize.HEIGHT, GridSize.WIDTH), dtype=np.float32) * self.tensor_multiplyer["wind"]

    def __get_tensor_by_param(self, param: str) -> torch.Tensor:
        if "rain" in param:
            return self.rain_tensor
        elif "temp" in param:
            return self.temperature_tensor
        elif "humid" in param:
            return self.humidity_tensor
        elif "wind" in param:
            return self.wind_tensor
