import unittest
from unittest.mock import MagicMock, patch
from typing import List, Dict

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

    def __generate_dummy_data_file_paths(self, dataset_length: int = 10) -> List:
        data_file_paths = []
        for dataset_idx in range(dataset_length):
            file_paths = {}
            for param_name in self.input_parameters:
                file_paths[param_name] = {"input": [], "label": []}
                for i in range(self.input_seq_length):
                    file_paths[param_name]["input"].append(f"/dataset{dataset_idx}/input{i}.csv")
                for i in range(self.label_seq_length):
                    file_paths[param_name]["label"].append(f"/dataset{dataset_idx}/label{i}.csv")
            data_file_paths.append(file_paths)
        return data_file_paths

    @patch("common.data_loader.store_input_data")
    @patch("common.data_loader.store_label_data")
    @patch("common.data_loader.json_loader")
    def test_train_data_loader(self, mock_json_loader: MagicMock, mock_store_label_data: MagicMock, mock_store_input_data: MagicMock):
        data_file_paths = self.__generate_dummy_data_file_paths(dataset_length=10)
        mock_json_loader.return_value = {"file_paths": data_file_paths}
        mock_store_input_data.return_value = (None, None)
        scaling_method = "min_max_standard"
        input_tensor, label_tensor = train_data_loader(path=self.meta_train_file_path, scaling_method=scaling_method)
        self.assertEqual(
            mock_store_input_data.call_count,
            input_tensor.size(0) * len(self.input_parameters),
        )
        self.assertEqual(
            mock_store_label_data.call_count,
            label_tensor.size(0) * len(self.input_parameters),
        )
        for dataset_idx in range(input_tensor.size(0)):
            for param_idx, param_name in enumerate(self.input_parameters):
                store_input_data_call_args = dict(mock_store_input_data.call_args_list[dataset_idx * len(self.input_parameters) + param_idx].kwargs)
                call_args_input_tensor = store_input_data_call_args.pop("input_tensor")

                store_label_data_call_args = dict(mock_store_label_data.call_args_list[dataset_idx * len(self.input_parameters) + param_idx].kwargs)
                call_args_label_tensor = store_label_data_call_args.pop("label_tensor")
                self.assertEqual(
                    store_input_data_call_args,
                    {
                        "dataset_idx": dataset_idx,
                        "param_idx": param_idx,
                        "input_dataset_paths": data_file_paths[dataset_idx][param_name]["input"],
                        "scaling_method": scaling_method,
                        "inplace": True,
                    },
                )
                self.assertEqual(
                    torch.equal(
                        torch.Tensor(
                            len(data_file_paths),
                            len(self.input_parameters),
                            self.input_seq_length,
                            GridSize.HEIGHT,
                            GridSize.WIDTH,
                        ),
                        call_args_input_tensor,
                    ),
                    True,
                )
                self.assertEqual(
                    store_label_data_call_args,
                    {
                        "dataset_idx": dataset_idx,
                        "param_idx": param_idx,
                        "label_dataset_paths": data_file_paths[dataset_idx][param_name]["label"],
                        "inplace": True,
                    },
                )
                self.assertEqual(
                    torch.equal(
                        torch.Tensor(
                            len(data_file_paths),
                            len(self.input_parameters),
                            self.input_seq_length,
                            GridSize.HEIGHT,
                            GridSize.WIDTH,
                        ),
                        call_args_label_tensor,
                    ),
                    True,
                )

    @patch("common.data_loader.json_loader")
    @patch("common.data_loader.store_input_data")
    @patch("common.data_loader.store_label_data")
    def test_test_data_loader(
        self,
        mock_store_label_data: MagicMock,
        mock_store_input_data: MagicMock,
        mock_json_loader: MagicMock,
    ):
        mock_store_input_data.return_value = ({"mean": 0.0, "std": 1.0}, None)
        sample_length = 10
        data_file_paths = self.__generate_dummy_test_data_files(sample_length=sample_length)
        mock_json_loader.return_value = {"file_paths": data_file_paths}
        scaling_method = "min_max_standard"
        output_data, features_dict = test_data_loader(
            path=self.meta_test_file_path,
            scaling_method=scaling_method,
            use_dummy_data=True,
        )
        self.assertEqual(
            features_dict,
            dict((idx, param_name) for idx, param_name in enumerate(self.input_parameters)),
        )
        self.assertEqual(mock_store_input_data.call_count, sample_length * len(self.input_parameters))
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
                torch.equal(input_tensor, torch.zeros((1, len(self.input_parameters), self.input_seq_length, GridSize.HEIGHT, GridSize.WIDTH))), True
            )
            self.assertEqual(
                torch.equal(label_tensor, torch.zeros((1, len(self.input_parameters), self.label_seq_length, GridSize.HEIGHT, GridSize.WIDTH))), True
            )
            self.assertEqual(
                standarize_info,
                dict(
                    (
                        param,
                        {"mean": 0.0, "std": 1.0},
                    )
                    for param in self.input_parameters
                ),
            )

            for param_idx, param_name in enumerate(self.input_parameters):
                store_input_data_call_args = dict(mock_store_input_data.call_args_list[sample_idx * len(self.input_parameters) + param_idx].kwargs)
                call_args_input_tensor = store_input_data_call_args.pop("input_tensor")

                store_label_data_call_args = dict(mock_store_label_data.call_args_list[sample_idx * len(self.input_parameters) + param_idx].kwargs)
                call_args_label_tensor = store_label_data_call_args.pop("label_tensor")
                self.assertEqual(
                    store_input_data_call_args,
                    {
                        "dataset_idx": 0,
                        "param_idx": param_idx,
                        "input_dataset_paths": data_file_paths[sample_name][param_name]["input"],
                        "scaling_method": scaling_method,
                        "inplace": True,
                    },
                )
                self.assertEqual(
                    torch.equal(
                        torch.zeros((1, len(self.input_parameters), self.input_seq_length, GridSize.HEIGHT, GridSize.WIDTH)),
                        call_args_input_tensor,
                    ),
                    True,
                )
                self.assertEqual(
                    store_label_data_call_args,
                    {
                        "dataset_idx": 0,
                        "param_idx": param_idx,
                        "label_dataset_paths": data_file_paths[sample_name][param_name]["label"],
                        "inplace": True,
                    },
                )
                self.assertEqual(
                    torch.equal(
                        torch.zeros(
                            (
                                1,
                                len(self.input_parameters),
                                self.input_seq_length,
                                GridSize.HEIGHT,
                                GridSize.WIDTH,
                            )
                        ),
                        call_args_label_tensor,
                    ),
                    True,
                )

    def __generate_dummy_test_data_files(self, sample_length: int = 10) -> Dict[str, Dict]:
        test_data_files = {}
        for sample_idx in range(sample_length):
            file_paths = {}
            file_paths["date"] = f"date{sample_idx}"
            file_paths["start"] = f"start{sample_idx}"
            for param_name in self.input_parameters:
                file_paths[param_name] = {"input": [], "label": []}
                for i in range(self.input_seq_length):
                    file_paths[param_name]["input"].append(f"/dataset{sample_idx}/input{i}.csv")
                for i in range(self.label_seq_length):
                    file_paths[param_name]["label"].append(f"/dataset{sample_idx}/label{i}.csv")
            test_data_files[f"sample{sample_idx}"] = file_paths
        return test_data_files

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
