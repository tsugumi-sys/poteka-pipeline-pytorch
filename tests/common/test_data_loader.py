import os
import shutil
import json
import unittest
from unittest.mock import MagicMock, patch
from typing import List, Dict, Union

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
        self.meta_train_file_path = "./tmp/meta_train.json"
        self.meta_test_file_path = "./tmp/meta_test.json"
        # valid tensor
        # NOTE: tensor is still cpu in train_test_loader
        self.tensor_multiplyer = {"rain": 1, "temperature": 2, "humidity": 3, "wind": 4}
        self.rain_tensor = torch.ones((GridSize.HEIGHT, GridSize.WIDTH), dtype=torch.float) * self.tensor_multiplyer["rain"]
        self.temperature_tensor = torch.ones((GridSize.HEIGHT, GridSize.WIDTH), dtype=torch.float) * self.tensor_multiplyer["temperature"]
        self.humidity_tensor = torch.ones((GridSize.HEIGHT, GridSize.WIDTH), dtype=torch.float) * self.tensor_multiplyer["humidity"]
        self.wind_tensor = torch.ones((GridSize.HEIGHT, GridSize.WIDTH), dtype=torch.float) * self.tensor_multiplyer["wind"]

        self.train_data_file_paths = self.__get_train_file_paths(dataset_length=10)
        self.test_data_file_paths = self.__get_test_file_paths(dataset_length=10)

    def setUp(self) -> None:
        os.makedirs("./tmp", exist_ok=True)
        # Save meta-train.json
        with open(self.meta_train_file_path, "w") as f:
            json.dump({"file_paths": self.train_data_file_paths}, f)
        # Save meta-test.json
        with open(self.meta_test_file_path, "w") as f:
            json.dump({"file_paths": self.test_data_file_paths}, f)
        return super().setUp()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        shutil.rmtree("./tmp", ignore_errors=True)

    def __get_dummy_file_paths(self, save_dir_path: str) -> Dict:
        file_paths = {}
        for param_name in self.input_parameters:
            file_paths[param_name] = {"input": [], "label": []}
            dummy_data_dir_path = os.path.join(save_dir_path, param_name)
            os.makedirs(dummy_data_dir_path, exist_ok=True)
            for i in range(self.input_seq_length):
                dummy_file_path = os.path.join(dummy_data_dir_path, f"input{i}.csv")
                dummy_df = pd.DataFrame({"sample": [1]})
                dummy_df.to_csv(dummy_file_path)
                file_paths[param_name]["input"].append(dummy_file_path)
            for i in range(self.label_seq_length):
                dummy_file_path = os.path.join(dummy_data_dir_path, f"label{i}.csv")
                dummy_df = pd.DataFrame({"sample": [2]})
                dummy_df.to_csv(dummy_file_path)
                file_paths[param_name]["label"].append(dummy_file_path)
        return file_paths

    def __get_train_file_paths(self, dataset_length: int = 10) -> List:
        # TODO: Change schemas and near to the real poteka directory
        data_file_paths = []
        for dataset_idx in range(dataset_length):
            save_dir_path = f"./tmp/train_dataset{dataset_idx}"
            os.makedirs(save_dir_path, exist_ok=True)
            dummy_file_paths = self.__get_dummy_file_paths(save_dir_path)
            data_file_paths.append(dummy_file_paths)
        return data_file_paths

    def __get_test_file_paths(self, dataset_length: int = 10) -> Dict:
        data_file_paths = {}
        for dataset_idx in range(dataset_length):
            save_dir_path = f"./tmp/test_dataset{dataset_idx}"
            os.makedirs(save_dir_path, exist_ok=True)
            dummy_file_paths = self.__get_dummy_file_paths(save_dir_path)
            dummy_file_paths["date"] = "date"
            dummy_file_paths["start"] = "12-0.csv"

            data_file_paths[f"sample{dataset_idx}"] = dummy_file_paths
        return data_file_paths

    @patch("common.data_loader.store_input_data")
    @patch("common.data_loader.store_label_data")
    def test_train_data_loader(
        self,
        mock_store_label_data: MagicMock,
        mock_store_input_data: MagicMock,
    ):
        mock_store_input_data.return_value = (None, None)
        scaling_method = "min_max_standard"
        input_tensor, label_tensor = train_data_loader(
            meta_data_file_path=self.meta_train_file_path,
            observation_point_file_path="./common/meta-data/observation_point.json",
            scaling_method=scaling_method,
        )
        self.assertEqual(
            mock_store_input_data.call_count,
            input_tensor.size(0) * len(self.input_parameters),
        )
        # self.assertEqual(
        #    mock_store_label_data.call_count,
        #    label_tensor.size(0) * len(self.input_parameters),
        # )
        for dataset_idx in range(input_tensor.size(0)):
            for param_idx, param_name in enumerate(self.input_parameters):
                store_input_data_call_args = dict(mock_store_input_data.call_args_list[dataset_idx * len(self.input_parameters) + param_idx].kwargs)
                call_args_input_tensor = store_input_data_call_args.pop("input_tensor")

                # store_label_data_call_args = dict(mock_store_label_data.call_args_list[dataset_idx * len(self.input_parameters) + param_idx].kwargs)
                # call_args_label_tensor = store_label_data_call_args.pop("label_tensor")
                # self.assertEqual(
                #    store_input_data_call_args,
                #    {
                #        "dataset_idx": dataset_idx,
                #        "param_idx": param_idx,
                #        "input_dataset_paths": self.train_data_file_paths[dataset_idx][param_name]["input"],
                #        "scaling_method": scaling_method,
                #        "inplace": True,
                #    },
                # )
                # self.assertEqual(
                #    torch.equal(
                #        torch.Tensor(
                #            len(self.train_data_file_paths),
                #            len(self.input_parameters),
                #            self.input_seq_length,
                #            GridSize.HEIGHT,
                #            GridSize.WIDTH,
                #        ),
                #        call_args_input_tensor,
                #    ),
                #    True,
                # )
                # self.assertEqual(
                #    store_label_data_call_args,
                #    {
                #        "dataset_idx": dataset_idx,
                #        "param_idx": param_idx,
                #        "label_dataset_paths": self.train_data_file_paths[dataset_idx][param_name]["label"],
                #        "inplace": True,
                #    },
                # )
                # self.assertEqual(
                #    torch.equal(
                #        torch.Tensor(
                #            len(self.train_data_file_paths),
                #            len(self.input_parameters),
                #            self.input_seq_length,
                #            GridSize.HEIGHT,
                #            GridSize.WIDTH,
                #        ),
                #        call_args_label_tensor,
                #    ),
                #    True,
                # )

    @patch("common.data_loader.store_input_data")
    @patch("common.data_loader.store_label_data")
    def test_test_data_loader(
        self,
        mock_store_label_data: MagicMock,
        mock_store_input_data: MagicMock,
    ):
        mock_store_input_data.return_value = ({"mean": 0.0, "std": 1.0}, None)
        sample_length = 10
        scaling_method = "min_max_standard"
        output_data, features_dict = test_data_loader(
            meta_data_file_path=self.meta_test_file_path,
            observation_point_file_path="./common/meta-data/observation_point.json",
            scaling_method=scaling_method,
            use_dummy_data=True,
        )
        self.assertEqual(
            features_dict,
            dict((idx, param_name) for idx, param_name in enumerate(self.input_parameters)),
        )
        self.assertEqual(mock_store_input_data.call_count, sample_length * len(self.input_parameters))
        for sample_idx, sample_name in enumerate(list(self.test_data_file_paths.keys())):
            input_tensor = output_data[sample_name]["input"]
            label_tensor = output_data[sample_name]["label"]
            date = output_data[sample_name]["date"]
            label_dfs = output_data[sample_name]["label_df"]
            standarize_info = output_data[sample_name]["standarize_info"]
            # test meta info
            self.assertEqual(date, self.test_data_file_paths[sample_name]["date"])
            for df in label_dfs.values():
                self.assertIsInstance(df, pd.DataFrame)
                self.assertEqual(df.columns.tolist(), PPOTEKACols.get_cols())

            self.assertEqual(
                torch.equal(input_tensor, torch.zeros((1, len(self.input_parameters), self.input_seq_length, GridSize.HEIGHT, GridSize.WIDTH))), True
            )
            # self.assertEqual(
            #    torch.equal(label_tensor, torch.zeros((1, len(self.input_parameters), self.label_seq_length, GridSize.HEIGHT, GridSize.WIDTH))), True
            # )
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

                # store_label_data_call_args = dict(mock_store_label_data.call_args_list[sample_idx * len(self.input_parameters) + param_idx].kwargs)
                # call_args_label_tensor = store_label_data_call_args.pop("label_tensor")
                self.assertEqual(
                    store_input_data_call_args,
                    {
                        "dataset_idx": 0,
                        "param_idx": param_idx,
                        "input_dataset_paths": self.test_data_file_paths[sample_name][param_name]["input"],
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
                # self.assertEqual(
                #    store_label_data_call_args,
                #    {
                #        "dataset_idx": 0,
                #        "param_idx": param_idx,
                #        "label_dataset_paths": self.test_data_file_paths[sample_name][param_name]["label"],
                #        "inplace": True,
                #    },
                # )
                # self.assertEqual(
                #    torch.equal(
                #        torch.zeros(
                #            (
                #                1,
                #                len(self.input_parameters),
                #                self.input_seq_length,
                #                GridSize.HEIGHT,
                #                GridSize.WIDTH,
                #            )
                #        ),
                #        call_args_label_tensor,
                #    ),
                #    True,
                # )
