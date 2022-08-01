import os
from typing import Dict
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np
from common.config import WEATHER_PARAMS, MinMaxScalingValue

from common.utils import timestep_csv_names
from preprocess.src.extract_dummy_data import get_dummy_data_files, save_dummy_data, generate_dummy_data, get_meta_test_info


class TestExtractDummyData(unittest.TestCase):
    @patch("shutil.rmtree")
    @patch("preprocess.src.extract_dummy_data.save_dummy_data")
    def test_get_dummy_data_files(self, mock_save_dummy_data: MagicMock, mock_shutil_rmtree: MagicMock):
        mock_save_dummy_data.return_value = {"test": 1234}
        with self.assertRaises(ValueError):
            _ = get_dummy_data_files(["dummy_param"], time_step_minutes=10, downstream_dir_path="./data/preprocess", dataset_length=1)

        with self.assertRaises(ValueError):
            _ = get_dummy_data_files(["rain", "dummy_param"], time_step_minutes=10, downstream_dir_path="./data/preprocess", dataset_length=1)
        # Arguments
        input_parameters = ["rain"]
        time_step_minutes = 10
        downstream_dir_path = "./data/preprocess"
        dataset_length = 3
        input_seq_length = 6
        label_seq_length = 1
        dummy_data_files = get_dummy_data_files(
            input_parameters=input_parameters,
            time_step_minutes=time_step_minutes,
            downstream_dir_path=downstream_dir_path,
            dataset_length=dataset_length,
            input_seq_length=input_seq_length,
            label_seq_length=label_seq_length,
        )
        # tests
        self.assertTrue(mock_shutil_rmtree.call_count == 0 or mock_shutil_rmtree.call_count == 1)
        self.assertEqual(len(dummy_data_files), 3)
        self.assertTrue(isinstance(dummy_data_files[0], Dict))
        self.assertEqual(mock_save_dummy_data.call_count, 3)
        self.assertEqual(
            mock_save_dummy_data.call_args.kwargs,
            {
                "input_parameters": input_parameters,
                "time_step_minutes": time_step_minutes,
                "downstream_dir_path": downstream_dir_path,
                "input_seq_length": input_seq_length,
                "label_seq_length": label_seq_length,
            },
        )

    @patch("common.utils.timestep_csv_names")
    @patch("os.makedirs")
    @patch("os.listdir")
    @patch("os.path.exists")
    @patch("preprocess.src.extract_dummy_data.generate_dummy_data")
    @patch("pandas.DataFrame.to_csv")
    def test_save_dummy_data(
        self,
        mock_pd_df_to_csv: MagicMock,
        mock_generate_dummy_data: MagicMock,
        mock_os_path_exists: MagicMock,
        mock_os_listdirs: MagicMock,
        mock_os_makedirs: MagicMock,
        mock_timestep_csv_names: MagicMock,
    ):
        # arguments
        input_parameters = ["rain", "temperature"]
        time_step_minutes = 10
        downstream_dir_path = "./dummy_directory"
        input_seq_length = 6
        label_seq_length = 1
        # Mock return_value
        mock_timestep_csv_names.return_value = timestep_csv_names()
        mock_generate_dummy_data.return_value = pd.DataFrame(np.random.rand(3, 3))
        mock_os_listdirs.return_value = [f"0-{i*10}.csv" for i in range(6)]
        mock_os_path_exists.side_effect = lambda x: x.split("/")[-1] in mock_os_listdirs.return_value
        dummy_data_paths = save_dummy_data(
            input_parameters=input_parameters,
            time_step_minutes=time_step_minutes,
            downstream_dir_path=downstream_dir_path,
            input_seq_length=input_seq_length,
            label_seq_length=label_seq_length,
        )
        # tests
        self.assertTrue(isinstance(dummy_data_paths, dict))
        self.assertEqual(mock_os_makedirs.call_count, 2)
        self.assertEqual(mock_os_listdirs.call_count, 2)
        self.assertEqual(mock_os_path_exists.call_count, (input_seq_length + label_seq_length) * 2)
        self.assertEqual(mock_pd_df_to_csv.call_count, 2)  # There've been already 6 files so only 1 file saved for each input parameters.
        for idx, param_name in enumerate(input_parameters):
            save_dir_path = os.path.join(downstream_dir_path, "dummy_data", param_name)
            self.assertEqual(dummy_data_paths[param_name]["input"], [os.path.join(save_dir_path, f"0-{i*10}.csv") for i in range(6)])
            self.assertEqual(dummy_data_paths[param_name]["label"], [os.path.join(save_dir_path, "1-0.csv")])
            self.assertEqual(mock_os_makedirs.call_args_list[idx].args, (save_dir_path,))
            self.assertEqual(mock_os_listdirs.call_args_list[idx].args, (save_dir_path,))
            self.assertEqual(mock_pd_df_to_csv.call_args_list[idx].args, (os.path.join(save_dir_path, "1-0.csv"),))
        # test when the all files are already exist
        mock_timestep_csv_names = ["0-0.csv"]
        dummy_data_paths = save_dummy_data(
            input_parameters=input_parameters,
            time_step_minutes=time_step_minutes,
            downstream_dir_path=downstream_dir_path,
            input_seq_length=input_seq_length,
            label_seq_length=label_seq_length,
        )
        self.assertTrue(dummy_data_paths, None)

    def test_generate_dummy_data(self):
        # arguments
        input_parameters = WEATHER_PARAMS.valid_params()
        for param_name in input_parameters:
            with self.subTest(param_name=param_name):
                df = generate_dummy_data(param_name, array_shape=(50, 50))
                arr = df.to_numpy()
                max_val, min_val = arr.max(), arr.min()
                param_name = "wind" if param_name in ["v_wind", "u_wind"] else param_name
                valid_min_val, valid_max_val = MinMaxScalingValue.get_minmax_values_by_weather_param(param_name)
                self.assertTrue(valid_max_val >= max_val and valid_min_val <= min_val)

    def test_get_meta_test_info(self):
        # argument
        input_parameters = ["rain", "temperature"]
        dummy_input_file_names = [f"0-{i*10}.csv" for i in range(6)]
        dummy_label_file_names = [f"1-{i*10}.csv" for i in range(6)]
        label_seq_length = 6
        test_data_files_path = [
            {
                input_parameters[0]: {"input": dummy_input_file_names, "label": dummy_label_file_names},
                input_parameters[1]: {"input": dummy_input_file_names, "label": [dummy_label_file_names[0]]},
            },
        ]
        meta_test_info = get_meta_test_info(test_data_files_path, label_seq_length=label_seq_length)
        self.assertEqual(
            meta_test_info,
            {
                "sample0": {
                    input_parameters[0]: {
                        "input": dummy_input_file_names,
                        "label": dummy_label_file_names,
                    },
                    input_parameters[1]: {
                        "input": dummy_input_file_names,
                        "label": [dummy_label_file_names[0]] * label_seq_length,
                    },
                    "date": "sample_date_0",
                    "start": "1-0.csv",
                }
            },
        )
