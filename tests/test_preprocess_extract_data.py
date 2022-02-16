import unittest
import pandas as pd

from preprocess.src.extract_data import get_train_data_files, get_test_data_files
from preprocess.src.constants import WEATHER_PARAMS_ENUM


class TestPreprocessExtractdata(unittest.TestCase):
    def test_get_train_data_files(self):
        train_list_df = pd.DataFrame(
            {
                "date": ["2020-01-01"],
                "start_time": ["10-0"],
                "end_time": ["11-0"],
            }
        )
        train_data_files = get_train_data_files(
            train_list_df=train_list_df,
            params=WEATHER_PARAMS_ENUM.valid_params(),
            delta=10,
            slides=3,
        )
        self.assertIsInstance(train_data_files, list)

        # input wind param, u-wind and v-wind loads
        expected_params = WEATHER_PARAMS_ENUM.valid_params()
        expected_params = [i for i in expected_params if i != "wind"]
        expected_params += ["u_wind", "v_wind"]
        for sample in train_data_files:
            with self.subTest(sample=sample):
                self.assertEqual(sorted(list(sample.keys())), sorted(expected_params))
                # Each params has input and label key

                for p in expected_params:
                    with self.subTest(param=p):
                        param_data_files = sample[p]
                        self.assertEqual(sorted(list(param_data_files.keys())), sorted(["input", "label"]))

                        # input length is 6, label length is 1
                        self.assertEqual(len(param_data_files["input"]), 6)
                        self.assertEqual(len(param_data_files["label"]), 1)

    def test_get_test_data_files(self):
        test_data_list = {
            "TC_case": {
                "sample1": {
                    0: {
                        "date": "2020-10-12",
                        "start": "5-0.csv",
                    },
                    1: {
                        "date": "2020-10-12",
                        "start": "6-0.csv",
                    },
                    2: {
                        "date": "2020-10-12",
                        "start": "7-0.csv",
                    },
                    3: {
                        "date": "2020-10-12",
                        "start": "8-0.csv",
                    },
                    4: {
                        "date": "2020-10-12",
                        "start": "9-0.csv",
                    },
                }
            },
            "NOT_TC_case": {
                "sample1": {
                    0: {
                        "date": "2020-07-04",
                        "start": "6-0.csv",
                    },
                    1: {
                        "date": "2020-07-04",
                        "start": "7-0.csv",
                    },
                    2: {
                        "date": "2020-07-04",
                        "start": "8-0.csv",
                    },
                    3: {
                        "date": "2020-07-04",
                        "start": "9-0.csv",
                    },
                    4: {
                        "date": "2020-07-04",
                        "start": "10-0.csv",
                    },
                }
            },
        }

        test_data_files = get_test_data_files(
            test_data_list=test_data_list,
            params=WEATHER_PARAMS_ENUM.valid_params(),
            delta=10,
        )

        # Keys should be sample names like TC_case_{date}_{start} without .csv
        expected_key_names = []
        for key in test_data_list.keys():
            for sample_name in test_data_list[key].keys():
                for case_idx in test_data_list[key][sample_name]:
                    case = test_data_list[key][sample_name][case_idx]
                    start = case["start"].replace(".csv", "")
                    date = case["date"]
                    expected_key_names += [f"{key}_{date}_{start}_start"]

        self.assertEqual(sorted(list(test_data_files.keys())), sorted(expected_key_names))

        # input wind param, u-wind and v-wind loads
        expected_params = WEATHER_PARAMS_ENUM.valid_params()
        expected_params = [i for i in expected_params if i != "wind"]
        expected_params += ["u_wind", "v_wind"]
        # test_data_files should has date and start
        test_data_expected_params = expected_params.copy()
        test_data_expected_params += ["date", "start"]
        for case in test_data_files.values():
            # each caase has each parameters
            with self.subTest(case=case):
                self.assertEqual(sorted(list(case.keys())), sorted(test_data_expected_params))

                for param in expected_params:
                    with self.subTest(param=param):
                        param_data = case[param]
                        self.assertEqual(sorted(list(param_data.keys())), sorted(["input", "label"]))

                        self.assertEqual(len(param_data["input"]), 6)
                        self.assertEqual(len(param_data["label"]), 6)
