from typing import List, Dict
import unittest
from unittest.mock import MagicMock

from hydra import initialize
import hydra
import torch
import numpy as np
import pandas as pd

from evaluate.src.base_evaluator import BaseEvaluator
from train.src.config import DEVICE
from common.config import WEATHER_PARAMS, GridSize, PPOTEKACols


class TestBaseEvaluator(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.model = MagicMock()
        self.model_name = "test_model"
        self.input_parameter_names = ["rain", "temperature", "humidity"]
        self.output_parameter_names = ["rain", "temperature", "humidity"]
        self.downstream_directory = "./tmp"
        self.test_dataset = generate_dummy_test_dataset(self.input_parameter_names)
        self.observation_point_file_path = "./common/meta-data/observation_point.json"

    def setUp(self) -> None:
        initialize(config_path="../../conf", version_base=None)
        return super().setUp()

    def tearDown(self) -> None:
        hydra.core.global_hydra.GlobalHydra.instance().clear()  # type:ignore
        return super().tearDown()

    def test__init__(self):
        """This tests evaluate initialziation of BaseEvaluator."""
        base_evaluator = BaseEvaluator(
            self.model,
            self.model_name,
            self.test_dataset,
            self.input_parameter_names,
            self.output_parameter_names,
            self.downstream_directory,
        )
        self.assertTrue(isinstance(base_evaluator.results_df, pd.DataFrame))
        self.assertTrue(isinstance(base_evaluator.metrics_df, pd.DataFrame))

    def test_load_test_case_dataset(self):
        """This function tests test dataset of a certain test case returned correctly."""
        base_evaluator = BaseEvaluator(
            self.model,
            self.model_name,
            self.test_dataset,
            self.input_parameter_names,
            self.output_parameter_names,
            self.downstream_directory,
        )
        for test_case_name, test_case_dataset in self.test_dataset.items():
            X_test, y_test = base_evaluator.load_test_case_dataset(test_case_name)

            self.assertTrue(torch.equal(X_test, test_case_dataset["input"]))
            self.assertTrue(torch.equal(y_test, test_case_dataset["label"]))

    def test_rescale_pred_tensor(self):
        """This function tests a given tensor is rescaled for a given parameter's scale."""
        base_evaluator = BaseEvaluator(
            self.model,
            self.model_name,
            self.test_dataset,
            self.input_parameter_names,
            self.output_parameter_names,
            self.downstream_directory,
        )

        tensor = (torch.rand((49, 50)) + (-0.50)) * 2  # This tensor is scaled as [-1, 1]
        rain_rescaled_tensor = base_evaluator.rescale_pred_tensor(tensor, target_param="rain")  # A given tensor scaled to [0, 100]
        self.assertTrue(rain_rescaled_tensor.min().item() >= 0.0)
        self.assertTrue(rain_rescaled_tensor.max().item() <= 100.0)

        temp_rescaled_tensor = base_evaluator.rescale_pred_tensor(tensor, target_param="temperature")
        self.assertTrue(temp_rescaled_tensor.min().item() >= 10.0)
        self.assertTrue(temp_rescaled_tensor.max().item() <= 45.0)

        humid_rescaled_tensor = base_evaluator.rescale_pred_tensor(tensor, target_param="humidity")
        self.assertTrue(humid_rescaled_tensor.min().item() >= 0.0)
        self.assertTrue(humid_rescaled_tensor.max().item() <= 100.0)

        wind_rescaled_tensor = base_evaluator.rescale_pred_tensor(tensor, target_param="u_wind")
        self.assertTrue(wind_rescaled_tensor.min().item() >= -10.0)
        self.assertTrue(wind_rescaled_tensor.max().item() <= 10.0)

    def test_rmse_from_label_df(self):
        base_evaluator = BaseEvaluator(
            self.model,
            self.model_name,
            self.test_dataset,
            self.input_parameter_names,
            self.output_parameter_names,
            self.downstream_directory,
        )

        label_df = pd.DataFrame({col: [idx] * 35 for idx, col in enumerate(PPOTEKACols.get_cols())})
        for idx, col in enumerate(PPOTEKACols.get_cols()):
            pred_tensor = torch.ones(GridSize.HEIGHT, GridSize.WIDTH) * idx
            rmse = base_evaluator.rmse_from_label_df(
                observation_point_file_path=self.observation_point_file_path,
                pred_tensor=pred_tensor,
                label_df=label_df,
                target_param=WEATHER_PARAMS.get_param_from_ppoteka_col(col),
            )
            self.assertTrue(rmse == 0)


def generate_dummy_test_dataset(input_parameter_names: List) -> Dict:
    """This function creates dummy test dataset."""
    dummy_tensor = torch.ones((5, len(input_parameter_names), 6, 50, 50), dtype=torch.float, device=DEVICE)
    sample1_input_tensor = dummy_tensor.clone().detach()
    sample1_label_tensor = dummy_tensor.clone().detach()
    sample2_input_tensor = dummy_tensor.clone().detach()
    sample2_label_tensor = dummy_tensor.clone().detach()
    # change value for each input parameters
    # rain -> 0, temperature -> 1, humidity -> 0.5)
    for i in range(len(input_parameter_names)):
        val = 1 / i if i > 0 else 0
        sample1_input_tensor[:, i, :, :, :] = val
        sample1_label_tensor[:, i, :, :, :] = val
        sample2_input_tensor[:, i, :, :, :] = val
        sample2_label_tensor[:, i, :, :, :] = val
    label_dfs = {}
    for i in range(sample1_input_tensor.size()[2]):
        data = {}
        for col in PPOTEKACols.get_cols():
            data[col] = np.ones((10))
            if col == "hour-rain":
                data[col] *= 0
            elif col == "RH1":
                data[col] /= 2
        label_dfs[i] = pd.DataFrame(data)

    test_dataset = {
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

    return test_dataset
