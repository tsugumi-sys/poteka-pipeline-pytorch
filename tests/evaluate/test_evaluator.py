import unittest
from unittest.mock import MagicMock, patch
from typing import Dict

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
        self.model = MagicMock()  # You need to set different shape torch.Tensor based on the evaluate_type
        self.model_name = "test_model"
        self.input_parameters = ["rain", "temperature", "humdity"]
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
                "start": "1-0",
                "input": sample1_input_tensor,
                "label": sample1_label_tensor,
                "label_df": label_dfs,
            },
            "sample2": {
                "date": "2022-01-02",
                "start": "2-0",
                "input": sample2_input_tensor,
                "label": sample2_label_tensor,
                "label_df": label_dfs,
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
