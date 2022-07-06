from typing import Dict
import unittest
from unittest.mock import MagicMock, mock_open, patch

import torch
import pandas as pd
import numpy as np
from evaluate.src.evaluate import evaluate

from train.src.model_for_test import TestModel


class TestEvaluate(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.input_parameters = ["rain", "temperature", "humidity"]
        input_tensor = torch.ones((1, 3, 6, 50, 50))
        label_tensor = torch.ones((1, 3, 6, 50, 50))
        for i in range(len(self.input_parameters)):
            val = 0 if i == 0 else 1 / i
            input_tensor[:, i, :, :, :] = val
            label_tensor[:, i, :, :, :] = val
        self.test_dataset = {
            "sample0": {
                "date": "xxx",
                "start": "1-0",
                "input": input_tensor,
                "label": label_tensor,
                "label_df": {"0": pd.DataFrame(np.ones((50, 50)))},
            },
        }
        self.upstream_directory = "./dummy_upstream_directory"
        self.downstream_directory = "./dummy_downstream_directory"
        self.preprocess_downstream_directory = "./dummy_preprocess_downstream_directory"
        self.scaling_method = "dummy_scaling_method"

    @patch("evaluate.src.evaluate.TestModel")
    @patch("evaluate.src.evaluate.Evaluator")
    @patch("evaluate.src.evaluate.test_data_loader")
    @patch("evaluate.src.evaluate.order_meta_models")
    @patch("evaluate.src.evaluate.torch.load")
    def test_evaluate(
        self,
        mock_torch_load: MagicMock,
        mock_order_meta_models: MagicMock,
        mock_common_test_data_loader: MagicMock,
        mock_evaluator: MagicMock,
        mock_test_model: MagicMock,
    ):
        # setting up mocked functions
        mock_torch_load.return_value = {"model_state_dict": {}}
        mock_order_meta_models.return_value = {}
        for param_name in self.input_parameters:
            mock_order_meta_models.return_value[param_name] = {"return_sequences": True, "input_parameters": [param_name], "output_parameters": [param_name]}
        mock_order_meta_models.return_value["model"] = {
            "return_sequences": False,
            "input_parameters": self.input_parameters,
            "output_parameters": self.input_parameters,
        }
        mock_common_test_data_loader.return_value = (self.test_dataset, {0: "rain", 1: "temperature", 2: "humidity"})
        dummy_evaluator_run_result = {"result": 111}
        mock_evaluator.return_value.run.return_value = dummy_evaluator_run_result
        mocked_test_model = TestModel  # NOTE: TestModel.return_sequences is changing in evaluate. Test by call_args here.
        mocked_test_model.load_state_dict = MagicMock()
        mock_test_model.return_value = mocked_test_model()

        mock_builtins_open = mock_open()
        # run evaluate.evaluate
        with patch("builtins.open", mock_builtins_open):
            with patch("json.load", MagicMock(side_effect=mock_order_meta_models.return_value)):
                result = evaluate(
                    upstream_directory=self.upstream_directory,
                    downstream_directory=self.downstream_directory,
                    preprocess_downstream_directory=self.preprocess_downstream_directory,
                    use_dummy_data=True,
                    use_test_model=True,
                    scaling_method=self.scaling_method,
                    input_parameters=self.input_parameters,
                )
        # test evaluator result
        self.assertEqual(mock_evaluator.call_count, 4)
        self.assertIsInstance(result, Dict)
        self.assertTrue("model" in result and self.input_parameters[0] in result and self.input_parameters[1] in result and self.input_parameters[2] in result)
        self.assertTrue(result["model"], mock_evaluator.return_value.run.return_value)
        # test evaluator call of single parameter model
        test_case_name = "sample0"
        for idx, param_name in enumerate(self.input_parameters):
            self.assertIsInstance(mock_evaluator.call_args_list[idx].kwargs["model"], TestModel)
            self.assertEqual(mock_evaluator.call_args_list[idx].kwargs["model_name"], param_name)
            self.assertTrue(
                torch.equal(
                    mock_evaluator.call_args_list[idx].kwargs["test_dataset"][test_case_name]["input"],
                    self.test_dataset[test_case_name]["input"][:, idx : idx + 1, :, :, :],  # noqa: E203
                )
            )
            self.assertTrue(
                torch.equal(
                    mock_evaluator.call_args_list[idx].kwargs["test_dataset"][test_case_name]["label"],
                    self.test_dataset[test_case_name]["label"][:, idx : idx + 1, :, :, :],  # noqa: E203
                )
            )
            self.assertEqual(mock_evaluator.call_args_list[idx].kwargs["input_parameter_names"], [param_name])
            self.assertEqual(mock_evaluator.call_args_list[idx].kwargs["output_parameter_names"], [param_name])
            self.assertEqual(mock_evaluator.call_args_list[idx].kwargs["downstream_directory"], self.downstream_directory)
        # test evaluator call of multi trained model
        self.assertIsInstance(mock_evaluator.call_args_list[3].kwargs["model"], TestModel)
        self.assertEqual(mock_evaluator.call_args_list[3].kwargs["model_name"], "model")
        # self.assertTrue(mock_evaluator.call_args_list[3].kwargs["test_dataset"], self.test_dataset)
        self.assertTrue(mock_evaluator.call_args_list[3].kwargs["input_parameter_names"], self.input_parameters)
        self.assertTrue(mock_evaluator.call_args_list[3].kwargs["output_parameter_names"], self.input_parameters)
        self.assertTrue(mock_evaluator.call_args_list[3].kwargs["downstream_directory"], self.downstream_directory)
        # test evaluator.run call
        self.assertEqual(mock_evaluator.return_value.run.call_count, 4)
        for idx in range(len(self.input_parameters)):
            self.assertEqual(mock_evaluator.return_value.run.call_args_list[idx].kwargs["evaluate_types"], ["normal"])
        self.assertEqual(mock_evaluator.return_value.run.call_args_list[3].kwargs["evaluate_types"], ["reuse_predict", "sequential", "combine_models"])
        # test other mock function
        self.assertEqual(mock_test_model.call_count, 4)
        for idx in range(len(self.input_parameters)):
            self.assertEqual(mock_test_model.call_args_list[idx].kwargs["return_sequences"], True)
        self.assertEqual(mock_test_model.call_args_list[3].kwargs["return_sequences"], False)
