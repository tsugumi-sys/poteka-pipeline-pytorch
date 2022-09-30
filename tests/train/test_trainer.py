import os
from typing import Dict
import unittest
from unittest.mock import MagicMock, mock_open, patch

from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import hydra
from hydra import initialize

from train.src.config import DEVICE
from train.src.model_for_test import TestModel
from train.src.seq_to_seq import PotekaDataset, Seq2Seq
from train.src.trainer import Trainer


class TestTrainer(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.input_parameters = ["rain", "temperature", "humidity"]
        self.train_input_tensor = torch.ones((5, 3, 6, 50, 50), dtype=torch.float, device=DEVICE)
        self.train_label_tensor = torch.ones((5, 3, 6, 50, 50), dtype=torch.float, device=DEVICE)
        self.valid_input_tensor = torch.ones((5, 3, 6, 50, 50), dtype=torch.float, device=DEVICE)
        self.valid_label_tensor = torch.ones((5, 3, 6, 50, 50), dtype=torch.float, device=DEVICE)
        # setup tensors
        # rain tensor is all 0 value matrix, temperature tensor is all 1 value matrix, humidity tensor is all 2 matrix.
        for i in range(len(self.input_parameters)):
            self.train_input_tensor[:, i, :, :, :] = i
            self.train_label_tensor[:, i, :, :, :] = i
            self.valid_input_tensor[:, i, :, :, :] = i
            self.valid_label_tensor[:, i, :, :, :] = i
        self.checkpoints_directory = "./dummy_check_points"
        self.use_test_model = True

    def setUp(self) -> None:
        initialize(config_path="../../conf", version_base=None)
        return super().setUp()

    def tearDown(self) -> None:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        return super().tearDown()

    def test_Trainer__initialize_hydra_conf(self):
        trainer = Trainer(
            self.input_parameters,
            self.train_input_tensor,
            self.train_label_tensor,
            self.valid_input_tensor,
            self.valid_label_tensor,
            self.checkpoints_directory,
            self.use_test_model,
        )
        self.assertTrue(isinstance(trainer.hydra_cfg, DictConfig))

    @patch("torchinfo.summary")
    def test_Trainer__initialize_model(self, mocked_torchinfo_summary: MagicMock):
        mocked_torchinfo_summary.return_value = "//dummy_model_info//"
        # use_test_model=True
        trainer = Trainer(
            self.input_parameters,
            self.train_input_tensor,
            self.train_label_tensor,
            self.valid_input_tensor,
            self.valid_label_tensor,
            self.checkpoints_directory,
            self.use_test_model,
        )
        model_name = "model"
        mock_builtins_open = mock_open()
        with patch("builtins.open", mock_builtins_open):
            model = trainer._Trainer__initialize_model(model_name=model_name, input_tensor_shape=self.train_input_tensor.shape, return_sequences=False)
        self.assertTrue(model, TestModel)
        # use_test_model=False
        trainer = Trainer(
            self.input_parameters,
            self.train_input_tensor,
            self.train_label_tensor,
            self.valid_input_tensor,
            self.valid_label_tensor,
            self.checkpoints_directory,
            use_test_model=False,
        )
        model_name = "model"
        mock_builtins_open = mock_open()
        with patch("builtins.open", mock_builtins_open):
            model = trainer._Trainer__initialize_model(model_name=model_name, input_tensor_shape=self.train_input_tensor.shape, return_sequences=False)
        self.assertTrue(model, Seq2Seq)

    @patch("train.src.trainer.validator")
    @patch("train.src.trainer.EarlyStopping")
    def test_Trainer__train(self, mock_earlystopping: MagicMock, mock_train_validator: MagicMock):
        # when return sequences is False
        return_sequences = False
        # setup mocked functions
        _, num_channels, _, height, width = self.train_input_tensor.size()
        mock_train_validator.return_value = (1.0, 1.0)
        # mock_earlystopping.side_effect = self.__earlystopping_side_effect()
        # initialize dataloader instance
        train_dataset = PotekaDataset(self.train_input_tensor, self.train_label_tensor)
        valid_dataset = PotekaDataset(self.valid_input_tensor, self.valid_label_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=self.train_input_tensor.size()[0], shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.valid_input_tensor.size()[0], shuffle=True)
        trainer = Trainer(
            self.input_parameters,
            self.train_input_tensor,
            self.train_label_tensor,
            self.valid_input_tensor,
            self.valid_label_tensor,
            self.checkpoints_directory,
            self.use_test_model,
        )
        trainer.hydra_cfg.train.epochs = 3
        dummy_model_name = "test_model"
        results: Dict = trainer._Trainer__train(
            model_name=dummy_model_name,
            model=TestModel(return_sequences=return_sequences),
            return_sequences=return_sequences,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
        )
        # test earlystopping call
        self.assertEqual(mock_earlystopping.call_count, 1)
        mock_earlystopping_call_kwargs = mock_earlystopping.call_args.kwargs
        self.assertEqual(mock_earlystopping_call_kwargs["patience"], trainer.hydra_cfg.train.earlystopping.patience)
        self.assertEqual(mock_earlystopping_call_kwargs["verbose"], trainer.hydra_cfg.train.earlystopping.verbose)
        self.assertEqual(mock_earlystopping_call_kwargs["delta"], trainer.hydra_cfg.train.earlystopping.delta)
        self.assertEqual(mock_earlystopping_call_kwargs["path"], os.path.join(self.checkpoints_directory, f"{dummy_model_name}.pth"))
        # test results dict
        self.assertTrue(isinstance(results, Dict))
        self.assertTrue("training_loss" in results and "validation_loss" in results and "validation_accuracy" in results and "return_sequences" in results)
        # test validator call
        self.assertEqual(mock_train_validator.call_count, 3)
        train_validator_call_args = mock_train_validator.call_args.args
        self.assertEqual(train_validator_call_args[1], valid_dataloader)
        self.assertEqual(train_validator_call_args[4], trainer.hydra_cfg.train.loss_only_rain)
        self.assertEqual(train_validator_call_args[5], return_sequences)

    @patch("train.src.trainer.PotekaDataset")
    @patch("train.src.trainer.DataLoader")
    @patch("train.src.trainer.Trainer._Trainer__train")
    @patch("train.src.trainer.Trainer._Trainer__initialize_model")
    def test_Trainer_run(
        self,
        mock_Trainer__initialize_model: MagicMock,
        mock_Trainer__train: MagicMock,
        mock_torch_dataloader: MagicMock,
        mock_train_potekadataloader: MagicMock,
    ):
        mock_Trainer__train.side_effect = self.__trainer__train_side_effect
        mock_Trainer__initialize_model.return_value = TestModel(return_sequences=False)
        trainer = Trainer(
            self.input_parameters,
            self.train_input_tensor,
            self.train_label_tensor,
            self.valid_input_tensor,
            self.valid_label_tensor,
            self.checkpoints_directory,
            self.use_test_model,
        )
        trainer.hydra_cfg.train.train_sepalately = True
        trainer.hydra_cfg.multi_parameters_model.return_sequences = False
        trainer.hydra_cfg.single_parameter_model.return_sequences = True
        results = trainer.run()
        # test results
        self.assertTrue(isinstance(results, Dict))
        self.assertTrue("model" in results)
        self.assertTrue("input_parameters" in results["model"] and "output_parameters" in results["model"])
        self.assertTrue(results["model"]["input_parameters"] == self.input_parameters)
        self.assertTrue(results["model"]["output_parameters"] == self.input_parameters)
        for param_name in self.input_parameters:
            self.assertTrue(param_name in results)
            self.assertTrue("input_parameters" in results[param_name] and "output_parameters" in results[param_name])
            self.assertTrue(results[param_name]["input_parameters"] == [param_name])
            self.assertTrue(results[param_name]["output_parameters"] == [param_name])
        # test PotekaDataset
        self.assertEqual(mock_train_potekadataloader.call_count, 8)
        self.assertEqual(
            mock_train_potekadataloader.call_args_list[0].kwargs, {"input_tensor": self.train_input_tensor, "label_tensor": self.train_label_tensor}
        )
        self.assertEqual(
            mock_train_potekadataloader.call_args_list[1].kwargs, {"input_tensor": self.valid_input_tensor, "label_tensor": self.valid_label_tensor}
        )
        for idx, param_name in enumerate(self.input_parameters):
            with self.subTest(param_name=param_name):
                self.assertEqual(mock_train_potekadataloader.call_args_list[idx * 2 + 2].kwargs["input_tensor"].mean(), idx)
                self.assertEqual(mock_train_potekadataloader.call_args_list[idx * 2 + 2].kwargs["label_tensor"].mean(), idx)
                self.assertEqual(mock_train_potekadataloader.call_args_list[idx * 2 + 3].kwargs["input_tensor"].mean(), idx)
                self.assertEqual(mock_train_potekadataloader.call_args_list[idx * 2 + 3].kwargs["label_tensor"].mean(), idx)
        # test Dataset
        self.assertEqual(mock_torch_dataloader.call_count, 8)
        # test Trainer._Trainer__train
        self.assertEqual(mock_Trainer__train.call_count, 4)
        self.assertEqual(mock_Trainer__train.call_args_list[0].kwargs["model_name"], "model")
        for idx, param_name in enumerate(self.input_parameters):
            self.assertEqual(mock_Trainer__train.call_args_list[idx + 1].kwargs["model_name"], param_name)
        # test mocked Trainer._Trainer__initialize_model
        # TODO: Add more concrete tests
        self.assertEqual(mock_Trainer__initialize_model.call_count, 4)

    def __trainer__train_side_effect(self, *args, **kwargs):
        return {"training_loss": [], "validation_loss": [], "validation_accuracy": [], "return_sequences": kwargs["return_sequences"]}
