import os
import unittest
from unittest.mock import MagicMock, mock_open, patch

from omegaconf import DictConfig
import torch
import hydra
from hydra import initialize

from train.src.config import DEVICE
from train.src.model_for_test import TestModel
from train.src.seq_to_seq import Seq2Seq
from train.src.trainer import Trainer


class TestTrainer(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.input_parameters = ["rain", "temperature"]
        self.train_input_tensor = torch.ones((5, 2, 6, 50, 50), dtype=torch.float, device=DEVICE)
        self.train_lanel_tensor = torch.ones((5, 2, 6, 50, 50), dtype=torch.float, device=DEVICE) * 2
        self.valid_input_tensor = torch.ones((5, 2, 6, 50, 50), dtype=torch.float, device=DEVICE) * 3
        self.valid_label_tensor = torch.ones((5, 2, 6, 50, 50), dtype=torch.float, device=DEVICE) * 4
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
            self.train_lanel_tensor,
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
            self.train_lanel_tensor,
            self.valid_input_tensor,
            self.valid_label_tensor,
            self.checkpoints_directory,
            self.use_test_model,
        )
        model_name = "model"
        mock_builtins_open = mock_open()
        with patch("builtins.open", mock_builtins_open):
            model = trainer._Trainer__initialize_model(model_name=model_name, return_sequences=False)
        self.assertTrue(model, TestModel)
        self.assertEqual(mock_builtins_open.call_args.args[0], os.path.join(self.checkpoints_directory, f"{model_name}_summary.txt"))
        self.assertEqual(mocked_torchinfo_summary.call_args.args, (model,))
        self.assertEqual(
            mocked_torchinfo_summary.call_args.kwargs,
            {
                "input_size": (
                    trainer.hydra_cfg.train.batch_size,
                    len(self.input_parameters),
                    trainer.hydra_cfg.input_seq_length,
                    trainer.hydra_cfg.tensor_height,
                    trainer.hydra_cfg.tensor_width,
                )
            },
        )
        # use_test_model=False
        trainer = Trainer(
            self.input_parameters,
            self.train_input_tensor,
            self.train_lanel_tensor,
            self.valid_input_tensor,
            self.valid_label_tensor,
            self.checkpoints_directory,
            use_test_model=False,
        )
        model_name = "model"
        mock_builtins_open = mock_open()
        with patch("builtins.open", mock_builtins_open):
            model = trainer._Trainer__initialize_model(model_name=model_name, return_sequences=False)
        self.assertTrue(model, Seq2Seq)
