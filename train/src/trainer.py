import logging
import os
import sys
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from hydra import initialize, compose
import torchinfo
from train.src.config import DEVICE, WeightsInitializer

from train.src.model_for_test import TestModel

sys.path.append("..")
from train.src.early_stopping import EarlyStopping
from train.src.validator import validator
from train.src.seq_to_seq import PotekaDataset, Seq2Seq

logger = logging.getLogger("Train_Logger")


class Trainer:
    def __init__(
        self,
        input_parameters: List[str],
        train_input_tensor: torch.Tensor,
        train_label_tensor: torch.Tensor,
        valid_input_tensor: torch.Tensor,
        valid_label_tensor: torch.Tensor,
        optimizer: nn.Module,
        loss_criterion: nn.Module,
        acc_criterion: nn.Module,
        checkpoints_directory: str = "/SimpleConvLSTM/model/",
        use_test_model: bool = False,
        train_sepalately: bool = False,
        loss_only_rain: bool = False,
    ) -> None:
        self.input_parameters = input_parameters
        self.train_input_tensor = train_input_tensor
        self.train_label_tensor = train_label_tensor
        self.valid_input_tensor = valid_input_tensor
        self.valid_label_tensor = valid_label_tensor
        self.optimizer = optimizer
        self.loss_criterion = loss_criterion
        self.acc_criterion = acc_criterion
        self.checkpoints_directory = checkpoints_directory
        self.use_test_model = use_test_model
        self.train_sepalately = train_sepalately
        self.loss_only_rain = loss_only_rain

        self.hydra_cfg = self.__initialize_hydra_conf()

    def __initialize_hydra_conf(self) -> DictConfig:
        initialize(vertion_base=None, config_path="../../conf")
        cfg = compose(config_name="train")
        return cfg

    def run(self) -> Dict[str, List]:
        """Trainer

        Args:
            model (nn.Module): _description_
            train_input_tensor (torch.Tensor): _description_
            train_label_tensor (torch.Tensor): _description_
            valid_input_tensor (torch.Tensor): _description_
            valid_label_tensor (torch.Tensor): _description_
            optimizer (nn.Module): _description_
            loss_criterion (nn.Module): _description_
            acc_criterion (nn.Module): _description_
            epochs (int, optional): _description_. Defaults to 32.
            checkpoints_directory (str, optional): _description_. Defaults to "/SimpleConvLSTM/model/".
            train_sepalately (bool, optional): If True, create model for each input parameters. Defaults to False.
            loss_only_rain (bool, optional): _description_. Defaults to False.

        Returns:
            Dict[str, List]: results of each model
                e.g. {model1: {training_loss: ...}, model2: {training_loss: ...}}
        """
        results = {}

        train_dataset = PotekaDataset(input_tensor=self.train_input_tensor, label_tensor=self.train_label_tensor)
        valid_dataset = PotekaDataset(input_tensor=self.valid_input_tensor, label_tensor=self.valid_label_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        results["model"] = self.__train(
            model_name="model",
            model=self.__initialize_model(return_sequences=False),
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
        )
        if self.train_sepalately is True:
            train_input_tensor_size, train_lanel_tensor_size = self.train_input_tensor.size(), self.train_label_tensor.size()
            valid_input_tensor_size, valid_label_tensor_size = self.valid_input_tensor.size(), self.valid_label_tensor.size()
            for idx, input_param in enumerate(self.input_parameters):
                train_dataset = PotekaDataset(
                    input_tensor=self.train_input_tensor[:, idx, :, :, :].reshape(train_input_tensor_size[0], 1, *train_input_tensor_size[2:]),
                    label_tensor=self.train_label_tensor[:, idx, :, :, :].reshape(train_lanel_tensor_size[0], 1, *train_lanel_tensor_size[2:]),
                )
                valid_dataset = PotekaDataset(
                    input_tensor=self.valid_input_tensor[:, idx, :, :, :].reshape(valid_input_tensor_size[0], 1, *valid_input_tensor_size[2:]),
                    label_tensor=self.valid_label_tensor[:, idx, :, :, :].reshape(valid_label_tensor_size[0], idx, *valid_label_tensor_size[2:]),
                )
                train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
                valid_dataloader = (DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True),)
                results[input_param] = self.__train(
                    model_name=input_param,
                    model=self.__initialize_model(return_sequences=True),
                    train_dataloader=train_dataloader,
                    valid_dataloader=valid_dataloader,
                )

        return results

    def __train(
        self,
        model_name: str,
        model: nn.Module,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
    ) -> Dict:
        """Train model

        Args:
            model (nn.Module): Model to train
            train_dataloader (DataLoader): Torch DataLoader of train dataset
            valid_dataloader (DataLoader): Torch DataLoader of valid dataset
            optimizer (nn.Module): Training Optimizer
            loss_criterion (nn.Module): Loss function for train
            acc_criterion (nn.Module): Criterion function for accuracy
            epochs (int, optional): Number of epochs. Defaults to 32.
            checkpoints_directory (str, optional): Directory path to save models. Defaults to "/SimpleConvLSTM/model/".
            loss_only_rain (bool, optional): Calculate loss and accuracy for only rain parameter or all parameters. Defaults to False.

        Returns:
            Dict: {"train_loss": List, "validation_loss": List, "validation_accuracy": List}
        """
        logger.info("start training ...")
        results = {"training_loss": [], "validation_loss": [], "validation_accuracy": []}
        early_stopping = EarlyStopping(
            patience=500, verbose=True, delta=0.0001, path=os.path.join(self.checkpoints_directory, f"{model_name}.pth"), trace_func=logger.info
        )

        for epoch in range(1, self.epochs + 1):
            train_loss = 0
            model.train()
            for _, (input, target) in enumerate(train_dataloader, start=1):
                self.optimizer.zero_grad()
                output: torch.Tensor = model(input)

                if torch.isnan(output).any():
                    logger.warning(f"Input tensor size: {input.size()}")
                    logger.warning(output)

                # input, target is the shape of (batch_size, num_channels, seq_len, height, width)
                if self.loss_only_rain is True:
                    output, target = output[:, 0, :, :, :], target[:, 0, :, :, :]

                # Outpuyt and target Validation
                if output.max().item() > 1.0 or output.min().item() < 0.0:
                    logger.error(f"Training output tensor is something wrong. Max value: {output.max().item()}, Min value: {output.min().item()}")

                if target.max().item() > 1.0 or target.min().item() < 0.0:
                    logger.error(f"Training target tensor is something wrong. Max value: {target.max().item()}, Min value: {target.min().item()}")

                loss = self.loss_criterion(output.flatten(), target.flatten())

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_dataloader)

            validation_loss, validation_accuracy = validator(model, valid_dataloader, self.loss_criterion, self.acc_criterion, self.loss_only_rain)
            results["training_loss"].append(train_loss)
            results["validation_loss"].append(validation_loss)
            results["validation_accuracy"].append(validation_accuracy)

            early_stopping(validation_loss, model)
            if early_stopping.early_stop is True:
                logger.info(f"Early Stopped at epoch {epoch}.")
                break
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch: {epoch} Training loss: {train_loss:.8f} Validation loss: {validation_loss:.8f} Validation accuracy: {validation_accuracy:.8f}\n"
                )

        results["model_state_dict"] = early_stopping.state_dict
        return results

    def __initialize_model(self, model_name: str, return_sequences: bool = False) -> nn.Module:
        num_channels, seq_length, HEIGHT, WIDTH = self.train_input_tensor.size()[1:]
        kernel_size = self.hydra_cfg.train.seq_to_seq.kernel_size
        num_kernels = self.hydra_cfg.train.num_kernels
        padding = self.hydra_cfg.padding
        activation = self.hydra_cfg.activation
        frame_size = (HEIGHT, WIDTH)
        num_layers = self.hydra_cfg.num_layers
        if self.use_test_model is True:
            model = TestModel(return_sequences=return_sequences).to(DEVICE).to(float)
        else:
            model = (
                Seq2Seq(
                    num_channels=num_channels,
                    kernel_size=kernel_size,
                    num_kernels=num_kernels,
                    padding=padding,
                    activation=activation,
                    frame_size=frame_size,
                    num_layers=num_layers,
                    weights_initializer=WeightsInitializer.He,
                    return_sequences=return_sequences,
                )
                .to(DEVICE)
                .to(float)
            )

        # Save summary
        model_summary_file_path = os.path.join(self.checkpoints_directory, f"{model_name}_summary.txt")
        with open(model_summary_file_path, "w") as f:
            f.write(repr(torchinfo.summary(model, input_size=(self.hydra_cfg.train.batch_size, num_channels, seq_length, HEIGHT, WIDTH))))

    def __initialize_optimiser(model: nn.Module) -> nn.Module:
    def __initialize_loss_criterion() -> nn.Module:
    def __initialize_accuracy_criterion() -> nn.Module: