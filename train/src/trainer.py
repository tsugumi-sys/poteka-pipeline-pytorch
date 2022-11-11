import logging
import os
import sys
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from omegaconf import DictConfig
from hydra import compose
import torchinfo

sys.path.append("..")
from train.src.config import DEVICE, WeightsInitializer  # noqa: E402
from train.src.model_for_test import TestModel  # noqa: E402
from train.src.early_stopping import EarlyStopping  # noqa: E402
from train.src.validator import validator  # noqa: E402
from train.src.seq_to_seq import PotekaDataset, RMSELoss  # noqa: E402
from train.src.obpoint_seq_to_seq import OBPointSeq2Seq  # noqa: E402

logger = logging.getLogger("Train_Logger")


class Trainer:
    def __init__(
        self,
        input_parameters: List[str],
        train_input_tensor: torch.Tensor,
        train_label_tensor: torch.Tensor,
        valid_input_tensor: torch.Tensor,
        valid_label_tensor: torch.Tensor,
        checkpoints_directory: str = "/SimpleConvLSTM/model/",
        use_test_model: bool = False,
        hydra_overrides: List[str] = [],
    ) -> None:
        self.input_parameters = input_parameters
        self.train_input_tensor = train_input_tensor
        self.train_label_tensor = train_label_tensor
        self.valid_input_tensor = valid_input_tensor
        self.valid_label_tensor = valid_label_tensor
        self.checkpoints_directory = checkpoints_directory
        self.use_test_model = use_test_model

        self.ob_point_count = train_label_tensor.size(-1)
        self.hydra_cfg = self.__initialize_hydra_conf(hydra_overrides)

    def __initialize_hydra_conf(self, overrides: List[str]) -> DictConfig:
        cfg = compose(config_name="config", overrides=overrides)
        return cfg

    def run(self) -> Dict[str, List]:
        results = {}

        logger.info("... model training with all parameters...")
        logger.info(f"Input parameters: {self.input_parameters}, input tensor shape: {self.train_input_tensor.shape}")

        if self.hydra_cfg.multi_parameters_model.return_sequences:
            # If return_sequences is True, the output (label) should be only rain.
            _, train_label_tensor, _, valid_label_tensor = self.extract_tensor_from_channel_dim(target_channel_dim=0)
            train_dataset = PotekaDataset(input_tensor=self.train_input_tensor, label_tensor=train_label_tensor)
            valid_dataset = PotekaDataset(input_tensor=self.valid_input_tensor, label_tensor=valid_label_tensor)
            train_dataloader = DataLoader(train_dataset, batch_size=self.hydra_cfg.train.batch_size, shuffle=True, drop_last=True)
            valid_dataloader = DataLoader(valid_dataset, batch_size=self.hydra_cfg.train.batch_size, shuffle=True, drop_last=True)
            logger.info(f"Output parameters: {self.input_parameters[0]}, label tensor shape: {train_label_tensor.shape}")
        else:
            train_dataset = PotekaDataset(input_tensor=self.train_input_tensor, label_tensor=self.train_label_tensor)
            valid_dataset = PotekaDataset(input_tensor=self.valid_input_tensor, label_tensor=self.valid_label_tensor)
            train_dataloader = DataLoader(train_dataset, batch_size=self.hydra_cfg.train.batch_size, shuffle=True, drop_last=True)
            valid_dataloader = DataLoader(valid_dataset, batch_size=self.hydra_cfg.train.batch_size, shuffle=True, drop_last=True)
            logger.info(f"Output parameters: {self.input_parameters}, label tensor shape: {self.train_label_tensor.shape}")

        model = self.__initialize_model(
            model_name="model", input_tensor_shape=self.train_input_tensor.shape, return_sequences=self.hydra_cfg.multi_parameters_model.return_sequences
        )
        results["model"] = self.__train(
            model_name="model",
            model=model,
            return_sequences=self.hydra_cfg.multi_parameters_model.return_sequences,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
        )
        results["model"]["input_parameters"] = self.input_parameters
        if self.hydra_cfg.multi_parameters_model.return_sequences:
            results["model"]["output_parameters"] = ["rain"]
        else:
            results["model"]["output_parameters"] = self.input_parameters

        if self.hydra_cfg.train.train_sepalately is True:
            train_input_tensor_size, train_lanel_tensor_size = self.train_input_tensor.size(), self.train_label_tensor.size()
            valid_input_tensor_size, valid_label_tensor_size = self.valid_input_tensor.size(), self.valid_label_tensor.size()
            for idx, input_param in enumerate(self.input_parameters):
                # Update train and valid tensors
                train_input_tensor, train_label_tensor, valid_input_tensor, valid_label_tensor = self.extract_tensor_from_channel_dim(target_channel_dim=idx)
                train_dataset = PotekaDataset(input_tensor=train_input_tensor, label_tensor=train_label_tensor)
                valid_dataset = PotekaDataset(input_tensor=valid_input_tensor, label_tensor=valid_label_tensor)
                train_dataloader = DataLoader(train_dataset, batch_size=self.hydra_cfg.train.batch_size, shuffle=True, drop_last=True)
                valid_dataloader = DataLoader(valid_dataset, batch_size=self.hydra_cfg.train.batch_size, shuffle=True, drop_last=True)
                logger.info(f"... model training with {input_param} ...")
                logger.info(f"Input parameter: {input_param}, input tensor shape: {train_input_tensor.shape}")
                logger.info(f"Output parameter: {input_param}, label tensor shape: {valid_label_tensor.shape}")
                # Run training
                model = self.__initialize_model(
                    model_name=input_param, input_tensor_shape=train_input_tensor.shape, return_sequences=self.hydra_cfg.single_parameter_model.return_sequences
                )
                results[input_param] = self.__train(
                    model_name=input_param,
                    model=model,
                    return_sequences=self.hydra_cfg.single_parameter_model.return_sequences,
                    train_dataloader=train_dataloader,
                    valid_dataloader=valid_dataloader,
                )
                results[input_param]["input_parameters"] = [input_param]
                results[input_param]["output_parameters"] = [input_param]

        return results

    def extract_tensor_from_channel_dim(self, target_channel_dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            This function extract tensor of target channel dimention and return the same shape as original tensor.
        
            Return:
                (train_input_tensor, train_label_tensor, valid_input_tensor, valid_label_tensor)
        """
        train_input_tensor_size, train_lanel_tensor_size = self.train_input_tensor.size(), self.train_label_tensor.size()
        valid_input_tensor_size, valid_label_tensor_size = self.valid_input_tensor.size(), self.valid_label_tensor.size()
        train_input_tensor = self.train_input_tensor[:, target_channel_dim, ...].reshape(train_input_tensor_size[0], 1, *train_input_tensor_size[2:])
        train_label_tensor = self.train_label_tensor[:, target_channel_dim, ...].reshape(train_lanel_tensor_size[0], 1, *train_lanel_tensor_size[2:])
        valid_input_tensor = self.valid_input_tensor[:, target_channel_dim, ...].reshape(valid_input_tensor_size[0], 1, *valid_input_tensor_size[2:])
        valid_label_tensor = self.valid_label_tensor[:, target_channel_dim, ...].reshape(valid_label_tensor_size[0], 1, *valid_label_tensor_size[2:])
        return (train_input_tensor, train_label_tensor, valid_input_tensor, valid_label_tensor)

    def __train(self, model_name: str, model: nn.Module, return_sequences: bool, train_dataloader: DataLoader, valid_dataloader: DataLoader,) -> Dict:
        """_summary_

        Args:
            model_name (str): _description_
            train_dataloader (DataLoader): _description_
            valid_dataloader (DataLoader): _description_

        Returns:
            Dict: {"train_loss": List, "validation_loss": List, "validation_accuracy": List}
        """
        logger.info("start training ...")
        optimizer = self.__initialize_optimiser(model)
        loss_criterion = self.__initialize_loss_criterion()
        acc_criterion = self.__initialize_accuracy_criterion()
        results = {"training_loss": [], "validation_loss": [], "validation_accuracy": [], "return_sequences": return_sequences}
        early_stopping = EarlyStopping(
            patience=self.hydra_cfg.train.earlystopping.patience,
            verbose=self.hydra_cfg.train.earlystopping.verbose,
            delta=self.hydra_cfg.train.earlystopping.delta,
            path=os.path.join(self.checkpoints_directory, f"{model_name}.pth"),
            trace_func=logger.info,
        )

        for epoch in range(1, self.hydra_cfg.train.epochs + 1):
            train_loss = 0
            model.train()
            for _, (input, target) in enumerate(train_dataloader, start=1):
                optimizer.zero_grad()
                output: torch.Tensor = model(input)

                if torch.isnan(output).any():
                    logger.warning(f"Input tensor size: {input.size()}")
                    logger.warning(output)

                # input, target is the shape of (batch_size, num_channels, seq_len, height, width)
                if self.hydra_cfg.train.loss_only_rain is True:
                    output, target = output[:, 0, ...], target[:, 0, ...]

                # Outpuyt and target Validation
                if output.max().item() > 1.0 or output.min().item() < 0.0:
                    logger.error(f"Training output tensor is something wrong. Max value: {output.max().item()}, Min value: {output.min().item()}")

                if target.max().item() > 1.0 or target.min().item() < 0.0:
                    logger.error(f"Training target tensor is something wrong. Max value: {target.max().item()}, Min value: {target.min().item()}")

                if return_sequences is False and target.size()[2] > 1:
                    target = target[:, :, 0, ...]

                loss = loss_criterion(output.flatten(), target.flatten())

                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_dataloader)

            validation_loss, validation_accuracy = validator(
                model, valid_dataloader, loss_criterion, acc_criterion, self.hydra_cfg.train.loss_only_rain, return_sequences
            )
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
        return results

    def __initialize_model(self, model_name: str, input_tensor_shape: Tuple, return_sequences: bool = False) -> nn.Module:
        _, num_channels, seq_length, HEIGHT, WIDTH = input_tensor_shape
        frame_size = (HEIGHT, WIDTH)
        if self.use_test_model is True:
            model = TestModel(return_sequences=return_sequences).to(DEVICE).to(torch.float)
        else:
            kernel_size = self.hydra_cfg.train.seq_to_seq.kernel_size
            num_kernels = self.hydra_cfg.train.seq_to_seq.num_kernels
            padding = self.hydra_cfg.train.seq_to_seq.padding
            activation = self.hydra_cfg.train.seq_to_seq.activation
            num_layers = self.hydra_cfg.train.seq_to_seq.num_layers
            input_seq_length = self.hydra_cfg.input_seq_length
            label_seq_length = self.hydra_cfg.label_seq_length
            model = (
                OBPointSeq2Seq(
                    num_channels=num_channels,
                    ob_point_count=self.ob_point_count,
                    kernel_size=kernel_size,
                    num_kernels=num_kernels,
                    padding=padding,
                    activation=activation,
                    frame_size=frame_size,
                    num_layers=num_layers,
                    input_seq_length=input_seq_length,
                    prediction_seq_length=label_seq_length,
                    out_channels=None if return_sequences is False else 1,
                    weights_initializer=WeightsInitializer.He.value,
                    return_sequences=return_sequences,
                )
                .to(DEVICE)
                .to(torch.float)
            )

        # Save summary
        model_summary_file_path = os.path.join(self.checkpoints_directory, f"{model_name}_summary.txt")
        with open(model_summary_file_path, "w") as f:
            f.write(repr(torchinfo.summary(model, input_size=(self.hydra_cfg.train.batch_size, num_channels, seq_length, HEIGHT, WIDTH))))
        return model

    def __initialize_optimiser(self, model: nn.Module) -> nn.Module:
        return Adam(model.parameters(), lr=self.hydra_cfg.train.optimizer_learning_rate)

    def __initialize_loss_criterion(self) -> nn.Module:
        return nn.BCELoss()

    def __initialize_accuracy_criterion(self) -> nn.Module:
        return RMSELoss(reduction="mean")
