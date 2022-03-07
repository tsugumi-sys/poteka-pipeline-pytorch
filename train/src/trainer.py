import logging
import os
import sys
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.append("..")
from train.src.early_stopping import EarlyStopping
from train.src.validator import validator

logger = logging.getLogger("Train_Logger")


def trainer(
    model: nn.Module,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    optimizer: nn.Module,
    loss_criterion: nn.Module,
    acc_criterion: nn.Module,
    epochs: int = 32,
    checkpoints_directory: str = "/SimpleConvLSTM/model/",
    loss_only_rain: bool = False,
) -> Dict[str, List]:
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
    early_stopping = EarlyStopping(patience=20, verbose=True, delta=0.0001, path=os.path.join(checkpoints_directory, "model.pth"), trace_func=logger.info)

    for epoch in range(1, epochs + 1):
        train_loss = 0
        model.train()
        for _, (input, target) in enumerate(train_dataloader, start=1):
            optimizer.zero_grad()
            output = model(input)

            if torch.isnan(output).any():
                logger.warning(f"Input tensor size: {input.size()}")
                logger.warning(output)

            # input, target is the shape of (batch_size, num_channels, seq_len, height, width)
            if loss_only_rain is True:
                output, target = output[:, 0, :, :, :], target[:, 0, :, :, :]

            loss = loss_criterion(output.flatten(), target.flatten())

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)

        validation_loss, validation_accuracy = validator(model, valid_dataloader, loss_criterion, acc_criterion, loss_only_rain)
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
