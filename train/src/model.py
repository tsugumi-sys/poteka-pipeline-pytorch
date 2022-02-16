import logging
from typing import Dict, Tuple, List

# import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger("Train_Logger")


def evaluate(
    model: nn.Module,
    valid_dataloader: DataLoader,
    loss_criterion: nn.Module,
    acc_criterion: nn.Module,
    loss_only_rain: bool = False,
) -> Tuple[float, float]:
    """Evaluate model

    Args:
        model (nn.Module): Model to evaluate.
        valid_dataloader (DataLoader): torch DataLoader of validation dataset.
        loss_criterion (nn.Module): loss function for evaluation.
        acc_criterion (nn.Module): accuracy function for evaluation.
        calc_only_rain (bool, optional): Calculate acc and loss for only rain or all parameters. Defaults to False (all parameters).

    Returns:
        Tuple[float, float]: validation_loss, accuracy
    """
    validation_loss = 0.0
    accuracy = 0.0

    model.eval()
    with torch.no_grad():
        for input, target in valid_dataloader:
            output = model(input)

            # input, target is the shape of (batch_size, num_channels, seq_len, height, width)
            if loss_only_rain is True:
                output, target = output[:, 0, :, :, :], target[:, 0, :, :, :]
            valid_loss = loss_criterion(output.flatten(), target.flatten())
            validation_loss += valid_loss.item()

            acc_loss = acc_criterion(output.flatten(), target.flatten())
            accuracy += acc_loss.item()

    dataset_length = len(valid_dataloader.dataset)
    validation_loss /= dataset_length
    accuracy /= dataset_length

    return validation_loss, accuracy


def train(
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
    results = {"train_loss": [], "validation_loss": [], "validation_accuracy": []}

    for epoch in range(1, epochs + 1):
        train_loss = 0
        model.train()
        for _, (input, target) in enumerate(train_dataloader, start=1):
            output = model(input)

            # input, target is the shape of (batch_size, num_channels, seq_len, height, width)
            if loss_only_rain is True:
                output, target = output[:, 0, :, :, :], target[:, 0, :, :, :]

            loss = loss_criterion(output.flatten(), target.flatten())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        train_loss /= len(train_dataloader.dataset)

        validation_loss, validation_accuracy = evaluate(model, valid_dataloader, loss_criterion, acc_criterion, loss_only_rain)
        results["train_loss"].append(train_loss)
        results["validation_loss"].append(validation_loss)
        results["validation_accuracy"].append(validation_accuracy)

        logger.info(
            f"Epoch: {epoch:.2f} Training loss: {train_loss:.2f} Validation loss: {validation_loss:.2f} Validation accuracy: {validation_accuracy:.2f}\n"
        )
    return results
