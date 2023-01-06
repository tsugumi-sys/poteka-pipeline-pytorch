from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


def validator(
    model: nn.Module,
    valid_dataloader: DataLoader,
    loss_criterion: nn.Module,
    acc_criterion: nn.Module,
    loss_only_rain: bool = False,
    return_sequences: bool = False,
) -> Tuple[float, float]:
    """Evaluate model

    Args:
        model (nn.Module): Model to evaluate.
        valid_dataloader (DataLoader): torch DataLoader of validation dataset.
        loss_criterion (nn.Module): loss function for evaluation.
        acc_criterion (nn.Module): accuracy function for evaluation.
        calc_only_rain (bool, optional): Calculate acc and loss for only rain or all parameters.
            Defaults to False (all parameters).

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
                output, target = output[:, 0, ...], target[:, 0, ...]

            if return_sequences is False and target.size()[2] > 1:
                target = target[:, :, -1, ...]

            valid_loss = loss_criterion(output.flatten(), target.flatten())
            validation_loss += valid_loss.item()

            acc_loss = acc_criterion(output.flatten(), target.flatten())
            accuracy += acc_loss.item()

    dataset_length = len(valid_dataloader)
    validation_loss /= dataset_length
    accuracy /= dataset_length

    return validation_loss, accuracy
