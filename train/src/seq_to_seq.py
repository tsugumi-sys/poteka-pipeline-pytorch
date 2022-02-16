from typing import Tuple, Union

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F

from src.convlstm import ConvLSTM


class Seq2Seq(nn.Module):
    def __init__(
        self,
        num_channels: int,
        kernel_size: Union[int, Tuple],
        num_kernels: int,
        padding: Union[str, Tuple],
        activation: str,
        frame_size: Tuple,
        num_layers: int,
    ) -> None:
        """Initialize SeqtoSeq

        Args:
            num_channels (int): [Number of input channels]
            kernel_size (int): [kernel size]
            num_kernels (int): [Number of kernels]
            padding (Union[str, Tuple]): ['same', 'valid' or (int, int)]
            activation (str): [the name of activation function]
            frame_size (Tuple): [height and width]
            num_layers (int): [the number of layers]
        """
        super(Seq2Seq, self).__init__()

        self.sequencial = nn.Sequential()

        # Add first layer (Different in_channels than the rest)
        self.sequencial.add_module(
            "convlstm1",
            ConvLSTM(
                in_channels=num_channels,
                out_channels=num_kernels,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation,
                frame_size=frame_size,
            ),
        )

        self.sequencial.add_module("bathcnorm1", nn.BatchNorm3d(num_features=num_kernels))

        # Add the rest of the layers
        for layer_idx in range(2, num_layers + 1):
            self.sequencial.add_module(
                f"convlstm{layer_idx}",
                ConvLSTM(
                    in_channels=num_kernels,
                    out_channels=num_kernels,
                    kernel_size=kernel_size,
                    padding=padding,
                    activation=activation,
                    frame_size=frame_size,
                ),
            )

            self.sequencial.add_module(f"batchnorm{layer_idx}", nn.BatchNorm3d(num_features=num_kernels))

        # Add Convolutional layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=num_kernels,
            out_channels=num_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, X: torch.Tensor):
        # Forward propagation through all the layers
        output = self.sequencial(X)

        # Return only the last output frame
        output = self.conv(output[:, :, -1])

        return nn.Sigmoid()(output)


class PotekaDataset(Dataset):
    def __init__(self, input_tensor: torch.Tensor, label_tensor: torch.Tensor) -> None:
        super().__init__()
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor

    def __len__(self):
        return self.input_tensor.size(0)

    def __getitem__(self, idx):
        return self.input_tensor[idx, :, :, :, :], self.label_tensor[idx, :, :, :, :]


class RMSELoss(_Loss):
    __constants__ = ["reduction"]

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super(RMSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(F.mse_loss(input, target, reduction=self.reduction))
