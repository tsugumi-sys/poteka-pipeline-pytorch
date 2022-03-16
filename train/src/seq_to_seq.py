from typing import Tuple, Union, Optional
import sys

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F

# Need to import from the parent directory to load pytorch model in evaluate directory.
sys.path.append("..")
from train.src.convlstm import ConvLSTM
from train.src.config import WeightsInitializer


class Seq2Seq(nn.Module):
    def __init__(
        self,
        num_channels: int,
        kernel_size: Union[int, Tuple],
        num_kernels: int,
        padding: Union[int, Tuple, str],
        activation: str,
        frame_size: Tuple,
        num_layers: int,
        weights_initializer: Optional[str] = WeightsInitializer.Zeros,
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
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.padding = padding
        self.activation = activation
        self.frame_size = frame_size
        self.num_layers = num_layers
        self.weights_initializer = weights_initializer

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
                weights_initializer=weights_initializer,
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
                    weights_initializer=weights_initializer,
                ),
            )

            self.sequencial.add_module(f"batchnorm{layer_idx}", nn.BatchNorm3d(num_features=num_kernels))

        # Add Convolutional layer to predict output frame
        # output shape is (batch_size, out_channels, height, width)
        # self.conv = nn.Conv2d(
        #     in_channels=num_kernels,
        #     out_channels=num_channels,
        #     kernel_size=kernel_size,
        #     padding=padding,
        # )
        self.sequencial.add_module(
            "convlstm_last",
            ConvLSTM(
                in_channels=num_kernels,
                out_channels=num_channels,
                kernel_size=kernel_size,
                padding=padding,
                activation="sigmoid",
                frame_size=frame_size,
                weights_initializer=weights_initializer,
            ),
        )

    def forward(self, X: torch.Tensor):
        # Forward propagation through all the layers
        output = self.sequencial(X)

        # Return only the last output frame
        # output = self.conv(output[:, :, -1])

        # batch_size, out_channels, height, width = output.size()

        # output = torch.reshape(output, (batch_size, out_channels, 1, height, width))

        output = output[:, :, -1, :, :]
        batch_size, out_channels, height, width = output.size()
        output = torch.reshape(output, (batch_size, out_channels, 1, height, width))
        return output


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
