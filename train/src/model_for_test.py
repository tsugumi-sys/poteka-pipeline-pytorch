from typing import Tuple, Union, Optional
import torch
from torch import nn


class TestModel(nn.Module):
    def __init__(
        self,
        num_channels: int = 10,
        kernel_size: Union[int, Tuple] = 10,
        num_kernels: int = 10,
        padding: Union[int, Tuple, str] = 10,
        activation: str = "dummy_activation",
        frame_size: Tuple = (50, 50),
        num_layers: int = 10,
        weights_initializer: Optional[str] = "dummy_initializer",
        return_sequences: bool = False,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.padding = padding
        self.activation = activation
        self.frame_size = frame_size
        self.num_layers = num_layers
        self.weights_initializer = weights_initializer
        self.w = nn.parameter.Parameter(torch.ones(1))
        self.return_sequences = return_sequences

    def forward(self, X: torch.Tensor):
        if self.return_sequences is True:
            output = torch.sigmoid(X[:, :, :, :, :] + self.w)
            return output

        output = torch.sigmoid(X[:, :, -1, :, :] + self.w)
        batch_size, out_channels, height, width = output.size()
        return torch.reshape(output, (batch_size, out_channels, 1, height, width))
