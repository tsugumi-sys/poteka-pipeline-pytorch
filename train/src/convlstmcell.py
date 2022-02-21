from typing import Tuple, Union
import sys

import torch
from torch import nn

sys.path.append("..")
from train.src.config import DEVICE


class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        padding: Union[int, Tuple],
        activation: str,
        frame_size: Tuple,
    ) -> None:
        """[Initialize ConvLSTMCell]

        Args:
            in_channels (int): [Number of channels of input tensor.]
            out_channels (int): [Number of channels of output tensor]
            kernel_size (Union[int, Tuple]): [Size of the convolution kernel.]
            padding (padding (Union[str, Tuple]): ['same', 'valid' or (int, int)]): ['same', 'valid' or (int, int)]
            activation (str): [Activation function]
            frame_size (Tuple): [height and width]
        """
        super(ConvLSTMCell, self).__init__()

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu

        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels,
            out_channels=4 * out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        # Initialize weights for Hadamard Products.
        self.W_ci = nn.parameter.Parameter(torch.Tensor(out_channels, *frame_size)).to(DEVICE)
        self.W_co = nn.parameter.Parameter(torch.Tensor(out_channels, *frame_size)).to(DEVICE)
        self.W_cf = nn.parameter.Parameter(torch.Tensor(out_channels, *frame_size)).to(DEVICE)

    def forward(self, X: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[forward function of ConvLSTMCell]

        Args:
            X (torch.Tensor): [input data with the shape of ]
            h_prev (torch.Tensor): [previous hidden state]
            c_prev (torch.Tensor): [previous cell state]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: [current_hidden_state, current_cell_state]
        """
        conv_output = self.conv(torch.cat([X, h_prev], dim=1))

        i_conv, f_conv, c_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * c_prev)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * c_prev)

        # Current cell output (state)
        C = forget_gate * c_prev + input_gate * self.activation(c_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C)

        # Current hidden state
        H = output_gate * self.activation(C)

        return H.to(DEVICE), C.to(DEVICE)
