"""
This model's output is [1, num_channels, num_sequences, ob_point_count]
 ob_point_count: Number of P-POTEKA observation points.
"""

from typing import Tuple, Union, Optional
import sys

import torch
from torch import nn

# Need to import from the parent directory to load pytorch model in evaluate directory.
sys.path.append("..")
from train.src.convlstm import ConvLSTM  # noqa: E402
from train.src.config import WeightsInitializer  # noqa: E402
from train.src.time_sequence_reshaper import TimeSequenceReshaper  # noqa: E402


class OBPointSeq2Seq(nn.Module):
    def __init__(
        self,
        num_channels: int,
        ob_point_count: int,
        kernel_size: Union[int, Tuple],
        num_kernels: int,
        padding: Union[int, Tuple, str],
        activation: str,
        frame_size: Tuple,
        num_layers: int,
        input_seq_length: int,
        prediction_seq_length: int,
        out_channels: Optional[int] = None,
        weights_initializer: Optional[str] = WeightsInitializer.Zeros.value,
        return_sequences: bool = False,
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
            input_seq_length (int): Number of time length per a dataset of input.
            prediction_seq_length (int): Number of predicton time length (if interval is 10min, prediction_length=6 means 1h prediction_length).
        """
        super(OBPointSeq2Seq, self).__init__()
        self.num_channels = num_channels
        self.ob_point_count = ob_point_count
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.padding = padding
        self.activation = activation
        self.frame_size = frame_size
        self.num_layers = num_layers
        self.input_seq_length = input_seq_length
        self.prediction_seq_length = prediction_seq_length
        self.out_channels = out_channels
        self.weights_initializer = weights_initializer
        self.return_sequences = return_sequences

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
        self.sequencial.add_module("bathcnorm0", nn.BatchNorm3d(num_features=num_kernels))
        self.sequencial.add_module(
            "convlstm2",
            ConvLSTM(
                in_channels=num_kernels,
                out_channels=num_channels if out_channels is None else out_channels,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation,
                frame_size=frame_size,
                weights_initializer=weights_initializer,
            ),
        )

        self.sequencial.add_module("bathcnorm1", nn.BatchNorm3d(num_features=num_channels if out_channels is None else out_channels))
        self.sequencial.add_module("maxpooling2d_1", nn.MaxPool3d(kernel_size=(1, 2, 2), padding=0))  # (..., 50, 50) -> (..., 25, 25)
        maxpooled_grid_size = (frame_size[0] // 2) * (frame_size[1] // 2)
        # TODO: Add custom layer to extract ob point values from the tensor.
        self.sequencial.add_module("flatten", nn.Flatten(start_dim=2))
        if self.prediction_seq_length < self.input_seq_length:
            self.sequencial.add_module(
                "dense0", nn.Linear(in_features=self.input_seq_length * 25 * 25, out_features=self.prediction_seq_length * maxpooled_grid_size,),
            )
        self.sequencial.add_module("reshape_time_sequence", TimeSequenceReshaper(output_seq_length=self.prediction_seq_length))
        self.sequencial.add_module("dense", nn.Linear(in_features=maxpooled_grid_size, out_features=self.ob_point_count))
        self.sequencial.add_module("sigmoid", nn.Sigmoid())

    def forward(self, X: torch.Tensor):
        # Forward propagation through all the layers
        output = self.sequencial(X)

        if self.return_sequences is True:
            return output

        output = output[:, :, -1, :]
        batch_size, out_channels, _ = output.size()
        output = torch.reshape(output, (batch_size, out_channels, 1, self.ob_point_count))
        return output
