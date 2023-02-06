from typing import Tuple

import torch
from torch import nn


class TimeSequenceReshaper(nn.Module):
    """Reshape the gridded tensor to single vector"""

    def __init__(self, output_seq_length: Tuple) -> None:
        super(TimeSequenceReshaper, self).__init__()
        self.output_seq_length = output_seq_length

    def forward(self, x):
        """Reshape a tensor that its seq_length and data are flatten.

        Args:
            x: [batch, num_channel, seq_length * data_size] -> The shape is [batch, num_channel, seq_length, data_size].
        """
        batch_size, num_channel, data_size = x.shape
        return torch.reshape(x, (batch_size, num_channel, self.output_seq_length, data_size // self.output_seq_length))
