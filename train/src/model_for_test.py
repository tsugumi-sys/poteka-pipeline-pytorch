import sys

import torch
from torch import nn

sys.path.append("..")
from train.src.config import DEVICE


class TestModel(nn.Module):
    def __init__(
        self,
        return_sequences: bool = False,
    ) -> None:
        super().__init__()
        self.w = nn.parameter.Parameter(torch.ones(1, dtype=torch.float)).to(DEVICE)
        self.return_sequences = return_sequences

    def forward(self, X: torch.Tensor):
        if self.return_sequences is True:
            output = torch.sigmoid(X[:, :, :, :, :] + self.w)
            return output

        output = torch.sigmoid(X[:, :, -1, :, :] + self.w)
        batch_size, out_channels, height, width = output.size()
        return torch.reshape(output, (batch_size, out_channels, 1, height, width))
