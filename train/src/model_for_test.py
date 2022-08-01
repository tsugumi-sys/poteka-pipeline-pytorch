import sys
from typing import Iterator

import torch
from torch import nn

sys.path.append("..")
from train.src.config import DEVICE


class TestModel(nn.Module):
    def __init__(
        self,
        return_sequences: bool = False,
    ) -> None:
        super(TestModel, self).__init__()
        self.return_sequences = return_sequences
        # https://github.com/pytorch/pytorch/issues/50402
        # Set device not only torch.ones but also nn.parameter.Parameter
        self.w1 = nn.parameter.Parameter(torch.ones(1, dtype=torch.float, device=DEVICE)).to(DEVICE)
        self.w2 = nn.parameter.Parameter(torch.ones(1, dtype=torch.float, device=DEVICE)).to(DEVICE)

    def forward(self, X: torch.Tensor):
        if self.return_sequences is True:
            output = torch.sigmoid(X * self.w1 + self.w2)
            return output

        if X.dim() == 5:
            output = torch.sigmoid(X[:, :, -1, :, :] * self.w1 + self.w2)
            batch_size, out_channels, height, width = output.size()
            return torch.reshape(output, (batch_size, out_channels, 1, height, width))
        elif X.dim() == 4:
            output = torch.sigmoid(X[:, :, -1, :] * self.w1 + self.w2)
            batch_size, out_channels, ob_point_count = output.size()
            return torch.reshape(output, (batch_size, out_channels, 1, ob_point_count))

    def parameters(self, recurse: bool = True) -> Iterator[nn.parameter.Parameter]:
        return iter((self.w1, self.w2))
