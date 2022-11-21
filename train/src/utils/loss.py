import torch
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F


class RMSELoss(_Loss):
    __constants__ = ["reduction"]

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super(RMSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(F.mse_loss(input, target, reduction=self.reduction))
