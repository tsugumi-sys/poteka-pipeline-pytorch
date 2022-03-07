import torch
from enum import Enum

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class WeightsInitializer(Enum):
    Zeros = "zeros"
    He = "he"
    Xavier = "xavier"
