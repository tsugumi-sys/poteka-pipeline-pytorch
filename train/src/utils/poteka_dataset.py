import torch
from torch.utils.data import Dataset


class PotekaDataset(Dataset):
    def __init__(self, input_tensor: torch.Tensor, label_tensor: torch.Tensor) -> None:
        super().__init__()
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor

    def __len__(self):
        return self.input_tensor.size(0)

    def __getitem__(self, idx):
        return self.input_tensor[idx, ...], self.label_tensor[idx, ...]
