from typing import Dict
from pydantic import BaseModel
import torch
import pandas as pd


class TestDataDictBase(BaseModel):
    date: str
    start: str
    input: torch.Tensor
    label: torch.Tensor
    label_df: Dict[int, pd.DataFrame]


class TestDataDict(BaseModel):
    __root__: Dict[str, TestDataDictBase]
