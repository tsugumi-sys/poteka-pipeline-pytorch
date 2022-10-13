from typing import List, Dict
import torch
import numpy as np
import pandas as pd
import json

from common.config import PPOTEKACols
from train.src.config import DEVICE


def generate_dummy_test_dataset(
    input_parameter_names: List,
    observation_point_file_path: str,
    input_seq_length: int = 6,
    label_seq_length: int = 6,
    is_ob_point_label: bool = False,
) -> Dict:
    """This function creates dummy test dataset."""
    dummy_tensor = torch.ones((1, len(input_parameter_names), input_seq_length, 50, 50), dtype=torch.float, device=DEVICE)
    sample1_input_tensor = dummy_tensor.clone().detach()
    sample2_input_tensor = dummy_tensor.clone().detach()

    with open(observation_point_file_path, "r") as f:
        ob_point_data = json.load(f)

    ob_point_names = list(ob_point_data.keys())
    if is_ob_point_label:
        dummy_label_tensor = torch.zeros((1, len(input_parameter_names), label_seq_length, len(ob_point_names)))
        sample1_label_tensor = dummy_label_tensor.clone().detach()
        sample2_label_tensor = dummy_label_tensor.clone().detach()
    else:
        dummy_label_tensor = torch.zeros((1, len(input_parameter_names), label_seq_length, 50, 50))
        sample1_label_tensor = dummy_label_tensor.clone().detach()
        sample2_label_tensor = dummy_label_tensor.clone().detach()

    # change value for each input parameters
    # rain -> 0, temperature -> 1, humidity -> 0.5)
    for i in range(len(input_parameter_names)):
        val = 1 / i if i > 0 else 0
        sample1_input_tensor[:, i, ...] = val
        sample1_label_tensor[:, i, ...] = val
        sample2_input_tensor[:, i, ...] = val
        sample2_label_tensor[:, i, ...] = val

    label_dfs = {}

    for i in range(sample1_input_tensor.size()[2]):
        data = {}
        for col in PPOTEKACols.get_cols():
            data[col] = np.ones((len(ob_point_names)))
            if col == "hour-rain":
                data[col] *= 0
            elif col == "RH1":
                data[col] /= 2
        label_dfs[i] = pd.DataFrame(data, index=ob_point_names)

    test_dataset = {
        "sample1": {
            "date": "2022-01-01",
            "start": "23-20.csv",
            "input": sample1_input_tensor,
            "label": sample1_label_tensor,
            "label_df": label_dfs,
            "standarized_info": {param_name: {"mean": 0.0, "std": 1.0} for param_name in input_parameter_names},
        },
        "sample2": {
            "date": "2022-01-02",
            "start": "1-0.csv",
            "input": sample2_input_tensor,
            "label": sample2_label_tensor,
            "label_df": label_dfs,
            "standarized_info": {param_name: {"mean": 0.0, "std": 1.0} for param_name in input_parameter_names},
        },
    }

    return test_dataset
