from typing import List, Dict
import torch
import numpy as np
import pandas as pd
import json

from common.config import PPOTEKACols
from train.src.config import DEVICE


def generate_dummy_test_dataset(input_parameter_names: List, observation_point_file_path: str) -> Dict:
    """This function creates dummy test dataset."""
    dummy_tensor = torch.ones((5, len(input_parameter_names), 6, 50, 50), dtype=torch.float, device=DEVICE)
    sample1_input_tensor = dummy_tensor.clone().detach()
    sample1_label_tensor = dummy_tensor.clone().detach()
    sample2_input_tensor = dummy_tensor.clone().detach()
    sample2_label_tensor = dummy_tensor.clone().detach()
    # change value for each input parameters
    # rain -> 0, temperature -> 1, humidity -> 0.5)
    for i in range(len(input_parameter_names)):
        val = 1 / i if i > 0 else 0
        sample1_input_tensor[:, i, :, :, :] = val
        sample1_label_tensor[:, i, :, :, :] = val
        sample2_input_tensor[:, i, :, :, :] = val
        sample2_label_tensor[:, i, :, :, :] = val
    label_dfs = {}
    with open(observation_point_file_path, "r") as f:
        ob_point_data = json.load(f)
    ob_point_names = list(ob_point_data.keys())
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
            "standarize_info": {"rain": {"mean": 1.0, "std": 0.1}, "temperature": {"mean": 2.0, "std": 0.2}, "humidity": {"mean": 3.0, "std": 0.3}},
        },
        "sample2": {
            "date": "2022-01-02",
            "start": "1-0.csv",
            "input": sample2_input_tensor,
            "label": sample2_label_tensor,
            "label_df": label_dfs,
            "standarize_info": {"rain": {"mean": 1.0, "std": 0.1}, "temperature": {"mean": 2.0, "std": 0.2}, "humidity": {"mean": 3.0, "std": 0.3}},
        },
    }

    return test_dataset
