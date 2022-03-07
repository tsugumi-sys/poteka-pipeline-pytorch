from typing import Dict, Tuple, Union
import json

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

from common.utils import load_scaled_data, load_standard_scaled_data
from common.custom_logger import CustomLogger

logger = CustomLogger("data_loader_Logger")


def data_loader(path: str, isMaxSizeLimit: bool = False, scale_method: str = "min_max", isTrain: bool = True) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict]:
    """Data loader

    Args:
        path (str): meta file path
        isMaxSizeLimit (bool, optional): Limit data size for test. Defaults to False.
        isTrain (bool, optional): Use for train (valid) data or test data. Defaults to True means train (valid) data.

    Returns:
        (Union[Tuple[torch.Tensor, torch.Tensor], Dict]):
            if isTrain is True:
                (Tuple[torch.Tensor, torch.Tensor]): input_tensor and label_tensor.
                input_tensor shape is (sample_number, num_channels, seq_len=6, height, width)
                label_tensor shape is (sample_number, num_channels, seq_len=1, height, width)
            if isTrain is False:
                (Dict): dict of test cases.
    """
    # [TODO]
    # You may add these args to data_laoder()?
    # HEIGHT, WIDTH = 50, 50
    HEIGHT, WIDTH = 50, 50
    meta_file = json_loader(path)
    meta_file_paths = meta_file["file_paths"]

    if isTrain and isinstance(meta_file_paths, list):
        # =============================
        # meta_file_paths: List[Dict]
        # [ param1: {
        #   input: [6 paths],
        #   label: [1 paths],
        #   },
        #   param2: {
        #       ...
        #   }, ...
        # }]

        num_channels = len(meta_file_paths[0].keys())
        input_seq_length = len(meta_file_paths[0]["rain"]["input"])
        label_seq_length = len(meta_file_paths[0]["rain"]["label"])

        meta_file_paths = meta_file_paths[:100] if isMaxSizeLimit else meta_file_paths

        # [TODO]
        # Tensor shape should be (batch_size, num_channels, seq_len, height, width)
        input_tensor = torch.zeros((len(meta_file_paths), num_channels, input_seq_length, HEIGHT, WIDTH), dtype=torch.float)
        label_tensor = torch.zeros((len(meta_file_paths), num_channels, label_seq_length, HEIGHT, WIDTH), dtype=torch.float)

        for dataset_idx, dataset_path in tqdm(enumerate(meta_file_paths), ascii=True, desc="Loading Train and Valid dataset"):
            # load input data
            for param_idx, param_name in enumerate(dataset_path.keys()):
                for seq_idx, path in enumerate(dataset_path[param_name]["input"]):
                    if scale_method == "min_max":
                        numpy_arr = load_scaled_data(path)  # shape: (50, 50)
                    elif scale_method == "standard":
                        numpy_arr = load_standard_scaled_data(path)

                    if np.isnan(numpy_arr).any():
                        logger.error(f"NaN contained in {path}")

                    input_tensor[dataset_idx, param_idx, seq_idx, :, :] = torch.from_numpy(numpy_arr)

            # load label data
            for param_idx, param_name in enumerate(dataset_path.keys()):
                if scale_method == "min_max":
                    numpy_arr = load_scaled_data(dataset_path[param_name]["label"][0])  # shape: (50, 50)
                elif scale_method == "standard":
                    numpy_arr = load_standard_scaled_data(dataset_path[param_name]["label"][0])

                if np.isnan(numpy_arr).any():
                    logger.error(f"NaN contained in {path}")

                label_tensor[dataset_idx, param_idx, 0, :, :] = torch.from_numpy(numpy_arr)

        logger.info(f"Training dataset shape: {input_tensor.shape}")
        return (input_tensor, label_tensor)

    elif not isTrain or isinstance(meta_file_paths, Dict):
        if isTrain:
            logger.warning("This data is regarded as test data because the type is Dict")

        # =============================
        # meta_file_paths: Dict
        # { sample1: {
        #     date: ###,
        #     start: ###,
        #     rain: {
        #       input: [6 paths],
        #       label: [6 paths],
        #     },
        #     humidity: { input: [...]},
        #     temperature: { input: [...]},
        #     ...
        #   },
        #   sample2: {...}
        # }]
        output_data = {}
        for sample_name in tqdm(meta_file_paths.keys(), ascii=True, desc="Loading Valid dataset"):
            feature_names = [v for v in meta_file_paths[sample_name].keys() if v not in ["date", "start"]]

            num_channels = len(feature_names)
            input_seq_length = len(meta_file_paths[sample_name]["rain"]["input"])
            label_seq_length = len(meta_file_paths[sample_name]["rain"]["label"])

            input_tensor = torch.zeros((1, num_channels, input_seq_length, HEIGHT, WIDTH), dtype=torch.float)
            label_tensor = torch.zeros((1, num_channels, label_seq_length, HEIGHT, WIDTH), dtype=torch.float)

            for param_idx, param_name in enumerate(feature_names):
                for seq_idx, path in enumerate(meta_file_paths[sample_name][param_name]["input"]):
                    if scale_method == "min_max":
                        numpy_arr = load_scaled_data(path)  # shape: (50, 50)
                    elif scale_method == "standard":
                        numpy_arr = load_standard_scaled_data(path)

                    if np.isnan(numpy_arr).any():
                        logger.error(f"NaN contained in {path}")

                    input_tensor[0, param_idx, seq_idx, :, :] = torch.from_numpy(numpy_arr)

                # load label data
                for seq_idx, path in enumerate(meta_file_paths[sample_name][param_name]["label"]):
                    if scale_method == "min_max":
                        numpy_arr = load_scaled_data(path)  # shape: (50, 50)
                    elif scale_method == "standard":
                        numpy_arr = load_standard_scaled_data(path)

                    if np.isnan(numpy_arr).any():
                        logger.error(f"NaN contained in {path}")

                    label_tensor[0, param_idx, seq_idx, :, :] = torch.from_numpy(numpy_arr)

            # Load One Day data for evaluation
            label_dfs = {}
            for i in range(label_seq_length):
                df_path = meta_file_paths[sample_name]["rain"]["label"][i]
                df_path = df_path.replace("rain_image", "one_day_data").replace(".csv", ".parquet.gzip")
                df = pd.read_parquet(df_path, engine="pyarrow")
                df = df.set_index("Unnamed: 0")
                label_dfs[i] = df

            output_data[sample_name] = {
                "date": meta_file_paths[sample_name]["date"],
                "start": meta_file_paths[sample_name]["start"],
                "input": input_tensor,
                "label": label_tensor,
                "label_df": label_dfs,
            }

        return output_data


def json_loader(path: str):
    f = open(path)
    return json.load(f)


def sample_data_loader(
    train_size: int,
    valid_size: int,
    x_batch: int,
    y_batch: int,
    height: int,
    width: int,
    vector_size: int,
):
    X_train = random_normalized_data(train_size, x_batch, height, width, vector_size)
    y_train = random_normalized_data(train_size, y_batch, height, width, vector_size)
    X_valid = random_normalized_data(valid_size, x_batch, height, width, vector_size)
    y_valid = random_normalized_data(valid_size, y_batch, height, width, vector_size)

    return (X_train, y_train), (X_valid, y_valid)


def random_normalized_data(
    sample_size: int,
    batch_num: int,
    height: int,
    width: int,
    vector_size: int,
):
    arr = np.array([[np.random.rand(height, width, vector_size)] * batch_num] * sample_size)
    return arr


if __name__ == "__main__":
    meta_valid_file = "../data/preprocess/meta_valid.json"
    valid_dataset = data_loader(meta_valid_file, isTrain=False)
    print(valid_dataset)
    for sample_name in valid_dataset.keys():
        X_valid = valid_dataset[sample_name]["input"]
        y_valid = valid_dataset[sample_name]["label"]
        print(X_valid.shape, y_valid.shape)
