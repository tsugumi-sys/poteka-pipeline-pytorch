import logging
from typing import Dict, OrderedDict, Tuple, Union
import json
from collections import OrderedDict as ordered_dict

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

from common.utils import load_scaled_data, load_standard_scaled_data
from common.custom_logger import CustomLogger
from common.config import WEATHER_PARAMS, GridSize, MinMaxScalingValue, PPOTEKACols, ScalingMethod

logger = CustomLogger("data_loader_Logger", level=logging.DEBUG)


def train_data_loader(
    path: str,
    isMaxSizeLimit: bool = False,
    scaling_method: str = "min_max",
    debug_mode: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not ScalingMethod.is_valid(scaling_method):
        raise ValueError("Invalid scaling method")
    # [TODO]
    # You may add these args to data_laoder()?
    # HEIGHT, WIDTH = 50, 50
    HEIGHT, WIDTH = GridSize.HEIGHT, GridSize.WIDTH
    meta_file = json_loader(path)
    meta_file_paths = meta_file["file_paths"]
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
    logger.info(f"Scaling method: {scaling_method}")
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
        # input data is scaled in 2 ways
        # 1. MinMax: scaled to [0, 1]
        # 2. MinMaxStandard: scaleed to [0, 1] first, then scaled with standarization
        for param_idx, param_name in enumerate(dataset_path.keys()):
            for seq_idx, path in enumerate(dataset_path[param_name]["input"]):
                # if param_name == WEATHER_PARAMS.RAIN.value:
                #     numpy_arr = load_scaled_data(path)
                # else:
                #     if scaling_method == ScalingMethod.MinMax.value:
                #         numpy_arr = load_scaled_data(path)  # shape: (50, 50)
                #     elif scaling_method == ScalingMethod.Standard.value:
                #         numpy_arr = load_standard_scaled_data(path)
                #     elif scaling_method == ScalingMethod.MinMaxStandard.value:
                #         numpy_arr = load_scaled_data(path)
                #         numpy_arr = (numpy_arr - numpy_arr.mean()) / numpy_arr.std()
                numpy_arr = load_scaled_data(path)
                if debug_mode is True:
                    logger.warning(
                        f"Input Tensor, {scaling_method}, parameter: {param_name}, max: {numpy_arr.max():.5f}, min: {numpy_arr.min():.5f}, mean: {numpy_arr.mean():.5f}, std: {numpy_arr.std():.5f}"  # noqa: E501
                    )
                if np.isnan(numpy_arr).any():
                    logger.error(f"NaN contained in {path}")
                    logger.error(numpy_arr)
                input_tensor[dataset_idx, param_idx, seq_idx, :, :] = torch.from_numpy(numpy_arr)
            if scaling_method == ScalingMethod.Standard.value or scaling_method == ScalingMethod.MinMaxStandard.value:
                means = torch.mean(input_tensor[dataset_idx, param_idx, :, :, :])
                stds = torch.std(input_tensor[dataset_idx, param_idx, :, :, :])
                input_tensor[dataset_idx, param_idx, :, :, :] = (input_tensor[dataset_idx, param_idx, :, :, :] - means) / stds
        # load label data
        # label data is scaled to [0, 1]
        for param_idx, param_name in enumerate(dataset_path.keys()):
            numpy_arr = load_scaled_data(dataset_path[param_name]["label"][0])  # shape: (50, 50)
            if debug_mode is True:
                logger.warning(
                    f"Label Tensor, min_max, parameter: {param_name}, max: {numpy_arr.max():.5f}, min: {numpy_arr.min():.5f}, mean: {numpy_arr.mean():.5f}, std: {numpy_arr.std():.5f}"  # noqa: E501
                )
            if np.isnan(numpy_arr).any():
                logger.error(f"NaN contained in {path}")
            label_tensor[dataset_idx, param_idx, 0, :, :] = torch.from_numpy(numpy_arr)
    logger.info(f"Training dataset shape: {input_tensor.shape}")
    return (input_tensor, label_tensor)


def test_data_loader(
    path: str,
    scaling_method: str = "min_max",
    debug_mode: bool = False,
    use_dummy_data: bool = False,
) -> Tuple[Dict, OrderedDict]:
    if not ScalingMethod.is_valid(scaling_method):
        raise ValueError("Invalid scaling method")
    # [TODO]
    # You may add these args to data_laoder()?
    # HEIGHT, WIDTH = 50, 50
    HEIGHT, WIDTH = GridSize.HEIGHT, GridSize.WIDTH
    meta_file = json_loader(path)
    meta_file_paths = meta_file["file_paths"]
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
    logger.info(f"Scaling method: {scaling_method}")
    output_data = {}
    features_dict = ordered_dict()
    for sample_name in tqdm(meta_file_paths.keys(), ascii=True, desc="Loading Valid dataset"):
        feature_names = [v for v in meta_file_paths[sample_name].keys() if v not in ["date", "start"]]
        if bool(features_dict) is False:
            for idx, name in enumerate(feature_names):
                features_dict[idx] = name
        num_channels = len(feature_names)
        input_seq_length = len(meta_file_paths[sample_name]["rain"]["input"])
        label_seq_length = len(meta_file_paths[sample_name]["rain"]["label"])
        input_tensor = torch.zeros((1, num_channels, input_seq_length, HEIGHT, WIDTH), dtype=torch.float)
        label_tensor = torch.zeros((1, num_channels, label_seq_length, HEIGHT, WIDTH), dtype=torch.float)
        standarize_info = {}
        # load input data
        # input data is scaled in 2 ways
        # 1. MinMax: scaled to [0, 1]
        # 2. MinMaxStandard: scaleed to [0, 1] first, then scaled with standarization
        for param_idx, param_name in enumerate(feature_names):
            for seq_idx, path in enumerate(meta_file_paths[sample_name][param_name]["input"]):
                # if param_name == WEATHER_PARAMS.RAIN.value:
                #     numpy_arr = load_scaled_data(path)
                # else:
                #     if scaling_method == ScalingMethod.MinMax.value:
                #         numpy_arr = load_scaled_data(path)  # shape: (50, 50)
                #     elif scaling_method == ScalingMethod.Standard.value:
                #         numpy_arr = load_standard_scaled_data(path)
                #     elif scaling_method == ScalingMethod.MinMaxStandard.value:
                #         numpy_arr = load_scaled_data(path)
                #         numpy_arr = (numpy_arr - numpy_arr.mean()) / numpy_arr.std()
                numpy_arr = load_scaled_data(path)
                if debug_mode is True:
                    logger.warning(
                        f"Input Tensor, {scaling_method}, parameter: {param_name}, max: {numpy_arr.max():.5f}, min: {numpy_arr.min():.5f}, mean: {numpy_arr.mean():.5f}, std: {numpy_arr.std():.5f}"  # noqa: E501
                    )
                if np.isnan(numpy_arr).any():
                    logger.error(f"NaN contained in {path}")
                input_tensor[0, param_idx, seq_idx, :, :] = torch.from_numpy(numpy_arr)
            if scaling_method == ScalingMethod.Standard.value or scaling_method == ScalingMethod.MinMaxStandard.value:
                means = torch.mean(input_tensor[0, param_idx, :, :, :])
                stds = torch.std(input_tensor[0, param_idx, :, :, :])
                input_tensor[0, param_idx, :, :, :] = (input_tensor[0, param_idx, :, :, :] - means) / stds
                standarize_info[param_name] = {"mean": means, "std": stds}
            # load label data
            # label data is scaled to [0, 1]
            for seq_idx, path in enumerate(meta_file_paths[sample_name][param_name]["label"]):
                numpy_arr = load_scaled_data(path)  # shape: (50, 50)
                if debug_mode is True:
                    logger.warning(
                        f"Label Tensor, min_max, parameter: {param_name}, max: {numpy_arr.max():.5f}, min: {numpy_arr.min():.5f}, mean: {numpy_arr.mean():.5f}, std: {numpy_arr.std():.5f}"  # noqa: E501
                    )
                if np.isnan(numpy_arr).any():
                    logger.error(f"NaN contained in {path}")
                label_tensor[0, param_idx, seq_idx, :, :] = torch.from_numpy(numpy_arr)
        # Load One Day data for evaluation
        label_dfs = {}
        if use_dummy_data:
            for i in range(label_seq_length):
                data = {}
                for col in PPOTEKACols.get_cols():
                    min_val, max_val = MinMaxScalingValue.get_minmax_values_by_ppoteka_cols(col)
                    data[col] = np.random.uniform(low=min_val, high=max_val, size=(10))
                label_dfs[i] = pd.DataFrame(data)

        else:
            # If you use dummy data, parqet files of one_data_data don't exist.
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
            "standarize_info": standarize_info,
        }

    return output_data, features_dict


# NOTE: Deprecated!
def data_loader(
    path: str,
    isMaxSizeLimit: bool = False,
    scaling_method: str = "min_max",
    isTrain: bool = True,
    debug_mode: bool = False,
    use_dummy_data: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[Dict, OrderedDict]]:
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
    if not ScalingMethod.is_valid(scaling_method):
        raise ValueError("Invalid scaling method")
    # [TODO]
    # You may add these args to data_laoder()?
    # HEIGHT, WIDTH = 50, 50
    HEIGHT, WIDTH = GridSize.HEIGHT, GridSize.WIDTH
    meta_file = json_loader(path)
    meta_file_paths = meta_file["file_paths"]

    logger.info(f"Scaling method: {scaling_method}")

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
                    if param_name == WEATHER_PARAMS.RAIN.value:
                        numpy_arr = load_scaled_data(path)
                    else:
                        if scaling_method == ScalingMethod.MinMax.value:
                            numpy_arr = load_scaled_data(path)  # shape: (50, 50)
                        elif scaling_method == ScalingMethod.Standard.value:
                            numpy_arr = load_standard_scaled_data(path)
                        elif scaling_method == ScalingMethod.MinMaxStandard.value:
                            numpy_arr = load_scaled_data(path)
                            numpy_arr = (numpy_arr - numpy_arr.mean()) / numpy_arr.std()

                    if debug_mode is True:
                        logger.warning(
                            f"Input Tensor, {scaling_method}, parameter: {param_name}, max: {numpy_arr.max():.5f}, min: {numpy_arr.min():.5f}, mean: {numpy_arr.mean():.5f}, std: {numpy_arr.std():.5f}"  # noqa: E501
                        )

                    if np.isnan(numpy_arr).any():
                        logger.error(f"NaN contained in {path}")
                        logger.error(numpy_arr)

                    input_tensor[dataset_idx, param_idx, seq_idx, :, :] = torch.from_numpy(numpy_arr)

            # load label data
            for param_idx, param_name in enumerate(dataset_path.keys()):
                numpy_arr = load_scaled_data(dataset_path[param_name]["label"][0])  # shape: (50, 50)

                if debug_mode is True:
                    logger.warning(
                        f"Label Tensor, min_max, parameter: {param_name}, max: {numpy_arr.max():.5f}, min: {numpy_arr.min():.5f}, mean: {numpy_arr.mean():.5f}, std: {numpy_arr.std():.5f}"  # noqa: E501
                    )

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
        features_dict = ordered_dict()
        for sample_name in tqdm(meta_file_paths.keys(), ascii=True, desc="Loading Valid dataset"):
            feature_names = [v for v in meta_file_paths[sample_name].keys() if v not in ["date", "start"]]
            if bool(features_dict) is False:
                for idx, name in enumerate(feature_names):
                    features_dict[idx] = name

            num_channels = len(feature_names)
            input_seq_length = len(meta_file_paths[sample_name]["rain"]["input"])
            label_seq_length = len(meta_file_paths[sample_name]["rain"]["label"])

            input_tensor = torch.zeros((1, num_channels, input_seq_length, HEIGHT, WIDTH), dtype=torch.float)
            label_tensor = torch.zeros((1, num_channels, label_seq_length, HEIGHT, WIDTH), dtype=torch.float)

            for param_idx, param_name in enumerate(feature_names):
                for seq_idx, path in enumerate(meta_file_paths[sample_name][param_name]["input"]):
                    if param_name == WEATHER_PARAMS.RAIN.value:
                        numpy_arr = load_scaled_data(path)
                    else:
                        if scaling_method == ScalingMethod.MinMax.value:
                            numpy_arr = load_scaled_data(path)  # shape: (50, 50)
                        elif scaling_method == ScalingMethod.Standard.value:
                            numpy_arr = load_standard_scaled_data(path)
                        elif scaling_method == ScalingMethod.MinMaxStandard.value:
                            numpy_arr = load_scaled_data(path)
                            numpy_arr = (numpy_arr - numpy_arr.mean()) / numpy_arr.std()

                    if debug_mode is True:
                        logger.warning(
                            f"Input Tensor, {scaling_method}, parameter: {param_name}, max: {numpy_arr.max():.5f}, min: {numpy_arr.min():.5f}, mean: {numpy_arr.mean():.5f}, std: {numpy_arr.std():.5f}"  # noqa: E501
                        )

                    if np.isnan(numpy_arr).any():
                        logger.error(f"NaN contained in {path}")

                    input_tensor[0, param_idx, seq_idx, :, :] = torch.from_numpy(numpy_arr)

                # load label data
                for seq_idx, path in enumerate(meta_file_paths[sample_name][param_name]["label"]):
                    numpy_arr = load_scaled_data(path)  # shape: (50, 50)

                    if debug_mode is True:
                        logger.warning(
                            f"Label Tensor, min_max, parameter: {param_name}, max: {numpy_arr.max():.5f}, min: {numpy_arr.min():.5f}, mean: {numpy_arr.mean():.5f}, std: {numpy_arr.std():.5f}"  # noqa: E501
                        )

                    if np.isnan(numpy_arr).any():
                        logger.error(f"NaN contained in {path}")

                    label_tensor[0, param_idx, seq_idx, :, :] = torch.from_numpy(numpy_arr)

            # Load One Day data for evaluation
            label_dfs = {}
            if use_dummy_data:
                for i in range(label_seq_length):
                    data = {}
                    for col in PPOTEKACols.get_cols():
                        min_val, max_val = MinMaxScalingValue.get_minmax_values_by_ppoteka_cols(col)
                        data[col] = np.random.uniform(low=min_val, high=max_val, size=(10))
                    label_dfs[i] = pd.DataFrame(data)

            else:
                # If you use dummy data, parqet files of one_data_data don't exist.
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

        return output_data, features_dict


def dummy_data_loader(meta_file_path: str, scaling_method: str = "min_max") -> Tuple[Dict, Dict]:
    """Load dummy data for evaluation
    In testing, dummy data is used and evaluate.src.predict.pred_obervation_point_values is skipped.
    So not needed to load one day data for evaluation.

    Args:
        meta_file_path (str): meta file path
        scaling_method (str, optional): scaling method. Defaults to "min_max".

    Returns:
        Tuple[Dict, Dict]: (output_data, features_dict)
            output_data: {sample1: {date:, start: str, input: torch.Tensor, label: torch.Tensor}, ...}
    """


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
