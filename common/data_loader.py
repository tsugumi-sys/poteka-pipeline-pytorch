from typing import Dict
import json
from tqdm import tqdm
import pandas as pd

import numpy as np
from common.utils import load_scaled_data
from common.custom_logger import CustomLogger

logger = CustomLogger("data_loader_Logger")


def data_loader(path: str, isMaxSizeLimit: bool = False, isTrain=True):
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

        feature_num = len(meta_file_paths[0].keys())
        input_batch_size = len(meta_file_paths[0]["rain"]["input"])
        label_batch_size = len(meta_file_paths[0]["rain"]["label"])

        meta_file_paths = meta_file_paths[:10] if isMaxSizeLimit else meta_file_paths

        # [TODO]
        # if label arrs shape is (n, 1, 50, 50, feature_num), learning is failed.
        # shape should be (n, 50, 50, feature_num)
        input_arrs = np.zeros([len(meta_file_paths), input_batch_size, HEIGHT, WIDTH, feature_num])
        label_arrs = np.zeros([len(meta_file_paths), HEIGHT, WIDTH, feature_num])
        for dataset_idx, dataset_path in tqdm(enumerate(meta_file_paths), ascii=True, desc="Loading Train and Test dataset"):
            # load input data
            for param_idx, param_name in enumerate(dataset_path.keys()):
                for batch_idx, path in enumerate(dataset_path[param_name]["input"]):
                    arr = load_scaled_data(path)  # shape: (50, 50)
                    input_arrs[dataset_idx][batch_idx][:, :, param_idx] = arr

            # load label data
            for param_idx, param_name in enumerate(dataset_path.keys()):
                arr = load_scaled_data(dataset_path[param_name]["label"][0])  # shape: (50, 50)
                label_arrs[dataset_idx][:, :, param_idx] = arr

        logger.info(f"Training dataset shape: {input_arrs.shape}")
        return (input_arrs, label_arrs)

    elif isTrain == False or isinstance(meta_file_paths, Dict):
        if isTrain:
            logger.warning("This data is regarded as Valid data because the type is Dict")

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

            feature_num = len(feature_names)
            input_batch_size = len(meta_file_paths[sample_name]["rain"]["input"])
            label_batch_size = len(meta_file_paths[sample_name]["rain"]["label"])

            input_arrs = np.zeros([1, input_batch_size, HEIGHT, WIDTH, feature_num])
            label_arrs = np.zeros([1, label_batch_size, HEIGHT, WIDTH, feature_num])

            for param_idx, param_name in enumerate(feature_names):
                for batch_idx, path in enumerate(meta_file_paths[sample_name][param_name]["input"]):
                    arr = load_scaled_data(path)
                    input_arrs[0][batch_idx][:, :, param_idx] = arr

                # load label data
                for batch_idx, path in enumerate(meta_file_paths[sample_name][param_name]["label"]):
                    arr = load_scaled_data(path)
                    label_arrs[0][batch_idx][:, :, param_idx] = arr

            # Load One Day data for evaluation
            label_dfs = {}
            for i in range(label_batch_size):
                df_path = meta_file_paths[sample_name]["rain"]["label"][i]
                df_path = df_path.replace("rain_image", "one_day_data").replace(".csv", ".parquet.gzip")
                df = pd.read_parquet(df_path, engine="pyarrow")
                label_dfs[i] = df[["hour-rain"]]

            output_data[sample_name] = {
                "date": meta_file_paths[sample_name]["date"],
                "start": meta_file_paths[sample_name]["start"],
                "input": input_arrs,
                "label": label_arrs,
                "label_df": label_dfs,
            }

        return output_data


def json_loader(path: str):
    f = open(path)
    return json.load(f)


def sample_data_loader(
    train_size: int,
    test_size: int,
    x_batch: int,
    y_batch: int,
    height: int,
    width: int,
    vector_size: int,
):
    X_train = random_normalized_data(train_size, x_batch, height, width, vector_size)
    y_train = random_normalized_data(train_size, y_batch, height, width, vector_size)
    X_test = random_normalized_data(test_size, x_batch, height, width, vector_size)
    y_test = random_normalized_data(test_size, y_batch, height, width, vector_size)

    return (X_train, y_train), (X_test, y_test)


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
