import numpy as np
import json
import sys
import os


sys.path.append("..")
from common.utils import load_scaled_data


def data_loader(path: str, isLimit: bool = False):
    input_arrs, label_arrs = [], []
    meta_file = json_loader(path)
    meta_file_paths = meta_file["file_paths"]
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

    # [TODO]
    # You may add these args to data_laoder()?
    HEIGHT, WIDTH = 50, 50

    meta_file_paths = meta_file_paths[:10] if isLimit else meta_file_paths
    for dataset_path in meta_file_paths:
        sub_input_arr = np.empty([input_batch_size, HEIGHT, WIDTH, feature_num])
        sub_label_arr = np.empty([1, HEIGHT, WIDTH, feature_num])

        # load input data
        for param_idx, param_name in enumerate(dataset_path.keys()):
            for batch_idx, path in enumerate(dataset_path[param_name]["input"]):
                arr = load_scaled_data(path)  # shape: (50, 50)

                # [TODO]
                # You may can do this faster?
                for i in range(HEIGHT):
                    for j in range(WIDTH):
                        sub_input_arr[batch_idx][i, j, param_idx] = arr[i, j]

        # load label data
        for param_idx, param_name in enumerate(dataset_path.keys()):
            for batch_idx, path in enumerate(dataset_path[param_name]["label"]):
                arr = load_scaled_data(path)  # shape: (50, 50)

                # [TODO]
                # You may can do this faster?
                for i in range(HEIGHT):
                    for j in range(WIDTH):
                        sub_label_arr[batch_idx][i, j, param_idx] = arr[i, j]

        input_arrs.append(sub_input_arr)
        label_arrs.append(sub_label_arr)

    input_arrs = np.array(input_arrs).reshape([len(input_arrs), input_batch_size, HEIGHT, WIDTH, feature_num])
    label_arrs = np.array(label_arrs).reshape(len(label_arrs), HEIGHT, WIDTH, feature_num)
    return (input_arrs, label_arrs)


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
    X_test = random_normalized_data(test_size, 1, height, width, vector_size)
    y_test = random_normalized_data(test_size, 1, height, width, vector_size)

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
