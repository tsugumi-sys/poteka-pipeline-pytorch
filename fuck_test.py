import json
import numpy as np
import pandas as pd
from common.utils import load_scaled_data, rescale_arr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def json_loader(path: str):
    f = open(path)
    return json.load(f)


def data_loader(path: str, isMaxSizeLimit: bool = False, isTrain=True):
    HEIGHT, WIDTH = 50, 50
    meta_file = json_loader(path)
    meta_file_paths = meta_file["file_paths"]

    if isTrain and isinstance(meta_file_paths, list):
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

        # [TODO]
        # You may add these args to data_laoder()?
        # HEIGHT, WIDTH = 50, 50

        meta_file_paths = meta_file_paths[:10] if isMaxSizeLimit else meta_file_paths

        # [TODO]
        # If you create ndarray first, the learning is somethig wrong...
        #
        input_arrs = np.empty([len(meta_file_paths), input_batch_size, HEIGHT, WIDTH, feature_num])
        label_arrs = np.empty([len(meta_file_paths), label_batch_size, HEIGHT, WIDTH, feature_num])
        # input_arrs = []
        # label_arrs = []
        for dataset_idx, dataset_path in tqdm(enumerate(meta_file_paths), ascii=True, desc="Loading Train and Test dataset"):
            assert meta_file_paths[dataset_idx] == dataset_path
            # input_tmp_arr = np.empty([input_batch_size, HEIGHT, WIDTH, feature_num])
            # label_tmp_arr = np.empty([label_batch_size, HEIGHT, WIDTH, feature_num])
            # load input data
            for param_idx, param_name in enumerate(dataset_path.keys()):
                for batch_idx, path in enumerate(dataset_path[param_name]["input"]):
                    arr = load_scaled_data(path)  # shape: (50, 50)
                    input_arrs[dataset_idx][batch_idx][:, :, param_idx] = arr
                    # input_tmp_arr[batch_idx][:, :, param_idx] = arr
                    # for i in range(HEIGHT):
                    #     for j in range(WIDTH):
                    #         input_tmp_arr[batch_idx][i, j, param_idx] = arr[i, j]

            # load label data
            for param_idx, param_name in enumerate(dataset_path.keys()):
                for batch_idx, path in enumerate(dataset_path[param_name]["label"]):
                    arr = load_scaled_data(path)  # shape: (50, 50)
                    label_arrs[dataset_idx][batch_idx][:, :, param_idx] = arr
                    # label_tmp_arr[batch_idx][:, :, param_idx] = arr
        #             for i in range(HEIGHT):
        #                 for j in range(WIDTH):
        #                     label_tmp_arr[batch_idx][i, j, param_idx] = arr[i, j]

        #     input_arrs.append(input_tmp_arr)
        #     label_arrs.append(label_tmp_arr)

        # input_arrs = np.array(input_arrs).reshape([len(meta_file_paths), input_batch_size, HEIGHT, WIDTH, feature_num])
        # label_arrs = np.array(label_arrs).reshape([len(meta_file_paths), label_batch_size, HEIGHT, WIDTH, feature_num])
        return (input_arrs, label_arrs)


# meta_file_paths = "./data/preprocess/meta_train.json"
# dataset = data_loader(meta_file_paths, isMaxSizeLimit=True)
# X_train, y_train = dataset[0], dataset[1]

# meta_file = json_loader(meta_file_paths)
# meta_file_paths = meta_file["file_paths"]
# meta_file_paths = meta_file_paths[:10]

# for dataset_idx, dataset_path in enumerate(meta_file_paths):
#     for param_idx, param_name in enumerate(dataset_path.keys()):
#         assert len(dataset_path[param_name]["input"]) == 6
#         assert len(dataset_path[param_name]["label"]) == 1
#         # input
#         for batch_idx, path in enumerate(dataset_path[param_name]["input"]):
#             original_arr = load_scaled_data(path)
#             arr = X_train[dataset_idx][batch_idx][:, :, param_idx]
#             assert (original_arr - arr).sum() == 0

#         # label
#         for batch_idx, path in enumerate(dataset_path[param_name]["label"]):
#             original_arr = load_scaled_data(path)
#             arr = y_train[dataset_idx][batch_idx][:, :, param_idx]
#             assert (original_arr - arr).sum() == 0

arr = np.zeros([1, 6, 2, 2, 3])
arr[0][0][:, :, 0] = np.array([[1, 1], [1, 1]])
arr[0][0][:, :, 1] = np.array([[2, 2], [2, 2]])
print(arr)
print(arr[0][0][:, :, 0])
