import os
import sys
import logging

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import mlflow
from src.create_image import save_rain_image

sys.path.append("..")
from common.utils import rescale_arr, timestep_csv_names

logger = logging.getLogger(__name__)


def save_parquet(arr, save_path: str) -> None:
    grid_lon, grid_lat = np.round(np.linspace(120.90, 121.150, 50), 3), np.round(np.linspace(14.350, 14.760, 50), 3)
    df = pd.DataFrame(arr, index=np.flip(grid_lat), columns=grid_lon)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    df.to_parquet(
        path=save_path,
        engine="pyarrow",
        compression="gzip",
    )


def create_prediction(model, valid_dataset, downstream_directory: str, preprocess_delta: int):
    # valid_dataset: Dict
    # { sample1: {
    #     date: ###,
    #     start: ###,
    #     input: {ndarray.shape(1, 6, 50, 50, feature_num)},
    #     label: {ndarray.shape(1, 6, 50, 50, feature_num)},
    #  },
    #  sample2: {...}
    # }
    HEIGHT, WIDTH = 50, 50

    _time_step_csvnames = timestep_csv_names(delta=preprocess_delta)

    results = {}
    for sample_name in valid_dataset.keys():
        logger.info(f"Evaluationg {sample_name}")
        X_valid = valid_dataset[sample_name]["input"]
        y_valid = valid_dataset[sample_name]["label"]

        feature_num = X_valid.shape[-1]
        input_batch_size = X_valid.shape[1]

        start = valid_dataset[sample_name]["start"]
        date = valid_dataset[sample_name]["date"]
        start_idx = _time_step_csvnames.index(start)
        start = start.replace(".csv", "")

        formatted_sample_name = f"{sample_name}_{date}_{start}start"
        save_dir = os.path.join(downstream_directory, formatted_sample_name)
        os.makedirs(save_dir, exist_ok=True)

        rmses = []

        # Copy X_valid because X_valid is re-used after the normal prediction.
        _X_valid = X_valid.copy()
        for t in range(_X_valid.shape[1]):
            preds = model.predict(_X_valid)
            preds = normalize_prediction(sample_name, preds)
            rain_arr = preds[0][:, :, 0]
            label_rain_arr = y_valid[0][t][:, :, 0]

            scaled_pred_arr = rescale_arr(0, 100, rain_arr)
            scaled_label_arr = rescale_arr(0, 100, label_rain_arr)

            rmse = mean_squared_error(
                np.ravel(scaled_label_arr),
                np.ravel(scaled_pred_arr),
                squared=False,
            )
            rmses.append(rmse)
            mlflow.log_metric(
                key=formatted_sample_name,
                value=rmse,
                step=t,
            )

            _X_valid = np.append(_X_valid[0][1:], preds, axis=0).reshape(1, input_batch_size, HEIGHT, WIDTH, feature_num)

            time_step_name = _time_step_csvnames[start_idx + t + 6].replace(".csv", "")
            save_rain_image(scaled_pred_arr, save_dir + f"/{time_step_name}.png")
            save_parquet(scaled_pred_arr, save_dir + f"/{time_step_name}.parquet.gzip")

        # [EXPERIMENTS]
        # Sequential prediction
        if sample_name == "sample4" or sample_name == "sample8":
            save_dir_name = f"{sample_name}_seq_pred"
            save_dir = os.path.join(downstream_directory, save_dir_name)
            os.makedirs(save_dir, exist_ok=True)

            for t in range(X_valid.shape[1]):
                preds = model.predict(X_valid)
                preds = normalize_prediction(sample_name, preds)
                rain_arr = preds[0][:, :, 0]
                label_rain_arr = y_valid[0][t][:, :, 0]

                scaled_pred_arr = rescale_arr(0, 100, rain_arr)
                scaled_label_arr = rescale_arr(0, 100, label_rain_arr)

                rmse = mean_squared_error(
                    np.ravel(scaled_label_arr),
                    np.ravel(scaled_pred_arr),
                    squared=False,
                )

                mlflow.log_metric(
                    key=save_dir_name,
                    value=rmse,
                    step=t,
                )

                X_valid = np.append(X_valid[0][1:], [y_valid[0][t]], axis=0).reshape(1, input_batch_size, HEIGHT, WIDTH, feature_num)

                time_step_name = _time_step_csvnames[start_idx + t + 6].replace(".csv", "")
                save_rain_image(scaled_pred_arr, save_dir + f"/{time_step_name}.png")
                save_parquet(scaled_pred_arr, save_dir + f"/{time_step_name}.parquet.gzip")

        results[formatted_sample_name] = sum(rmses) / len(rmses)

    return results


def normalize_prediction(sample_name, pred_arr):
    if pred_arr.max() > 1 or pred_arr.min() < 0:
        logger.warning(f"The predictions in {sample_name} contains more 1 or less 0 value. Autoscaleing is applyed.")

    pred_arr = np.where(pred_arr > 1, 1, pred_arr)
    pred_arr = np.where(pred_arr < 0, 0, pred_arr)
    return pred_arr
