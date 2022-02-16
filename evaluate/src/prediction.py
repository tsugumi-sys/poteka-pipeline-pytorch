import os
import sys
import logging
import itertools

import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import mlflow

from src.create_image import save_rain_image, all_cases_plot, sample_plot, casetype_plot

sys.path.append("..")
from common.utils import rescale_tensor, timestep_csv_names
from common import schemas

logger = logging.getLogger("Evaluate_Logger")


def save_parquet(tensor: torch.Tensor, save_path: str) -> None:
    grid_lon, grid_lat = np.round(np.linspace(120.90, 121.150, 50), 3), np.round(np.linspace(14.350, 14.760, 50), 3)
    df = pd.DataFrame(tensor.numpy(), index=np.flip(grid_lat), columns=grid_lon)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    df.to_parquet(
        path=save_path,
        engine="pyarrow",
        compression="gzip",
    )


def pred_obervation_point_values(rain_tensor: torch.Tensor) -> pd.DataFrame:
    """Prediction value near the observation points

    Args:
        rain_tensor (torch.Tensor): The shape is (HEIGHT, WIDTH)

    Returns:
        (pd.DataFrame): DataFrame that has `Pred_Value` column and `observation point name` index.
    """
    HEIGHT, WIDTH = 50, 50
    grid_lons = np.round(np.linspace(120.90, 121.150, WIDTH), decimals=3).tolist()
    grid_lats = np.round(np.linspace(14.350, 14.760, HEIGHT), decimals=3).tolist()
    grid_lats = grid_lats[::-1]

    current_dir = os.getcwd()
    observe_points_df = pd.read_csv(
        os.path.join(current_dir, "src/observation_point.csv"),
        index_col="Name",
    )

    idxs_of_arr = {}
    for i in observe_points_df.index:
        ob_lon, ob_lat = observe_points_df.loc[i, "LON"], observe_points_df.loc[i, "LAT"]
        idxs_of_arr[i] = {"lon": [], "lat": []}

        pred_tensor_lon_idxs = []
        pred_tensor_lat_idxs = []
        # Check longitude
        for before_lon, next_lon in zip(grid_lons[:-1], grid_lons[1:]):
            if ob_lon > before_lon and ob_lon < next_lon:
                pred_tensor_lon_idxs += [grid_lons.index(before_lon), grid_lons.index(next_lon)]

        # Check latitude
        for before_lat, next_lat in zip(grid_lats[:-1], grid_lats[1:]):
            if ob_lat < before_lat and ob_lat > next_lat:
                pred_tensor_lat_idxs += [grid_lats.index(before_lat), grid_lats.index(next_lat)]

        idxs_of_arr[i]["lon"] += pred_tensor_lon_idxs
        idxs_of_arr[i]["lat"] += pred_tensor_lat_idxs

    pred_df = pd.DataFrame(columns=["Pred_Value"], index=observe_points_df.index)
    for ob_name in idxs_of_arr.keys():
        _pred_values = []
        for lon_lat in list(itertools.product(idxs_of_arr[ob_name]["lon"], idxs_of_arr[ob_name]["lat"])):
            _pred_values.append(rain_tensor[lon_lat[1], lon_lat[0]])

        pred_df.loc[ob_name, "Pred_Value"] = np.round(sum(_pred_values) / len(_pred_values), decimals=3)

    return pred_df


def create_prediction(model: nn.Module, test_dataset: schemas.TestDataDict, downstream_directory: str, preprocess_delta: int):
    # test_dataset: Dict
    # { sample1: {
    #     date: str,
    #     start: str,
    #     input: Tensor shape is (batch_size, num_channels, seq_len, height, width),
    #     label: Tensor shape is (batch_size, num_channels, seq_len, height, width),
    #     label_df: Dict[int, pd.DataFrame]
    #  },
    #  sample2: {...}
    # }

    _time_step_csvnames = timestep_csv_names(delta=preprocess_delta)

    rmses_df = pd.DataFrame(columns=["isSequential", "case_type", "date", "date_time", "hour-rain", "Pred_Value"])
    for sample_name in test_dataset.keys():
        logger.info(f"Evaluationg {sample_name}")
        X_test = test_dataset[sample_name]["input"]
        y_test = test_dataset[sample_name]["label"]
        label_oneday_dfs = test_dataset[sample_name]["label_df"]

        input_seq_length = X_test.shape[2]

        date = test_dataset[sample_name]["date"]
        start = test_dataset[sample_name]["start"]
        start_idx = _time_step_csvnames.index(start)
        start = start.replace(".csv", "")

        save_dir = os.path.join(downstream_directory, sample_name)
        os.makedirs(save_dir, exist_ok=True)

        # Normal prediction.
        # Copy X_test because X_test is re-used after the normal prediction.
        _X_test = torch.clone(X_test)
        for t in range(input_seq_length):
            # pred_tensor: Tensor shape is (batch_size=1, num_channels, seq_len=1, height, width)
            pred_tensor: torch.Tensor = model(_X_test)
            pred_tensor = normalize_prediction(sample_name, pred_tensor)

            rain_tensor = pred_tensor[0, 0, 0, :, :]

            scaled_pred_tensor = rescale_tensor(min_value=0, max_value=100, tensor=rain_tensor)

            label_oneday_df = label_oneday_dfs[t]
            pred_oneday_df = pred_obervation_point_values(scaled_pred_tensor)
            label_pred_oneday_df = label_oneday_df.merge(pred_oneday_df, how="outer", left_index=True, right_index=True)
            label_pred_oneday_df = label_pred_oneday_df.dropna()

            label_pred_oneday_df["isSequential"] = False
            label_pred_oneday_df["case_type"] = sample_name.split("_case_")[0]
            label_pred_oneday_df["date"] = date
            label_pred_oneday_df["date_time"] = f"{date}_{start}"
            rmses_df = rmses_df.append(label_pred_oneday_df[["isSequential", "case_type", "date", "date_time", "hour-rain", "Pred_Value"]], ignore_index=True)

            rmse = mean_squared_error(
                np.ravel(label_pred_oneday_df["hour-rain"].values),
                np.ravel(label_pred_oneday_df["Pred_Value"].values),
                squared=False,
            )
            mlflow.log_metric(
                key=sample_name,
                value=rmse,
                step=t,
            )

            _X_test[0, :, :-1, :, :] = _X_test[0, :, 1:, :, :]
            _X_test[0, :, -1, :, :] = pred_tensor

            time_step_name = _time_step_csvnames[start_idx + t + 6].replace(".csv", "")
            save_rain_image(scaled_pred_tensor, save_dir + f"/{time_step_name}.png")
            label_pred_oneday_df.to_csv(save_dir + f"/pred_observ_df_{time_step_name}.csv")
            save_parquet(scaled_pred_tensor, save_dir + f"/{time_step_name}.parquet.gzip")

        # Sequential prediction
        save_dir_name = f"Sequential_{sample_name}"
        save_dir = os.path.join(downstream_directory, save_dir_name)
        os.makedirs(save_dir, exist_ok=True)

        for t in range(X_test.shape[1]):
            pred_tensor = model(X_test)
            pred_tensor = normalize_prediction(sample_name, pred_tensor)

            rain_tensor = pred_tensor[0, 0, 0, :, :]

            scaled_pred_tensor = rescale_tensor(min_value=0, max_value=100, tensor=rain_tensor)

            label_oneday_df = label_oneday_dfs[t]
            pred_oneday_df = pred_obervation_point_values(scaled_pred_tensor)
            label_pred_oneday_df = label_oneday_df.merge(pred_oneday_df, how="outer", left_index=True, right_index=True)
            label_pred_oneday_df = label_pred_oneday_df.dropna()

            label_pred_oneday_df["isSequential"] = True
            label_pred_oneday_df["case_type"] = sample_name.split("_case_")[0]
            label_pred_oneday_df["date"] = date
            label_pred_oneday_df["date_time"] = f"{date}_{start}"
            rmses_df = rmses_df.append(label_pred_oneday_df[["isSequential", "case_type", "date", "date_time", "hour-rain", "Pred_Value"]], ignore_index=True)

            rmse = mean_squared_error(
                np.ravel(label_pred_oneday_df["hour-rain"].values),
                np.ravel(label_pred_oneday_df["Pred_Value"].values),
                squared=False,
            )

            mlflow.log_metric(
                key=save_dir_name,
                value=rmse,
                step=t,
            )

            X_test[0, :, :-1, :, :] = X_test[0, :, 1:, :, :]
            X_test[0, :, -1, :, :] = y_test[0, :, t, :, :]

            time_step_name = _time_step_csvnames[start_idx + t + 6].replace(".csv", "")
            save_rain_image(scaled_pred_tensor, save_dir + f"/{time_step_name}.png")
            save_parquet(scaled_pred_tensor, save_dir + f"/{time_step_name}.parquet.gzip")

    # Visualize prediction results
    sample_plot(rmses_df, downstream_directory)
    all_cases_plot(rmses_df, downstream_directory)
    casetype_plot("tc", rmses_df, downstream_directory)
    casetype_plot("not_tc", rmses_df, downstream_directory)

    sample_plot(rmses_df, downstream_directory, isSequential=True)
    all_cases_plot(rmses_df, downstream_directory, isSequential=True)
    casetype_plot("tc", rmses_df, downstream_directory, isSequential=True)
    casetype_plot("not_tc", rmses_df, downstream_directory, isSequential=True)

    all_sample_rmse = mean_squared_error(
        np.ravel(rmses_df["hour-rain"]),
        np.ravel(rmses_df["Pred_Value"]),
        squared=False,
    )

    not_sequential_df = rmses_df.loc[rmses_df["isSequential"] is False]
    one_h_prediction_rmse = mean_squared_error(
        np.ravel(not_sequential_df["hour-rain"]),
        np.ravel(not_sequential_df["Pred_Value"]),
        squared=False,
    )
    return {"All_sample_RMSE": all_sample_rmse, "One_Hour_Prediction_RMSE": one_h_prediction_rmse}


def normalize_prediction(sample_name: str, pred_tensor: torch.Tensor) -> torch.Tensor:
    if pred_tensor.max() > 1 or pred_tensor.min() < 0:
        logger.warning(f"The predictions in {sample_name} contains more 1 or less 0 value. Autoscaleing is applyed.")

    pred_tensor = torch.where(pred_tensor > 1, 1.0, pred_tensor)
    pred_tensor = np.where(pred_tensor < 0, 0.0, pred_tensor)
    return pred_tensor
