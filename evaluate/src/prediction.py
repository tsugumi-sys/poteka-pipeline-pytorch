import os
import sys
import logging
import itertools

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import mlflow
from src.create_image import save_rain_image, all_cases_plot, sample_plot, casetype_plot

sys.path.append("..")
from common.utils import rescale_arr, timestep_csv_names

logger = logging.getLogger("Evaluate_Logger")


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


def pred_obervation_point_values(rain_arr):
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

        preds_lon_idxs = []
        preds_lat_idxs = []
        # Check longitude
        for before_lon, next_lon in zip(grid_lons[:-1], grid_lons[1:]):
            if ob_lon > before_lon and ob_lon < next_lon:
                preds_lon_idxs += [grid_lons.index(before_lon), grid_lons.index(next_lon)]

        # Check latitude
        for before_lat, next_lat in zip(grid_lats[:-1], grid_lats[1:]):
            if ob_lat < before_lat and ob_lat > next_lat:
                preds_lat_idxs += [grid_lats.index(before_lat), grid_lats.index(next_lat)]

        idxs_of_arr[i]["lon"] += preds_lon_idxs
        idxs_of_arr[i]["lat"] += preds_lat_idxs

    pred_df = pd.DataFrame(columns=["Pred_Value"], index=observe_points_df.index)
    for ob_name in idxs_of_arr.keys():
        _pred_values = []
        for lon_lat in list(itertools.product(idxs_of_arr[ob_name]["lon"], idxs_of_arr[ob_name]["lat"])):
            _pred_values.append(rain_arr[lon_lat[1], lon_lat[0]])

        pred_df.loc[ob_name, "Pred_Value"] = np.round(sum(_pred_values) / len(_pred_values), decimals=3)

    return pred_df


def create_prediction(model, test_dataset, downstream_directory: str, preprocess_delta: int):
    # test_dataset: Dict
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

    rmses_df = pd.DataFrame(columns=["isSequential", "case_type", "date", "date_time", "hour-rain", "Pred_Value"])
    for sample_name in test_dataset.keys():
        logger.info(f"Evaluationg {sample_name}")
        X_test = test_dataset[sample_name]["input"]
        y_test = test_dataset[sample_name]["label"]
        label_oneday_dfs = test_dataset[sample_name]["label_df"]

        feature_num = X_test.shape[-1]
        input_batch_size = X_test.shape[1]

        date = test_dataset[sample_name]["date"]
        start = test_dataset[sample_name]["start"]
        start_idx = _time_step_csvnames.index(start)
        start = start.replace(".csv", "")

        save_dir = os.path.join(downstream_directory, sample_name)
        os.makedirs(save_dir, exist_ok=True)

        # Normal prediction.
        # Copy X_test because X_test is re-used after the normal prediction.
        _X_test = X_test.copy()
        for t in range(_X_test.shape[1]):
            preds = model.predict(_X_test)
            preds = normalize_prediction(sample_name, preds)
            rain_arr = preds[0][:, :, 0]
            label_rain_arr = y_test[0][t][:, :, 0]

            scaled_pred_arr = rescale_arr(0, 100, rain_arr)
            scaled_label_arr = rescale_arr(0, 100, label_rain_arr)

            label_oneday_df = label_oneday_dfs[t]
            pred_oneday_df = pred_obervation_point_values(scaled_pred_arr)
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

            _X_test = np.append(_X_test[0][1:], preds, axis=0).reshape(1, input_batch_size, HEIGHT, WIDTH, feature_num)

            time_step_name = _time_step_csvnames[start_idx + t + 6].replace(".csv", "")
            save_rain_image(scaled_pred_arr, save_dir + f"/{time_step_name}.png")
            label_pred_oneday_df.to_csv(save_dir + f"/pred_observ_df_{time_step_name}.csv")
            save_parquet(scaled_pred_arr, save_dir + f"/{time_step_name}.parquet.gzip")

        # Sequential prediction
        save_dir_name = f"Sequential_{sample_name}"
        save_dir = os.path.join(downstream_directory, save_dir_name)
        os.makedirs(save_dir, exist_ok=True)

        for t in range(X_test.shape[1]):
            preds = model.predict(X_test)
            preds = normalize_prediction(sample_name, preds)
            rain_arr = preds[0][:, :, 0]
            label_rain_arr = y_test[0][t][:, :, 0]

            scaled_pred_arr = rescale_arr(0, 100, rain_arr)
            scaled_label_arr = rescale_arr(0, 100, label_rain_arr)

            label_oneday_df = label_oneday_dfs[t]
            pred_oneday_df = pred_obervation_point_values(scaled_pred_arr)
            label_pred_oneday_df = label_oneday_df.merge(pred_oneday_df, how="outer", left_index=True, right_index=True)
            label_pred_oneday_df = label_pred_oneday_df.dropna()

            label_pred_oneday_df["isSequential"] = True
            label_pred_oneday_df["case_type"] = sample_name.split("_case_")[0]
            label_pred_oneday_df["date"] = date
            label_pred_oneday_df["date_time"] = f"{date}_{start}"
            rmses_df = rmses_df.append(label_pred_oneday_df[["isSequential", "case_type", "date", "date_time", "hour-rain", "Pred_Value"]], ignore_index=True)

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

            X_test = np.append(X_test[0][1:], [y_test[0][t]], axis=0).reshape(1, input_batch_size, HEIGHT, WIDTH, feature_num)

            time_step_name = _time_step_csvnames[start_idx + t + 6].replace(".csv", "")
            save_rain_image(scaled_pred_arr, save_dir + f"/{time_step_name}.png")
            save_parquet(scaled_pred_arr, save_dir + f"/{time_step_name}.parquet.gzip")

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


def normalize_prediction(sample_name, pred_arr):
    if pred_arr.max() > 1 or pred_arr.min() < 0:
        logger.warning(f"The predictions in {sample_name} contains more 1 or less 0 value. Autoscaleing is applyed.")

    pred_arr = np.where(pred_arr > 1, 1, pred_arr)
    pred_arr = np.where(pred_arr < 0, 0, pred_arr)
    return pred_arr
