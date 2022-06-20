import os
import sys
import logging
from typing import Dict

import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import mlflow

from src.create_image import save_rain_image, all_cases_plot, sample_plot, casetype_plot

sys.path.append("..")
from common.config import WEATHER_PARAMS, ScalingMethod
from common.utils import rescale_tensor, timestep_csv_names
from train.src.config import DEVICE
from evaluate.src.utils import re_standard_scale, normalize_tensor, standard_scaler_torch_tensor, validate_scaling

logger = logging.getLogger("Evaluate_Logger")

# # Move to ./utils.py
# def save_parquet(tensor: np.ndarray, save_path: str) -> None:
#     grid_lon, grid_lat = np.round(np.linspace(120.90, 121.150, 50), 3), np.round(np.linspace(14.350, 14.760, 50), 3)
#     df = pd.DataFrame(tensor, index=np.flip(grid_lat), columns=grid_lon)
#     df.index = df.index.astype(str)
#     df.columns = df.columns.astype(str)
#     df.to_parquet(
#         path=save_path,
#         engine="pyarrow",
#         compression="gzip",
#     )

# # Move to utils.py
# def pred_obervation_point_values(rain_tensor: np.ndarray) -> pd.DataFrame:
#     """Prediction value near the observation points

#     Args:
#         rain_tensor (torch.Tensor): The shape is (HEIGHT, WIDTH)

#     Returns:
#         (pd.DataFrame): DataFrame that has `Pred_Value` column and `observation point name` index.
#     """
#     HEIGHT, WIDTH = 50, 50
#     grid_lons = np.round(np.linspace(120.90, 121.150, WIDTH), decimals=3).tolist()
#     grid_lats = np.round(np.linspace(14.350, 14.760, HEIGHT), decimals=3).tolist()
#     grid_lats = grid_lats[::-1]

#     current_dir = os.getcwd()
#     observe_points_df = pd.read_csv(
#         os.path.join(current_dir, "src/observation_point.csv"),
#         index_col="Name",
#     )

#     idxs_of_arr = {}
#     for i in observe_points_df.index:
#         ob_lon, ob_lat = observe_points_df.loc[i, "LON"], observe_points_df.loc[i, "LAT"]
#         idxs_of_arr[i] = {"lon": [], "lat": []}

#         pred_tensor_lon_idxs = []
#         pred_tensor_lat_idxs = []
#         # Check longitude
#         for before_lon, next_lon in zip(grid_lons[:-1], grid_lons[1:]):
#             if ob_lon > before_lon and ob_lon < next_lon:
#                 pred_tensor_lon_idxs += [grid_lons.index(before_lon), grid_lons.index(next_lon)]

#         # Check latitude
#         for before_lat, next_lat in zip(grid_lats[:-1], grid_lats[1:]):
#             if ob_lat < before_lat and ob_lat > next_lat:
#                 pred_tensor_lat_idxs += [grid_lats.index(before_lat), grid_lats.index(next_lat)]

#         idxs_of_arr[i]["lon"] += pred_tensor_lon_idxs
#         idxs_of_arr[i]["lat"] += pred_tensor_lat_idxs

#     pred_df = pd.DataFrame(columns=["Pred_Value"], index=observe_points_df.index)
#     for ob_name in idxs_of_arr.keys():
#         _pred_values = []
#         for lon_lat in list(itertools.product(idxs_of_arr[ob_name]["lon"], idxs_of_arr[ob_name]["lat"])):
#             _pred_values.append(rain_tensor[lon_lat[1], lon_lat[0]])

#         pred_df.loc[ob_name, "Pred_Value"] = np.round(sum(_pred_values) / len(_pred_values), decimals=3)

#     return pred_df


# [TODO]: refactoring is needed
def create_prediction(
    model: nn.Module,
    test_dataset: Dict,
    downstream_directory: str,
    preprocess_time_step_minutes: int,
    scaling_method: str,
    feature_names: Dict[int, str],
    use_dummy_data: bool = False,
):
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
    model.eval()
    with torch.no_grad():
        _time_step_csvnames = timestep_csv_names(time_step_minutes=preprocess_time_step_minutes)

        rmses_df = pd.DataFrame(columns=["isSequential", "case_type", "date", "date_time", "hour-rain", "Pred_Value"])
        for sample_name in test_dataset.keys():
            logger.info(f"Evaluate {sample_name}")
            X_test: torch.Tensor = test_dataset[sample_name]["input"]
            y_test: torch.Tensor = test_dataset[sample_name]["label"]
            y_test_size = y_test.size()
            y_test_batch_size = y_test_size[0]
            num_channels = y_test_size[1]
            height, width = y_test_size[3], y_test_size[4]

            validate_scaling(y_test, scaling_method=ScalingMethod.MinMax.value, logger=logger)
            validate_scaling(X_test, scaling_method=scaling_method, logger=logger)

            X_test, y_test = X_test.to(device=DEVICE), y_test.to(device=DEVICE)

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
            _X_test = X_test.clone().detach()
            logger.info("- Evaluating 1 hour prediction ...")
            for t in range(input_seq_length):
                # pred_tensor: Tensor shape is (batch_size=1, num_channels, seq_len=1, height, width)
                # [TODO]: sometime predict vlaues are all nan.
                pred_tensor: torch.Tensor = model(_X_test)

                if torch.isnan(pred_tensor).any():
                    logger.warning(model.state_dict())
                    logger.warning(pred_tensor)

                pred_tensor = normalize_tensor(pred_tensor, device=DEVICE)
                validate_scaling(tensor=pred_tensor, scaling_method=ScalingMethod.MinMax.value, logger=logger)

                rain_tensor = pred_tensor[0, 0, 0, :, :]

                scaled_pred_tensor = rescale_tensor(min_value=0, max_value=100, tensor=rain_tensor)

                scaled_pred_ndarr = scaled_pred_tensor.cpu().detach().numpy().copy()

                # Calculate RMSE
                if use_dummy_data is False:
                    # If you use dummy data, pred_overvation_point_values is skipped.
                    label_oneday_df = label_oneday_dfs[t]
                    pred_oneday_df = pred_obervation_point_values(scaled_pred_ndarr)
                    label_pred_oneday_df = label_oneday_df.merge(pred_oneday_df, how="outer", left_index=True, right_index=True)
                    label_pred_oneday_df = label_pred_oneday_df.dropna()

                    label_pred_oneday_df["isSequential"] = False
                    label_pred_oneday_df["case_type"] = sample_name.split("_case_")[0]
                    label_pred_oneday_df["date"] = date
                    label_pred_oneday_df["date_time"] = f"{date}_{start}"
                    rmses_df = rmses_df.append(
                        label_pred_oneday_df[["isSequential", "case_type", "date", "date_time", "hour-rain", "Pred_Value"]], ignore_index=True
                    )

                    rmse = mean_squared_error(
                        np.ravel(label_pred_oneday_df["hour-rain"].values),
                        np.ravel(label_pred_oneday_df["Pred_Value"].values),
                        squared=False,
                    )
                else:
                    label_tensor = y_test[0, 0, t, :, :].cpu().detach().numpy().copy()
                    rmse = mean_squared_error(np.ravel(scaled_pred_ndarr), np.ravel(label_tensor), squared=False)

                mlflow.log_metric(
                    key=sample_name,
                    value=rmse,
                    step=t,
                )

                time_step_idx = start_idx + t + 6
                if time_step_idx > len(_time_step_csvnames) - 1:
                    time_step_idx -= len(_time_step_csvnames)
                time_step_name = _time_step_csvnames[time_step_idx].replace(".csv", "")
                # [TODO]
                # Solve unknown error of "free(): invalid size" in poetry env. Use conda environment to visualize for now....
                if use_dummy_data is False:
                    save_rain_image(scaled_pred_ndarr, save_dir + f"/{time_step_name}.png")
                    label_pred_oneday_df.to_csv(save_dir + f"/pred_observ_df_{time_step_name}.csv")
                save_parquet(scaled_pred_ndarr, save_dir + f"/{time_step_name}.parquet.gzip")

                # Rescale pred_tensor for next prediction
                if scaling_method == ScalingMethod.Standard.value or scaling_method == ScalingMethod.MinMaxStandard.value:
                    for idx, name in feature_names.items():
                        pred_feature_tensor = pred_tensor[0, idx, 0, :, :]
                        pred_feature_tensor = normalize_tensor(pred_feature_tensor, device=DEVICE)
                        if name != WEATHER_PARAMS.RAIN.value:
                            if scaling_method == ScalingMethod.Standard.value:
                                pred_tensor[0, idx, 0, :, :] = re_standard_scale(pred_feature_tensor, feature_name=name, device=DEVICE, logger=logger)

                            elif scaling_method == ScalingMethod.MinMaxStandard.value:
                                pred_tensor[0, idx, 0, :, :] = standard_scaler_torch_tensor(pred_feature_tensor, device=DEVICE)

                        else:
                            pred_tensor[0, idx, 0, :, :] = pred_feature_tensor

                    validate_scaling(pred_tensor, scaling_method=ScalingMethod.Standard.value, logger=logger)
                elif scaling_method == ScalingMethod.MinMax.value:
                    validate_scaling(y_test, scaling_method=ScalingMethod.MinMax.value, logger=logger)

                _X_test = torch.cat((_X_test[:, :, 1:, :, :], pred_tensor), dim=2)
                validate_scaling(_X_test, scaling_method=scaling_method, logger=logger)

            # Sequential prediction
            save_dir_name = f"Sequential_{sample_name}"
            save_dir = os.path.join(downstream_directory, save_dir_name)
            os.makedirs(save_dir, exist_ok=True)

            logger.info("- Evaluating 10 minutes prediction ...")
            for t in range(input_seq_length):
                pred_tensor = model(X_test)

                if torch.isnan(pred_tensor).any():
                    logger.warning(pred_tensor)

                # pred_tensor = torch.rand(pred_tensor.shape, device=DEVICE)
                pred_tensor = normalize_tensor(pred_tensor, device=DEVICE)
                validate_scaling(tensor=pred_tensor, scaling_method=ScalingMethod.MinMax.value, logger=logger)

                rain_tensor = pred_tensor[0, 0, 0, :, :]

                scaled_pred_tensor = rescale_tensor(min_value=0, max_value=100, tensor=rain_tensor)

                scaled_pred_ndarr = scaled_pred_tensor.cpu().detach().numpy().copy()

                # Calculate RMSE
                if use_dummy_data is False:
                    # If you use dummy data, pred_overvation_point_values is skipped.
                    label_oneday_df = label_oneday_dfs[t]
                    pred_oneday_df = pred_obervation_point_values(scaled_pred_ndarr)
                    label_pred_oneday_df = label_oneday_df.merge(pred_oneday_df, how="outer", left_index=True, right_index=True)
                    label_pred_oneday_df = label_pred_oneday_df.dropna()

                    label_pred_oneday_df["isSequential"] = True
                    label_pred_oneday_df["case_type"] = sample_name.split("_case_")[0]
                    label_pred_oneday_df["date"] = date
                    label_pred_oneday_df["date_time"] = f"{date}_{start}"
                    rmses_df = rmses_df.append(
                        label_pred_oneday_df[["isSequential", "case_type", "date", "date_time", "hour-rain", "Pred_Value"]], ignore_index=True
                    )

                    rmse = mean_squared_error(
                        np.ravel(label_pred_oneday_df["hour-rain"].values),
                        np.ravel(label_pred_oneday_df["Pred_Value"].values),
                        squared=False,
                    )
                else:
                    label_tensor = y_test[0, 0, t, :, :].cpu().detach().numpy().copy()
                    rmse = mean_squared_error(np.ravel(scaled_pred_ndarr), np.ravel(label_tensor), squared=False)

                mlflow.log_metric(
                    key=save_dir_name,
                    value=rmse,
                    step=t,
                )

                # Rescale pred_tensor for next prediction
                if scaling_method == ScalingMethod.Standard.value or scaling_method == ScalingMethod.MinMaxStandard.value:
                    for idx, name in feature_names.items():
                        if name != WEATHER_PARAMS.RAIN.value:

                            if scaling_method == ScalingMethod.Standard.value:
                                y_test[0, idx, t, :, :] = re_standard_scale(y_test[0, idx, t, :, :], feature_name=name, device=DEVICE, logger=logger)

                            elif scaling_method == ScalingMethod.MinMaxStandard.value:
                                y_test[0, idx, t, :, :] = standard_scaler_torch_tensor(y_test[0, idx, t, :, :], device=DEVICE)

                        else:
                            continue

                        if torch.isnan(y_test).any():
                            logger.error(f"{name}")
                            logger.error(y_test[0, idx, t, :, :])

                    validate_scaling(y_test, scaling_method=ScalingMethod.Standard.value, logger=logger)
                elif scaling_method == ScalingMethod.MinMax.value:
                    validate_scaling(y_test, scaling_method=ScalingMethod.MinMax.value, logger=logger)

                # X_test[0, :, -1, :, :] = y_test[0, :, t, :, :]
                X_test = torch.cat((X_test[:, :, 1:, :, :], torch.reshape(y_test[0, :, t, :, :], (y_test_batch_size, num_channels, 1, height, width))), dim=2)
                validate_scaling(X_test, scaling_method=scaling_method, logger=logger)

                if torch.isnan(X_test).any():
                    logger.error(X_test)

                time_step_idx = start_idx + t + 6
                if time_step_idx > len(_time_step_csvnames) - 1:
                    time_step_idx = -len(_time_step_csvnames)
                time_step_name = _time_step_csvnames[time_step_idx].replace(".csv", "")
                # [TODO]
                # Solve unknown error of "free(): invalid size" in poetry env. Use conda environment to visualize for now....
                if use_dummy_data is False:
                    # If you use dummy data, cartopy is not installed. So save_rain_image is skipped.
                    save_rain_image(scaled_pred_ndarr, save_dir + f"/{time_step_name}.png")
                save_parquet(scaled_pred_ndarr, save_dir + f"/{time_step_name}.parquet.gzip")

        # Initialize results metrics dict
        result_metrics = {}

        # Visualize prediction results
        if use_dummy_data is False:
            sample_plot(rmses_df, downstream_directory=downstream_directory, result_metrics=result_metrics)
            all_cases_plot(rmses_df, downstream_directory=downstream_directory, result_metrics=result_metrics)
            casetype_plot("tc", rmses_df, downstream_directory=downstream_directory, result_metrics=result_metrics)
            casetype_plot("not_tc", rmses_df, downstream_directory=downstream_directory, result_metrics=result_metrics)

            sample_plot(rmses_df, downstream_directory=downstream_directory, result_metrics=result_metrics, isSequential=True)
            all_cases_plot(rmses_df, downstream_directory=downstream_directory, result_metrics=result_metrics, isSequential=True)
            casetype_plot("tc", rmses_df, downstream_directory=downstream_directory, result_metrics=result_metrics, isSequential=True)
            casetype_plot("not_tc", rmses_df, downstream_directory=downstream_directory, result_metrics=result_metrics, isSequential=True)

            all_sample_rmse = mean_squared_error(
                np.ravel(rmses_df["hour-rain"]),
                np.ravel(rmses_df["Pred_Value"]),
                squared=False,
            )

            not_sequential_df = rmses_df.loc[rmses_df["isSequential"] == True]  # noqa: E712
            one_h_prediction_rmse = mean_squared_error(
                np.ravel(not_sequential_df["hour-rain"]),
                np.ravel(not_sequential_df["Pred_Value"]),
                squared=False,
            )
            result_metrics["All_sample_RMSE"] = all_sample_rmse
            result_metrics["One_Hour_Prediction_RMSE"] = one_h_prediction_rmse
        else:
            result_metrics["sample_metrics"] = 0.1
    return result_metrics
