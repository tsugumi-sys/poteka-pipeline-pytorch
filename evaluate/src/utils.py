import logging
import sys
import os

import torch
import numpy as np
import pandas as pd
import itertools

sys.path.append("..")
from common.config import MinMaxScalingValue, WEATHER_PARAMS, ScalingMethod
from common.utils import rescale_tensor


def pred_obervation_point_values(rain_tensor: np.ndarray, use_dummy_data: bool = False) -> pd.DataFrame:
    """Prediction value near the observation points

    Args:
        rain_tensor (torch.Tensor): The shape is (HEIGHT, WIDTH)

    Returns:
        (pd.DataFrame): DataFrame that has `Pred_Value` column and `observation point name` index.
    """
    HEIGHT, WIDTH = 50, 50
    grid_lons = np.around(np.linspace(120.90, 121.150, WIDTH), decimals=3).tolist()
    grid_lats = np.around(np.linspace(14.350, 14.760, HEIGHT), decimals=3).tolist()
    grid_lats = grid_lats[::-1]

    current_dir = os.getcwd()
    if use_dummy_data:
        observe_points_df = pd.DataFrame(
            {
                "LON": np.random.uniform(min(grid_lons), max(grid_lons), 10),
                "LAT": np.random.uniform(min(grid_lats), max(grid_lats), 10),
            }
        )
    else:
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


def save_parquet(tensor: np.ndarray, save_path: str) -> None:
    grid_lon, grid_lat = np.round(np.linspace(120.90, 121.150, 50), 3), np.round(np.linspace(14.350, 14.760, 50), 3)
    df = pd.DataFrame(tensor, index=np.flip(grid_lat), columns=grid_lon)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    df.to_parquet(
        path=save_path,
        engine="pyarrow",
        compression="gzip",
    )


def re_standard_scale(tensor: torch.Tensor, feature_name: str, device: str, logger: logging.Logger) -> torch.Tensor:
    """Re scaling tensor
        1. tensor is [0, 1] sacled.
        2. re scale original scale for each weather parameter.
        3. Standard normalizing

    Args:
        tensor (torch.Tensor): input tensor with [0, 1] sacling.
        feature_name (str): feature name.
        logger (logging.Logger): logger

    Returns:
        torch.Tensor: Standard normalized tensor
    """
    rescaled_tensor = rescale_pred_tensor(tensor=tensor, feature_name=feature_name)
    if torch.isnan(rescaled_tensor).any():
        logger.error(f"{feature_name} has nan values")
        logger.error(rescale_tensor)
    return standard_scaler_torch_tensor(rescaled_tensor, device)


def rescale_pred_tensor(tensor: torch.Tensor, feature_name: str) -> torch.Tensor:
    # Tensor is scaled as [0, 1]
    # Rescale tensor again for standarization
    if feature_name == WEATHER_PARAMS.RAIN.value:
        return rescale_tensor(min_value=MinMaxScalingValue.RAIN_MIN, max_value=MinMaxScalingValue.RAIN_MAX, tensor=tensor)

    elif feature_name == WEATHER_PARAMS.TEMPERATURE.value:
        return rescale_tensor(min_value=MinMaxScalingValue.TEMPERATURE_MIN, max_value=MinMaxScalingValue.TEMPERATURE_MAX, tensor=tensor)

    elif feature_name == WEATHER_PARAMS.HUMIDITY.value:
        return rescale_tensor(min_value=MinMaxScalingValue.HUMIDITY_MIN, max_value=MinMaxScalingValue.HUMIDITY_MAX, tensor=tensor)

    elif feature_name in [WEATHER_PARAMS.WIND.value, WEATHER_PARAMS.U_WIND.value, WEATHER_PARAMS.V_WIND.value]:
        return rescale_tensor(min_value=MinMaxScalingValue.WIND_MIN, max_value=MinMaxScalingValue.WIND_MAX, tensor=tensor)

    elif feature_name == WEATHER_PARAMS.ABS_WIND.value:
        return rescale_tensor(min_value=MinMaxScalingValue.ABS_WIND_MIN, max_value=MinMaxScalingValue.ABS_WIND_MAX, tensor=tensor)

    elif feature_name == WEATHER_PARAMS.STATION_PRESSURE.value:
        return rescale_tensor(min_value=MinMaxScalingValue.STATION_PRESSURE_MIN, max_value=MinMaxScalingValue.STATION_PRESSURE_MAX, tensor=tensor)

    elif feature_name == WEATHER_PARAMS.SEALEVEL_PRESSURE.value:
        return rescale_tensor(min_value=MinMaxScalingValue.SEALEVEL_PRESSURE_MIN, max_value=MinMaxScalingValue.SEALEVEL_PRESSURE_MAX, tensor=tensor)

    else:
        raise ValueError(f"Invalid feature name {feature_name}")


def standard_scaler_torch_tensor(tensor: torch.Tensor, device: str) -> torch.Tensor:
    std, mean = torch.std_mean(tensor, unbiased=False)
    # [WARN]:
    # If tensor has same values, mean goes to zero and standarized tensor has NaN values.
    # Artificially add noises to avoid this.
    if std < 0.0001:
        delta = 1.0
        tensor = tensor + torch.rand(size=tensor.size()).to(device=device) * delta
        std, mean = torch.std_mean(tensor, unbiased=False)
    return ((tensor - mean.item()) / std.item()).to(dtype=torch.float)


def normalize_tensor(tensor: torch.Tensor, device: str) -> torch.Tensor:
    ones_tensor = torch.ones(tensor.shape, dtype=torch.float).to(device)
    zeros_tensor = torch.zeros(tensor.shape, dtype=torch.float).to(device)
    tensor = torch.where(tensor > 1, ones_tensor, tensor)
    tensor = torch.where(tensor < 0, zeros_tensor, tensor)
    return tensor.to(dtype=torch.float)


def validate_scaling(tensor: torch.Tensor, scaling_method: str, logger: logging.Logger) -> None:
    if scaling_method is ScalingMethod.MinMax.value:
        max_value, min_value = tensor.max().item(), tensor.min().item()
        if max_value > 1 or min_value < 0:
            logger.error(f"Tensor is faild to be min-max scaled. Max: {max_value}, Min: {min_value}")

    elif scaling_method is ScalingMethod.Standard.value:
        std_val, mean_val = tensor.std().item(), tensor.mean().item()
        if abs(1 - std_val) > 0.001 or abs(mean_val) > 0.001:
            logger.error(f"Tensor is faild to be standard scaled. Std: {std_val}, Mean: {mean_val}")
