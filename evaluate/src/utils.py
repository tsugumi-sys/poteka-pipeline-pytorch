import logging
import sys
import json

import torch
import numpy as np
import pandas as pd

sys.path.append("..")
from common.config import MinMaxScalingValue, WEATHER_PARAMS, ScalingMethod, GridSize
from common.utils import rescale_tensor

logger = logging.getLogger(__name__)


def pred_observation_point_values(rain_tensor: np.ndarray) -> pd.DataFrame:
    """Prediction value near the observation points

    Args:
        rain_tensor (torch.Tensor): The shape is (HEIGHT, WIDTH)

    Returns:
        (pd.DataFrame): DataFrame that has `Pred_Value` column and `observation point name` index.
    """
    grid_lons = np.linspace(120.90, 121.150, GridSize.WIDTH)
    grid_lats = np.linspace(14.350, 14.760, GridSize.HEIGHT)[::-1]  # Flip latitudes because latitudes are in desending order.
    grid_lats = grid_lats[::-1]

    with open("../common/meta-data/observation_point.json", "r") as f:
        ob_point_data = json.load(f)

    ob_point_names = [k for k in ob_point_data.keys()]
    ob_lons, ob_lats = [val["longitude"] for val in ob_point_data.values()], [val["latitude"] for val in ob_point_data.values()]

    pred_df = pd.DataFrame(columns=["Pred_Value"], index=ob_point_names)
    for ob_point_idx, ob_point_name in enumerate(ob_point_names):
        if rain_tensor.ndim == 1:
            pred_df.loc[ob_point_name, "Pred_Value"] = rain_tensor[ob_point_idx]
        else:
            ob_lon, ob_lat = ob_lons[ob_point_idx], ob_lats[ob_point_idx]
            target_lon, target_lat = 0, 0
            # Check longitude
            for before_lon, next_lon in zip(grid_lons[:-1], grid_lons[1:]):
                if ob_lon > before_lon and ob_lon < next_lon:
                    target_lon = before_lon
            # Check latitude
            for next_lat, before_lat in zip(grid_lats[:-1], grid_lats[1:]):  # NOTE: grid_lats are flipped and in descending order.
                if ob_lat < before_lat and ob_lat > next_lat:
                    target_lat = before_lat

            pred_df.loc[ob_point_name, "Pred_Value"] = rain_tensor[target_lat - 1 : target_lat + 2, target_lon - 1 : target_lon + 2]
    return pred_df


def save_parquet(ndarray: np.ndarray, save_path: str) -> None:
    if ndarray.shape[0] != GridSize.HEIGHT or ndarray.shape[1] != GridSize.WIDTH:
        #logger.warning(f"Tensor is not grid data. The shape is {ndarray.shape}")

        with open("../common/meta-data/observation_point.json", "r") as f:
            ob_point_data = json.load(f)
        ob_point_names = list(ob_point_data.keys())
        df = pd.DataFrame(ndarray, index=ob_point_names, columns=["prediction_value"])
        df.to_parquet(save_path, engine="pyarrow", compression="gzip")
    else:
        grid_lon, grid_lat = np.round(np.linspace(120.90, 121.150, 50), 3), np.round(np.linspace(14.350, 14.760, 50), 3)
        df = pd.DataFrame(ndarray, index=np.flip(grid_lat), columns=grid_lon)
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)
        df.to_parquet(path=save_path, engine="pyarrow", compression="gzip")


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
    std, mean = std.item(), mean.item()
    # [WARN]:
    # If tensor has same values, mean goes to zero and standarized tensor has NaN values.
    # Artificially add noises to avoid this.
    # if std < 0.0001:
    #     delta = 1.0
    #     tensor = tensor + torch.rand(size=tensor.size()).to(device=device) * delta
    #     std, mean = torch.std_mean(tensor, unbiased=False)
    # Handle zewros in standarization
    # see https://github.com/scikit-learn/scikit-learn/blob/7389dbac82d362f296dc2746f10e43ffa1615660/sklearn/preprocessing/data.py#L70
    if std == 0:
        std = 1
    return ((tensor - mean) / std).to(dtype=torch.float)


def normalize_tensor(tensor: torch.Tensor, device: str) -> torch.Tensor:
    ones_tensor = torch.ones(tensor.shape, dtype=torch.float).to(device)
    zeros_tensor = torch.zeros(tensor.shape, dtype=torch.float).to(device)
    tensor = torch.where(tensor > 1, ones_tensor, tensor)
    tensor = torch.where(tensor < 0, zeros_tensor, tensor)
    return tensor.to(dtype=torch.float)


def validate_scaling(tensor: torch.Tensor, scaling_method: str, logger: logging.Logger) -> None:
    if scaling_method is ScalingMethod.MinMax.value:
        max_value, min_value = torch.max(tensor).item(), torch.min(tensor).item()
        if max_value > 1 or min_value < 0:
            logger.error(f"Tensor is faild to be min-max scaled. Max: {max_value}, Min: {min_value}")

    elif scaling_method is ScalingMethod.Standard.value:
        std_val, mean_val = torch.std(tensor).item(), torch.mean(tensor).item()
        if abs(1 - std_val) > 0.001 or abs(mean_val) > 0.001:
            logger.error(f"Tensor is faild to be standard scaled. Std: {std_val}, Mean: {mean_val}")
