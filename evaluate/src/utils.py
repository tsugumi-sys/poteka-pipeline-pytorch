import logging
import sys

import torch

sys.path.append("..")
from common.config import MinMaxScalingValue, WEATHER_PARAMS, ScalingMethod
from common.utils import rescale_tensor


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
