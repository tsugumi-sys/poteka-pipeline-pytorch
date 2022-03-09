import sys

import torch

sys.path.append("..")
from common.config import MinMaxScalingValue, WEATHER_PARAMS
from common.utils import rescale_tensor


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


def re_standard_scale(tensor: torch.Tensor, feature_name: str) -> torch.Tensor:
    rescaled_tensor = rescale_pred_tensor(tensor=tensor, feature_name=feature_name)
    return standard_scaler_torch_tensor(rescaled_tensor)


def standard_scaler_torch_tensor(tensor: torch.Tensor) -> torch.Tensor:
    std, mean = torch.std_mean(tensor, unbiased=False)
    return (tensor - mean.item()) / std.item()
