from typing import Generator, List, Optional
from datetime import datetime, timedelta
import tracemalloc

import pandas as pd
import numpy as np
from pandas.core.generic import json
import torch
from sklearn.preprocessing import StandardScaler

import sys

sys.path.append(".")
from common.custom_logger import CustomLogger  # noqa: E402
from common.config import GridSize, MinMaxScalingValue  # noqa: E402

logger = CustomLogger("utils_Logger")


def calc_u_v(df: pd.DataFrame, ob_point: str) -> list:
    """Calculate u, v wind from wind direction and wind speed of p-poteka  `one_day_data`

    Args:
        df (pd.DataFrame): [description]
        ob_point (str): [description]

    Returns:
        [list]: [index, u wind, v wind] (u: X (East-West) v: Y(North-South))
    """
    wind_dir = float(df["WD1"])
    wind_speed = float(df["WS1"])

    rads = np.radians(float(wind_dir))
    wind_u, wind_v = -1 * wind_speed * np.cos(rads), -1 * wind_speed * np.sin(rads)
    # wind_u_v = wind_components(wind_speed * units("m/s"), wind_dir * units.deg)

    return [
        ob_point,
        round(wind_u, 5),
        round(wind_v, 5),
    ]  # (index, u wind, v wind) u: X (East-West) v: Y(North-South)


def get_mlflow_tag_from_input_parameters(input_parameters: list) -> str:
    tag_str = ""
    for p in input_parameters:
        tag_str += p[0].upper() + p[0:]
    return tag_str


def split_input_parameters_str(input_parameters_str: str) -> list:
    return input_parameters_str.split("/")


def datetime_range(start: datetime, end: datetime, delta: timedelta) -> Generator[datetime, None, None]:
    current = start
    while current <= end:
        yield current
        current += delta


def convert_two_digit_date(x: str) -> str:
    if len(str(x)) == 2:
        return str(x)
    else:
        return "0" + str(x)


def timestep_csv_names(year: int = 2020, month: int = 1, date: int = 1, time_step_minutes: int = 10) -> List[str]:
    dts = [
        f"{dt.hour}-{dt.minute}.csv"
        for dt in datetime_range(datetime(year, month, date, 0), datetime(year, month, date, 23, 59), timedelta(minutes=time_step_minutes))
    ]
    return dts


def format_bytes(size: int) -> str:
    power = 2**10
    n = 0
    power_labels = ["B", "KB", "MB", "GB", "TB"]
    while size > power and n <= len(power_labels):
        size /= power
        n += 1
    return f"current used memory: {size} {power_labels[n]}"


def log_memory() -> None:
    snapshot = tracemalloc.take_snapshot()
    size = sum([stat.size for stat in snapshot.statistics("filename")])
    print(format_bytes(size))


def min_max_scaler(min_value: float, max_value: float, arr: np.ndarray) -> np.ndarray:
    return (arr - min_value) / (max_value - min_value)


def rescale_tensor(min_value: float, max_value: float, tensor: torch.Tensor):
    return ((max_value - min_value) * tensor + min_value).to(dtype=torch.float)


def load_standard_scaled_data(path: str) -> np.ndarray:
    df = pd.read_csv(path, index_col=0, dtype=np.float32)
    scaler = StandardScaler()
    return scaler.fit_transform(df.values)


# return: ndarray
def load_scaled_data(path: str) -> np.ndarray:
    if path.endswith(".csv"):
        df = pd.read_csv(path, index_col=0, dtype=np.float32)
    elif path.endswith(".parquet.gzip"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Data file shoud be csv or parquet.gzip. (Given path is {path})")

    if "rain" in path:
        # df = df + 50
        # Scale [0, 100]
        return min_max_scaler(MinMaxScalingValue.RAIN_MIN, MinMaxScalingValue.RAIN_MAX, df.values)

    elif "temp" in path:
        # Scale [10, 45]
        return min_max_scaler(MinMaxScalingValue.TEMPERATURE_MIN, MinMaxScalingValue.TEMPERATURE_MAX, df.values)

    elif "abs_wind" in path:
        nd_arr = np.where(df > MinMaxScalingValue.ABS_WIND_MAX, MinMaxScalingValue.ABS_WIND_MAX, df)
        return min_max_scaler(MinMaxScalingValue.ABS_WIND_MIN, MinMaxScalingValue.ABS_WIND_MAX, nd_arr)

    elif "wind" in path:
        # Scale [-10, 10]
        return min_max_scaler(MinMaxScalingValue.WIND_MIN, MinMaxScalingValue.WIND_MAX, df.values)

    elif "humidity" in path:
        return min_max_scaler(MinMaxScalingValue.HUMIDITY_MIN, MinMaxScalingValue.HUMIDITY_MAX, df.values)

    elif "pressure" in path:
        return min_max_scaler(MinMaxScalingValue.SEALEVEL_PRESSURE_MIN, MinMaxScalingValue.SEALEVEL_PRESSURE_MAX, df.values)
    else:
        raise ValueError(f"Invalid data path: {path}")


def param_date_path(param_name: str, year, month, date) -> Optional[str]:
    if "rain" in param_name:
        return f"data/rain_image/{year}/{month}/{date}"
    elif "abs_wind" in param_name:
        return f"data/abs_wind_image/{year}/{month}/{date}"
    elif "wind" in param_name:
        return f"data/wind_image/{year}/{month}/{date}"
    elif "temperature" in param_name:
        return f"data/temp_image/{year}/{month}/{date}"
    elif "humidity" in param_name:
        return f"data/humidity_image/{year}/{month}/{date}"
    elif "station_pressure" in param_name:
        return f"data/station_pressure_image/{year}/{month}/{date}"
    elif "seaLevel_pressure" in param_name:
        return f"data/seaLevel_pressure_image/{year}/{month}/{date}"
    else:
        raise ValueError(f"Invalid param name: {param_name}")


def create_time_list(year: int = 2020, month: int = 1, date: int = 1, delta: int = 10) -> List[datetime]:
    dts = [dt for dt in datetime_range(datetime(year, month, date, 0), datetime(year, month, date, 23, 59), timedelta(minutes=delta))]
    return dts


def get_ob_point_values_from_tensor(tensor: torch.Tensor, observation_point_file_path: str) -> torch.Tensor:
    """
    This function extract observation point values of a given tensor (gridded p-poteka data).

    Args:
        tensor(torch.Tensor): A gridded p-poteka data of one weather para,eter.
    Returns:
        (numpy.ndarray): observation point values with the shape of (Number of observation point)
    """
    with open(observation_point_file_path, "r") as f:
        ob_point_data = json.load(f)
    ob_point_lons = [item["longitude"] for _, item in ob_point_data.items()]
    ob_point_lats = [item["latitude"] for _, item in ob_point_data.items()]

    grid_lons = np.linspace(120.90, 121.150, GridSize.WIDTH)
    grid_lats = np.linspace(14.350, 14.760, GridSize.HEIGHT)[::-1]  # Flip grid latitudes because the latitudes are in descending order.

    ob_point_values = torch.zeros((len(ob_point_lons)), dtype=torch.float)
    # Extract 9 data points in grid data near the each observation points.
    for ob_point_idx, (target_lon, target_lat) in enumerate(zip(ob_point_lons, ob_point_lats)):
        target_lon_idx, target_lat_idx = 0, 0
        for before_lon, next_lon in zip(grid_lons[:-1], grid_lons[1:]):
            target_lon_idx, target_lat_idx = 0, 0
            if before_lon < target_lon and target_lon < next_lon:
                target_lon_idx = np.where(grid_lons == before_lon)[0][0]
                break
        for next_lat, before_lat in zip(grid_lats[:-1], grid_lats[1:]):  # `grid_lats` is in descending order.
            if before_lat < target_lat and target_lat < next_lat:
                target_lat_idx = np.where(grid_lats == before_lat)[0][0]
                break
        if target_lon_idx == 0 or target_lon_idx == 0:
            raise ValueError(f"longitude or latitude is too small for the area of longigude (120.90, 120,150) and latitude (14.350, 14.760)")

        if target_lon_idx == len(grid_lons) - 1 or target_lon_idx == len(grid_lats) - 1:
            raise ValueError(f"longitude or latitude is too big for the area of longigude (120.90, 120,150) and latitude (14.350, 14.760)")
        # Extract values from gird data (tensor)
        ob_point_values[ob_point_idx] = tensor[target_lat_idx - 1 : target_lat_idx + 2, target_lon_idx - 1 : target_lon_idx + 2].mean().item()
    return ob_point_values


if __name__ == "__main__":
    tensor = torch.ones((50, 50))
    result = get_ob_point_values_from_tensor(tensor)
    print(result)
