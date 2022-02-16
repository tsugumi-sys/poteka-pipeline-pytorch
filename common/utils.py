from typing import Generator, List, Optional
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
import tracemalloc

from .custom_logger import CustomLogger

logger = CustomLogger("utils_Logger")


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


def timestep_csv_names(year: int = 2020, month: int = 1, date: int = 1, delta: int = 10) -> List[str]:
    dts = [f"{dt.hour}-{dt.minute}.csv" for dt in datetime_range(datetime(year, month, date, 0), datetime(year, month, date, 23, 59), timedelta(minutes=delta))]
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
    return (max_value - min_value) * tensor + min_value


# return: ndarray
def load_scaled_data(path: str):
    df = pd.read_csv(path, index_col=0, dtype=np.float32)
    if "rain" in path:
        # df = df + 50
        # Scale [0, 100]
        return min_max_scaler(0, 100, df.values)

    elif "temp" in path:
        # Scale [10, 45]
        return min_max_scaler(10, 45, df.values)

    elif "abs_wind" in path:
        df = np.where(df > 15, 15, df)
        return min_max_scaler(0, 15, df.values)

    elif "wind" in path:
        # Scale [-10, 10]
        return min_max_scaler(-10, 10, df.values)

    elif "humidity" in path:
        return min_max_scaler(0, 100, df.values)

    elif "pressure" in path:
        return min_max_scaler(990, 1025, df.values)


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
