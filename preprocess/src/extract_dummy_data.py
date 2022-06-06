from typing import Dict, List, Tuple
import os
import sys

import pandas as pd
import numpy as np

sys.path.append("..")
from common.utils import timestep_csv_names
from common.config import WEATHER_PARAMS, MinMaxScalingValue, GridSize


def get_dummy_data_files(
    input_parameters: List[str],
    time_step_minutes: int,
    root_dir_path: str,
    train_dataset_length: int = 100,
    test_dataset_length: int = 10,
) -> Dict:
    if WEATHER_PARAMS.RAIN.value not in input_parameters:
        raise ValueError("'rain' should be in `input_parameters`")

    if not WEATHER_PARAMS.is_params_valid(input_parameters):
        raise ValueError("Invalid input parametes in `input_parameters`")

    paths = []
    for _ in range(train_dataset_length):
        dummy_data_path = save_dummy_data(input_parameters=input_parameters, time_step_minutes=time_step_minutes, root_dir_path=root_dir_path)
        paths.append(dummy_data_path)

    return paths


def save_dummy_data(input_parameters: List[str], time_step_minutes: int, root_dir_path: str) -> Dict:
    _timestep_csv_names = timestep_csv_names(time_step_minutes=time_step_minutes)
    csv_file_names = _timestep_csv_names[0:7]

    paths = {}
    for param in input_parameters:
        save_csv_dir = os.path.join(root_dir_path, "dummy_data", param)
        os.makedirs(save_csv_dir, exist_ok=True)
        # Input & Label data (csv_file_names[:6] for input, csv_file_names[6] for label)
        for csv_file_name in csv_file_names:
            dummy_data_df = generate_dummy_data(input_parameter=param, array_shape=(GridSize.HEIGHT.value, GridSize.WIDTH.value))
            save_csv_file_path = os.path.join(save_csv_dir, csv_file_name)
            dummy_data_df.to_csv(save_csv_file_path)
        paths[param] = {"input": [os.path.join(save_csv_dir, f) for f in csv_file_names[:6]], "label": [os.path.join(save_csv_dir, csv_file_names[-1])]}
    return paths


def generate_dummy_data(input_parameter: str, array_shape: Tuple = (50, 50)) -> pd.DataFrame:
    if not WEATHER_PARAMS.is_params_valid(input_parameter):
        raise ValueError(f"Invalid `input_parameter`: {input_parameter}")

    arr = np.random.rand(array_shape)
    dummy_cols = [f"col{i}" for i in range(array_shape[0])]
    if input_parameter == WEATHER_PARAMS.RAIN.value:
        max_rain_val, min_rain_val = MinMaxScalingValue.RAIN_MAX.value, MinMaxScalingValue.RAIN_MIN.value
        arr = (max_rain_val - min_rain_val) * arr + min_rain_val

    elif input_parameter == WEATHER_PARAMS.TEMPERATURE.value:
        max_temp_val, min_temp_val = MinMaxScalingValue.TEMPERATURE_MAX.value, MinMaxScalingValue.TEMPERATURE_MIN.value
        arr = (max_temp_val - min_temp_val) * arr + min_temp_val

    elif input_parameter == WEATHER_PARAMS.HUMIDITY.value:
        max_humidity_val, min_humidity_val = MinMaxScalingValue.HUMIDITY_MAX.value, MinMaxScalingValue.HUMIDITY_MIN.value
        arr = (max_humidity_val - min_humidity_val) * arr + min_humidity_val

    elif input_parameter == WEATHER_PARAMS.ABS_WIND.value:
        max_abs_wind_val, min_abs_wind_val = MinMaxScalingValue.ABS_WIND_MAX.value, MinMaxScalingValue.ABS_WIND_MIN.value
        arr = (max_abs_wind_val - min_abs_wind_val) * arr + min_abs_wind_val

    elif input_parameter == WEATHER_PARAMS.WIND.value:
        max_wind_val, min_wind_val = MinMaxScalingValue.WIND_MAX.value, MinMaxScalingValue.WIND_MIN.value
        arr = (max_wind_val - min_wind_val) * arr + min_wind_val

    elif input_parameter == WEATHER_PARAMS.SEALEVEL_PRESSURE.value:
        max_slp_val, min_slp_val = MinMaxScalingValue.SEALEVEL_PRESSURE_MAX.value, MinMaxScalingValue.SEALEVEL_PRESSURE_MIN.value
        arr = (max_slp_val - min_slp_val) * arr + min_slp_val

    elif input_parameter == WEATHER_PARAMS.STATION_PRESSURE.value:
        max_stp_val, min_stp_val = MinMaxScalingValue.STATION_PRESSURE_MAX.value, MinMaxScalingValue.STATION_PRESSURE_MIN.value
        arr = (max_stp_val - min_stp_val) * arr + min_stp_val

    return pd.DataFrame(arr, columns=dummy_cols)


# Sample test code
def valid_data_length(param_data_paths: Dict):
    for pa in param_data_paths.keys():
        _input = param_data_paths[pa]["input"]
        _label = param_data_paths[pa]["label"]

        try:
            assert len(_input) == 6
            assert len(_label) == 6
        except AssertionError:
            print("_input file length or _label files length is wrong")
            print("input", len(_input))
            print("label", len(_label))


def valid_path(param_data_paths: Dict):
    for pa in param_data_paths.keys():
        for typ in param_data_paths[pa].keys():
            for path in param_data_paths[pa][typ]:
                assert os.path.exists(path)
