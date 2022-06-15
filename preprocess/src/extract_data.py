import json
import logging
from typing import Dict, List
import sys
import os

import pandas as pd

sys.path.append("..")
from common.utils import timestep_csv_names, param_date_path
from common.config import WEATHER_PARAMS, DIRECTORYS

logger = logging.getLogger(__name__)


def get_train_data_files(
    train_list_df: pd.DataFrame,
    input_parameters: List[str] = ["rain", "temperature"],
    time_step_minutes: int = 10,
    time_slides_delta: int = 3,
) -> List[Dict]:
    """Get train data file paths

    Args:
        train_list_df (pd.DataFrame): pandas.DataFrame with training data informations with columns ['date', 'start_time', 'end_time']
        input_parameters (List[str], optional): Input parameters list. Defaults to ["rain", "temperature"].
        time_step_minutes (int, optional): Time step of datesets in minutes. Defaults to 10.
        time_slides_delta (int, optional): Time slides used for creating datesets in minutes. Defaults to 3.

    Raises:
        ValueError: `rain` must be in `input_parameters`.
        ValueError: check if all input parameters in `input_paramerers` are valid.

    Returns:
        List[Dict]: list of dictioaryis contains data file paths of each input tarameters.
            {
                "rain": {"input": ['path/to/datafiles/0-0.csv', 'path/to/datafiles/0-10.csv', ...], "label": [...]},
                "temperature": {"input": [...], "label": [...]},
                ...
            }
    """
    if WEATHER_PARAMS.RAIN.value not in input_parameters:
        logger.error(f"rain is not in {input_parameters}")
        raise ValueError("input_parameters should have 'rain'.")

    if not WEATHER_PARAMS.is_params_valid(input_parameters):
        logger.error(f"{input_parameters} is invalid name.")
        raise ValueError(f"preprocess_input_parameters should be in {WEATHER_PARAMS.valid_params()}")

    _timestep_csv_names = timestep_csv_names(time_step_minutes=time_step_minutes)
    paths = []
    for idx in train_list_df.index:
        date = train_list_df.loc[idx, "date"]
        year, month = date.split("-")[0], date.split("-")[1]

        input_parameters_date_paths = {}
        if len(input_parameters) > 0:
            for pa in input_parameters:
                input_parameters_date_paths[pa] = os.path.join(
                    DIRECTORYS.project_root_dir,
                    param_date_path(pa, year, month, date),
                )

        start, end = train_list_df.loc[idx, "start_time"], train_list_df.loc[idx, "end_time"]
        idx_start, idx_end = _timestep_csv_names.index(str(start) + ".csv"), _timestep_csv_names.index(str(end) + ".csv")
        idx_start = idx_start - 7 if idx_start > 6 else 0
        idx_end = idx_end + 7 if idx_end < len(_timestep_csv_names) - 7 else len(_timestep_csv_names) - 1
        for i in range(idx_start, idx_end - 7, time_slides_delta):
            next_i = i + 7
            h_m_csv_names = _timestep_csv_names[i:next_i]

            _tmp = {}
            for pa in input_parameters:
                if pa == WEATHER_PARAMS.WIND.value:
                    for name in [WEATHER_PARAMS.U_WIND.value, WEATHER_PARAMS.V_WIND.value]:
                        _tmp[name] = {
                            "input": [],
                            "label": [],
                        }
                else:
                    _tmp[pa] = {
                        "input": [],
                        "label": [],
                    }

            for h_m_csv_name in h_m_csv_names[:6]:
                for pa in input_parameters:
                    if pa == WEATHER_PARAMS.WIND.value:
                        _tmp[WEATHER_PARAMS.U_WIND.value]["input"] += [input_parameters_date_paths[pa] + f"/{h_m_csv_name}".replace(".csv", "U.csv")]
                        _tmp[WEATHER_PARAMS.V_WIND.value]["input"] += [input_parameters_date_paths[pa] + f"/{h_m_csv_name}".replace(".csv", "V.csv")]
                    else:
                        _tmp[pa]["input"] += [input_parameters_date_paths[pa] + f"/{h_m_csv_name}"]

            # for h_m_csv_name in h_m_csv_names[6]:
            label_h_m_csv_name = h_m_csv_names[6]
            for pa in input_parameters:
                if pa == WEATHER_PARAMS.WIND.value:
                    _tmp[WEATHER_PARAMS.U_WIND.value]["label"] += [input_parameters_date_paths[pa] + f"/{label_h_m_csv_name}".replace(".csv", "U.csv")]
                    _tmp[WEATHER_PARAMS.V_WIND.value]["label"] += [input_parameters_date_paths[pa] + f"/{label_h_m_csv_name}".replace(".csv", "V.csv")]
                else:
                    _tmp[pa]["label"] += [input_parameters_date_paths[pa] + f"/{label_h_m_csv_name}"]

            paths.append(_tmp)
    return paths


# Deprecated!!
def get_test_data_files(
    test_data_list: Dict,
    input_parameters: List[str] = ["rain", "temperature"],
    time_step_minutes: int = 10,
) -> Dict:
    """Get test data file informations

    Args:
        test_data_list (Dict): test data information contains date, start, end.
        input_parameters (List[str], optional): Input parameters list. Defaults to ["rain", "temperature"].
        time_step_minutes (int, optional): time step minutes. Defaults to 10.

    Raises:
        ValueError: when rain is not in input_parameters
        ValueError: when invalid parameter name contains

    Returns:
        Dict: data file paths of each test cases like following.
            {
                "case1": {
                    "rain": {"input": ['path/to/datafiles/0-0.csv', 'path/to/datafiles/0-10.csv', ...], "label": [...]},
                    "temperature": {"input": [...], "label": [...]},
                    ...,
                    "date": "2020/01/05",
                    "start": "1000UTC",
                },
                "case2": {...}
            }
    """
    if WEATHER_PARAMS.RAIN.value not in input_parameters:
        logger.error(f"rain is not in {input_parameters}")
        raise ValueError("preprocess_input_parameters should have 'rain'.")

    if not WEATHER_PARAMS.is_params_valid(input_parameters):
        logger.error(f"{input_parameters} is invalid name.")
        raise ValueError(f"preprocess_input_parameters should be in {WEATHER_PARAMS.is_params_valid()}")

    _timestep_csv_names = timestep_csv_names(time_step_minutes=time_step_minutes)
    paths = {}
    for case_name in test_data_list.keys():
        for sample_name in test_data_list[case_name].keys():
            for idx in test_data_list[case_name][sample_name].keys():
                sample_info = test_data_list[case_name][sample_name][idx]
                date = sample_info["date"]
                year, month = date.split("-")[0], date.split("-")[1]

                input_parameters_date_paths = {}
                if len(input_parameters) > 0:
                    for pa in input_parameters:
                        input_parameters_date_paths[pa] = os.path.join(
                            DIRECTORYS.project_root_dir,
                            param_date_path(pa, year, month, date),
                        )

                start = sample_info["start"]
                idx_start = _timestep_csv_names.index(str(start))
                idx_end = idx_start + 12
                h_m_csv_names = _timestep_csv_names[idx_start:idx_end]

                _tmp = {}
                for pa in input_parameters:
                    if pa == WEATHER_PARAMS.WIND.value:
                        for name in [WEATHER_PARAMS.U_WIND.value, WEATHER_PARAMS.V_WIND.value]:
                            _tmp[name] = {
                                "input": [],
                                "label": [],
                            }
                    else:
                        _tmp[pa] = {
                            "input": [],
                            "label": [],
                        }

                # Load input data
                for h_m_csv_name in h_m_csv_names[:6]:
                    for pa in input_parameters:
                        if pa == WEATHER_PARAMS.WIND.value:
                            _tmp[WEATHER_PARAMS.U_WIND.value]["input"] += [input_parameters_date_paths[pa] + f"/{h_m_csv_name}".replace(".csv", "U.csv")]
                            _tmp[WEATHER_PARAMS.V_WIND.value]["input"] += [input_parameters_date_paths[pa] + f"/{h_m_csv_name}".replace(".csv", "V.csv")]
                        else:
                            _tmp[pa]["input"] += [input_parameters_date_paths[pa] + f"/{h_m_csv_name}"]

                # Load label data
                # contains other parameters value for sequential prediction
                for label_h_m_csv_name in h_m_csv_names[6:]:
                    for pa in input_parameters:
                        if pa == WEATHER_PARAMS.WIND.value:
                            _tmp[WEATHER_PARAMS.U_WIND.value]["label"] += [input_parameters_date_paths[pa] + f"/{label_h_m_csv_name}".replace(".csv", "U.csv")]
                            _tmp[WEATHER_PARAMS.V_WIND.value]["label"] += [input_parameters_date_paths[pa] + f"/{label_h_m_csv_name}".replace(".csv", "V.csv")]
                        else:
                            _tmp[pa]["label"] += [input_parameters_date_paths[pa] + f"/{label_h_m_csv_name}"]

                _sample_name = f"{case_name}_{date}_{start.replace('.csv', '')}_start"
                paths[_sample_name] = _tmp
                paths[_sample_name]["date"] = date
                paths[_sample_name]["start"] = start
    return paths


def data_file_path(
    input_parameters: List[str] = ["rain", "temperature"],
    isTrain=True,
    time_step_minutes: int = 10,
    time_slides_delta: int = 3,
) -> List[Dict]:
    if WEATHER_PARAMS.RAIN.value not in input_parameters:
        logger.error(f"rain is not in {input_parameters}")
        raise ValueError("preprocess_input_parameters should have 'rain'.")

    if not WEATHER_PARAMS.is_input_parameters_valid(input_parameters):
        logger.error(f"{input_parameters} is invalid name.")
        raise ValueError(f"preprocess_input_parameters should be in {WEATHER_PARAMS.is_params_valid()}")

    current_dir = os.getcwd()
    if isTrain:
        # [TODO]
        # Shoud I use json file?
        train_list = pd.read_csv(
            os.path.join(current_dir, "src/train_dataset.csv"),
        )

        _timestep_csv_names = timestep_csv_names(time_step_minutes=time_step_minutes)
        paths = []
        for idx in train_list.index:
            date = train_list.loc[idx, "date"]
            year, month = date.split("-")[0], date.split("-")[1]

            input_parameters_date_paths = {}
            if len(input_parameters) > 0:
                for pa in input_parameters:
                    input_parameters_date_paths[pa] = os.path.join(
                        DIRECTORYS.project_root_dir,
                        param_date_path(pa, year, month, date),
                    )

            start, end = train_list.loc[idx, "start_time"], train_list.loc[idx, "end_time"]
            idx_start, idx_end = _timestep_csv_names.index(str(start) + ".csv"), _timestep_csv_names.index(str(end) + ".csv")
            idx_start = idx_start - 7 if idx_start > 6 else 0
            idx_end = idx_end + 7 if idx_end < len(_timestep_csv_names) - 7 else len(_timestep_csv_names) - 1
            for i in range(idx_start, idx_end - 7, time_slides_delta):
                next_i = i + 7
                h_m_csv_names = _timestep_csv_names[i:next_i]

                _tmp = {}
                for pa in input_parameters:
                    if pa == WEATHER_PARAMS.WIND.value:
                        for name in [WEATHER_PARAMS.U_WIND.value, WEATHER_PARAMS.V_WIND.value]:
                            _tmp[name] = {
                                "input": [],
                                "label": [],
                            }
                    else:
                        _tmp[pa] = {
                            "input": [],
                            "label": [],
                        }

                for h_m_csv_name in h_m_csv_names[:6]:
                    for pa in input_parameters:
                        if pa == WEATHER_PARAMS.WIND.value:
                            _tmp[WEATHER_PARAMS.U_WIND.value]["input"] += [input_parameters_date_paths[pa] + f"/{h_m_csv_name}".replace(".csv", "U.csv")]
                            _tmp[WEATHER_PARAMS.V_WIND.value]["input"] += [input_parameters_date_paths[pa] + f"/{h_m_csv_name}".replace(".csv", "V.csv")]
                        else:
                            _tmp[pa]["input"] += [input_parameters_date_paths[pa] + f"/{h_m_csv_name}"]

                # for h_m_csv_name in h_m_csv_names[6]:
                label_h_m_csv_name = h_m_csv_names[6]
                for pa in input_parameters:
                    if pa == WEATHER_PARAMS.WIND.value:
                        _tmp[WEATHER_PARAMS.U_WIND.value]["label"] += [input_parameters_date_paths[pa] + f"/{label_h_m_csv_name}".replace(".csv", "U.csv")]
                        _tmp[WEATHER_PARAMS.V_WIND.value]["label"] += [input_parameters_date_paths[pa] + f"/{label_h_m_csv_name}".replace(".csv", "V.csv")]
                    else:
                        _tmp[pa]["label"] += [input_parameters_date_paths[pa] + f"/{label_h_m_csv_name}"]

                paths.append(_tmp)
        return paths

    else:
        f = open(os.path.join(current_dir, "src/test_dataset.json"))
        test_data_list = json.load(f)

        _timestep_csv_names = timestep_csv_names(time_step_minutes=time_step_minutes)
        paths = {}
        for case_name in test_data_list.keys():
            for sample_name in test_data_list[case_name].keys():
                for idx in test_data_list[case_name][sample_name].keys():
                    sample_info = test_data_list[case_name][sample_name][idx]
                    date = sample_info["date"]
                    year, month = date.split("-")[0], date.split("-")[1]

                    input_parameters_date_paths = {}
                    if len(input_parameters) > 0:
                        for pa in input_parameters:
                            input_parameters_date_paths[pa] = os.path.join(
                                DIRECTORYS.project_root_dir,
                                param_date_path(pa, year, month, date),
                            )

                    start = sample_info["start"]
                    idx_start = _timestep_csv_names.index(str(start))
                    idx_end = idx_start + 12
                    h_m_csv_names = _timestep_csv_names[idx_start:idx_end]

                    _tmp = {}
                    for pa in input_parameters:
                        if pa == WEATHER_PARAMS.WIND.value:
                            for name in [WEATHER_PARAMS.U_WIND.value, WEATHER_PARAMS.V_WIND.value]:
                                _tmp[name] = {
                                    "input": [],
                                    "label": [],
                                }
                        else:
                            _tmp[pa] = {
                                "input": [],
                                "label": [],
                            }

                    # Load input data
                    for h_m_csv_name in h_m_csv_names[:6]:
                        for pa in input_parameters:
                            if pa == WEATHER_PARAMS.WIND.value:
                                _tmp[WEATHER_PARAMS.U_WIND.value]["input"] += [input_parameters_date_paths[pa] + f"/{h_m_csv_name}".replace(".csv", "U.csv")]
                                _tmp[WEATHER_PARAMS.V_WIND.value]["input"] += [input_parameters_date_paths[pa] + f"/{h_m_csv_name}".replace(".csv", "V.csv")]
                            else:
                                _tmp[pa]["input"] += [input_parameters_date_paths[pa] + f"/{h_m_csv_name}"]

                    # Load label data
                    # contains other parameters value for sequential prediction
                    for label_h_m_csv_name in h_m_csv_names[6:]:
                        for pa in input_parameters:
                            if pa == WEATHER_PARAMS.WIND.value:
                                _tmp[WEATHER_PARAMS.U_WIND.value]["label"] += [
                                    input_parameters_date_paths[pa] + f"/{label_h_m_csv_name}".replace(".csv", "U.csv")
                                ]
                                _tmp[WEATHER_PARAMS.V_WIND.value]["label"] += [
                                    input_parameters_date_paths[pa] + f"/{label_h_m_csv_name}".replace(".csv", "V.csv")
                                ]
                            else:
                                _tmp[pa]["label"] += [input_parameters_date_paths[pa] + f"/{label_h_m_csv_name}"]

                    _sample_name = f"{case_name}_{date}_{start.replace('.csv', '')}_start"
                    paths[_sample_name] = _tmp
                    paths[_sample_name]["date"] = date
                    paths[_sample_name]["start"] = start
    return paths
