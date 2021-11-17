import json
import logging
from typing import Dict, List
import sys
import os

import pandas as pd
from src.constants import WEATHER_PARAMS_ENUM, DIRECTORYS

sys.path.append("..")
from common.utils import timestep_csv_names, param_date_path

logger = logging.getLogger(__name__)


def data_file_path(
    params: List[str] = ["rain", "humidity", "temperature", "wind"],
    isTrain=True,
    delta: int = 10,
    slides: int = 3,
) -> List[Dict]:
    if "rain" not in params:
        logger.error(f"rain is not in {params}")
        raise ValueError("preprocess_params should have 'rain'.")

    if not WEATHER_PARAMS_ENUM.is_params_valid(params):
        logger.error(f"{params} is invalid name.")
        raise ValueError(f"preprocess_params should be in {WEATHER_PARAMS_ENUM.valid_params()}")

    current_dir = os.getcwd()
    if isTrain:
        # [TODO]
        # Shoud I use json file?
        train_list = pd.read_csv(
            os.path.join(current_dir, "src/train_dataset.csv"),
        )

        _timestep_csv_names = timestep_csv_names(delta=delta)
        paths = []
        for idx in train_list.index:
            date = train_list.loc[idx, "date"]
            year, month = date.split("-")[0], date.split("-")[1]

            params_date_paths = {}
            if len(params) > 0:
                for pa in params:
                    params_date_paths[pa] = os.path.join(
                        DIRECTORYS.project_root_dir,
                        param_date_path(pa, year, month, date),
                    )

            start, end = train_list.loc[idx, "start_time"], train_list.loc[idx, "end_time"]
            idx_start, idx_end = _timestep_csv_names.index(str(start) + ".csv"), _timestep_csv_names.index(str(end) + ".csv")
            idx_start = idx_start - 7 if idx_start > 6 else 0
            idx_end = idx_end + 7 if idx_end < len(_timestep_csv_names) - 7 else len(_timestep_csv_names) - 1
            for i in range(idx_start, idx_end - 7, slides):
                next_i = i + 7
                h_m_csv_names = _timestep_csv_names[i:next_i]

                _tmp = {}
                for pa in params:
                    if pa == "wind":
                        for name in ["u_wind", "v_wind"]:
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
                    for pa in params:
                        if "wind" == pa:
                            _tmp["u_wind"]["input"] += [params_date_paths[pa] + f"/{h_m_csv_name}".replace(".csv", "U.csv")]
                            _tmp["v_wind"]["input"] += [params_date_paths[pa] + f"/{h_m_csv_name}".replace(".csv", "V.csv")]
                        else:
                            _tmp[pa]["input"] += [params_date_paths[pa] + f"/{h_m_csv_name}"]

                # for h_m_csv_name in h_m_csv_names[6]:
                label_h_m_csv_name = h_m_csv_names[6]
                for pa in params:
                    if "wind" == pa:
                        _tmp["u_wind"]["label"] += [params_date_paths[pa] + f"/{label_h_m_csv_name}".replace(".csv", "U.csv")]
                        _tmp["v_wind"]["label"] += [params_date_paths[pa] + f"/{label_h_m_csv_name}".replace(".csv", "V.csv")]
                    else:
                        _tmp[pa]["label"] += [params_date_paths[pa] + f"/{label_h_m_csv_name}"]

                paths.append(_tmp)
        return paths

    else:
        f = open(os.path.join(current_dir, "src/valid_dataset.json"))
        valid_data_list = json.load(f)

        _timestep_csv_names = timestep_csv_names(delta=delta)
        paths = {}
        for case_name in valid_data_list.keys():
            for sample_name in valid_data_list[case_name].keys():
                for idx in valid_data_list[case_name][sample_name].keys():
                    sample_info = valid_data_list[case_name][sample_name][idx]
                    date = sample_info["date"]
                    year, month = date.split("-")[0], date.split("-")[1]

                    params_date_paths = {}
                    if len(params) > 0:
                        for pa in params:
                            params_date_paths[pa] = os.path.join(
                                DIRECTORYS.project_root_dir,
                                param_date_path(pa, year, month, date),
                            )

                    start = sample_info["start"]
                    idx_start = _timestep_csv_names.index(str(start))
                    idx_end = idx_start + 12
                    h_m_csv_names = _timestep_csv_names[idx_start:idx_end]

                    _tmp = {}
                    for pa in params:
                        if pa == "wind":
                            for name in ["u_wind", "v_wind"]:
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
                        for pa in params:
                            if "wind" == pa:
                                _tmp["u_wind"]["input"] += [params_date_paths[pa] + f"/{h_m_csv_name}".replace(".csv", "U.csv")]
                                _tmp["v_wind"]["input"] += [params_date_paths[pa] + f"/{h_m_csv_name}".replace(".csv", "V.csv")]
                            else:
                                _tmp[pa]["input"] += [params_date_paths[pa] + f"/{h_m_csv_name}"]

                    # Load label data
                    # contains other parameters value for sequential prediction
                    for label_h_m_csv_name in h_m_csv_names[6:]:
                        for pa in params:
                            if "wind" == pa:
                                _tmp["u_wind"]["label"] += [params_date_paths[pa] + f"/{label_h_m_csv_name}".replace(".csv", "U.csv")]
                                _tmp["v_wind"]["label"] += [params_date_paths[pa] + f"/{label_h_m_csv_name}".replace(".csv", "V.csv")]
                            else:
                                _tmp[pa]["label"] += [params_date_paths[pa] + f"/{label_h_m_csv_name}"]

                    _sample_name = f"{case_name}_{date}_{start.replace('.csv', '')}_start"
                    paths[_sample_name] = _tmp
                    paths[_sample_name]["date"] = date
                    paths[_sample_name]["start"] = start
    return paths


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


if __name__ == "__main__":
    res = data_file_path(isTrain=False)
    for key in res.keys():
        item = res[key]
        valid_data_length(item)
        valid_path(item)
