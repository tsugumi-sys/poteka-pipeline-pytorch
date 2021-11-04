import logging
from typing import Dict, List
import sys
import os

import pandas as pd
from src.constants import WEATHER_PARAMS_ENUM, DIRECTORYS

sys.path.append("..")
from common.utils import timestep_csv_names, param_date_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :%(message)s",
)
logger = logging.getLogger(__name__)


def data_file_path(params: List[str] = ["rain", "humidity", "temperature", "wind"], delta: int = 10, slides: int = 3) -> List[Dict]:
    if "rain" not in params:
        logger.error(f"rain is not in {params}")
        raise ValueError("preprocess_params should have 'rain'.")

    if not WEATHER_PARAMS_ENUM.is_params_valid(params):
        logger.error(f"{params} is invalid name.")
        raise ValueError(f"preprocess_params should be in {WEATHER_PARAMS_ENUM.valid_params()}")

    current_dir = os.getcwd()
    train_list = pd.read_csv(
        os.path.join(current_dir, "src/train_data_list.csv"),
        index_col="date",
    )

    _timestep_csv_names = timestep_csv_names(delta=delta)
    paths = []

    for date in train_list.index:
        year, month = date.split("-")[0], date.split("-")[1]

        params_date_paths = {}
        if len(params) > 0:
            for pa in params:
                params_date_paths[pa] = os.path.join(
                    DIRECTORYS.project_root_dir,
                    param_date_path(pa, year, month, date),
                )

        start, end = train_list.loc[date, "start"], train_list.loc[date, "end"]
        idx_start, idx_end = _timestep_csv_names.index(str(start)), _timestep_csv_names.index(str(end))
        idx_start = idx_start - 12 if idx_start > 11 else 0
        idx_end = idx_end + 12 if idx_end < len(_timestep_csv_names) - 12 else len(_timestep_csv_names) - 1
        for i in range(idx_start, idx_end - 12, slides):
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


# Sample test code
def valid_data_length(param_data_paths: Dict):
    for pa in param_data_paths.keys():
        _input = param_data_paths[pa]["input"]
        _label = param_data_paths[pa]["label"]

        try:
            assert len(_input) == 6
            assert len(_label) == 6
        except AssertionError:
            print("_input file length or _label fle length is wrong")
            print("input", _input)
            print("label", _label)


def valid_path(param_data_paths: Dict):
    for pa in param_data_paths.keys():
        for typ in param_data_paths[pa].keys():
            for path in param_data_paths[pa][typ]:
                assert os.path.exists(path)


if __name__ == "__main__":
    res = data_file_path()
    for item in res:
        valid_data_length(item)
        valid_path(item)
