import os
import json
import logging
import sys
from typing import Dict

import hydra
from omegaconf import DictConfig
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from src.extract_data import get_train_data_files, get_test_data_files
from src.extract_dummy_data import get_dummy_data_files, get_meta_test_info

sys.path.append("..")
from common.custom_logger import CustomLogger
from common.utils import get_mlflow_tag_from_input_parameters, split_input_parameters_str

logging.basicConfig(
    level=logging.INFO,
)
logger = CustomLogger("Preprocess_Logger")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    input_parameters = split_input_parameters_str(cfg.input_parameters)
    mlflow.set_tag("mlflow.runName", get_mlflow_tag_from_input_parameters(input_parameters) + "_preprcess")

    downstream_dir_path: str = cfg.preprocess.downstream_dir_path
    os.makedirs(downstream_dir_path, exist_ok=True)

    time_step_minutes = cfg.preprocess.time_step_minutes
    time_slides_delta = cfg.preprocess.time_slides_delta

    if cfg.use_dummy_data is True:
        logger.warning("... Using dummy data ...")
        data_files = get_dummy_data_files(
            input_parameters=input_parameters,
            time_step_minutes=time_step_minutes,
            downstream_dir_path=downstream_dir_path,
            dataset_length=100,
            input_seq_length=cfg.input_seq_length,
            label_seq_length=cfg.label_seq_length,
        )

        data_file_length = len(data_files)
        train_data_size = data_file_length // 2
        valid_test_data_size = data_file_length // 4
        train_data_files, valid_data_files, test_data_files = (
            data_files[:train_data_size],
            data_files[train_data_size : train_data_size + valid_test_data_size],  # noqa: E203
            data_files[train_data_size + valid_test_data_size :],  # noqa: E203
        )

    else:
        # train_dataset.csv comes from https://github.com/tsugumi-sys/poteka_data_analysis/blob/main/EDA/rain/rain_durations.ipynb
        current_dir = os.getcwd()
        train_list_df = pd.read_csv(os.path.join(current_dir, "src/train_dataset.csv"))
        train_data_files = get_train_data_files(
            train_list_df=train_list_df,
            input_parameters=input_parameters,
            time_step_minutes=time_step_minutes,
            time_slides_delta=time_slides_delta,
            input_seq_length=cfg.input_seq_length,
            label_seq_length=cfg.label_seq_length,
        )
        train_data_files, valid_data_files = train_test_split(train_data_files, test_size=0.2, random_state=11)

        # test_dataset.json comes from https://github.com/tsugumi-sys/poteka_data_analysis/blob/main/EDA/rain/select_test_dataset.ipynb
        with open(os.path.join(current_dir, "src/test_dataset.json")) as f:
            test_data_list = json.load(f)
        test_data_files = get_test_data_files(
            test_data_list=test_data_list,
            input_parameters=input_parameters,
            time_step_minutes=time_step_minutes,
            input_seq_length=cfg.input_seq_length,
            label_seq_length=cfg.label_seq_length,
        )

    meta_train = {"file_paths": train_data_files}
    meta_valid = {"file_paths": valid_data_files}
    meta_test = {"file_paths": test_data_files if isinstance(test_data_files, Dict) else get_meta_test_info(test_data_files, cfg.label_seq_length)}

    meta_train_filepath = os.path.join(
        downstream_dir_path,
        "meta_train.json",
    )
    meta_valid_filepath = os.path.join(
        downstream_dir_path,
        "meta_valid.json",
    )
    meta_test_filepath = os.path.join(
        downstream_dir_path,
        "meta_test.json",
    )

    with open(meta_train_filepath, "w") as f:
        json.dump(meta_train, f)
    with open(meta_valid_filepath, "w") as f:
        json.dump(meta_valid, f)
    with open(meta_test_filepath, "w") as f:
        json.dump(meta_test, f)

    for path in [meta_train_filepath, meta_valid_filepath, meta_test_filepath]:
        mlflow.log_artifact(path)
    logger.info("meta info files have saved")


if __name__ == "__main__":
    main()
