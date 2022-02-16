import argparse
import os
import json
import logging
import sys

import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from src.extract_data import get_train_data_files, get_test_data_files  # , data_file_path

sys.path.append("..")
from common.custom_logger import CustomLogger

logging.basicConfig(
    level=logging.INFO,
)
logger = CustomLogger("Preprocess_Logger")


def main():
    parser = argparse.ArgumentParser(
        description="Make dataset",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--parent_run_name",
        type=str,
        default="defaultRun",
        help="Parent Run Name",
    )
    parser.add_argument(
        "--downstream",
        type=str,
        default="/data/preprocess/",
        help="downstream directory",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="rain humidity temperature wind",
        help="input params",
    )
    parser.add_argument(
        "--delta",
        type=int,
        default=10,
        help="time resolution of the data. This should be even number. (max, min) = (10, 2)",
    )
    parser.add_argument(
        "--slides",
        type=int,
        default=3,
        help="slide number of collecting data.",
    )

    args = parser.parse_args()

    mlflow.set_tag("mlflow.runName", args.parent_run_name + "_preprcess")

    downstream_directory = args.downstream
    os.makedirs(downstream_directory, exist_ok=True)

    params = args.params.split()
    delta = args.delta
    slides = args.slides

    # _data_file_path = data_file_path(params=params, delta=delta, slides=slides)
    # train_file_paths, valid_file_paths = train_test_split(_data_file_path, test_size=0.2, random_state=11)
    # test_file_paths = data_file_path(params=params, delta=delta, isTrain=False)

    # train_dataset.csv comes from https://github.com/tsugumi-sys/poteka_data_analysis/blob/main/EDA/rain/rain_durations.ipynb
    current_dir = os.getcwd()
    train_list_df = pd.read_csv(os.path.join(current_dir, "src/train_dataset.csv"))
    train_data_files = get_train_data_files(train_list_df=train_list_df, params=params, delta=delta, slides=slides)
    train_data_files, valid_data_files = train_test_split(train_data_files, test_size=0.2, random_state=11)

    # test_dataset.json comes from https://github.com/tsugumi-sys/poteka_data_analysis/blob/main/EDA/rain/select_test_dataset.ipynb
    with open(os.path.join(current_dir, "src/test_dataset.json")) as f:
        test_data_list = json.load(f)
    test_data_files = get_test_data_files(test_data_list=test_data_list, params=params, delta=delta)

    meta_train = {"file_paths": train_data_files}
    meta_valid = {"file_paths": valid_data_files}
    meta_test = {"file_paths": test_data_files}

    meta_train_filepath = os.path.join(
        downstream_directory,
        "meta_train.json",
    )
    meta_valid_filepath = os.path.join(
        downstream_directory,
        "meta_valid.json",
    )
    meta_test_filepath = os.path.join(
        downstream_directory,
        "meta_test.json",
    )

    with open(meta_train_filepath, "w") as f:
        json.dump(meta_train, f)
    with open(meta_valid_filepath, "w") as f:
        json.dump(meta_valid, f)
    with open(meta_test_filepath, "w") as f:
        json.dump(meta_test, f)

    mlflow.log_artifacts(
        downstream_directory,
        artifact_path="downstream_directory",
    )
    logger.info(f"meta info files have saved in {downstream_directory}")


if __name__ == "__main__":
    main()
