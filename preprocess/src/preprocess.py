import argparse
import os
import json
import logging

import random
import mlflow
from sklearn.model_selection import train_test_split
from src.extract_data import data_file_path

import sys

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

    downstream_directory = args.downstream
    os.makedirs(downstream_directory, exist_ok=True)

    params = args.params.split()
    delta = args.delta
    slides = args.slides

    _data_file_path = data_file_path(params=params, delta=delta, slides=slides)
    train_file_paths, test_file_paths = train_test_split(_data_file_path, test_size=0.2, random_state=11)
    valid_file_paths = data_file_path(params=params, delta=delta, isTrain=False)

    meta_train = {"file_paths": train_file_paths}
    meta_test = {"file_paths": test_file_paths}
    meta_valid = {"file_paths": valid_file_paths}

    meta_train_filepath = os.path.join(
        downstream_directory,
        "meta_train.json",
    )
    meta_test_filepath = os.path.join(
        downstream_directory,
        "meta_test.json",
    )
    meta_valid_filepath = os.path.join(
        downstream_directory,
        "meta_valid.json",
    )

    with open(meta_train_filepath, "w") as f:
        json.dump(meta_train, f)
    with open(meta_test_filepath, "w") as f:
        json.dump(meta_test, f)
    with open(meta_valid_filepath, "w") as f:
        json.dump(meta_valid, f)

    mlflow.log_artifacts(
        downstream_directory,
        artifact_path="downstream_directory",
    )
    logger.info(f"meta info files have saved in {downstream_directory}")


if __name__ == "__main__":
    main()
