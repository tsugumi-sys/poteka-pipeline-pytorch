import argparse
import os
import json

import random
import mlflow
from src.extract_data import data_file_path


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
    _data_file_path = data_file_path(params, delta, slides)
    random.Random(12345).shuffle(_data_file_path)

    split_length = int(len(_data_file_path) * 0.7)
    train_file_paths, test_file_paths = _data_file_path[:split_length], _data_file_path[split_length:]

    meta_train = {"file_paths": train_file_paths}
    meta_test = {"file_paths": test_file_paths}

    meta_train_filepath = os.path.join(
        downstream_directory,
        "meta_train.json",
    )
    meta_test_filepath = os.path.join(
        downstream_directory,
        "meta_test.json",
    )

    with open(meta_train_filepath, "w") as f:
        json.dump(meta_train, f)
    with open(meta_test_filepath, "w") as f:
        json.dump(meta_test, f)

    mlflow.log_artifacts(
        downstream_directory,
        artifact_path="downstream_directory",
    )


if __name__ == "__main__":
    main()
