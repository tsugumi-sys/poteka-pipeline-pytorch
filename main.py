import os
import argparse

import mlflow
from logging import getLogger, basicConfig, INFO

logger = getLogger(__name__)
logger.setLevel(INFO)
basicConfig(level=INFO)


def main():
    parser = argparse.ArgumentParser(
        description="Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--preprocess_downstream",
        type=str,
        default="./preprocess/data/preprocess",
        help="preprocess downstream directory",
    )
    parser.add_argument(
        "--train_upstream",
        type=str,
        default="./preprocess/data/preprocess",
        help="upsteam directory",
    )
    parser.add_argument(
        "--train_downstream",
        type=str,
        default="../train/data/model",
        help="downstream directory",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=1,
        help="epochs",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="batch size",
    )
    parser.add_argument(
        "--train_learning_rate",
        type=float,
        default=0.001,
        help="learning rate",
    )
    parser.add_argument(
        "--train_model_type",
        type=str,
        default="simplenet",
        choices=["simplenet", "skreg"],
        help="model type.",
    )

    parser.add_argument(
        "--evaluate_downstream",
        type=str,
        default="./data/evaluate",
        help="evaluate downstream direcotry",
    )

    args = parser.parse_args()
    mlflow_experiment_id = int(os.getenv("MLFLOW_EXPERIMENT_ID", 0))

    with mlflow.start_run():
        preprocess_run = mlflow.run(
            uri="./preprocess",
            entry_point="preprocess",
            backend="local",
            parameters={
                "downstream": args.preprocess_downstream,
            },
            use_conda=False,
        )
        preprocess_run = mlflow.tracking.MlflowClient().get_run(preprocess_run.run_id)

        dataset = os.path.join(
            "../mlruns/",
            str(mlflow_experiment_id),
            preprocess_run.info.run_id,
            "artifacts/downstream_directory",
        )

        train_run = mlflow.run(
            uri="./train",
            entry_point="train",
            backend="local",
            parameters={
                "upstream": dataset,
                "downstream": args.train_downstream,
                "epochs": args.train_epochs,
                "batch_size": args.train_batch_size,
                "learning_rate": args.train_learning_rate,
                "model_type": args.train_model_type,
            },
            use_conda=False,
        )
        train_run = mlflow.tracking.MlflowClient().get_run(train_run.run_id)
        model_file_path = (
            os.path.join(train_run.info.artifact_uri, "model")
            if args.train_model_type == "simplenet"
            else os.path.join(train_run.info.artifact_uri, "skregression_model.joblib")
        )

        evaluate_run = mlflow.run(
            uri="./evaluate",
            entry_point="evaluate",
            backend="local",
            parameters={
                "downstream": args.evaluate_downstream,
                "valid_data_directory": os.path.join(dataset, "valid"),
                "train_data_directory": os.path.join(dataset, "train"),
                "model_file_path": model_file_path,
                "model_type": args.train_model_type,
            },
            use_conda=False,
        )
        evaluate_run = mlflow.tracking.MlflowClient().get_run(evaluate_run.run_id)


if __name__ == "__main__":
    main()
