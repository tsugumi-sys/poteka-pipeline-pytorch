import argparse
import logging
import os

import mlflow
import pandas as pd
import tensorflow as tf
from src.model import SimpleNet, SKRegressor, train, evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def start_run(
    mlflow_experiment_id: str,
    upstream_directory: str,
    downstream_directory: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    model_type: str,
):
    train_dataset = pd.read_parquet(
        os.path.join(upstream_directory, "train"),
        engine="pyarrow",
    )
    test_dataset = pd.read_parquet(
        os.path.join(upstream_directory, "test"),
        engine="pyarrow",
    )

    if model_type == "gbr":
        params = {"alpha": learning_rate}
        model = SKRegressor(params)
    elif model_type == "simplenet":
        model = SimpleNet(input_shape=[len(train_dataset.columns) - 1])
    else:
        raise ValueError("Unknown model.")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model_path = train(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        checkpoints_directory=os.path.join(downstream_directory, mlflow_experiment_id),
    )

    accuracy, loss = evaluate(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )

    logger.info(f"Latest performance: Accuracy: {accuracy}, Loss: {loss}")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("loss", loss)
    mlflow.log_artifact(model_path)


def main():
    parser = argparse.ArgumentParser(
        description="Train model",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--upstream",
        type=str,
        default="/data/preprocess",
        help="upstream directory",
    )
    parser.add_argument(
        "--downstream",
        type=str,
        default="/model/",
        help="downstream directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="learning rate",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="gbr",
        choices=[
            "gbr",
            "simplenet",
        ],
        help="simplenet, gbr",
    )
    args = parser.parse_args()
    mlflow_experiment_id = str(os.getenv("MLFLOW_EXPERIMENT_ID", 0))

    upstream_directory = args.upstream
    downstream_directory = args.downstream

    os.makedirs(downstream_directory, exist_ok=True)

    start_run(
        mlflow_experiment_id=mlflow_experiment_id,
        upstream_directory=upstream_directory,
        downstream_directory=downstream_directory,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
