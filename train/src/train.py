import argparse
import logging
import os

import mlflow
import tensorflow as tf
from src.model import Simple_ConvLSTM, train, evaluate
from src.data_loader import sample_data_loader, data_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def start_run(
    mlflow_experiment_id: str,
    upstream_directory: str,
    downstream_directory: str,
    batch_size: int,
    epochs: int,
    optimizer_learning_rate: float,
):
    train_data_paths = os.path.join(upstream_directory, "meta_train.json")
    test_data_paths = os.path.join(upstream_directory, "meta_test.json")

    train_dataset = data_loader(train_data_paths, isLimit=True)
    test_dataset = data_loader(test_data_paths)

    model = Simple_ConvLSTM(
        feature_num=train_dataset[0].shape[-1],
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=optimizer_learning_rate)

    mlflow.tensorflow.autolog()
    model_file_path = train(
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
        test_dataset=test_dataset,
    )

    logger.info(f"Latest performance: Accuracy: {accuracy}, Loss: {loss}")
    logger.info(f"Model saved at {model_file_path}")

    # [TODO] why autlog run is got deactivated?
    # after tensorflow.autolog, the run is got deactivated
    # before log metric and loss, then save model.

    # mlflow.log_artifacts(model_file_path, "model")
    # print("ACTIVE RUN OBJECT", active_run_obj)
    # if active_run_obj:
    #     with mlflow.start_run(run_id=active_run_obj.info.run_id):
    #         mlflow.log_artifacts(model_path, "model")


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
        "--optimizer_learning_rate",
        type=float,
        default=0.001,
        help="optimizer learning rate",
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
        optimizer_learning_rate=args.optimizer_learning_rate,
    )


if __name__ == "__main__":
    main()
