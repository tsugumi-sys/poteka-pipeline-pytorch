import argparse
import logging
import os
import sys
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import mlflow

from src.model import train
from src.seq_to_seq import Seq2Seq, PotekaDataset, RMSELoss
from src.learning_curve_plot import learning_curve_plot

# from src.tuning import optimize_params

sys.path.append("..")
from common.data_loader import data_loader
from common.custom_logger import CustomLogger

logging.basicConfig(
    level=logging.INFO,
)
logger = CustomLogger("Train_Logger")


def start_run(
    mlflow_experiment_id: str,
    upstream_directory: str,
    downstream_directory: str,
    batch_size: int,
    epochs: int,
    optimizer_learning_rate: float,
):
    train_data_paths = os.path.join(upstream_directory, "meta_train.json")
    valid_data_paths = os.path.join(upstream_directory, "meta_valid.json")

    train_input_tensor, train_label_tensor = data_loader(train_data_paths, isMaxSizeLimit=False)
    valid_input_tensor, valid_label_tensor = data_loader(valid_data_paths, isMaxSizeLimit=False)

    train_dataset = PotekaDataset(input_tensor=train_input_tensor, label_tensor=train_label_tensor)
    valid_dataset = PotekaDataset(input_tensor=valid_input_tensor, label_tensor=valid_label_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    # best_params = optimize_params(
    #     train_dataset,
    #     valid_dataset,
    #     epochs=epochs,
    #     batch_size=batch_size,
    # )
    # best_params = {"filter_num": 32, "adam_learning_rate": optimizer_learning_rate}

    model = Seq2Seq(num_channels=3, kernel_size=3, num_kernels=32, padding=1, activation="relu", frame_size=(30, 30), num_layers=3)

    optimizer = Adam(model.parameters(), lr=optimizer_learning_rate)
    loss_criterion = nn.MSELoss(reduction="mean")
    acc_criterion = RMSELoss(reduction="mean")

    loss_only_rain = False

    results = train(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optiizer=optimizer,
        loss_criterion=loss_criterion,
        acc_criterion=acc_criterion,
        epochs=epochs,
        checkpoints_directory=os.path.join(downstream_directory, mlflow_experiment_id),
        loss_only_rain=loss_only_rain,
    )

    saved_figure_path = learning_curve_plot(
        save_dir_path=downstream_directory,
        training_losses=results["training_losses"],
        validation_losses=results["validation_losses"],
        validation_accuracy=results["validation_accuracy"],
    )

    # Save model
    mlflow.pytorch.save_model(model, "model")

    # Save results to mlflow
    mlflow.log_artifact(saved_figure_path, "training_results")

    logger.info("Training finished")


def main():
    parser = argparse.ArgumentParser(
        description="Train model",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--parent_run_name",
        type=str,
        default="defaultRun",
        help="Parent Run Name",
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

    mlflow.set_tag("mlflow.runName", args.parent_run_name + "_train & tuning")

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
