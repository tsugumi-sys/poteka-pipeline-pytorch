import argparse
import logging
import os
import sys
from torch import nn
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import mlflow

from src.model import train
from src.seq_to_seq import Seq2Seq, PotekaDataset, RMSELoss
from src.learning_curve_plot import learning_curve_plot
from src.config import DEVICE

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

    train_input_tensor, train_label_tensor = data_loader(train_data_paths, isMaxSizeLimit=True)
    valid_input_tensor, valid_label_tensor = data_loader(valid_data_paths, isMaxSizeLimit=True)

    train_input_tensor, train_label_tensor = train_input_tensor.to(DEVICE), train_label_tensor.to(DEVICE)
    valid_input_tensor, valid_label_tensor = valid_input_tensor.to(DEVICE), valid_label_tensor.to(DEVICE)

    train_dataset = PotekaDataset(input_tensor=train_input_tensor, label_tensor=train_label_tensor)
    valid_dataset = PotekaDataset(input_tensor=valid_input_tensor, label_tensor=valid_label_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    num_channels = train_input_tensor.size(1)
    HEIGHT, WIDTH = train_input_tensor.size(3), train_input_tensor.size(4)

    # best_params = optimize_params(
    #     train_dataset,
    #     valid_dataset,
    #     epochs=epochs,
    #     batch_size=batch_size,
    # )
    # best_params = {"filter_num": 32, "adam_learning_rate": optimizer_learning_rate}
    kernel_size = 3
    num_kernels = 32
    padding = 1
    activation = "relu"
    frame_size = (HEIGHT, WIDTH)
    num_layers = 3

    model = Seq2Seq(
        num_channels=num_channels,
        kernel_size=kernel_size,
        num_kernels=num_kernels,
        padding=padding,
        activation=activation,
        frame_size=frame_size,
        num_layers=num_layers,
    )
    model.to(DEVICE)
    model.float()

    optimizer = Adam(model.parameters(), lr=optimizer_learning_rate)
    loss_criterion = nn.MSELoss(reduction="mean")
    acc_criterion = RMSELoss(reduction="mean")

    loss_only_rain = False

    results = train(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        loss_criterion=loss_criterion,
        acc_criterion=acc_criterion,
        epochs=epochs,
        checkpoints_directory=os.path.join(downstream_directory, mlflow_experiment_id),
        loss_only_rain=loss_only_rain,
    )

    _ = learning_curve_plot(
        save_dir_path=downstream_directory,
        training_losses=results["training_loss"],
        validation_losses=results["validation_loss"],
        validation_accuracy=results["validation_accuracy"],
    )

    # Save model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_channels": num_channels,
            "kernel_size": kernel_size,
            "num_kernels": num_kernels,
            "padding": padding,
            "activation": activation,
            "frame_size": frame_size,
            "num_layers": num_layers,
        },
        os.path.join(downstream_directory, "model.pth"),
    )

    # Save results to mlflow
    mlflow.log_artifacts(downstream_directory)

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
