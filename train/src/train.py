import logging
import os
import sys

import hydra
from omegaconf import DictConfig
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchinfo
import mlflow

sys.path.append("..")
from train.src.trainer import trainer
from train.src.seq_to_seq import Seq2Seq, PotekaDataset, RMSELoss
from train.src.model_for_test import TestModel
from train.src.learning_curve_plot import learning_curve_plot
from train.src.config import DEVICE, WeightsInitializer
from common.utils import get_mlflow_tag_from_input_parameters, split_input_parameters_str
from common.data_loader import data_loader
from common.custom_logger import CustomLogger
from common.config import ScalingMethod

logger = CustomLogger("Train_Logger", level=logging.INFO)


def start_run(
    mlflow_experiment_id: str,
    upstream_directory: str,
    downstream_directory: str,
    batch_size: int,
    epochs: int,
    optimizer_learning_rate: float,
    debug_mode: bool = False,
):
    train_data_paths = os.path.join(upstream_directory, "meta_train.json")
    valid_data_paths = os.path.join(upstream_directory, "meta_valid.json")

    is_maxsize_limit: bool = False
    scaling_method = ScalingMethod.MinMaxStandard.value
    train_input_tensor, train_label_tensor = data_loader(train_data_paths, scaling_method=scaling_method, isMaxSizeLimit=is_maxsize_limit, debug_mode=False)
    valid_input_tensor, valid_label_tensor = data_loader(valid_data_paths, scaling_method=scaling_method, isMaxSizeLimit=is_maxsize_limit, debug_mode=False)

    train_input_tensor, train_label_tensor = train_input_tensor.to(DEVICE), train_label_tensor.to(DEVICE)
    valid_input_tensor, valid_label_tensor = valid_input_tensor.to(DEVICE), valid_label_tensor.to(DEVICE)

    train_dataset = PotekaDataset(input_tensor=train_input_tensor, label_tensor=train_label_tensor)
    valid_dataset = PotekaDataset(input_tensor=valid_input_tensor, label_tensor=valid_label_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    num_channels = train_input_tensor.size(1)
    seq_length = train_input_tensor.size(2)
    HEIGHT, WIDTH = train_input_tensor.size(3), train_input_tensor.size(4)
    if debug_mode is True:
        model = TestModel()
    else:
        kernel_size = 3
        num_kernels = 32
        padding = "same"
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
            weights_initializer=WeightsInitializer.He,
        )

    model.to(DEVICE)
    model.float()

    model_sumary_file = os.path.join(downstream_directory, "model_summary.txt")
    with open(model_sumary_file, "w") as f:
        f.write(repr(torchinfo.summary(model, input_size=(batch_size, num_channels, seq_length, HEIGHT, WIDTH))))

    optimizer = Adam(model.parameters(), lr=optimizer_learning_rate)
    # loss_criterion = nn.MSELoss(reduction="mean")
    loss_criterion = nn.BCELoss()
    acc_criterion = RMSELoss(reduction="mean")

    loss_only_rain = False

    results = trainer(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        loss_criterion=loss_criterion,
        acc_criterion=acc_criterion,
        epochs=epochs,
        checkpoints_directory=downstream_directory,
        loss_only_rain=loss_only_rain,
    )

    _ = learning_curve_plot(
        save_dir_path=downstream_directory,
        training_losses=results["training_loss"],
        validation_losses=results["validation_loss"],
        validation_accuracy=results["validation_accuracy"],
    )

    # # Save model
    # torch.save(
    #     {
    #         "model_state_dict": model.state_dict(),
    #         "num_channels": num_channels,
    #         "kernel_size": kernel_size,
    #         "num_kernels": num_kernels,
    #         "padding": padding,
    #         "activation": activation,
    #         "frame_size": frame_size,
    #         "num_layers": num_layers,
    #     },
    #     os.path.join(downstream_directory, "model.pth"),
    # )

    # Save results to mlflow
    mlflow.log_artifacts(downstream_directory)

    logger.info("Training finished")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.debug_mode is True:
        logger.info("Running train in debug mode ...")

    input_parameters = split_input_parameters_str(cfg.input_parameters)
    mlflow.set_tag("mlflow.runName", get_mlflow_tag_from_input_parameters(input_parameters) + "_train & tuning")

    mlflow_experiment_id = str(os.getenv("MLFLOW_EXPERIMENT_ID", 0))

    upstream_dir_path = cfg.train.upstream_dir_path
    downstream_dir_path = cfg.train.downstream_dir_path

    os.makedirs(downstream_dir_path, exist_ok=True)

    start_run(
        mlflow_experiment_id=mlflow_experiment_id,
        upstream_directory=upstream_dir_path,
        downstream_directory=downstream_dir_path,
        batch_size=cfg.train.batch_size,
        epochs=cfg.train.epochs,
        optimizer_learning_rate=cfg.train.optimizer_learning_rate,
        debug_mode=cfg.debug_mode,
    )


if __name__ == "__main__":
    main()
