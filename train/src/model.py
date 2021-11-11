import os
import logging
from typing import Tuple

# import numpy as np
from tensorflow.keras import layers, callbacks, models, losses, metrics
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)


def Simple_ConvLSTM(
    feature_num: int,
    filter_num: int = 32,
    grid_height: int = 50,
    grid_width: int = 50,
):

    inp = layers.Input(shape=(None, grid_height, grid_width, feature_num))
    x = layers.ConvLSTM2D(
        filters=filter_num,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=filter_num,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=filter_num,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=feature_num,
        kernel_size=3,
        padding="same",
        return_sequences=False,
        activation="sigmoid",
    )(x)

    model = models.Model(inp, x)

    return model


def evaluate(
    model,
    test_dataset,
) -> Tuple[float, float]:
    X_test, y_test = test_dataset[0], test_dataset[1]
    y_pred = model.predict(X_test)
    y_test, y_pred = y_test.reshape(-1), y_pred.reshape(-1)

    mse = losses.MeanSquaredError()
    loss = mse(y_test, y_pred).numpy()
    acc = r2_score(y_test, y_pred)

    return acc, loss


def train(
    model,
    train_dataset,
    test_dataset,
    optimizer,
    epochs: int = 32,
    batch_size: int = 10,
    checkpoints_directory: str = "/SimpleConvLSTM/model/",
):
    logger.info("start training ...")
    X_train, y_train = train_dataset[0], train_dataset[1]
    X_test, y_test = test_dataset[0], test_dataset[1]

    model.compile(
        optimizer=optimizer,
        loss=losses.BinaryCrossentropy(),
        metrics=["mae", "mse", metrics.RootMeanSquaredError()],
    )
    model.summary()

    early_stopping = callbacks.EarlyStopping(
        monitor="loss",
        min_delta=0.001,
        patience=20,
        restore_best_weights=True,
    )

    checkpoints_path = os.path.join(checkpoints_directory, "model")

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[early_stopping],
    )

    model.save(checkpoints_path)

    return checkpoints_path
