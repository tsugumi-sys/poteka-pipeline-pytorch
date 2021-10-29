import os
import logging
from typing import Tuple

# import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# from skl2onnx import convert_sklearn
from joblib import dump

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def SimpleNet(input_shape: int):
    model = keras.Sequential(
        [
            layers.Dense(64, activation="relu", input_shape=input_shape),
            layers.Dense(64, activation="relu"),
            layers.Dense(1),
        ]
    )

    return model


def SKRegressor(params):
    return Ridge(**params)


def evaluate(
    model,
    train_dataset,
    test_dataset,
) -> Tuple[float, float]:
    X_test, y_test = test_dataset.drop(columns="target", axis=1), test_dataset["target"]

    if isinstance(model, Ridge):
        y_pred = model.predict(X_test)
        loss = mean_squared_error(y_test, y_pred, squared=False)
        acc = r2_score(y_test, y_pred)

        return acc, loss

    else:
        # Scaling
        scaler = StandardScaler()
        scaler.fit(train_dataset.drop(columns="target", axis=1))
        X_test = scaler.transform(X_test)

        y_pred = model.predict(X_test)
        loss = mean_squared_error(y_test, y_pred, squared=False)
        acc = r2_score(y_test, y_pred)

        return acc, loss


def train(
    model,
    train_dataset,
    test_dataset,
    optimizer,
    epochs: int = 32,
    batch_size: int = 10,
    checkpoints_directory: str = "/SimpleDNN/model/",
):
    logger.info("start training ...")
    X_train, y_train = train_dataset.drop(columns="target", axis=1), train_dataset["target"]
    X_test, y_test = test_dataset.drop(columns="target", axis=1), test_dataset["target"]

    if isinstance(model, Ridge):
        model.fit(X_train, y_train)

        checkpoints_path = os.path.join(checkpoints_directory, "skregression_model.joblib")
        # initial_type = train_dataset[:1].astype(np.float32)
        # onx = convert_sklearn(model, initial_types=initial_type)
        # with open(checkpoints_path, "wb") as f:
        #     f.write(onx.SerializeToString())
        dump(model, checkpoints_path)

        logger.info(f"save model {checkpoints_path}")

        return checkpoints_path

    else:  # DNN keras model
        # Scaling
        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

        model.compile(optimizer=optimizer, loss="mse", metrics=["mse"])
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10)

        checkpoints_path = os.path.join(checkpoints_directory, "cp.ckpt")
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoints_path, save_weights_only=True, verbose=1)

        model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=[early_stopping, cp_callback],
            validation_data=(X_test, y_test),
        )

        return os.path.join(checkpoints_directory, "cp.ckpt.index")
