import mlflow
import optuna
import logging
from tensorflow.keras import callbacks, losses, metrics, backend, optimizers
from src.model import Simple_ConvLSTM

logger = logging.getLogger("Train_Logger")


def optimize_params(
    train_dataset, valid_dataset, epochs: int, batch_size: int, verbose: int = 0,
):
    def create_model(trial: optuna.trial.Trial, feature_num: int):
        filter_num = trial.suggest_int("filter_num", 16, 128)
        adam_learning_rate = trial.suggest_loguniform("adam_learning_rate", 1e-5, 1e-1)

        model = Simple_ConvLSTM(feature_num=feature_num, filter_num=filter_num)
        optimizer = optimizers.Adam(learning_rate=adam_learning_rate)
        model.compile(
            optimizer=optimizer, loss=losses.BinaryCrossentropy(), metrics=["mse", metrics.RootMeanSquaredError(),],
        )

        return model

    def objective(trial: optuna.trial.Trial):
        backend.clear_session()
        feature_num = X_train.shape[-1]
        model = create_model(trial, feature_num=feature_num)

        early_stopping = callbacks.EarlyStopping(monitor="loss", min_delta=0.01, patience=10, restore_best_weights=True,)
        model.fit(
            X_train, y_train, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=[early_stopping],
        )
        score = model.evaluate(X_valid, y_valid, verbose=verbose)

        mlflow.log_metric(
            key="HP_Optimization_Score", value=score[0], step=trial.number,
        )

        trial_params = trial.params
        for key, value in trial_params.items():
            mlflow.log_metric(
                key="HP_Optimization_" + key, value=value, step=trial.number,
            )

        logger.info(f"Trial{trial.number}: evaluate LOSS: {score[0]} MSE: {score[1]}")
        filter_num, adam_lr = trial_params["filter_num"], trial_params["adam_learning_rate"]
        logger.info(f"-- filter_num: {filter_num} adam_learning_rate: {adam_lr}")
        return score[1]

    X_train, y_train = train_dataset[0], train_dataset[1]
    X_valid, y_valid = valid_dataset[0], valid_dataset[1]

    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective, n_trials=30, gc_after_trial=True, show_progress_bar=True,
    )

    return study.best_trial.params
