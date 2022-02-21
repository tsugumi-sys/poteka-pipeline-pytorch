import os
import argparse

import mlflow


def main():
    parser = argparse.ArgumentParser(
        description="Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Preprocess args
    parser.add_argument(
        "--preprocess_downstream",
        type=str,
        default="./preprocess/data/preprocess",
        help="preprocess downstream directory",
    )
    parser.add_argument(
        "--preprocess_params",
        type=str,
        default="rain humidity temperature wind",
        help="input weather parameters",
    )
    parser.add_argument(
        "--preprocess_delta",
        type=int,
        default=10,
        help="time resolution of input dataset (minute). Minimum is 2, max is 10.",
    )
    parser.add_argument(
        "--preprocess_slides",
        type=int,
        default=3,
        help="Time slides when load datasets. Ex. 10:00~12:00, 10:06~12:06 ... (slides=3) slides * 2min step.",
    )
    # Train args
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
        "--train_optim_learning_rate",
        type=float,
        default=0.001,
        help="optimizers learning rate",
    )

    # Evaluate args
    parser.add_argument(
        "--evaluate_downstream",
        type=str,
        default="./data/evaluate",
        help="evaluate downstream direcotry",
    )

    args = parser.parse_args()
    mlflow_experiment_id = int(os.getenv("MLFLOW_EXPERIMENT_ID", 0))

    run_name = ""
    for param in args.preprocess_params.split():
        run_name += param[0].upper() + param[1:]

    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", run_name)
        preprocess_run = mlflow.run(
            uri="./preprocess",
            entry_point="preprocess",
            backend="local",
            parameters={
                "parent_run_name": run_name,
                "downstream": args.preprocess_downstream,
                "params": args.preprocess_params,
                "delta": args.preprocess_delta,
                "slides": args.preprocess_slides,
            },
            use_conda=False,
        )
        preprocess_run = mlflow.tracking.MlflowClient().get_run(preprocess_run.run_id)

        current_dir = os.getcwd()
        dataset = os.path.join(
            current_dir,
            "mlruns/",
            str(mlflow_experiment_id),
            preprocess_run.info.run_id,
            "artifacts/downstream_directory",
        )

        train_run = mlflow.run(
            uri="./train",
            entry_point="train",
            backend="local",
            parameters={
                "parent_run_name": run_name,
                "upstream": dataset,
                "downstream": args.train_downstream,
                "epochs": args.train_epochs,
                "batch_size": args.train_batch_size,
                "optimizer_learning_rate": args.train_optim_learning_rate,
            },
            use_conda=False,
        )
        train_run = mlflow.tracking.MlflowClient().get_run(train_run.run_id)

        model_file_path = train_run.info.artifact_uri
        model_file_path = model_file_path.replace("file://", "")
        evaluate_run = mlflow.run(
            uri="./evaluate",
            entry_point="evaluate",
            backend="local",
            parameters={
                "parent_run_name": run_name,
                "upstream": model_file_path,
                "downstream": args.evaluate_downstream,
                "preprocess_downstream": dataset,
                "preprocess_delta": args.preprocess_delta,
            },
            use_conda=False,
        )
        evaluate_run = mlflow.tracking.MlflowClient().get_run(evaluate_run.run_id)


if __name__ == "__main__":
    main()
