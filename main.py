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

    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
