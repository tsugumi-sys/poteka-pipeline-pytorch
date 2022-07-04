import os
from typing import Dict
import sys
import hydra
from omegaconf import DictConfig
import json
from collections import OrderedDict

import torch
import mlflow

sys.path.append("..")
from common.data_loader import test_data_loader
from common.custom_logger import CustomLogger
from common.utils import get_mlflow_tag_from_input_parameters, split_input_parameters_str
from train.src.seq_to_seq import Seq2Seq
from train.src.model_for_test import TestModel
from evaluate.src.evaluator import Evaluator

logger = CustomLogger("Evaluate_Logger")

device = "cuda" if torch.cuda.is_available() else "cpu"


def order_meta_models(meta_models: Dict) -> OrderedDict:
    # Move `model` to the end so that evaluating for combined models.
    ordered_dic = OrderedDict(meta_models)
    ordered_dic.move_to_end("model")
    return ordered_dic


def evaluate(
    upstream_directory: str,
    downstream_directory: str,
    preprocess_downstream_directory: str,
    use_dummy_data: bool,
    use_test_model: bool,
    scaling_method: str,
) -> Dict:
    test_data_paths = os.path.join(preprocess_downstream_directory, "meta_test.json")
    debug_mode = False
    test_dataset, _ = test_data_loader(test_data_paths, scaling_method=scaling_method, isTrain=False, debug_mode=debug_mode, use_dummy_data=use_dummy_data)
    with open(os.path.join(upstream_directory, "meta_models.json"), "r") as f:
        meta_models = json.load(f)
    meta_models = order_meta_models(meta_models)
    results = {}
    for model_name, info in meta_models.items():
        trained_model = torch.load(os.path.join(upstream_directory, f"{model_name}.pth"))
        if use_test_model is True:
            logger.info("... using test model ...")
            model = TestModel(return_sequences=info["return_sequences"])
        else:
            model = Seq2Seq(
                num_channels=trained_model["num_channels"],
                kernel_size=trained_model["kernel_size"],
                num_kernels=trained_model["num_kernels"],
                padding=trained_model["padding"],
                activation=trained_model["activation"],
                frame_size=trained_model["frame_size"],
                num_layers=trained_model["num_layers"],
                weights_initializer=trained_model["weights_initializer"],
                return_sequences=info["return_sequences"],
            )
        model.load_state_dict(trained_model["model_state_dict"])
        model.to(device)
        model.float()
        # Run main evaluation process
        evaluator = Evaluator(
            model=model,
            model_name=model_name,
            test_dataset=test_dataset,
            input_parameter_names=info["input_parameters"],
            output_parameter_names=info["output_parameters"],
            downstream_directory=downstream_directory,
        )
        if not info["return_sequences"]:
            if model_name == "model":
                results[model_name] = evaluator.run(evaluate_types=["reuse_predict", "sequential", "combine_models"])
            else:
                results[model_name] = evaluator.run(evaluate_types=["reuse_predict", "sequential"])
        else:
            results[model_name] = evaluator.run(evaluate_types=["normal"])

    return results


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    input_parameters = split_input_parameters_str(cfg.input_parameters)
    mlflow.set_tag("mlflow.runName", get_mlflow_tag_from_input_parameters(input_parameters) + "_evaluate")

    # mlflow_experiment_id = int(os.getenv("MLFLOW_EXPERIMENT_ID", 0))

    upstream_directory = cfg.evaluate.model_file_dir_path
    downstream_directory = cfg.evaluate.downstream_dir_path
    preprocess_downstream_directory = cfg.evaluate.preprocess_meta_file_dir_path

    os.makedirs(downstream_directory, exist_ok=True)

    logger.info(upstream_directory)
    results = evaluate(
        upstream_directory=upstream_directory,
        downstream_directory=downstream_directory,
        preprocess_downstream_directory=preprocess_downstream_directory,
        use_dummy_data=cfg.use_dummy_data,
        use_test_model=cfg.train.use_test_model,
        scaling_method=cfg.scaling_method,
    )
    for model_name, result in results.items():
        for evaluate_type, metrics in result.items():
            for key, val in metrics.items():
                mlflow.log_metric(f"{model_name}-{evaluate_type}-{key}", val)

    mlflow.log_artifacts(
        downstream_directory,
        artifact_path="evaluations",
    )
    logger.info("Evaluation successfully ended.")


if __name__ == "__main__":
    main()
