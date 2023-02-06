import argparse
import json
import os
import sys
from collections import OrderedDict
from typing import Dict

import mlflow
import torch
from omegaconf import DictConfig

sys.path.append("..")
from common.config import DEVICE  # noqa: E402
from common.custom_logger import CustomLogger  # noqa: E402
from common.data_loader import test_data_loader  # noqa: E402
from common.omegaconf_manager import OmegaconfManager  # noqa: E402
from common.utils import get_mlflow_tag_from_input_parameters, split_input_parameters_str  # noqa: E402
from evaluate.src.combine_models_evaluator import CombineModelsEvaluator  # noqa: E402
from evaluate.src.normal_evaluator import NormalEvaluator  # noqa: E402
from evaluate.src.sequential_evaluator import SequentialEvaluator  # noqa: E402
from train.src.utils.model_interactor import ModelInteractor  # noqa: E402

logger = CustomLogger("Evaluate_Logger")


def order_meta_models(meta_models: Dict) -> OrderedDict:
    """Order meta models for managing evaluation order.

    If CombineModelEvaluation is executed, all the other single parameter
    models evaluation should be executed.
    """
    # Move `model` to the end so that evaluating for combined models.
    ordered_dic = OrderedDict(meta_models)
    ordered_dic.move_to_end("model")
    return ordered_dic


def main(hydra_cfg: DictConfig):
    """The main process of `evaluation` step.

    There are 3 evaluation processes.
        1. NormalEvaluation: Evaluation of the model if `return_sequences=false` like (1h prediction).
        2. SequentialEvaluation: Evaluation of the model if `return_sequences=ture`.
        3. CombineModelsEvaluation: Evaluation of the multi parameter model with `return_sequences=false`.
            Executed if `train_separately=true`.
    """
    input_parameters = split_input_parameters_str(hydra_cfg.input_parameters)
    upstream_directory = hydra_cfg.evaluate.model_file_dir_path
    downstream_directory = hydra_cfg.evaluate.downstream_dir_path
    preprocess_downstream_directory = hydra_cfg.evaluate.preprocess_meta_file_dir_path
    scaling_method = hydra_cfg.scaling_method
    is_obpoint_labeldata = hydra_cfg.is_obpoint_labeldata
    trained_model_name = hydra_cfg.model_name
    train_separately = hydra_cfg.train.train_separately
    use_dummy_data = hydra_cfg.use_dummy_data

    mlflow.set_tag("mlflow.runName", get_mlflow_tag_from_input_parameters(input_parameters) + "_evaluate")
    os.makedirs(downstream_directory, exist_ok=True)

    test_data_paths = os.path.join(preprocess_downstream_directory, "meta_test.json")
    observation_point_file_path = "../common/meta-data/observation_point.json"
    # NOTE: test_data_loader loads all parameters tensor. So num_channels are maximum.
    test_dataset, features_dict = test_data_loader(
        test_data_paths,
        observation_point_file_path,
        scaling_method=scaling_method,
        isObPointLabelData=is_obpoint_labeldata,
        use_dummy_data=use_dummy_data,
    )

    with open(os.path.join(upstream_directory, "meta_models.json"), "r") as f:
        meta_models = json.load(f)

    model_interactor = ModelInteractor()
    # NOTE: all parameter trained model (model) should be evaluate
    #   in the end so that combine models prediction can be executed.
    meta_models = order_meta_models(meta_models)

    for model_name, info in meta_models.items():
        trained_model = torch.load(os.path.join(upstream_directory, f"{model_name}.pth"))
        model_state_dict = trained_model.pop("model_state_dict")
        model = model_interactor.initialize_model(trained_model_name, **trained_model)
        model.load_state_dict(model_state_dict)
        model.to(DEVICE)
        model.float()
        # NOTE: Clone test dataset
        # You cannot use dict.copy() because you need to clone the input and label tensor.
        _test_dataset = {}
        for test_case_name in test_dataset.keys():
            _test_dataset[test_case_name] = {}
            # Copy date, start, label_df, standarized_info
            for key, val in test_dataset[test_case_name].items():
                if key not in ["input", "label"]:
                    _test_dataset[test_case_name][key] = val

        # Copy input and label
        if len(info["input_parameters"]) == 1 and len(info["output_parameters"]) == 1:
            param_idx = list(features_dict.values()).index(info["input_parameters"][0])
            for test_case_name in test_dataset.keys():
                _test_dataset[test_case_name]["input"] = (
                    test_dataset[test_case_name]["input"].clone().detach()[:, param_idx : param_idx + 1, ...]
                )  # noqa: E203
                _test_dataset[test_case_name]["label"] = (
                    test_dataset[test_case_name]["label"].clone().detach()[:, param_idx : param_idx + 1, ...]
                )  # noqa: E203
        else:
            for test_case_name in test_dataset.keys():
                _test_dataset[test_case_name]["input"] = test_dataset[test_case_name]["input"].clone().detach()
                _test_dataset[test_case_name]["label"] = test_dataset[test_case_name]["label"].clone().detach()

        if info["return_sequences"]:
            # Run normal evaluator process
            evaluator = NormalEvaluator(
                model=model,
                model_name=model_name,
                test_dataset=_test_dataset,
                input_parameter_names=info["input_parameters"],
                output_parameter_names=info["output_parameters"],
                downstream_directory=downstream_directory,
                observation_point_file_path=observation_point_file_path,
                hydra_cfg=hydra_cfg,
            )
            normal_eval_results = evaluator.run()
            mlflow.log_metrics(normal_eval_results)

        else:
            sequential_evaluator = SequentialEvaluator(
                model=model,
                model_name=model_name,
                test_dataset=_test_dataset,
                input_parameter_names=info["input_parameters"],
                output_parameter_names=info["output_parameters"],
                downstream_directory=downstream_directory,
                observation_point_file_path=observation_point_file_path,
                hydra_cfg=hydra_cfg,
                evaluate_type="reuse_predict",
            )

            combine_models_evaluator = CombineModelsEvaluator(
                model=model,
                model_name=model_name,
                test_dataset=_test_dataset,
                input_parameter_names=info["input_parameters"],
                output_parameter_names=info["output_parameters"],
                downstream_directory=downstream_directory,
                observation_point_file_path=observation_point_file_path,
                hydra_cfg=hydra_cfg,
            )
            if model_name == "model":
                # Run seqiential evaluation process..
                # Reuse Predict Evaluation
                sequential_evaluator.evaluate_type = "reuse_predict"
                reuse_predict_eval_result = sequential_evaluator.run()

                # update inputs evaluation
                sequential_evaluator.evaluate_type = "update_inputs"
                sequential_evaluator.clean_dfs()
                update_inputs_eval_result = sequential_evaluator.run()

                # save metrics to mlflow
                mlflow.log_metrics(reuse_predict_eval_result)
                mlflow.log_metrics(update_inputs_eval_result)

                # Run combine models evaluation process.
                # NOTE: order_meta_models() sorts order of evaluation and normal evaluation process ends here.
                if train_separately:
                    results = combine_models_evaluator.run()
                    mlflow.log_metrics(results)

    mlflow.log_artifacts(
        downstream_directory,
        artifact_path="evaluations",
    )
    logger.info("Evaluation successfully ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train arguments")
    parser.add_argument("--hydra_file_path", type=str, help="Hydra configuration file saved in main.py.")
    args = parser.parse_args()
    omegaconf_manager = OmegaconfManager()
    hydra_cfg = omegaconf_manager.load(args.hydra_file_path)
    main(hydra_cfg)
