import os
from typing import Dict, List
import sys
import hydra
from omegaconf import DictConfig
import json
from collections import OrderedDict

import torch
import mlflow

sys.path.append("..")
from common.data_loader import test_data_loader  # noqa: E402
from common.custom_logger import CustomLogger  # noqa: E402
from common.utils import get_mlflow_tag_from_input_parameters, split_input_parameters_str  # noqa: E402

# from train.src.seq_to_seq import Seq2Seq  # noqa: E402
from evaluate.src.normal_evaluator import NormalEvaluator  # noqa: E402
from evaluate.src.sequential_evaluator import SequentialEvaluator  # noqa: E402
from evaluate.src.combine_models_evaluator import CombineModelsEvaluator  # noqa: E402
from common.config import DEVICE
from train.src.utils.model_interactor import ModelInteractor

logger = CustomLogger("Evaluate_Logger")


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
    is_obpoint_labeldata: bool,
    input_parameters: List[str],
    trained_model_name: str,
):
    test_data_paths = os.path.join(preprocess_downstream_directory, "meta_test.json")
    observation_point_file_path = "../common/meta-data/observation_point.json"
    # NOTE: test_data_loader loads all parameters tensor. So num_channels are maximum.
    test_dataset, features_dict = test_data_loader(
        test_data_paths, observation_point_file_path, scaling_method=scaling_method, isObPointLabelData=is_obpoint_labeldata, use_dummy_data=use_dummy_data,
    )

    with open(os.path.join(upstream_directory, "meta_models.json"), "r") as f:
        meta_models = json.load(f)

    model_interactor = ModelInteractor()
    # NOTE: all parameter trained model (model) should be evaluate in the end so that combine models prediction can be executed.
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
                _test_dataset[test_case_name]["input"] = test_dataset[test_case_name]["input"].clone().detach()[:, param_idx : param_idx + 1, ...]  # noqa: E203
                _test_dataset[test_case_name]["label"] = test_dataset[test_case_name]["label"].clone().detach()[:, param_idx : param_idx + 1, ...]  # noqa: E203
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
                hydra_overrides=[f"use_dummy_data={use_dummy_data}", f"train.use_test_model={use_test_model}", f"input_parameters={input_parameters}"],
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
                hydra_overrides=[f"use_dummy_data={use_dummy_data}", f"train.use_test_model={use_test_model}", f"input_parameters={input_parameters}"],
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
                hydra_overrides=[f"use_dummy_data={use_dummy_data}", f"train.use_test_model={use_test_model}", f"input_parameters={input_parameters}"],
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
                results = combine_models_evaluator.run()
                mlflow.log_metrics(results)


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
    evaluate(
        upstream_directory=upstream_directory,
        downstream_directory=downstream_directory,
        preprocess_downstream_directory=preprocess_downstream_directory,
        use_dummy_data=cfg.use_dummy_data,
        use_test_model=cfg.train.use_test_model,
        scaling_method=cfg.scaling_method,
        is_obpoint_labeldata=cfg.is_obpoint_labeldata,
        input_parameters=cfg.input_parameters,
        trained_model_name=cfg.model_name,
    )

    mlflow.log_artifacts(
        downstream_directory, artifact_path="evaluations",
    )
    logger.info("Evaluation successfully ended.")


if __name__ == "__main__":
    main()
