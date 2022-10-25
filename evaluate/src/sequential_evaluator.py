import sys
from typing import Dict, List
import logging
import os

import torch
from torch import nn

sys.path.append("..")
from evaluate.src.base_evaluator import BaseEvaluator  # noqa: E402
from common.custom_logger import CustomLogger  # noqa: E402

logger = CustomLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SequentialEvaluator(BaseEvaluator):
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        test_dataset: Dict,
        input_parameter_names: List[str],
        output_parameter_names: List[str],
        downstream_directory: str,
        observation_point_file_path: str,
        hydra_overrides: List[str] = [],
        evaluate_type: str = "reuse_predict",
    ) -> None:
        self.maxDiff = None
        super().__init__(
            model,
            model_name,
            test_dataset,
            input_parameter_names,
            output_parameter_names,
            downstream_directory,
            observation_point_file_path,
            hydra_overrides,
        )
        if evaluate_type not in ["reuse_predict", "update_inputs"]:
            raise ValueError(f"Invalid evaluate_type: {evaluate_type}. Shoud be in ['reuse_predict', 'update_inputs']")
        self.evaluate_type = evaluate_type

    def run(self):
        with torch.no_grad():
            for test_case_name in self.test_dataset.keys():
                self.evaluate_test_case(test_case_name)

        save_dir_path = os.path.join(self.downstream_direcotry, self.model_name, "sequential_evaluation", self.evaluate_type)
        os.makedirs(save_dir_path, exist_ok=True)

        self.scatter_plot(save_dir_path)
        self.save_results_df_to_csv(save_dir_path)
        self.save_metrics_df_to_csv(save_dir_path)

        results = {
            f"{self.model_name}_sequential_{self.evaluate_type}_r2": self.r2_score_from_results_df(self.output_parameter_names[0]),
            f"{self.model_name}_sequential_{self.evaluate_type}_rmse": self.rmse_from_results_df(self.output_parameter_names[0]),
        }
        return results

    def evaluate_test_case(self, test_case_name: str):
        logger.info(f"Sequential Evaluation - case: {test_case_name}")
        X_test, y_test = self.load_test_case_dataset(test_case_name)
        output_param_name = self.output_parameter_names[0]
        before_standarized_info = self.test_dataset[test_case_name]["standarize_info"].copy()

        _X_test = X_test.clone().detach()
        rescaled_pred_tensors = y_test.clone().detach()
        for time_step in range(self.hydra_cfg.label_seq_length):
            # NOTE: model return predict tensors with chunnels. All_pred_tensors are scaled to [0, 1] because of the sigmoid fucntion.
            all_pred_tensors = self.model(_X_test)
            # NOTE: channel 0 is target weather parameter channel. Rescaled pred tensor are scaled to original scale of output parameter name.
            rescaled_pred_tensor = self.rescale_pred_tensor(all_pred_tensors[0, 0, 0, ...], output_param_name)
            rescaled_pred_tensors[0, 0, time_step, ...] = rescaled_pred_tensor
            label_df = self.test_dataset[test_case_name]["label_df"][time_step]
            self.add_result_df_from_pred_tensor(
                test_case_name,
                time_step,
                rescaled_pred_tensor,
                label_df,
                output_param_name,
            )
            self.add_metrics_df_from_pred_tensor(
                test_case_name,
                time_step,
                rescaled_pred_tensor,
                label_df,
                output_param_name,
            )

            if self.evaluate_type == "reuse_predict":
                _X_test, before_standarized_info = self.update_input_tensor(_X_test, before_standarized_info, all_pred_tensors[0, :, 0, ...])
            elif self.evaluate_type == "update_inputs":
                _X_test, before_standarized_info = self.update_input_tensor(_X_test, before_standarized_info, y_test[0, :, time_step, ...])

        save_dir_path = os.path.join(self.downstream_direcotry, self.model_name, "sequential_evaluation", self.evaluate_type, test_case_name)
        os.makedirs(save_dir_path, exist_ok=True)
        self.geo_plot(test_case_name, save_dir_path, rescaled_pred_tensors)
