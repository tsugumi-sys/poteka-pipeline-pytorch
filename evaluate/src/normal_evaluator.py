import sys
from typing import Dict, List
import logging
import os

import torch
from torch import nn

sys.path.append("..")
from evaluate.src.base_evaluator import BaseEvaluator  # noqa: E402
from common.custom_logger import CustomLogger  # noqa: E402
from common.interpolate_by_gpr import interpolate_by_gpr  # noqa: E402

logger = CustomLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NormalEvaluator(BaseEvaluator):
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
    ) -> None:
        super().__init__(
            model, model_name, test_dataset, input_parameter_names, output_parameter_names, downstream_directory, observation_point_file_path, hydra_overrides,
        )

    def run(self) -> Dict[str, float]:
        with torch.no_grad():
            for test_case_name in self.test_dataset.keys():
                self.evaluate_test_case(test_case_name)

        save_dir_path = os.path.join(self.downstream_direcotry, self.model_name, "normal_evaluation")
        os.makedirs(save_dir_path, exist_ok=True)
        # Save scatter plot of prediction and observation data of each P-POTEKA data point.
        self.scatter_plot(save_dir_path)

        self.save_results_df_to_csv(save_dir_path)
        self.save_metrics_df_to_csv(save_dir_path)

        results = {
            f"{self.model_name}_normal_r2": self.r2_score_from_results_df(self.output_parameter_names[0]),
            f"{self.model_name}_normal_rmse": self.rmse_from_results_df(self.output_parameter_names[0]),
        }
        return results

    def evaluate_test_case(self, test_case_name: str):
        logger.info(f"Normal Evaluation - case: {test_case_name}")
        X_test, _ = self.load_test_case_dataset(test_case_name)
        output_param_name = self.output_parameter_names[0]

        all_pred_tensors: torch.Tensor = self.model(X_test)
        all_rescaled_pred_tensors = self.rescale_pred_tensor(all_pred_tensors, output_param_name)

        for time_step in range(self.hydra_cfg.label_seq_length):
            rescaled_pred_tensor = all_rescaled_pred_tensors[0, 0, time_step, ...]
            label_df = self.test_dataset[test_case_name]["label_df"][time_step]
            self.add_result_df_from_pred_tensor(
                test_case_name, time_step, rescaled_pred_tensor, label_df, output_param_name,
            )
            self.add_metrics_df_from_pred_tensor(
                test_case_name, time_step, rescaled_pred_tensor, label_df, output_param_name,
            )

        save_dir_path = os.path.join(self.downstream_direcotry, self.model_name, "normal_evaluation", test_case_name)
        os.makedirs(save_dir_path, exist_ok=True)
        self.geo_plot(test_case_name, save_dir_path, all_rescaled_pred_tensors)
