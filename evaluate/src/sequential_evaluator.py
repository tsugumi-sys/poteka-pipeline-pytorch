import sys
from typing import Dict, List, Tuple
import logging
import os

import torch
from torch import nn
from common.config import ScalingMethod
from common.utils import param_date_path

from evaluate.src.utils import normalize_tensor, rescale_pred_tensor
from train.src.config import DEVICE

sys.path.append("..")
from evaluate.src.base_evaluator import BaseEvaluator  # noqa: E402
from common.custom_logger import CustomLogger  # noqa: E402
from common.interpolate_by_gpr import interpolate_by_gpr  # noqa: E402

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

        save_dir_path = os.path.join(self.downstream_direcotry, self.model_name, "sequential_evaluation")
        os.makedirs(save_dir_path, exist_ok=True)

        self.scatter_plot(save_dir_path)
        self.save_results_df_to_csv(save_dir_path)
        self.save_metrics_df_to_csv(save_dir_path)

        results = {
            "r2": self.r2_score_from_results_df(self.output_parameter_names[0]),
            "rmse": self.rmse_from_results_df(self.output_parameter_names[0]),
        }
        return results

    def evaluate_test_case(self, test_case_name: str):
        X_test, y_test = self.load_test_case_dataset(test_case_name)
        output_param_name = self.output_parameter_names[0]
        before_standarized_info = self.test_dataset[test_case_name]["standarized_info"].copy()

        _X_test = X_test.clone().detach()
        rescaled_pred_tensors = y_test.clone().detach()
        for time_step in range(self.hydra_cfg.label_seq_length):
            # NOTE: model return predict tensors with chunnels.
            all_pred_tensors = self.model(_X_test)
            # NOTE: channel 0 is target weather parameter channel.
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
                _X_test, before_standarized_info = self._update_input_tensor(_X_test, before_standarized_info, all_pred_tensors[0, :, 0, ...])
            elif self.evaluate_type == "update_inputs":
                _X_test, before_standarized_info = self._update_input_tensor(_X_test, before_standarized_info, y_test[0, :, time_step, ...])

        save_dir_path = os.path.join(self.downstream_direcotry, self.model_name, "sequential_evaluation", self.evaluate_type, test_case_name)
        os.makedirs(save_dir_path, exist_ok=True)
        self.geo_plot(test_case_name, save_dir_path, rescaled_pred_tensors)

    def _update_input_tensor(
        self, before_input_tensor: torch.Tensor, before_standarized_info: Dict, next_frame_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Update input tensor X_test for next prediction. A next one frame tensor (prediction or label tensor) a given
        and update the first frame of X_test with that given tensor.

        Args:
            before_input_tensor (torch.Tensor):
            before_standarized_info (Dict): Stadarization information to rescale tensor using this.
            next_frame_tensor (torch.Tensor): This tensor should be scaled to [0, 1].
        """
        if next_frame_tensor.max().item() > 1 or next_frame_tensor.min().item():
            raise ValueError(f"next_frame_tensor is not scaled to [0, 1], but [{next_frame_tensor.min().item(), next_frame_tensor.max().item()}]")

        # Convert next_frame tensor to grids if ob point tensor is given.
        _, num_channels, _, height, width = before_input_tensor.size()
        if next_frame_tensor.ndim == 1:
            # The case of next_frame_tensor is [ob_point values]
            _next_frame_tensor = next_frame_tensor.cpu().detach()
            _next_frame_tensor = normalize_tensor(_next_frame_tensor, device=DEVICE)
            _next_frame_tensor = _next_frame_tensor.numpy().copy()
            next_frame_tensor = torch.zeros((len(self.input_parameter_names), width, height), dtype=torch.float)
            for param_dim in range(len(self.input_parameter_names)):
                next_frame_tensor[param_dim, ...] = interpolate_by_gpr(_next_frame_tensor[param_dim, ...], return_torch_tensor=True)
            next_frame_tensor = normalize_tensor(next_frame_tensor, device=DEVICE)

        scaling_method = self.hydra_cfg.scaling_method

        if scaling_method == ScalingMethod.MinMax.value:
            return torch.cat((before_input_tensor[:, :, 1:, ...], torch.reshape(next_frame_tensor, (1, num_channels, 1, height, width))), dim=2), {}

        # elif scaling_method == ScalingMethod.Standard.value or scaling_method == ScalingMethod.MinMaxStandard.value:
        else:
            for param_dim, param_name in enumerate(self.input_parameter_names):
                means, stds = before_standarized_info[param_name]["mean"], before_standarized_info[param_name]["std"]
                before_input_tensor[:, param_dim, ...] = before_input_tensor[:, param_dim, ...] * stds + means

            if before_input_tensor.ndim == 5:
                updated_input_tensor = torch.cat(
                    (before_input_tensor[:, :, 1:, ...], torch.reshape(next_frame_tensor, (1, num_channels, 1, height, width))), dim=2
                )
            else:
                # before_input_tensor's shape is like [1, num_channels, seq_length, ob_point_counts]
                ob_point_counts = next_frame_tensor.size(dim=3)
                updated_input_tensor = torch.cat(
                    (before_input_tensor[:, :, 1:, ...], torch.reshape(next_frame_tensor, (1, num_channels, 1, ob_point_counts))), dim=2
                )

            standarized_info = {}
            for param_dim, param_name in enumerate(self.input_parameter_names):
                standarized_info[param_name] = {}
                means = torch.mean(updated_input_tensor[:, param_dim, ...])
                stds = torch.std(updated_input_tensor[:, param_dim, ...])
                updated_input_tensor[:, param_dim, ...] = (updated_input_tensor[:, param_dim, ...] - means) / stds
                standarized_info[param_name]["mean"] = means
                standarized_info[param_name]["std"] = stds
            return updated_input_tensor, standarized_info
