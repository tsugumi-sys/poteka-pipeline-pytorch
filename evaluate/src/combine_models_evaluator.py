import sys
from typing import Dict, List
import logging
import os

import torch
from torch import nn
import numpy as np

from evaluate.src.utils import normalize_tensor
from train.src.config import DEVICE

sys.path.append("..")
from evaluate.src.base_evaluator import BaseEvaluator  # noqa: E402
from common.custom_logger import CustomLogger  # noqa: E402
from common.interpolate_by_gpr import interpolate_by_gpr  # noqa: E402
from common.config import GridSize  # noqa: E402
from common.utils import load_scaled_data  # noqa: E402

logger = CustomLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CombineModelsEvaluator(BaseEvaluator):
    """CombineModelsEvaluator execute an evaluation using other models predictions.
    The evaluation target model predicts only next frame like predict.
    The other models predict normally (The prediction lendth is `label_seq_length`).
    The other models predictions are saved in other directories and scaled as its original scale.
    Load and re-scale data for given scaling method is needed.

    """

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

    def run(self):
        with torch.no_grad():
            for test_case_name in self.test_dataset.keys():
                self.evaluate_test_case(test_case_name)

        save_dir_path = os.path.join(self.downstream_direcotry, self.model_name, "combine_models_evaluation")
        os.makedirs(save_dir_path, exist_ok=True)

        self.scatter_plot(save_dir_path)
        self.timeseries_metrics_boxplot(target_param_name=self.output_parameter_names[0], target_metrics_name="rmse", downstream_directory=save_dir_path)
        self.timeseries_metrics_boxplot(target_param_name=self.output_parameter_names[0], target_metrics_name="r2_score", downstream_directory=save_dir_path)
        self.save_results_df_to_csv(save_dir_path)
        self.save_metrics_df_to_csv(save_dir_path)

        results = {
            f"{self.model_name}_combine_models_r2": self.r2_score_from_results_df(self.output_parameter_names[0]),
            f"{self.model_name}_combine_models_rmse": self.rmse_from_results_df(self.output_parameter_names[0]),
        }
        return results

    def evaluate_test_case(self, test_case_name: str):
        logger.info(f"Combine Models Evaluation - case: {test_case_name}")
        X_test, y_test = self.load_test_case_dataset(test_case_name)
        before_standarized_info = self.test_dataset[test_case_name]["standarize_info"].copy()
        output_param_name = self.output_parameter_names[0]

        # NOTE: This tensor is grid data. The shape is like [1, 3, 6, 50, 50]
        sub_models_predict_tensors = self.load_sub_models_predict_tensor(test_case_name)

        _X_test = X_test.clone().detach()
        rescaled_pred_tensors = y_test.clone().detach()
        for time_step in range(self.hydra_cfg.label_seq_length):
            all_pred_tensors = self.model(_X_test)

            # NOTE: rescaled_pred_tensors are used in geo_plot. rescaled_pred_tensor is scaled as its original scale.
            rescaled_pred_tensor = self.rescale_pred_tensor(all_pred_tensors[0, 0, 0, ...], output_param_name)
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

            rescaled_pred_tensors[0, 0, time_step, ...] = rescaled_pred_tensor

            # NOTE: all_pred_tensors are scaled as [0, 1]
            pred_rain_tensor = all_pred_tensors[0, 0, 0, ...]
            if pred_rain_tensor.ndim == 1:
                # NOTE: pred_rain_tensor is like [35]
                pred_rain_ndarr = interpolate_by_gpr(pred_rain_tensor.cpu().detach().numpy().copy(), self.observation_point_file_path)
                pred_rain_tensor = torch.from_numpy(pred_rain_ndarr.copy()).to(DEVICE)
                pred_rain_tensor = normalize_tensor(pred_rain_tensor, device=DEVICE)

            sub_models_predict_tensors[0, 0, time_step, ...] = pred_rain_tensor
            _X_test, before_standarized_info = self.update_input_tensor(_X_test, before_standarized_info, sub_models_predict_tensors[0, :, time_step, ...])

        save_dir_path = os.path.join(self.downstream_direcotry, self.model_name, "combine_models_evaluation", test_case_name)
        os.makedirs(save_dir_path, exist_ok=True)

        self.geo_plot(test_case_name, save_dir_path, rescaled_pred_tensors)

    def load_sub_models_predict_tensor(self, test_case_name):
        """This function load other models prediction data to tensor.

        1. load other models prediction data frok parquet.gzip file.
        2. If this data is not a grid data, interpolate data with GPR.
        3. This data is scaled as its original scale. So scale again.

        """
        sub_models_predict_tensors = torch.zeros(
            (1, len(self.input_parameter_names), self.hydra_cfg.label_seq_length, GridSize.HEIGHT, GridSize.WIDTH), dtype=torch.float, device=DEVICE
        ).to(DEVICE)

        for param_dim, param_name in enumerate(self.input_parameter_names):
            if param_name != "rain":
                results_dir_path = os.path.join(self.downstream_direcotry, param_name, "normal_evaluation", test_case_name)
                for time_step in range(self.hydra_cfg.label_seq_length):
                    pred_result_file_path = os.path.join(results_dir_path, f"{self.get_prediction_utc_time(test_case_name, time_step)}.parquet.gzip")
                    if not os.path.exists(pred_result_file_path):
                        raise ValueError(f"prediction data is not found(path: {pred_result_file_path})")

                    # NOTE: The shape is two way. Like a grid [50, 50] or ob_point_data [35, 1].
                    ndarr = load_scaled_data(pred_result_file_path)
                    pred_ndarray = ndarr.astype(np.float32)
                    if pred_ndarray.shape[0] != GridSize.WIDTH or pred_ndarray.shape[1] != GridSize.HEIGHT:
                        # NOTE: Interpoate is needed because input data is grid data. Change ndarray shape to 1 dimention.
                        pred_ndarray = interpolate_by_gpr(pred_ndarray.reshape((pred_ndarray.shape[0])), self.observation_point_file_path)
                    pred_tensor = torch.from_numpy(pred_ndarray.copy()).to(DEVICE)
                    pred_tensor = normalize_tensor(pred_tensor, device=DEVICE)

                    sub_models_predict_tensors[0, param_dim, time_step, ...] = pred_tensor

        return sub_models_predict_tensors
