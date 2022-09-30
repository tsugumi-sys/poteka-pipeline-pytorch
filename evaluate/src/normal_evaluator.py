import sys
from typing import Dict, List, Tuple
import logging
import os

from omegaconf import DictConfig
import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import mean_squared_error
import torch
from torch import nn
from hydra import compose

from common.interpolate_by_gpr import interpolate_by_gpr

sys.path.append("..")
from common.custom_logger import CustomLogger
from common.config import MinMaxScalingValue, PPOTEKACols, ScalingMethod, GridSize
from common.utils import rescale_tensor, timestep_csv_names
from train.src.config import DEVICE
from evaluate.src.utils import pred_observation_point_values, normalize_tensor, save_parquet, validate_scaling
from evaluate.src.create_image import all_cases_plot, casetype_plot, sample_plot, save_rain_image


logger = CustomLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NormalEvaluator:
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        test_dataset: Dict,
        input_parameter_names: List[str],
        output_parameter_names: List[str],
        downstream_directory: str,
        hydra_overrides: List[str] = [],
    ) -> None:
        """This class is used to evaluate model's prediction normally. The model predicts one time per one input.
            e.g. just simple 1 hour prediction

        Args:
            model (nn.Module): target model
            test_dataset (Dict): Test dataset information like the following ...
                test_dataset: Dict
                { sample1: {
                    date: str,
                    start: str,
                    input: Tensor shape is (batch_size, num_channels, seq_len, height, width),
                    label: Tensor shape is (batch_size, num_channels, seq_len, height, width),
                    label_df: Dict[int, pd.DataFrame]
                    standarize_info: {"rain": {"mean": 1.0, "std": 0.3}, ...}
                 },
                 sample2: {...}
                }
            downstream_directory (str): _description_
        """
        self.model = model
        self.model_name = model_name
        self.test_dataset = test_dataset
        self.input_parameter_names = input_parameter_names
        self.output_parameter_names = output_parameter_names
        self.downstream_direcotry = downstream_directory
        self.results_df = None
        self.hydra_cfg = self.__initialize_hydar_conf(hydra_overrides)

    def __initialize_hydar_conf(self, overrides: List[str]) -> DictConfig:
        cfg = compose(config_name="config", overrides=overrides)
        return cfg

    def __initialize_results_df(self):
        self.results_df = pd.DataFrame(columns=["isSequential", "case_type", "date", "date_time", "hour-rain", "Pred_Value",])

    def run(self) -> Dict:
        results = self.__evaluate()
        return results

    def __evaluate(self) -> Dict:
        logger.info("... Evaluating 1 hour prediction ...")
        self.__initialize_results_df()
        self.model.eval()
        with torch.no_grad():
            for test_case_name in self.test_dataset.keys():
                self.__eval_test_case(test_case_name=test_case_name, evaluate_type=evaluate_type)
        return self.__visualize_results(evaluate_type=evaluate_type)

    def __eval_test_case(self, test_case_name: str, evaluate_type: str) -> None:
        """Evaluate a test case.

        Args:
            test_case_name (str): test case name
        """
        valid_evaluate_types = ["normal", "sequential", "reuse_predict", "combine_models"]
        if evaluate_type in valid_evaluate_types:
            # Load X_test and y_train
            X_test: torch.Tensor = self.test_dataset[test_case_name]["input"]
            y_test: torch.Tensor = self.test_dataset[test_case_name]["label"]
            X_test = X_test.to(DEVICE)
            y_test = y_test.to(DEVICE)
            validate_scaling(y_test, scaling_method=ScalingMethod.MinMax.value, logger=logger)
            validate_scaling(X_test, scaling_method=self.hydra_cfg.scaling_method, logger=logger)
            # Run evaluation based on evaluate_type
            save_dir_path = os.path.join(self.downstream_direcotry, self.model_name, evaluate_type, test_case_name)
            os.makedirs(save_dir_path, exist_ok=True)
            logger.info(f"... Evaluating model_name: {self.model_name}, evaluate_type: {evaluate_type}, test_case_name: {test_case_name} ...")
            if evaluate_type == "normal":
                rmses = self.__eval_normal(X_test=X_test, save_results_dir_path=save_dir_path, test_case_name=test_case_name)
            elif evaluate_type in ["sequential", "reuse_predict"]:
                rmses = self.__eval_successibely(
                    X_test=X_test, y_test=y_test, save_results_dir_path=save_dir_path, test_case_name=test_case_name, evaluate_type=evaluate_type
                )
            elif evaluate_type == "combine_models":
                rmses = self.__eval_combine_models(X_test=X_test, save_results_dir_path=save_dir_path, test_case_name=test_case_name)
        else:
            raise ValueError(f"Invalid evaluate_type: {evaluate_type}")
        # Log mlflow
        rmse_log = "... - RMSE: "
        for time_step, rmse in rmses.items():
            mlflow.log_metric(key=f"{self.model_name}-{evaluate_type}-{test_case_name}", value=rmse, step=time_step)
            rmse_log += f"{time_step}: {np.round(rmse, 3)} mm/h, "
        logger.info(rmse_log)

    def __eval_normal(self, X_test: torch.Tensor, save_results_dir_path: str, test_case_name: str) -> Dict:
        # [TODO]: Use pydantic to define test_dataset
        input_seq_length = X_test.shape[2]
        _time_step_csvnames = timestep_csv_names(time_step_minutes=self.hydra_cfg.preprocess.time_step_minutes)
        date, start = self.test_dataset[test_case_name]["date"], self.test_dataset[test_case_name]["start"]
        start_idx = _time_step_csvnames.index(start)
        start = start.replace(".csv", "")
        label_dfs = self.test_dataset[test_case_name]["label_df"]
        # Evaluate for each timestep
        pred_tensor: torch.Tensor = self.model(X_test)
        pred_tensor = normalize_tensor(pred_tensor, device=DEVICE)
        self.__validate_pred_tensor(pred_tensor)
        output_param_name = self.output_parameter_names[0]  # get first output parameter names
        rmses = {}
        for time_step in range(input_seq_length):
            min_val, max_val = MinMaxScalingValue.get_minmax_values_by_weather_param(output_param_name)
            scaled_pred_tensor = rescale_tensor(min_value=min_val, max_value=max_val, tensor=pred_tensor[0, 0, time_step, ...])
            scaled_pred_ndarray = scaled_pred_tensor.cpu().detach().numpy().copy()
            # Calculate RMSE
            rmse, result_df = self.__calc_rmse(
                pred_ndarray=scaled_pred_ndarray, label_df=label_dfs[time_step], test_case_name=test_case_name, date=date, start=start, time_step=time_step,
            )
            rmses[time_step] = rmse
            # Save predict informations
            self.results_df = pd.concat([self.results_df, result_df], axis=0)
            utc_time_idx = start_idx + time_step + 6
            if utc_time_idx > len(_time_step_csvnames) - 1:
                utc_time_idx -= len(_time_step_csvnames)
            utc_time_name = _time_step_csvnames[utc_time_idx].replace(".csv", "")
            if self.hydra_cfg.use_dummy_data is False and output_param_name == "rain":
                save_rain_image(scaled_pred_ndarray, os.path.join(save_results_dir_path, f"{utc_time_name}.png"))
            result_df.to_csv(os.path.join(save_results_dir_path, f"pred_observ_df_{utc_time_name}.csv"))
            save_parquet(scaled_pred_ndarray, os.path.join(save_results_dir_path, f"{utc_time_name}.parquet.gzip"))
        return rmses
