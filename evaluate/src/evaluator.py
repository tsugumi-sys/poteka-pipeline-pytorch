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

sys.path.append("..")
from common.custom_logger import CustomLogger
from common.config import WEATHER_PARAMS, MinMaxScalingValue, PPOTEKACols, ScalingMethod
from common.utils import rescale_tensor, timestep_csv_names
from train.src.config import DEVICE
from evaluate.src.utils import pred_obervation_point_values, re_standard_scale, normalize_tensor, save_parquet, standard_scaler_torch_tensor, validate_scaling
from evaluate.src.create_image import all_cases_plot, casetype_plot, sample_plot, save_rain_image


logger = CustomLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        test_dataset: Dict,
        input_parameter_names: Dict[int, str],
        output_parameter_names: Dict[int, str],
        downstream_directory: str,
    ) -> None:
        """Evaluator

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
        self.hydra_cfg = self.__initialize_hydar_conf()

    def __initialize_hydar_conf(self) -> DictConfig:
        cfg = compose(config_name="config")
        return cfg

    def __initialize_results_df(self):
        self.results_df = pd.DataFrame(
            columns=[
                "isSequential",
                "case_type",
                "date",
                "date_time",
                "hour-rain",
                "Pred_Value",
            ]
        )

    def run(self, evaluate_types: List[str]) -> Dict:
        results = {}
        for evaluate_type in evaluate_types:
            results[evaluate_type] = self.__evaluate(evaluate_type)
        return results

    def __evaluate(self, evaluate_type: str) -> Dict:
        """Evaluate model

        Args:
            evaluate_type (str): evalutation type
            if normal: Evaluate a model which the normal prediction.
                        The medel should predict with the single num channels.
            if reuse_predict: Evaluate a model which reuses the predicted data for next input data.
                                The model should be trained with the inputs and next frame output (so, the label length is one).
            if sequential: Evaluate the model of sequential prediction.
                            The sequential prediction here means that the model trained to predict next frame
                            and predict updateing the next input data with the obersavation data
                            (inverse of reusing the predict data like __evaluate_retuse_predict).

        Returns:
            Dict: _description_
        """
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
        X_test: torch.Tensor = self.test_dataset[test_case_name]["input"]
        y_test: torch.Tensor = self.test_dataset[test_case_name]["label"]
        X_test = X_test.to(DEVICE)
        y_test = y_test.to(DEVICE)

        validate_scaling(y_test, scaling_method=ScalingMethod.MinMax.value, logger=logger)
        validate_scaling(X_test, scaling_method=self.hydra_cfg.scaling_method, logger=logger)

        save_dir_path = os.path.join(self.downstream_direcotry, self.model_name, evaluate_type, test_case_name)
        os.makedirs(save_dir_path, exist_ok=True)
        logger.info(f"... Evaluating model_name: {self.model_name}, evaluate_type: {evaluate_type}, test_case_name: {test_case_name} ...")

        if evaluate_type == "normal":
            rmses = self.__eval_normal(X_test=X_test, save_results_dir_path=save_dir_path, test_case_name=test_case_name)
        elif evaluate_type in ["sequential", "reuse_predict"]:
            rmses = self.__eval_successibely(
                X_test=X_test, y_test=y_test, save_results_dir_path=save_dir_path, test_case_name=test_case_name, evaluate_type=evaluate_type
            )
        else:
            raise ValueError(f"Invalid evaluate_type: {evaluate_type}")

        for time_step, rmse in rmses.items():
            mlflow.log_metric(key=f"{self.model_name}-{evaluate_type}-{test_case_name}", value=rmse, step=time_step)

    def __eval_normal(self, X_test: torch.Tensor, save_results_dir_path: str, test_case_name: str) -> Dict:
        # [TODO]: Use pydantic to define test_dataset
        input_seq_length = X_test.shape[2]
        _time_step_csvnames = timestep_csv_names(time_step_minutes=self.hydra_cfg.preprocess.time_step_minutes)
        date, start = self.test_dataset[test_case_name]["date"], self.test_dataset[test_case_name]["start"]
        start_idx = _time_step_csvnames.index(start)
        start = start.replace(".csv", "")
        label_dfs = self.test_dataset[test_case_name]["label_df"]

        pred_tensor: torch.Tensor = self.model(X_test)
        output_param_name = self.output_parameter_names[0]  # get first output parameter names
        rmses = {}
        for time_step in range(input_seq_length):
            min_val, max_val = MinMaxScalingValue.get_minmax_values_by_weather_param(output_param_name)
            scaled_pred_tensor = rescale_tensor(min_value=min_val, max_value=max_val, tensor=pred_tensor[0, 0, time_step, :, :])
            scaled_pred_ndarray = scaled_pred_tensor.cpu().detach().numpy().copy()
            rmse, result_df = self.__calc_rmse(
                pred_ndarray=scaled_pred_ndarray,
                label_df=label_dfs[time_step],
                test_case_name=test_case_name,
                date=date,
                start=start,
                time_step=time_step,
            )
            rmses[time_step] = rmse
            self.results_df = pd.concat([self.results_df, result_df], axis=0)
            utc_time_idx = start_idx + time_step + 6
            if utc_time_idx > len(_time_step_csvnames) - 1:
                utc_time_idx -= len(_time_step_csvnames)
            utc_time_name = _time_step_csvnames[utc_time_idx].replace(".csv", "")

            # Save predict informations
            if self.hydra_cfg.use_dummy_data is False and output_param_name == "rain":
                save_rain_image(scaled_pred_ndarray, os.path.join(save_results_dir_path, f"{utc_time_name}.png"))
            result_df.to_csv(os.path.join(save_results_dir_path, f"pred_observ_df_{utc_time_name}.csv"))
            save_parquet(scaled_pred_ndarray, os.path.join(save_results_dir_path, f"{utc_time_name}.parquet.gzip"))
        return rmses

    def __eval_successibely(self, X_test: torch.Tensor, y_test: torch.Tensor, save_results_dir_path: str, test_case_name: str, evaluate_type: str) -> Dict:
        input_seq_length = X_test.shape[2]
        _time_step_csvnames = timestep_csv_names(time_step_minutes=self.hydra_cfg.preprocess.time_step_minutes)
        date, start = self.test_dataset[test_case_name]["date"], self.test_dataset[test_case_name]["start"]
        start_idx = _time_step_csvnames.index(start)
        start = start.replace(".csv", "")
        label_dfs = self.test_dataset[test_case_name]["label_df"]
        _X_test = X_test.clone().detach()
        output_param_name = self.output_parameter_names[0]

        rmses = {}
        for time_step in range(input_seq_length):
            pred_tensor: torch.Tensor = self.model(_X_test)
            # Check if tensor containes nan values.
            if torch.isnan(pred_tensor).any():
                # logger.warning(self.model.state_dict())
                # logger.warning(pred_tensor)
                logger.warning(f"predicting {test_case_name} in normal way failed and predict tensor contains nan value.")

            pred_tensor = normalize_tensor(pred_tensor, device=DEVICE)
            predict_rain_tensor = pred_tensor[0, 0, 0, :, :]
            rain_min_val, rain_max_val = MinMaxScalingValue.get_minmax_values_by_weather_param("rain")
            scaled_rain_pred_tensor = rescale_tensor(min_value=rain_min_val, max_value=rain_max_val, tensor=predict_rain_tensor)
            scaled_rain_pred_ndarray = scaled_rain_pred_tensor.cpu().detach().numpy().copy()
            rmse, result_df = self.__calc_rmse(
                pred_ndarray=scaled_rain_pred_ndarray,
                label_df=label_dfs[time_step],
                test_case_name=test_case_name,
                date=date,
                start=start,
                time_step=time_step,
            )
            rmses[time_step] = rmse
            self.results_df = pd.concat([self.results_df, result_df], axis=0)
            utc_time_idx = start_idx + time_step + 6
            if utc_time_idx > len(_time_step_csvnames) - 1:
                utc_time_idx -= len(_time_step_csvnames)
            utc_time_name = _time_step_csvnames[utc_time_idx].replace(".csv", "")

            # Save predict informations
            if self.hydra_cfg.use_dummy_data is False and output_param_name == "rain":
                save_rain_image(scaled_rain_pred_ndarray, os.path.join(save_results_dir_path, f"{utc_time_name}.png"))

            result_df.to_csv(os.path.join(save_results_dir_path, f"pred_observ_df_{utc_time_name}.csv"))
            save_parquet(scaled_rain_pred_ndarray, os.path.join(save_results_dir_path, f"{utc_time_name}.parquet.gzip"))

            if evaluate_type == "sequential":
                _X_test = self.__update_input_tensor(_X_test, y_test[0, :, time_step, :, :])
            elif evaluate_type == "reuse_predict":
                _X_test = self.__update_input_tensor(_X_test, pred_tensor[0, :, 0, :, :])
            validate_scaling(_X_test, scaling_method=self.hydra_cfg.scaling_method, logger=logger)
        return rmses

    def __eval_combine_models(self, main_model_name: str, main_model_input_parameters: List[str]):
        """Evaluate the mutlti parameters trained model and single parameters models. Like
           Main model is

        Args:
            main_model_name (str): _description_
            main_model_input_parameters (List[str]): _description_
        """

    def __update_input_tensor(self, before_input_tensor: torch.Tensor, next_input_tensor: torch.Tensor) -> torch.Tensor:
        """Update input tensor (X_test) for next prediction step.
           In `sequential` evaluation, the input tensor will be updated with the label tensor, which is the observed tensor.
           In `reuse_predict` evaluation, the input tensor will be updated with the predict tensor.

        Args:
            before_input_tensor (torch.Tensor): tensor with the shape of (batch_size, num_channels, seq_length, height, width)
            next_input_tensor (torch.Tensor): tensor with the shape of (num_channels, height, width)

        Returns:
            torch.Tensor: tensor with the shape of (batch_size, num_channels, seq_length, height, width)
        """
        next_input_tensor = normalize_tensor(next_input_tensor, device=DEVICE)
        scaling_method = self.hydra_cfg.scaling_method
        if scaling_method is ScalingMethod.Standard.value or scaling_method is ScalingMethod.MinMaxStandard.value:
            for param_idx, param_name in self.input_parameter_names.items():
                if param_name != WEATHER_PARAMS.RAIN.value:
                    if scaling_method == ScalingMethod.Standard.value:
                        next_input_tensor[param_idx, :, :] = re_standard_scale(
                            next_input_tensor[param_idx, :, :], feature_name=param_name, device=DEVICE, logger=logger
                        )
                    elif scaling_method == ScalingMethod.MinMaxStandard.value:
                        next_input_tensor[param_idx, :, :] = standard_scaler_torch_tensor(next_input_tensor[param_idx, :, :], device=DEVICE)
                else:
                    continue
        validate_scaling(next_input_tensor, scaling_method=scaling_method, logger=logger)

        _, num_channels, _, height, width = before_input_tensor.size()
        return torch.cat((before_input_tensor[:, :, 1:, :, :], torch.reshape(next_input_tensor, (1, num_channels, 1, height, width))), dim=2)

    def __calc_rmse(
        self, pred_ndarray: np.ndarray, label_df: pd.DataFrame, test_case_name: str, date: str, start: str, time_step: int, target_param: str = "rain"
    ) -> Tuple[float, pd.DataFrame]:
        """Calculate RMSE of predict and oberved values and log rmse to mlflow.

        Args:
            pred_ndarray (np.ndarray): _description_
            label_df (pd.DataFrame): _description_
            test_case_name (str): _description_
            date (str): _description_
            start (str): _description_

        Returns:
            pd.DataFrame: Contains columns of hour-rain, Pred_Value, isSequential, case_type, date, observation_point_name
        """
        pred_df = pred_obervation_point_values(pred_ndarray, use_dummy_data=self.hydra_cfg.use_dummy_data)
        result_df = label_df.merge(pred_df, how="outer", left_index=True, right_index=True)
        result_df.dropna()

        result_df["isSequential"] = False
        result_df["case_type"] = test_case_name.split("_case_")[0]
        result_df["date"] = date
        result_df["date_time"] = f"{date}_{start}"
        target_param = PPOTEKACols.get_col_from_weather_param(target_param)
        rmse = mean_squared_error(np.ravel(result_df[target_param].to_numpy()), np.ravel(result_df["Pred_Value"].to_numpy()), squared=False)
        logger.info(f"time_step: {time_step}, RMSE: {rmse}")
        return (rmse, result_df)

    def __visualize_results(self, evaluate_type: str) -> Dict:
        metrics = {}
        output_param_name = self.output_parameter_names[0]
        if self.hydra_cfg.use_dummy_data is True:
            output_param_name = PPOTEKACols.get_col_from_weather_param(output_param_name)
            all_sample_rmse = mean_squared_error(
                np.ravel(self.results_df[output_param_name].to_numpy()), np.ravel(self.results_df["Pred_Value"].to_numpy()), squared=False
            )

            metrics[f"{self.model_name}_{evaluate_type}_All_sample_RMSE"] = all_sample_rmse
        else:
            save_dir_path = os.path.join(self.downstream_direcotry, self.model_name, evaluate_type)
            os.makedirs(save_dir_path, exist_ok=True)
            sample_plot(self.results_df, downstream_directory=save_dir_path, result_metrics=metrics)
            all_cases_plot(self.results_df, downstream_directory=save_dir_path, result_metrics=metrics)
            casetype_plot("tc", self.results_df, downstream_directory=save_dir_path, result_metrics=metrics)
            casetype_plot("not_tc", self.results_df, downstream_directory=save_dir_path, result_metrics=metrics)

            all_sample_rmse = mean_squared_error(
                np.ravel(self.results_df[output_param_name].to_numpy()), np.ravel(self.results_df["Pred_Value"].to_numpy()), squared=False
            )
            metrics[f"{self.model_name}_{evaluate_type}_All_sample_RMSE"] = all_sample_rmse
        return metrics
