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
from common.config import MinMaxScalingValue, PPOTEKACols, ScalingMethod
from common.utils import rescale_tensor, timestep_csv_names
from train.src.config import DEVICE
from evaluate.src.utils import pred_observation_point_values, normalize_tensor, save_parquet, validate_scaling
from evaluate.src.create_image import all_cases_plot, casetype_plot, sample_plot, save_rain_image


logger = CustomLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Evaluator:
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
            if evaluate_type == "combine_models" and not self.hydra_cfg.train.train_sepalately:
                logger.warning("train.sepalate_train is False so `cobine_models` evaluation is skipped.")
            else:
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
            if combine_models: Evaluate the multi parameters trained model and single parameter models.

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
                pred_ndarray=scaled_pred_ndarray,
                label_df=label_dfs[time_step],
                test_case_name=test_case_name,
                date=date,
                start=start,
                time_step=time_step,
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

    def __eval_successibely(self, X_test: torch.Tensor, y_test: torch.Tensor, save_results_dir_path: str, test_case_name: str, evaluate_type: str) -> Dict:
        input_seq_length = X_test.shape[2]
        _time_step_csvnames = timestep_csv_names(time_step_minutes=self.hydra_cfg.preprocess.time_step_minutes)
        date, start = self.test_dataset[test_case_name]["date"], self.test_dataset[test_case_name]["start"]
        start_idx = _time_step_csvnames.index(start)
        start = start.replace(".csv", "")
        label_dfs = self.test_dataset[test_case_name]["label_df"]
        # Evaluate in each timestep
        _X_test = X_test.clone().detach()
        output_param_name = self.output_parameter_names[0]
        rmses = {}
        before_standarized_info = self.test_dataset[test_case_name]["standarize_info"].copy()
        for time_step in range(input_seq_length):
            pred_tensor: torch.Tensor = self.model(_X_test)
            pred_tensor = normalize_tensor(pred_tensor, device=DEVICE)
            self.__validate_pred_tensor(pred_tensor)
            # Check if tensor containes nan values.
            if torch.isnan(pred_tensor).any():
                # logger.warning(self.model.state_dict())
                # logger.warning(pred_tensor)
                logger.warning(f"predicting {test_case_name} in normal way failed and predict tensor contains nan value.")
            # Extract rain tensor
            predict_rain_tensor = pred_tensor[0, 0, 0, ...]
            rain_min_val, rain_max_val = MinMaxScalingValue.get_minmax_values_by_weather_param("rain")
            scaled_rain_pred_tensor = rescale_tensor(min_value=rain_min_val, max_value=rain_max_val, tensor=predict_rain_tensor)
            scaled_rain_pred_ndarray = scaled_rain_pred_tensor.cpu().detach().numpy().copy()
            # Calculate RMSE
            rmse, result_df = self.__calc_rmse(
                pred_ndarray=scaled_rain_pred_ndarray,
                label_df=label_dfs[time_step],
                test_case_name=test_case_name,
                date=date,
                start=start,
                time_step=time_step,
            )
            rmses[time_step] = rmse
            # Save predict informations
            self.results_df = pd.concat([self.results_df, result_df], axis=0)
            utc_time_idx = start_idx + time_step + 6
            if utc_time_idx > len(_time_step_csvnames) - 1:
                utc_time_idx -= len(_time_step_csvnames)
            utc_time_name = _time_step_csvnames[utc_time_idx].replace(".csv", "")
            if self.hydra_cfg.use_dummy_data is False and output_param_name == "rain":
                save_rain_image(scaled_rain_pred_ndarray, os.path.join(save_results_dir_path, f"{utc_time_name}.png"))
            result_df.to_csv(os.path.join(save_results_dir_path, f"pred_observ_df_{utc_time_name}.csv"))
            save_parquet(scaled_rain_pred_ndarray, os.path.join(save_results_dir_path, f"{utc_time_name}.parquet.gzip"))
            # Update next input tensor
            if evaluate_type == "sequential":
                _X_test, before_standarized_info = self.__update_input_tensor(_X_test, before_standarized_info, y_test[0, :, time_step, ...])
            elif evaluate_type == "reuse_predict":
                _X_test, before_standarized_info = self.__update_input_tensor(_X_test, before_standarized_info, pred_tensor[0, :, 0, ...])
            validate_scaling(_X_test, scaling_method=self.hydra_cfg.scaling_method, logger=logger)
        return rmses

    def __eval_combine_models(self, X_test: torch.Tensor, save_results_dir_path: str, test_case_name: str) -> Dict:
        """Evaluate the mutlti parameters trained model and single parameter models.
           e.g. Main model is trained with rain, temperature and humidity (`return_sequence` should be false). And the other two single parameter models
                are trained with temperature and humidity respectively. Then the main model predicts sequentially updating the input data by two single
                parameter models.

                <main model (rain, temp., humid)> --> update rain dimention of the main model's next input tensor
                    <temperature model>           --> update temperature dimention of the main model's next input tensor
                    <humidity model>              --> update humidity dimention of the main model's next input tensor

        Args:
            main_model_name (str): _description_
            main_model_input_parameters (List[str]): _description_
        """
        _, _, input_seq_length, height, width = X_test.size()
        _time_step_csvnames = timestep_csv_names(time_step_minutes=self.hydra_cfg.preprocess.time_step_minutes)
        date, start = self.test_dataset[test_case_name]["date"], self.test_dataset[test_case_name]["start"]
        start_idx = _time_step_csvnames.index(start)
        start = start.replace(".csv", "")
        label_dfs = self.test_dataset[test_case_name]["label_df"]
        # Load sub-models' prediction data
        sub_models_predict_tensor = torch.zeros(
            size=(1, len(self.input_parameter_names), self.hydra_cfg.label_seq_length, height, width), dtype=torch.float, device=DEVICE
        )
        for param_dim, param_name in enumerate(self.input_parameter_names):
            if param_name != "rain":
                results_dir_path = os.path.join(self.downstream_direcotry, param_name, "normal", test_case_name)
                file_paths = self.__sort_predict_data_files(results_dir_path, filename_extention=".parquet.gzip")
                for time_step in range(input_seq_length):
                    pred_df = pd.read_parquet(file_paths[time_step])
                    # pred_tensor shape is (width, height)
                    pred_tensor = torch.from_numpy(pred_df.to_numpy(dtype=np.float32)).to(DEVICE)
                    sub_models_predict_tensor[0, param_dim, time_step, ...] = pred_tensor
        # Evaluate in each timestep
        _X_test = X_test.clone().detach()
        rmses = {}
        before_standarized_info = self.test_dataset[test_case_name]["standarize_info"].copy()
        for time_step in range(input_seq_length):
            pred_tensor = self.model(_X_test)
            pred_tensor = normalize_tensor(pred_tensor, device=DEVICE)
            self.__validate_pred_tensor(pred_tensor)
            # Extract rain tensor
            pred_rain_tensor = pred_tensor[0, 0, 0, ...]
            rain_min_val, rain_max_val = MinMaxScalingValue.get_minmax_values_by_weather_param("rain")
            scaled_rain_pred_tensor = rescale_tensor(min_value=rain_min_val, max_value=rain_max_val, tensor=pred_rain_tensor)
            scaled_rain_pred_ndarray = scaled_rain_pred_tensor.cpu().detach().numpy().copy()
            # Calculate RMSE
            rmse, result_df = self.__calc_rmse(
                pred_ndarray=scaled_rain_pred_ndarray,
                label_df=label_dfs[time_step],
                test_case_name=test_case_name,
                date=date,
                start=start,
                time_step=time_step,
            )
            rmses[time_step] = rmse
            # Save predict information
            self.results_df = pd.concat([self.results_df, result_df], axis=0)
            utc_time_idx = start_idx + time_step + self.hydra_cfg.input_seq_length
            if utc_time_idx > len(_time_step_csvnames):
                utc_time_idx -= len(_time_step_csvnames)
            utc_time_name = _time_step_csvnames[utc_time_idx].replace(".csv", "")
            if self.hydra_cfg.use_dummy_data is False:
                save_rain_image(scaled_rain_pred_ndarray, os.path.join(save_results_dir_path, f"{utc_time_name}.png"))
            result_df.to_csv(os.path.join(save_results_dir_path, f"pred_observ_df_{utc_time_name}.csv"))
            save_parquet(scaled_rain_pred_ndarray, os.path.join(save_results_dir_path, f"{utc_time_name}.parquet.gzip"))
            # Update rain tensor of next input
            sub_models_predict_tensor[0, 0, time_step, ...] = pred_rain_tensor
            _X_test, before_standarized_info = self.__update_input_tensor(_X_test, before_standarized_info, sub_models_predict_tensor[0, :, time_step, ...])
        return rmses

    def __sort_predict_data_files(self, dir_path: str, filename_extention: str) -> List[str]:
        file_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(filename_extention)]
        if len(file_paths) > self.hydra_cfg.label_seq_length:
            raise ValueError(f"Too much predict files: {file_paths}")
        return sorted(file_paths)

    def __update_input_tensor(
        self, before_input_tensor: torch.Tensor, before_standarized_info: Dict, next_input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Update input tensor (X_test) for next prediction step.
           In `sequential` evaluation, the input tensor will be updated with the label tensor, which is the observed tensor.
           In `reuse_predict` evaluation, the input tensor will be updated with the predict tensor.

        Args:
            before_input_tensor (torch.Tensor): tensor with the shape of (batch_size, num_channels, seq_length, height, width)
            next_input_tensor (torch.Tensor): tensor with the shape of (num_channels, height, width). Scaled as original min max values.

        Returns:
            Tuple(torch.Tensor, Dict): tensor with the shape of (batch_size, num_channels, seq_length, height, width)
            and standarize information (mean and std values)
        """
        _, num_channels, _, height, width = before_input_tensor.size()
        # scale next_input_tensor to [0, 1]
        next_input_tensor = normalize_tensor(next_input_tensor, device=DEVICE)
        scaling_method = self.hydra_cfg.scaling_method
        # standarization
        if scaling_method == ScalingMethod.Standard.value or scaling_method == ScalingMethod.MinMaxStandard.value:
            for param_dim, param_name in enumerate(self.input_parameter_names):
                # Rescale using before mean and std
                means, stds = before_standarized_info[param_name]["mean"], before_standarized_info[param_name]["std"]
                before_input_tensor[:, param_dim, ...] = before_input_tensor[:, param_dim, ...] * stds + means
            if before_input_tensor.ndim == 5: # tensor like [1, nun_channels, seq_length, height, width]:
                updated_input_tensor = torch.cat(
                    (before_input_tensor[:, :, 1:, ...], torch.reshape(next_input_tensor, (1, num_channels, 1, height, width))), dim=2
                )
            else:  # tensor like [1, num_channels, seq_len, ob_point_counts]
                ob_point_counts = next_input_tensor.size(dim=3)
                updated_input_tensor = torch.cat(
                        (before_input_tensor[:, :, 1:, ...], torch.reshape(next_input_tensor, (1, num_channels, 1, ob_point_counts)))
                        )
            standarized_info = {}
            for param_dim, param_name in enumerate(self.input_parameter_names):
                standarized_info[param_name] = {}
                means = torch.mean(updated_input_tensor[:, param_dim, ...])
                stds = torch.std(updated_input_tensor[:, param_dim, ...])
                updated_input_tensor[:, param_dim, ...] = (updated_input_tensor[:, param_dim, ...] - means) / stds
                standarized_info[param_name]["mean"] = means
                standarized_info[param_name]["std"] = stds
            validate_scaling(updated_input_tensor, scaling_method=scaling_method, logger=logger)
            return updated_input_tensor, standarized_info
            # if param_name != WEATHER_PARAMS.RAIN.value:
            #     if scaling_method == ScalingMethod.Standard.value:
            #         next_input_tensor[param_dim, :, :] = re_standard_scale(
            #             next_input_tensor[param_dim, :, :], feature_name=param_name, device=DEVICE, logger=logger
            #         )
            #     elif scaling_method == ScalingMethod.MinMaxStandard.value:
            #         next_input_tensor[param_dim, :, :] = standard_scaler_torch_tensor(next_input_tensor[param_dim, :, :], device=DEVICE)
            # else:
            #     continue
        return torch.cat((before_input_tensor[:, :, 1:, ...], torch.reshape(next_input_tensor, (1, num_channels, 1, height, width))), dim=2), {}

    def __calc_rmse(
        self,
        pred_ndarray: np.ndarray,
        label_df: pd.DataFrame,
        test_case_name: str,
        date: str,
        start: str,
        time_step: int,
        target_param: str = "rain",
        show_rmse: bool = False,
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
        pred_df = pred_observation_point_values(pred_ndarray)
        result_df = label_df.merge(pred_df, how="outer", left_index=True, right_index=True)
        result_df.dropna(inplace=True)

        result_df["isSequential"] = False
        result_df["case_type"] = test_case_name.split("_case_")[0]
        result_df["date"] = date
        result_df["date_time"] = f"{date}_{start}"
        target_param = PPOTEKACols.get_col_from_weather_param(target_param)
        rmse = mean_squared_error(np.ravel(result_df[target_param].to_numpy()), np.ravel(result_df["Pred_Value"].to_numpy()), squared=False)
        if show_rmse:
            logger.info(f"time_step: {time_step}, RMSE: {rmse}")
        return (rmse, result_df)

    def __validate_pred_tensor(self, pred_tensor: torch.Tensor) -> None:
        # Check if tensor containes nan values.
        if torch.isnan(pred_tensor).any():
            # logger.warning(self.model.state_dict())
            # logger.warning(pred_tensor)
            logger.warning("Predict tensor contains nan value.")

    def __visualize_results(self, evaluate_type: str) -> Dict:
        metrics = {}
        output_param_name = self.output_parameter_names[0]
        target_poteka_col = PPOTEKACols.get_col_from_weather_param(output_param_name)
        if self.hydra_cfg.use_dummy_data is True:
            all_sample_rmse = mean_squared_error(
                np.ravel(self.results_df[target_poteka_col].to_numpy()), np.ravel(self.results_df["Pred_Value"].to_numpy()), squared=False
            )

            metrics[f"{self.model_name}_{evaluate_type}_All_sample_RMSE"] = all_sample_rmse
        else:
            save_dir_path = os.path.join(self.downstream_direcotry, self.model_name, evaluate_type)
            os.makedirs(save_dir_path, exist_ok=True)
            sample_plot(self.results_df, downstream_directory=save_dir_path, output_param_name=output_param_name, result_metrics=metrics)
            all_cases_plot(self.results_df, downstream_directory=save_dir_path, output_param_name=output_param_name, result_metrics=metrics)
            casetype_plot("tc", self.results_df, downstream_directory=save_dir_path, output_param_name=output_param_name, result_metrics=metrics)
            casetype_plot("not_tc", self.results_df, downstream_directory=save_dir_path, output_param_name=output_param_name, result_metrics=metrics)

            all_sample_rmse = mean_squared_error(
                np.ravel(self.results_df[target_poteka_col].to_numpy()), np.ravel(self.results_df["Pred_Value"].to_numpy()), squared=False
            )
            metrics[f"{self.model_name}_{evaluate_type}_All_sample_RMSE"] = all_sample_rmse
        return metrics
