import sys
from typing import Dict, List, Optional, Tuple
import logging
import os
import json

from omegaconf import DictConfig
import pandas as pd
import numpy as np
from sklearn.base import check_array
from sklearn.metrics import mean_squared_error, r2_score
import torch
from torch import nn
from hydra import compose

sys.path.append("..")
from common.custom_logger import CustomLogger  # noqa: E402
from common.config import GridSize, MinMaxScalingValue, PPOTEKACols  # noqa: E402
from common.utils import get_ob_point_values_from_tensor, rescale_tensor, timestep_csv_names  # noqa: E402
from train.src.config import DEVICE  # noqa: E402
from evaluate.src.utils import normalize_tensor, save_parquet  # noqa: E402
from evaluate.src.create_image import all_cases_scatter_plot, date_scatter_plot, save_rain_image  # noqa: E402


logger = CustomLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseEvaluator:
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
        """This class is a base class for evaluators.

        Args:
            model (nn.Module): target model.
            model_name (str): target model name.
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
            input_parameter_names (List[str]): Input parameter names.
            output_parameter_names (List[str]): Output parameter names.
            downstream_directory (str): Downstream directory path.
            observation_point_file_path (str): A observation point file path (observation_point.json)
            hydra_overrides (List[str]): Override information of hydra configurations.
        """
        self.model = model
        self.model_name = model_name
        self.test_dataset = test_dataset
        self.input_parameter_names = input_parameter_names
        self.output_parameter_names = output_parameter_names
        self.downstream_direcotry = downstream_directory
        self.observation_point_file_path = observation_point_file_path
        self.results_df = pd.DataFrame()
        self.metrics_df = pd.DataFrame()
        self.hydra_cfg = self.__initialize_hydar_conf(hydra_overrides)

    def __initialize_hydar_conf(self, overrides: List[str]) -> DictConfig:
        cfg = compose(config_name="config", overrides=overrides)
        return cfg

    def load_test_case_dataset(self, test_case_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        X_test = self.test_dataset[test_case_name]["input"].to(DEVICE)
        y_test = self.test_dataset[test_case_name]["label"].to(DEVICE)
        return X_test, y_test

    def rescale_pred_tensor(self, target_tensor: torch.Tensor, target_param: str) -> torch.Tensor:
        """This function rescale target_tensor to target parameter's original scale.
        e.g. rain tensor [0, 1] -> [0, 100]
        """
        target_tensor = normalize_tensor(target_tensor, DEVICE)
        min_val, max_val = MinMaxScalingValue.get_minmax_values_by_weather_param(target_param)
        rescaled_tensor = rescale_tensor(min_value=min_val, max_value=max_val, tensor=target_tensor.cpu().detach())
        return rescaled_tensor

    def add_result_df_from_pred_tensor(
        self, test_case_name: str, time_step: int, pred_tensor: torch.Tensor, label_df: pd.DataFrame, target_param: str,
    ) -> None:
        """This function is a interface for add result_df to self.result_df.

        Args:
            test_case_name (str):
            time_step (int):
            pred_tensor (torch.Tensor): A prediction tensor of a certain test case. This tensor should be scaled to its original scale.
            label_df (pd.DataFrame): A pandas dataframe of observation data.
            target_param (str):
        """
        pred_df = self.get_pred_df_from_tensor(pred_tensor)  # This dataframe has columns ["Pred_Value"] and index is observation names.
        result_df = label_df.merge(pred_df, right_index=True, left_index=True)
        result_df["test_case_name"] = test_case_name
        result_df["date"] = self.test_dataset[test_case_name]["date"]
        result_df["predict_utc_time"] = self.get_prediction_utc_time(test_case_name, time_step)
        result_df["target_parameter"] = target_param
        self.results_df = pd.concat([self.results_df, result_df], axis=0)

    def add_metrics_df_from_pred_tensor(
        self, test_case_name: str, time_step: int, pred_tensor: torch.Tensor, label_df: pd.DataFrame, target_param: str,
    ):
        """This function is a interface to add metrics_df from pred_tensor and label_df

        Args:
            test_case_name (str):
            time_step (int):
            pred_tensor (torch.Tensor):
            label_df (pd.DataFrame):
            target_param (str):
        """
        rmse_score = self.rmse_from_pred_tensor(pred_tensor, label_df, target_param)
        r2_score = self.r2_score_from_pred_tensor(pred_tensor, label_df, target_param)
        pred_utc_time = self.get_prediction_utc_time(test_case_name, time_step)

        metrics_df = pd.DataFrame(
            {
                "test_case_name": [test_case_name],
                "predict_utc_time": [pred_utc_time],
                "target_parameter": [target_param],
                "r2": [r2_score],
                "rmse": [rmse_score],
            }
        )
        self.metrics_df = pd.concat([self.metrics_df, metrics_df])

    def get_prediction_utc_time(self, test_case_name: str, time_step: int) -> str:
        """This function get UTC time of a prediction time of given test case and time step.

        Args:
            test_case_name (str): A test case name.
            time_step (int): A prediction time step from start time of a given test case.
        """
        _time_step_csvnames = timestep_csv_names(time_step_minutes=self.hydra_cfg.preprocess.time_step_minutes)
        predict_start = self.test_dataset[test_case_name]["start"]  # "hh-mm.csv"
        predict_start_idx = _time_step_csvnames.index(predict_start)

        utc_time_idx = predict_start_idx + time_step

        if utc_time_idx > len(_time_step_csvnames) - 1:
            utc_time_idx -= len(_time_step_csvnames)

        return _time_step_csvnames[utc_time_idx].replace(".csv", "")

    # def __get_metrics_overview():
    def rmse_from_pred_tensor(self, pred_tensor: torch.Tensor, label_df: pd.DataFrame, target_param: str) -> float:
        """This function return rmse value between prediction and observation values.

        Args:
            pred_tensor (torch.Tensor): A prediction tensor of a certain test case. This tensor should be scaled to its original scale.
            label_df (pd.DataFrame): A pandas dataframe of observation data.
            target_param (str): A target weather parameter name.
        """
        # Pred_Value contains prediction values of each observation points.
        pred_df = self.get_pred_df_from_tensor(pred_tensor)
        check_df = pred_df.merge(label_df, right_index=True, left_index=True)
        rmse = mean_squared_error(
            np.ravel(check_df["Pred_Value"].astype(float).to_numpy()),
            np.ravel(check_df[PPOTEKACols.get_col_from_weather_param(target_param)].astype(float).to_numpy()),
            squared=False,
        )
        return rmse

    def rmse_from_results_df(self, output_param_name: str, target_date: Optional[str] = None, is_tc_case: Optional[bool] = None) -> float:
        """This function calculate r2 score from results_df. Querying results_df with date and case_type:

        Args:
            output_param_name (str): A output weather parameter.
            target_date (Optional[str]): A target date for querying results_df.
            is_tc_case (Optional[bool]): A tc or not tc case type for querying results_df.

        """
        target_poteka_col = PPOTEKACols.get_col_from_weather_param(output_param_name)

        df = self.query_result_df(target_date=target_date, is_tc_case=is_tc_case)
        rmse = mean_squared_error(np.ravel(df[target_poteka_col].astype(float).to_numpy()), np.ravel(df["Pred_Value"].astype(float).to_numpy()),)

        return rmse

    def r2_score_from_pred_tensor(self, pred_tensor: torch.Tensor, label_df: pd.DataFrame, target_param: str) -> float:
        """This funtion return r2 score between prediction and observation values.

        Args:
            pred_tensor (torch.Tensor): A prediction tensor of a certain test case. This tensor should be scaled to its original scale.
            label_df (pd.DataFrame): A pandas dataframe of observation data.
            target_param (str): A target weather parameter name.
        """
        pred_df = self.get_pred_df_from_tensor(pred_tensor)
        check_df = pred_df.merge(label_df, right_index=True, left_index=True)
        r2_score_val = r2_score(
            np.ravel(check_df["Pred_Value"].astype(float).to_numpy()),
            np.ravel(check_df[PPOTEKACols.get_col_from_weather_param(target_param)].astype(float).to_numpy()),
        )
        return r2_score_val

    def r2_score_from_results_df(self, output_param_name: str, target_date: Optional[str] = None, is_tc_case: Optional[bool] = None) -> float:
        """This function calculate r2 score from results_df. Querying results_df with date and case_type:

        Args:
            output_param_name (str): A output weather parameter.
            target_date (Optional[str]): A target date for querying results_df.
            is_tc_case (Optional[bool]): A tc or not tc case type for querying results_df.

        """
        target_poteka_col = PPOTEKACols.get_col_from_weather_param(output_param_name)

        df = self.query_result_df(target_date=target_date, is_tc_case=is_tc_case)

        r2_score_value = r2_score(np.ravel(df[target_poteka_col].astype(float).to_numpy()), np.ravel(df["Pred_Value"].astype(float).to_numpy()),)
        return r2_score_value

    def query_result_df(self, target_date: Optional[str] = None, is_tc_case: Optional[bool] = None):
        "This function get results_df queried with target date and is_tc_case flag."
        df = self.results_df.copy()
        if target_date is not None:
            query = [target_date in d for d in self.results_df["date"]]
            df = df.loc[query]

        if is_tc_case is not None:
            if is_tc_case is True:
                query = df["case_type"] == "tc"
            else:
                query = df["case_type"] == "not_tc"
            df = df.loc[query]

        return df

    def get_pred_df_from_tensor(self, pred_tensor: torch.Tensor) -> pd.DataFrame:
        """This function generates prediction dataframe from prediction tensor.

        Return (pd.DataFrame): A prediction dataframe (columns: [`Pred_Value`], index: obsevation points name).
        """
        if pred_tensor.ndim > 2:
            raise ValueError(f"Invalid tensor dimentions for pred_tensor. The shape shold be less than 2, but {pred_tensor.shape}.")

        if pred_tensor.shape == torch.Size([GridSize.WIDTH, GridSize.HEIGHT]):
            pred_ob_point_tensor = get_ob_point_values_from_tensor(pred_tensor, self.observation_point_file_path)
        elif pred_tensor.shape == torch.Size([35]):
            pred_ob_point_tensor = pred_tensor
        else:
            raise ValueError(f"Invalid predict tensor shape.: {pred_tensor.shape}")

        with open(self.observation_point_file_path, "r") as f:
            ob_point_data = json.load(f)

        pred_df = pd.DataFrame(index=list(ob_point_data.keys()))
        pred_df["Pred_Value"] = pred_ob_point_tensor.clone().detach().numpy().copy()  # torch.float convert to numpy.float32
        return pred_df

    def save_results_df_to_csv(self, save_dir_path: str) -> None:
        if isinstance(self.results_df, pd.DataFrame):
            self.results_df.to_csv(os.path.join(save_dir_path, "predict_result.csv"))
        else:
            logger.warning("Results dataframe is not initialized")

    def save_metrics_df_to_csv(self, save_dir_path: str) -> None:
        self.metrics_df.to_csv(os.path.join(save_dir_path, "predict_metrics.csv"))

    def scatter_plot(self, save_dir_path: str) -> None:
        """This function creates scatter plot of prediction and observation data of `results_df`."""
        output_param_name = self.output_parameter_names[0]
        all_cases_scatter_plot(
            result_df=self.query_result_df(),
            downstream_directory=save_dir_path,
            output_param_name=output_param_name,
            r2_score=self.r2_score_from_results_df(output_param_name=output_param_name),
        )

        unique_dates = self.results_df["date"].unique().tolist()
        for date in unique_dates:
            date_scatter_plot(
                result_df=self.query_result_df(target_date=date),
                downstream_directory=save_dir_path,
                output_param_name=output_param_name,
                date=date,
                r2_score=self.r2_score_from_results_df(output_param_name=output_param_name, target_date=date),
            )

        # casetype_scatter_plot(
        #    result_df=self.query_result_df(is_tc_case=True),
        #    case_type="tc",
        #    downstream_directory=save_dir_path,
        #    output_param_name=output_param_name,
        #    r2_score=self.r2_score_from_results_df(output_param_name=output_param_name, is_tc_case=True),
        # )
        # casetype_scatter_plot(
        #    result_df=self.query_result_df(is_tc_case=True),
        #    case_type="not_tc",
        #    downstream_directory=save_dir_path,
        #    output_param_name=output_param_name,
        #    r2_score=self.r2_score_from_results_df(output_param_name=output_param_name, is_tc_case=True),
        # )

    def geo_plot(self, test_case_name: str, save_dir_path: str, pred_tensors: torch.Tensor) -> None:
        """This function create and save geo plotted images (Mainly used for rainfall images).
            And saving prediction grid array.

        Args:
            test_case_name (str): A test case name of the prediction.
            save_dir_path (str): Save directory path.
            pred_tensors (torch.Tensor): Prediction tensors of a test case. The shape is [batch=1, n_featuures=1, seq_length, height, width].
                This tensor should be scaled.
        """
        for time_step in range(self.hydra_cfg.label_seq_length):
            pred_ndarray = pred_tensors[0, 0, time_step, ...].numpy().copy()
            utc_time_name = self.get_prediction_utc_time(test_case_name, time_step)
            if self.hydra_cfg.use_dummy_data is False:
                save_rain_image(pred_ndarray, os.path.join(save_dir_path, f"{utc_time_name}.png"))
            save_parquet(pred_ndarray, os.path.join(save_dir_path, f"{utc_time_name}.parquet.gzip"), self.observation_point_file_path)
