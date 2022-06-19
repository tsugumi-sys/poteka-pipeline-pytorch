import sys
from typing import Dict
import logging
import os

from omegaconf import DictConfig
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import torch
from torch import nn
from hydra import compose

sys.path.append("..")
from common.config import WEATHER_PARAMS, ScalingMethod
from common.utils import rescale_tensor, timestep_csv_names
from train.src.config import DEVICE
from evaluate.src.utils import pred_obervation_point_values, re_standard_scale, normalize_tensor, standard_scaler_torch_tensor, validate_scaling


logger = logging.getLogger("EvaluateLogger")

class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        test_dataset: Dict,
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
        self.test_dataset = test_dataset
        self.downstream_direcotry = downstream_directory
        self.rmse_df = pd.DataFrame(columns=["isSequential", "case_type", "date", "date_time", "hour-rain", ])
        self.hydra_cfg = self.__initialize_hydar_conf()
        pass

    def __initialize_hydar_conf(self) -> DictConfig:
        cfg = compose(config_name="config")
        return cfg

    def run(self) -> Dict:
        return
    def __evaluate_normal(self) -> pd.DataFrame:
        logger.info("- Evaluating 1 hour prediction")
        results_df = pd.DataFrame(columns=["isSequential", "case_type", "date", "date_time", "hour-rain", "Pred_Value"])
        self.model.eval()
        with torch.no_grad():
            _time_step_csvnames = timestep_csv_names(time_step_minutes=self.hydra_cfg.preprocess.time_step_minutes)
            for sample_name in self.test_dataset.keys():
                logger.info(f"Evaluate {sample_name}")
                X_test: torch.Tensor = self.test_dataset[sample_name]["input"]
                y_test: torch.Tensor = self.test_dataset[sample_name]["label"]
                X_test = X_test.to(DEVICE)
                y_test = y_test.to(DEVICE)

                y_test_size = y_test.size()
                y_test_batch_size = y_test_size[0]
                num_channels = y_test_size[1]
                height, width = y_test_size[3], y_test_size[4]

                input_seq_length = X_test.shape[2]

                validate_scaling(y_test, scaling_method=ScalingMethod.MinMax.value, logger=logger)
                validate_scaling(X_test, scaling_method=self.hydra_cfg.scaling_method)

                # [TODO]: Use pydantic to define test_dataset
                date, start = self.test_dataset[sample_name]["date"], self.test_dataset[sample_name]["start"]
                start_idx = _time_step_csvnames.index(start)
                start = start.replace()
                label_dfs = self.test_dataset[sample_name]["label_df"]

                save_dir = os.path.join(self.downstream_direcotry, sample_name)
                os.makedirs(save_dir, exist_ok=True)

                _X_test = X_test.clone().detach()
                for t in range(input_seq_length):
                    # pred_tensor: Predict tensor shape is (batch_size=1, num_channels, seq_length, height, width)
                    pred_tensor: torch.Tensor = self.model(_X_test)

                    # Check if tensor containes nan values.
                    if torch.isnan(pred_tensor).any():
                        # logger.warning(self.model.state_dict())
                        # logger.warning(pred_tensor)
                        logger.warning(f"predicting {sample_name} in normal way failed and predict tensor contains nan value.")

                    pred_tensor = normalize_tensor(pred_tensor, device=DEVICE)
                    predict_rain_tensor = pred_tensor[0, 0, 0, :, :]
                    scaled_pred_tensor = rescale_tensor(min_value=0, max_value=100, tensor=predict_rain_tensor)
                    scaled_pred_ndarray = scaled_pred_tensor.cpu().detach().numpy().copy()


            

    def __evaluate_sequential(self) -> pd.DataFrame:
        return

    def __calc_rmse(self, pred_ndarray: np.ndarray, label_df: pd.DataFrame, sample_name: str, date: str, start: str,) -> pd.DataFrame:
        pred_df = pred_obervation_point_values(pred_ndarray, use_dummy_data=self.hydra_cfg.use_dummy_data)
        result_df = label_df.merge(pred_df, how="outer", left_index=True, right_index=True)
        result_df.dropna()

        result_df["isSequential"] = False
        result_df["case_type"] = sample_name.split("_case_")[0]
        result_df["date"] = date
        result_df = pd.concat([result_df, ])


    def __visualize_retults(self) -> None:

    
