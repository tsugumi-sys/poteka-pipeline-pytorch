import sys
import numpy as np
import torch
import matplotlib.colors as mcolors
from matplotlib import cm

sys.path.append(".")
from common.utils import get_ob_point_values_from_tensor
from common.config import WEATHER_PARAMS
from evaluate.src.geoimg_generator.geoimg_generator_interface import GeoimgGeneratorInterface
from evaluate.src.geoimg_generator.utils import interpolate_img_data, ob_point_df_from_ndarray, save_img


class WindimgGenerator(GeoimgGeneratorInterface):
    def __init__(self, weather_param_name: str) -> None:
        if not WEATHER_PARAMS.is_weather_param_wind(weather_param_name):
            raise ValueError(f"The weather param is invalid ({weather_param_name}). Shoud be wind parameter.")

        if weather_param_name == WEATHER_PARAMS.ABS_WIND.value:
            self.color_levels = [i for i in range(0, 101, 5)]
            self.color_map = cm.viridis
        else:
            self.color_levels = [i for i in range(-10, 11)]
            self.color_map = cm.coolwarm

        self.weather_param_unit_label = "m/s"
        super().__init__()

    def gen_img(self, scaled_ndarray: np.ndarray, observation_point_file_path: str, save_img_path: str) -> None:
        if scaled_ndarray.ndim == 1:
            ob_point_scaled_ndarray = scaled_ndarray.copy()
            grid_data = interpolate_img_data(scaled_ndarray, WEATHER_PARAMS.RAIN.value, observation_point_file_path)
        elif scaled_ndarray.ndim == 2:
            ob_point_scaled_tensor = get_ob_point_values_from_tensor(torch.from_numpy(scaled_ndarray.copy()), observation_point_file_path)
            ob_point_scaled_ndarray = ob_point_scaled_tensor.cpu().detach().numpy().cppy()
            grid_data = scaled_ndarray.copy()
        else:
            raise ValueError("Invalid ndarray shape for scaled_ndarray")

        ob_point_df = ob_point_df_from_ndarray(ob_point_scaled_ndarray, observation_point_file_path)
        save_img(grid_data, ob_point_df, self.color_levels, self.color_map, self.weather_param_unit_label, save_img_path)
