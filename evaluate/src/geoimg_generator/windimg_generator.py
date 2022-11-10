import sys
import numpy as np
import torch
import matplotlib.colors as mcolors
from matplotlib import cm

sys.path.append(".")
from common.utils import get_ob_point_values_from_tensor
from common.config import WEATHER_PARAMS
from evaluate.src.geoimg_generator.geoimg_generator_interface import GeoimgGeneratorInterface
from evaluate.src.geoimg_generator.utils import obpoint_grid_handler, save_img_from_griddata, ob_point_df_from_ndarray


class WindimgGenerator(GeoimgGeneratorInterface):
    def __init__(self, weather_param_name: str) -> None:
        if not WEATHER_PARAMS.is_weather_param_wind(weather_param_name):
            raise ValueError(f"The weather param is invalid ({weather_param_name}). Shoud be wind parameter.")

        self.weather_param_name = weather_param_name

        if weather_param_name == WEATHER_PARAMS.ABS_WIND.value:
            self.color_levels = [i for i in range(0, 16)]
            self.color_map = cm.viridis
        else:
            self.color_levels = [i for i in range(-10, 11)]
            self.color_map = cm.coolwarm

        self.weather_param_unit_label = "m/s"
        super().__init__()

    def gen_img(self, scaled_ndarray: np.ndarray, observation_point_file_path: str, save_img_path: str) -> None:
        ob_point_scaled_ndarray, grid_data = obpoint_grid_handler(self.weather_param_name, scaled_ndarray, observation_point_file_path, save_img_path)
        ob_point_df = ob_point_df_from_ndarray(ob_point_scaled_ndarray, observation_point_file_path)
        save_img_from_griddata(grid_data, ob_point_df, self.color_levels, self.color_map, self.weather_param_unit_label, save_img_path)
