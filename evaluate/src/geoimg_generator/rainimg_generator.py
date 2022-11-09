import sys
import numpy as np
import torch
import matplotlib.colors as mcolors

sys.path.append(".")
from common.config import WEATHER_PARAMS
from common.utils import get_ob_point_values_from_tensor
from evaluate.src.geoimg_generator.geoimg_generator_interface import GeoimgGeneratorInterface
from evaluate.src.geoimg_generator.utils import interpolate_img_data, ob_point_df_from_ndarray, save_img


class RainimgGenerator(GeoimgGeneratorInterface):
    def __init__(self) -> None:
        self.color_levels = [0, 5, 7.5, 10, 15, 20, 30, 40, 50, 70, 100]
        cmap_data = [
            (1.0, 1.0, 1.0),
            (0.3137255012989044, 0.8156862854957581, 0.8156862854957581),
            (0.0, 1.0, 1.0),
            (0.0, 0.8784313797950745, 0.501960813999176),
            (0.0, 0.7529411911964417, 0.0),
            (0.501960813999176, 0.8784313797950745, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.627451002597808, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.125490203499794, 0.501960813999176),
            (0.9411764740943909, 0.250980406999588, 1.0),
            (0.501960813999176, 0.125490203499794, 1.0),
        ]
        self.color_map = mcolors.ListedColormap(cmap_data, "precipitation")
        self.weather_param_unit_label = "millimater"
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
