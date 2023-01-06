import os
import shutil
import sys
import unittest

import numpy as np

sys.path.append(".")
from common.config import WEATHER_PARAMS, GridSize, MinMaxScalingValue  # noqa: E402
from evaluate.src.geoimg_generator.humidiyimg_generator import HumidityimgGenerator  # noqa: E402

try:
    import cartopy  # noqa

    is_cartopy_available = True
except ImportError:
    is_cartopy_available = False


class TestHumidityimgGenerator(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        self.save_dir_path = "./tmp"
        super().__init__(methodName)

    def setUp(self) -> None:
        if os.path.exists(self.save_dir_path):
            shutil.rmtree(self.save_dir_path)
        os.makedirs(self.save_dir_path, exist_ok=True)
        return super().setUp()

    def tearDown(self) -> None:
        shutil.rmtree(self.save_dir_path)
        return super().tearDown()

    @unittest.skipIf(not is_cartopy_available, "skipped because cartopy is not available")
    def test_gen_img(self):
        geoimg_generator = HumidityimgGenerator()
        observation_point_file_path = "./common/meta-data/observation_point.json"
        save_img_path = os.path.join(self.save_dir_path, "geoimg.png")

        min_val, max_val = MinMaxScalingValue.get_minmax_values_by_weather_param(WEATHER_PARAMS.HUMIDITY.value)
        obpoint_ndarray = np.random.rand(35) * (max_val - min_val) + min_val
        grid_data = np.random.rand(GridSize.HEIGHT, GridSize.WIDTH) * (max_val - min_val) + min_val

        with self.subTest(shape=obpoint_ndarray.shape):
            geoimg_generator.gen_img(obpoint_ndarray, observation_point_file_path, save_img_path)
            self.assertTrue(os.path.exists(save_img_path))

        with self.subTest(shape=grid_data.shape):
            geoimg_generator.gen_img(grid_data, observation_point_file_path, save_img_path)
            self.assertTrue(os.path.exists(save_img_path))
