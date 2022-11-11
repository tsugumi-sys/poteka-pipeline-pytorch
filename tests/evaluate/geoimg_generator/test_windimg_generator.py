import unittest
import sys
import numpy as np
import os
import shutil

sys.path.append(".")
from common.config import WEATHER_PARAMS, GridSize, MinMaxScalingValue
from evaluate.src.geoimg_generator.windimg_generator import WindimgGenerator


class TestWindimgGenerator(unittest.TestCase):
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

    def test_gen_img(self):
        with self.assertRaises(ValueError):
            _ = WindimgGenerator("invalid-param")

        self._test_gen_img(WEATHER_PARAMS.ABS_WIND.value)
        self._test_gen_img(WEATHER_PARAMS.U_WIND.value)
        self._test_gen_img(WEATHER_PARAMS.V_WIND.value)

    def _test_gen_img(self, weather_param: str):
        observation_point_file_path = "./common/meta-data/observation_point.json"

        min_val, max_val = MinMaxScalingValue.get_minmax_values_by_weather_param(weather_param)
        obpoint_ndarray = np.random.rand(35) * (max_val - min_val) + min_val
        grid_data = np.random.rand(GridSize.HEIGHT, GridSize.WIDTH) * (max_val - min_val) + min_val

        with self.subTest(weather_param=weather_param):
            geoimg_generator = WindimgGenerator(weather_param_name=weather_param)
            save_img_path = os.path.join(self.save_dir_path, f"{weather_param}geoimg.png")

            with self.subTest(shape=obpoint_ndarray.shape):
                geoimg_generator.gen_img(obpoint_ndarray, observation_point_file_path, save_img_path)
                self.assertTrue(os.path.exists(save_img_path))

            with self.subTest(shape=grid_data.shape):
                geoimg_generator.gen_img(grid_data, observation_point_file_path, save_img_path)
                self.assertTrue(os.path.exists(save_img_path))
