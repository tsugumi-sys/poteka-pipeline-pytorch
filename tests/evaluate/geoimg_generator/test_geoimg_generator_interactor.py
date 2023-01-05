import unittest
import sys
import numpy as np
import os
import shutil

sys.path.append(".")
from common.config import WEATHER_PARAMS, GridSize  # noqa: E402
from evaluate.src.geoimg_generator.geoimg_generator_interactor import GeoimgGenratorInteractor  # noqa: e402
from evaluate.src.geoimg_generator.rainimg_generator import RainimgGenerator  # noqa: E402
from evaluate.src.geoimg_generator.temperatureimg_generator import TemperatureimgGenerator  # noqa: E402
from evaluate.src.geoimg_generator.humidiyimg_generator import HumidityimgGenerator  # noqa: E402
from evaluate.src.geoimg_generator.windimg_generator import WindimgGenerator  # noqa: E402
from evaluate.src.geoimg_generator.pressureimg_generator import PressureimgGenerator  # noqa: E402

try:
    import cartopy  # noqa

    is_cartopy_available = True
except ImportError:
    is_cartopy_available = False


class TestGeoimgGeneratorInteractor(unittest.TestCase):
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

    def test_get_img_generator(self):
        geoimg_interactor = GeoimgGenratorInteractor()
        with self.subTest(weather_param=WEATHER_PARAMS.RAIN.value):
            self.assertIsInstance(geoimg_interactor.get_img_generator(WEATHER_PARAMS.RAIN.value), RainimgGenerator)

        with self.subTest(weather_param=WEATHER_PARAMS.TEMPERATURE.value):
            self.assertIsInstance(
                geoimg_interactor.get_img_generator(WEATHER_PARAMS.TEMPERATURE.value), TemperatureimgGenerator
            )

        with self.subTest(weather_param=WEATHER_PARAMS.HUMIDITY.value):
            self.assertIsInstance(
                geoimg_interactor.get_img_generator(WEATHER_PARAMS.HUMIDITY.value), HumidityimgGenerator
            )

        with self.subTest(weather_param=WEATHER_PARAMS.SEALEVEL_PRESSURE.value):
            self.assertIsInstance(
                geoimg_interactor.get_img_generator(WEATHER_PARAMS.SEALEVEL_PRESSURE.value), PressureimgGenerator
            )

        with self.subTest(weather_param=WEATHER_PARAMS.STATION_PRESSURE.value):
            self.assertIsInstance(
                geoimg_interactor.get_img_generator(WEATHER_PARAMS.STATION_PRESSURE.value), PressureimgGenerator
            )

        with self.subTest(weather_param=WEATHER_PARAMS.ABS_WIND.value):
            self.assertIsInstance(geoimg_interactor.get_img_generator(WEATHER_PARAMS.ABS_WIND.value), WindimgGenerator)

        with self.subTest(weather_param=WEATHER_PARAMS.U_WIND.value):
            self.assertIsInstance(geoimg_interactor.get_img_generator(WEATHER_PARAMS.U_WIND.value), WindimgGenerator)

        with self.subTest(weather_param=WEATHER_PARAMS.V_WIND.value):
            self.assertIsInstance(geoimg_interactor.get_img_generator(WEATHER_PARAMS.V_WIND.value), WindimgGenerator)

    @unittest.skipIf(not is_cartopy_available, "skipped because cartopy is not available")
    def test_save_img(self):
        geoimg_generator = GeoimgGenratorInteractor()
        observation_point_file_path = "./common/meta-data/observation_point.json"
        save_img_path = os.path.join(self.save_dir_path, "geoimg.png")
        obpoint_ndarray = np.random.rand(35)
        grid_data = np.random.rand(GridSize.HEIGHT, GridSize.WIDTH)

        with self.subTest(shape=obpoint_ndarray.shape):
            geoimg_generator.save_img(
                WEATHER_PARAMS.RAIN.value, obpoint_ndarray, observation_point_file_path, save_img_path
            )
            self.assertTrue(os.path.exists(save_img_path))

        with self.subTest(shape=grid_data.shape):
            geoimg_generator.save_img(WEATHER_PARAMS.RAIN.value, grid_data, observation_point_file_path, save_img_path)
            self.assertTrue(os.path.exists(save_img_path))
