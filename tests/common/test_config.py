import unittest
import sys

sys.path.append(".")
from common.config import WEATHER_PARAMS


class TestConfig(unittest.TestCase):
    def test_WEATHER_PARAMS(self):
        with self.subTest(msg="Test is_weather_params_wind"):
            self.assertTrue(WEATHER_PARAMS.is_weather_param_wind("wind"))
            self.assertTrue(WEATHER_PARAMS.is_weather_param_wind("u_wind"))
            self.assertTrue(WEATHER_PARAMS.is_weather_param_wind("v_wind"))
            self.assertTrue(WEATHER_PARAMS.is_weather_param_wind("abs_wind"))
            self.assertFalse(WEATHER_PARAMS.is_weather_param_wind("rain"))

        with self.subTest(msg="test for is_wather_params_pressure"):
            self.assertTrue(WEATHER_PARAMS.is_weather_param_pressure("station_pressure"))
            self.assertTrue(WEATHER_PARAMS.is_weather_param_pressure("seaLevel_pressure"))
            self.assertFalse(WEATHER_PARAMS.is_weather_param_pressure("wind"))
