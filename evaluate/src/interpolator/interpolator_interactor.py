import sys
import numpy as np

sys.path.append(".")
from common.config import WEATHER_PARAMS
from evaluate.src.interpolator.rain_interpolator import RainInterpolator
from evaluate.src.interpolator.temperature_interpolator import TemperatureInterpolator
from evaluate.src.interpolator.humidity_interpolator import HumidityInterplator
from evaluate.src.interpolator.wind_interpolator import WindInterpolator
from evaluate.src.interpolator.pressure_interpolator import PressureInterpolator
from evaluate.src.interpolator.interpolator_interface import InterpolatorInterface


class InterpolatorInteractor:
    def get_interpolator(self, weather_param: str) -> InterpolatorInterface:
        if weather_param == WEATHER_PARAMS.RAIN.value:
            return RainInterpolator()
        elif weather_param == WEATHER_PARAMS.TEMPERATURE.value:
            return TemperatureInterpolator()
        elif weather_param == WEATHER_PARAMS.HUMIDITY.value:
            return HumidityInterplator()
        elif WEATHER_PARAMS.is_weather_param_wind(weather_param):
            return WindInterpolator()
        elif WEATHER_PARAMS.is_weather_param_pressure(weather_param):
            return PressureInterpolator()
        else:
            raise ValueError(f"Invalid weather_param: {weather_param}. Shoud be in {WEATHER_PARAMS.valid_params()}")

    def interpolate(self, weather_param: str, ndarray: np.ndarray, observation_file_path: str) -> np.ndarray:
        interpolator = self.get_interpolator(weather_param)
        return interpolator.interpolate(ndarray, observation_file_path)
