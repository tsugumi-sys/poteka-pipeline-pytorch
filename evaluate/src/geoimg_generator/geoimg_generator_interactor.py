import sys
import numpy as np

sys.path.append(".")
from common.config import WEATHER_PARAMS
from evaluate.src.geoimg_generator.geoimg_generator_interface import GeoimgGeneratorInterface
from evaluate.src.geoimg_generator.rainimg_generator import RainimgGenerator
from evaluate.src.geoimg_generator.temperatureimg_generator import TemperatureimgGenerator
from evaluate.src.geoimg_generator.humidiyimg_generator import HumidityimgGenerator
from evaluate.src.geoimg_generator.windimg_generator import WindimgGenerator
from evaluate.src.geoimg_generator.pressureimg_generator import PressureimgGenerator


class GeoimgGenratorInteractor:
    def get_img_generator(self, weather_param: str) -> GeoimgGeneratorInterface:
        if weather_param == WEATHER_PARAMS.RAIN.value:
            return RainimgGenerator()
        elif weather_param == WEATHER_PARAMS.TEMPERATURE.value:
            return TemperatureimgGenerator()
        elif weather_param == WEATHER_PARAMS.HUMIDITY.value:
            return HumidityimgGenerator()
        elif WEATHER_PARAMS.is_weather_param_wind(weather_param):
            return WindimgGenerator(weather_param)
        elif WEATHER_PARAMS.is_weather_param_pressure(weather_param):
            return PressureimgGenerator(weather_param)
        else:
            raise ValueError(f"Invalid weather_param: {weather_param}. Should be in {WEATHER_PARAMS.valid_params()}")

    def save_img(self, weather_param: str, scaled_ndarray: np.ndarray, observation_point_file_path: str, save_img_path: str) -> None:
        img_generator = self.get_img_generator(weather_param)
        img_generator.gen_img(scaled_ndarray, observation_point_file_path, save_img_path)
