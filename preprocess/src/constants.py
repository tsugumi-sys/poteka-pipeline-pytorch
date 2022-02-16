from typing import List
import enum


class WEATHER_PARAMS_ENUM(enum.Enum):
    RAIN = "rain"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    WIND = "wind"
    ABS_WIND = "abs_wind"
    STATION_PRESSURE = "station_pressure"
    SEALEVEL_PRESSURE = "seaLevel_pressure"

    @staticmethod
    def has_value(item):
        return item in [v.value for v in WEATHER_PARAMS_ENUM.__members__.values()]

    @staticmethod
    def valid_params():
        return [v.value for v in WEATHER_PARAMS_ENUM.__members__.values()]

    @staticmethod
    def is_params_valid(params: List[str]) -> bool:
        isValid = True
        for p in params:
            isValid = isValid & WEATHER_PARAMS_ENUM.has_value(p)
        return isValid


class DIRECTORYS:
    project_root_dir = "/home/akira/Desktop/p-poteka/"


def isParamsValid(params: List[str]) -> bool:
    isValid = True
    for p in params:
        isValid = isValid & WEATHER_PARAMS_ENUM.has_value(p)
    return isValid
