from typing import List
from enum import Enum, IntEnum


class ScalingMethod(Enum):
    MinMax = "min_max"
    Standard = "standard"
    MinMaxStandard = "min_max_standard"


class GridSize(IntEnum):
    WIDTH = 50
    HEIGHT = 50


class MinMaxScalingValue(IntEnum):
    RAIN_MIN = 0.0
    RAIN_MAX = 100.0

    TEMPERATURE_MIN = 10.0
    TEMPERATURE_MAX = 45.0

    HUMIDITY_MIN = 0.0
    HUMIDITY_MAX = 100.0

    WIND_MIN = -10.0
    WIND_MAX = 10.0

    ABS_WIND_MIN = 0.0
    ABS_WIND_MAX = 15.0

    STATION_PRESSURE_MIN = 990.0
    STATION_PRESSURE_MAX = 1025.0

    SEALEVEL_PRESSURE_MIN = 990.0
    SEALEVEL_PRESSURE_MAX = 1025.0


class WEATHER_PARAMS(Enum):
    RAIN = "rain"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    WIND = "wind"
    U_WIND = "u_wind"
    V_WIND = "v_wind"
    ABS_WIND = "abs_wind"
    STATION_PRESSURE = "station_pressure"
    SEALEVEL_PRESSURE = "seaLevel_pressure"

    @staticmethod
    def has_value(item):
        return item in [v.value for v in WEATHER_PARAMS.__members__.values()]

    @staticmethod
    def valid_params():
        return [v.value for v in WEATHER_PARAMS.__members__.values()]

    @staticmethod
    def is_params_valid(params: List[str]) -> bool:
        isValid = True
        for p in params:
            isValid = isValid & WEATHER_PARAMS.has_value(p)
        return isValid


class DIRECTORYS:
    project_root_dir = "/home/akira/Desktop/p-poteka/"
    pipeline_dir = "/home/akira/Desktop/p-poteka/poteka-pipeline-pytorch/"


def isParamsValid(params: List[str]) -> bool:
    isValid = True
    for p in params:
        isValid = isValid & WEATHER_PARAMS.has_value(p)
    return isValid
