from typing import List, Tuple
from enum import Enum, IntEnum


class ScalingMethod(Enum):
    MinMax = "min_max"
    Standard = "standard"
    MinMaxStandard = "min_max_standard"

    @staticmethod
    def is_valid(scaling_method: str) -> bool:
        return scaling_method in [v.value for v in ScalingMethod.__members__.values()]


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

    WIND_DIRECTION_MIN = 0
    WIND_DIRECTION_MAX = 360

    ABS_WIND_MIN = 0.0
    ABS_WIND_MAX = 15.0

    STATION_PRESSURE_MIN = 990.0
    STATION_PRESSURE_MAX = 1025.0

    SEALEVEL_PRESSURE_MIN = 990.0
    SEALEVEL_PRESSURE_MAX = 1025.0

    @staticmethod
    def get_minmax_values_by_weather_param(weather_param_name: str) -> Tuple[float, float]:
        if weather_param_name == WEATHER_PARAMS.RAIN.value:
            return (MinMaxScalingValue.RAIN_MIN.value, MinMaxScalingValue.RAIN_MAX.value)
        elif weather_param_name == WEATHER_PARAMS.TEMPERATURE.value:
            return (MinMaxScalingValue.TEMPERATURE_MIN.value, MinMaxScalingValue.TEMPERATURE_MAX.value)
        elif weather_param_name == WEATHER_PARAMS.WIND.value:
            return (MinMaxScalingValue.WIND_MIN.value, MinMaxScalingValue.WIND_MAX.value)
        elif weather_param_name == WEATHER_PARAMS.ABS_WIND.value:
            return (MinMaxScalingValue.ABS_WIND_MIN.value, MinMaxScalingValue.ABS_WIND_MAX.value)
        elif weather_param_name == WEATHER_PARAMS.STATION_PRESSURE.value:
            return (MinMaxScalingValue.STATION_PRESSURE_MIN.value, MinMaxScalingValue.STATION_PRESSURE_MAX.value)
        elif weather_param_name == WEATHER_PARAMS.SEALEVEL_PRESSURE.value:
            return (MinMaxScalingValue.SEALEVEL_PRESSURE_MIN.value, MinMaxScalingValue.SEALEVEL_PRESSURE_MAX.value)
        else:
            raise ValueError(f"Invalid weather_param_name: {weather_param_name}")
        return

    @staticmethod
    def get_minmax_values_by_ppoteka_cols(ppoteka_col: str) -> Tuple[float, float]:
        if ppoteka_col == PPOTEKACols.RAIN.value:
            return (MinMaxScalingValue.RAIN_MIN.value, MinMaxScalingValue.RAIN_MAX.value)
        elif ppoteka_col == PPOTEKACols.TEMPERATURE.value:
            return (MinMaxScalingValue.TEMPERATURE_MIN.value, MinMaxScalingValue.TEMPERATURE_MAX.value)
        elif ppoteka_col == PPOTEKACols.HUMIDITY.value:
            return (MinMaxScalingValue.HUMIDITY_MIN.value, MinMaxScalingValue.HUMIDITY_MAX.value)
        elif ppoteka_col == PPOTEKACols.WIND_SPEED.value:
            return (MinMaxScalingValue.WIND_MIN.value, MinMaxScalingValue.WIND_MAX.value)
        elif ppoteka_col == PPOTEKACols.WIND_DIRECTION.value:
            return (MinMaxScalingValue.WIND_DIRECTION_MIN.value, MinMaxScalingValue.WIND_DIRECTION_MAX.value)
        elif ppoteka_col == PPOTEKACols.STATION_PRESSURE.value:
            return (MinMaxScalingValue.STATION_PRESSURE_MIN.value, MinMaxScalingValue.STATION_PRESSURE_MAX.value)
        elif ppoteka_col == PPOTEKACols.SEALEVEL_PRESSURE.value:
            return (MinMaxScalingValue.SEALEVEL_PRESSURE_MIN.value, MinMaxScalingValue.SEALEVEL_PRESSURE_MAX.value)
        else:
            raise ValueError(f"Invalid ppoteka_col: {ppoteka_col}")


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
        if not isinstance(params, list):
            raise ValueError(f"`params` should be list. {params}")
        isValid = True
        for p in params:
            isValid = isValid & WEATHER_PARAMS.has_value(p)
        return isValid


class PPOTEKACols(Enum):
    RAIN = "hour-rain"
    TEMPERATURE = "AT1"
    HUMIDITY = "RH1"
    WIND_SPEED = "WS1"
    WIND_DIRECTION = "WD1"
    STATION_PRESSURE = "PRS"
    SEALEVEL_PRESSURE = "SLP"

    @staticmethod
    def get_cols():
        return [v.value for v in PPOTEKACols.__members__.values()]

    @staticmethod
    def get_col_from_weather_param(weather_param_name: str):
        if not WEATHER_PARAMS.is_params_valid([weather_param_name]):
            raise ValueError(f"Invalid weather_param_name: {weather_param_name}")

        if weather_param_name == WEATHER_PARAMS.RAIN.value:
            return PPOTEKACols.RAIN.value

        if weather_param_name == WEATHER_PARAMS.TEMPERATURE.value:
            return PPOTEKACols.TEMPERATURE.value

        if weather_param_name == WEATHER_PARAMS.ABS_WIND.value:
            return PPOTEKACols.WIND_SPEED.value

        if (
            weather_param_name == WEATHER_PARAMS.WIND.value
            or weather_param_name == WEATHER_PARAMS.U_WIND.value  # noqa: W503
            or weather_param_name == WEATHER_PARAMS.V_WIND.value  # noqa: W503
        ):
            return PPOTEKACols.WIND_SPEED.value, PPOTEKACols.WIND_DIRECTION.value

        if weather_param_name == WEATHER_PARAMS.STATION_PRESSURE.value:
            return PPOTEKACols.STATION_PRESSURE.value

        if weather_param_name == WEATHER_PARAMS.SEALEVEL_PRESSURE.value:
            return PPOTEKACols.SEALEVEL_PRESSURE.value


class DIRECTORYS:
    project_root_dir = "/home/akira/Desktop/p-poteka/"
    pipeline_dir = "/home/akira/Desktop/p-poteka/poteka-pipeline-pytorch/"


def isParamsValid(params: List[str]) -> bool:
    isValid = True
    for p in params:
        isValid = isValid & WEATHER_PARAMS.has_value(p)
    return isValid
