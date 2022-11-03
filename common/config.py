from typing import List, Tuple, Union
from enum import Enum, IntEnum


class TargetManilaErea:
    MAX_LONGITUDE = 121.150
    MIN_LONGITUDE = 120.90

    MAX_LATITUDE = 14.760
    MIN_LATITUDE = 14.350


class ScalingMethod(Enum):
    MinMax = "min_max"
    Standard = "standard"
    MinMaxStandard = "min_max_standard"

    @staticmethod
    def is_valid(scaling_method: str) -> bool:
        return scaling_method in ScalingMethod.get_methods()

    @staticmethod
    def get_methods() -> List[str]:
        return [v.value for v in ScalingMethod.__members__.values()]


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
        elif weather_param_name == WEATHER_PARAMS.HUMIDITY.value:
            return (MinMaxScalingValue.HUMIDITY_MIN.value, MinMaxScalingValue.HUMIDITY_MAX.value)
        elif (
            weather_param_name == WEATHER_PARAMS.WIND.value
            or weather_param_name == WEATHER_PARAMS.U_WIND.value  # noqa: W503
            or weather_param_name == WEATHER_PARAMS.V_WIND.value  # noqa: W503
        ):
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
        elif ppoteka_col == PPOTEKACols.WIND_SPEED.value or ppoteka_col == PPOTEKACols.U_WIND.value or ppoteka_col == PPOTEKACols.V_WIND.value:
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

    @staticmethod
    def get_param_from_ppoteka_col(ppoteka_col: str) -> Union[str]:
        if ppoteka_col == PPOTEKACols.RAIN.value:
            return WEATHER_PARAMS.RAIN.value
        elif ppoteka_col == PPOTEKACols.TEMPERATURE.value:
            return WEATHER_PARAMS.TEMPERATURE.value
        elif ppoteka_col == PPOTEKACols.HUMIDITY.value:
            return WEATHER_PARAMS.HUMIDITY.value
        elif ppoteka_col == PPOTEKACols.WIND_SPEED.value:
            return WEATHER_PARAMS.ABS_WIND.value
        elif ppoteka_col == PPOTEKACols.U_WIND.value:
            return WEATHER_PARAMS.U_WIND.value
        elif ppoteka_col == PPOTEKACols.V_WIND.value:
            return WEATHER_PARAMS.V_WIND.value
        elif ppoteka_col == PPOTEKACols.STATION_PRESSURE.value:
            return WEATHER_PARAMS.STATION_PRESSURE.value
        elif ppoteka_col == PPOTEKACols.SEALEVEL_PRESSURE.value:
            return WEATHER_PARAMS.SEALEVEL_PRESSURE.value
        elif ppoteka_col == "WD1":
            return "WindDirection"
        else:
            raise ValueError(f"Unknown ppoteka col: {ppoteka_col}")


class PPOTEKACols(Enum):
    RAIN = "hour-rain"
    TEMPERATURE = "AT1"
    HUMIDITY = "RH1"
    WIND_SPEED = "WS1"
    WIND_DIRECTION = "WD1"
    STATION_PRESSURE = "PRS"
    SEALEVEL_PRESSURE = "SLP"
    # calculate in test_data_loader label_dfs
    U_WIND = "U-WIND"
    V_WIND = "V-WIND"

    @staticmethod
    def has_value(item):
        return item in [v.value for v in PPOTEKACols.__members__.values()]

    @staticmethod
    def get_cols():
        return [v.value for v in PPOTEKACols.__members__.values()]

    @staticmethod
    def get_col_from_weather_param(weather_param_name: str) -> str:
        if not WEATHER_PARAMS.is_params_valid([weather_param_name]):
            raise ValueError(f"Invalid weather_param_name: {weather_param_name}")
        if weather_param_name == WEATHER_PARAMS.RAIN.value:
            return PPOTEKACols.RAIN.value
        elif weather_param_name == WEATHER_PARAMS.TEMPERATURE.value:
            return PPOTEKACols.TEMPERATURE.value
        elif weather_param_name == WEATHER_PARAMS.HUMIDITY.value:
            return PPOTEKACols.HUMIDITY.value
        elif weather_param_name == WEATHER_PARAMS.ABS_WIND.value:
            return PPOTEKACols.WIND_SPEED.value
        elif weather_param_name == WEATHER_PARAMS.WIND.value:
            return PPOTEKACols.WIND_SPEED.value
        elif weather_param_name == WEATHER_PARAMS.U_WIND.value:
            return PPOTEKACols.U_WIND.value
        elif weather_param_name == WEATHER_PARAMS.V_WIND.value:
            return PPOTEKACols.V_WIND.value
        elif weather_param_name == WEATHER_PARAMS.STATION_PRESSURE.value:
            return PPOTEKACols.STATION_PRESSURE.value
        elif weather_param_name == WEATHER_PARAMS.SEALEVEL_PRESSURE.value:
            return PPOTEKACols.SEALEVEL_PRESSURE.value

    @staticmethod
    def get_unit(param: str):
        if not PPOTEKACols.has_value(param):
            raise ValueError(f"Ivalid name `{param}` for PPOTEKACols")
        if param == PPOTEKACols.RAIN.value:
            return "mm/h"
        elif param == PPOTEKACols.HUMIDITY.value:
            return "%"
        elif param == PPOTEKACols.TEMPERATURE.value:
            return "℃"
        elif param == PPOTEKACols.WIND_SPEED.value or param == PPOTEKACols.U_WIND.value or PPOTEKACols.V_WIND.value:
            return "m/s"
        elif param == PPOTEKACols.STATION_PRESSURE.value or param == PPOTEKACols.SEALEVEL_PRESSURE.value:
            return "hPa"


class DIRECTORYS:
    project_root_dir = "/home/akira/Desktop/p-poteka/"
    pipeline_dir = "/home/akira/Desktop/p-poteka/poteka-pipeline-pytorch/"


def isParamsValid(params: List[str]) -> bool:
    isValid = True
    for p in params:
        isValid = isValid & WEATHER_PARAMS.has_value(p)
    return isValid
