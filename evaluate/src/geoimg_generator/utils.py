import logging
import json
from typing import Tuple

import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from common.config import MinMaxScalingValue, TargetManilaErea
from common.utils import get_ob_point_values_from_tensor
from evaluate.src.interpolator.interpolator_interactor import InterpolatorInteractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def interpolate_img_data(scaled_ndarray: np.ndarray, weather_param: str, observation_point_file_path: str) -> np.ndarray:
    min_val, max_val = MinMaxScalingValue.get_minmax_values_by_weather_param(weather_param)
    normalized_ndarray = (scaled_ndarray - min_val) / (max_val - min_val)

    interpolator_interactor = InterpolatorInteractor()
    grid_data = interpolator_interactor.interpolate(weather_param, normalized_ndarray, observation_point_file_path)
    return grid_data * (max_val - min_val) + min_val


def ob_point_df_from_ndarray(ob_point_ndarr: np.ndarray, observation_point_file_path: str) -> pd.DataFrame:
    """
    This function returns ob point dataframe like oneday data.

    Args:
        ob_point_data(np.ndarray): ndarray with one dimention.
        observation_point_file_path(str):

    Return:
        pd.DataFrame: columns are Pred_Value, Lon (longitude) and LAT (latitude). index is observation point name from observation_point_file_path
    """
    with open(observation_point_file_path, "r") as f:
        ob_point_data = json.load(f)

    ob_point_df = pd.DataFrame(
        {"LON": [d["longitude"] for d in ob_point_data.values()], "LAT": [d["latitude"] for d in ob_point_data.values()]}, index=list(ob_point_data.keys())
    )
    pred_df = pd.DataFrame({"Pred_Value": ob_point_ndarr}, index=list(ob_point_data.keys()))
    return ob_point_df.merge(pred_df, right_index=True, left_index=True)


def obpoint_grid_handler(
    weather_param_name: str, scaled_ndarray: np.ndarray, observation_point_file_path: str, save_img_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
        This function convert ndarray like follows.
            1. if scaled_ndarray.ndim == 1, interpolate data to create grid data.
            2. if scaled_ndarray.ndim == 2, extract observation point values from grid data.
            3. else raise ValueError
    """
    if scaled_ndarray.ndim == 1:
        ob_point_scaled_ndarray = scaled_ndarray.copy()
        grid_data = interpolate_img_data(scaled_ndarray, weather_param_name, observation_point_file_path)
    elif scaled_ndarray.ndim == 2:
        ob_point_scaled_tensor = get_ob_point_values_from_tensor(torch.from_numpy(scaled_ndarray.copy()), observation_point_file_path)
        ob_point_scaled_ndarray = ob_point_scaled_tensor.cpu().detach().numpy().copy()
        grid_data = scaled_ndarray.copy()
    else:
        raise ValueError("Invalid ndarray shape for scaled_ndarray")

    return ob_point_scaled_ndarray, grid_data


def save_img_from_griddata(
    grid_data: np.ndarray, ob_point_df: pd.DataFrame, color_levels: list, color_map: mcolors.Colormap, weather_param_unit_label: str, save_img_path: str
) -> None:
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ModuleNotFoundError:
        logger.warning("Cartopy not found in the current env. Skip creating geo image.")
        return None

    grid_lon = np.round(np.linspace(TargetManilaErea.MIN_LONGITUDE, TargetManilaErea.MAX_LONGITUDE), decimals=3)
    grid_lat = np.round(np.linspace(TargetManilaErea.MIN_LATITUDE, TargetManilaErea.MAX_LATITUDE), decimals=3)

    xi, yi = np.meshgrid(grid_lon, grid_lat)

    fig = plt.figure(figsize=(7, 8), dpi=80)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(
        [TargetManilaErea.MIN_LONGITUDE, TargetManilaErea.MAX_LONGITUDE, TargetManilaErea.MIN_LATITUDE, TargetManilaErea.MAX_LATITUDE], crs=ccrs.PlateCarree()
    )

    gl = ax.gridlines(draw_labels=True, alpha=0)
    gl.right_labels = False
    gl.top_labels = False

    cs = ax.contourf(xi, np.flip(yi), grid_data, color_levels, cmap=color_map, norm=mcolors.BoundaryNorm(color_levels, color_map.N))

    color_bar = plt.colorbar(cs, orientation="vertical")
    color_bar.set_label(weather_param_unit_label)

    ax.scatter(ob_point_df["LON"], ob_point_df["LAT"], marker="D", color="dimgrey")

    for idx, pred_val in enumerate(ob_point_df["Pred_Value"]):
        ax.annotate(round(pred_val, 1), (ob_point_df["LON"][idx], ob_point_df["LAT"][idx]))

    ax.add_feature(cfeature.COASTLINE)

    plt.savefig(save_img_path)
    plt.close()
