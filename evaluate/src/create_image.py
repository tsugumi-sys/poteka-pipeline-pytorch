import json
import logging
import os
from typing import Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from common.config import MinMaxScalingValue, PPOTEKACols
from common.utils import get_ob_point_values_from_tensor
from evaluate.src.interpolator.interpolator_interactor import InterpolatorInteractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_rain_image(
    scaled_rain_ndarray: np.ndarray,
    observation_point_file_path: str,
    save_path: str,
):
    """Save prediction images of rainfalls."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ModuleNotFoundError:
        logger.warning("Cartopy not found in the current env. Skip creating image with cartopy.")
        return None

    if scaled_rain_ndarray.ndim == 1:
        ob_point_pred_ndarray = scaled_rain_ndarray
        min_val, max_val = MinMaxScalingValue.get_minmax_values_by_weather_param("rain")
        scaled_rain_ndarray = (scaled_rain_ndarray - min_val) / (max_val - min_val)

        interpolator_interactor = InterpolatorInteractor()
        scaled_rain_ndarray = interpolator_interactor.interpolate(
            "rain", scaled_rain_ndarray, observation_point_file_path
        )
        scaled_rain_ndarray = scaled_rain_ndarray * (max_val - min_val) + min_val
    else:
        ob_point_pred_tensor = get_ob_point_values_from_tensor(
            torch.from_numpy(scaled_rain_ndarray), observation_point_file_path
        )
        ob_point_pred_ndarray = ob_point_pred_tensor.cpu().detach().numpy().copy()

    if scaled_rain_ndarray.ndim != 2:
        raise ValueError("Invalid ndarray shape for `scaled_rain_ndarray`. The shape should be (Height, Widht).")

    with open(observation_point_file_path, "r") as f:
        ob_point_data = json.load(f)

    ob_point_df = pd.DataFrame(
        {
            "LON": [d["longitude"] for d in ob_point_data.values()],
            "LAT": [d["latitude"] for d in ob_point_data.values()],
        },
        index=list(ob_point_data.keys()),
    )
    pred_df = pd.DataFrame({"Pred_Value": ob_point_pred_ndarray}, index=list(ob_point_data.keys()))
    ob_point_df = ob_point_df.merge(pred_df, right_index=True, left_index=True)

    grid_lon = np.round(np.linspace(120.90, 121.150, 50), decimals=3)
    grid_lat = np.round(np.linspace(14.350, 14.760, 50), decimals=3)
    xi, yi = np.meshgrid(grid_lon, grid_lat)
    fig = plt.figure(figsize=(7, 8), dpi=80)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([120.90, 121.150, 14.350, 14.760], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, alpha=0)
    gl.right_labels = False
    gl.top_labels = False

    clevs = [0, 5, 7.5, 10, 15, 20, 30, 40, 50, 70, 100]
    cmap_data = [
        (1.0, 1.0, 1.0),
        (0.3137255012989044, 0.8156862854957581, 0.8156862854957581),
        (0.0, 1.0, 1.0),
        (0.0, 0.8784313797950745, 0.501960813999176),
        (0.0, 0.7529411911964417, 0.0),
        (0.501960813999176, 0.8784313797950745, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.627451002597808, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.125490203499794, 0.501960813999176),
        (0.9411764740943909, 0.250980406999588, 1.0),
        (0.501960813999176, 0.125490203499794, 1.0),
    ]
    cmap = mcolors.ListedColormap(cmap_data, "precipitation")
    norm = mcolors.BoundaryNorm(clevs, cmap.N)

    cs = ax.contourf(xi, np.flip(yi), scaled_rain_ndarray, clevs, cmap=cmap, norm=norm)
    cbar = plt.colorbar(cs, orientation="vertical")
    cbar.set_label("millimeter")
    ax.scatter(ob_point_df["LON"], ob_point_df["LAT"], marker="D", color="dimgrey")
    for idx, pred_value in enumerate(ob_point_df["Pred_Value"]):
        ax.annotate(round(pred_value, 1), (ob_point_df["LON"][idx], ob_point_df["LAT"][idx]))

    # The white color of rain erase the coastline so add here.
    ax.add_feature(cfeature.COASTLINE)
    plt.savefig(save_path)
    plt.close()


def get_r2score_text_position(max_val: float, min_val: float) -> Tuple[float, float]:
    x_pos = min_val + (max_val - min_val) * 0.03
    y_pos = max_val * 0.95
    return x_pos, y_pos


def all_cases_scatter_plot(
    result_df: pd.DataFrame,
    downstream_directory: str,
    output_param_name: str,
    r2_score: float,
    save_fig_name: Optional[str] = None,
):
    """Save scatter plots of predictions and observations of all test cases."""
    r2_score = np.round(r2_score, 4)
    target_poteka_col = PPOTEKACols.get_col_from_weather_param(output_param_name)
    target_param_unit = PPOTEKACols.get_unit(target_poteka_col)
    target_param_min_val, target_param_max_val = MinMaxScalingValue.get_minmax_values_by_ppoteka_cols(target_poteka_col)
    text_position_x, text_position_y = get_r2score_text_position(
        max_val=target_param_max_val, min_val=target_param_min_val
    )
    # With TC, NOT TC hue.
    plt.figure(figsize=(6, 6))
    ax = sns.scatterplot(data=result_df, x=target_poteka_col, y="Pred_Value", hue="date")
    ax.text(text_position_x, text_position_y, f"R2-Score: {r2_score}", size=15)

    x = np.linspace(
        target_param_min_val, target_param_max_val, int((target_param_max_val - target_param_min_val) // 10)
    )
    ax.plot(x, x, color="blue", linestyle="--")

    ax.set_xlim(target_param_min_val, target_param_max_val)
    ax.set_ylim(target_param_min_val, target_param_max_val)
    ax.set_title(f"{output_param_name} Scatter plot of all validation cases.")
    ax.set_xlabel(f"Observation value {target_param_unit}")
    ax.set_ylabel(f"Prediction value {target_param_unit}")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_fig_name is None:
        plt.savefig(os.path.join(downstream_directory, "all_cases.png"))
    else:
        plt.savefig(os.path.join(downstream_directory, save_fig_name))
    plt.close()


def date_scatter_plot(
    result_df: pd.DataFrame,
    date: str,
    downstream_directory: str,
    output_param_name: str,
    r2_score: float,
    save_fig_name: Optional[str] = None,
):
    """Save scatter plots of prediction vs obervation of a given date.

    Args:
        rmses_df (pd.DataFrame):
        downstream_directory (str):
        output_param_name (str): weather param name
        r2_score: r2_score of given datasets.
    """
    r2_score = np.round(r2_score, 4)
    # Add date_time column
    result_df["date_time"] = result_df["date"] + "_" + result_df["predict_utc_time"] + "start"
    # create each sample scatter plot. hue is date_time.
    target_poteka_col = PPOTEKACols.get_col_from_weather_param(output_param_name)
    target_param_unit = PPOTEKACols.get_unit(target_poteka_col)
    target_param_min_val, target_param_max_val = MinMaxScalingValue.get_minmax_values_by_ppoteka_cols(target_poteka_col)
    text_position_x, text_position_y = get_r2score_text_position(
        max_val=target_param_max_val, min_val=target_param_min_val
    )

    plt.figure(figsize=(6, 6))

    # NOTE: hue=date_time is too huge for plot. The legend makes the plot odd shape.
    ax = sns.scatterplot(data=result_df, x=target_poteka_col, y="Pred_Value")
    # plot r2 score line.
    ax.text(text_position_x, text_position_y, f"R2-Score: {r2_score}", size=15)
    # plot base line (cc = 1)
    x = np.linspace(
        target_param_min_val, target_param_max_val, int((target_param_max_val - target_param_min_val) // 10)
    )
    ax.plot(x, x, color="blue", linestyle="--")

    ax.set_xlim(target_param_min_val, target_param_max_val)
    ax.set_ylim(target_param_min_val, target_param_max_val)
    ax.set_title(f"{output_param_name} Scatter plot of {date} cases.")
    ax.set_xlabel(f"Observation value {target_param_unit}")
    ax.set_ylabel(f"Prediction value {target_param_unit}")
    # ax.legend(loc="lower right")
    plt.tight_layout()

    if save_fig_name is None:
        plt.savefig(os.path.join(downstream_directory, f"{date}_cases.png"))
    else:
        plt.savefig(os.path.join(downstream_directory, save_fig_name))

    plt.close()


def casetype_scatter_plot(
    result_df: pd.DataFrame,
    case_type: str,
    downstream_directory: str,
    output_param_name: str,
    r2_score: float,
    isSequential: bool = False,
    save_fig_name: Optional[str] = None,
) -> None:
    """Save scatter plots of predictions and observations with each case type (`tc` and `not_tc`)"""
    case_type = case_type.upper()
    if case_type not in ["TC", "NOT_TC"]:
        raise ValueError("Invalid case type. TC or NOT_TC")

    _title_tag = "(Sequential prediction)" if isSequential else ""
    _fig_name_tag = "Sequential_prediction_" if isSequential else ""

    r2_score = np.round(r2_score, 4)
    target_poteka_col = PPOTEKACols.get_col_from_weather_param(output_param_name)
    target_param_unit = PPOTEKACols.get_unit(target_poteka_col)
    target_param_min_val, target_param_max_val = MinMaxScalingValue.get_minmax_values_by_ppoteka_cols(target_poteka_col)
    text_position_x, text_position_y = (
        target_param_min_val + (target_param_max_val - target_param_min_val) / 2,
        target_param_max_val * 0.97,
    )

    plt.figure(figsize=(6, 6))
    ax = sns.scatterplot(data=result_df, x=target_poteka_col, y="Pred_Value", hue="date")
    ax.text(text_position_x, text_position_y, f"R2-Score: {r2_score}", size=15)
    # plot base line (cc = 1)
    x = np.linspace(
        target_param_min_val, target_param_max_val, int((target_param_max_val - target_param_min_val) // 10)
    )
    ax.plot(x, x, color="blue", linestyle="--")
    ax.set_xlim(target_param_min_val, target_param_max_val)
    ax.set_ylim(target_param_min_val, target_param_max_val)
    ax.set_title(f"{output_param_name} Scatter plot of tropical affected validation cases. {_title_tag}")
    ax.set_xlabel(f"Observation value {target_param_unit}")
    ax.set_ylabel(f"Prediction value {target_param_unit}")
    ax.legend(loc="lower left")
    plt.tight_layout()

    if save_fig_name is None:
        plt.savefig(os.path.join(downstream_directory, f"{_fig_name_tag}{case_type}_affected_cases.png"))
    else:
        plt.savefig(os.path.join(downstream_directory, save_fig_name))

    plt.close()
