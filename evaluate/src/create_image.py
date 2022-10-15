import os
from typing import Tuple
import logging

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from common.config import MinMaxScalingValue, PPOTEKACols
from common.interpolate_by_gpr import interpolate_by_gpr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_rain_image(
    scaled_rain_ndarray: np.ndarray,
    save_path: str,
    observation_point_file_path: str,
):
    try:
        import cartopy.crs as ccrs
        import cartopy.cfeature as cfeature
    except ModuleNotFoundError:
        logger.warning("Cartopy not found in the current env. Skip creating image with cartopy.")
        return None

    if scaled_rain_ndarray.ndim == 1:
        scaled_rain_ndarray = interpolate_by_gpr(scaled_rain_ndarray, observation_point_file_path)

    if scaled_rain_ndarray.ndim != 2:
        raise ValueError("Invalid ndarray shape for `scaled_rain_ndarray`. The shape should be (Height, Widht).")
    current_dir = os.getcwd()
    original_df = pd.read_csv(
        os.path.join(current_dir, "src/observation_point.csv"),
        index_col="Name",
    )

    grid_lon = np.round(np.linspace(120.90, 121.150, 50), decimals=3)
    grid_lat = np.round(np.linspace(14.350, 14.760, 50), decimals=3)
    xi, yi = np.meshgrid(grid_lon, grid_lat)
    plt.figure(figsize=(7, 8), dpi=80)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([120.90, 121.150, 14.350, 14.760])
    ax.add_feature(cfeature.COASTLINE)
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

    cs = ax.contourf(xi, np.flip(yi, axis=0), scaled_rain_ndarray, clevs, cmap=cmap, norm=norm)
    cbar = plt.colorbar(cs, orientation="vertical")
    cbar.set_label("millimeter")
    ax.scatter(original_df["LON"], original_df["LAT"], marker="D", color="dimgrey")

    plt.savefig(save_path)
    plt.close()


def get_r2score_text_position(max_val: float, min_val: float) -> Tuple[float, float]:
    x_pos = min_val + (max_val - min_val) * 0.03
    y_pos = max_val * 0.95
    return x_pos, y_pos


def all_cases_scatter_plot(result_df: pd.DataFrame, downstream_directory: str, output_param_name: str, r2_score: float):
    target_poteka_col = PPOTEKACols.get_col_from_weather_param(output_param_name)
    target_param_unit = PPOTEKACols.get_unit(target_poteka_col)
    target_param_min_val, target_param_max_val = MinMaxScalingValue.get_minmax_values_by_ppoteka_cols(target_poteka_col)
    text_position_x, text_position_y = get_r2score_text_position(max_val=target_param_max_val, min_val=target_param_min_val)
    # With TC, NOT TC hue.
    plt.figure(figsize=(6, 6))
    ax = sns.scatterplot(data=result_df, x=target_poteka_col, y="Pred_Value", hue="date")
    ax.text(text_position_x, text_position_y, f"R2-Score: {r2_score}", size=15)

    x = np.linspace(target_param_min_val, target_param_max_val, int((target_param_max_val - target_param_min_val) // 10))
    ax.plot(x, x, color="blue", linestyle="--")

    ax.set_xlim(target_param_min_val, target_param_max_val)
    ax.set_ylim(target_param_min_val, target_param_max_val)
    ax.set_title(f"{output_param_name} Scatter plot of all validation cases.")
    ax.set_xlabel(f"Observation value {target_param_unit}")
    ax.set_ylabel(f"Prediction value {target_param_unit}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(downstream_directory, "all_cases.png"))
    plt.close()


def date_scatter_plot(result_df: pd.DataFrame, date: str, downstream_directory: str, output_param_name: str, r2_score: float):
    """plot scatter plots of prediction vs obervation of a given date.

    Args:
        rmses_df (pd.DataFrame):
        downstream_directory (str):
        output_param_name (str): weather param name
        r2_score: r2_score of given datasets.
    """
    # Add date_time column
    result_df["date_time"] = result_df["date"] + "_" + result_df["predict_utc_time"] + "start"
    # create each sample scatter plot. hue is date_time.
    target_poteka_col = PPOTEKACols.get_col_from_weather_param(output_param_name)
    target_param_unit = PPOTEKACols.get_unit(target_poteka_col)
    target_param_min_val, target_param_max_val = MinMaxScalingValue.get_minmax_values_by_ppoteka_cols(target_poteka_col)
    text_position_x, text_position_y = get_r2score_text_position(max_val=target_param_max_val, min_val=target_param_min_val)

    plt.figure(figsize=(6, 6))

    ax = sns.scatterplot(data=result_df, x=target_poteka_col, y="Pred_Value", hue="date_time")
    # plot r2 score line.
    ax.text(text_position_x, text_position_y, f"R2-Score: {r2_score}", size=15)
    # plot base line (cc = 1)
    x = np.linspace(target_param_min_val, target_param_max_val, int((target_param_max_val - target_param_min_val) // 10))
    ax.plot(x, x, color="blue", linestyle="--")

    ax.set_xlim(target_param_min_val, target_param_max_val)
    ax.set_ylim(target_param_min_val, target_param_max_val)
    ax.set_title(f"{output_param_name} Scatter plot of {date} cases.")
    ax.set_xlabel(f"Observation value {target_param_unit}")
    ax.set_ylabel(f"Prediction value {target_param_unit}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(downstream_directory, f"{date}_cases.png"))
    plt.close()


def casetype_scatter_plot(
    result_df: pd.DataFrame, case_type: str, downstream_directory: str, output_param_name: str, r2_score: float, isSequential: bool = False
) -> None:
    case_type = case_type.upper()
    if case_type not in ["TC", "NOT_TC"]:
        raise ValueError("Invalid case type. TC or NOT_TC")

    _title_tag = "(Sequential prediction)" if isSequential else ""
    _fig_name_tag = "Sequential_prediction_" if isSequential else ""

    target_poteka_col = PPOTEKACols.get_col_from_weather_param(output_param_name)
    target_param_unit = PPOTEKACols.get_unit(target_poteka_col)
    target_param_min_val, target_param_max_val = MinMaxScalingValue.get_minmax_values_by_ppoteka_cols(target_poteka_col)
    text_position_x, text_position_y = target_param_min_val + (target_param_max_val - target_param_min_val) / 2, target_param_max_val * 0.97

    plt.figure(figsize=(6, 6))
    ax = sns.scatterplot(data=result_df, x=target_poteka_col, y="Pred_Value", hue="date")
    ax.text(text_position_x, text_position_y, f"R2-Score: {r2_score}", size=15)
    # plot base line (cc = 1)
    x = np.linspace(target_param_min_val, target_param_max_val, int((target_param_max_val - target_param_min_val) // 10))
    ax.plot(x, x, color="blue", linestyle="--")
    ax.set_xlim(target_param_min_val, target_param_max_val)
    ax.set_ylim(target_param_min_val, target_param_max_val)
    ax.set_title(f"{output_param_name} Scatter plot of tropical affected validation cases. {_title_tag}")
    ax.set_xlabel(f"Observation value {target_param_unit}")
    ax.set_ylabel(f"Prediction value {target_param_unit}")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(downstream_directory, f"{_fig_name_tag}{case_type}_affected_cases.png"))
    plt.close()
