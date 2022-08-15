import os
from typing import Dict
import logging

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from common.config import MinMaxScalingValue, PPOTEKACols
from common.interpolate_by_gpr import interpolate_by_gpr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# [NOTE]: Cartopy can be used only local conda environment.
# Avoid ModuleNotFoundError when run test or poetry environments.
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ModuleNotFoundError:
    logger.warning("Cartopy not found in the current env.")


def save_rain_image(
    scaled_rain_ndarray: np.ndarray,
    save_path: str,
):
    logger.warning("Skip create and save rain image.")
    if scaled_rain_ndarray.ndim == 1:
        scaled_rain_ndarray = interpolate_by_gpr(scaled_rain_ndarray)

    if scaled_rain_ndarray.ndim != 2:
        raise ValueError("Invalid ndarray shape for `scaled_rain_ndarray`. The shape should be (Height, Widht).")
    # current_dir = os.getcwd()
    # original_df = pd.read_csv(
    #    os.path.join(current_dir, "src/observation_point.csv"),
    #    index_col="Name",
    # )

    # grid_lon = np.round(np.linspace(120.90, 121.150, 50), decimals=3)
    # grid_lat = np.round(np.linspace(14.350, 14.760, 50), decimals=3)
    # xi, yi = np.meshgrid(grid_lon, grid_lat)
    # plt.figure(figsize=(7, 8), dpi=80)
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.set_extent([120.90, 121.150, 14.350, 14.760])
    # ax.add_feature(cfeature.COASTLINE)
    # gl = ax.gridlines(draw_labels=True, alpha=0)
    # gl.right_labels = False
    # gl.top_labels = False

    # clevs = [0, 5, 7.5, 10, 15, 20, 30, 40, 50, 70, 100]
    # cmap_data = [
    #    (1.0, 1.0, 1.0),
    #    (0.3137255012989044, 0.8156862854957581, 0.8156862854957581),
    #    (0.0, 1.0, 1.0),
    #    (0.0, 0.8784313797950745, 0.501960813999176),
    #    (0.0, 0.7529411911964417, 0.0),
    #    (0.501960813999176, 0.8784313797950745, 0.0),
    #    (1.0, 1.0, 0.0),
    #    (1.0, 0.627451002597808, 0.0),
    #    (1.0, 0.0, 0.0),
    #    (1.0, 0.125490203499794, 0.501960813999176),
    #    (0.9411764740943909, 0.250980406999588, 1.0),
    #    (0.501960813999176, 0.125490203499794, 1.0),
    # ]
    # cmap = mcolors.ListedColormap(cmap_data, "precipitation")
    # norm = mcolors.BoundaryNorm(clevs, cmap.N)

    # cs = ax.contourf(xi, np.flip(yi, axis=0), scaled_rain_ndarray, clevs, cmap=cmap, norm=norm)
    # cbar = plt.colorbar(cs, orientation="vertical")
    # cbar.set_label("millimeter")
    # ax.scatter(original_df["LON"], original_df["LAT"], marker="D", color="dimgrey")

    # plt.savefig(save_path)
    # plt.close()


def all_cases_plot(rmses_df: pd.DataFrame, downstream_directory: str, output_param_name: str, result_metrics: Dict, isSequential: bool = False):
    # Create scatter of all data. hue is case_type (TC or NOT_TC).
    data = rmses_df.loc[rmses_df["isSequential"] == isSequential]
    _title_tag = "(Sequential prediction)" if isSequential else ""
    _fig_name_tag = "Sequential_prediction_" if isSequential else ""
    target_poteka_col = PPOTEKACols.get_col_from_weather_param(output_param_name)
    target_param_unit = PPOTEKACols.get_unit(target_poteka_col)
    target_param_min_val, target_param_max_val = MinMaxScalingValue.get_minmax_values_by_ppoteka_cols(target_poteka_col)
    text_position_x, text_position_y = target_param_min_val + (target_param_max_val + target_param_min_val) / 2, target_param_max_val * 0.97
    # With TC, NOT TC hue.
    plt.figure(figsize=(6, 6))
    ax = sns.scatterplot(data=data, x=target_poteka_col, y="Pred_Value", hue="case_type")
    # caluculate r2 score
    r2 = r2_score(data[target_poteka_col].astype(float).values, data["Pred_Value"].astype(float).values)
    r2 = np.round(r2, decimals=3)
    result_metrics[f"r2_{_fig_name_tag}all_validation_cases"] = r2
    ax.text(text_position_x, text_position_y, f"R2-Score: {r2}", size=15)

    x = np.linspace(target_param_min_val, target_param_max_val, (target_param_max_val - target_param_min_val) // 10)
    ax.plot(x, x, color="blue", linestyle="--")

    ax.set_xlim(target_param_min_val, target_param_max_val)
    ax.set_ylim(target_param_min_val, target_param_max_val)
    ax.set_title(f"{output_param_name} Scatter plot of all validation cases. {_title_tag}")
    ax.set_xlabel(f"Observation value {target_param_unit}")
    ax.set_ylabel(f"Prediction value {target_param_unit}")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(downstream_directory, f"{_fig_name_tag}all_validation_cases.png"))
    plt.close()

    # Without TC, NOT TC hue.
    plt.figure(figsize=(6, 6))
    ax = sns.scatterplot(data=data, x="hour-rain", y="Pred_Value", hue="case_type")
    # calculate r2 score
    r2 = r2_score(data["hour-rain"].astype(float).values, data["Pred_Value"].astype(float).values)
    r2 = np.round(r2, decimals=3)
    ax.text(text_position_x, text_position_y, f"R2-Score: {r2}", size=15)
    x = np.linspace(target_param_min_val, target_param_max_val, (target_param_max_val - target_param_min_val) // 10)
    ax.plot(x, x, color="blue", linestyle="--")
    ax.set_xlim(target_param_min_val, target_param_max_val)
    ax.set_ylim(target_param_min_val, target_param_max_val)
    ax.set_title(f"{output_param_name} Scatter plot of all validation cases. {_title_tag}")
    ax.set_xlabel(f"Observation value {target_param_unit}")
    ax.set_ylabel(f"Prediction value {target_param_unit}")
    plt.tight_layout()
    plt.savefig(os.path.join(downstream_directory, f"{_fig_name_tag}all_validation_cases.png"))
    plt.close()


def sample_plot(rmses_df: pd.DataFrame, downstream_directory: str, result_metrics: Dict, output_param_name: str, isSequential: bool = False):
    """plot scatter plots of prediction vs obervation

    Args:
        rmses_df (pd.DataFrame):
        downstream_directory (str):
        result_metrics (Dict): dictionary to save retults
        output_param_name (str): weather param name
        isSequential (bool, optional): [description]. Defaults to False.
    """
    data = rmses_df.loc[rmses_df["isSequential"] == isSequential]
    _title_tag = "(Sequential prediction)" if isSequential else ""
    _fig_name_tag = "Sequential_prediction_" if isSequential else ""

    # create each sample scatter plot. hue is date_time.
    sample_dates = data["date"].unique().tolist()
    target_poteka_col = PPOTEKACols.get_col_from_weather_param(output_param_name)
    target_param_unit = PPOTEKACols.get_unit(target_poteka_col)
    target_param_min_val, target_param_max_val = MinMaxScalingValue.get_minmax_values_by_ppoteka_cols(target_poteka_col)
    text_position_x, text_position_y = target_param_min_val + (target_param_max_val + target_param_min_val) / 2, target_param_max_val * 0.97
    for sample_date in sample_dates:
        query = [sample_date in e for e in data["date"]]
        _rmses_each_sample = data.loc[query]

        plt.figure(figsize=(6, 6))

        ax = sns.scatterplot(data=_rmses_each_sample, x=target_poteka_col, y="Pred_Value", hue="date_time")
        # plot r2 score line.
        r2 = r2_score(_rmses_each_sample[target_poteka_col].astype(float).values, _rmses_each_sample["Pred_Value"].astype(float).values)
        r2 = np.round(r2, decimals=3)
        result_metrics[f"r2_{_fig_name_tag}{sample_date}_cases"] = r2
        ax.text(text_position_x, text_position_y, f"R2-Score: {r2}", size=15)
        # plot base line (cc = 1)
        x = np.linspace(target_param_min_val, target_param_max_val, (target_param_max_val - target_param_min_val) // 10)
        ax.plot(x, x, color="blue", linestyle="--")

        ax.set_xlim(target_param_min_val, target_param_max_val)
        ax.set_ylim(target_param_min_val, target_param_max_val)
        ax.set_title(f"{output_param_name} Scatter plot of {sample_date} cases. {_title_tag}")
        ax.set_xlabel(f"Observation value {target_param_unit}")
        ax.set_ylabel(f"Prediction value {target_param_unit}")
        ax.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(downstream_directory, f"{_fig_name_tag}{sample_date}_cases.png"))
        plt.close()


def casetype_plot(casetype: str, rmses_df: pd.DataFrame, downstream_directory: str, output_param_name: str, result_metrics: Dict, isSequential: bool = False):
    data = rmses_df.loc[rmses_df["isSequential"] == isSequential]
    _title_tag = "(Sequential prediction)" if isSequential else ""
    _fig_name_tag = "Sequential_prediction_" if isSequential else ""
    casetype = casetype.upper()
    if casetype not in ["TC", "NOT_TC"]:
        raise ValueError("Invalid case type. TC or NOT_TC")
    query = [e == casetype for e in data["case_type"]]
    _rmses_tc_cases = data.loc[query]
    # get PPOTKA cols
    target_poteka_col = PPOTEKACols.get_col_from_weather_param(output_param_name)
    target_param_unit = PPOTEKACols.get_unit(target_poteka_col)
    target_param_min_val, target_param_max_val = MinMaxScalingValue.get_minmax_values_by_ppoteka_cols(target_poteka_col)
    text_position_x, text_position_y = target_param_min_val + (target_param_max_val + target_param_min_val) / 2, target_param_max_val * 0.97
    # Create figure
    plt.figure(figsize=(6, 6))
    ax = sns.scatterplot(data=_rmses_tc_cases, x=target_poteka_col, y="Pred_Value", hue="date")
    # plot r2 score line.
    r2 = r2_score(_rmses_tc_cases[target_poteka_col].astype(float).values, _rmses_tc_cases["Pred_Value"].astype(float).values)
    r2 = np.round(r2, decimals=3)
    result_metrics[f"r2_{_fig_name_tag}{casetype}_affected_cases"] = r2
    ax.text(text_position_x, text_position_y, f"R2-Score: {r2}", size=15)
    # plot base line (cc = 1)
    x = np.linspace(target_param_min_val, target_param_max_val, (target_param_max_val - target_param_min_val) // 10)
    ax.plot(x, x, color="blue", linestyle="--")
    ax.set_xlim(target_param_min_val, target_param_max_val)
    ax.set_ylim(target_param_min_val, target_param_max_val)
    ax.set_title(f"{output_param_name} Scatter plot of tropical affected validation cases. {_title_tag}")
    ax.set_xlabel(f"Observation value {target_param_unit}")
    ax.set_ylabel(f"Prediction value {target_param_unit}")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(downstream_directory, f"{_fig_name_tag}{casetype}_affected_cases.png"))
    plt.close()
