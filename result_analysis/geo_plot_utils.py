import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from test_case_utils import WeatherParams, TestCase

###
# Utils for plotting
###
class TargetManilaErea:
    MAX_LONGITUDE = 121.150
    MIN_LONGITUDE = 120.90

    MAX_LATITUDE = 14.760
    MIN_LATITUDE = 14.350


def create_img_from_griddata(ax, grid_data, color_levels, color_map, contour: bool = False):
    grid_lon = np.round(np.linspace(TargetManilaErea.MIN_LONGITUDE, TargetManilaErea.MAX_LONGITUDE), decimals=3)
    grid_lat = np.round(np.linspace(TargetManilaErea.MIN_LATITUDE, TargetManilaErea.MAX_LATITUDE), decimals=3)

    # fig = plt.figure(figsize=(7, 8), dpi=80)
    # ax = fig.add_subplot(1, 1, 1)

    xi, yi = np.meshgrid(grid_lon, grid_lat)
    if contour:
        # Add contour lines
        c = ax.contour(xi, np.flip(yi), grid_data, colors=['black'], linestyles='dashed')

    # Add heat map
    cs = ax.contourf(
        xi, np.flip(yi), grid_data, color_levels, cmap=color_map, norm=mcolors.BoundaryNorm(color_levels, color_map.N)
    )
    return cs


def create_geo_plot(ax, target_param, grid_data, xlabel=None):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=15, labelpad=20)
        ax.set_xticks([]) # So that ax.set_xlabel works
    ax.set_extent(
        [
            TargetManilaErea.MIN_LONGITUDE,
            TargetManilaErea.MAX_LONGITUDE,
            TargetManilaErea.MIN_LATITUDE,
            TargetManilaErea.MAX_LATITUDE,
        ],
        crs=ccrs.PlateCarree(),
    )
    gl = ax.gridlines(draw_labels=True, alpha=0)
    gl.right_labels = False
    gl.top_labels = False
    gl.ylabel_style = {'rotation': 45, 'fontsize': 18}
    gl.xlabel_style = {'fontsize': 18}

    cs = create_img_from_griddata(
        ax,
        grid_data,
        WeatherParams.get_clevels(target_param) if target_param != WeatherParams.attention_map else WeatherParams.get_clevels(target_param, grid_data.max()),
        WeatherParams.get_cmap(target_param),
        contour=False if target_param in [WeatherParams.rainfall, WeatherParams.attention_map] else True
    )

    if target_param == WeatherParams.attention_map:
        x_center = (
        TargetManilaErea.MIN_LONGITUDE + (TargetManilaErea.MAX_LONGITUDE - TargetManilaErea.MIN_LONGITUDE) / 2
        )
        y_center = TargetManilaErea.MIN_LATITUDE + (TargetManilaErea.MAX_LATITUDE - TargetManilaErea.MIN_LATITUDE) / 2
        ax.plot(x_center, y_center, color="black", marker="+", markersize=12)
    ax.add_feature(cfeature.COASTLINE)
    cbar = plt.colorbar(cs, ax=ax)
    cbar.ax.tick_params(labelsize=18)
    if target_param == WeatherParams.attention_map:
        cbar.ax.ticklabel_format(style="sci", scilimits=(-3, -3), useMathText=True)
        cbar.ax.yaxis.get_offset_text().set_fontsize(18)
    cbar.set_label(WeatherParams.unit(target_param), fontsize=20)


def save_geo_plots(
    test_case: TestCase,
    target_params=None,
    target_types: str = ['input', 'label', 'predict', 'attention_map'],
    model_prefix: str = 'rainonly',
):
    for target_type in target_types:
        if target_type not in ['input', 'label', 'predict', 'attention_map']:
            raise ValueError(f'target_type must be in [input, label] instaed of {target_type}.')

    if target_params == None:
        target_params = [t for t in WeatherParams._member_names_ if t != WeatherParams.attention_map]

    if isinstance(target_params, str):
        target_params = [target_params]

    def save_plots(data: list[np.ndarray], target_param: str, save_fig_prefix: str):
        for idx, grid_data in enumerate(data):
            fig = plt.figure(figsize=(7, 8), dpi=80)
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            create_geo_plot(ax, target_param, grid_data)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'{save_fig_prefix}-{idx}.png')
            # plt.show()
            plt.close()


    root_dir_path = 'geo_plots'
    # plot_data = {'input': {'rainfall': [np.ndarray, ...], 'temperature': ...}, 'label': {...}, 'predict': {...}]
    plot_data = get_data_for_plots(test_case, target_types, target_params)
    for target_type in target_types:
        save_dir_path = os.path.join(root_dir_path, test_case.test_case_name, target_type)
        if target_type in ['predict', 'attention_map']:
            save_dir_path = os.path.join(save_dir_path, model_prefix)
        os.makedirs(save_dir_path, exist_ok=True)
        if target_type in ['input', 'label']:
            for target_param in target_params:
                data = plot_data[target_type][target_param]
                save_plots(data, target_param, save_fig_prefix=os.path.join(save_dir_path, target_param))
        elif target_type == 'predict':
            data = plot_data[target_type]
            save_plots(data, WeatherParams.rainfall, save_fig_prefix=os.path.join(save_dir_path, WeatherParams.rainfall))
        else:
            for layer_num, data in plot_data[target_type].items():
                save_plots(data, WeatherParams.attention_map, save_fig_prefix=os.path.join(save_dir_path, f'layer{layer_num}'))




def get_data_for_plots(test_case: TestCase, target_types: list[str], target_params: list[str]):
    results = {}
    data_paths = test_case.data_pathes
    for target_type in target_types:
        results[target_type] = {}
        if target_type in ['input', 'label']:
            for target_param in target_params:
                seq_length = test_case.input_seq_length if target_type == 'input' else test_case.label_seq_length
                if target_param == WeatherParams.rainfall:
                    data = [
                        pd.read_csv(data_paths[target_param][target_type][col], index_col=0).to_numpy() for col in range(seq_length)
                    ]
                else:
                    data = [
                        interpolate(pd.read_csv(data_paths['one_day_data'][target_type][idx]).rename(columns={'Unnamed: 0': 'Station_Name'}), param_name=target_param)
                        for idx in range(seq_length)
                    ]
                results[target_type][target_param] = data
        elif target_type == 'predict':
            results[target_type] = [pd.read_parquet(data_paths[WeatherParams.rainfall]['predict'][col]).to_numpy() for col in range(test_case.label_seq_length)]
        else:
            results[target_type] = {}
            target_layer_nums = [1, 4]
            for layer_num in target_layer_nums:
                results[target_type][layer_num] = [
                    np.load(f) for f in test_case.data_pathes['attention_maps'][f'layer{layer_num}']
                ]
    return results
                

def timeseries_geo_plot(
    subfig,
    suptitle: str,
    target_param: str, 
    data: list,
    xlabels = None
):
    subfig.suptitle(suptitle, fontsize=20)
    axs = subfig.subplots(nrows=1, ncols=len(data), subplot_kw=dict(projection=ccrs.PlateCarree()))
    for col, ax in enumerate(axs):
        # Add time step
        if xlabels:
            ax.set_xlabel(xlabels[col], fontsize=15, labelpad=20)
            ax.set_xticks([]) # So that ax.set_xlabel works
        ax.set_extent(
            [
                TargetManilaErea.MIN_LONGITUDE,
                TargetManilaErea.MAX_LONGITUDE,
                TargetManilaErea.MIN_LATITUDE,
                TargetManilaErea.MAX_LATITUDE,
            ],
            crs=ccrs.PlateCarree(),
        )
        gl = ax.gridlines(draw_labels=True, alpha=0)
        gl.right_labels = False
        gl.top_labels = False

        cs = create_img_from_griddata(
            ax,
            data[col],
            WeatherParams.get_clevels(target_param) if target_param != WeatherParams.attention_map else WeatherParams.get_clevels(target_param, data[col].max()),
            WeatherParams.get_cmap(target_param),
            contour=False if target_param in [WeatherParams.rainfall, WeatherParams.attention_map] else True
        )

        if target_param == WeatherParams.attention_map:
            x_center = (
            TargetManilaErea.MIN_LONGITUDE + (TargetManilaErea.MAX_LONGITUDE - TargetManilaErea.MIN_LONGITUDE) / 2
            )
            y_center = TargetManilaErea.MIN_LATITUDE + (TargetManilaErea.MAX_LATITUDE - TargetManilaErea.MIN_LATITUDE) / 2
            ax.plot(x_center, y_center, color="black", marker="+", markersize=12)
        ax.add_feature(cfeature.COASTLINE)

    cbar = plt.colorbar(cs, ax=axs)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(WeatherParams.unit(target_param), fontsize=20)

def xlabels_from_times(times: list[str]):
    times = [s.replace('-', ':') for s in times]
    times = [t.replace(':0', ':00') if t.endswith(':0') else t for t in times]
    return [f'{t} UTC' for t in times]


###
# Data Cleaning
###
def get_zero_diff_count(series: pd.Series):
    return (series == 0).sum()

def check_abnormal_datapoint(data_df):
    y_labels = {
        "hour-rain": "Hourly rainfall (mm/h)",
        "AT1": "Temperature (℃)",
        "RH1": "Relative Humidity (%)",
        "WS1": "Wind Speed (m/s)",
        "V-Wind": "North-south wind speed(m/s)",
        "U-Wind": "East-west wind speed (m/s)",
        "PRS": "Station Pressure (hPa)",
    }
    for col in ['AT1', 'RH1']:
        if col in list(y_labels.keys()):
            # Check outliers
            ## 1. same value is continuing (except for hour-rain)
            grouped_by_station = data_df.groupby(by=["Station_Name"])
            if col != "hour-rain":
                check_same_value_df = data_df.sort_values(by=["Station_Name", "time_step"])
                check_same_value_df["diff"] = check_same_value_df.groupby(by=["Station_Name"])[col].diff()
                zero_diff_count_series = check_same_value_df.groupby(by=["Station_Name"])["diff"].agg(get_zero_diff_count)
                for observation_name in zero_diff_count_series.index:
                    zero_diff_count = zero_diff_count_series[observation_name]
                    print(zero_diff_count)
                    if zero_diff_count > 50:
                        print(f"{col} data of {observation_name} has {zero_diff_count} zero different values.")

###
# Interpolate with exclude obpoints
# Poteka has abnormal observation values such as same value everytime.
###
from scipy.interpolate import RBFInterpolator
import sys
sys.path.append('../')
from common.config import PPOTEKACols, MinMaxScalingValue
from evaluate.src.interpolator.humidity_interpolator import HumidityInterplator
from evaluate.src.interpolator.temperature_interpolator import TemperatureInterpolator

def interpolate(df: pd.DataFrame, param_name: str, exclude_obpoints: list = ['Vlz-Bagbaguin_00174731', 'MM-EFCOS_00173456', 'CentPark_00181288']):
    WeatherParams.valid(param_name)
    if param_name == WeatherParams.rainfall:
        param_name = 'rain'
    df = df.loc[~df['Station_Name'].isin(exclude_obpoints)]
    obpoint_lons, obpoint_lats = df['LON'].tolist(), df['LAT'].tolist()
    target_col = PPOTEKACols.get_col_from_weather_param(param_name)

    min_val, max_val = MinMaxScalingValue.get_minmax_values_by_ppoteka_cols(target_col)
    data = (df[target_col].to_numpy() - min_val) / (max_val - min_val)
    rbfi = RBFInterpolator(
        y=np.column_stack([obpoint_lons, obpoint_lats]), d=data, kernel="linear", epsilon=10
    )
    grid_coordinate = np.mgrid[120.90:121.150:50j, 14.350:14.760:50j]

    y_pred = rbfi(grid_coordinate.reshape(2, -1).T)
    grid_data = np.reshape(y_pred, (50, 50)).T
    grid_data = np.flipud(grid_data)

    grid_data = np.where(grid_data > 0, grid_data, 0)
    grid_data = np.where(grid_data > 1, 1, grid_data)

    return grid_data.astype(np.float32) * (max_val - min_val) + min_val