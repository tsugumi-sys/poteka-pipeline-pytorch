import sys
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from common.config import GridSize

sys.path.append(".")
from common.utils import get_ob_point_values_from_tensor
from evaluate.src.geoimg_generator.geoimg_generator_interface import GeoimgGeneratorInterface
from evaluate.src.geoimg_generator.utils import obpoint_grid_handler, save_img_from_griddata, ob_point_df_from_ndarray
from common.config import MinMaxScalingValue, TargetManilaErea

logger = logging.getLogger(__name__)


class AttentionMapImgGenerator(GeoimgGeneratorInterface):
    def __init__(self) -> None:
        self.weather_param_name = "attention_map"
        self.color_map = cm.bwr

        self.weather_param_unit_label = ""
        super().__init__()

    def gen_img(self, scaled_ndarray: np.ndarray, observation_point_file_path: str, save_img_path: str) -> None:
        if scaled_ndarray.shape != (GridSize.HEIGHT, GridSize.WIDTH):
            raise ValueError("AttentionMapImgGenerator only allowed to give (HEIGHT, WIDTH) shape tensor.")

        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
        except ModuleNotFoundError:
            logger.warning("Cartopy not found in the current env. Skip creating geo image.")
            return None

        self.color_levels = np.linspace(0, scaled_ndarray.max(), 50)
        grid_lon = np.round(np.linspace(TargetManilaErea.MIN_LONGITUDE, TargetManilaErea.MAX_LONGITUDE), decimals=3)
        grid_lat = np.round(np.linspace(TargetManilaErea.MIN_LATITUDE, TargetManilaErea.MAX_LATITUDE), decimals=3)

        xi, yi = np.meshgrid(grid_lon, grid_lat)

        fig = plt.figure(figsize=(7, 8), dpi=80)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent(
            [TargetManilaErea.MIN_LONGITUDE, TargetManilaErea.MAX_LONGITUDE, TargetManilaErea.MIN_LATITUDE, TargetManilaErea.MAX_LATITUDE],
            crs=ccrs.PlateCarree(),
        )

        gl = ax.gridlines(draw_labels=True, alpha=0)
        gl.right_labels = False
        gl.top_labels = False

        cs = ax.contourf(
            xi, np.flip(yi), scaled_ndarray, self.color_levels, cmap=self.color_map, norm=mcolors.BoundaryNorm(self.color_levels, self.color_map.N)
        )

        color_bar = plt.colorbar(cs, orientation="vertical")

        x_center = TargetManilaErea.MIN_LONGITUDE + (TargetManilaErea.MAX_LONGITUDE - TargetManilaErea.MIN_LONGITUDE) / 2
        y_center = TargetManilaErea.MIN_LATITUDE + (TargetManilaErea.MAX_LATITUDE - TargetManilaErea.MIN_LATITUDE) / 2
        ax.plot(x_center, y_center, color="black", marker="+", markersize=12)
        ax.add_feature(cfeature.COASTLINE)

        plt.savefig(save_img_path)
        plt.close()
