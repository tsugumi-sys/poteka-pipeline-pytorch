import json
import unittest

import numpy as np

from common.config import GridSize, TargetManilaErea
from evaluate.src.interpolator.pressure_interpolator import PressureInterpolator


class TestPressureInterpolator(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

        self.interpolator = PressureInterpolator()

    def test_interpolate(self):
        """
        TODO: test with input tensor like one point that is located near the edge.
            And test the interpolate results checking the maximum value of one of the
            four quandrants and the maximum and minimum value of other three.
        """
        ob_point_file_path = "./common/meta-data/observation_point.json"
        with open(ob_point_file_path, "r") as f:
            ob_point_data = json.load(f)

        lons_lats = [(item["longitude"], item["latitude"]) for item in ob_point_data.values()]
        target_point = lons_lats[0]

        ob_point_ndarray = np.asarray([0] * len(lons_lats))
        ob_point_ndarray[0] = 1

        grid_array = self.interpolator.interpolate(ob_point_ndarray, ob_point_file_path)
        grid_lons = np.linspace(TargetManilaErea.MIN_LONGITUDE, TargetManilaErea.MAX_LONGITUDE, GridSize.WIDTH)
        grid_lats = np.linspace(TargetManilaErea.MIN_LATITUDE, TargetManilaErea.MAX_LATITUDE, GridSize.HEIGHT)

        target_point_lon_idx, target_point_lat_idx = 0, 0
        for idx, lon in enumerate(grid_lons):
            if lon > target_point[0]:
                target_point_lon_idx = idx
                break

        for idx, lat in enumerate(grid_lats):
            if lat > target_point[1]:
                target_point_lat_idx = GridSize.HEIGHT - idx
                break

        # NOTE: numpy ndarray indexing is array[y, x] if array is two-dimentional
        grid_data_max_idxs = np.unravel_index(np.argmax(grid_array, axis=None), grid_array.shape)

        self.assertTrue((target_point_lat_idx, target_point_lon_idx) == grid_data_max_idxs)
