import unittest
from typing import List
import json

from datetime import datetime, timedelta
import numpy as np
from pandas.core.frame import itertools
import torch
from common.config import GridSize

from common.utils import datetime_range, convert_two_digit_date, min_max_scaler, timestep_csv_names, get_ob_point_values_from_tensor


class TestUtils(unittest.TestCase):
    def test_datetime_range(self):
        result = list(datetime_range(start=datetime(2020, 1, 1, 0, 0, 0), end=datetime(2020, 1, 1, 1, 0, 0), delta=timedelta(minutes=10)))
        expected_result = [datetime(2020, 1, 1, 0, 0, 0) + timedelta(minutes=m) for m in range(0, 70, 10)]
        self.assertEqual(result, expected_result)

    def test_makedates(self):
        str_date: str = convert_two_digit_date("10")
        self.assertEqual(str_date, "10")

        str_date: str = convert_two_digit_date("1")
        self.assertEqual(str_date, "01")

    def test_timestep_csv_names(self):
        csv_filenames: List = timestep_csv_names(2020, 1, 1, 60)
        self.assertEqual(len(csv_filenames), 24)

        for filename in csv_filenames:
            with self.subTest(filename=filename):
                self.assertRegex(csv_filenames[0], r".+\.csv$")

    def test_min_max_scaler(self):
        sample_arr = np.asarray([0, 1, 2, 3, 10])
        scaled_arr = min_max_scaler(min_value=0.0, max_value=10.0, arr=sample_arr)
        self.assertIsInstance(scaled_arr, np.ndarray)
        self.assertEqual(scaled_arr.min(), 0.0)
        self.assertEqual(scaled_arr.max(), 1.0)

    def test_get_ob_point_values_from_tensor(self):
        # Generate dummy tensor
        with open("./common/meta-data/observation_point.json", "r") as f:
            ob_points_data = json.load(f)
        grid_lons = np.linspace(120.90, 121.150, GridSize.WIDTH)
        grid_lats = np.linspace(14.350, 14.760, GridSize.HEIGHT)
        ob_point_lons = [item["longitude"] for item in ob_points_data.values()]
        ob_point_lats = [item["latitude"] for item in ob_points_data.values()][::-1]
        tensor = torch.zeros((GridSize.HEIGHT, GridSize.WIDTH))
        for ob_point_idx, (lon, lat) in enumerate(zip(ob_point_lons, ob_point_lats)):
            target_lon_idx, target_lat_idx = 0, 0
            for before_lon, after_lon in zip(grid_lons[:-1], grid_lons[1:]):
                if before_lon < lon and lon < after_lon:
                    target_lon_idx = np.where(grid_lons == before_lon)[0][0]
                    break
            for before_lat, after_lat in zip(grid_lats[:-1], grid_lats[1:]):
                if before_lat < lat and lat < after_lat:
                    target_lat_idx = np.where(grid_lats == before_lat)[0][0]
                    break
            for tensor_lon, tensor_lat in itertools.product([target_lon_idx-1, target_lon_idx, target_lon_idx+1], [target_lat_idx-1, target_lat_idx, target_lat_idx+1]):
                tensor[tensor_lat, tensor_lon] = ob_point_idx
        # Test
        result = get_ob_point_values_from_tensor(tensor)
        for idx in range(len(ob_point_lons)):
            self.assertEqual(result[idx, 0], idx)

if __name__ == "__main__":
    unittest.main()