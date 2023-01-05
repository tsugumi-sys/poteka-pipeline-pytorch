import json
import unittest
from datetime import datetime, timedelta
from typing import List

import numpy as np
import torch

from common.config import GridSize
from common.utils import (
    convert_two_digit_date,
    datetime_range,
    get_ob_point_values_from_tensor,
    min_max_scaler,
    timestep_csv_names,
)


class TestUtils(unittest.TestCase):
    def test_datetime_range(self):
        result = list(
            datetime_range(
                start=datetime(2020, 1, 1, 0, 0, 0), end=datetime(2020, 1, 1, 1, 0, 0), delta=timedelta(minutes=10)
            )
        )
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
        grid_lats = np.linspace(14.350, 14.760, GridSize.HEIGHT)[
            ::-1
        ]  # Flip the grid latitudes because the latitudes are in descending order.
        ob_point_lons = [item["longitude"] for item in ob_points_data.values()]
        ob_point_lats = [item["latitude"] for item in ob_points_data.values()]
        tensor = torch.rand((GridSize.HEIGHT, GridSize.WIDTH))
        true_result = torch.zeros((len(ob_point_lons)))
        for ob_point_idx, (lon, lat) in enumerate(zip(ob_point_lons, ob_point_lats)):
            target_lon_idx, target_lat_idx = 0, 0
            for before_lon, after_lon in zip(grid_lons[:-1], grid_lons[1:]):
                if before_lon < lon and lon < after_lon:
                    target_lon_idx = np.where(grid_lons == before_lon)[0][0]
                    break
            for after_lat, before_lat in zip(
                grid_lats[:-1], grid_lats[1:]
            ):  # NOTE: `grid_lats` are in the descending order.
                if before_lat < lat and lat < after_lat:
                    target_lat_idx = np.where(grid_lats == before_lat)[0][0]
                    break
            true_result[ob_point_idx] = (
                tensor[target_lat_idx - 1 : target_lat_idx + 2, target_lon_idx - 1 : target_lon_idx + 2].mean().item()
            )
        # Test
        result = get_ob_point_values_from_tensor(
            observation_point_file_path="./common/meta-data/observation_point.json", tensor=tensor
        )
        self.assertTrue(torch.equal(true_result, result))


if __name__ == "__main__":
    unittest.main()
