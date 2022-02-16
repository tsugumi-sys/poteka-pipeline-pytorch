import unittest
from typing import List
import sys
from datetime import datetime, timedelta
import numpy as np

sys.path.append(".")
from common.utils import datetime_range, convert_two_digit_date, min_max_scaler, timestep_csv_names


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


if __name__ == "__main__":
    unittest.main()
