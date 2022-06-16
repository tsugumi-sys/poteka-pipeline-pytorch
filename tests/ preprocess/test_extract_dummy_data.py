import sys
from typing import Dict
import unittest
from unittest.mock import MagicMock, patch

from preprocess.src.extract_dummy_data import get_dummy_data_files, save_dummy_data, generate_dummy_data, get_meta_test_info


class TestExtractDummyData(unittest.TestCase):
    @patch("shutil.rmtree")
    @patch("preprocess.src.extract_dummy_data.save_dummy_data")
    def test_get_dummy_data_files(self, mock_save_dummy_data: MagicMock, mock_shutil_rmtree: MagicMock):
        mock_save_dummy_data.return_value = {"test": 1234}
        with self.assertRaises(ValueError):
            _ = get_dummy_data_files(["dummy_param"], time_step_minutes=10, downstream_dir_path="./data/preprocess", dataset_length=1)

        with self.assertRaises(ValueError):
            _ = get_dummy_data_files(["rain", "dummy_param"], time_step_minutes=10, downstream_dir_path="./data/preprocess", dataset_length=1)

        dummy_data_files = get_dummy_data_files(["rain"], 10, "./data/preprocess", 1)
        self.assertEqual(len(dummy_data_files), 1)
        self.assertTrue(isinstance(dummy_data_files[0], Dict))
        self.assertEqual(mock_save_dummy_data.call_count, 1)
        self.assertEqual(mock_shutil_rmtree.call_count, 1)
