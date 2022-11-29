import os
import unittest
import shutil

from omegaconf import OmegaConf, errors

from common.omegaconf_manager import OmegaconfManager


class TestOmegaconfManager(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        self.tmp_dir = "./data"
        super().__init__(methodName)

    def setUp(self) -> None:
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
        os.makedirs(self.tmp_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save(self):
        manager = OmegaconfManager()
        file_path = os.path.join(self.tmp_dir, "sample.yaml")
        cfg = OmegaConf.create({"key1": 1, "key2": 2})
        manager.save(cfg, file_path, ["key2"])

        self.assertTrue(os.path.exists(file_path))

        saved_cfg = OmegaConf.load(file_path)
        expect_cfg = OmegaConf.create({"key1": 1})
        self.assertEqual(saved_cfg, expect_cfg)

    def test_load(self):
        file_path = os.path.join(self.tmp_dir, "sample.yaml")
        expect_cfg = OmegaConf.create({"key": 1})
        OmegaConf.save(expect_cfg, file_path)

        manager = OmegaconfManager()
        cfg = manager.load(file_path)

        self.assertEqual(cfg, expect_cfg)

    def test_pop(self):
        cfg = OmegaConf.create({"key1": 1, "key2": 2})
        expect_cfg = OmegaConf.create({"key1": 1})

        manager = OmegaconfManager()
        poped_cfg = manager.pop(cfg, "key2")
        self.assertTrue(poped_cfg, expect_cfg)

        with self.assertRaises(ValueError):
            _ = manager.pop(cfg, 1)  # type: ignore

        with self.assertRaises(errors.ConfigKeyError):
            _ = manager.pop(cfg, "unkownKey")

    def test_update(self):
        cfg = OmegaConf.create({"a": "b", "c": {"d": "e"}, "e": {"f": {"g": "h"}}})

        manager = OmegaconfManager()
        updated_cfg = manager.update(cfg, {"a": 123, "c": {"d": 456}})
        expect_cfg = OmegaConf.create({"a": 123, "c": {"d": 456}, "e": {"f": {"g": "h"}}})
        self.assertEqual(updated_cfg, expect_cfg)

        updated_cfg = manager.update(cfg, {"c.d": 789, "e.f.g": 101})
        expect_cfg = OmegaConf.create({"a": 123, "c": {"d": 789}, "e": {"f": {"g": 101}}})
        self.assertEqual(updated_cfg, expect_cfg)

        updated_cfg = manager.update(cfg, {"c": {"d": 102}, "e.f.g": 103})
        expect_cfg = OmegaConf.create({"a": 123, "c": {"d": 102}, "e": {"f": {"g": 103}}})
        self.assertEqual(updated_cfg, expect_cfg)

    def test_convert_nested_dict_to_single_depth(self):
        dic = {"a": "b", "c": {"d": "e"}, "f": {"g": {"h": "i"}, "j": "k"}}
        expect_dic = {"a": "b", "c.d": "e", "f.g.h": "i", "f.j": "k"}
        manager = OmegaconfManager()
        single_depth_dic = manager._convert_nested_dict_to_single_depth(dic)

        self.assertEqual(single_depth_dic, expect_dic)
