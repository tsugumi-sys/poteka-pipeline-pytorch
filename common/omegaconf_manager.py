from typing import Union, Optional, List, Dict, Any, Tuple
import warnings

from omegaconf import ListConfig, OmegaConf, DictConfig, open_dict


class OmegaconfManager:
    def __init__(self):
        super().__init__()

    def save(self, cfg: DictConfig, save_path: str, except_keys: Optional[Union[str, List]] = None) -> None:
        if except_keys is not None:
            cfg = self.pop(cfg.copy(), except_keys)

        with open(save_path, "w") as f:
            OmegaConf.save(cfg, save_path)

    def load(self, file_path: str) -> DictConfig:
        cfg = OmegaConf.load(file_path)
        if isinstance(cfg, ListConfig):
            raise ValueError(f"{file_path} is not DictConfig but ListConfig. In this project, hydra conf sopposed to be DictConfig.")

        return cfg

    def pop(self, cfg: DictConfig, except_keys: Union[str, List]) -> DictConfig:
        with open_dict(cfg):
            if isinstance(except_keys, str):
                _ = cfg.pop(except_keys)
            elif isinstance(except_keys, List):
                for key in except_keys:
                    _ = cfg.pop(key)
            else:
                raise ValueError("except_keys must be str or List[str]")
        return cfg

    def update(self, cfg: DictConfig, update_items: Dict) -> DictConfig:
        warnings.warn("OmegaconfManager.update changes original DictConfig.")
        single_depth_dict = self._convert_nested_dict_to_single_depth(update_items)
        for key, value in single_depth_dict.items():
            with open_dict(cfg):
                OmegaConf.update(cfg, key, value, merge=False)
        return cfg

    def _convert_nested_dict_to_single_depth(self, items: Union[Dict, DictConfig]) -> Dict:
        """
            This function returns 1-depth key-value dict from nested dict.
            e.g.
                (before) {'key1': {'key2': 3}}
             -> (after) {'key1.key2': 3}
        """
        if isinstance(items, DictConfig):
            items = OmegaConf.to_object(items)  # type: ignore

        results = {}
        for key, item in items.items():
            if isinstance(item, Dict):
                nested_dict = self._convert_nested_dict_to_single_depth(item)
                for nested_key, nested_value in nested_dict.items():
                    results[".".join([str(key), nested_key])] = nested_value
            else:
                results[str(key)] = item
        return results
