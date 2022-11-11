from abc import ABCMeta, abstractmethod
import numpy as np


class GeoimgGeneratorInterface(metaclass=ABCMeta):
    @abstractmethod
    def gen_img(self, scaled_ndarray: np.ndarray, observation_point_file_path: str, save_img_path: str) -> None:
        """
        This function save geo image with given ndarray.

        Args:
            scaled_ndarray (np.ndarray): The scaled_ndarray shoud be scaled its original scale. NOT like [0, 1]
            save_img_path (str): save_path to save image.
        """
        pass
