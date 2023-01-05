from abc import ABCMeta, abstractmethod

import numpy as np


class InterpolatorInterface(metaclass=ABCMeta):
    @abstractmethod
    def interpolate(self, ndarray: np.ndarray, observation_point_file_path: str) -> np.ndarray:
        """
        This function interpolate the given array to make grid data.

        Args:
            ndarray (np.ndarray): This array shoud be scaled to [0, 1] and dimention is one.

        Return:
            np.ndarray: Grid data (two-dimention)
        """
        pass

    def interpolate_by_gpr(self, ndarray: np.ndarray, observation_point_file_path: str) -> np.ndarray:
        pass
