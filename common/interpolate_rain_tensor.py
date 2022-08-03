from typing import Union
import json

import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

def interpolate_rain_data(tensor: torch.Tensor) -> torch.Tensor:
    """This function interploate observation point rainfall data to create grid data.

    """
    with open("../common/meta-data/observation_point.json", "r") as f:
        ob_point_data = json.load(f)

    ob_point_names = list(ob_point_data.keys())
    ob_point_lons = [val["longitude"] for val in ob_point_data.values()]
    ob_point_lats = [val["latitude"] for val in ob_point_data.values()]
    
    grid_lons = np.linspace(120.90, 121.150, 50)
    grid_lats = np.linspace(14.350, 14.760, 50)
    grid_coordinate = 
