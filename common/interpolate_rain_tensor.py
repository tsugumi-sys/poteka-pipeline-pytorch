from typing import Union
import json
import sys

import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

sys.path.append(".")
from train.src.config import DEVICE

def interpolate_rain_data(ndarray: np.ndarray, return_torch_rensor: bool = False) -> Union[torch.Tensor, np.ndarray]:
    """This function interploate observation point rainfall data to create grid data.

    ndarray (numpy.ndarray): ndarray dimention should be one (e.g. (35,)) because this function is used for interpolate OBPointSeq2Seq modles' function.
    """
    if ndarray.ndim != 1:
        raise ValueError(f"Invalid dimention of ndarray (ndim: {ndarray.ndim}, shape: {ndarray.shape}) for interpolation. ")
    with open("../common/meta-data/observation_point.json", "r") as f:
        ob_point_data = json.load(f)

    ob_point_lons = [val["longitude"] for val in ob_point_data.values()]
    ob_point_lats = [val["latitude"] for val in ob_point_data.values()]
    
    grid_coordinate = np.mgrid[120.90:121.150:50j, 14.350:14.760:50j]

    kernel = ConstantKernel(1, (1e-5, 1e5)) * RBF(1, (1e-5, 1e5))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, random_state=123)

    x = np.column_stack([ob_point_lons, ob_point_lats])
    gp.fit(x, ndarray)

    y_pred, _ = gp.predict(grid_coordinate.reshape(2, -1).T, return_std=True)

    rain_grid_data = np.reshape(y_pred, (50, 50))
    rain_grid_data = np.where(rain_grid_data > 0, rain_grid_data, 0)
    rain_grid_data = np.where(rain_grid_data > 100, 100, rain_grid_data)
    rain_grid_data = rain_grid_data.astype(np.float32)
    if return_torch_rensor:
        return torch.from_numpy(rain_grid_data).to(DEVICE)
    return rain_grid_data.astype(np.float32)


if __name__ == "__main__":
    ndarray = np.zeros((35))
    for i in range(35):
        ndarray[i] = i
    print(ndarray)
    grid_data = interpolate_rain_data(ndarray)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 8))
    grid_coordinate = np.mgrid[120.90:121.150:50j, 14.350:14.760:50j]
    cs = ax.contourf(*grid_coordinate, grid_data, cmap="rainbow")

    with open("./common/meta-data/observation_point.json", "r") as f:
        ob_point_data = json.load(f)

    ob_point_lons = [val["longitude"] for val in ob_point_data.values()]
    ob_point_lats = [val["latitude"] for val in ob_point_data.values()]

    ax.scatter(ob_point_lons, ob_point_lats)

    for idx, (lon, lat) in enumerate(zip(ob_point_lons, ob_point_lats)):
        ax.annotate(ndarray[idx], (lon, lat))

    plt.savefig("./interp.png")
    plt.close()
