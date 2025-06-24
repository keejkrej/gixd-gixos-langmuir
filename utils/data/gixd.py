import numpy as np
import xarray as xr
from pathlib import Path
from utils.math.transform import cartesian2polar

def load_gixd_xarray(data_path, name, index):
    data_path = Path(data_path) / name
    intensity = np.loadtxt(data_path / f'{name}_{index}_{index}_combined_I.dat')
    qxy = np.loadtxt(data_path / f'{name}_{index}_{index}_combined_Qxy.dat')
    qz = np.loadtxt(data_path / f'{name}_{index}_{index}_combined_Qz.dat')
    intensity = np.nan_to_num(intensity, nan=0)
    da_cart = xr.DataArray(
        intensity,
        dims=('qz', 'qxy'),
        coords={'qz': qz, 'qxy': qxy},
        name='intensity'
    )
    return da_cart

def gixd_cartesian2polar(da_cart, dr=0.01, dtheta=0.01):
    intensity_polar, q, theta = cartesian2polar(
        da_cart.values, da_cart['qxy'].values, da_cart['qz'].values, dr, dtheta
    )
    da_polar = xr.DataArray(
        intensity_polar,
        dims=('theta', 'q'),
        coords={'theta': np.rad2deg(theta), 'q': q},
        name='intensity_polar'
    )
    return da_polar

def extract_intensity_q(da_polar, q_range=None, theta_range=None, method='mean'):
    da = da_polar
    if q_range is not None:
        da = da.sel(q=slice(q_range[0], q_range[1]))
    if theta_range is not None:
        da = da.sel(theta=slice(theta_range[0], theta_range[1]))
    if method == 'mean':
        return da.mean(dim='theta')
    elif method == 'sum':
        return da.sum(dim='theta')
    else:
        raise ValueError("method must be 'mean' or 'sum'")

def extract_intensity_theta(da_polar, q_range=None, theta_range=None, method='mean'):
    da = da_polar
    if q_range is not None:
        da = da.sel(q=slice(q_range[0], q_range[1]))
    if theta_range is not None:
        da = da.sel(theta=slice(theta_range[0], theta_range[1]))
    if method == 'mean':
        return da.mean(dim='q')
    elif method == 'sum':
        return da.sum(dim='q')
    else:
        raise ValueError("method must be 'mean' or 'sum'") 