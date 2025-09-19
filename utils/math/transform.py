import numpy as np
from typing import Callable


def rebin(
    f: Callable, I: np.ndarray, x: np.ndarray, y: np.ndarray, du: float, dv: float
):
    X, Y = np.meshgrid(x, y)
    U, V = f(X, Y)
    u_min, u_max = np.min(U), np.max(U)
    v_min, v_max = np.min(V), np.max(V)
    u_ = np.arange(u_min, u_max, du)
    v_ = np.arange(v_min, v_max, dv)
    I_hist, _ = np.histogramdd(
        (U.flatten(), V.flatten()),
        bins=[u_.flatten(), v_.flatten()],
        weights=I.flatten(),
    )
    n_hist, _ = np.histogramdd(
        (U.flatten(), V.flatten()), bins=[u_.flatten(), v_.flatten()]
    )
    I = np.divide(I_hist, n_hist, out=np.zeros_like(I_hist), where=n_hist != 0)
    I = np.transpose(I, (1, 0))
    return I, u_[:-1], v_[:-1]


def cartesian2polar(I, x, y, dr, dtau):
    def f(x, y):
        r = np.sqrt(x**2 + y**2)
        tau = np.arctan2(y, x)
        return r, tau

    return rebin(f, I, x, y, dr, dtau)
