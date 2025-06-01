import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

def rebin(f: Callable, I: np.ndarray, x: np.ndarray, y: np.ndarray, du: float, dv: float):
    """
    Rebin an image using a function f(x, y) that returns u and v coordinates.
    
    Parameters:
    -----------
    f : Callable
        Function that returns u and v coordinates for each pixel in the image
    I : np.ndarray
        Input image array to be rebinned.
    x : np.ndarray
        x coordinates of the image
    y : np.ndarray
        y coordinates of the image
    du : float
        bin size in u direction
    dv : float
        bin size in v direction
    
    Returns:
    --------
    I : np.ndarray
        Rebinned image
    u : np.ndarray
        u coordinates of the rebinned image
    v : np.ndarray
        v coordinates of the rebinned image
    """
    X, Y = np.meshgrid(x, y)
    U, V = f(X, Y)
    u_min, u_max = np.min(U), np.max(U)
    v_min, v_max = np.min(V), np.max(V)
    u_ = np.arange(u_min, u_max, du)
    v_ = np.arange(v_min, v_max, dv)
    I_hist, _ = np.histogramdd((U.flatten(), V.flatten()), bins=[u_.flatten(), v_.flatten()], weights=I.flatten())
    n_hist, _ = np.histogramdd((U.flatten(), V.flatten()), bins=[u_.flatten(), v_.flatten()])
    I = np.divide(I_hist, n_hist, out=np.zeros_like(I_hist), where=n_hist != 0)
    I = np.transpose(I, (1, 0))
    return I, u_[:-1], v_[:-1]

def cartesian2polar(I, x, y, dr, dtheta):
    """
    Convert a Cartesian coordinate image to polar coordinates and rebin it.
    
    This function converts an image from Cartesian (x, y) coordinates to polar
    (r, theta) coordinates and then rebins it using the specified bin sizes.
    
    Parameters:
    -----------
    I : np.ndarray
        Input image array in Cartesian coordinates
    x : np.ndarray
        x coordinates of the input image
    y : np.ndarray
        y coordinates of the input image
    dr : float
        Bin size in radial (r) direction
    dtheta : float
        Bin size in angular (theta) direction
    
    Returns:
    --------
    I : np.ndarray
        Rebinned image in polar coordinates
    r : np.ndarray
        Radial coordinates of the rebinned image
    theta : np.ndarray
        Angular coordinates of the rebinned image
    """
    def f(x, y):
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return r, theta
    return rebin(f, I, x, y, dr, dtheta)

if __name__ == "__main__":
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-2, 2, 200)
    x_0, y_0 = 0, 0
    w_x, w_y = 1, 2
    X, Y = np.meshgrid(x, y)
    I_cart = np.exp(-((X-x_0)/w_x)**2 - ((Y-y_0)/w_y)**2)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    im1 = ax1.imshow(I_cart, extent=[np.min(x), np.max(x), np.min(y), np.max(y)])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Original 2D Gaussian')
    plt.colorbar(im1, ax=ax1)
    # Rebin and plot polar image
    I_polar, r, theta = cartesian2polar(I_cart, x, y, 0.05, 0.1)
    im2 = ax2.imshow(I_polar, extent=[np.min(r), np.max(r), np.max(theta), np.min(theta)])
    ax2.set_xlabel('r')
    ax2.set_ylabel('theta')
    ax2.set_title('Polar Rebinning')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()