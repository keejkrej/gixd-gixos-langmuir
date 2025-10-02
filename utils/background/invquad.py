import numpy as np
import xarray as xr
from typing import Tuple, Optional
from scipy.optimize import curve_fit


def _invquad_model(qxy: np.ndarray, A: float, B: float) -> np.ndarray:
    """
    Internal inverse quadratic model: I = A*qxy^-2 + B

    Parameters:
    -----------
    qxy : np.ndarray
        Qxy coordinates (must be non-zero)
    A : float
        Amplitude parameter for this slice
    B : float
        Background offset parameter for this slice
    Returns:
    --------
    np.ndarray
        Modeled intensity values for this slice
    """
    # Safely handle zero values by adding epsilon
    qxy_safe = np.where(qxy == 0, 1e-10, qxy)
    return A * qxy_safe**-2 + B


def _find_data_edges(intensity_slice: np.ndarray) -> Tuple[int, int]:
    """
    Find where actual data starts/ends by scanning for first non-zero values.

    Parameters:
    -----------
    intensity_slice : np.ndarray
        1D intensity array for a single qz slice

    Returns:
    --------
    tuple[int, int]
        Start and end indices where real data begins/ends
    """
    n = len(intensity_slice)

    # Find start edge: scan from left until we hit non-zero
    start_idx = 0
    for i in range(n):
        if intensity_slice[i] != 0:
            start_idx = i
            break

    # Find end edge: scan from right until we hit non-zero
    end_idx = n - 1
    for i in range(n - 1, -1, -1):
        if intensity_slice[i] != 0:
            end_idx = i
            break

    return start_idx, end_idx


def fit_slice_invquad_background(
    da_cart: xr.DataArray, num_fitting_points: int = 5
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Fit inverse quadratic background I = A*qxy^-2 + B slice-by-slice with auto edge detection.

    For each horizontal slice (constant qz):
    1. Auto-detect data edges by scanning for jumps from zero-filled regions
    2. Use num_fitting_points from edges for fitting
    3. Fit individual parameters for each slice

    Parameters:
    -----------
    da_cart : xr.DataArray
        Cartesian data array with dimensions (qz, qxy)
    num_fitting_points : int, optional
        Number of points to use from each edge after edge detection (default: 5)

    Returns:
    --------
    tuple[np.ndarray, np.ndarray, dict]
        Fitted parameters for each slice (A_values, B_values, quality_metrics)
    """
    if num_fitting_points < 1:
        raise ValueError("num_fitting_points must be positive")

    qxy_coords = da_cart.coords["qxy"].values
    qz_coords = da_cart.coords["qz"].values
    n_qz = len(qz_coords)

    A_values = np.zeros(n_qz)
    B_values = np.zeros(n_qz)
    failed_slices = 0

    for i, qz_val in enumerate(qz_coords):
        try:
            # Extract intensity slice
            intensity_slice = da_cart.isel(qz=i).values

            # Auto-detect edge positions
            start_idx, end_idx = _find_data_edges(intensity_slice)

            # Define fitting regions
            start_fit_begin = start_idx
            start_fit_end = min(start_idx + num_fitting_points, end_idx)
            end_fit_begin = max(end_idx - num_fitting_points, start_fit_end)
            end_fit_end = end_idx + 1

            # Extract fitting regions
            qxy_fit_points = np.concatenate(
                [
                    qxy_coords[start_fit_begin:start_fit_end],
                    qxy_coords[end_fit_begin:end_fit_end],
                ]
            )
            intensity_points = np.concatenate(
                [
                    intensity_slice[start_fit_begin:start_fit_end],
                    intensity_slice[end_fit_begin:end_fit_end],
                ]
            )

            # Remove any negative/zero qxy values
            valid_mask = qxy_fit_points > 0
            if not np.any(valid_mask):
                raise ValueError("No valid qxy points")

            qxy_final = qxy_fit_points[valid_mask]
            intensity_final = intensity_points[valid_mask]

            # Remove NaN/inf values
            finite_mask = np.isfinite(qxy_final) & np.isfinite(intensity_final)
            if not np.any(finite_mask):
                raise ValueError("No finite values")

            qxy_fit = qxy_final[finite_mask]
            intensity_fit = intensity_final[finite_mask]

            if len(qxy_fit) < 2:
                raise ValueError("Insufficient points for fitting")

            # Parameter initial estimates
            B_init = np.percentile(intensity_fit, 25)  # Background level
            A_init = np.median(intensity_fit * qxy_fit**2)  # Amplitude level

            # Fit the model
            popt, pcov = curve_fit(
                _invquad_model,
                qxy_fit,
                intensity_fit,
                p0=[A_init, B_init],
                bounds=([0, -np.inf], [np.inf, np.inf]),
                maxfev=1000,
            )

            A_values[i], B_values[i] = popt

        except Exception as e:
            # Fallback for failed slices
            print(f"Slice {i} failed ({qz_val:.3f}): {e}")
            failed_slices += 1
            A_values[i] = 0
            B_values[i] = np.percentile(intensity_slice, 30)

    # Quality metrics
    quality = {
        "failed_slices": failed_slices,
        "total_slices": n_qz,
        "success_rate": (n_qz - failed_slices) / n_qz,
        "A_range": (np.min(A_values), np.max(A_values)),
        "B_range": (np.min(B_values), np.max(B_values)),
        "A_mean": np.mean(A_values),
        "B_mean": np.mean(B_values),
        "A_std": np.std(A_values),
        "B_std": np.std(B_values),
    }

    return A_values, B_values, quality


def create_slice_invquad_background(
    da_cart: xr.DataArray, A_values: np.ndarray, B_values: np.ndarray
) -> xr.DataArray:
    """
    Create inverse quadratic background array using slice-wise fitted parameters.

    Parameters:
    -----------
    da_cart : xr.DataArray
        Original cartesian data array (used for coordinates)
    A_values, B_values : np.ndarray
        Fitted parameters for each slice

    Returns:
    --------
    xr.DataArray
        Full 2D background array
    """
    qxy_coords = da_cart.coords["qxy"].values
    qz_coords = da_cart.coords["qz"].values
    n_qz = len(qz_coords)
    n_qxy = len(qxy_coords)

    # Create background slice by slice
    background_2d = np.zeros((n_qz, n_qxy))

    for i in range(n_qz):
        if i < len(A_values) and i < len(B_values):
            A_slice = A_values[i]
            B_slice = B_values[i]
        else:
            A_slice = 0
            B_slice = 0

        # Create background using individual slice model
        background_slice = _invquad_model(qxy_coords, A_slice, B_slice)
        background_2d[i, :] = background_slice

    return xr.DataArray(
        background_2d,
        dims=("qz", "qxy"),
        coords={"qz": qz_coords, "qxy": qxy_coords},
        name="invquad_background",
        attrs={
            "description": "Slice-wise inverse quadratic background",
            "model": "I(qxy,qz) = A(qz)*qxy^(-2) + B(qz)",
        },
    )


def subtract_invquad_background(
    da_cart: xr.DataArray,
    num_fitting_points: int = 5,
    return_background: bool = False,
) -> Tuple[xr.DataArray, Optional[xr.DataArray]]:
    """
    Subtract inverse quadratic background from cartesian data slice-by-slice.

    Parameters:
    -----------
    da_cart : xr.DataArray
        Cartesian data to process
    num_fitting_points : int
        Number of points to use from each edge for fitting
    return_background : bool
        Whether to return the background array

    Returns:
    --------
    tuple[xr.DataArray, xr.DataArray] or tuple[xr.DataArray, None]
        Subtracted data and optional background
    """

    # Fit slice-by-slice parameters
    A_vals, B_vals, quality = fit_slice_invquad_background(
        da_cart, num_fitting_points=num_fitting_points
    )


    # Create full 2D background
    background = create_slice_invquad_background(da_cart, A_vals, B_vals)

    # Subtract and return
    subtracted = da_cart - background

    if return_background:
        return subtracted, background
    else:
        return subtracted, None
