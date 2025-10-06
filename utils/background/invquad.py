import numpy as np
import xarray as xr
from typing import Tuple
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


def _global_invquad_model(X: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    """
    Global inverse quadratic model: I(qxy,qz) = (A*qz + B)*qxy^-2 + C

    Parameters:
    -----------
    X : np.ndarray
        Array with shape (n_points, 2) where columns are [qxy, qz]
    A : float
        Linear coefficient for qz in amplitude
    B : float
        Constant term in amplitude
    C : float
        Constant background offset
    Returns:
    --------
    np.ndarray
        Modeled intensity values
    """
    qxy, qz = X[:, 0], X[:, 1]
    # Safely handle zero values by adding epsilon
    qxy_safe = np.where(qxy == 0, 1e-10, qxy)
    return (A * qz + B) * qxy_safe**-2 + C


def _find_data_edges(intensity_slice: np.ndarray) -> Tuple[int, int]:
    """
    Find where actual data starts/ends by identifying the span of non-NaN values.

    Parameters:
    -----------
    intensity_slice : np.ndarray
        1D intensity array for a single qz slice

    Returns:
    --------
    tuple[int, int]
        Start and end indices of the valid (non-NaN) data region
    """
    n = len(intensity_slice)
    valid_mask = ~np.isnan(intensity_slice)

    if not np.any(valid_mask):
        return 0, n - 1

    valid_indices = np.where(valid_mask)[0]
    start_idx = valid_indices[0]
    end_idx = valid_indices[-1]

    return start_idx, end_idx


def create_fit_mask(da_cart: xr.DataArray, num_fitting_points: int = 5) -> xr.DataArray:
    """
    Create a boolean mask of pixels used for inverse quadratic background fitting.

    For each qz slice:
    1. Detect data edges
    2. Select num_fitting_points from start and end regions
    3. Apply valid (qxy > 0) and finite masks
    4. Mark those positions as True in the slice

    Parameters:
    -----------
    da_cart : xr.DataArray
        Cartesian data array with dimensions (qz, qxy)
    num_fitting_points : int, optional
        Number of points to use from each edge (default: 5)

    Returns:
    --------
    xr.DataArray
        Boolean mask with same shape as da_cart, True where fitting points are used
    """
    if num_fitting_points < 1:
        raise ValueError("num_fitting_points must be positive")

    qxy_coords = da_cart.coords["qxy"].values
    qz_coords = da_cart.coords["qz"].values
    n_qz = len(qz_coords)
    n_qxy = len(qxy_coords)

    fit_mask_data = np.zeros((n_qz, n_qxy), dtype=bool)

    for i, qz_val in enumerate(qz_coords):
        intensity_slice = da_cart.isel(qz=i).values

        # Auto-detect edge positions
        start_idx, end_idx = _find_data_edges(intensity_slice)

        if start_idx >= end_idx:
            continue  # No valid data in this slice

        # Define fitting regions
        start_fit_begin = start_idx
        start_fit_end = min(start_idx + num_fitting_points, end_idx + 1)
        end_fit_begin = max(end_idx - num_fitting_points + 1, start_fit_end)
        end_fit_end = end_idx + 1

        # Start region
        if start_fit_begin < start_fit_end:
            start_qxy = qxy_coords[start_fit_begin:start_fit_end]
            start_intensity = intensity_slice[start_fit_begin:start_fit_end]
            start_valid = start_qxy > 0
            start_finite = np.isfinite(start_qxy) & np.isfinite(start_intensity)
            start_used = start_valid & start_finite
            # Mark used positions in the full slice indices
            used_start_local = np.where(start_used)[0]
            global_start_indices = start_fit_begin + used_start_local
            fit_mask_data[i, global_start_indices] = True

        # End region
        if end_fit_begin < end_fit_end:
            end_qxy = qxy_coords[end_fit_begin:end_fit_end]
            end_intensity = intensity_slice[end_fit_begin:end_fit_end]
            end_valid = end_qxy > 0
            end_finite = np.isfinite(end_qxy) & np.isfinite(end_intensity)
            end_used = end_valid & end_finite
            used_end_local = np.where(end_used)[0]
            global_end_indices = end_fit_begin + used_end_local
            fit_mask_data[i, global_end_indices] = True

    return xr.DataArray(
        fit_mask_data,
        dims=("qz", "qxy"),
        coords={"qz": qz_coords, "qxy": qxy_coords},
        name="fit_mask",
        attrs={
            "description": "Boolean mask of pixels used in inverse quadratic background fitting",
            "num_fitting_points": num_fitting_points,
        },
    )


def fit_slice_invquad_background(
    da_cart: xr.DataArray, num_fitting_points: int = 5
) -> Tuple[np.ndarray, np.ndarray, dict, xr.DataArray]:
    """
    Fit inverse quadratic background I = A*qxy^-2 + B slice-by-slice with auto edge detection.

    For each horizontal slice (constant qz):
    1. Auto-detect data edges by identifying non-NaN span
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
    tuple[np.ndarray, np.ndarray, dict, xr.DataArray]
        Fitted parameters for each slice (A_values, B_values, quality_metrics, fit_mask)
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

    fit_mask = create_fit_mask(da_cart, num_fitting_points)

    quality["num_fitting_points"] = num_fitting_points
    return A_values, B_values, quality, fit_mask


def fit_global_invquad_background(
    da_cart: xr.DataArray, num_fitting_points: int = 5
) -> Tuple[float, float, float, dict, xr.DataArray]:
    """
    Fit global inverse quadratic background I(qxy,qz) = (A*qz + B)*qxy^-2 + C.

    Collects fitting points from all slices (head and tail regions) and fits
    a single global model across the entire 2D dataset.

    Parameters:
    -----------
    da_cart : xr.DataArray
        Cartesian data array with dimensions (qz, qxy)
    num_fitting_points : int, optional
        Number of points to use from each edge after edge detection (default: 5)

    Returns:
    --------
    tuple[float, float, float, dict, xr.DataArray]
        Fitted global parameters (A, B, C), quality metrics, and fit_mask
    """
    if num_fitting_points < 1:
        raise ValueError("num_fitting_points must be positive")

    qxy_coords = da_cart.coords["qxy"].values
    qz_coords = da_cart.coords["qz"].values
    n_qz = len(qz_coords)

    # Collect all fitting points across all slices
    all_qxy_points = []
    all_qz_points = []
    all_intensity_points = []

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

            # Collect points for global fit
            all_qxy_points.extend(qxy_fit)
            all_qz_points.extend([qz_val] * len(qxy_fit))
            all_intensity_points.extend(intensity_fit)

        except Exception as e:
            failed_slices += 1
            print(f"Slice {i} failed ({qz_val:.3f}): {e}")

    # Convert to arrays
    qxy_all = np.array(all_qxy_points)
    qz_all = np.array(all_qz_points)
    intensity_all = np.array(all_intensity_points)

    if len(qxy_all) < 3:
        raise ValueError("Insufficient total points for global fitting")

    # Create X matrix for curve_fit (n_points, 2) -> [qxy, qz]
    X = np.column_stack([qxy_all, qz_all])

    # Initial parameter estimates
    C_init = np.percentile(intensity_all, 25)  # Background level
    B_init = np.median(intensity_all * qxy_all**2)  # Constant amplitude
    A_init = 0.0  # Linear qz coefficient (start with zero)

    try:
        # Fit the global model
        popt, pcov = curve_fit(
            _global_invquad_model,
            X,
            intensity_all,
            p0=[A_init, B_init, C_init],
            bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=5000,
        )

        A, B, C = popt

    except Exception as e:
        print(f"Global fit failed: {e}")
        # Fallback to median values
        A = 0.0
        B = np.median(intensity_all * qxy_all**2)
        C = np.percentile(intensity_all, 25)

    # Quality metrics
    quality = {
        "failed_slices": failed_slices,
        "total_slices": n_qz,
        "success_rate": (n_qz - failed_slices) / n_qz,
        "total_fitting_points": len(qxy_all),
        "A": A,
        "B": B,
        "C": C,
    }

    fit_mask = create_fit_mask(da_cart, num_fitting_points)

    quality["num_fitting_points"] = num_fitting_points
    return A, B, C, quality, fit_mask


def create_global_invquad_background(
    da_cart: xr.DataArray, A: float, B: float, C: float
) -> xr.DataArray:
    """
    Create inverse quadratic background array using global fitted parameters.

    Parameters:
    -----------
    da_cart : xr.DataArray
        Original cartesian data array (used for coordinates)
    A, B, C : float
        Global fitted parameters for I(qxy,qz) = (A*qz + B)*qxy^-2 + C

    Returns:
    --------
    xr.DataArray
        Full 2D background array
    """
    qxy_coords = da_cart.coords["qxy"].values
    qz_coords = da_cart.coords["qz"].values
    n_qz = len(qz_coords)
    n_qxy = len(qxy_coords)

    # Create background using global model
    background_2d = np.zeros((n_qz, n_qxy))

    for i, qz_val in enumerate(qz_coords):
        # Create background using global model
        background_slice = _global_invquad_model(
            np.column_stack([qxy_coords, np.full(n_qxy, qz_val)]), A, B, C
        )
        background_2d[i, :] = background_slice

    return xr.DataArray(
        background_2d,
        dims=("qz", "qxy"),
        coords={"qz": qz_coords, "qxy": qxy_coords},
        name="global_invquad_background",
        attrs={
            "description": "Global inverse quadratic background",
            "model": "I(qxy,qz) = (A*qz + B)*qxy^(-2) + C",
            "A": A,
            "B": B,
            "C": C,
        },
    )


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
    use_global_fit: bool = True,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Subtract inverse quadratic background from cartesian data.

    Parameters:
    -----------
    da_cart : xr.DataArray
        Cartesian data to process
    num_fitting_points : int
        Number of points to use from each edge for fitting
    use_global_fit : bool
        Whether to use global fitting (I(qxy,qz) = (A*qz + B)*qxy^-2 + C)
        or slice-wise fitting (I(qxy,qz) = A(qz)*qxy^-2 + B(qz))

    Returns:
    --------
    tuple[xr.DataArray, xr.DataArray, xr.DataArray]
        (subtracted data, background, fit_mask)
    """
    if use_global_fit:
        A, B, C, quality, fit_mask = fit_global_invquad_background(
            da_cart, num_fitting_points=num_fitting_points
        )
        background = create_global_invquad_background(da_cart, A, B, C)
    else:
        A_vals, B_vals, quality, fit_mask = fit_slice_invquad_background(
            da_cart, num_fitting_points=num_fitting_points
        )
        background = create_slice_invquad_background(da_cart, A_vals, B_vals)

    subtracted = da_cart - background

    return subtracted, background, fit_mask
