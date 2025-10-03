import numpy as np
from scipy.signal import medfilt
from scipy.optimize import curve_fit
from typing import Tuple, Optional


def detect_peaks_median(data, coords, window_size, sigma_threshold):
    """
    Detect peaks using median filter with reflection for boundary handling.

    Parameters:
    -----------
    data : array-like
        The intensity data
    coords : array-like
        The coordinate values (q or tau)
    window_size : int
        Size of the median filter window (must be odd)
    sigma_threshold : float
        Number of standard deviations above median for peak detection

    Returns:
    --------
    list of tuples
        List of (coordinate, intensity) pairs for detected peaks
    """
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Apply median filter with reflection boundary handling
    pad_width = window_size // 2
    padded_data = np.pad(data, pad_width, mode="reflect")
    median_filtered = medfilt(padded_data, kernel_size=window_size)
    median_filtered = median_filtered[pad_width:-pad_width]  # Remove padding

    # Calculate noise level (std of residuals)
    residuals = data - median_filtered
    noise_std = np.std(residuals)

    # Set threshold
    threshold = median_filtered + sigma_threshold * noise_std

    # Find peaks (points above threshold that are local maxima)
    peaks = []
    for i in range(len(data)):
        # Check if above threshold
        if data[i] < threshold[i]:
            continue

        # Check if local maximum (including endpoints)
        if i == 0:  # First point
            is_peak = len(data) > 1 and data[i] > data[i + 1]
        elif i == len(data) - 1:  # Last point
            is_peak = len(data) > 1 and data[i] > data[i - 1]
        else:  # Interior points
            is_peak = data[i] > data[i - 1] and data[i] > data[i + 1]

        if is_peak:
            peaks.append(i)

    return [(coords[i], data[i]) for i in peaks]


def mirrored_gaussian(
    x: np.ndarray, amplitude: float, center: float, sigma: float, offset: float = 0.0
) -> np.ndarray:
    """
    Mirrored Gaussian function: gauss(x0) + gauss(-x0) + offset

    Parameters:
    -----------
    x : array-like
        Input coordinates
    amplitude : float
        Amplitude of each Gaussian component
    center : float
        Center position x0
    sigma : float
        Standard deviation of each Gaussian
    offset : float, optional
        Baseline offset (default: 0.0)

    Returns:
    --------
    array-like
        Fitted intensity values
    """
    gauss1 = amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    gauss2 = amplitude * np.exp(-0.5 * ((x + center) / sigma) ** 2)
    return gauss1 + gauss2 + offset


def fit_mirrored_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    initial_guess: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Fit a mirrored Gaussian function: gauss(x0) + gauss(-x0) + offset to data.
    Ignores points with tau < 5 for fitting.

    Parameters:
    -----------
    x : array-like
        Coordinate values (tau values)
    y : array-like
        Intensity values
    initial_guess : tuple of (amplitude, center, sigma, offset), optional
        Initial parameter guesses. If None, uses automatic estimation.

    Returns:
    --------
    tuple of (amplitude, center, sigma, offset, fitted_y)
        Fitted parameters and fitted curve evaluated on full x range
    """
    # Filter out points with tau < 5
    mask = x >= 5
    x_fit = x[mask]
    y_fit = y[mask]

    # Check if we have enough points to fit
    if len(x_fit) < 5:
        print("Warning: Not enough points (tau >= 5) for mirrored Gaussian fitting")
        # Return a simple fallback: just use the overall maximum
        center_idx = np.argmax(y)
        center = abs(x[center_idx])
        amplitude = np.max(y) / 2.0
        sigma = 0.5
        offset = np.min(y)  # Estimate baseline as minimum value
        fitted_y = mirrored_gaussian(x, amplitude, center, sigma, offset)
        return amplitude, center, sigma, offset, fitted_y

    if initial_guess is None:
        # Estimate initial parameters from filtered data (tau >= 5)
        # Amplitude: maximum intensity divided by 2 (since two Gaussians)
        amplitude = np.max(y_fit) / 2.0

        # Center: position of maximum intensity in filtered data
        center_idx_fit = np.argmax(y_fit)
        center = abs(x_fit[center_idx_fit])  # Take absolute value for positive center

        # Sigma: estimate from FWHM of the central peak in filtered data
        # Find half-maximum points around the center in filtered data
        half_max = amplitude
        left_idx = center_idx_fit
        while left_idx > 0 and y_fit[left_idx] > half_max:
            left_idx -= 1
        right_idx = center_idx_fit
        while right_idx < len(y_fit) - 1 and y_fit[right_idx] > half_max:
            right_idx += 1

        if right_idx > left_idx:
            fwhm = x_fit[right_idx] - x_fit[left_idx]
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        else:
            sigma = 0.1  # fallback

        # Offset: estimate as the minimum value in the filtered data
        offset = np.min(y_fit)

        initial_guess = (amplitude, center, sigma, offset)

    try:
        # Fit the mirrored Gaussian using only tau >= 5 data
        popt, _ = curve_fit(
            mirrored_gaussian,
            x_fit,
            y_fit,
            p0=initial_guess,
            bounds=([0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
        )

        amplitude_fit, center_fit, sigma_fit, offset_fit = popt
        # Return fitted curve evaluated on full x range
        fitted_y = mirrored_gaussian(
            x, amplitude_fit, center_fit, sigma_fit, offset_fit
        )

        return amplitude_fit, center_fit, sigma_fit, offset_fit, fitted_y

    except Exception as e:
        # If fitting fails, return initial guess and original data
        print(f"Warning: Mirrored Gaussian fitting failed: {e}")
        amplitude, center, sigma, offset = initial_guess
        fitted_y = mirrored_gaussian(x, amplitude, center, sigma, offset)
        return amplitude, center, sigma, offset, fitted_y
