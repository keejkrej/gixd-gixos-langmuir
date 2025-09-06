import numpy as np
from scipy.signal import medfilt


def detect_peaks_median(data, coords, window_size, sigma_threshold):
    """
    Detect peaks using median filter with reflection for boundary handling.

    Parameters:
    -----------
    data : array-like
        The intensity data
    coords : array-like
        The coordinate values (q or theta)
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
