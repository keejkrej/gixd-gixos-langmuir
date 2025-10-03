#!/usr/bin/env python3
"""
Test script for the mirrored Gaussian fitting function.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.math.peak import mirrored_gaussian, fit_mirrored_gaussian


def test_mirrored_gaussian():
    """Test the mirrored Gaussian fitting on synthetic data."""

    # Create synthetic data with mirrored Gaussian
    x = np.linspace(-2, 2, 100)
    true_amplitude = 1.5
    true_center = 0.8
    true_sigma = 0.3
    true_offset = 0.2
    noise_level = 0.05

    # Generate true signal
    y_true = mirrored_gaussian(x, true_amplitude, true_center, true_sigma, true_offset)

    # Add noise
    np.random.seed(42)
    y_noisy = y_true + np.random.normal(0, noise_level, len(x))

    # Fit the mirrored Gaussian
    amplitude_fit, center_fit, sigma_fit, offset_fit, y_fitted = fit_mirrored_gaussian(
        x, y_noisy
    )

    # Print results
    print("True parameters:")
    print(f"  Amplitude: {true_amplitude:.3f}")
    print(f"  Center: {true_center:.3f}")
    print(f"  Sigma: {true_sigma:.3f}")
    print(f"  Offset: {true_offset:.3f}")
    print("\nFitted parameters:")
    print(f"  Amplitude: {amplitude_fit:.3f}")
    print(f"  Center: {center_fit:.3f}")
    print(f"  Sigma: {sigma_fit:.3f}")
    print(f"  Offset: {offset_fit:.3f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, "b-", label="True signal", linewidth=2)
    plt.plot(x, y_noisy, "k.", label="Noisy data", alpha=0.6)
    plt.plot(x, y_fitted, "r--", label="Fitted mirrored Gaussian", linewidth=2)
    plt.xlabel("x (tau)")
    plt.ylabel("Intensity")
    plt.title("Mirrored Gaussian Fitting Test")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("test_mirrored_gauss.png", dpi=150)
    plt.close()

    print(f"\nTest plot saved as 'test_mirrored_gauss.png'")
    print("Fitting test completed successfully!")


if __name__ == "__main__":
    test_mirrored_gaussian()
