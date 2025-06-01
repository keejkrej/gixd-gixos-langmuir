import numpy as np
import matplotlib.pyplot as plt

def fit_gaussian(x, intensity, fit_range=None):
    if fit_range is None:
        fit_range = (np.min(x), np.max(x))
    mask = np.isnan(intensity) | np.isinf(intensity) | (x < fit_range[0]) | (x > fit_range[1])
    x_masked = x[~mask]
    intensity_masked = intensity[~mask]
    
    # Polynomial fit (degree 2 for parabolic fit to find extremum)
    coeffs = np.polyfit(x_masked, intensity_masked, 2)
    intensity_fit = np.polyval(coeffs, x_masked)
    
    # Find extremum by taking derivative and setting to zero
    # For polynomial ax^2 + bx + c, derivative is 2ax + b = 0, so x = -b/(2a)
    a, b, c = coeffs
    if a != 0:
        x_max_fit = -b / (2 * a)
        # If multiple extrema or outside range, use the first valid one
        if x_max_fit < fit_range[0] or x_max_fit > fit_range[1]:
            # Fall back to argmax of fitted values
            x_max_fit = x_masked[np.argmax(intensity_fit)]
    else:
        # Linear case, use argmax
        x_max_fit = x_masked[np.argmax(intensity_fit)]
    
    return x_masked, intensity_fit, x_max_fit

def test_fit_gaussian():
    x = np.linspace(0, 10, 100)
    # Create a parabolic peak for testing
    intensity = -(x - 5)**2 + 10 + np.random.normal(0, 0.1, 100)
    x_masked, intensity_fit, x_max_fit = fit_gaussian(x, intensity, (2, 8))
    plt.scatter(x, intensity)
    plt.plot(x_masked, intensity_fit)
    plt.axvline(x_max_fit, color='red', label=f'x_max = {x_max_fit:.2f}')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    test_fit_gaussian()