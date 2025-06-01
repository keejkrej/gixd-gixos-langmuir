import numpy as np
from lmfit import Model
import matplotlib.pyplot as plt

def gaussian(x, amplitude, x_max, width, background, slope):
    return amplitude * np.exp(-(x - x_max) ** 2 / (2 * width ** 2)) + slope * x + background

def lorentzian(x, amplitude, x_max, width, background, slope):
    return amplitude / (1 + ((x - x_max) / width) ** 2) + slope * x + background

def fit_gaussian(x, intensity, fit_range=None, ignore_range=None):
    if fit_range is None:
        fit_range = (np.min(x), np.max(x))
    if ignore_range is None:
        ignore_range = []
    mask = np.isnan(intensity) | np.isinf(intensity) | (x < fit_range[0]) | (x > fit_range[1])
    for r in ignore_range:
        mask |= (x > r[0]) & (x < r[1])
    x_masked = x[~mask]
    intensity_masked = intensity[~mask]
    mod = Model(gaussian)
    params = mod.make_params(
        amplitude=np.max(intensity_masked)-np.min(intensity_masked),
        x_max=x_masked[np.argmax(intensity_masked)],
        width=np.std(x_masked),
        background=np.min(intensity_masked),
        slope=(intensity_masked[0]-intensity_masked[-1])/(x_masked[0]-x_masked[-1])
    )
    params['amplitude'].min = 0
    params['width'].min = 0
    params['x_max'].min = fit_range[0]
    params['x_max'].max = fit_range[1]
    result = mod.fit(intensity_masked, params, x=x_masked)
    intensity_fit = np.array(result.eval(x=x_masked))
    x_max_fit = result.params['x_max'].value
    return x_masked, intensity_fit, x_max_fit

def fit_lorentzian(x, intensity, fit_range=None, ignore_range=None):
    if fit_range is None:
        fit_range = (np.min(x), np.max(x))
    if ignore_range is None:
        ignore_range = []
    mask = np.isnan(intensity) | np.isinf(intensity) | (x < fit_range[0]) | (x > fit_range[1])
    for r in ignore_range:
        mask |= (x > r[0]) & (x < r[1])
    x_masked = x[~mask]
    intensity_masked = intensity[~mask]
    mod = Model(lorentzian)
    params = mod.make_params(
        amplitude=np.max(intensity_masked)-np.min(intensity_masked),
        x_max=x_masked[np.argmax(intensity_masked)],
        width=np.abs(x_masked[0]-x_masked[-1])/2,
        background=np.min(intensity_masked),
        slope=(intensity_masked[0]-intensity_masked[-1])/(x_masked[0]-x_masked[-1])
    )
    params['amplitude'].min = 0
    params['width'].min = 0
    params['x_max'].min = fit_range[0]
    params['x_max'].max = fit_range[1]
    result = mod.fit(intensity_masked, params, x=x_masked)
    intensity_fit = np.array(result.eval(x=x_masked))
    x_max_fit = result.params['x_max'].value
    return x_masked, intensity_fit, x_max_fit

def test_fit_gaussian():
    # Generate synthetic Gaussian data
    x = np.linspace(0, 10, 200)
    true_params = dict(amplitude=5, x_max=4, width=0.5, background=1, slope=0.1)
    intensity = gaussian(x, **true_params) + 0.1 * np.random.randn(len(x))
    x_masked, intensity_fit, x_max_fit = fit_gaussian(x, intensity, fit_range=(2, 6))
    print("Gaussian Fit Test:")
    print(f"Original max intensity: {np.max(intensity):.3f}")
    print(f"Fitted max intensity: {np.max(intensity_fit):.3f}")
    plt.figure()
    plt.plot(x, intensity, 'ko', label='Data')
    plt.plot(x_masked, intensity_fit, 'r-', label='Fit')
    plt.axvline(x_max_fit, color='g', linestyle='--', label=f'x_max_fit = {x_max_fit:.3f}')
    plt.xlabel('x')
    plt.ylabel('Intensity')
    plt.title('Gaussian Fit Test')
    plt.legend()
    plt.show()
    # Assert fit captures the peak reasonably well
    correlation = np.corrcoef(intensity.astype(float), intensity_fit.astype(float))[0, 1]
    assert correlation > 0.8  # Correlation should be high

def test_fit_lorentzian():
    # Generate synthetic Lorentzian data
    q = np.linspace(0, 10, 200)
    true_params = dict(amplitude=5, x_max=6, width=0.7, background=0.5, slope=-0.05)
    intensity = lorentzian(q, **true_params) + 0.1 * np.random.randn(len(q))
    x_masked, intensity_fit, x_max_fit = fit_lorentzian(q, intensity, fit_range=(4, 8))
    print("Lorentzian Fit Test:")
    print(f"Original max intensity: {np.max(intensity):.3f}")
    print(f"Fitted max intensity: {np.max(intensity_fit):.3f}")
    plt.figure()
    plt.plot(q, intensity, 'ko', label='Data')
    plt.plot(x_masked, intensity_fit, 'r-', label='Fit')
    plt.axvline(x_max_fit, color='g', linestyle='--', label=f'x_max_fit = {x_max_fit:.3f}')
    plt.xlabel('q')
    plt.ylabel('Intensity')
    plt.title('Lorentzian Fit Test')
    plt.legend()
    plt.show()
    # Assert fit captures the peak reasonably well
    correlation = np.corrcoef(intensity.astype(float), intensity_fit.astype(float))[0, 1]
    assert correlation > 0.8  # Correlation should be high

if __name__ == "__main__":
    test_fit_gaussian()
    test_fit_lorentzian()
