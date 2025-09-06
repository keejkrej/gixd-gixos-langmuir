import numpy as np

def linear_detrend(y):
    """
    Remove linear background from 1D data by connecting first and last points.
    
    Parameters:
    y : array-like
        Input 1D data array
        
    Returns:
    detrended : array
        Data with linear background removed
    """
    y = np.asarray(y)
    
    if len(y) < 2:
        return y
    
    # Create linear background from first to last point
    x = np.arange(len(y))
    first_point = y[0]
    last_point = y[-1]
    
    # Calculate slope and intercept
    slope = (last_point - first_point) / (len(y) - 1)
    background = first_point + slope * x
    
    # Subtract background
    detrended = y - background
    
    return detrended