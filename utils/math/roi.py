import numpy as np

def get_crop(bbox, I, x, y):
    """
    Crop an image to a specified bounding box.
    
    Parameters:
    -----------
    bbox : tuple
        Bounding box (x_min, x_max, y_min, y_max)
    I : np.ndarray
        Input image array to be cropped
    x : np.ndarray
        x coordinates of the image
    y : np.ndarray
        y coordinates of the image
    
    Returns:
    --------
    I : np.ndarray
        Cropped image
    """
    x_min, x_max, y_min, y_max = bbox
    X, Y = np.meshgrid(x, y)
    mask = (X > x_min) & (X < x_max) & (Y > y_min) & (Y < y_max)
    I_copy = np.copy(I)
    I_copy[~mask] = np.nan
    return I_copy

def get_avg(bbox, I, x, y, axis):
    """
    Calculate the average of an image within a specified bounding box.
    
    Parameters:
    -----------
    bbox : tuple
        Bounding box (x_min, x_max, y_min, y_max)
    I : np.ndarray
        Input image array
    x : np.ndarray
        x coordinates of the image
    y : np.ndarray
        y coordinates of the image
    axis : str
        Axis along which to calculate the average ('x' or 'y')
    
    Returns:
    --------
    avg : float
        Average of the image within the specified bounding box
    """
    crop = get_crop(bbox, I, x, y)
    if axis == 'x':
        axis = 1
    elif axis == 'y':
        axis = 0
    else:
        raise ValueError("axis must be 'x' or 'y'")
    avg = np.nanmean(crop, axis=axis)
    return avg

def get_sum(bbox, I, x, y, axis):
    """
    Calculate the sum of an image within a specified bounding box.
    
    Parameters:
    -----------
    bbox : tuple
        Bounding box (x_min, x_max, y_min, y_max)
    I : np.ndarray
        Input image array
    x : np.ndarray
        x coordinates of the image
    y : np.ndarray
        y coordinates of the image
    axis : str
        Axis along which to calculate the sum ('x' or 'y')
    
    Returns:
    --------
    sum : float
        Sum of the image within the specified bounding box
    """
    crop = get_crop(bbox, I, x, y)
    if axis == 'x':
        axis = 1
    elif axis == 'y':
        axis = 0
    else:
        raise ValueError("axis must be 'x' or 'y'")
    sum = np.nansum(crop, axis=axis)
    return sum
