import numpy as np

def exponential_weighted_center(intensity, coordinates, temperature=0.1):
    """
    Calculate softmax-weighted center of mass.
    
    Parameters:
    intensity : array-like
        Intensity values (can be positive or negative)
    coordinates : array-like  
        Coordinate values corresponding to intensities
    temperature : float, optional
        Temperature parameter controlling exponential scaling (default: 0.1)
        Lower temperature = more aggressive exponential weighting (more peak-sensitive)
        
    Returns:
    weighted_center : float
        Softmax-weighted center coordinate
    """
    intensity = np.asarray(intensity)
    coordinates = np.asarray(coordinates)
    
    if len(intensity) == 0:
        return 0.0
    
    if len(intensity) != len(coordinates):
        raise ValueError("intensity and coordinates must have the same length")
    
    # Apply softmax weighting: exp(I / temperature) / sum(exp(I / temperature))
    # Subtract max for numerical stability
    max_intensity = np.max(intensity) if len(intensity) > 0 else 0
    scaled_intensity = (intensity - max_intensity) / temperature
    weights = np.exp(scaled_intensity)
    
    # Calculate weighted center
    if np.sum(weights) > 0:
        weighted_center = np.sum(weights * coordinates) / np.sum(weights)
    else:
        # Fallback to simple maximum if weights sum to zero
        weighted_center = coordinates[np.argmax(intensity)]
    
    return weighted_center