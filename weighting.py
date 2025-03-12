import numpy as np

def compute_weights(margins, slider_value):
    """
    Compute observation weights based on race margins and a user-defined slider value.
    
    Parameters:
        margins (array-like): Differences in finishing times between boats.
        slider_value (float): User input from 0 to 100 (higher = more discounting for large margins).
        
    Returns:
        weights (array-like): Adjusted weights for GLM.
    """
    slider_scaled = slider_value / 100  # Convert to 0-1 range
    base_weight = 1 - (0.5 * slider_scaled)  # Ensure minimum weight is 0.5 at most

    weights = base_weight / (1 + slider_scaled * np.abs(margins))  # Inverse relationship with margin
    return np.clip(weights, 0.5, 1)  # Keep weights in a reasonable range

# Example usage
margins = np.array([0.1, 1.5, 3.0, 5.0, 10.0])  # Example margins in seconds
slider_value = 75  # User-selected value (0-100)

weights = compute_weights(margins, slider_value)
print(weights)


