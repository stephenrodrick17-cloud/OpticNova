import numpy as np

def calculate_area(mask: np.ndarray, pixel_spacing: float = 1.0) -> float:
    """
    Calculates the area from a binary mask.
    
    Args:
        mask (np.ndarray): Binary mask (0s and 1s).
        pixel_spacing (float): Physical distance between pixels if known (e.g., mm/pixel).
                               Defaults to 1.0 for pixel area.
                               
    Returns:
        float: Calculated area.
    """
    return np.sum(mask > 0) * (pixel_spacing ** 2)

def calculate_rim_area(disc_mask: np.ndarray, cup_mask: np.ndarray, pixel_spacing: float = 1.0) -> float:
    """
    Calculates the neuroretinal rim area (Disc Area - Cup Area).
    
    Args:
        disc_mask (np.ndarray): Binary mask for optic disc.
        cup_mask (np.ndarray): Binary mask for optic cup.
        pixel_spacing (float): Physical distance between pixels.
        
    Returns:
        float: Calculated rim area.
    """
    disc_area = calculate_area(disc_mask, pixel_spacing)
    cup_area = calculate_area(cup_mask, pixel_spacing)
    
    # Rim area is the difference
    rim_area = max(0.0, disc_area - cup_area)
    return rim_area
