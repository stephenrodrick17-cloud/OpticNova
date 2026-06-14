import numpy as np

def calculate_vertical_diameter(mask: np.ndarray) -> int:
    """
    Calculates the vertical diameter of a binary mask.
    """
    # Find rows that have at least one True value
    rows = np.any(mask, axis=1)
    if not np.any(rows):
        return 0
    # Get the indices of these rows
    ymin, ymax = np.where(rows)[0][[0, -1]]
    return ymax - ymin

def calculate_cdr(disc_mask: np.ndarray, cup_mask: np.ndarray) -> float:
    """
    Calculates the vertical Cup-to-Disc Ratio (vCDR).
    
    Args:
        disc_mask (np.ndarray): Binary mask for optic disc.
        cup_mask (np.ndarray): Binary mask for optic cup.
        
    Returns:
        float: Cup-to-Disc ratio.
    """
    cup_diameter = calculate_vertical_diameter(cup_mask)
    disc_diameter = calculate_vertical_diameter(disc_mask)
    
    if disc_diameter == 0:
        return 0.0
        
    return cup_diameter / disc_diameter
