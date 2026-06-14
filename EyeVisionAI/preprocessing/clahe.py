import cv2
import numpy as np

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.
    Works well for retinal fundus images to enhance vessel and optic disc visibility.
    
    Args:
        image (np.ndarray): The input RGB image.
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (tuple): Size of grid for histogram equalization.
        
    Returns:
        np.ndarray: CLAHE-enhanced RGB image.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Split the LAB channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L-channel back with A and B channels
    limg = cv2.merge((cl, a, b))
    
    # Convert LAB back to RGB
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    return enhanced_image
