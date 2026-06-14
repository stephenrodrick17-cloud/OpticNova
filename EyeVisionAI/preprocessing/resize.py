import cv2
import numpy as np

def resize_image(image: np.ndarray, target_size: tuple = (512, 512)) -> np.ndarray:
    """
    Resizes an image to the target size.
    
    Args:
        image (np.ndarray): The input image.
        target_size (tuple): The desired output size (width, height).
        
    Returns:
        np.ndarray: The resized image.
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def pad_image_to_square(image: np.ndarray, fill_value: int = 0) -> np.ndarray:
    """
    Pads an image to a square aspect ratio.
    """
    h, w = image.shape[:2]
    if h == w:
        return image
    
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    
    if len(image.shape) == 3:
        padded = np.full((size, size, image.shape[2]), fill_value, dtype=image.dtype)
        padded[pad_h:pad_h+h, pad_w:pad_w+w, :] = image
    else:
        padded = np.full((size, size), fill_value, dtype=image.dtype)
        padded[pad_h:pad_h+h, pad_w:pad_w+w] = image
        
    return padded
