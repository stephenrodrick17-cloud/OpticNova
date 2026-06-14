import cv2
import numpy as np

def overlay_heatmap(original_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5, colormap=cv2.COLORMAP_JET) -> np.ndarray:
    """
    Overlays a Grad-CAM heatmap on the original image.
    
    Args:
        original_image (np.ndarray): Original RGB image (H, W, 3) in uint8 format (0-255).
        heatmap (np.ndarray): Grad-CAM heatmap (H', W') in float format (0-1).
        alpha (float): Overlay transparency.
        colormap (int): OpenCV colormap to use.
        
    Returns:
        np.ndarray: Blended image.
    """
    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # Convert heatmap to uint8
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Convert BGR (OpenCV default) to RGB for heatmap
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Blend the original image and the heatmap
    blended = cv2.addWeighted(original_image, 1 - alpha, heatmap_color, alpha, 0)
    
    return blended
