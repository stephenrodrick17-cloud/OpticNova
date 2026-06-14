import cv2
import numpy as np

def detect_macular_drusen(image: np.ndarray, disc_mask: np.ndarray, eye_side: str = "Right Eye") -> dict:
    """
    Locates the estimated macular region relative to the optic disc centroid
    and segments potential yellow drusen deposits (indicators of AMD).
    
    Args:
        image (np.ndarray): Enhanced RGB fundus scan (512x512).
        disc_mask (np.ndarray): Optic disc mask to reference position.
        eye_side (str): "Right Eye" (OD) or "Left Eye" (OS).
        
    Returns:
        dict: Macular ROI circle coordinates, segmented drusen mask, and Drusen Area Ratio (DAR).
    """
    h, w, _ = image.shape
    
    # 1. Determine Optic Disc Centroid
    disc_indices = np.argwhere(disc_mask > 0.5)
    if len(disc_indices) > 0:
        dy, dx = np.mean(disc_indices, axis=0)
        disc_radius = np.sqrt(len(disc_indices) / np.pi)
    else:
        # Fallback to center if disc is missing
        dy, dx = h // 2, w // 2
        disc_radius = 50.0
        
    # 2. Locate the Macula (Optic disc temporal offsets)
    # Right Eye (OD): The macula is located temporarily (to the right of the optic disc).
    # Left Eye (OS): The macula is located temporarily (to the left of the optic disc).
    # Distance is roughly 2.5 times the disc diameter (~5 times disc radius).
    temporal_sign = 1 if eye_side == "Right Eye" else -1
    macula_cx = int(dx + temporal_sign * (5.0 * disc_radius))
    macula_cy = int(dy)
    
    # Bound within frame limits
    macula_cx = max(50, min(w - 50, macula_cx))
    macula_cy = max(50, min(h - 50, macula_cy))
    
    # Macular region of interest radius (roughly twice the disc radius)
    macula_radius = int(2.2 * disc_radius)
    
    # 3. Create Macula ROI mask
    macula_roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(macula_roi_mask, (macula_cx, macula_cy), macula_radius, 255, -1)
    
    # 4. Drusen Segmentation using color characteristics
    # Drusen deposits are bright yellow (high red and green, low blue values)
    # Subtraction method: (Red + Green)/2 - Blue enhances yellow deposits
    r, g, b = cv2.split(image)
    yellow_enhancement = cv2.addWeighted(r, 0.5, g, 0.5, 0)
    yellow_enhancement = cv2.subtract(yellow_enhancement, b)
    
    # Mask out only the macular region to run localized segmentations
    localized_yellow = cv2.bitwise_and(yellow_enhancement, yellow_enhancement, mask=macula_roi_mask)
    
    # Localized adaptive thresholding to detect drusen spots
    # Drusen are small discrete spots. An adaptive threshold isolates them from the background retina color.
    blurred = cv2.GaussianBlur(localized_yellow, (5, 5), 0)
    drusen_binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 21, -8
    )
    
    # Mask the final thresholded output strictly inside the Macula ROI
    drusen_final = cv2.bitwise_and(drusen_binary, macula_roi_mask)
    
    # Calculate Drusen Area Ratio (DAR)
    total_macula_pixels = np.sum(macula_roi_mask > 0)
    drusen_pixels = np.sum(drusen_final > 0)
    
    dar = float(drusen_pixels / total_macula_pixels) if total_macula_pixels > 0 else 0.0
    
    return {
        "macula_center": (macula_cx, macula_cy),
        "macula_radius": macula_radius,
        "drusen_mask": drusen_final,
        "drusen_area_ratio": dar
    }
