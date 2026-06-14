import cv2
import numpy as np

def extract_blood_vessels(image: np.ndarray) -> np.ndarray:
    """
    Extracts blood vessels from a fundus image using a high-precision green-channel
    CLAHE enhancement and adaptive thresholding process.
    
    Args:
        image (np.ndarray): Input RGB fundus image.
        
    Returns:
        np.ndarray: Binary vessel mask (255 for vessel, 0 for background).
    """
    # 1. Extract green channel (highest contrast for blood vessels)
    green = image[:, :, 1]
    
    # 2. Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(green)
    
    # 3. Apply background exclusion (subtract average smoothed background to remove illumination variance)
    background = cv2.medianBlur(contrast_enhanced, 25)
    subtracted = cv2.subtract(background, contrast_enhanced)
    
    # 4. Filter high-frequency noise and enhance lines via Bilateral filter
    filtered = cv2.bilateralFilter(subtracted, 9, 75, 75)
    
    # 5. Thresholding (Adaptive thresholding gives excellent detailed extraction)
    vessels_binary = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 15, -2
    )
    
    # 6. Post-processing: Remove small isolated noise pixels using morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_vessels = cv2.morphologyEx(vessels_binary, cv2.MORPH_OPEN, kernel)
    
    # Additionally mask out the boundary artifact of the circular fundus mask if any
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    # Erode the boundary mask to avoid edge artifacts
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    eroded_mask = cv2.erode(mask, kernel_erode)
    
    final_vessels = cv2.bitwise_and(cleaned_vessels, eroded_mask)
    return final_vessels

def calculate_vessel_density(vessel_mask: np.ndarray, roi_mask: np.ndarray = None) -> float:
    """
    Calculates blood vessel density (percentage of vessel pixels in the ROI).
    If no ROI is provided, calculates density over the entire active fundus region.
    """
    if roi_mask is not None:
        total_pixels = np.sum(roi_mask > 0)
        vessel_pixels = np.sum((vessel_mask > 0) & (roi_mask > 0))
    else:
        # Assuming non-black region is active fundus
        total_pixels = np.sum(vessel_mask >= 0)
        vessel_pixels = np.sum(vessel_mask > 0)
        
    if total_pixels == 0:
        return 0.0
    return float(vessel_pixels / total_pixels)

def calculate_vessel_tortuosity(vessel_mask: np.ndarray) -> dict:
    """
    Finds individual vessel segments, traces their skeleton paths, and calculates
    Tortuosity T = L_curve / L_chord.
    
    Returns:
        dict: Mean tortuosity, max tortuosity, and segment counts.
    """
    # Skeletonize to get single-pixel width lines
    skeleton = cv2.ximgproc.thinning(vessel_mask) if hasattr(cv2, 'ximgproc') else vessel_mask
    
    # Find contours/segments of the skeleton
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    tortuosities = []
    
    for cnt in contours:
        # We only consider sufficiently long vessel segments to avoid noise calculations
        arc_len = cv2.arcLength(cnt, closed=False)
        if arc_len < 30.0: # pixels
            continue
            
        # Get start and end points of the contour to calculate straight line chord length
        p_start = cnt[0][0]
        p_end = cnt[-1][0]
        chord_len = np.linalg.norm(p_start - p_end)
        
        if chord_len > 5.0: # avoid division by zero
            # T = Arc Length / Chord Length
            t = arc_len / chord_len
            # Clamp extreme outliers due to loops
            if t < 5.0:
                tortuosities.append(t)
                
    if len(tortuosities) == 0:
        return {"mean_tortuosity": 1.0, "max_tortuosity": 1.0, "segments": 0}
        
    return {
        "mean_tortuosity": float(np.mean(tortuosities)),
        "max_tortuosity": float(np.max(tortuosities)),
        "segments": len(tortuosities)
    }
