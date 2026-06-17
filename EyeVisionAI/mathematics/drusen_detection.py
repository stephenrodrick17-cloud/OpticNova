import cv2
import numpy as np
from scipy import ndimage


def detect_macular_drusen(image: np.ndarray, disc_mask: np.ndarray, eye_side: str = "Right Eye") -> dict:
    """
    Locates the estimated macular region relative to the optic disc centroid
    and segments potential yellow drusen deposits (indicators of AMD) with
    comprehensive mathematical analysis.
    
    Mathematical Background for Drusen Analysis:
        Drusen are extracellular deposits that appear as yellowish spots.
        They are characterized by:
        1. High intensity in red and green channels
        2. Low intensity in blue channel
        3. Round/oval shape
        4. Variable size (small: <63μm, medium: 63-125μm, large: >125μm)
    
    Args:
        image (np.ndarray): Enhanced RGB fundus scan (512x512).
        disc_mask (np.ndarray): Optic disc mask to reference position.
        eye_side (str): "Right Eye" (OD) or "Left Eye" (OS).
        
    Returns:
        dict: Comprehensive drusen analysis.
    """
    h, w, _ = image.shape
    
    # 1. Determine Optic Disc Centroid
    disc_indices = np.argwhere(disc_mask > 0.5)
    if len(disc_indices) > 0:
        dy, dx = np.mean(disc_indices, axis=0)
        disc_radius = np.sqrt(len(disc_indices) / np.pi)
    else:
        dy, dx = h // 2, w // 2
        disc_radius = 50.0
        
    # 2. Locate the Macula (Optic disc temporal offsets)
    temporal_sign = 1 if eye_side == "Right Eye" else -1
    macula_cx = int(dx + temporal_sign * (5.0 * disc_radius))
    macula_cy = int(dy)
    
    macula_cx = max(50, min(w - 50, macula_cx))
    macula_cy = max(50, min(h - 50, macula_cy))
    macula_radius = int(2.2 * disc_radius)
    
    # 3. Create Macula ROI mask
    macula_roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(macula_roi_mask, (macula_cx, macula_cy), macula_radius, 255, -1)
    
    # 4. Multi-channel drusen enhancement
    drusen_mask = multi_channel_drusen_enhancement(image, macula_roi_mask)
    
    # 5. Analyze individual drusen
    drusen_analysis = analyze_drusen_particles(drusen_mask, macula_roi_mask)
    
    # 6. Calculate severity metrics
    severity = calculate_drusen_severity(drusen_analysis)
    
    return {
        "macula_center": (macula_cx, macula_cy),
        "macula_radius": macula_radius,
        "drusen_mask": drusen_mask,
        "drusen_analysis": drusen_analysis,
        "severity": severity
    }


def multi_channel_drusen_enhancement(image: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """
    Enhances drusen using multiple color space transformations.
    
    Mathematical Background:
        We use:
        1. RGB space: Y1 = (R + G)/2 - B
        2. HSV space: Y2 = S * (1 - V)
        3. Lab space: Y3 = a*
        
        Final enhancement: Y = w1*Y1 + w2*Y2 + w3*Y3
        where w1 = 0.5, w2 = 0.3, w3 = 0.2
    """
    h, w, _ = image.shape
    
    # RGB enhancement
    r, g, b = cv2.split(image)
    y1 = cv2.addWeighted(r, 0.5, g, 0.5, 0)
    y1 = cv2.subtract(y1, b)
    
    # HSV enhancement
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_hsv, s, v = cv2.split(hsv)
    y2 = cv2.multiply(s, cv2.subtract(255, v))
    y2 = cv2.normalize(y2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Lab enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b_lab = cv2.split(lab)
    y3 = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Weighted combination
    enhanced = cv2.addWeighted(y1, 0.5, y2, 0.3, 0)
    enhanced = cv2.addWeighted(enhanced, 1.0, y3, 0.2, 0)
    
    # Apply ROI mask
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=roi_mask)
    
    # Adaptive thresholding
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, -10
    )
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return cleaned


def analyze_drusen_particles(drusen_mask: np.ndarray, roi_mask: np.ndarray) -> dict:
    """
    Analyzes individual drusen particles using mathematical morphology.
    
    Mathematical Background:
        For each drusen:
        - Area: A = πr² (for circle approximation)
        - Circularity: C = 4πA/P²
        - Aspect ratio: AR = major_axis/minor_axis
        - Solidarity: S = A_object/A_convex_hull
    """
    contours, hierarchy = cv2.findContours(
        drusen_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    particles = []
    total_area = 0
    small_count = 0
    medium_count = 0
    large_count = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < 5:
            continue
        
        perimeter = cv2.arcLength(contour, closed=True)
        
        if perimeter == 0:
            continue
        
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            aspect_ratio = minor_axis / major_axis if major_axis > 0 else 0
        else:
            major_axis = np.sqrt(area / np.pi) * 2
            minor_axis = major_axis
            aspect_ratio = 1.0
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        
        diameter_eq = 2 * np.sqrt(area / np.pi)
        if diameter_eq < 20:
            small_count += 1
        elif diameter_eq < 40:
            medium_count += 1
        else:
            large_count += 1
        
        particles.append({
            "area": round(area, 2),
            "perimeter": round(perimeter, 2),
            "circularity": round(circularity, 4),
            "aspect_ratio": round(aspect_ratio, 4),
            "solidity": round(solidity, 4),
            "center": (cx, cy),
            "major_axis": round(major_axis, 2),
            "minor_axis": round(minor_axis, 2),
            "equivalent_diameter": round(diameter_eq, 2)
        })
        
        total_area += area
    
    total_macula_pixels = np.sum(roi_mask > 0)
    drusen_area_ratio = total_area / total_macula_pixels if total_macula_pixels > 0 else 0
    
    areas = [p["area"] for p in particles]
    if len(areas) > 0:
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        max_area = np.max(areas)
    else:
        mean_area = 0
        std_area = 0
        max_area = 0
    
    return {
        "count": len(particles),
        "total_area": round(total_area, 2),
        "mean_area": round(mean_area, 2),
        "std_area": round(std_area, 2),
        "max_area": round(max_area, 2),
        "area_ratio": round(drusen_area_ratio, 4),
        "size_distribution": {
            "small": small_count,
            "medium": medium_count,
            "large": large_count
        },
        "particles": particles
    }


def calculate_drusen_severity(analysis: dict) -> dict:
    """
    Calculates drusen severity using a weighted scoring system.
    
    Mathematical Background:
        Severity Score S = w1*C + w2*A + w3*N
        
        where:
        - C = 1 - exp(-(count/5)^2) [count component]
        - A = 1 - exp(-(area_ratio/0.1)^2) [area component]
        - N = (N_large + 0.5*N_medium) / (N_total + 1) [size component]
        - w1 = 0.3, w2 = 0.4, w3 = 0.3
    """
    count = analysis["count"]
    area_ratio = analysis["area_ratio"]
    size_dist = analysis["size_distribution"]
    
    count_component = 1.0 - np.exp(-(count / 5.0) ** 2)
    area_component = 1.0 - np.exp(-(area_ratio / 0.1) ** 2)
    
    n_large = size_dist["large"]
    n_medium = size_dist["medium"]
    n_total = count if count > 0 else 1
    size_component = (n_large + 0.5 * n_medium) / n_total
    
    w1, w2, w3 = 0.3, 0.4, 0.3
    severity_score = w1 * count_component + w2 * area_component + w3 * size_component
    severity_score = min(1.0, max(0.0, severity_score))
    
    if severity_score < 0.1:
        category = "No Drusen"
    elif severity_score < 0.3:
        category = "Mild"
    elif severity_score < 0.6:
        category = "Moderate"
    else:
        category = "Severe"
    
    return {
        "score": round(severity_score, 4),
        "category": category,
        "components": {
            "count": round(count_component, 4),
            "area": round(area_component, 4),
            "size": round(size_component, 4)
        }
    }


def calculate_drusen_density_map(drusen_mask: np.ndarray, roi_mask: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """
    Creates a density map of drusen using Gaussian kernel density estimation.
    
    Mathematical Background:
        Density D(x, y) = Σ_{i=1 to N} K_σ(x - x_i, y - y_i)
        where K_σ is a 2D Gaussian kernel with bandwidth σ.
    """
    h, w = drusen_mask.shape
    
    density_map = np.zeros((h, w), dtype=np.float32)
    
    contours, _ = cv2.findContours(
        drusen_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            area = cv2.contourArea(contour)
            weight = np.sqrt(area)
            
            x, y = np.ogrid[0:h, 0:w]
            gaussian = np.exp(-((y - cx) ** 2 + (x - cy) ** 2) / (2 * sigma ** 2))
            density_map += weight * gaussian
    
    density_map = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    density_map = cv2.bitwise_and(density_map, density_map, mask=roi_mask)
    
    return density_map


def drusen_spatial_statistics(analysis: dict, macula_center: tuple, macula_radius: float) -> dict:
    """
    Calculates spatial statistics of drusen distribution.
    
    Mathematical Background:
        - Mean distance from fovea: μ = (1/N) Σ ||p_i - f||
        - Standard distance: σ = sqrt( (1/N) Σ (||p_i - f|| - μ)^2 )
        - Ripley's K-function (simplified)
    """
    particles = analysis["particles"]
    
    if len(particles) == 0:
        return {
            "mean_distance": 0.0,
            "std_distance": 0.0,
            "dispersion": 0.0
        }
    
    distances = []
    fx, fy = macula_center
    
    for p in particles:
        px, py = p["center"]
        dist = np.sqrt((px - fx) ** 2 + (py - fy) ** 2)
        distances.append(dist)
    
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    dispersion = std_dist / mean_dist if mean_dist > 0 else 0
    
    return {
        "mean_distance": round(mean_dist, 2),
        "std_distance": round(std_dist, 2),
        "dispersion_index": round(dispersion, 4)
    }
