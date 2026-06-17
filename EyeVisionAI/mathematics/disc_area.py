import numpy as np
import cv2


def calculate_area(mask: np.ndarray, pixel_spacing: float = 1.0) -> float:
    """
    Calculates the area from a binary mask.
    
    Mathematical Background:
        The area A of a binary image is given by:
        
        A = N * s²
        
        where N is the number of foreground pixels, and s is the pixel spacing.
        
        For a continuous region, this approximates the Lebesgue measure.
    
    Args:
        mask (np.ndarray): Binary mask (0s and 1s).
        pixel_spacing (float): Physical distance between pixels (e.g., mm/pixel).
                               Defaults to 1.0 for pixel area.
                               
    Returns:
        float: Calculated area.
    """
    return np.sum(mask > 0) * (pixel_spacing ** 2)


def calculate_perimeter(mask: np.ndarray, pixel_spacing: float = 1.0) -> float:
    """
    Calculates the perimeter of a binary mask.
    
    Mathematical Background:
        The perimeter P is calculated using the contour length. For a binary image,
        the perimeter is approximated as the length of the object's boundary.
        
        Using the chain code representation of contours, the perimeter is:
        
        P = s * Σ_{i=1 to M} L_i
        
        where L_i is the length of each chain code segment (1 for horizontal/vertical,
        √2 for diagonals), and M is the number of segments.
    
    Args:
        mask (np.ndarray): Binary mask.
        pixel_spacing (float): Physical distance between pixels.
        
    Returns:
        float: Perimeter length.
    """
    contours, _ = cv2.findContours(
        (mask > 0.5).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(contours) == 0:
        return 0.0
    
    perimeter = cv2.arcLength(contours[0], closed=True)
    return perimeter * pixel_spacing


def calculate_circularity(mask: np.ndarray) -> float:
    """
    Calculates the circularity (roundness) of a binary mask.
    
    Mathematical Background:
        Circularity C is defined as:
        
        C = (4πA) / P²
        
        where A is the area and P is the perimeter.
        
        A perfect circle has C = 1. Values less than 1 indicate irregular shapes.
    
    Args:
        mask (np.ndarray): Binary mask.
        
    Returns:
        float: Circularity score between 0 and 1.
    """
    area = calculate_area(mask)
    perimeter = calculate_perimeter(mask)
    
    if perimeter <= 0:
        return 0.0
    
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return min(1.0, max(0.0, circularity))


def calculate_solidity(mask: np.ndarray) -> float:
    """
    Calculates the solidity (convexity) of a binary mask.
    
    Mathematical Background:
        Solidity S is the ratio of the object area to its convex hull area:
        
        S = A_object / A_convex_hull
        
        A value of 1 indicates a perfectly convex shape.
    
    Args:
        mask (np.ndarray): Binary mask.
        
    Returns:
        float: Solidity score between 0 and 1.
    """
    contours, _ = cv2.findContours(
        (mask > 0.5).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(contours) == 0:
        return 0.0
    
    contour = contours[0]
    hull = cv2.convexHull(contour)
    
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(hull)
    
    if hull_area <= 0:
        return 0.0
    
    return area / hull_area


def calculate_eccentricity(mask: np.ndarray) -> float:
    """
    Calculates the eccentricity of a binary mask using image moments.
    
    Mathematical Background:
        Eccentricity ε is derived from the second-order central moments μ₂₀, μ₀₂, μ₁₁.
        
        The covariance matrix is:
            [ μ₂₀  μ₁₁ ]
            [ μ₁₁  μ₀₂ ]
        
        The eigenvalues λ₁ and λ₂ (λ₁ ≥ λ₂) of this matrix give the major and minor axes.
        
        ε = sqrt(1 - (λ₂ / λ₁))
        
        ε = 0 for a circle, ε approaches 1 for a very elongated ellipse.
    
    Args:
        mask (np.ndarray): Binary mask.
        
    Returns:
        float: Eccentricity between 0 and 1.
    """
    contours, _ = cv2.findContours(
        (mask > 0.5).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(contours) == 0:
        return 0.0
    
    contour = contours[0]
    
    if len(contour) < 5:
        return 0.0
    
    ellipse = cv2.fitEllipse(contour)
    major_axis = max(ellipse[1])
    minor_axis = min(ellipse[1])
    
    if major_axis <= 0:
        return 0.0
    
    eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
    return eccentricity


def fit_ellipse(mask: np.ndarray) -> dict:
    """
    Fits an ellipse to a binary mask and returns ellipse parameters.
    
    Mathematical Background:
        An ellipse can be represented by:
        - Center (c_x, c_y)
        - Major and minor axes lengths (a, b)
        - Rotation angle θ
        
        The general equation of an ellipse is:
        ( (x - c_x) cosθ + (y - c_y) sinθ )² / a² +
        ( -(x - c_x) sinθ + (y - c_y) cosθ )² / b² = 1
    
    Args:
        mask (np.ndarray): Binary mask.
        
    Returns:
        dict: Ellipse parameters.
    """
    contours, _ = cv2.findContours(
        (mask > 0.5).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(contours) == 0 or len(contours[0]) < 5:
        return {
            "center": (0.0, 0.0),
            "major_axis": 0.0,
            "minor_axis": 0.0,
            "angle": 0.0,
            "area": 0.0,
            "perimeter": 0.0
        }
    
    ellipse = cv2.fitEllipse(contours[0])
    
    center = ellipse[0]
    axes = ellipse[1]
    angle = ellipse[2]
    
    major_axis = max(axes)
    minor_axis = min(axes)
    
    ellipse_area = np.pi * (major_axis / 2) * (minor_axis / 2)
    ellipse_perimeter = np.pi * (3 * (major_axis + minor_axis) / 2 - 
                                  np.sqrt((3 * major_axis + minor_axis) * 
                                          (major_axis + 3 * minor_axis))) / 2
    
    return {
        "center": (round(center[0], 2), round(center[1], 2)),
        "major_axis": round(major_axis, 2),
        "minor_axis": round(minor_axis, 2),
        "axis_ratio": round(minor_axis / major_axis, 4) if major_axis > 0 else 0.0,
        "angle": round(angle, 2),
        "area": round(ellipse_area, 2),
        "perimeter": round(ellipse_perimeter, 2),
        "eccentricity": round(calculate_eccentricity(mask), 4)
    }


def calculate_moments(mask: np.ndarray) -> dict:
    """
    Calculates image moments up to order 3.
    
    Mathematical Background:
        The raw moment M_ij of order i+j is:
            M_ij = Σ_x Σ_y x^i y^j I(x, y)
        
        Central moments μ_ij are translation-invariant:
            μ_ij = Σ_x Σ_y (x - x̄)^i (y - ȳ)^j I(x, y)
            where x̄ = M₁₀ / M₀₀, ȳ = M₀₁ / M₀₀
        
        Normalized central moments η_ij are scale-invariant:
            η_ij = μ_ij / M₀₀^((i+j)/2 + 1)
        
        Hu invariants are rotation, translation, and scale invariant.
    
    Args:
        mask (np.ndarray): Binary mask.
        
    Returns:
        dict: Image moments and Hu invariants.
    """
    moments = cv2.moments((mask > 0.5).astype(np.uint8))
    
    if moments["m00"] == 0:
        return {
            "centroid": (0.0, 0.0),
            "hu_moments": [0.0] * 7
        }
    
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    
    hu_moments = cv2.HuMoments(moments).flatten()
    
    return {
        "centroid": (round(cx, 2), round(cy, 2)),
        "hu_moments": [round(h, 6) for h in hu_moments]
    }


def calculate_rim_area(disc_mask: np.ndarray, cup_mask: np.ndarray, pixel_spacing: float = 1.0) -> dict:
    """
    Calculates comprehensive neuroretinal rim metrics.
    
    Mathematical Background:
        Rim area A_rim = A_disc - A_cup
        
        Rim volume (approximate): V_rim ≈ A_rim * t_avg
        where t_avg is average rim thickness.
    
    Args:
        disc_mask (np.ndarray): Binary mask for optic disc.
        cup_mask (np.ndarray): Binary mask for optic cup.
        pixel_spacing (float): Physical distance between pixels.
        
    Returns:
        dict: Comprehensive rim metrics.
    """
    disc_area = calculate_area(disc_mask, pixel_spacing)
    disc_perim = calculate_perimeter(disc_mask, pixel_spacing)
    disc_circ = calculate_circularity(disc_mask)
    disc_ellipse = fit_ellipse(disc_mask)
    
    cup_area = calculate_area(cup_mask, pixel_spacing)
    cup_perim = calculate_perimeter(cup_mask, pixel_spacing)
    cup_circ = calculate_circularity(cup_mask)
    cup_ellipse = fit_ellipse(cup_mask)
    
    rim_area = max(0.0, disc_area - cup_area)
    
    rim_area_ratio = rim_area / disc_area if disc_area > 0 else 0.0
    
    return {
        "rim_area": round(rim_area, 2),
        "rim_area_ratio": round(rim_area_ratio, 4),
        "disc": {
            "area": round(disc_area, 2),
            "perimeter": round(disc_perim, 2),
            "circularity": round(disc_circ, 4),
            "solidity": round(calculate_solidity(disc_mask), 4),
            "ellipse": disc_ellipse
        },
        "cup": {
            "area": round(cup_area, 2),
            "perimeter": round(cup_perim, 2),
            "circularity": round(cup_circ, 4),
            "solidity": round(calculate_solidity(cup_mask), 4),
            "ellipse": cup_ellipse
        }
    }


def calculate_shape_descriptors(mask: np.ndarray) -> dict:
    """
    Calculates a comprehensive set of shape descriptors.
    
    Args:
        mask (np.ndarray): Binary mask.
        
    Returns:
        dict: Shape descriptors.
    """
    area = calculate_area(mask)
    perimeter = calculate_perimeter(mask)
    circularity = calculate_circularity(mask)
    solidity = calculate_solidity(mask)
    eccentricity = calculate_eccentricity(mask)
    ellipse = fit_ellipse(mask)
    moments = calculate_moments(mask)
    
    rect_similarity = 0.0
    contours, _ = cv2.findContours(
        (mask > 0.5).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) > 0:
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box_area = cv2.contourArea(box)
        rect_similarity = area / box_area if box_area > 0 else 0.0
    
    return {
        "area": round(area, 2),
        "perimeter": round(perimeter, 2),
        "circularity": round(circularity, 4),
        "solidity": round(solidity, 4),
        "eccentricity": round(eccentricity, 4),
        "rectangularity": round(rect_similarity, 4),
        "ellipse": ellipse,
        "moments": moments
    }
