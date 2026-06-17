import numpy as np
import cv2


def calculate_vertical_diameter(mask: np.ndarray) -> int:
    """
    Calculates the vertical diameter of a binary mask.
    
    Mathematical Background:
        The vertical diameter D_v is defined as the distance between the topmost and 
        bottommost pixels of the binary segmentation mask.
        
        D_v = y_max - y_min
        
        where y_min = argmin{ y | ∃x, mask[y, x] = 1 },
              y_max = argmax{ y | ∃x, mask[y, x] = 1 }
    
    Args:
        mask (np.ndarray): Binary segmentation mask.
        
    Returns:
        int: Vertical diameter in pixels.
    """
    rows = np.any(mask, axis=1)
    if not np.any(rows):
        return 0
    ymin, ymax = np.where(rows)[0][[0, -1]]
    return ymax - ymin


def calculate_horizontal_diameter(mask: np.ndarray) -> int:
    """
    Calculates the horizontal diameter of a binary mask.
    
    Mathematical Background:
        The horizontal diameter D_h is defined as the distance between the leftmost and 
        rightmost pixels of the binary segmentation mask.
        
        D_h = x_max - x_min
        
        where x_min = argmin{ x | ∃y, mask[y, x] = 1 },
              x_max = argmax{ x | ∃y, mask[y, x] = 1 }
    
    Args:
        mask (np.ndarray): Binary segmentation mask.
        
    Returns:
        int: Horizontal diameter in pixels.
    """
    cols = np.any(mask, axis=0)
    if not np.any(cols):
        return 0
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return xmax - xmin


def calculate_centroid(mask: np.ndarray) -> tuple:
    """
    Calculates the centroid (center of mass) of a binary mask.
    
    Mathematical Background:
        The centroid (c_x, c_y) of a binary region is given by:
        
        c_x = (1 / N) * Σ_{i=1 to N} x_i
        c_y = (1 / N) * Σ_{i=1 to N} y_i
        
        where N is the number of pixels in the mask, and (x_i, y_i) are the 
        coordinates of each pixel in the mask.
    
    Args:
        mask (np.ndarray): Binary mask.
        
    Returns:
        tuple: (cx, cy) centroid coordinates.
    """
    coords = np.argwhere(mask > 0.5)
    if len(coords) == 0:
        return (0.0, 0.0)
    cy, cx = np.mean(coords, axis=0)
    return (cx, cy)


def calculate_max_diameter(mask: np.ndarray) -> tuple:
    """
    Calculates the maximum diameter (Feret diameter) between any two points on the mask boundary.
    
    Mathematical Background:
        The maximum Feret diameter D_max is the maximum distance between any two 
        points on the boundary of the object. It is found using the rotating calipers method.
        
        D_max = max_{p,q ∈ ∂M} ||p - q||₂
        
        where ∂M is the boundary of the mask M, and ||·||₂ is the Euclidean norm.
    
    Args:
        mask (np.ndarray): Binary mask.
        
    Returns:
        tuple: (max_diameter, angle, point1, point2)
    """
    coords = np.argwhere(mask > 0.5)
    if len(coords) < 2:
        return (0.0, 0.0, (0, 0), (0, 0))
    
    contours, _ = cv2.findContours(
        (mask > 0.5).astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(contours) == 0:
        return (0.0, 0.0, (0, 0), (0, 0))
    
    contour = contours[0]
    max_dist = 0.0
    best_p1 = (0, 0)
    best_p2 = (0, 0)
    
    for i in range(len(contour)):
        p1 = contour[i][0]
        for j in range(i + 1, len(contour)):
            p2 = contour[j][0]
            dist = np.linalg.norm(p1 - p2)
            if dist > max_dist:
                max_dist = dist
                best_p1 = tuple(p1)
                best_p2 = tuple(p2)
    
    angle = np.arctan2(best_p2[1] - best_p1[1], best_p2[0] - best_p1[0]) * 180 / np.pi
    return (max_dist, angle, best_p1, best_p2)


def calculate_cdr(disc_mask: np.ndarray, cup_mask: np.ndarray) -> dict:
    """
    Calculates comprehensive Cup-to-Disc Ratio (CDR) metrics including:
    - Vertical CDR (vCDR)
    - Horizontal CDR (hCDR)
    - Area-based CDR (aCDR)
    - Diameter-based CDR (dCDR)
    - Confidence estimates for each metric
    
    Mathematical Background:
        
        1. Vertical CDR:
            vCDR = D_v^cup / D_v^disc
        
        2. Horizontal CDR:
            hCDR = D_h^cup / D_h^disc
        
        3. Area-based CDR:
            aCDR = A_cup / A_disc
        
        4. Mean CDR:
            mCDR = (vCDR + hCDR) / 2
        
        where D_v and D_h are vertical and horizontal diameters,
        A_cup and A_disc are cup and disc areas.
        
    Clinical Relevance:
        A vCDR > 0.6 is generally considered suspicious for glaucoma.
        The ISNT rule complements CDR for more robust diagnosis.
    
    Args:
        disc_mask (np.ndarray): Binary mask for optic disc.
        cup_mask (np.ndarray): Binary mask for optic cup.
        
    Returns:
        dict: Comprehensive CDR metrics with confidence estimates.
    """
    v_cup = calculate_vertical_diameter(cup_mask)
    v_disc = calculate_vertical_diameter(disc_mask)
    
    h_cup = calculate_horizontal_diameter(cup_mask)
    h_disc = calculate_horizontal_diameter(disc_mask)
    
    a_cup = np.sum(cup_mask > 0.5)
    a_disc = np.sum(disc_mask > 0.5)
    
    d_cup, _, _, _ = calculate_max_diameter(cup_mask)
    d_disc, _, _, _ = calculate_max_diameter(disc_mask)
    
    vcdr = v_cup / v_disc if v_disc > 0 else 0.0
    hcdr = h_cup / h_disc if h_disc > 0 else 0.0
    acdr = a_cup / a_disc if a_disc > 0 else 0.0
    dcdr = d_cup / d_disc if d_disc > 0 else 0.0
    mcdr = (vcdr + hcdr) / 2.0
    
    conf_vcdr = calculate_cdr_confidence(v_disc, v_cup, "vertical")
    conf_hcdr = calculate_cdr_confidence(h_disc, h_cup, "horizontal")
    conf_acdr = calculate_cdr_confidence(a_disc, a_cup, "area")
    
    return {
        "vertical_cdr": round(vcdr, 4),
        "horizontal_cdr": round(hcdr, 4),
        "area_cdr": round(acdr, 4),
        "max_diameter_cdr": round(dcdr, 4),
        "mean_cdr": round(mcdr, 4),
        "confidence": {
            "vertical": round(conf_vcdr, 4),
            "horizontal": round(conf_hcdr, 4),
            "area": round(conf_acdr, 4),
            "overall": round((conf_vcdr + conf_hcdr + conf_acdr) / 3.0, 4)
        },
        "dimensions": {
            "disc": {
                "vertical": v_disc,
                "horizontal": h_disc,
                "area": a_disc,
                "max_diameter": round(d_disc, 2)
            },
            "cup": {
                "vertical": v_cup,
                "horizontal": h_cup,
                "area": a_cup,
                "max_diameter": round(d_cup, 2)
            }
        }
    }


def calculate_cdr_confidence(disc_measure: float, cup_measure: float, measure_type: str) -> float:
    """
    Calculates confidence score for CDR measurements.
    
    Mathematical Background:
        The confidence C is a function of:
        1. Absolute size of the disc (larger discs have more stable measurements)
        2. Ratio of cup to disc (avoid edge cases near 0 or 1)
        
        C = f(D_disc) * g(r), where r = D_cup / D_disc
        
        f(D_disc) = 1 - exp(-(D_disc / D_0)^2)
        g(r) = 4 * r * (1 - r)
        
        where D_0 is a scaling constant (40 pixels).
    
    Args:
        disc_measure (float): Disc measurement.
        cup_measure (float): Cup measurement.
        measure_type (str): Type of measurement.
        
    Returns:
        float: Confidence score between 0 and 1.
    """
    if disc_measure <= 0:
        return 0.0
    
    D0 = 40.0
    size_conf = 1.0 - np.exp(-(disc_measure / D0) ** 2)
    
    ratio = cup_measure / disc_measure
    ratio_conf = 4.0 * ratio * (1.0 - ratio)
    
    confidence = 0.7 * size_conf + 0.3 * ratio_conf
    return min(1.0, max(0.0, confidence))


def calculate_cdr_uncertainty(disc_mask: np.ndarray, cup_mask: np.ndarray, num_bootstraps: int = 100) -> dict:
    """
    Estimates CDR uncertainty using bootstrap resampling.
    
    Mathematical Background:
        Bootstrap uncertainty estimation involves:
        1. Sampling with replacement from the boundary pixels
        2. Recalculating CDR for each bootstrap sample
        3. Computing the standard deviation of bootstrap estimates
        
        σ_CDR = sqrt( (1/(B-1)) * Σ_{b=1 to B} (CDR_b - μ_CDR)^2 )
        
        where B is the number of bootstrap samples, CDR_b is the CDR for
        bootstrap sample b, and μ_CDR is the mean CDR across all samples.
    
    Args:
        disc_mask (np.ndarray): Binary disc mask.
        cup_mask (np.ndarray): Binary cup mask.
        num_bootstraps (int): Number of bootstrap samples.
        
    Returns:
        dict: Uncertainty estimates.
    """
    disc_coords = np.argwhere(disc_mask > 0.5)
    cup_coords = np.argwhere(cup_mask > 0.5)
    
    if len(disc_coords) < 10 or len(cup_coords) < 10:
        return {
            "vcdr_std": 0.0,
            "hcdr_std": 0.0,
            "acdr_std": 0.0,
            "vcdr_ci": (0.0, 0.0),
            "hcdr_ci": (0.0, 0.0),
            "acdr_ci": (0.0, 0.0)
        }
    
    bootstrap_vcdr = []
    bootstrap_hcdr = []
    bootstrap_acdr = []
    
    for _ in range(num_bootstraps):
        idx_disc = np.random.choice(len(disc_coords), size=len(disc_coords), replace=True)
        idx_cup = np.random.choice(len(cup_coords), size=len(cup_coords), replace=True)
        
        boot_disc_coords = disc_coords[idx_disc]
        boot_cup_coords = cup_coords[idx_cup]
        
        h, w = disc_mask.shape
        boot_disc = np.zeros((h, w), dtype=bool)
        boot_cup = np.zeros((h, w), dtype=bool)
        
        for coord in boot_disc_coords:
            if 0 <= coord[0] < h and 0 <= coord[1] < w:
                boot_disc[coord[0], coord[1]] = True
        
        for coord in boot_cup_coords:
            if 0 <= coord[0] < h and 0 <= coord[1] < w:
                boot_cup[coord[0], coord[1]] = True
        
        cdr_result = calculate_cdr(boot_disc, boot_cup)
        bootstrap_vcdr.append(cdr_result["vertical_cdr"])
        bootstrap_hcdr.append(cdr_result["horizontal_cdr"])
        bootstrap_acdr.append(cdr_result["area_cdr"])
    
    bootstrap_vcdr = np.array(bootstrap_vcdr)
    bootstrap_hcdr = np.array(bootstrap_hcdr)
    bootstrap_acdr = np.array(bootstrap_acdr)
    
    return {
        "vcdr_std": round(np.std(bootstrap_vcdr), 4),
        "hcdr_std": round(np.std(bootstrap_hcdr), 4),
        "acdr_std": round(np.std(bootstrap_acdr), 4),
        "vcdr_ci": (
            round(np.percentile(bootstrap_vcdr, 2.5), 4),
            round(np.percentile(bootstrap_vcdr, 97.5), 4)
        ),
        "hcdr_ci": (
            round(np.percentile(bootstrap_hcdr, 2.5), 4),
            round(np.percentile(bootstrap_hcdr, 97.5), 4)
        ),
        "acdr_ci": (
            round(np.percentile(bootstrap_acdr, 2.5), 4),
            round(np.percentile(bootstrap_acdr, 97.5), 4)
        )
    }
