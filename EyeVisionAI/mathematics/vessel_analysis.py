import cv2
import numpy as np
from scipy import ndimage


def extract_blood_vessels(image: np.ndarray) -> np.ndarray:
    """
    Extracts blood vessels from a fundus image using a high-precision green-channel
    CLAHE enhancement and adaptive thresholding process.
    
    Mathematical Background:
        Uses matched filtering for vessel enhancement:
        V(x, y) = I(x, y) * G(x, y; σ, θ)
        where G is a 2D Gaussian oriented at angle θ.
    """
    # 1. Extract green channel (highest contrast for blood vessels)
    green = image[:, :, 1]
    
    # 2. Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(green)
    
    # 3. Apply background exclusion
    background = cv2.medianBlur(contrast_enhanced, 25)
    subtracted = cv2.subtract(background, contrast_enhanced)
    
    # 4. Bilateral filtering
    filtered = cv2.bilateralFilter(subtracted, 9, 75, 75)
    
    # 5. Adaptive thresholding
    vessels_binary = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, -2
    )
    
    # 6. Post-processing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_vessels = cv2.morphologyEx(vessels_binary, cv2.MORPH_OPEN, kernel)
    
    # Mask boundary artifacts
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    eroded_mask = cv2.erode(mask, kernel_erode)
    
    final_vessels = cv2.bitwise_and(cleaned_vessels, eroded_mask)
    return final_vessels


def calculate_vessel_density(vessel_mask: np.ndarray, roi_mask: np.ndarray = None) -> float:
    """
    Calculates blood vessel density (percentage of vessel pixels in the ROI).
    
    Mathematical Background:
        Density ρ = N_vessel / N_total
        where N_vessel is vessel pixels, N_total is total ROI pixels.
    """
    if roi_mask is not None:
        total_pixels = np.sum(roi_mask > 0)
        vessel_pixels = np.sum((vessel_mask > 0) & (roi_mask > 0))
    else:
        total_pixels = np.sum(vessel_mask >= 0)
        vessel_pixels = np.sum(vessel_mask > 0)
    
    if total_pixels == 0:
        return 0.0
    return float(vessel_pixels / total_pixels)


def calculate_vessel_tortuosity(vessel_mask: np.ndarray) -> dict:
    """
    Calculates tortuosity metrics for vessel segments.
    
    Mathematical Background:
        Tortuosity T = L_curve / L_chord
        Distance Metric (DM) = (L_curve - L_chord) / L_chord
        Inflection Count (IC) = number of direction changes
    """
    # Skeletonize
    skeleton = skeletonize(vessel_mask)
    
    # Find contours
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    tortuosities = []
    distance_metrics = []
    inflection_counts = []
    
    for cnt in contours:
        arc_len = cv2.arcLength(cnt, closed=False)
        if arc_len < 30.0:
            continue
        
        p_start = cnt[0][0]
        p_end = cnt[-1][0]
        chord_len = np.linalg.norm(p_start - p_end)
        
        if chord_len > 5.0:
            t = arc_len / chord_len
            if t < 5.0:
                tortuosities.append(t)
                dm = (arc_len - chord_len) / chord_len
                distance_metrics.append(dm)
                ic = count_inflections(cnt)
                inflection_counts.append(ic)
    
    if len(tortuosities) == 0:
        return {
            "mean_tortuosity": 1.0,
            "max_tortuosity": 1.0,
            "mean_distance_metric": 0.0,
            "mean_inflection_count": 0,
            "segments": 0
        }
    
    return {
        "mean_tortuosity": float(np.mean(tortuosities)),
        "max_tortuosity": float(np.max(tortuosities)),
        "mean_distance_metric": float(np.mean(distance_metrics)),
        "mean_inflection_count": float(np.mean(inflection_counts)),
        "segments": len(tortuosities)
    }


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """
    Skeletonizes a binary mask using morphological operations.
    
    Mathematical Background:
        Skeleton S(X) = ∪ (X ⊖ nB) \ (X ⊖ nB)∘B
        where ⊖ is erosion, ∘ is opening, B is structuring element.
    """
    if hasattr(cv2, 'ximgproc'):
        return cv2.ximgproc.thinning(mask)
    
    # Fallback: morphological skeletonization
    skeleton = np.zeros_like(mask)
    img = mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    while True:
        eroded = cv2.erode(img, kernel)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
        temp = cv2.subtract(eroded, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        
        if cv2.countNonZero(img) == 0:
            break
    
    return skeleton


def count_inflections(contour: np.ndarray) -> int:
    """
    Counts inflection points (direction changes) in a contour.
    """
    if len(contour) < 3:
        return 0
    
    count = 0
    prev_dir = None
    
    for i in range(1, len(contour) - 1):
        p0 = contour[i - 1][0]
        p1 = contour[i][0]
        p2 = contour[i + 1][0]
        
        # Vectors
        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        
        # Cross product for direction change
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        curr_dir = 1 if cross > 0 else (-1 if cross < 0 else 0)
        
        if prev_dir is not None and curr_dir != 0 and curr_dir != prev_dir:
            count += 1
        
        if curr_dir != 0:
            prev_dir = curr_dir
    
    return count


def calculate_fractal_dimension(vessel_mask: np.ndarray) -> dict:
    """
    Calculates fractal dimension using box-counting method.
    
    Mathematical Background:
        N(s) ∝ s^(-D)
        where N(s) is number of boxes of size s covering the set,
        D is fractal dimension.
        
        Estimated by linear regression of log(N(s)) vs log(1/s).
    """
    # Convert to binary
    binary = (vessel_mask > 0).astype(np.uint8)
    
    # Box sizes (powers of 2)
    sizes = [2, 4, 8, 16, 32, 64]
    
    counts = []
    for s in sizes:
        count = 0
        h, w = binary.shape
        for y in range(0, h, s):
            for x in range(0, w, s):
                box = binary[y:y+s, x:x+s]
                if np.any(box):
                    count += 1
        counts.append(count)
    
    # Remove zeros
    valid = [(s, c) for s, c in zip(sizes, counts) if c > 0]
    if len(valid) < 2:
        return {"fractal_dimension": 1.0, "r_squared": 0.0}
    
    sizes_valid, counts_valid = zip(*valid)
    
    # Linear regression in log-log space
    log_inv_s = np.log(1.0 / np.array(sizes_valid))
    log_n = np.log(np.array(counts_valid))
    
    # Compute regression
    n = len(log_inv_s)
    sum_x = np.sum(log_inv_s)
    sum_y = np.sum(log_n)
    sum_xy = np.sum(log_inv_s * log_n)
    sum_x2 = np.sum(log_inv_s ** 2)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    intercept = (sum_y - slope * sum_x) / n
    
    # R-squared
    y_pred = slope * log_inv_s + intercept
    ss_tot = np.sum((log_n - np.mean(log_n)) ** 2)
    ss_res = np.sum((log_n - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        "fractal_dimension": float(slope),
        "r_squared": float(r2),
        "box_counts": counts,
        "box_sizes": sizes
    }


def analyze_branching(vessel_mask: np.ndarray) -> dict:
    """
    Analyzes vessel branching points and network structure.
    
    Mathematical Background:
        Branch point detection using pixel connectivity:
        C(p) = 0.5 * Σ |b_i - b_{i+1}|
        where b_i are 8-neighbor values in cyclic order.
        C(p) = 2 indicates branch point.
    """
    skeleton = skeletonize(vessel_mask)
    
    # Find branch points
    branch_points = 0
    endpoints = 0
    
    # 8-neighbor indices in clockwise order
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, 1), (1, 1), (1, 0),
                 (1, -1), (0, -1)]
    
    h, w = skeleton.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x] == 0:
                continue
            
            # Get neighbor values
            vals = []
            for dy, dx in neighbors:
                vals.append(1 if skeleton[y + dy, x + dx] > 0 else 0)
            
            # Count transitions
            transitions = 0
            for i in range(8):
                if vals[i] != vals[(i + 1) % 8]:
                    transitions += 1
            
            # Classify point
            n_ones = sum(vals)
            if n_ones == 1:
                endpoints += 1
            elif n_ones >= 3 and transitions >= 4:
                branch_points += 1
    
    # Analyze segments
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    total_length = 0.0
    lengths = []
    
    for cnt in contours:
        length = cv2.arcLength(cnt, closed=False)
        if length > 10:
            total_length += length
            lengths.append(length)
    
    if len(lengths) == 0:
        return {
            "branch_points": 0,
            "endpoints": 0,
            "total_length": 0.0,
            "mean_segment_length": 0.0,
            "branching_density": 0.0
        }
    
    area = np.sum(vessel_mask >= 0)
    branching_density = branch_points / (area / 1000000.0) if area > 0 else 0
    
    return {
        "branch_points": branch_points,
        "endpoints": endpoints,
        "total_length": float(total_length),
        "mean_segment_length": float(np.mean(lengths)),
        "max_segment_length": float(np.max(lengths)),
        "branching_density": float(branching_density)
    }


def calculate_vessel_caliber(vessel_mask: np.ndarray) -> dict:
    """
    Estimates vessel caliber (diameter) using distance transform.
    
    Mathematical Background:
        Distance transform D(p) = min distance to background
        Diameter ≈ 2*D(p) for vessel centerline points.
    """
    # Distance transform
    dist_transform = cv2.distanceTransform(vessel_mask, cv2.DIST_L2, 5)
    
    # Skeletonize
    skeleton = skeletonize(vessel_mask)
    
    # Get diameters at skeleton points
    diameters = []
    h, w = skeleton.shape
    for y in range(h):
        for x in range(w):
            if skeleton[y, x] > 0:
                diameter = 2.0 * dist_transform[y, x]
                if diameter > 0.5:
                    diameters.append(diameter)
    
    if len(diameters) == 0:
        return {
            "mean_caliber": 0.0,
            "max_caliber": 0.0,
            "min_caliber": 0.0,
            "std_caliber": 0.0
        }
    
    return {
        "mean_caliber": float(np.mean(diameters)),
        "max_caliber": float(np.max(diameters)),
        "min_caliber": float(np.min(diameters)),
        "std_caliber": float(np.std(diameters)),
        "percentiles": {
            "p5": float(np.percentile(diameters, 5)),
            "p25": float(np.percentile(diameters, 25)),
            "p50": float(np.percentile(diameters, 50)),
            "p75": float(np.percentile(diameters, 75)),
            "p95": float(np.percentile(diameters, 95))
        }
    }


def comprehensive_vessel_analysis(image: np.ndarray, roi_mask: np.ndarray = None) -> dict:
    """
    Performs comprehensive vessel analysis combining all metrics.
    """
    vessel_mask = extract_blood_vessels(image)
    
    density = calculate_vessel_density(vessel_mask, roi_mask)
    tortuosity = calculate_vessel_tortuosity(vessel_mask)
    fractal = calculate_fractal_dimension(vessel_mask)
    branching = analyze_branching(vessel_mask)
    caliber = calculate_vessel_caliber(vessel_mask)
    
    return {
        "density": density,
        "tortuosity": tortuosity,
        "fractal_dimension": fractal,
        "branching": branching,
        "caliber": caliber,
        "vessel_mask": vessel_mask
    }
