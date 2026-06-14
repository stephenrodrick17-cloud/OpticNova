import numpy as np
import cv2

def calculate_isnt_metrics(disc_mask: np.ndarray, cup_mask: np.ndarray, eye_side: str = "Right Eye") -> dict:
    """
    Divides the optic disc and cup into Inferior, Superior, Nasal, and Temporal quadrants.
    Calculates rim thickness (distance from cup boundary to disc boundary) along multiple radials.
    
    Args:
        disc_mask (np.ndarray): Binary mask of optic disc (512x512).
        cup_mask (np.ndarray): Binary mask of optic cup (512x512).
        eye_side (str): "Right Eye" (OD) or "Left Eye" (OS).
        
    Returns:
        dict: Metrics including sector thickness averages, ISNT rule compliance, and coordinates for plotting.
    """
    # 1. Find centers of disc and cup using centroids
    disc_coords = np.argwhere(disc_mask > 0.5)
    if len(disc_coords) == 0:
        return {
            "compliant": True,
            "thicknesses": {"Inferior": 0.0, "Superior": 0.0, "Nasal": 0.0, "Temporal": 0.0},
            "rule_violated_sectors": [],
            "rim_profile": []
        }
    
    cy, cx = np.mean(disc_coords, axis=0)
    
    # We will sample thickness along 360 radial angles (0 to 359 degrees)
    # 0 degrees is right (positive x), 90 degrees is down (positive y in image coords)
    rim_thicknesses = []
    angles = np.arange(0, 360, 5) # sample every 5 degrees for smoothness
    
    # Max search radius
    h, w = disc_mask.shape
    max_radius = int(np.sqrt(h**2 + w**2))
    
    # Store profile coordinates for visual representation
    profile_coords = []
    
    for angle_deg in angles:
        angle_rad = np.deg2rad(angle_deg)
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        
        # Trace from center outwards to find cup boundary and disc boundary
        cup_boundary_r = None
        disc_boundary_r = None
        
        for r in range(1, max_radius):
            px = int(cx + r * dx)
            py = int(cy + r * dy)
            
            if px < 0 or px >= w or py < 0 or py >= h:
                break
                
            # If we exit the cup mask
            if cup_boundary_r is None and cup_mask[py, px] <= 0.5 and r > 1:
                # cup boundary was at r-1
                cup_boundary_r = r - 1
                
            # If we exit the disc mask
            if disc_mask[py, px] <= 0.5:
                disc_boundary_r = r
                break
                
        if cup_boundary_r is None:
            cup_boundary_r = 0
        if disc_boundary_r is None:
            disc_boundary_r = 0
            
        thickness = max(0, disc_boundary_r - cup_boundary_r)
        rim_thicknesses.append((angle_deg, thickness, cup_boundary_r, disc_boundary_r))
        
        # Save points for plotting overlays
        disc_pt = (int(cx + disc_boundary_r * dx), int(cy + disc_boundary_r * dy))
        cup_pt = (int(cx + cup_boundary_r * dx), int(cy + cup_boundary_r * dy))
        profile_coords.append({"angle": angle_deg, "disc_pt": disc_pt, "cup_pt": cup_pt, "thickness": thickness})

    # 2. Map angles to Clinical Quadrants (ISNT)
    # Standard image orientation:
    # 0 deg = Right, 90 deg = Down (Inferior), 180 deg = Left, 270 deg = Up (Superior)
    # Sector definitions:
    # Superior: 225 to 315 deg (centered around 270 deg)
    # Inferior: 45 to 135 deg (centered around 90 deg)
    # Nasal & Temporal depend on Left vs. Right Eye:
    # Right Eye (OD): Nasal is Left (135 to 225 deg), Temporal is Right (315 to 45 deg)
    # Left Eye (OS): Nasal is Right (315 to 45 deg), Temporal is Left (135 to 225 deg)
    
    sector_thicknesses = {"Superior": [], "Inferior": [], "Nasal": [], "Temporal": []}
    
    for angle_deg, thickness, _, _ in rim_thicknesses:
        # Normalize angle to 0-360
        angle_deg = angle_deg % 360
        
        if 45 <= angle_deg < 135:
            sector_thicknesses["Inferior"].append(thickness)
        elif 225 <= angle_deg < 315:
            sector_thicknesses["Superior"].append(thickness)
        else:
            # Either Nasal or Temporal
            is_right_half = (angle_deg >= 315 or angle_deg < 45)
            if eye_side == "Right Eye":
                if is_right_half:
                    sector_thicknesses["Temporal"].append(thickness)
                else:
                    sector_thicknesses["Nasal"].append(thickness)
            else: # Left Eye
                if is_right_half:
                    sector_thicknesses["Nasal"].append(thickness)
                else:
                    sector_thicknesses["Temporal"].append(thickness)

    # Average thickness in each quadrant
    avg_thicknesses = {sec: float(np.mean(vals)) if len(vals) > 0 else 0.0 for sec, vals in sector_thicknesses.items()}
    
    # 3. Validate ISNT Rule: I >= S >= N >= T
    # Check violations:
    violated = []
    # I >= S
    if avg_thicknesses["Inferior"] < avg_thicknesses["Superior"]:
        violated.append("Inferior < Superior")
    # S >= N
    if avg_thicknesses["Superior"] < avg_thicknesses["Nasal"]:
        violated.append("Superior < Nasal")
    # N >= T
    if avg_thicknesses["Nasal"] < avg_thicknesses["Temporal"]:
        violated.append("Nasal < Temporal")
        
    is_compliant = len(violated) == 0
    
    return {
        "center": (int(cx), int(cy)),
        "compliant": is_compliant,
        "thicknesses": avg_thicknesses,
        "rule_violated_sectors": violated,
        "rim_profile": profile_coords
    }
