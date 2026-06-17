import numpy as np
import cv2
from scipy import stats


def calculate_isnt_metrics(disc_mask: np.ndarray, cup_mask: np.ndarray, eye_side: str = "Right Eye") -> dict:
    """
    Divides the optic disc and cup into Inferior, Superior, Nasal, and Temporal quadrants.
    Calculates rim thickness along multiple radials with comprehensive mathematical analysis.
    
    Mathematical Background for ISNT Rule:
        The ISNT rule states that in a healthy optic disc:
            Inferior ≥ Superior ≥ Nasal ≥ Temporal
        
        Rim thickness at angle θ: t(θ) = R_disc(θ) - R_cup(θ)
        where R_disc(θ) is the distance from center to disc boundary at angle θ
        and R_cup(θ) is the distance from center to cup boundary at angle θ.
        
        Statistical analysis:
        - Mean thickness per quadrant
        - Standard deviation
        - Confidence intervals
        - T-tests for pairwise comparisons
    """
    disc_coords = np.argwhere(disc_mask > 0.5)
    if len(disc_coords) == 0:
        return {
            "compliant": True,
            "thicknesses": {"Inferior": 0.0, "Superior": 0.0, "Nasal": 0.0, "Temporal": 0.0},
            "rule_violated_sectors": [],
            "rim_profile": [],
            "statistics": {}
        }
    
    cy, cx = np.mean(disc_coords, axis=0)
    
    rim_thicknesses = []
    angles = np.arange(0, 360, 2)
    h, w = disc_mask.shape
    max_radius = int(np.sqrt(h**2 + w**2))
    profile_coords = []
    
    for angle_deg in angles:
        angle_rad = np.deg2rad(angle_deg)
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        
        cup_boundary_r = None
        disc_boundary_r = None
        
        for r in range(1, max_radius):
            px = int(cx + r * dx)
            py = int(cy + r * dy)
            
            if px < 0 or px >= w or py < 0 or py >= h:
                break
                
            if cup_boundary_r is None and cup_mask[py, px] <= 0.5 and r > 1:
                cup_boundary_r = r - 1
                
            if disc_mask[py, px] <= 0.5:
                disc_boundary_r = r
                break
                
        if cup_boundary_r is None:
            cup_boundary_r = 0
        if disc_boundary_r is None:
            disc_boundary_r = 0
            
        thickness = max(0, disc_boundary_r - cup_boundary_r)
        rim_thicknesses.append((angle_deg, thickness, cup_boundary_r, disc_boundary_r))
        
        disc_pt = (int(cx + disc_boundary_r * dx), int(cy + disc_boundary_r * dy))
        cup_pt = (int(cx + cup_boundary_r * dx), int(cy + cup_boundary_r * dy))
        profile_coords.append({"angle": angle_deg, "disc_pt": disc_pt, "cup_pt": cup_pt, "thickness": thickness})
    
    sector_thicknesses = {"Superior": [], "Inferior": [], "Nasal": [], "Temporal": []}
    
    for angle_deg, thickness, _, _ in rim_thicknesses:
        angle_deg = angle_deg % 360
        
        if 45 <= angle_deg < 135:
            sector_thicknesses["Inferior"].append(thickness)
        elif 225 <= angle_deg < 315:
            sector_thicknesses["Superior"].append(thickness)
        else:
            is_right_half = (angle_deg >= 315 or angle_deg < 45)
            if eye_side == "Right Eye":
                if is_right_half:
                    sector_thicknesses["Temporal"].append(thickness)
                else:
                    sector_thicknesses["Nasal"].append(thickness)
            else:
                if is_right_half:
                    sector_thicknesses["Nasal"].append(thickness)
                else:
                    sector_thicknesses["Temporal"].append(thickness)
    
    sector_stats = {}
    for sector, vals in sector_thicknesses.items():
        if len(vals) > 0:
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            ci_low, ci_high = stats.t.interval(0.95, len(vals)-1, loc=mean_val, scale=stats.sem(vals))
            sector_stats[sector] = {
                "mean": float(mean_val),
                "std": float(std_val),
                "ci_95_low": float(ci_low) if not np.isnan(ci_low) else float(mean_val - std_val),
                "ci_95_high": float(ci_high) if not np.isnan(ci_high) else float(mean_val + std_val),
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "count": len(vals)
            }
        else:
            sector_stats[sector] = {
                "mean": 0.0, "std": 0.0, "ci_95_low": 0.0, "ci_95_high": 0.0,
                "median": 0.0, "min": 0.0, "max": 0.0, "count": 0
            }
    
    avg_thicknesses = {sec: stat["mean"] for sec, stat in sector_stats.items()}
    
    comparisons = pairwise_comparisons(sector_stats)
    
    violated = []
    violation_confidence = {}
    
    I, S, N, T = (avg_thicknesses["Inferior"], avg_thicknesses["Superior"],
                  avg_thicknesses["Nasal"], avg_thicknesses["Temporal"])
    
    if I < S:
        violated.append("Inferior < Superior")
        violation_confidence["Inferior < Superior"] = comparisons["Inferior-Superior"]["p_value"]
    if S < N:
        violated.append("Superior < Nasal")
        violation_confidence["Superior < Nasal"] = comparisons["Superior-Nasal"]["p_value"]
    if N < T:
        violated.append("Nasal < Temporal")
        violation_confidence["Nasal < Temporal"] = comparisons["Nasal-Temporal"]["p_value"]
    
    is_compliant = len(violated) == 0
    compliance_score = calculate_compliance_score(sector_stats)
    
    return {
        "center": (int(cx), int(cy)),
        "compliant": is_compliant,
        "compliance_score": round(compliance_score, 4),
        "thicknesses": avg_thicknesses,
        "sector_statistics": sector_stats,
        "pairwise_comparisons": comparisons,
        "rule_violated_sectors": violated,
        "violation_confidence": violation_confidence,
        "rim_profile": profile_coords
    }


def pairwise_comparisons(sector_stats: dict) -> dict:
    """
    Performs pairwise t-tests between quadrant thicknesses.
    
    Mathematical Background:
        For two samples X and Y:
        t = (μ_x - μ_y) / sqrt(s_x²/n_x + s_y²/n_y)
        Degrees of freedom using Welch-Satterthwaite approximation
    """
    sectors = ["Inferior", "Superior", "Nasal", "Temporal"]
    comparisons = {}
    
    for i in range(len(sectors)):
        for j in range(i + 1, len(sectors)):
            s1, s2 = sectors[i], sectors[j]
            stat1, stat2 = sector_stats[s1], sector_stats[s2]
            
            if stat1["count"] > 1 and stat2["count"] > 1:
                n1, n2 = stat1["count"], stat2["count"]
                m1, m2 = stat1["mean"], stat2["mean"]
                v1, v2 = stat1["std"] ** 2, stat2["std"] ** 2
                
                t_stat, p_val = stats.ttest_ind_from_stats(
                    m1, stat1["std"], n1, m2, stat2["std"], n2,
                    equal_var=False
                )
                
                cohens_d = (m1 - m2) / np.sqrt((v1 + v2) / 2)
                
                comparisons[f"{s1}-{s2}"] = {
                    "t_statistic": float(t_stat) if not np.isnan(t_stat) else 0.0,
                    "p_value": float(p_val) if not np.isnan(p_val) else 1.0,
                    "cohens_d": float(cohens_d) if not np.isnan(cohens_d) else 0.0,
                    "mean_difference": float(m1 - m2)
                }
            else:
                comparisons[f"{s1}-{s2}"] = {
                    "t_statistic": 0.0,
                    "p_value": 1.0,
                    "cohens_d": 0.0,
                    "mean_difference": 0.0
                }
    
    return comparisons


def calculate_compliance_score(sector_stats: dict) -> float:
    """
    Calculates a continuous compliance score for the ISNT rule.
    
    Mathematical Background:
        Score S = (w1*I + w2*S + w3*N + w4*T) - penalty
        where weights w1 > w2 > w3 > w4
        Penalty for violations with statistical confidence
    """
    I = sector_stats["Inferior"]["mean"]
    S = sector_stats["Superior"]["mean"]
    N = sector_stats["Nasal"]["mean"]
    T = sector_stats["Temporal"]["mean"]
    
    total = I + S + N + T
    if total == 0:
        return 0.0
    
    w1, w2, w3, w4 = 0.4, 0.3, 0.2, 0.1
    base_score = (w1 * I + w2 * S + w3 * N + w4 * T) / total
    
    penalty = 0.0
    if I < S:
        penalty += 0.2 * (1 - I/(S + 1e-10))
    if S < N:
        penalty += 0.2 * (1 - S/(N + 1e-10))
    if N < T:
        penalty += 0.2 * (1 - N/(T + 1e-10))
    
    compliance_score = max(0.0, min(1.0, base_score - penalty))
    
    return compliance_score


def calculate_rim_asymmetry(rim_profile: list) -> dict:
    """
    Calculates rim asymmetry using Fourier analysis.
    
    Mathematical Background:
        Rim thickness profile t(θ) can be decomposed into Fourier series:
        t(θ) = a0/2 + Σ [an cos(nθ) + bn sin(nθ)]
        
        Asymmetry is measured by the magnitude of odd harmonics.
    """
    angles = np.deg2rad([p["angle"] for p in rim_profile])
    thicknesses = np.array([p["thickness"] for p in rim_profile])
    
    n_samples = len(thicknesses)
    if n_samples == 0:
        return {"asymmetry_index": 0.0, "harmonics": {}}
    
    fft_vals = np.fft.fft(thicknesses)
    freqs = np.fft.fftfreq(n_samples)
    
    harmonics = {}
    for n in range(1, 5):
        idx = np.where(np.isclose(np.abs(freqs), n / n_samples))[0]
        if len(idx) > 0:
            magnitude = np.abs(fft_vals[idx[0]]) / n_samples
            harmonics[f"n{n}"] = float(magnitude)
        else:
            harmonics[f"n{n}"] = 0.0
    
    total_power = np.sum(np.abs(fft_vals) ** 2) / n_samples
    odd_power = harmonics.get("n1", 0) ** 2 + harmonics.get("n3", 0) ** 2
    
    asymmetry_index = np.sqrt(odd_power) / (np.sqrt(total_power) + 1e-10)
    
    return {
        "asymmetry_index": round(float(asymmetry_index), 4),
        "harmonics": {k: round(v, 4) for k, v in harmonics.items()}
    }


def calculate_rim_volume(disc_mask: np.ndarray, cup_mask: np.ndarray) -> dict:
    """
    Calculates neuroretinal rim volume and related metrics.
    
    Mathematical Background:
        Rim volume V ≈ Σ (A_disc - A_cup) * t_avg
        where t_avg is the average rim thickness.
        
        Using paraboloid approximation:
        V_rim = (π/3) * (R_disc² H_disc - R_cup² H_cup)
        where H is the height of the paraboloid.
    """
    disc_area = np.sum(disc_mask > 0.5)
    cup_area = np.sum(cup_mask > 0.5)
    rim_area = disc_area - cup_area
    
    disc_coords = np.argwhere(disc_mask > 0.5)
    if len(disc_coords) == 0:
        return {"rim_volume": 0.0, "disc_volume": 0.0, "cup_volume": 0.0}
    
    cy, cx = np.mean(disc_coords, axis=0)
    disc_radius = np.sqrt(disc_area / np.pi)
    cup_radius = np.sqrt(cup_area / np.pi) if cup_area > 0 else 0
    
    disc_height = disc_radius * 0.3
    cup_height = cup_radius * 0.4
    
    disc_volume = (np.pi / 3) * disc_radius ** 2 * disc_height
    cup_volume = (np.pi / 3) * cup_radius ** 2 * cup_height
    rim_volume = disc_volume - cup_volume
    
    return {
        "rim_volume": round(rim_volume, 2),
        "disc_volume": round(disc_volume, 2),
        "cup_volume": round(cup_volume, 2),
        "rim_area": round(rim_area, 2),
        "rim_volume_ratio": round(rim_volume / disc_volume if disc_volume > 0 else 0, 4)
    }
