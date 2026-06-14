def compute_severity_score(glaucoma_prob: float, cdr: float, rim_area: float, isnt_compliant: bool = True, vessel_density: float = 0.12, vessel_tortuosity: float = 1.15) -> dict:
    """
    Computes a research-grade disease severity score:
    Severity = α(CDR) + β(RimLoss) + γ(VesselDensityLoss) + δ(CNNScore) + ε(ISNT Violation Penalty)
    """
    alpha = 35.0
    beta = 20.0
    gamma = 15.0 # Active vessel density loss factor
    delta = 20.0
    epsilon = 10.0 # ISNT violation penalty
    
    # 1. CDR Factor (scaled 0.0 to 1.0)
    cdr_factor = min(1.0, cdr)
    
    # 2. RimLoss: Assume healthy rim area is roughly 30000 px for a 512x512 image, 
    # so we penalize smaller rim areas.
    healthy_rim = 30000.0
    rim_loss_factor = max(0.0, 1.0 - (rim_area / healthy_rim))
    
    # 3. Vessel Density Loss: Normal density is around 12-16% in fundus images. 
    # We penalize densities drop below 10%.
    vessel_loss_factor = max(0.0, 1.0 - (vessel_density / 0.12))
    
    # 4. Deep learning confidence
    heatmap_score = glaucoma_prob
    
    # 5. ISNT penalty
    isnt_penalty = 1.0 if not isnt_compliant else 0.0
    
    score = (alpha * cdr_factor) + (beta * rim_loss_factor) + (gamma * vessel_loss_factor) + (delta * heatmap_score) + (epsilon * isnt_penalty)
    score = min(100.0, max(0.0, score))
    
    if score < 25:
        category = "Normal / Low Risk"
    elif score < 50:
        category = "Suspect / Mild Glaucoma"
    elif score < 75:
        category = "Moderate Glaucoma"
    else:
        category = "Severe Glaucoma"
        
    return {
        "score": round(score, 2),
        "category": category,
        "factors": {
            "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta, "epsilon": epsilon,
            "cdr_factor": cdr_factor, "rim_loss": rim_loss_factor, "vessel_loss": vessel_loss_factor, 
            "heatmap": heatmap_score, "isnt_penalty": isnt_penalty
        }
    }
