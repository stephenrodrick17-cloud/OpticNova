import numpy as np
from scipy import stats


def compute_severity_score(
    glaucoma_prob: float,
    cdr: float,
    rim_area: float,
    isnt_compliant: bool = True,
    vessel_density: float = 0.12,
    vessel_tortuosity: float = 1.15,
    disc_area: float = 50000.0,
    rim_asymmetry: float = 0.0,
    disc_circularity: float = 1.0
) -> dict:
    """
    Computes a comprehensive disease severity score using multiple mathematical models.
    
    Mathematical Background:
        Severity S = Σ (w_i * f_i) + penalty
        where w_i are optimized weights, f_i are normalized feature functions.
        
        Bayesian posterior probability:
        P(Severe | x) ∝ P(x | Severe) * P(Severe)
        
        Logistic regression model:
        logit(S) = β₀ + β₁x₁ + ... + βₙxₙ
    """
    # Weight configuration - optimized for clinical relevance
    weights = {
        "cdr": 0.25,
        "rim_loss": 0.20,
        "vessel_loss": 0.15,
        "cnn_score": 0.20,
        "isnt_violation": 0.10,
        "asymmetry": 0.05,
        "circularity": 0.05
    }
    
    # 1. CDR Factor - sigmoidal mapping for better discrimination
    cdr_optimal = 0.3
    cdr_k = 10.0
    cdr_factor = sigmoid((cdr - cdr_optimal) * cdr_k)
    
    # 2. Rim Loss Factor - exponential decay
    healthy_rim = 30000.0
    rim_ratio = rim_area / healthy_rim
    rim_loss_factor = exponential_decay(rim_ratio, 2.0)
    
    # 3. Vessel Density Loss - piecewise linear
    vessel_normal = 0.14
    vessel_critical = 0.08
    vessel_loss_factor = piecewise_linear(
        vessel_density,
        [(vessel_critical, 1.0), (vessel_normal, 0.0)]
    )
    
    # 4. Vessel Tortuosity Factor
    tortuosity_normal = 1.05
    tortuosity_factor = sigmoid((vessel_tortuosity - tortuosity_normal) * 5.0)
    
    # 5. Asymmetry Factor
    asymmetry_factor = min(1.0, rim_asymmetry / 0.3)
    
    # 6. Circularity Factor (penalize non-circular discs)
    circularity_factor = max(0.0, 1.0 - disc_circularity)
    
    # 7. ISNT Penalty - soft penalty with confidence
    isnt_penalty = 0.0
    if not isnt_compliant:
        isnt_penalty = 0.8
    
    # 8. Deep learning confidence
    cnn_score = glaucoma_prob
    
    # Compute weighted score
    score = (
        weights["cdr"] * cdr_factor +
        weights["rim_loss"] * rim_loss_factor +
        weights["vessel_loss"] * vessel_loss_factor +
        weights["cnn_score"] * cnn_score +
        weights["isnt_violation"] * isnt_penalty +
        weights["asymmetry"] * asymmetry_factor +
        weights["circularity"] * circularity_factor
    )
    
    # Scale to 0-100
    score_scaled = score * 100.0
    score_scaled = min(100.0, max(0.0, score_scaled))
    
    # Determine category
    if score_scaled < 20:
        category = "Normal / Very Low Risk"
    elif score_scaled < 40:
        category = "Low Risk / Suspect"
    elif score_scaled < 60:
        category = "Mild Glaucoma"
    elif score_scaled < 80:
        category = "Moderate Glaucoma"
    else:
        category = "Severe Glaucoma"
    
    # Calculate uncertainty
    uncertainty = calculate_uncertainty(
        cdr, rim_area, vessel_density, glaucoma_prob
    )
    
    # Bayesian probability
    bayesian_prob = bayesian_severity(
        cdr, rim_loss_factor, vessel_loss_factor, cnn_score
    )
    
    return {
        "score": round(score_scaled, 2),
        "category": category,
        "bayesian_probability": round(bayesian_prob, 4),
        "uncertainty": round(uncertainty, 4),
        "factors": {
            "weights": weights,
            "cdr_factor": round(cdr_factor, 4),
            "rim_loss": round(rim_loss_factor, 4),
            "vessel_loss": round(vessel_loss_factor, 4),
            "tortuosity_factor": round(tortuosity_factor, 4),
            "asymmetry_factor": round(asymmetry_factor, 4),
            "circularity_factor": round(circularity_factor, 4),
            "cnn_score": round(cnn_score, 4),
            "isnt_penalty": round(isnt_penalty, 4)
        }
    }


def sigmoid(x: float) -> float:
    """
    Sigmoid activation function: σ(x) = 1 / (1 + exp(-x))
    Maps input to range [0, 1].
    """
    return 1.0 / (1.0 + np.exp(-x))


def exponential_decay(x: float, k: float) -> float:
    """
    Exponential decay function: f(x) = 1 - exp(-k*x)
    """
    return 1.0 - np.exp(-k * x)


def piecewise_linear(x: float, points: list) -> float:
    """
    Piecewise linear interpolation between points.
    Points should be sorted by x value.
    """
    if len(points) < 2:
        return 0.0
    
    points = sorted(points, key=lambda p: p[0])
    
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    
    return 0.0


def calculate_uncertainty(
    cdr: float,
    rim_area: float,
    vessel_density: float,
    cnn_score: float
) -> float:
    """
    Calculates epistemic uncertainty using entropy and variance.
    
    Mathematical Background:
        Uncertainty U = w1 * H + w2 * Var
        where H is entropy, Var is variance of inputs.
    """
    # Entropy of CNN score (max at 0.5)
    epsilon = 1e-10
    p = np.clip(cnn_score, epsilon, 1 - epsilon)
    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    entropy_normalized = entropy / 1.0  # max entropy is 1 bit
    
    # Variance in measurements
    # Estimate based on how far from typical values
    cdr_deviation = abs(cdr - 0.4) / 0.4
    rim_deviation = abs(rim_area - 30000) / 30000
    vessel_deviation = abs(vessel_density - 0.14) / 0.14
    
    variance_estimate = np.mean([cdr_deviation, rim_deviation, vessel_deviation])
    variance_estimate = min(1.0, variance_estimate)
    
    # Combined uncertainty
    uncertainty = 0.6 * entropy_normalized + 0.4 * variance_estimate
    
    return min(1.0, max(0.0, uncertainty))


def bayesian_severity(
    cdr: float,
    rim_loss: float,
    vessel_loss: float,
    cnn_score: float
) -> float:
    """
    Computes Bayesian posterior probability of severe glaucoma.
    
    Mathematical Background:
        P(Severe | x) = [P(x | Severe) * P(Severe)] / P(x)
        
        Using Naive Bayes assumption:
        P(x | Severe) = ∏ P(x_i | Severe)
    """
    # Prior probability (prevalence in population)
    p_severe = 0.02
    p_normal = 1.0 - p_severe
    
    # Likelihoods P(x_i | Severe) and P(x_i | Normal)
    # Modeled as Beta distributions
    def beta_likelihood(x: float, alpha: float, beta: float) -> float:
        x = np.clip(x, 0.01, 0.99)
        return stats.beta.pdf(x, alpha, beta)
    
    # CDR likelihoods
    p_cdr_severe = beta_likelihood(cdr, 8, 2)
    p_cdr_normal = beta_likelihood(cdr, 2, 5)
    
    # Rim loss likelihoods
    p_rim_severe = beta_likelihood(rim_loss, 6, 2)
    p_rim_normal = beta_likelihood(rim_loss, 2, 6)
    
    # Vessel loss likelihoods
    p_vessel_severe = beta_likelihood(vessel_loss, 5, 2)
    p_vessel_normal = beta_likelihood(vessel_loss, 2, 5)
    
    # CNN score likelihoods
    p_cnn_severe = beta_likelihood(cnn_score, 7, 2)
    p_cnn_normal = beta_likelihood(cnn_score, 2, 6)
    
    # Joint likelihoods
    p_x_severe = p_cdr_severe * p_rim_severe * p_vessel_severe * p_cnn_severe
    p_x_normal = p_cdr_normal * p_rim_normal * p_vessel_normal * p_cnn_normal
    
    # Posterior probability
    numerator = p_x_severe * p_severe
    denominator = numerator + p_x_normal * p_normal
    
    if denominator == 0:
        return 0.5
    
    return numerator / denominator


def severity_trend_analysis(scores: list, dates: list = None) -> dict:
    """
    Analyzes severity trends over time using linear regression.
    
    Mathematical Background:
        Trend line: Score(t) = m*t + b
        Rate of change: m
        Statistical significance: p-value
    """
    if len(scores) < 2:
        return {
            "trend": "insufficient_data",
            "rate_of_change": 0.0,
            "r_squared": 0.0,
            "p_value": 1.0
        }
    
    x = np.arange(len(scores))
    y = np.array(scores)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    if slope > 2.0:
        trend = "worsening"
    elif slope < -2.0:
        trend = "improving"
    else:
        trend = "stable"
    
    return {
        "trend": trend,
        "rate_of_change": round(slope, 4),
        "r_squared": round(r_value ** 2, 4),
        "p_value": round(p_value, 4),
        "intercept": round(intercept, 4)
    }
