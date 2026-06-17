import numpy as np
import cv2
from scipy import stats, ndimage


def calculate_entropy(image: np.ndarray) -> float:
    """
    Calculates Shannon entropy of an image.
    
    Mathematical Background:
        H(X) = -Σ p(x_i) log2 p(x_i)
        where p(x_i) is probability of pixel value x_i.
    """
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))
    return float(entropy)


def calculate_statistical_metrics(arr: np.ndarray) -> dict:
    """
    Calculates comprehensive statistical metrics for an array.
    
    Returns:
        Dictionary with mean, std, median, min, max,
        skewness, kurtosis, percentiles.
    """
    arr_flat = arr.flatten()
    if len(arr_flat) == 0:
        return {}
    
    mean_val = np.mean(arr_flat)
    std_val = np.std(arr_flat)
    median_val = np.median(arr_flat)
    min_val = np.min(arr_flat)
    max_val = np.max(arr_flat)
    
    # Higher-order moments
    if std_val > 0:
        skewness = stats.skew(arr_flat)
        kurtosis_val = stats.kurtosis(arr_flat)
    else:
        skewness = 0.0
        kurtosis_val = 0.0
    
    # Percentiles
    percentiles = {}
    for p in [5, 10, 25, 50, 75, 90, 95]:
        percentiles[f"p{p}"] = float(np.percentile(arr_flat, p))
    
    return {
        "mean": float(mean_val),
        "std": float(std_val),
        "median": float(median_val),
        "min": float(min_val),
        "max": float(max_val),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis_val),
        "percentiles": percentiles
    }


def gaussian_kernel(size: int, sigma: float = None) -> np.ndarray:
    """
    Generates a 2D Gaussian kernel.
    
    Mathematical Background:
        G(x, y) = (1/(2πσ²)) exp(-(x²+y²)/(2σ²))
    """
    if sigma is None:
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    
    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel


def laplacian_of_gaussian(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Applies Laplacian of Gaussian (LoG) filter.
    
    Mathematical Background:
        LoG(x, y) = -(1/(πσ⁴)) (1 - (x²+y²)/(2σ²)) exp(-(x²+y²)/(2σ²))
    """
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    # Apply Laplacian
    log = cv2.Laplacian(blurred, cv2.CV_64F)
    return log


def calculate_hu_moments(mask: np.ndarray) -> np.ndarray:
    """
    Calculates Hu's 7 invariant moments.
    
    Mathematical Background:
        Hu moments are invariant to translation, scale, and rotation.
        Derived from centralized and normalized moments.
    """
    moments = cv2.moments(mask.astype(np.uint8))
    hu_moments = cv2.HuMoments(moments)
    # Log transform for better scale invariance
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    return hu_moments.flatten()


def zernike_moments(mask: np.ndarray, radius: int, order: int = 8) -> np.ndarray:
    """
    Calculates Zernike moments up to given order.
    
    Mathematical Background:
        Z_n^m(r, θ) = R_n^m(r) exp(i m θ)
        where R_n^m are radial polynomials.
    """
    # Normalize to circle of radius 'radius'
    mask = mask.astype(np.uint8)
    moments = []
    
    # Calculate centroid
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return np.zeros((order + 1) * (order + 2) // 2)
    
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    
    # Get coordinates
    y, x = np.indices(mask.shape)
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r_normalized = r / radius
    theta = np.arctan2(y - cy, x - cx)
    
    # Calculate moments
    for n in range(order + 1):
        for m in range(-n, n + 1, 2):
            if abs(m) > n:
                continue
            
            # Radial polynomial
            R = np.zeros_like(r_normalized)
            for k in range((n - abs(m)) // 2 + 1):
                coeff = ((-1)**k) * np.math.factorial(n - k)
                coeff /= (np.math.factorial(k) *
                         np.math.factorial((n + abs(m)) // 2 - k) *
                         np.math.factorial((n - abs(m)) // 2 - k))
                R += coeff * (r_normalized ** (n - 2*k))
            
            # Zernike moment
            zernike = R * np.exp(-1j * m * theta)
            moment = np.sum(zernike * (mask > 0))
            moments.append(moment)
    
    return np.array(moments)


def calculate_curvature(contour: np.ndarray) -> np.ndarray:
    """
    Calculates curvature along a contour.
    
    Mathematical Background:
        κ = (x'y'' - x''y') / (x'² + y'²)^(3/2)
        where ' denotes derivative.
    """
    if len(contour) < 3:
        return np.zeros(len(contour))
    
    # Reshape contour
    contour = contour.reshape(-1, 2)
    x = contour[:, 0]
    y = contour[:, 1]
    
    # First derivatives (central differences)
    dx = np.gradient(x)
    dy = np.gradient(y)
    
    # Second derivatives
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Curvature
    numerator = dx * ddy - dy * ddx
    denominator = (dx**2 + dy**2)**1.5
    denominator[denominator == 0] = 1e-10
    curvature = numerator / denominator
    
    return curvature


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int = 2) -> tuple:
    """
    Fits a polynomial to data points.
    
    Returns:
        (coefficients, residuals, r_squared)
    """
    coeffs, residuals, rank, singular_values, rcond = np.polyfit(x, y, degree, full=True)
    
    # Calculate R-squared
    y_pred = np.polyval(coeffs, x)
    ss_tot = np.sum((y - np.mean(y))**2)
    ss_res = np.sum((y - y_pred)**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return coeffs, float(ss_res[0]) if len(residuals) > 0 else 0, float(r2)


def sigmoid_function(x: np.ndarray, x0: float = 0, k: float = 1) -> np.ndarray:
    """
    Sigmoid function: σ(x) = 1 / (1 + exp(-k(x - x0)))
    """
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def gaussian_function(x: np.ndarray, mu: float = 0, sigma: float = 1, A: float = 1) -> np.ndarray:
    """
    Gaussian function: G(x) = A exp(-(x - μ)²/(2σ²))
    """
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


def exponential_moving_average(data: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Calculates exponential moving average.
    
    Mathematical Background:
        EMA_t = α x_t + (1 - α) EMA_{t-1}
    """
    ema = np.zeros_like(data, dtype=np.float64)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


def calculate_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """
    Calculates Pearson correlation coefficient matrix.
    
    Mathematical Background:
        ρ(X, Y) = Cov(X, Y) / (σ_X σ_Y)
    """
    return np.corrcoef(data, rowvar=False)


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 32) -> float:
    """
    Calculates mutual information between two variables.
    
    Mathematical Background:
        I(X, Y) = Σ Σ p(x, y) log(p(x, y)/(p(x) p(y)))
    """
    # Joint histogram
    hist_xy, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    
    # Marginal histograms
    hist_x = np.sum(hist_xy, axis=1)
    hist_y = np.sum(hist_xy, axis=0)
    
    # Probabilities
    p_xy = hist_xy / np.sum(hist_xy)
    p_x = hist_x / np.sum(hist_x)
    p_y = hist_y / np.sum(hist_y)
    
    # Mutual information
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
    
    return float(mi)


def fft_analysis(signal: np.ndarray, sample_rate: float = 1.0) -> dict:
    """
    Performs Fast Fourier Transform analysis on a signal.
    
    Returns:
        Dictionary with frequencies, magnitudes, phases,
        dominant frequency, power spectrum.
    """
    n = len(signal)
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n, 1 / sample_rate)
    
    # Positive frequencies
    positive_idx = freqs >= 0
    freqs_pos = freqs[positive_idx]
    magnitudes_pos = np.abs(fft_vals[positive_idx]) / n
    phases_pos = np.angle(fft_vals[positive_idx])
    
    # Power spectrum
    power_spectrum = np.abs(fft_vals) ** 2 / n
    
    # Dominant frequency (excluding DC)
    if len(freqs_pos) > 1:
        dominant_idx = np.argmax(magnitudes_pos[1:]) + 1
        dominant_freq = freqs_pos[dominant_idx]
        dominant_mag = magnitudes_pos[dominant_idx]
    else:
        dominant_freq = 0.0
        dominant_mag = 0.0
    
    return {
        "frequencies": freqs_pos,
        "magnitudes": magnitudes_pos,
        "phases": phases_pos,
        "power_spectrum": power_spectrum,
        "dominant_frequency": float(dominant_freq),
        "dominant_magnitude": float(dominant_mag)
    }


def wavelet_transform(signal: np.ndarray, wavelet: str = 'haar', level: int = 3) -> list:
    """
    Performs discrete wavelet transform.
    Note: This is a simple implementation. Consider using PyWavelets for full features.
    """
    # Haar wavelet transform
    coeffs = []
    approx = signal.copy()
    
    for _ in range(level):
        n = len(approx)
        # Even and odd indices
        even = approx[::2]
        odd = approx[1::2]
        
        # Detail coefficients
        detail = (even - odd) / np.sqrt(2)
        # Approximation coefficients
        approx = (even + odd) / np.sqrt(2)
        
        coeffs.append(detail)
    
    coeffs.append(approx)
    return coeffs


def calculate_image_gradients(image: np.ndarray) -> tuple:
    """
    Calculates image gradients using Sobel operators.
    
    Mathematical Background:
        G_x = I * K_x, G_y = I * K_y
        Magnitude = √(G_x² + G_y²)
        Orientation = arctan2(G_y, G_x)
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = np.arctan2(grad_y, grad_x)
    
    return grad_x, grad_y, magnitude, orientation


def local_binary_pattern(image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    """
    Calculates Local Binary Pattern (LBP) texture descriptor.
    
    Mathematical Background:
        LBP(P, R) = Σ 2^p * s(I_p - I_c)
        where s(x) = 1 if x ≥ 0 else 0
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    h, w = image.shape
    lbp = np.zeros_like(image, dtype=np.uint8)
    
    # Get coordinates
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = radius * np.cos(angles)
    y = -radius * np.sin(angles)  # Negative because y increases downward
    
    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            center = image[i, j]
            code = 0
            
            for p in range(n_points):
                # Interpolate pixel value
                x_p = j + x[p]
                y_p = i + y[p]
                
                x0 = int(np.floor(x_p))
                x1 = x0 + 1
                y0 = int(np.floor(y_p))
                y1 = y0 + 1
                
                # Bilinear interpolation
                fx = x_p - x0
                fy = y_p - y0
                
                v00 = image[y0, x0]
                v01 = image[y0, x1]
                v10 = image[y1, x0]
                v11 = image[y1, x1]
                
                neighbor = (v00 * (1 - fx) * (1 - fy) +
                           v01 * fx * (1 - fy) +
                           v10 * (1 - fx) * fy +
                           v11 * fx * fy)
                
                if neighbor >= center:
                    code |= 1 << p
            
            lbp[i, j] = code
    
    return lbp


def histogram_equalization(image: np.ndarray, clip_limit: float = 0.0) -> np.ndarray:
    """
    Performs histogram equalization with optional clipping.
    
    Mathematical Background:
        Transforms image using cumulative distribution function:
        T(r_k) = (L-1) Σ p(r_j) for j=0 to k
    """
    if clip_limit > 0:
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(image)
    else:
        return cv2.equalizeHist(image)


def canny_edge_detection(image: np.ndarray, low_threshold: float = 50, high_threshold: float = 150) -> np.ndarray:
    """
    Canny edge detection with Gaussian smoothing.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 1.4)
    # Canny edges
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges


def hough_transform(image: np.ndarray, rho: float = 1, theta: float = np.pi/180,
                   threshold: int = 100) -> tuple:
    """
    Hough transform for line detection.
    """
    edges = canny_edge_detection(image)
    lines = cv2.HoughLines(edges, rho, theta, threshold)
    
    if lines is not None:
        lines = lines.reshape(-1, 2)
    else:
        lines = np.array([])
    
    return lines, edges
