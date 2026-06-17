
<!-- PROJECT LOGO/TITLE -->
<br />
<div align="center">
  <a href="https://github.com/stephenrodrick17-cloud/OpticNova">
    <img src="assets/opticnova-logo.png" alt="opticNova Logo" width="140" height="140" style="border-radius: 20px; box-shadow: 0 8px 30px rgba(47, 137, 252, 0.35);">
  </a>

  <h1 align="center">opticNova</h1>

  <p align="center">
    Advanced Ophthalmic Eye Detection & Biomarker Suite
    <br />
    <a href="https://github.com/stephenrodrick17-cloud/OpticNova"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    ·
    <a href="https://github.com/stephenrodrick17-cloud/OpticNova/issues">Report Bug</a>
    ·
    <a href="https://github.com/stephenrodrick17-cloud/OpticNova/issues">Request Feature</a>
  </p>

  <!-- BADGES -->
  <p align="center">
    <img src="https://img.shields.io/github/license/stephenrodrick17-cloud/OpticNova?style=flat-square&color=2F89FC" alt="License">
    <img src="https://img.shields.io/github/last-commit/stephenrodrick17-cloud/OpticNova?style=flat-square&color=3BE8B0" alt="Last Commit">
    <img src="https://img.shields.io/github/languages/top/stephenrodrick17-cloud/OpticNova?style=flat-square&color=8B5CF6" alt="Top Language">
    <img src="https://img.shields.io/github/stars/stephenrodrick17-cloud/OpticNova?style=flat-square&color=F59E0B" alt="Stars">
  </p>
</div>

---

## 📖 Table of Contents
1. [About the Project](#about-the-project)
2. [Key Features](#key-features)
3. [Tech Stack](#tech-stack)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Mathematical Foundations](#mathematical-foundations)
7. [AI Accessibility Mode](#ai-accessibility-mode)
8. [Contributing](#contributing)
9. [License](#license)
10. [Contact](#contact)

---

## 👁️ About the Project
**opticNova** is a cutting-edge, AI-powered ophthalmic diagnostic suite designed to assist medical professionals and researchers in analyzing fundus photographs. It combines deep learning, classical computer vision, and rigorous mathematical modeling to provide comprehensive, explainable insights into ocular health.

### Why opticNova?
- **Clinically Relevant Metrics**: Calculates Cup-to-Disc Ratio (CDR), ISNT rule compliance, vessel tortuosity, and drusen load.
- **Explainable AI**: Uses Grad-CAM to visualize which regions of the image influenced model decisions.
- **Mathematical Rigor**: Every measurement is backed by formal mathematical proofs and visualizations.
- **Accessible Design**: Built-in AI-powered accessibility features for visually impaired users.

---

## ✨ Key Features

### 🧠 Diagnostic Workstation
- **Upload & Preprocess**: Enhance fundus images using CLAHE and resizing.
- **Deep Learning Analysis**:
  - Glaucoma classification using EfficientNet.
  - Optic nerve segmentation using U-Net.
  - Lesion detection using YOLO.
- **Grad-CAM Visualization**: See exactly where the model is focusing!
- **Comprehensive Biomarkers**:
  - CDR (Vertical, Horizontal, Area-based)
  - ISNT Rule Compliance
  - Vessel Density & Tortuosity
  - Drusen Count & Load

### 📐 Mathematical Proofs & Reports
Every metric comes with:
- Formal mathematical equations in LaTeX.
- Real-time visualizations using the uploaded image.
- Clinical interpretation guidelines.

### 🔊 AI Accessibility Mode
- **AI-Powered Analysis**: Uses OpenRouter (Gemini/LLaMA) to analyze results.
- **Text-to-Speech**: Reads out comprehensive diagnostic reports for visually impaired users.
- **Natural Language Explanations**: Translates technical metrics into easy-to-understand language.

### 🎨 Beautiful UI
- Dark, eye-friendly theme.
- Glassmorphism design.
- Responsive layout.
- Interactive dashboard.

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Frontend** | Streamlit, CSS, HTML |
| **Deep Learning** | PyTorch, EfficientNet, U-Net, YOLO |
| **Computer Vision** | OpenCV, PIL, SciPy |
| **Mathematics** | NumPy, Matplotlib, Scikit-image |
| **AI Integration** | OpenRouter API |
| **Version Control** | Git, GitHub |

---

## 🚀 Installation

### Prerequisites
- Python 3.9+
- Git
- (Optional but recommended) Virtual environment tool (venv, conda, etc.)

### Step-by-Step Guide

1. **Clone the repository**
   ```bash
   git clone https://github.com/stephenrodrick17-cloud/OpticNova.git
   cd OpticNova
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   cd EyeVisionAI
   pip install -r requirements.txt
   ```

4. **Set up OpenRouter API Key** (Optional, for Accessibility Mode)
   ```bash
   # Windows (PowerShell)
   $env:OPENROUTER_API_KEY="your-openrouter-api-key-here"

   # macOS/Linux
   export OPENROUTER_API_KEY="your-openrouter-api-key-here"
   ```

5. **Run the app!**
   ```bash
   streamlit run app/streamlit_app.py
   ```

---

## 📱 Usage

### Home Page
- Hero section with opticNova branding.
- Feature cards showing key capabilities.
- Patient info booking area.

### Diagnostic Workstation Tabs
1. **Clinical View & Explainability**: Enhanced image, Grad-CAM, YOLO detections.
2. **Segmentation & Polar Profile**: Optic disc/cup segmentation, CDR, ISNT metrics, polar plot.
3. **Retinal Vascular Model**: Vessel segmentation, density, tortuosity.
4. **Macular Drusen (AMD)**: Drusen detection, area ratio, severity.
5. **Mathematical Proofs & Reports**: All equations, histograms, convolution, and final clinical report.

### Accessibility Mode
1. Upload a fundus image.
2. Wait for processing.
3. Click **🎤 Start AI Accessibility Analysis**.
4. The AI will generate a comprehensive report and read it out loud!

---

## 🧮 Mathematical Foundations

opticNova's strength lies in its mathematical rigor! Here are the core equations powering the analysis:

---

### 1. Histogram Equalization (Image Enhancement)
Improves visibility of blood vessels and lesions:

$$
s_k = (L-1) \sum_{j=0}^{k} p(r_j)
$$

- \(L\): Number of intensity levels.
- \(p(r_j)\): Probability of pixel intensity \(r_j\).

---

### 2. Convolution Operations (Linear Algebra)
CNNs use convolution for feature extraction:

$$
S(i,j) = \sum_{m} \sum_{n} I(i+m,j+n) K(m,n)
$$

- Detects blood vessels, optic disc boundaries, exudates, and hemorrhages.

---

### 3. Cross-Entropy Loss (Disease Classification)
Neural networks optimize this loss function:

$$
L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)
$$

- \(y_i\): True label.
- \(\hat{y}_i\): Predicted probability.
- \(C\): Number of disease classes.

---

### 4. Gradient Descent (Optimization)
Model weights are updated using:

$$
w_{t+1} = w_t - \eta \nabla L(w_t)
$$

- \(w_t\): Current weights.
- \(\eta\): Learning rate.
- \(\nabla L\): Gradient of loss.

---

### 5. Cup-to-Disc Ratio (CDR) - Glaucoma Detection
Key clinical metric:

$$
\text{CDR} = \frac{D_{\text{cup}}}{D_{\text{disc}}}
$$

- \(D_{\text{cup}}\): Optic cup diameter.
- \(D_{\text{disc}}\): Optic disc diameter.
- If CDR > 0.6, glaucoma risk is elevated.

---

### 6. Vessel Tortuosity Index
Measures blood vessel curvature:

$$
T = \frac{1}{N} \sum_{i=1}^{N} \frac{L_{\text{arc}}^{(i)}}{L_{\text{chord}}^{(i)}}
$$

- \(L_{\text{arc}}\): Actual vessel length.
- \(L_{\text{chord}}\): Straight-line distance.

---

## ♿ AI Accessibility Mode

### How it Works
1. Collects all clinical metrics (CDR, ISNT, vessels, drusen, severity).
2. Sends to OpenRouter's `openrouter/free` model.
3. Receives a natural language explanation tailored for visually impaired users.
4. Uses browser-native Text-to-Speech (TTS) to read the report aloud.

### Privacy
- Images are processed locally on your machine.
- Only text-based metrics are sent to the AI (no raw images).
- You can use any OpenRouter-supported model.

---

## 🤝 Contributing

Contributions make opticNova better! If you'd like to contribute:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add some AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request!

Please make sure to:
- Follow the existing code style.
- Add comments for complex functions.
- Test your changes.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 📬 Contact

**Stephen Rodrick** - [GitHub Profile](https://github.com/stephenrodrick17-cloud)

Project Link: [https://github.com/stephenrodrick17-cloud/OpticNova](https://github.com/stephenrodrick17-cloud/OpticNova)

---

<div align="center">
  <p>Made with ❤️ for ophthalmic AI research</p>
</div>

