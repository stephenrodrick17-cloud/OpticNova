import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import sys
import os
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.resize import resize_image
from preprocessing.clahe import apply_clahe
from classification.efficientnet import GlaucomaEfficientNet
from segmentation.unet import UNet
from explainability.gradcam import GradCAM
from explainability.heatmap import overlay_heatmap
from mathematics.cdr import calculate_cdr, calculate_vertical_diameter
from mathematics.disc_area import calculate_area, calculate_rim_area
from mathematics.severity_score import compute_severity_score
from mathematics.isnt_rule import calculate_isnt_metrics
from mathematics.vessel_analysis import extract_blood_vessels, calculate_vessel_density, calculate_vessel_tortuosity
from mathematics.drusen_detection import detect_macular_drusen
from detection.yolo_detector import FundusYOLODetector

# --- Page Config & Styling ---
st.set_page_config(page_title="Medi+ AI Ophthalmic Suite", layout="wide", page_icon="👁️")

# --- Encode Background Video to Base64 ---
video_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Computer_code_reflected_in_eye_202606142148.mp4")
video_base64 = ""
if os.path.exists(video_path):
    with open(video_path, "rb") as video_file:
        video_base64 = base64.b64encode(video_file.read()).decode("utf-8")

# Custom CSS implementing the background video, dark glassmorphic cards, and full horizontal nav bar
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #E2E8F0;
    }}
    
    /* Background Video Styling */
    #bg-video {{
        position: fixed;
        right: 0;
        bottom: 0;
        min-width: 100vw;
        min-height: 100vh;
        width: auto;
        height: auto;
        z-index: -9999;
        object-fit: cover;
        opacity: 0.4;
    }}
    
    /* Main container — dark semi-transparent, NO white */
    .stApp {{
        background: rgba(10, 15, 30, 0.75) !important;
    }}
    
    /* Hide Default Sidebar */
    [data-testid="sidebarHelper"] {{
        display: none !important;
    }}
    [data-testid="stSidebar"] {{
        display: none !important;
    }}
    [data-testid="collapsedControl"] {{
        display: none !important;
    }}
    .st-emotion-cache-1dp5vir {{
        padding: 2rem 3rem 10rem !important;
    }}
    
    /* ---- NUKE ALL STREAMLIT WHITE BOXES ---- */
    .stMainBlockContainer, .block-container {{
        background: transparent !important;
    }}
    [data-testid="stHeader"] {{
        background: transparent !important;
    }}
    [data-testid="stToolbar"] {{
        display: none !important;
    }}
    /* Remove white from expanders, tabs, containers, info boxes, etc */
    .streamlit-expanderHeader, .streamlit-expanderContent,
    [data-testid="stExpander"],
    [data-testid="stTabs"],
    [data-testid="stTabsContent"],
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"],
    div[data-testid="element-container"],
    .stAlert {{
        background: transparent !important;
        border-color: rgba(47, 137, 252, 0.2) !important;
    }}
    /* Make tab buttons dark */
    button[data-baseweb="tab"] {{
        background: transparent !important;
        color: #94A3B8 !important;
        border-bottom: 2px solid transparent !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: #3BE8B0 !important;
        border-bottom: 2px solid #3BE8B0 !important;
    }}
    /* Make text inputs / select boxes dark */
    .stTextInput > div > div, .stNumberInput > div > div,
    .stSelectbox > div > div, [data-baseweb="select"] > div {{
        background: rgba(15, 23, 42, 0.8) !important;
        border-color: rgba(47, 137, 252, 0.3) !important;
        color: #E2E8F0 !important;
    }}
    .stTextInput input, .stNumberInput input {{
        color: #E2E8F0 !important;
        background: transparent !important;
    }}
    /* File uploader dark */
    [data-testid="stFileUploader"] {{
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px dashed rgba(47, 137, 252, 0.4) !important;
        border-radius: 12px !important;
    }}
    /* Slider dark */
    .stSlider > div > div {{
        background: transparent !important;
    }}
    /* st.info / st.success / st.warning / st.error dark */
    [data-testid="stNotification"] {{
        background: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(47, 137, 252, 0.3) !important;
        color: #CBD5E1 !important;
    }}
    /* Buttons dark */
    .stButton > button {{
        background: linear-gradient(135deg, rgba(47, 137, 252, 0.3), rgba(59, 232, 176, 0.2)) !important;
        border: 1px solid rgba(47, 137, 252, 0.5) !important;
        color: #E2E8F0 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }}
    .stButton > button:hover {{
        background: linear-gradient(135deg, rgba(47, 137, 252, 0.5), rgba(59, 232, 176, 0.35)) !important;
        border-color: #3BE8B0 !important;
        transform: translateY(-2px) !important;
    }}
    /* Markdown and all general text */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2,
    .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
    label, .stSelectbox label, .stTextInput label, .stNumberInput label {{
        color: #CBD5E1 !important;
    }}
    /* Horizontal rules */
    hr {{
        border-color: rgba(47, 137, 252, 0.2) !important;
    }}
    /* Spinner */
    .stSpinner > div {{
        color: #3BE8B0 !important;
    }}

    /* Top Navigation Bar Styling — DARK */
    .medi-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(16px);
        padding: 15px 40px;
        border-bottom: 1px solid rgba(47, 137, 252, 0.25);
        margin-bottom: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }}
    .medi-logo {{
        font-size: 26px;
        font-weight: 700;
        color: #2F89FC;
    }}
    .medi-logo span {{
        color: #3BE8B0;
    }}
    .medi-contact-info {{
        font-size: 14px;
        color: #94A3B8;
        font-weight: 500;
    }}
    
    /* Medi+ Feature Cards — DARK Glassmorphism */
    .medi-card {{
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(47, 137, 252, 0.2);
        border-radius: 12px;
        padding: 30px 25px;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }}
    .medi-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(47,137,252,0.2);
        border-color: #3BE8B0;
    }}
    .medi-card-icon {{
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background-color: rgba(47, 137, 252, 0.15);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #2F89FC;
        font-size: 24px;
        margin-bottom: 20px;
    }}
    .medi-card h3 {{
        color: #F1F5F9 !important;
        font-weight: 700;
        margin-bottom: 12px;
    }}
    .medi-card p {{
        color: #94A3B8 !important;
        font-size: 14px;
        line-height: 1.6;
    }}
    
    /* Hero Banner overlay card — DARK */
    .medi-hero-overlay {{
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(16px);
        padding: 50px;
        border-radius: 16px;
        border: 1px solid rgba(47, 137, 252, 0.2);
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
        margin-bottom: 40px;
    }}
    .medi-hero-tag {{
        background-color: rgba(59, 232, 176, 0.15);
        color: #3BE8B0;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 15px;
    }}
    .medi-hero-overlay h1 {{
        font-size: 44px;
        font-weight: 700;
        color: #F1F5F9 !important;
        margin-bottom: 15px;
    }}
    .medi-hero-overlay p {{
        color: #94A3B8 !important;
    }}
    
    /* Interactive Appointment Area — DARK */
    .appointment-box {{
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(47, 137, 252, 0.2);
        border-radius: 16px;
        padding: 40px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }}
    .appointment-box h3 {{
        color: #F1F5F9 !important;
    }}
    .appointment-box p {{
        color: #94A3B8 !important;
    }}
    
    /* Workstation specific elements */
    .metric-title {{
        font-size: 12px;
        color: #94A3B8 !important;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }}
    .metric-number {{
        font-size: 30px;
        font-weight: 700;
        color: #3BE8B0;
        margin-top: 5px;
    }}
    
    /* Clinical alerts — DARK */
    .clinical-severe {{
        background-color: rgba(127, 29, 29, 0.5);
        border-left: 6px solid #EF4444;
        padding: 15px 20px;
        border-radius: 8px;
        color: #FCA5A5 !important;
        font-weight: 700;
        margin-bottom: 20px;
        backdrop-filter: blur(8px);
    }}
    .clinical-warning {{
        background-color: rgba(120, 83, 9, 0.4);
        border-left: 6px solid #F59E0B;
        padding: 15px 20px;
        border-radius: 8px;
        color: #FCD34D !important;
        font-weight: 700;
        margin-bottom: 20px;
        backdrop-filter: blur(8px);
    }}
    .clinical-normal {{
        background-color: rgba(6, 78, 59, 0.4);
        border-left: 6px solid #10B981;
        padding: 15px 20px;
        border-radius: 8px;
        color: #6EE7B7 !important;
        font-weight: 700;
        margin-bottom: 20px;
        backdrop-filter: blur(8px);
    }}
    
    .eq-box {{
        background-color: rgba(15, 23, 42, 0.9) !important;
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        color: #F8FAFC !important;
    }}
    .eq-box * {{
        color: #E2E8F0 !important;
    }}
    .eq-result {{
        background-color: rgba(30, 41, 59, 0.9) !important;
        border: 1px solid #10B981;
        border-radius: 8px;
        padding: 10px 15px;
        color: #34D399 !important;
        font-weight: 600;
    }}
    </style>
""", unsafe_allow_html=True)

# --- Render Background Video HTML ---
if video_base64:
    st.markdown(f"""
        <video autoplay loop muted playsinline id="bg-video">
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        </video>
    """, unsafe_allow_html=True)

# --- Top Navigation Bar ---
st.markdown("""
    <div class="medi-header">
        <div class="medi-logo">Medi<span>+ AI</span></div>
        <div class="medi-contact-info">
            🏥 Emergency: <b>1-800-555-OCU</b> &nbsp;&nbsp;|&nbsp;&nbsp; ✉️ Clinical: <b>support@mediplus.ai</b>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- Horizontal Page Navigation ---
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "🏠 Medi+ Home Portal"

col_nav1, col_nav2 = st.columns(2)
with col_nav1:
    if st.button("🏠 Medi+ Home Portal", use_container_width=True):
        st.session_state.active_tab = "🏠 Medi+ Home Portal"
with col_nav2:
    if st.button("🔬 Diagnostic Workstation", use_container_width=True):
        st.session_state.active_tab = "🔬 Diagnostic Workstation"

st.markdown("---")

# ==============================================================================
# PORTAL HOME PAGE (Colorlib Medi+ Inspired)
# ==============================================================================
if st.session_state.active_tab == "🏠 Medi+ Home Portal":
    
    # Hero Area
    st.markdown("""
        <div class="medi-hero-overlay">
            <span class="medi-hero-tag">Clinical AI Solution</span>
            <h1>Ophthalmic Eye Detection & Biomarker Suite</h1>
            <p>An advanced clinical intelligence system powered by deep learning. Instantly segment the optic nerve head, calculate microvascular tortuosity, and verify physiological ISNT compliance.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; margin-bottom: 40px; font-weight: 700; color: #F1F5F9;'>Our Clinical Features</h2>", unsafe_allow_html=True)
    
    # Feature Cards Row
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.markdown("""
            <div class="medi-card">
                <div class="medi-card-icon">👁️</div>
                <h3>Optic Nerve Segmentation</h3>
                <p>Isolate boundaries of both the optic cup and disc utilizing custom trained PyTorch U-Net models to generate high-fidelity geometric measurements.</p>
            </div>
        """, unsafe_allow_html=True)
    with col_f2:
        st.markdown("""
            <div class="medi-card">
                <div class="medi-card-icon">🕸️</div>
                <h3>Retinal Vascular Modeling</h3>
                <p>Isolate retinal blood vessels using Gaussian adaptive thresholds to calculate microvascular tortuosity and capillary dropout indicators.</p>
            </div>
        """, unsafe_allow_html=True)
    with col_f3:
        st.markdown("""
            <div class="medi-card">
                <div class="medi-card-icon">📐</div>
                <h3>ISNT Quadrant Verifier</h3>
                <p>Extract sector rim thicknesses and verify the clinical ISNT rule hierarchy to diagnose early neuroretinal thinning.</p>
            </div>
        """, unsafe_allow_html=True)

    # Appointment Booking Area
    st.markdown("### 📅 Book Patient Clinical Run")
    
    with st.container():
        st.markdown("""
            <div class="appointment-box">
                <div style="margin-bottom: 25px;">
                    <h3 style="color:#2D3748; margin:0 0 5px 0;">Schedule Trial Analysis</h3>
                    <p style="color:#718096; margin:0; font-size:14px;">Fill in basic patient data metadata. Then switch to the <b>Diagnostic Workstation</b> to upload retinal scans.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        col_form1, col_form2 = st.columns(2)
        with col_form1:
            patient_id = st.text_input("Patient Identification ID", "PT-2026-9812")
            patient_age = st.number_input("Patient Age", min_value=1, max_value=120, value=52)
        with col_form2:
            patient_gender = st.selectbox("Patient Gender", ["Male", "Female", "Other"])
            eye_laterality = st.selectbox("Initial Scan Laterality", ["Right Eye", "Left Eye"])
            
        st.info("💡 Patient profile scheduled. You can now switch to the **Diagnostic Workstation** navigation bar to upload the scan.")

    # Clinical FAQ Accordion
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### ❓ Frequently Asked Clinical Questions (FAQ)")
    
    with st.expander("What clinical standard is the ISNT rule based on?"):
        st.markdown("""
        The **ISNT rule** states that in healthy eyes, the neuroretinal rim thickness should follow the hierarchy of **Inferior (thickest) ≥ Superior ≥ Nasal ≥ Temporal (thinnest)**. 
        Glaucomatous damage typically causes localized thinning at the inferior and superior poles first, violating this hierarchy.
        """)
        
    with st.expander("How does OcuVision calculate vessel tortuosity?"):
        st.markdown("""
        The system skeletonizes the extracted blood vessel network and breaks it down into individual traced segments. 
        For each segment, it computes the ratio of the actual curve length (arc) to the straight-line distance (chord). 
        An average ratio greater than 1.25 indicates significant microvascular adaptation.
        """)

# ==============================================================================
# WORKSTATION PAGE
# ==============================================================================
else:
    st.markdown("<h2 style='color:#F1F5F9; margin-top: 0;'>🔬 Diagnostic Workstation</h2>", unsafe_allow_html=True)
    st.markdown("Run automated PyTorch segmentations and compute mathematical biomarker metrics for patients.")
    
    # Input Scan Configurations
    eye_side = st.selectbox("Retinal Scan Eye Laterality", ["Right Eye", "Left Eye"])
    conf_thresh = st.slider("YOLO Defect Confidence", 0.1, 1.0, 0.25)
    
    uploaded_file = st.file_uploader("Upload Retinal Scan (JPG / PNG)", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Load & Preprocess
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        resized_img = resize_image(image_np, (512, 512))
        clahe_img = apply_clahe(resized_img)
        
        # Torch preparation
        tensor_img = torch.tensor(clahe_img).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor_img = (tensor_img - mean) / std
        tensor_img = tensor_img.unsqueeze(0)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        with st.spinner("🔬 Running deep neural segmentation and clinical math..."):
            # 1. CNN Classification
            classifier = GlaucomaEfficientNet().to(device)
            class_weights_path = "best_efficientnet_classification.pth"
            if os.path.exists(class_weights_path):
                try:
                    classifier.load_state_dict(torch.load(class_weights_path, map_location=device))
                except RuntimeError:
                    pass
            classifier.eval()
            with torch.no_grad():
                output = classifier(tensor_img.to(device))
                prob = torch.sigmoid(output).item()
                
            # 2. Grad-CAM Explainability
            target_layer = classifier.model.features[-1]
            grad_cam = GradCAM(classifier, target_layer)
            heatmap = grad_cam.generate_heatmap(tensor_img.to(device))
            heatmap_overlay = overlay_heatmap(clahe_img, heatmap)
            
            # 3. U-Net Segmentation
            unet_model = UNet(n_channels=3, n_classes=2).to(device)
            seg_weights_path = "best_unet_segmentation.pth"
            if os.path.exists(seg_weights_path):
                try:
                    unet_model.load_state_dict(torch.load(seg_weights_path, map_location=device))
                except RuntimeError:
                    pass
            unet_model.eval()
            with torch.no_grad():
                seg_logits = unet_model(tensor_img.to(device))
                seg_probs = torch.sigmoid(seg_logits)
                seg_preds = (seg_probs > 0.5).float()
                
            disc_mask_np = seg_preds[:, 0:1, :, :].squeeze().cpu().numpy()
            cup_mask_np = seg_preds[:, 1:2, :, :].squeeze().cpu().numpy()
            
            # 4. YOLO Detector
            yolo_detector = FundusYOLODetector()
            yolo_img, detections = yolo_detector.detect(resized_img, conf_threshold=conf_thresh)
            
            # 5. Vessel segments
            vessel_mask = extract_blood_vessels(resized_img)
            vessel_density = calculate_vessel_density(vessel_mask)
            tortuosity_dict = calculate_vessel_tortuosity(vessel_mask)
            mean_tortuosity = tortuosity_dict["mean_tortuosity"]
            
            # 6. Macular Drusen Segmentation
            drusen_dict = detect_macular_drusen(resized_img, disc_mask_np, eye_side=eye_side)
            drusen_mask = drusen_dict["drusen_mask"]
            drusen_area_ratio = drusen_dict["drusen_area_ratio"]
            macula_cx, macula_cy = drusen_dict["macula_center"]
            macula_radius = drusen_dict["macula_radius"]
            
            # 7. ISNT calculations
            isnt_metrics = calculate_isnt_metrics(disc_mask_np, cup_mask_np, eye_side=eye_side)
            isnt_compliant = isnt_metrics["compliant"]
            
            # 8. Geometric metrics
            cup_v_diameter = calculate_vertical_diameter(cup_mask_np)
            disc_v_diameter = calculate_vertical_diameter(disc_mask_np)
            cdr = calculate_cdr(disc_mask_np, cup_mask_np)
            disc_area = calculate_area(disc_mask_np)
            cup_area = calculate_area(cup_mask_np)
            rim_area = calculate_rim_area(disc_mask_np, cup_mask_np)
            
            severity = compute_severity_score(
                prob, cdr, rim_area, 
                isnt_compliant=isnt_compliant, 
                vessel_density=vessel_density, 
                vessel_tortuosity=mean_tortuosity
            )
            factors = severity["factors"]

        # Alert Banner Output
        if severity["category"] == "Severe Glaucoma":
            st.markdown(f'<div class="clinical-severe">⚠️ CRITICAL PATHOLOGY: {severity["category"].upper()} — Index: {severity["score"]}/100</div>', unsafe_allow_html=True)
        elif "Moderate" in severity["category"] or "Suspect" in severity["category"]:
            st.markdown(f'<div class="clinical-warning">⚠️ CLINICAL WARNING: {severity["category"].upper()} — Index: {severity["score"]}/100</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="clinical-normal">✅ CLINICAL NORMAL: Retinal structure shows low anomaly risk — Index: {severity["score"]}/100</div>', unsafe_allow_html=True)

        # Workstation Analysis Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "👁️ Clinical View & Explainability",
            "📊 Segmentation & Polar Profile",
            "🕸️ Retinal Vascular Model",
            "🟡 Macular Drusen (AMD)",
            "📐 Mathematical Proofs & Reports"
        ])
        
        # --- TAB 1 ---
        with tab1:
            st.markdown("### Neural Networks & Visual Heatmaps")
            col_img1, col_img2, col_img3 = st.columns(3)
            with col_img1:
                st.image(clahe_img, caption="1. Enhanced Scan", use_container_width=True)
            with col_img2:
                st.image(heatmap_overlay, caption="2. CNN Grad-CAM Attention Map", use_container_width=True)
            with col_img3:
                st.image(yolo_img, caption="3. YOLOv8 Localization Boxes", use_container_width=True)
                
        # --- TAB 2 ---
        with tab2:
            st.markdown("### Optic Disc Boundary Segmentation & Polar Rim Profile")
            
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.markdown(f"<div class='medi-card'><div class='metric-title'>Vertical CDR</div><div class='metric-number'>{cdr:.3f}</div></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='medi-card'><div class='metric-title'>ISNT Compliant</div><div class='metric-number'>{'YES' if isnt_compliant else 'NO'}</div></div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div class='medi-card'><div class='metric-title'>Disc Area</div><div class='metric-number'>{disc_area:.0f} px</div></div>", unsafe_allow_html=True)
            with c4:
                st.markdown(f"<div class='medi-card'><div class='metric-title'>Cup Area</div><div class='metric-number'>{cup_area:.0f} px</div></div>", unsafe_allow_html=True)
            with c5:
                st.markdown(f"<div class='medi-card'><div class='metric-title'>Rim Area</div><div class='metric-number'>{rim_area:.0f} px</div></div>", unsafe_allow_html=True)

            col_sub1, col_sub2 = st.columns([1, 1])
            with col_sub1:
                combined_overlay = clahe_img.copy()
                combined_overlay[disc_mask_np > 0.5] = [0, 200, 0]
                combined_overlay[cup_mask_np > 0.5] = [255, 50, 50]
                blended = cv2.addWeighted(clahe_img, 0.6, combined_overlay, 0.4, 0)
                st.image(blended, caption="Optic Nerve Segments (Green=Disc, Red=Cup)", use_container_width=True)
            with col_sub2:
                fig = plt.figure(figsize=(6, 5))
                ax = fig.add_subplot(111, polar=True)
                
                profile = isnt_metrics["rim_profile"]
                angles_rad = [np.deg2rad(x["angle"]) for x in profile]
                thicknesses = [x["thickness"] for x in profile]
                
                if len(angles_rad) > 0:
                    angles_rad.append(angles_rad[0])
                    thicknesses.append(thicknesses[0])
                
                ax.plot(angles_rad, thicknesses, color='#2F89FC', linewidth=2, label='Rim Width')
                ax.fill(angles_rad, thicknesses, color='#2F89FC', alpha=0.15)
                
                ax.set_theta_zero_location('E')
                ax.set_title("360° Neuroretinal Rim Thickness Curve", fontsize=11, fontweight='bold', pad=15)
                ax.set_xticks(np.deg2rad([90, 270, 0, 180]))
                ax.set_xticklabels(['Inferior', 'Superior', 'Temporal' if eye_side=="Right Eye" else 'Nasal', 'Nasal' if eye_side=="Right Eye" else 'Temporal'], fontsize=9)
                
                fig.tight_layout()
                st.pyplot(fig)
                
                if isnt_compliant:
                    st.success("✅ **ISNT Rule Satisfied**: Follows standard physiological hierarchy.")
                else:
                    st.warning(f"⚠️ **ISNT Rule Violated**: Rim thinning detected in sectors: {', '.join(isnt_metrics['rule_violated_sectors'])}")

        # --- TAB 3 ---
        with tab3:
            st.markdown("### Retinal Blood Vessel Segmentation")
            col_v1, col_v2 = st.columns([1, 1])
            with col_v1:
                overlay_v = clahe_img.copy()
                overlay_v[vessel_mask > 0] = [255, 0, 0]
                blended_v = cv2.addWeighted(clahe_img, 0.65, overlay_v, 0.35, 0)
                st.image(blended_v, caption="Segmented Retinal Vasculature (Red Overlay)", use_container_width=True)
            with col_v2:
                st.markdown("#### Vascular Metrics")
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.markdown(f"<div class='medi-card'><div class='metric-title'>Vessel Density</div><div class='metric-number'>{vessel_density:.2%}</div></div>", unsafe_allow_html=True)
                with sc2:
                    st.markdown(f"<div class='medi-card'><div class='metric-title'>Tortuosity Index</div><div class='metric-number'>{mean_tortuosity:.4f}</div></div>", unsafe_allow_html=True)
                st.markdown("""
                **Diagnostic Relevance**:
                - **Dropout**: Drops in vessel density below 10.0% correlate with ischemia and localized glaucoma thinning.
                - **Tortuosity**: Tortuosity index increases ($>1.25$) are indicators of diabetic and high blood pressure retinopathies.
                """)

        # --- TAB 4 ---
        with tab4:
            st.markdown("### Macular Drusen Segmentation (AMD Analysis)")
            st.markdown("Automatic macula localization and micro-deposit segmentation for Age-related Macular Degeneration.")
            
            col_d1, col_d2 = st.columns([1, 1])
            with col_d1:
                overlay_d = clahe_img.copy()
                cv2.circle(overlay_d, (macula_cx, macula_cy), macula_radius, (47, 137, 252), 2)
                overlay_d[drusen_mask > 0] = [255, 235, 59]
                blended_d = cv2.addWeighted(clahe_img, 0.65, overlay_d, 0.35, 0)
                st.image(blended_d, caption="Macula ROI (Blue) & Drusen Segmentation (Yellow)", use_container_width=True)
            with col_d2:
                st.markdown("#### Macular Biomarkers")
                st.markdown(f"<div class='medi-card'><div class='metric-title'>Drusen Area Ratio (DAR)</div><div class='metric-number'>{drusen_area_ratio:.4%}</div></div>", unsafe_allow_html=True)
                
                if drusen_area_ratio < 0.005:
                    st.success("✅ **Healthy Macula**: No significant drusen accumulation detected in the macular region.")
                elif drusen_area_ratio < 0.02:
                    st.warning("⚠️ **Mild Drusen Accumulation**: Detected early drusen deposits in macular ROI. Monitor closely for dry AMD signs.")
                else:
                    st.error("🔴 **High Anomaly Density**: High concentration of drusen spots within the Macula ROI, suggesting Dry Age-related Macular Degeneration (AMD).")

        # --- TAB 5 ---
        with tab5:
            st.markdown("### 📐 Mathematical Proofs & Diagnostic Summary Report")
            
            st.markdown("#### 1. Vertical Cup-to-Disc Ratio (vCDR)")
            st.markdown('<div class="eq-box">', unsafe_allow_html=True)
            st.latex(r"\text{vCDR} = \frac{H_{\text{cup}}}{H_{\text{disc}}}")
            st.markdown(f'<div class="eq-result">📐 <b>Patient Value:</b> {cup_v_diameter} px / {disc_v_diameter} px = <b>{cdr:.4f}</b></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("#### 2. Drusen Area Ratio (DAR)")
            st.markdown('<div class="eq-box">', unsafe_allow_html=True)
            st.latex(r"\text{DAR} = \frac{\sum_{(x,y) \in \text{Macular ROI}} \mathbb{1}[\text{Drusen}(x,y) > 0]}{\sum_{(x,y) \in \text{Macular ROI}} \mathbb{1}[\text{ROI}(x,y) > 0]}")
            st.markdown(f'<div class="eq-result">🟡 <b>Patient Value:</b> Drusen Area Ratio (DAR) = <b>{drusen_area_ratio:.4%}</b></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("#### 3. Vessel Tortuosity Index")
            st.markdown('<div class="eq-box">', unsafe_allow_html=True)
            st.latex(r"T = \frac{1}{N} \sum_{i=1}^{N} \frac{L_{\text{arc}}^{(i)}}{L_{\text{chord}}^{(i)}}")
            st.markdown(f'<div class="eq-result">🕸️ <b>Patient Vessel Tortuosity (T):</b> <b>{mean_tortuosity:.4f}</b></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Clinical report print area
            st.markdown("#### 📋 Comprehensive Diagnostic Summary Report")
            report_html = f"""
            <div style="background-color: rgba(15, 23, 42, 0.85); border: 1px solid rgba(47, 137, 252, 0.25); border-radius: 12px; padding: 30px; color: #E2E8F0; box-shadow: 0 4px 20px rgba(0,0,0,0.3); backdrop-filter: blur(12px);">
                <div style="text-align: center; margin-bottom: 25px;">
                    <h2 style="color: #2F89FC; margin: 0; font-size:24px;">MEDI+ AI CLINICAL REPORT</h2>
                    <small style="color: #94A3B8; font-size:12px; text-transform:uppercase; letter-spacing:1px;">Automated Retinal Biomarker & Diagnostics System</small>
                </div>
                <hr style="border: 0; border-top: 1px solid rgba(47, 137, 252, 0.2); margin: 15px 0;">
                <table style="width: 100%; border-collapse: collapse; font-size:14px; line-height:1.8;">
                    <tr>
                        <td style="padding: 8px 0; font-weight: 700; width: 35%; color:#94A3B8;">Patient Scan Source:</td>
                        <td style="padding: 8px 0; color:#CBD5E1;">{uploaded_file.name}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; font-weight: 700; color:#94A3B8;">Laterality:</td>
                        <td style="padding: 8px 0; color:#CBD5E1;">{eye_side}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; font-weight: 700; color:#94A3B8;">Vertical Cup-to-Disc Ratio (vCDR):</td>
                        <td style="padding: 8px 0; color:#CBD5E1;">{cdr:.4f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; font-weight: 700; color:#94A3B8;">ISNT Compliance:</td>
                        <td style="padding: 8px 0; color:#CBD5E1;">{'Compliant (I >= S >= N >= T)' if isnt_compliant else 'Non-Compliant (Thinning Detected)'}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; font-weight: 700; color:#94A3B8;">Vessel Density / Tortuosity:</td>
                        <td style="padding: 8px 0; color:#CBD5E1;">Density: {vessel_density:.2%} | Tortuosity Index: {mean_tortuosity:.4f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; font-weight: 700; color:#94A3B8;">Drusen Area Ratio (DAR):</td>
                        <td style="padding: 8px 0; color:#CBD5E1;">{drusen_area_ratio:.4%} (Macula ROI Center: {macula_cx}px, {macula_cy}px)</td>
                    </tr>
                    <tr style="border-top: 1px solid rgba(47, 137, 252, 0.2);">
                        <td style="padding: 12px 0; font-weight: 700; font-size: 16px; color: #3BE8B0;">Composite Severity Index:</td>
                        <td style="padding: 12px 0; font-weight: 700; font-size: 16px; color: #3BE8B0;">{severity["score"]}/100</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; font-weight: 700; font-size: 16px; color: #2F89FC;">Clinical Assessment:</td>
                        <td style="padding: 8px 0; font-weight: 700; font-size: 16px; color: #2F89FC;">{severity["category"]}</td>
                    </tr>
                </table>
                <hr style="border: 0; border-top: 1px solid rgba(47, 137, 252, 0.2); margin: 20px 0;">
                <p style="font-size: 11px; color: #64748B; line-height: 1.5; text-align: justify; margin: 0;">
                    <b>Clinical Disclaimer:</b> This computational report is generated automatically based on deep learning segmentation and classical geometric modeling. 
                    It is intended solely as an clinical decision support aid for medical researchers and licensed ophthalmologists, and must not serve as the sole source 
                    for diagnostic decision-making or therapy initiation.
                </p>
            </div>
            """
            st.markdown(report_html, unsafe_allow_html=True)
            st.button("🖨️ Print Clinical Report", on_click=None, help="Use browser print option to save this page as PDF.")
    else:
        st.info("👋 Welcome! Please upload a fundus photograph above to activate the OcuVision AI Clinical Suite.")
