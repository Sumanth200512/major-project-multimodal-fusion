"""
Biometric Security Dashboard - Main Application Entrypoint.
Multimodal Fingerprint + Iris Gender Classification & Integrity Verification.
"""
import streamlit as st

st.set_page_config(
    page_title="Biometric Security Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.utils.ui_helpers import render_sidebar, check_artifacts_status

# Render unified sidebar
render_sidebar()

# Custom Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
    }
    .main-title {
        color: #f8fafc;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .main-subtitle {
        color: #38bdf8;
        font-size: 15px;
        font-weight: 600;
        margin-top: 4px;
        margin-bottom: 8px;
    }
    .main-desc {
        color: #94a3b8;
        font-size: 14px;
        margin: 0;
        line-height: 1.5;
    }
    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Main Application Header
st.markdown("""
<div class="main-header">
    <div class="main-title">BIOMETRIC SECURITY DASHBOARD</div>
    <div class="main-subtitle">Multimodal Fingerprint + Iris Gender Classification</div>
    <p class="main-desc">
        Fingerprint integrity verification followed by multimodal biometric classification.
        Demonstrates pre-classification alteration/spoof detection and multimodal feature fusion (Late Fusion Stacking).
    </p>
</div>
""", unsafe_allow_html=True)

# Quick Navigation and System Overview
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("### 📌 Application Workflow")
    st.markdown("""
    The application is divided into two primary operational modules:
    
    1. **Prediction Pipeline (`pages/1_Prediction_Pipeline.py`)**:
       - **Step 1: Fingerprint Verification**: Upload fingerprint image (`.bmp`, `.png`, `.jpg`). The system computes morphological minutiae, GLCM textures, LBP, and 288D HOG features ($295$ total dimensions) and tests for alteration using `HistGradientBoostingClassifier`.
       - **Security Gate**: If the fingerprint is detected as **Altered / Spoofed**, the pipeline immediately **HALTS**.
       - **Step 2: Multimodal Gender Classification**: If genuine, an Iris scan is uploaded. Features ($297$D each) are extracted, scaled, concatenated, and evaluated by the **Late Fusion Stacking Classifier** to determine gender with probabilistic confidence.
       
    2. **Model Comparison (`pages/2_Model_Comparison.py`)**:
       - Inspect the exact experimental benchmark comparing **SVM (RBF Kernel)**, **Random Forest**, **Hist Gradient Boosting**, and **Late Fusion Stacking**.
       - Evaluate unimodal baselines (**Iris Only**, **Fingerprint Only**) versus **Early Fusion** and **Late Fusion Stacking**.
       - Interactive charts reproducing the notebook's Accuracy comparison, Confusion Matrices, ROC curves, and extended metrics.
    """)

with col_right:
    st.markdown("### 📊 Architecture & State Overview")
    status = check_artifacts_status()
    all_ready = all(status.values())

    if all_ready:
        st.success("✅ **Production Models Ready**: All model weights, scalers, and benchmark metrics are loaded and ready for immediate inference.")
    else:
        st.warning("⚠️ **Models Incomplete or Missing**: Open the sidebar under **Model Management** and select **Train / Retrain Models** to compile the dataset and train models.")

    st.markdown("""
    <div style="background:#1e293b; padding:16px; border-radius:8px; border:1px solid #334155;">
        <div style="font-weight:700; color:#38bdf8; margin-bottom:8px;">Pipeline Dataflow</div>
        <div style="font-family:monospace; font-size:12px; color:#cbd5e1; line-height:1.6;">
            Fingerprint ──> Alteration Check [295D]<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Altered ──> 🛑 <b>HALT</b><br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── Real ──> Iris Upload<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Iris [297D]<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── FP [297D]<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── Late Fusion Stacking ──> <b>Gender</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.info("💡 **Getting Started**: Use the sidebar navigation on the left to go to **Prediction Pipeline** or **Model Comparison**.")
