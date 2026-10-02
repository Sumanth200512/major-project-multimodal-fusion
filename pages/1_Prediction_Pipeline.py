"""
Page 1: Prediction Pipeline
Fingerprint -> Alteration Detection -> Iris -> Multimodal Gender Classification
"""
import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(
    page_title="Prediction Pipeline - Biometric Security",
    page_icon="🛡️",
    layout="wide"
)

from src.utils.ui_helpers import render_sidebar, check_artifacts_status
from src.utils.validation import convert_uploaded_file_to_cv2
from src.inference.pipeline import BiometricInferencePipeline

# Render shared sidebar
render_sidebar()

# Custom styles
st.markdown("""
<style>
    .step-banner {
        background-color: #1e293b;
        border-left: 5px solid #38bdf8;
        padding: 12px 18px;
        margin-bottom: 20px;
        border-radius: 4px;
    }
    .security-alert {
        background-color: #450a0a;
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 20px;
        color: #fecaca;
        margin-top: 15px;
    }
    .security-pass {
        background-color: #064e3b;
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 20px;
        color: #d1fae5;
        margin-top: 15px;
    }
    .result-box {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Prediction Pipeline")
st.caption("Fingerprint → Alteration Detection → Iris → Multimodal Gender Classification")

# Verify model availability
status = check_artifacts_status()
if not all(status.values()):
    st.error("⚠️ **Required trained models are unavailable.** Please use the sidebar to train the models before proceeding.")
    st.stop()

# Initialize Cached Inference Engine
@st.cache_resource
def get_inference_pipeline():
    pipeline = BiometricInferencePipeline()
    pipeline.load_models()
    return pipeline

pipeline = get_inference_pipeline()

# Synchronize dynamically selected classifier from sidebar
active_model_file = st.session_state.get("active_classifier_file", "late_fusion_stacking.joblib")
pipeline.set_active_classifier(active_model_file)

# Manage Session State
if "fp_uploaded" not in st.session_state:
    st.session_state.fp_uploaded = False
if "fp_img_gray" not in st.session_state:
    st.session_state.fp_img_gray = None
if "fp_result" not in st.session_state:
    st.session_state.fp_result = None

if "iris_uploaded" not in st.session_state:
    st.session_state.iris_uploaded = False
if "iris_img_gray" not in st.session_state:
    st.session_state.iris_img_gray = None
if "gender_result" not in st.session_state:
    st.session_state.gender_result = None

# Top Action Toolbar
col_title, col_reset = st.columns([5, 1])
with col_reset:
    if st.button("🔄 Reset Pipeline", use_container_width=True):
        st.session_state.fp_uploaded = False
        st.session_state.fp_img_gray = None
        st.session_state.fp_result = None
        st.session_state.iris_uploaded = False
        st.session_state.iris_img_gray = None
        st.session_state.gender_result = None
        st.rerun()

# Visual Architecture Flow
with st.expander("ℹ️ Multimodal Inference Flow Architecture", expanded=False):
    st.markdown("""
    ```text
    Fingerprint Upload
           ↓
    Fingerprint Alteration Detection (295D: Minutiae + GLCM + LBP + Sobel + HOG)
           ↓
       ┌────────────────────────┐
       │                        │
    Altered                  Unaltered (Real)
       │                        │
    🛑 STOP                     ↓
                           Iris Upload
                                ↓
                     Feature Extraction (297D)
                                ↓
                        MinMax Normalization
                                ↓
                       Multimodal Fusion (594D)
                                ↓
                     Late Fusion Stacking (SVM + Meta Logistic Regression)
                                ↓
                         Gender Prediction
    ```
    """)

st.markdown("---")

# =========================================================================
# STEP 1: FINGERPRINT VERIFICATION
# =========================================================================
st.markdown("""
<div class="step-banner">
    <span style="font-weight: 800; font-size: 16px; color: #38bdf8;">STEP 1: FINGERPRINT VERIFICATION</span>
    <span style="color: #94a3b8; margin-left: 10px;">Upload fingerprint to evaluate integrity and detect spoofing/alteration.</span>
</div>
""", unsafe_allow_html=True)

fp_file = st.file_uploader(
    "Upload Fingerprint Image (.BMP, .PNG, .JPG, .JPEG):",
    type=["bmp", "png", "jpg", "jpeg"],
    key="fp_uploader"
)

if fp_file is not None:
    # Check if a new file was uploaded compared to session state
    file_id = f"{fp_file.name}_{fp_file.size}"
    if st.session_state.get("current_fp_file_id") != file_id:
        st.session_state.current_fp_file_id = file_id
        st.session_state.fp_result = None
        st.session_state.iris_uploaded = False
        st.session_state.iris_img_gray = None
        st.session_state.gender_result = None

    # 1. Immediately convert and display
    img_gray, err = convert_uploaded_file_to_cv2(fp_file)
    if err:
        st.error(f"❌ {err}")
    else:
        st.session_state.fp_uploaded = True
        st.session_state.fp_img_gray = img_gray

        col_img, col_status = st.columns([1, 1])
        with col_img:
            st.markdown("**Uploaded Fingerprint**")
            st.image(img_gray, caption=f"Fingerprint ({img_gray.shape[1]}x{img_gray.shape[0]})", width=260, clamp=True)

        # 2. Run Alteration Check if not already executed for this image
        if st.session_state.fp_result is None:
            with st.spinner("Analyzing fingerprint morphology, texture, and gradient structure..."):
                res = pipeline.verify_fingerprint(img_gray)
                st.session_state.fp_result = res

        fp_res = st.session_state.fp_result

        with col_status:
            st.markdown("**Fingerprint Verification Result**")
            if not fp_res["success"]:
                st.error(f"Verification Error: {fp_res.get('error', 'Unknown')}")
            elif not fp_res["is_real"]:
                # ALTERED DETECTED -> HALT PIPELINE
                st.markdown(f"""
                <div class="security-alert">
                    <div style="font-size: 20px; font-weight: 800; color: #f87171;">🚨 FINGERPRINT ALTERED</div>
                    <div style="font-size: 14px; margin-top: 8px; font-weight: 600;">Pipeline halted.</div>
                    <p style="margin-top: 8px; font-size: 13px; line-height: 1.5;">
                        The uploaded fingerprint was detected as <b>altered / spoofed</b>.<br>
                        Confidence: <b>{fp_res['confidence']:.2%}</b><br>
                        The iris stage cannot proceed due to biometric integrity failure.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # UNALTERED / REAL CONFIRMED
                st.markdown(f"""
                <div class="security-pass">
                    <div style="font-size: 20px; font-weight: 800; color: #34d399;">✅ FINGERPRINT VERIFIED</div>
                    <p style="margin-top: 8px; font-size: 13px; line-height: 1.5;">
                        Fingerprint confirmed as <b>Real / Genuine</b>.<br>
                        Confidence: <b>{fp_res['confidence']:.2%}</b><br>
                        Integrity verified. Proceed to Step 2 below.
                    </p>
                </div>
                """, unsafe_allow_html=True)

        # Explainability section
        with st.expander("🔍 Show Alteration Extracted Features (295D)", expanded=False):
            if fp_res["success"]:
                raw_feats = fp_res["features_raw"]
                st.json({
                    "minutiae_endings": int(raw_feats[0]),
                    "minutiae_density": float(raw_feats[1]),
                    "glcm_contrast": float(raw_feats[2]),
                    "glcm_dissimilarity": float(raw_feats[3]),
                    "glcm_homogeneity": float(raw_feats[4]),
                    "lbp_variance": float(raw_feats[5]),
                    "sobel_gradient_variance": float(raw_feats[6]),
                    "hog_descriptor_dimension": 288,
                    "total_features": len(raw_feats)
                })

# =========================================================================
# STEP 2: IRIS UPLOAD & MULTIMODAL CLASSIFICATION
# (Enabled ONLY if fingerprint is verified as REAL)
# =========================================================================
if st.session_state.fp_result and st.session_state.fp_result.get("is_real", False):
    st.markdown("---")
    st.markdown("""
    <div class="step-banner">
        <span style="font-weight: 800; font-size: 16px; color: #38bdf8;">STEP 2: IRIS UPLOAD & MULTIMODAL GENDER CLASSIFICATION</span>
        <span style="color: #94a3b8; margin-left: 10px;">Upload iris scan for multimodal feature fusion.</span>
    </div>
    """, unsafe_allow_html=True)

    iris_file = st.file_uploader(
        "Upload Iris Image (.BMP, .PNG, .JPG, .JPEG, .TIFF):",
        type=["bmp", "png", "jpg", "jpeg", "tiff"],
        key="iris_uploader"
    )

    if iris_file is not None:
        iris_file_id = f"{iris_file.name}_{iris_file.size}"
        if st.session_state.get("current_iris_file_id") != iris_file_id:
            st.session_state.current_iris_file_id = iris_file_id
            st.session_state.gender_result = None

        iris_gray, i_err = convert_uploaded_file_to_cv2(iris_file)
        if i_err:
            st.error(f"❌ {i_err}")
        else:
            st.session_state.iris_uploaded = True
            st.session_state.iris_img_gray = iris_gray

            # Side-by-side Biometric Display
            st.markdown("#### Dual Biometric Modalities")
            disp_col1, disp_col2 = st.columns(2)
            with disp_col1:
                st.image(st.session_state.fp_img_gray, caption="Verified Fingerprint Modality", width=250, clamp=True)
            with disp_col2:
                st.image(iris_gray, caption=f"Uploaded Iris Modality ({iris_gray.shape[1]}x{iris_gray.shape[0]})", width=250, clamp=True)

            # Multimodal Inference
            if st.session_state.gender_result is None:
                with st.spinner("Extracting multimodal features & running Late Fusion Stacking..."):
                    g_res = pipeline.predict_multimodal_gender(st.session_state.fp_img_gray, iris_gray)
                    st.session_state.gender_result = g_res

            g_res = st.session_state.gender_result

            if not g_res["success"]:
                st.error(f"Multimodal Classification Error: {g_res.get('error')}")
            else:
                st.markdown("---")
                st.markdown("### 🏆 Multimodal Classification Outcome")

                gender_str = g_res["gender"]
                conf = g_res["confidence"]
                badge_color = "#3b82f6" if gender_str == "Male" else "#ec4899"
                gender_icon = "👦" if gender_str == "Male" else "👧"

                st.markdown(f"""
                <div class="result-box">
                    <div style="font-size: 14px; text-transform: uppercase; color: #9ca3af; letter-spacing: 1px;">
                        MULTIMODAL GENDER CLASSIFICATION RESULT
                    </div>
                    <div style="font-size: 38px; font-weight: 800; color: {badge_color}; margin-top: 8px;">
                        {gender_str} {gender_icon}
                    </div>
                    <div style="font-size: 18px; font-weight: 600; color: #e5e7eb; margin-top: 6px;">
                        Model Confidence: {conf:.2%}
                    </div>
                    <div style="font-size: 13px; color: #94a3b8; margin-top: 10px;">
                        Active Model: <b>{st.session_state.get('selected_active_model_name', 'Late Fusion Stacking')}</b><br>
                        P(Male) = {g_res['prob_male']:.2%} &nbsp;|&nbsp; P(Female) = {g_res['prob_female']:.2%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Explainability for Multimodal Feature Fusion
                with st.expander("🔍 Show Multimodal Feature Vectors & Fusion Details", expanded=False):
                    col_f1, col_f2, col_f3 = st.columns(3)
                    with col_f1:
                        st.metric("Iris Features", f"{g_res['iris_feature_dim']}D")
                    with col_f2:
                        st.metric("Fingerprint Features", f"{g_res['fp_feature_dim']}D")
                    with col_f3:
                        st.metric("Concatenated Fusion", f"{g_res['fusion_feature_dim']}D")

                    st.markdown("""
                    **Late Fusion Stacking Execution**:
                    1. **Base Iris Model**: Evaluated on scaled 297D Iris feature vector.
                    2. **Base Fingerprint Model**: Evaluated on scaled 297D Fingerprint feature vector.
                    3. **Meta-Learner**: Probability vectors aggregated through meta Logistic Regression classifier to yield final prediction.
                    """)
