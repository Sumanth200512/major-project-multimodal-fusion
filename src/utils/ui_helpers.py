"""
Shared Streamlit UI helpers and sidebar components.
"""
from typing import Dict, Any, Tuple
import streamlit as st
import pandas as pd
from pathlib import Path

from src.config import (
    ALTERATION_MODELS_DIR, CLASSIFICATION_MODELS_DIR, SCALERS_DIR,
    METRICS_DIR, FIGURES_DIR
)
from src.data.dataset_manager import DatasetManager
from src.models.trainer import ModelTrainer
from src.utils.io import load_json

def check_artifacts_status() -> Dict[str, bool]:
    """Check existence of all key artifacts."""
    return {
        "alteration_detector": (ALTERATION_MODELS_DIR / "alteration_detector.joblib").exists(),
        "alteration_scaler": (SCALERS_DIR / "alteration_scaler.joblib").exists(),
        "iris_scaler": (SCALERS_DIR / "iris_scaler.joblib").exists(),
        "fingerprint_scaler": (SCALERS_DIR / "fingerprint_scaler.joblib").exists(),
        "fusion_classifier": (CLASSIFICATION_MODELS_DIR / "late_fusion_stacking.joblib").exists(),
        "experiment_results": (METRICS_DIR / "experiment_results.csv").exists()
    }

def render_sidebar():
    """Render unified professional sidebar for both pages."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding-bottom: 15px;">
            <div style="font-size: 26px; font-weight: 800; color: #14b8a6; letter-spacing: 0.5px;">BIOGUARD AI</div>
            <div style="font-size: 11px; text-transform: uppercase; color: #9ca3af; letter-spacing: 1px;">Biometric Security Research</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        
        # 1. Model Status Section
        st.markdown("#### 🛡️ Model Artifact Status")
        status = check_artifacts_status()
        
        col1, col2 = st.columns(2)
        with col1:
            if status["alteration_detector"] and status["alteration_scaler"]:
                st.markdown("<span style='color:#10b981; font-size:13px;'>● Alteration: Ready</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:#ef4444; font-size:13px;'>○ Alteration: Missing</span>", unsafe_allow_html=True)
                
            if status["iris_scaler"] and status["fingerprint_scaler"]:
                st.markdown("<span style='color:#10b981; font-size:13px;'>● Scalers: Ready</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:#ef4444; font-size:13px;'>○ Scalers: Missing</span>", unsafe_allow_html=True)

        with col2:
            if status["fusion_classifier"]:
                st.markdown("<span style='color:#10b981; font-size:13px;'>● Fusion: Ready</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:#ef4444; font-size:13px;'>○ Fusion: Missing</span>", unsafe_allow_html=True)

            if status["experiment_results"]:
                st.markdown("<span style='color:#10b981; font-size:13px;'>● Metrics: Ready</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:#ef4444; font-size:13px;'>○ Metrics: Missing</span>", unsafe_allow_html=True)

        st.markdown("---")

        # 2. Dataset Status Section
        st.markdown("#### 📁 Local Dataset Status")
        dm = DatasetManager()
        s_avail, s_info = dm.check_socofing_available()
        i_avail, i_info = dm.check_iris_available()

        if s_avail:
            st.markdown(f"<span style='color:#10b981; font-size:13px;'>● SOCOFing: Available ({s_info.get('real_count', 0)} Real)</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#f59e0b; font-size:13px;'>○ SOCOFing: Missing</span>", unsafe_allow_html=True)

        if i_avail:
            st.markdown(f"<span style='color:#10b981; font-size:13px;'>● Iris GFI: Available ({i_info.get('image_count', 0)} Imgs)</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#f59e0b; font-size:13px;'>○ Iris GFI: Missing</span>", unsafe_allow_html=True)

        if not (s_avail and i_avail):
            if st.button("📥 Download Datasets (KaggleHub)", use_container_width=True):
                with st.spinner("Downloading datasets via KaggleHub..."):
                    try:
                        dm.download_datasets()
                        st.success("Datasets downloaded successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Download failed: {e}")

        st.markdown("---")

        # 3. Model Management & Training Controls
        st.markdown("#### ⚙️ Model Management")
        workflow_choice = st.radio(
            "Select Pipeline Source:",
            ["Use Existing Models", "Train / Retrain Models"],
            index=0,
            key="pipeline_workflow_choice"
        )

        if workflow_choice == "Use Existing Models":
            st.markdown("##### 🎯 Multimodal Model Selector")
            # Map friendly names to saved model artifacts
            model_options = {
                "Late Fusion Stacking (Recommended)": "late_fusion_stacking.joblib",
                "SVM (RBF Kernel) - Fusion": "fusion_classifier.joblib",
                "Random Forest - Fusion": "random_forest_model_3.joblib",
                "Hist Gradient Boosting - Fusion": "hist_gradient_boosting_model_3.joblib"
            }

            # Filter to only existing files
            available_options = {
                k: v for k, v in model_options.items()
                if (CLASSIFICATION_MODELS_DIR / v).exists()
            }

            if not available_options:
                st.warning("No classification models found in models/classification.")
            else:
                selected_model_name = st.selectbox(
                    "Active Classifier for Prediction:",
                    list(available_options.keys()),
                    index=0,
                    key="selected_active_model_name"
                )
                selected_file = available_options[selected_model_name]
                st.session_state["active_classifier_file"] = selected_file
                st.caption(f"Artifact: `{selected_file}`")

        elif workflow_choice == "Train / Retrain Models":
            st.warning("Training will execute feature extraction, train/test splitting (80/20), model fitting, and evaluation benchmark.")
            
            with st.expander("Training Configuration", expanded=False):
                alt_samples = st.slider("Alteration Samples (per class)", 200, 3000, 3000, 200, key="training_config_alt_samples_v2")
                multi_cohort = st.slider("Multimodal Cohort Size (per gender)", 200, 1000, 1000, 100, key="training_config_multi_cohort_v2")
                force_extract = st.checkbox("Force Re-extract Features", value=False)

            if st.button("🚀 Start Training Pipeline", type="primary", use_container_width=True):
                trainer = ModelTrainer(dm)
                prog_bar = st.progress(0, text="Starting training...")
                status_box = st.empty()

                def update_progress(pct: float, text: str):
                    prog_bar.progress(int(pct * 100), text=text)
                    status_box.info(text)

                try:
                    res = trainer.train_all_models(
                        force_recompute_features=force_extract,
                        alteration_samples=alt_samples,
                        multimodal_cohort_samples=multi_cohort,
                        progress_cb=update_progress
                    )
                    st.success("🎉 Models successfully trained and all artifacts persisted!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")

        st.markdown("---")
        st.caption("🔒 Research Demonstration Application. Data processed locally in memory. No biometric data permanently saved.")
