"""
Page 2: Model Comparison
Display experimentally obtained model-comparison results and graphs from the notebook.
"""
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Model Comparison - Biometric Security",
    page_icon="📊",
    layout="wide"
)

from src.config import METRICS_DIR, FIGURES_DIR
from src.utils.ui_helpers import render_sidebar, check_artifacts_status
from src.utils.io import load_json
from src.evaluation.plots import (
    plot_accuracy_comparison,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_metric_comparison,
    plot_metric_heatmap
)

# Render shared sidebar
render_sidebar()

st.title("Model Comparison & Experimental Benchmarks")
st.caption("Empirical evaluation across classifiers, biometric modalities, and fusion paradigms.")

status = check_artifacts_status()
csv_path = METRICS_DIR / "experiment_results.csv"
cms_path = METRICS_DIR / "confusion_matrices.json"
rocs_path = METRICS_DIR / "roc_curves.json"
meta_path = METRICS_DIR.parent.parent / "models" / "metadata.json"

if not (csv_path.exists() and cms_path.exists() and rocs_path.exists()):
    st.warning("⚠️ **Experiment results are not available yet.** Please run model training from the sidebar under 'Train / Retrain Models'.")
    st.stop()

@st.cache_data
def load_all_results():
    df = pd.read_csv(csv_path)
    cms = load_json(cms_path)
    rocs = load_json(rocs_path)
    meta = load_json(meta_path) if meta_path.exists() else {}
    return df, cms, rocs, meta

results_df, cms, rocs, metadata = load_all_results()

# =========================================================================
# EXPERIMENT SUMMARY CARDS
# =========================================================================
st.markdown("### 📋 Experiment Overview")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Benchmark Classifiers", "4", "SVM, RF, HGB, Late Fusion")
with c2:
    st.metric("Modalities Tested", "3", "Iris, FP, Multimodal")
with c3:
    st.metric("Validation Split", "80 / 20", "Zero data leakage")
with c4:
    st.metric("Random State", "42", "Deterministic")

st.markdown("---")

# =========================================================================
# FILTERABLE RESULTS TABLE
# =========================================================================
st.markdown("### 📑 Full Experimental Results Table")

f_col1, f_col2 = st.columns(2)
with f_col1:
    all_classifiers = ["All"] + sorted(results_df["Classifier"].unique().tolist())
    selected_clf = st.selectbox("Filter by Classifier:", all_classifiers, index=0)

with f_col2:
    all_modalities = ["All"] + sorted(results_df["Modality"].unique().tolist())
    selected_mod = st.selectbox("Filter by Modality:", all_modalities, index=0)

filtered_df = results_df.copy()
if selected_clf != "All":
    filtered_df = filtered_df[filtered_df["Classifier"] == selected_clf]
if selected_mod != "All":
    filtered_df = filtered_df[filtered_df["Modality"] == selected_mod]

# Format metrics as percentages
display_df = filtered_df.copy()
for col in ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]:
    display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# =========================================================================
# NOTEBOOK GRAPH 1: ACCURACY COMPARISON
# =========================================================================
st.markdown("### 📈 Accuracy Comparison (Single Modality vs. Multimodal Fusion)")
st.caption("Direct reproduction of the notebook's primary benchmark chart.")
fig_acc = plot_accuracy_comparison(filtered_df if len(filtered_df) > 0 else results_df, interactive=True)
st.plotly_chart(fig_acc, use_container_width=True)

st.markdown("---")

# =========================================================================
# NOTEBOOK GRAPH 2: CONFUSION MATRICES (LATE FUSION STACKING)
# =========================================================================
st.markdown("### 🎯 Confusion Matrices (Late Fusion Stacking Classifier)")
st.caption("Distribution of True vs. Predicted classifications for Iris Only, Fingerprint Only, and Multimodal Fusion.")

norm_toggle = st.toggle("Show Normalized Confusion Matrices", value=False)
fig_cms = plot_confusion_matrices(cms, normalize=norm_toggle, interactive=True)
st.plotly_chart(fig_cms, use_container_width=True)

st.markdown("---")

# =========================================================================
# NOTEBOOK GRAPH 3: ROC CURVES (LATE FUSION STACKING)
# =========================================================================
st.markdown("### 📉 ROC Curves across Biometric Modalities")
st.caption("Receiver Operating Characteristic (FPR vs. TPR) demonstrating area under the curve (AUC).")
fig_rocs = plot_roc_curves(rocs, interactive=True)
st.plotly_chart(fig_rocs, use_container_width=True)

st.markdown("---")

# =========================================================================
# ADDITIONAL COMPARISON GRAPHS (PRECISION, RECALL, F1, AUC)
# =========================================================================
st.markdown("### 📊 Additional Comparative Metric Benchmarks")

tab_prec, tab_rec, tab_f1, tab_auc = st.tabs([
    "Precision Comparison",
    "Recall Comparison",
    "F1 Score Comparison",
    "AUC Comparison"
])

with tab_prec:
    fig_p = plot_metric_comparison(filtered_df if len(filtered_df) > 0 else results_df, "Precision", interactive=True)
    st.plotly_chart(fig_p, use_container_width=True)

with tab_rec:
    fig_r = plot_metric_comparison(filtered_df if len(filtered_df) > 0 else results_df, "Recall", interactive=True)
    st.plotly_chart(fig_r, use_container_width=True)

with tab_f1:
    fig_f = plot_metric_comparison(filtered_df if len(filtered_df) > 0 else results_df, "F1 Score", interactive=True)
    st.plotly_chart(fig_f, use_container_width=True)

with tab_auc:
    fig_a = plot_metric_comparison(filtered_df if len(filtered_df) > 0 else results_df, "AUC", interactive=True)
    st.plotly_chart(fig_a, use_container_width=True)

st.markdown("---")

# =========================================================================
# RECOMMENDED ADDITIONAL VISUALIZATION: METRIC HEATMAP
# =========================================================================
st.markdown("### 🗺️ Comprehensive Performance Heatmap")
st.caption("Multi-attribute comparison across all model configurations.")
fig_hm = plot_metric_heatmap(results_df, interactive=True)
st.plotly_chart(fig_hm, use_container_width=True)

st.markdown("---")

# =========================================================================
# INFERENCE PIPELINE ARCHITECTURE DIAGRAM
# =========================================================================
st.markdown("### 📐 End-to-End System Pipeline Architecture")
st.markdown("""
```text
                      [ Input Fingerprint Image ]
                                   │
                      ┌────────────┴────────────┐
                      ▼                         ▼
             Morphological Thinning       HOG & GLCM Analysis
             (CN Minutiae Analysis)      (Gradient & Texture)
                      │                         │
                      └────────────┬────────────┘
                                   ▼
               [ Alteration Classifier (HistGradientBoosting) ]
                                   │
                   ┌───────────────┴───────────────┐
                   │                               │
            [ Altered (0) ]                 [ Genuine (1) ]
                   │                               │
             🛑 HALT & LOG                         ▼
                                        [ Input Iris Image ]
                                                   │
                            ┌──────────────────────┴──────────────────────┐
                            ▼                                             ▼
                  Iris Feature Extraction                     Fingerprint Feature Extraction
                  (9 Base + 288 HOG = 297D)                     (9 Base + 288 HOG = 297D)
                            │                                             │
                            └──────────────────────┬──────────────────────┘
                                                   ▼
                                      [ Normalization (MinMax) ]
                                                   │
                            ┌──────────────────────┴──────────────────────┐
                            ▼                                             ▼
                    Early Fusion (594D)                           Late Fusion Stacking
               (Direct Vector Concatenation)                  (Out-of-fold Meta-Classifier)
                            │                                             │
                            └──────────────────────┬──────────────────────┘
                                                   ▼
                                  [ Final Gender Prediction & Metrics ]
```
""")
