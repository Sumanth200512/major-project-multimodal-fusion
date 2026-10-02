"""
Plotting utilities to faithfully reproduce all notebook graphs and additional required charts.
Supports both interactive Plotly figures and static Matplotlib figures.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import FIGURES_DIR
from src.utils.io import get_logger

logger = get_logger(__name__)

# Set clean default theme
sns.set_theme(style="whitegrid")

def plot_accuracy_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[Path] = None,
    interactive: bool = True
) -> Any:
    """
    Notebook Graph 1:
    Accuracy Comparison across Classifiers and Modalities.
    Titled: "Gender Classification: Single Modality vs. Multimodal Fusion"
    """
    if interactive:
        fig = px.bar(
            results_df,
            x="Modality",
            y="Accuracy",
            color="Classifier",
            barmode="group",
            text_auto=".1%",
            title="Gender Classification: Single Modality vs. Multimodal Fusion",
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig.update_layout(
            yaxis=dict(range=[0.5, 1.0], title="Accuracy"),
            xaxis=dict(title="Biometric Modality"),
            legend=dict(title="Classifier", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="sans-serif", size=13),
            margin=dict(l=40, r=40, t=80, b=40)
        )
        if save_path:
            fig.write_image(str(save_path))
        return fig
    else:
        plt.figure(figsize=(11, 5))
        ax = sns.barplot(data=results_df, x="Modality", y="Accuracy", hue="Classifier", palette="viridis")
        plt.title("Gender Classification: Single Modality vs. Multimodal Fusion", fontsize=14, fontweight='bold')
        plt.ylim(0.5, 1.0)
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(str(save_path), dpi=300)
            plt.close()
        return plt.gcf()

def plot_confusion_matrices(
    cms_dict: Dict[str, Any],
    normalize: bool = False,
    save_path: Optional[Path] = None,
    interactive: bool = True
) -> Any:
    """
    Notebook Graph 2:
    Confusion Matrices for Late Fusion Stacking across the 3 Modalities:
    - Iris Only
    - Fingerprint Only
    - Iris + Fingerprint (Fusion)
    """
    target_keys = [k for k in cms_dict.keys() if "Late Fusion Stacking" in k]
    if not target_keys:
        target_keys = list(cms_dict.keys())[:3]

    if interactive:
        fig = make_subplots(
            rows=1, cols=len(target_keys),
            subplot_titles=[k.split(" - ")[1] if " - " in k else k for k in target_keys]
        )
        for idx, key in enumerate(target_keys, 1):
            cm = np.array(cms_dict[key])
            if normalize and cm.sum() > 0:
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                text_vals = [[f"{cm[i, j]}<br>({cm_norm[i, j]:.1%})" for j in range(2)] for i in range(2)]
                z_vals = cm_norm
            else:
                text_vals = [[str(cm[i, j]) for j in range(2)] for i in range(2)]
                z_vals = cm

            heatmap = go.Heatmap(
                z=z_vals,
                x=['Female', 'Male'],
                y=['Female', 'Male'],
                text=text_vals,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=(idx == len(target_keys))
            )
            fig.add_trace(heatmap, row=1, col=idx)
            fig.update_xaxes(title_text="Predicted", row=1, col=idx)
            fig.update_yaxes(title_text="Actual" if idx == 1 else "", row=1, col=idx)

        fig.update_layout(
            title_text=f"Confusion Matrices (Late Fusion Stacking Classifier){' - Normalized' if normalize else ''}",
            font=dict(family="sans-serif", size=12),
            margin=dict(l=40, r=40, t=80, b=40)
        )
        if save_path:
            fig.write_image(str(save_path))
        return fig
    else:
        fig, axes = plt.subplots(1, len(target_keys), figsize=(15, 4))
        for idx, key in enumerate(target_keys):
            cm = np.array(cms_dict[key])
            if normalize and cm.sum() > 0:
                cm_disp = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                fmt = '.2%'
            else:
                cm_disp = cm
                fmt = 'd'
            sns.heatmap(cm_disp, annot=True, fmt=fmt, cmap='Blues', ax=axes[idx], cbar=False,
                        xticklabels=['Female', 'Male'], yticklabels=['Female', 'Male'])
            title_part = key.split(" - ")[1] if " - " in key else key
            axes[idx].set_title(title_part, fontweight='bold')
            axes[idx].set_xlabel("Predicted")
            axes[idx].set_ylabel("Actual")
        plt.suptitle("Confusion Matrices (Late Fusion Stacking Classifier)", fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        if save_path:
            plt.savefig(str(save_path), dpi=300)
            plt.close()
        return fig

def plot_roc_curves(
    rocs_dict: Dict[str, Any],
    save_path: Optional[Path] = None,
    interactive: bool = True
) -> Any:
    """
    Notebook Graph 3:
    ROC Curves for Late Fusion Stacking across modalities.
    """
    target_keys = [k for k in rocs_dict.keys() if "Late Fusion Stacking" in k]
    if not target_keys:
        target_keys = list(rocs_dict.keys())[:3]

    if interactive:
        fig = go.Figure()
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for idx, key in enumerate(target_keys):
            roc_data = rocs_dict[key]
            fpr = roc_data["fpr"] if isinstance(roc_data, dict) else roc_data[0]
            tpr = roc_data["tpr"] if isinstance(roc_data, dict) else roc_data[1]
            roc_auc = roc_data["auc"] if isinstance(roc_data, dict) else roc_data[2]
            name_part = key.split(" - ")[1] if " - " in key else key
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f"{name_part} (AUC = {roc_auc:.3f})",
                line=dict(width=2.5, color=colors[idx % len(colors)])
            ))

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Baseline (AUC = 0.500)',
            line=dict(dash='dash', color='gray')
        ))
        fig.update_layout(
            title="ROC Curves - Modality Comparison (Late Fusion Stacking)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.55, y=0.1, bgcolor='rgba(255,255,255,0.8)'),
            font=dict(family="sans-serif", size=13),
            margin=dict(l=40, r=40, t=80, b=40)
        )
        if save_path:
            fig.write_image(str(save_path))
        return fig
    else:
        plt.figure(figsize=(8, 5))
        for key in target_keys:
            roc_data = rocs_dict[key]
            fpr = roc_data["fpr"] if isinstance(roc_data, dict) else roc_data[0]
            tpr = roc_data["tpr"] if isinstance(roc_data, dict) else roc_data[1]
            roc_auc = roc_data["auc"] if isinstance(roc_data, dict) else roc_data[2]
            name_part = key.split(" - ")[1] if " - " in key else key
            plt.plot(fpr, tpr, lw=2, label=f"{name_part} (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Modality Comparison (Late Fusion Stacking)', fontweight='bold')
        plt.legend(loc="lower right")
        plt.tight_layout()
        if save_path:
            plt.savefig(str(save_path), dpi=300)
            plt.close()
        return plt.gcf()

def plot_metric_comparison(
    results_df: pd.DataFrame,
    metric_name: str,
    save_path: Optional[Path] = None,
    interactive: bool = True
) -> Any:
    """
    Additional Graphs:
    Precision, Recall, F1 Score, or AUC comparison across classifiers and modalities.
    """
    if interactive:
        fig = px.bar(
            results_df,
            x="Modality",
            y=metric_name,
            color="Classifier",
            barmode="group",
            text_auto=".1%",
            title=f"{metric_name} Comparison: Classifier × Modality",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig.update_layout(
            yaxis=dict(range=[0.5, 1.0], title=metric_name),
            xaxis=dict(title="Biometric Modality"),
            legend=dict(title="Classifier", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="sans-serif", size=13),
            margin=dict(l=40, r=40, t=80, b=40)
        )
        if save_path:
            fig.write_image(str(save_path))
        return fig
    else:
        plt.figure(figsize=(11, 5))
        ax = sns.barplot(data=results_df, x="Modality", y=metric_name, hue="Classifier", palette="crest")
        plt.title(f"{metric_name} Comparison: Classifier × Modality", fontsize=14, fontweight='bold')
        plt.ylim(0.5, 1.0)
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(str(save_path), dpi=300)
            plt.close()
        return plt.gcf()

def plot_metric_heatmap(
    results_df: pd.DataFrame,
    save_path: Optional[Path] = None,
    interactive: bool = True
) -> Any:
    """
    Recommended Additional Visualization:
    Heatmap of all metrics across all (Classifier, Modality) combinations.
    """
    df = results_df.copy()
    df["Model Configuration"] = df["Classifier"] + " (" + df["Modality"].str.replace("Model [123]: ", "") + ")"
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
    heatmap_data = df.set_index("Model Configuration")[metrics]

    if interactive:
        fig = px.imshow(
            heatmap_data,
            text_auto=".1%",
            aspect="auto",
            color_continuous_scale="Viridis",
            title="Comprehensive Performance Heatmap (All Models & Modalities)"
        )
        fig.update_layout(
            font=dict(family="sans-serif", size=12),
            margin=dict(l=100, r=40, t=60, b=40)
        )
        if save_path:
            fig.write_image(str(save_path))
        return fig
    else:
        plt.figure(figsize=(10, 7))
        sns.heatmap(heatmap_data, annot=True, fmt=".1%", cmap="viridis", cbar=True)
        plt.title("Comprehensive Performance Heatmap", fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(str(save_path), dpi=300)
            plt.close()
        return plt.gcf()
