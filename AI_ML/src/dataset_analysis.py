"""
EMC-GUARD: Physics-Informed Digital Twin for Elevator EMC Pre-Compliance
=========================================================================
MODULE 1 | Script 4: Dataset Analysis and Visualisations
File   : dataset_analysis.py
Author : EMC-GUARD Development Team

This script generates all visualisations for Module 1:
  1.  Missing value heatmap
  2.  Feature distribution histograms (numerical)
  3.  Target distribution histograms
  4.  Correlation heatmap (numerical features + targets)
  5.  Boxplots for numerical feature outlier inspection
  6.  Boxplots for target variable outlier inspection
  7.  Categorical feature bar charts
  8.  Feature vs. Target scatter matrix (sampled)
  9.  Outlier_Flag distribution (post-cleaning)
  10. Shield_Type × Victim_Voltage violin plot

All figures are saved to reports/visualisations/.

Usage
-----33
    python src/dataset_analysis.py

Outputs
-------
    reports/visualisations/*.png
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV    = os.path.join(BASE_DIR, "dataset", "dummy_simulation_dataset.csv")
CLEAN_CSV  = os.path.join(BASE_DIR, "dataset", "EMC_GUARD_Physics_Dataset.csv")
VIS_DIR    = os.path.join(BASE_DIR, "reports", "visualisations")

NUMERICAL_FEATURES = [
    "Separation_mm", "Overlap_mm", "Cable_Length_mm", "Frequency_Hz",
    "Voltage_V", "Conductor_Radius_mm", "Dielectric_Constant",
]
CATEGORICAL_FEATURES = ["Shield_Type", "Cable_Configuration"]
TARGET_COLS = [
    "MutualInductance_nH", "MutualCapacitance_pF",
    "Victim_Voltage_V", "Victim_RMS_V", "Victim_PeakToPeak_V",
]

# Global plot aesthetics
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
TITLE_SUFFIX = "\n[SYNTHETIC DUMMY DATA — FOR PIPELINE TESTING ONLY]"
DPI = 120


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_csv(path: str, label: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, comment="#")
        print(f"  Loaded {label}: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)


def save_fig(fig, name: str) -> None:
    os.makedirs(VIS_DIR, exist_ok=True)
    filepath = os.path.join(VIS_DIR, name)
    fig.savefig(filepath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: reports/visualisations/{name}")


def disclaimer_text(ax=None) -> None:
    """Add a small disclaimer watermark to the bottom of a figure."""
    if ax is not None:
        ax.figure.text(
            0.5, 0.01,
            "SYNTHETIC DUMMY DATA — EMC-GUARD pipeline testing only",
            ha="center", fontsize=7, color="grey", style="italic"
        )


# ---------------------------------------------------------------------------
# PLOT FUNCTIONS
# ---------------------------------------------------------------------------

def plot_missing_values(df_raw: pd.DataFrame) -> None:
    """Heatmap of missing values in the raw dataset."""
    numerical_only = df_raw.select_dtypes(include=[np.number])
    missing_matrix = numerical_only.isnull()

    if missing_matrix.sum().sum() == 0:
        # No missing values — create an informational bar chart instead
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No missing values in numerical columns",
                ha="center", va="center", fontsize=14, color="green")
        ax.set_axis_off()
        ax.set_title("Missing Value Analysis" + TITLE_SUFFIX, pad=14)
        save_fig(fig, "01_missing_value_analysis.png")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(missing_matrix, cbar=True, yticklabels=False,
                cmap="viridis", ax=ax)
    ax.set_title("Missing Value Heatmap (Raw Dataset)" + TITLE_SUFFIX, pad=14)
    ax.set_xlabel("Features")
    disclaimer_text(ax)
    save_fig(fig, "01_missing_value_analysis.png")


def plot_feature_distributions(df_clean: pd.DataFrame) -> None:
    """Histogram + KDE for each numerical input feature."""
    cols = [c for c in NUMERICAL_FEATURES if c in df_clean.columns]
    n = len(cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        data = df_clean[col].dropna()
        sns.histplot(data, kde=True, ax=axes[i], color="steelblue", bins=30)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Count")

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numerical Feature Distributions (Cleaned Dataset)" + TITLE_SUFFIX,
                 fontsize=12, y=1.01)
    fig.tight_layout()
    disclaimer_text()
    save_fig(fig, "02_feature_distributions.png")


def plot_target_distributions(df_clean: pd.DataFrame) -> None:
    """Histogram + KDE for each target variable."""
    cols = [c for c in TARGET_COLS if c in df_clean.columns]
    n = len(cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        data = df_clean[col].dropna()
        sns.histplot(data, kde=True, ax=axes[i], color="coral", bins=30)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Count")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Target Variable Distributions (Cleaned Dataset)" + TITLE_SUFFIX,
                 fontsize=12, y=1.01)
    fig.tight_layout()
    save_fig(fig, "03_target_distributions.png")


def plot_correlation_heatmap(df_clean: pd.DataFrame) -> None:
    """Pearson correlation heatmap for all numerical columns."""
    num_cols = [c for c in NUMERICAL_FEATURES + TARGET_COLS if c in df_clean.columns]
    corr = df_clean[num_cols].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))   # upper triangle
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8}
    )
    ax.set_title("Pearson Correlation Heatmap\n(Numerical Features + Targets)" + TITLE_SUFFIX,
                 pad=14)
    fig.tight_layout()
    save_fig(fig, "04_correlation_heatmap.png")


def plot_feature_boxplots(df_clean: pd.DataFrame) -> None:
    """Boxplots for numerical features to highlight outliers."""
    cols = [c for c in NUMERICAL_FEATURES if c in df_clean.columns]
    n = len(cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.boxplot(y=df_clean[col].dropna(), ax=axes[i],
                    color="lightblue", flierprops=dict(marker="o", color="red", ms=3))
        axes[i].set_title(col, fontsize=10)
        axes[i].set_ylabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numerical Feature Boxplots — Outlier Inspection" + TITLE_SUFFIX,
                 fontsize=12, y=1.01)
    fig.tight_layout()
    save_fig(fig, "05_feature_boxplots.png")


def plot_target_boxplots(df_clean: pd.DataFrame) -> None:
    """Boxplots for target variables."""
    cols = [c for c in TARGET_COLS if c in df_clean.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 5))
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        sns.boxplot(y=df_clean[col].dropna(), ax=ax,
                    color="lightsalmon", flierprops=dict(marker="o", color="navy", ms=3))
        ax.set_title(col, fontsize=9)
        ax.set_ylabel("")

    fig.suptitle("Target Variable Boxplots — Outlier Inspection" + TITLE_SUFFIX,
                 fontsize=12, y=1.03)
    fig.tight_layout()
    save_fig(fig, "06_target_boxplots.png")


def plot_categorical_distributions(df_clean: pd.DataFrame) -> None:
    """Count plots for categorical features, coloured by Cable_Configuration."""
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df_clean.columns]
    fig, axes = plt.subplots(1, len(cat_cols), figsize=(6 * len(cat_cols), 5))
    if len(cat_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cat_cols):
        order = df_clean[col].value_counts().index.tolist()
        sns.countplot(data=df_clean, x=col, ax=ax, order=order,
                      palette="Set2", edgecolor="black")
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height())}",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha="center", va="bottom", fontsize=9)

    fig.suptitle("Categorical Feature Distributions" + TITLE_SUFFIX,
                 fontsize=12, y=1.01)
    fig.tight_layout()
    save_fig(fig, "07_categorical_distributions.png")


def plot_scatter_feature_vs_target(df_clean: pd.DataFrame) -> None:
    """
    Scatter plots: key features vs. MutualInductance_nH coloured by Shield_Type.
    Sampled to 300 points for readability.
    """
    if "MutualInductance_nH" not in df_clean.columns:
        return

    key_features = [c for c in ["Separation_mm", "Overlap_mm", "Frequency_Hz"]
                    if c in df_clean.columns]
    if not key_features:
        return

    sample = df_clean.sample(min(300, len(df_clean)), random_state=42)

    fig, axes = plt.subplots(1, len(key_features), figsize=(6 * len(key_features), 5))
    if len(key_features) == 1:
        axes = [axes]

    palette = {"Shielded": "#2196F3", "Unshielded": "#FF5722"}

    for ax, feat in zip(axes, key_features):
        if "Shield_Type" in sample.columns:
            for stype, grp in sample.groupby("Shield_Type"):
                ax.scatter(grp[feat], grp["MutualInductance_nH"],
                           label=stype, alpha=0.6, s=20,
                           color=palette.get(stype, "grey"))
        else:
            ax.scatter(sample[feat], sample["MutualInductance_nH"],
                       alpha=0.6, s=20)
        ax.set_xlabel(feat)
        ax.set_ylabel("MutualInductance_nH")
        ax.legend(title="Shield_Type", fontsize=8)

    fig.suptitle(
        "Key Features vs. MutualInductance_nH\n(sample of 300; coloured by Shield_Type)"
        + TITLE_SUFFIX,
        fontsize=11, y=1.01
    )
    fig.tight_layout()
    save_fig(fig, "08_scatter_feature_vs_target.png")


def plot_outlier_flag_distribution(df_clean: pd.DataFrame) -> None:
    """Bar chart showing proportion of flagged vs. clean records."""
    if "Outlier_Flag" not in df_clean.columns:
        return

    counts = df_clean["Outlier_Flag"].value_counts().sort_index()
    labels = ["Clean (0)", "Flagged outlier (1)"]
    colors = ["#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels[:len(counts)], counts.values, color=colors[:len(counts)],
                  edgecolor="black")
    for bar in bars:
        ax.annotate(f"{int(bar.get_height())}",
                    (bar.get_x() + bar.get_width() / 2., bar.get_height()),
                    ha="center", va="bottom")
    ax.set_ylabel("Number of Records")
    ax.set_title("Outlier_Flag Distribution (Post-Cleaning)" + TITLE_SUFFIX)
    save_fig(fig, "09_outlier_flag_distribution.png")


def plot_shield_violin(df_clean: pd.DataFrame) -> None:
    """Violin plot: Shield_Type × Victim_Voltage_V, split by Cable_Configuration."""
    if not all(c in df_clean.columns for c in
               ["Shield_Type", "Victim_Voltage_V", "Cable_Configuration"]):
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=df_clean, x="Shield_Type", y="Victim_Voltage_V",
        hue="Cable_Configuration", split=False, inner="quartile",
        palette="Set3", ax=ax, cut=0
    )
    ax.set_title("Victim Voltage by Shield Type and Cable Configuration" + TITLE_SUFFIX)
    ax.set_xlabel("Shield Type")
    ax.set_ylabel("Victim_Voltage_V")
    ax.legend(title="Cable Configuration", loc="upper right")
    save_fig(fig, "10_shield_victim_voltage_violin.png")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("EMC-GUARD  |  Module 1  |  Dataset Analysis & Visualisations")
    print("=" * 65)

    os.makedirs(VIS_DIR, exist_ok=True)

    print("\n[1/2] Loading datasets …")
    df_raw   = load_csv(RAW_CSV,   "raw dataset")
    df_clean = load_csv(CLEAN_CSV, "cleaned dataset")

    print("\n[2/2] Generating visualisations …")

    plot_missing_values(df_raw)
    plot_feature_distributions(df_clean)
    plot_target_distributions(df_clean)
    plot_correlation_heatmap(df_clean)
    plot_feature_boxplots(df_clean)
    plot_target_boxplots(df_clean)
    plot_categorical_distributions(df_clean)
    plot_scatter_feature_vs_target(df_clean)
    plot_outlier_flag_distribution(df_clean)
    plot_shield_violin(df_clean)

    print("\n" + "=" * 65)
    print("[DONE] All visualisations saved to: reports/visualisations/")
    print("=" * 65)


if __name__ == "__main__":
    main()
