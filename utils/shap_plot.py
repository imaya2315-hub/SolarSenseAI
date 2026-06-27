"""
shap_plot.py – Generate SHAP-style explainability charts using matplotlib.

Since the `shap` package is not available in this environment, we use
gradient-based perturbation analysis to mimic SHAP bar charts.
All plots are saved to static/plots/ and referenced by filename.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "static", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Colour palette matching the SolarSense brand
SOLAR_ORANGE  = "#F97316"
SOLAR_AMBER   = "#FBBF24"
SOLAR_BLUE    = "#0EA5E9"
SOLAR_DARK    = "#0F172A"
BACKGROUND    = "#F8FAFC"


def _save_path(prefix: str) -> tuple[str, str]:
    """Return (abs_path, filename) for a new plot file."""
    fname = f"{prefix}_{int(time.time()*1000)}.png"
    return os.path.join(PLOT_DIR, fname), fname


def feature_importance_bar(importances: list[dict]) -> str:
    """
    Horizontal bar chart of feature importances (SHAP-style).
    Returns the filename (relative to static/plots/).
    """
    names  = [d["name"] for d in importances]
    values = [d["importance"] for d in importances]
    colors = [SOLAR_ORANGE if v == max(values) else SOLAR_AMBER for v in values]

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    bars = ax.barh(names, values, color=colors, height=0.55, zorder=3)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", ha="left",
            fontsize=10, fontweight="bold", color=SOLAR_DARK,
        )

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Relative Importance", fontsize=10, color=SOLAR_DARK)
    ax.set_title("Feature Importance (Perturbation-Based)", fontsize=12,
                 fontweight="bold", color=SOLAR_DARK, pad=12)
    ax.tick_params(colors=SOLAR_DARK, labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, color="#E2E8F0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout(pad=1.2)
    path, fname = _save_path("importance")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    return fname


def prediction_gauge(power: float, confidence: float,
                     max_power: float = 500.0) -> str:
    """
    Semi-circular gauge showing predicted power vs capacity.
    Returns the filename.
    """
    fraction = min(power / max_power, 1.0)

    fig, ax = plt.subplots(figsize=(5, 3), facecolor=BACKGROUND,
                           subplot_kw={"polar": False})
    ax.set_facecolor(BACKGROUND)
    ax.axis("off")

    # Draw gauge arc background
    theta_bg = np.linspace(np.pi, 0, 300)
    r_outer, r_inner = 1.0, 0.6
    ax.fill_between(
        r_outer * np.cos(theta_bg),
        r_inner * np.sin(theta_bg),
        r_outer * np.sin(theta_bg),
        color="#E2E8F0", zorder=1,
    )

    # Filled arc proportional to prediction
    theta_fill = np.linspace(np.pi, np.pi - fraction * np.pi, 300)
    grad_color = SOLAR_ORANGE if fraction > 0.6 else (SOLAR_AMBER if fraction > 0.3 else SOLAR_BLUE)
    ax.fill_between(
        r_outer * np.cos(theta_fill),
        r_inner * np.sin(theta_fill),
        r_outer * np.sin(theta_fill),
        color=grad_color, zorder=2,
    )

    # Needle
    angle = np.pi - fraction * np.pi
    ax.annotate(
        "", xy=(0.72 * np.cos(angle), 0.72 * np.sin(angle)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=SOLAR_DARK, lw=2),
    )
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.1)

    ax.text(0, 0.25, f"{power:.2f} kWh", ha="center", va="center",
            fontsize=14, fontweight="bold", color=SOLAR_DARK)
    ax.text(0, 0.07, f"Confidence: {confidence:.0f}%", ha="center",
            fontsize=9, color="#64748B")
    ax.text(0, -0.1, "Predicted Solar Output", ha="center",
            fontsize=10, color=SOLAR_DARK)

    plt.tight_layout(pad=0)
    path, fname = _save_path("gauge")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    return fname


def batch_timeline(powers: list[float]) -> str:
    """
    Line chart for batch prediction results.
    Returns the filename.
    """
    x = list(range(1, len(powers) + 1))

    fig, ax = plt.subplots(figsize=(9, 4), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    ax.fill_between(x, powers, alpha=0.15, color=SOLAR_ORANGE)
    ax.plot(x, powers, color=SOLAR_ORANGE, linewidth=2.0, zorder=3)
    ax.scatter(x, powers, color=SOLAR_ORANGE, s=18, zorder=4)

    ax.set_xlabel("Sample Index", fontsize=10, color=SOLAR_DARK)
    ax.set_ylabel("Predicted Power (kWh)", fontsize=10, color=SOLAR_DARK)
    ax.set_title("Batch Prediction Timeline", fontsize=12,
                 fontweight="bold", color=SOLAR_DARK, pad=10)
    ax.tick_params(colors=SOLAR_DARK, labelsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.yaxis.grid(True, color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)

    plt.tight_layout(pad=1.2)
    path, fname = _save_path("batch")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    return fname


def input_radar(values_normalised: list[float],
                feature_names: list[str]) -> str:
    """
    Radar (spider) chart of normalised input values.
    Returns the filename.
    """
    N      = len(feature_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    vals   = values_normalised + [values_normalised[0]]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), facecolor=BACKGROUND,
                           subplot_kw={"polar": True})
    ax.set_facecolor(BACKGROUND)
    fig.patch.set_facecolor(BACKGROUND)

    ax.plot(angles, vals, color=SOLAR_ORANGE, linewidth=2)
    ax.fill(angles, vals, alpha=0.25, color=SOLAR_AMBER)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, size=9, color=SOLAR_DARK)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=7, color="#94A3B8")
    ax.grid(color="#CBD5E1", linewidth=0.7)
    ax.spines["polar"].set_visible(False)
    ax.set_title("Input Feature Profile", fontsize=11,
                 fontweight="bold", color=SOLAR_DARK, pad=14)

    plt.tight_layout(pad=1)
    path, fname = _save_path("radar")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    return fname
