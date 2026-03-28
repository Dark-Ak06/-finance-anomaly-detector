"""
Visualization helpers for anomaly detection results.
All functions return matplotlib Figure objects (or save to file).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from typing import Optional

from .detector import AnomalyResult


# Consistent color palette
COLORS = {
    "normal":  "#6ee7b7",
    "anomaly": "#f87171",
    "accent":  "#818cf8",
    "warn":    "#fbbf24",
    "bg":      "#0d0f14",
    "grid":    "#1e2130",
    "text":    "#9ca3af",
}

CAT_COLORS = {
    "Food": "#6ee7b7", "Shopping": "#818cf8", "Transport": "#fbbf24",
    "Health": "#f87171", "Entertainment": "#34d399",
    "Salary": "#60a5fa", "Other": "#9ca3af",
}


def _dark_style():
    plt.rcParams.update({
        "figure.facecolor":  COLORS["bg"],
        "axes.facecolor":    COLORS["bg"],
        "axes.edgecolor":    COLORS["grid"],
        "axes.labelcolor":   COLORS["text"],
        "xtick.color":       COLORS["text"],
        "ytick.color":       COLORS["text"],
        "grid.color":        COLORS["grid"],
        "text.color":        COLORS["text"],
        "font.family":       "monospace",
    })


def plot_anomaly_scatter(
    results: list[AnomalyResult],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter plot: Amount vs Hour, colored by anomaly status.
    Point size encodes anomaly score.
    """
    _dark_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    normal = [r for r in results if not r.is_anomaly]
    anomalies = [r for r in results if r.is_anomaly]

    ax.scatter(
        [r.hour for r in normal],
        [abs(r.amount) for r in normal],
        c=COLORS["normal"], alpha=0.6, s=40, label="Normal", zorder=2,
    )
    ax.scatter(
        [r.hour for r in anomalies],
        [abs(r.amount) for r in anomalies],
        c=COLORS["anomaly"], alpha=0.9,
        s=[r.anomaly_score * 300 for r in anomalies],
        label="Anomaly", marker="X", zorder=3,
    )

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Amount (₸)")
    ax.set_title("Transaction Anomaly Scatter — Amount vs Hour", color="white", pad=12)
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.2, labelcolor="white")

    _format_yaxis_thousands(ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_score_distribution(
    results: list[AnomalyResult],
    threshold: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Histogram of anomaly scores with threshold line."""
    _dark_style()
    fig, ax = plt.subplots(figsize=(9, 4))

    scores = [r.anomaly_score for r in results]
    normal_scores  = [r.anomaly_score for r in results if not r.is_anomaly]
    anomaly_scores = [r.anomaly_score for r in results if r.is_anomaly]

    bins = np.linspace(0, 1, 30)
    ax.hist(normal_scores,  bins=bins, color=COLORS["normal"],  alpha=0.7, label="Normal")
    ax.hist(anomaly_scores, bins=bins, color=COLORS["anomaly"], alpha=0.8, label="Anomaly")
    ax.axvline(threshold, color=COLORS["warn"], lw=1.5, ls="--", label=f"Threshold {threshold:.3f}")

    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title("Anomaly Score Distribution", color="white", pad=12)
    ax.grid(True, alpha=0.2, axis="y")
    ax.legend(framealpha=0.2, labelcolor="white")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_spending_timeline(
    results: list[AnomalyResult],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Daily spending bar chart with anomaly days highlighted."""
    _dark_style()
    fig, ax = plt.subplots(figsize=(12, 4))

    daily: dict[str, float] = {}
    daily_anom: dict[str, bool] = {}
    for r in results:
        if r.amount < 0:
            daily[r.date] = daily.get(r.date, 0) + abs(r.amount)
            if r.is_anomaly:
                daily_anom[r.date] = True

    dates = sorted(daily.keys())
    amounts = [daily[d] for d in dates]
    colors = [COLORS["anomaly"] if daily_anom.get(d) else COLORS["normal"] for d in dates]

    ax.bar(range(len(dates)), amounts, color=colors, width=0.7)
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([d[5:] for d in dates], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Spending (₸)")
    ax.set_title("Daily Spending (red = anomaly detected)", color="white", pad=12)
    ax.grid(True, alpha=0.2, axis="y")
    _format_yaxis_thousands(ax)

    patches = [
        mpatches.Patch(color=COLORS["normal"],  label="Normal day"),
        mpatches.Patch(color=COLORS["anomaly"], label="Anomaly detected"),
    ]
    ax.legend(handles=patches, framealpha=0.2, labelcolor="white")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_category_breakdown(
    results: list[AnomalyResult],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Spending by category — donut chart."""
    _dark_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    cat_totals: dict[str, float] = {}
    for r in results:
        if r.amount < 0:
            cat_totals[r.category] = cat_totals.get(r.category, 0) + abs(r.amount)

    cats = list(cat_totals.keys())
    vals = [cat_totals[c] for c in cats]
    colors = [CAT_COLORS.get(c, "#9ca3af") for c in cats]

    wedges, texts, autotexts = ax.pie(
        vals, labels=cats, colors=colors,
        autopct="%1.0f%%", startangle=140,
        wedgeprops={"width": 0.55, "edgecolor": COLORS["bg"]},
        textprops={"color": COLORS["text"], "fontsize": 9},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(8)

    ax.set_title("Spending by Category", color="white", pad=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _format_yaxis_thousands(ax):
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"₸{int(x/1000)}k" if x >= 1000 else f"₸{int(x)}")
    )
