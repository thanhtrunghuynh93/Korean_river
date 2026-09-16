"""Shared matplotlib/seaborn style for the EDA figures.

Palette follows the dataviz reference instance: categorical slots assigned in fixed order
to the four site groups (validated all-pairs with `validate_palette.js`: blue, orange, aqua,
violet), a single-hue blue ramp for magnitude, blue<->red with a grey midpoint for diverging.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"

GROUP_COLORS = {          # fixed order, never cycled
    "main_stem": "#2a78d6",   # blue   (slot 1)
    "tributary": "#eb6834",   # orange (slot 2)
    "reservoir": "#1baf7a",   # aqua   (slot 3) - <3:1 contrast, so always legend + labels
    "treatment": "#4a3aa7",   # violet (slot 7)
}
TARGET_COLORS = {"THMFP": "#2a78d6", "HAAFP": "#eb6834"}
WTP_COLORS = {"Gumi": "#2a78d6", "Goryeong": "#eb6834", "Bansong": "#1baf7a"}

SEQ_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_STEPS)
DIV_CMAP = LinearSegmentedColormap.from_list("div_blue_red", ["#104281", "#2a78d6", "#f0efec", "#e34948", "#8a1f1f"])


def apply() -> None:
    sns.set_theme(style="white", context="notebook", font="DejaVu Sans")
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.linewidth": 1.0,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "text.color": TEXT, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 2.0, "lines.markersize": 6,
        "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def savefig(fig: plt.Figure, name: str) -> Path:
    """Save to figures/<name>.png and return the path (also closes nothing; caller shows it)."""
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    return path
