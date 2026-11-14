"""Shared figure style of Figures 2 and 3 of docs/VALIDATION.md (plot_mystic_states.py, plot_paranal_twilight.py).

  - widths: one column 3.4 in, two columns 7.0 in; text at final size 7-9 pt
  - colours: Okabe-Ito (distinguishable under the common colour-vision deficiencies), assigned to entities in a fixed order
    and never cycled; the measured or reference curve is black, uncertainty bands are light grey or a light tint of the
    entity colour; identity never rests on colour alone (markers or line styles differ too)
  - panel labels "(a)", "(b)" bold at the top left, outside the axes
  - output: a 300 dpi PNG with no date or software version, so repeated runs with one matplotlib version give the same
    bytes
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ONE_COL = 3.4   # inches
TWO_COL = 7.0

# Okabe-Ito, fixed order. Entities take colours by role, not by rank.
BLUE, VERMILLION, GREEN, PURPLE, ORANGE, SKY, YELLOW = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442")
INK = "0.10"        # text, measured curves, reference lines
MUTED = "0.40"      # secondary text
BAND = "0.86"       # neutral uncertainty band
FLOOR = "0.45"      # night-sky floor line

# Fixed entity colours across all figures.
STATE_COLOUR = {"clear_89": BLUE, "clear_92": VERMILLION, "clear_86": GREEN, "clear_80": PURPLE}
STATE_MARKER = {"clear_89": "o", "clear_92": "s", "clear_86": "^", "clear_80": "D"}


def apply():
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
                "svg.hashsalt": "sosfuse2",
        "path.simplify": False,
    })


def panel_label(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)


def save(fig, path):
    """Write the PNG (300 dpi) with no date or software metadata."""
    path = Path(path)
    fig.savefig(path, dpi=300, metadata={"Software": None}, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path
