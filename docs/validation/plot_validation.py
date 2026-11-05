#!/usr/bin/env python3
"""Draw the figures of docs/VALIDATION.md from docs/validation/mystic_compare.csv.

    uv run --no-project --with matplotlib --with numpy python docs/validation/plot_validation.py

Reads the CSV next to this file and writes mystic_vs_engine.png into the same folder. The output carries no
date or software version, so repeated runs with one matplotlib version give identical bytes.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
CSV = HERE / "mystic_compare.csv"
OUT = HERE / "mystic_vs_engine.png"

# Okabe-Ito colours (distinguishable under the common colour-vision deficiencies); markers differ as well.
STYLE = {93.0: ("#0072B2", "o", -3.0), 96.0: ("#D55E00", "s", 0.0), 99.0: ("#009E73", "^", 3.0)}


def load(path):
    """Rows of the CSV as a dict of float arrays; lines starting with '#' are comments."""
    with path.open(newline="") as f:
        lines = [ln for ln in f if ln.strip() and not ln.startswith("#")]
    reader = csv.reader(lines)
    names = next(reader)
    rows = [[float(v) for v in r] for r in reader]
    table = np.array(rows)
    return {n: table[:, i] for i, n in enumerate(names)}


def main():
    d = load(CSV)
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(7.2, 6.4), gridspec_kw={"height_ratios": [3, 2], "hspace": 0.07})

    for sza, (colour, marker, shift) in STYLE.items():
        m = d["sza_deg"] == sza
        wl = d["wavelength_nm"][m]
        ax1.plot(wl, d["engine_I"][m], "-", color=colour, lw=1.4)
        ax1.plot(wl, d["mystic_I"][m], linestyle="none", marker=marker, ms=4.5, mfc="white", mec=colour, mew=1.1)
        # Wavelength offsets keep the three sets of error bars from lying on top of each other.
        ax2.errorbar(wl + shift, d["ratio"][m], yerr=d["ratio_se"][m], linestyle="none", marker=marker, ms=3.8,
                     color=colour, elinewidth=0.9, capsize=0)

    ax1.set_yscale("log")
    ax1.set_ylim(6e-7, 4e-3)
    ax1.set_ylabel(r"radiance $L/E_0$ (sr$^{-1}$)")
    handles = [Line2D([], [], color=c, marker=mk, mfc="white", mec=c, mew=1.1, ms=4.5, lw=1.4, label=f"SZA {int(s)}°")
               for s, (c, mk, _) in STYLE.items()]
    handles += [Line2D([], [], color="0.3", lw=1.4, label="engine"),
                Line2D([], [], color="0.3", linestyle="none", marker="o", mfc="white", mec="0.3", mew=1.1, ms=4.5,
                       label="MYSTIC")]
    ax1.legend(handles=handles, loc="center right", bbox_to_anchor=(1.0, 0.34), ncol=2, frameon=False, handlelength=2.2,
               columnspacing=1.6)

    ax2.axhline(1.0, color="0.15", lw=0.8)
    ax2.set_ylim(0.90, 1.10)
    ax2.set_ylabel("MYSTIC / engine")
    ax2.set_xlabel("wavelength (nm)")
    ax2.set_xlim(350, 840)
    ax1.tick_params(labelbottom=False)

    fig.align_ylabels([ax1, ax2])
    fig.savefig(OUT, dpi=150, metadata={"Software": None}, facecolor="white", transparent=False,
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {OUT.name}: {OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
