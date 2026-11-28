#!/usr/bin/env python3
"""Draw Figure 1 of docs/VALIDATION.md from docs/validation/mystic_compare.csv, mystic_compare_102.csv and
mystic_compare_105.csv.

    uv run --no-project --with matplotlib --with numpy python docs/validation/plot_validation.py

Reads the CSVs next to this file and writes mystic_vs_engine.png into the same folder. The output carries no
date or software version, so repeated runs with one matplotlib version give identical bytes.

One panel per solar zenith angle on a linear axis: both spectra are divided by the engine's 30-wavelength mean, so the
panels compare spectral shape at radiance levels six orders of magnitude apart; the mean level is printed in each panel.
The sixth panel gives the inverse-variance weighted mean of the MYSTIC / engine ratio over the 30 wavelengths.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
CSVS = [HERE / "mystic_compare.csv",       # solar zenith angles 93, 96 and 99
        HERE / "mystic_compare_102.csv",   # 102, from separate MYSTIC runs
        HERE / "mystic_compare_105.csv"]   # 105, from separate MYSTIC runs
OUT = HERE / "mystic_vs_engine.png"

# Okabe-Ito colours (distinguishable under the common colour-vision deficiencies); markers differ as well.
STYLE = {93.0: ("#0072B2", "o"), 96.0: ("#D55E00", "s"), 99.0: ("#009E73", "^"), 102.0: ("#CC79A7", "D"),
         105.0: ("#E69F00", "v")}


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
    parts = [load(p) for p in CSVS]
    keys = ["sza_deg", "wavelength_nm", "engine_I", "mystic_I", "ratio", "ratio_se"]
    d = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig, axs = plt.subplots(2, 3, figsize=(10.0, 6.4), gridspec_kw={"wspace": 0.28, "hspace": 0.42})
    flat = axs.ravel()

    for ax, (sza, (colour, marker)) in zip(flat, STYLE.items()):
        m = d["sza_deg"] == sza
        wl, eng, mys, rse = d["wavelength_nm"][m], d["engine_I"][m], d["mystic_I"][m], d["ratio_se"][m]
        level = eng.mean()
        ax.plot(wl, eng / level, "-", color=colour, lw=1.6)
        # ratio_se * engine_I is the standard error of mystic_I.
        ax.errorbar(wl, mys / level, yerr=rse * eng / level, linestyle="none", marker=marker, ms=3.6, mfc="white",
                    mec=colour, mew=1.0, color=colour, elinewidth=0.8, capsize=0)
        ax.set_title(f"SZA {int(sza)}°  (sun −{int(sza) - 90}°)", fontsize=10, loc="left")
        ax.text(0.03, 0.97, f"mean {level:.1e} sr$^{{-1}}$", transform=ax.transAxes, ha="left", va="top", fontsize=8,
                color="0.3")
        ax.set_xlim(350, 840)
        ax.set_xticks([400, 500, 600, 700, 800])
        ax.set_ylim(0.3, 1.6)
    for ax in axs[:, 0]:
        ax.set_ylabel("radiance / engine mean")
    for ax in list(axs[1, :2]) + [axs[0, 2]]:
        ax.set_xlabel("wavelength (nm)")

    summary = flat[5]
    for sza, (colour, marker) in STYLE.items():
        m = d["sza_deg"] == sza
        w = 1.0 / d["ratio_se"][m] ** 2
        mean, err = np.sum(w * d["ratio"][m]) / np.sum(w), 1.0 / np.sqrt(np.sum(w))
        summary.errorbar([sza], [mean], yerr=[err], marker=marker, ms=6, color=colour, mfc="white", mec=colour, mew=1.2,
                         elinewidth=1.1, capsize=2)
    summary.axhline(1.0, color="0.15", lw=0.8)
    summary.set_xticks(list(STYLE))
    summary.set_xlim(91, 107)
    summary.set_ylim(0.97, 1.03)
    summary.set_title("30 wavelengths, weighted", fontsize=10, loc="left")
    summary.set_xlabel("solar zenith angle (°)")
    summary.set_ylabel("MYSTIC / engine")
    fig.text(0.5, 0.955, "lines: engine   markers: MYSTIC (±1 standard error)", ha="center", fontsize=9.5, color="0.3")

    fig.savefig(OUT, dpi=150, metadata={"Software": None}, facecolor="white", transparent=False,
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {OUT.name}: {OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
