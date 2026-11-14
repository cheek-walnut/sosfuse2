#!/usr/bin/env python3
"""Draw Figure 3 of docs/VALIDATION.md, paranal_twilight.png, from paranal_twilight.csv next to this file.

    uv run --no-project --with matplotlib --with numpy python docs/validation/plot_paranal_twilight.py

Panels: (a) zenith luminance at Paranal, the Patat et al. (2006) fit and the engine with and without a night-sky floor;
(b) engine / measured with and without the floor; (c) the floor as a share of the measured luminance. The engine rows
used are those with all three US Standard Atmosphere options (run all_three_options of the CSV).
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figure_style as style  # noqa: E402  (shared figure style, next to this file)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

STATES = ("clear_89", "clear_92")
FIT_RMS_MAG, FLOOR_V, FLOOR_SIGMA_MAG = 0.18, 21.61, 0.20
FLOOR = 10.0 ** ((12.58 - FLOOR_V) / 2.5)
K_LO, K_HI = 10.0 ** (-0.4 * FLOOR_SIGMA_MAG), 10.0 ** (0.4 * FLOOR_SIGMA_MAG)
R_LO, R_HI = 10.0 ** (-0.4 * FIT_RMS_MAG), 10.0 ** (0.4 * FIT_RMS_MAG)
XLIM = (-4.6, -15.4)
XTICKS = np.arange(-5, -16, -2)


def read(name):
    with (HERE / name).open(newline="") as f:
        return list(csv.DictReader(ln for ln in f if not ln.startswith("#")))


def series(rows, run, state, key):
    sel = sorted((r for r in rows if r["run"] == run and r["state"] == state), key=lambda r: -float(r["solar_elevation_deg"]))
    return np.array([float(r["solar_elevation_deg"]) for r in sel]), np.array([float(r[key]) for r in sel])


def patat(h):
    z = 90.0 - np.asarray(h, float)
    return 10.0 ** ((12.58 - (11.84 + 1.518 * (z - 95.0) - 0.057 * (z - 95.0) ** 2)) / 2.5)


def elev_axis(ax, label=True):
    ax.set_xlim(*XLIM)
    ax.set_xticks(XTICKS)
    if label:
        ax.set_xlabel("solar elevation (°)")


def mk(state, filled):
    c = style.STATE_COLOUR[state]
    return dict(marker=style.STATE_MARKER[state], ms=3.6, mec=c, mew=0.9, mfc=c if filled else "white")


def fig_twilight(rows):
    fig, axs = plt.subplots(1, 3, figsize=(style.TWO_COL, 2.45), gridspec_kw={"width_ratios": [1.2, 1.2, 0.75], "wspace": 0.42})
    ax, bx, cx = axs
    run = "all_three_options"
    hh = np.linspace(-5, -15, 201)
    fit = patat(hh)

    # (a) luminance
    ax.fill_between(hh, fit * R_LO, fit * R_HI, color=style.BAND, lw=0)
    ax.plot(hh, fit, color=style.INK, lw=1.3)
    ax.axhspan(FLOOR * K_LO, FLOOR * K_HI, color=style.BAND, lw=0, alpha=0.7)
    ax.axhline(FLOOR, color=style.FLOOR, lw=0.9, ls=(0, (1.2, 1.6)))
    for s in STATES:
        h, m = series(rows, run, s, "model_cd_m2")
        _, mf = series(rows, run, s, "model_plus_floor_cd_m2")
        ax.plot(h, mf, color=style.STATE_COLOUR[s], lw=1.1, **mk(s, True), zorder=3)
        ax.plot(h, m, color=style.STATE_COLOUR[s], lw=0.7, ls=(0, (2.5, 1.5)), **mk(s, False), zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(2e-5, 4)
    ax.set_ylabel(r"zenith luminance (cd m$^{-2}$)")
    ax.text(-5.0, FLOOR * 0.62, "night-sky floor V = 21.61", ha="left", va="top", fontsize=7, color=style.MUTED)
    ax.text(-6.4, patat(-6.4) * 2.3, "Patat et al. (2006)", ha="left", va="bottom", fontsize=7, color=style.INK)
    elev_axis(ax)
    style.panel_label(ax, "a", x=-0.26)

    # (b) ratio
    bx.axhspan(R_LO, R_HI, color=style.BAND, lw=0)
    bx.axhline(1.0, color=style.INK, lw=0.6)
    for s in STATES:
        h, meas = series(rows, run, s, "measured_cd_m2")
        _, m = series(rows, run, s, "model_cd_m2")
        c = style.STATE_COLOUR[s]
        bx.fill_between(h, (m + K_LO * FLOOR) / meas, (m + K_HI * FLOOR) / meas, color=c, alpha=0.15, lw=0)
        bx.plot(h, (m + FLOOR) / meas, color=c, lw=1.1, **mk(s, True), zorder=3)
        bx.plot(h, m / meas, color=c, lw=0.7, ls=(0, (2.5, 1.5)), **mk(s, False), zorder=4)
    bx.set_ylim(0, 1.25)
    bx.set_ylabel("engine / measured")
    bx.text(-5.0, R_HI - 0.02, f"fit RMS ±{FIT_RMS_MAG:.2f} mag", ha="left", va="top", fontsize=7, color=style.MUTED)
    elev_axis(bx)
    style.panel_label(bx, "b", x=-0.22)

    # (c) floor share
    h, meas = series(rows, run, "clear_89", "measured_cd_m2")
    share = FLOOR / meas * 100
    cx.fill_between(h, FLOOR * K_LO / meas * 100, FLOOR * K_HI / meas * 100, color=style.BAND, lw=0)
    cx.plot(h, share, color=style.INK, lw=1.1, marker="o", ms=2.8, mfc=style.INK, mec=style.INK)
    cx.set_ylim(0, 100)
    cx.set_ylabel("floor / measured (%)")
    elev_axis(cx)
    cx.set_xticks([-5, -10, -15])
    style.panel_label(cx, "c", x=-0.36)

    handles = [
        Line2D([], [], color=style.STATE_COLOUR["clear_89"], lw=1.1, **mk("clear_89", True), label="clear_89"),
        Line2D([], [], color=style.STATE_COLOUR["clear_92"], lw=1.1, **mk("clear_92", True), label="clear_92"),
        Line2D([], [], color=style.MUTED, lw=1.1, marker="o", ms=3.6, mfc=style.MUTED, mec=style.MUTED, label="engine + floor"),
        Line2D([], [], color=style.MUTED, lw=0.7, ls=(0, (2.5, 1.5)), marker="o", ms=3.6, mfc="white", mec=style.MUTED,
               label="engine, no floor"),
        Line2D([], [], color=style.INK, lw=1.3, label="measured (fit)"),
        Patch(color=style.BAND, lw=0, label="fit RMS / floor ±0.2 mag"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=6, handlelength=2.2, columnspacing=1.3)
    return fig


def main():
    style.apply()
    rows = read("paranal_twilight.csv")
    out = style.save(fig_twilight(rows), HERE / "paranal_twilight.png")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
