#!/usr/bin/env python3
"""Draw Figure 2 of docs/VALIDATION.md, mystic_states.png, from mystic_states.csv next to this file.

    uv run --no-project --with matplotlib --with numpy python docs/validation/plot_mystic_states.py

Panels: (a) radiance spectra of one state, engine (lines) and MYSTIC (markers), at the pixel 20.25 deg / 0.5 deg;
(b) MYSTIC / engine spectral ratio at that pixel, 8 states (grey) and their mean (colour) with its standard error;
(c) MYSTIC / engine luminance ratio with 95 % intervals at six pixels, 8 states each. Rows (b) and (c) share the y axis:
top SZA 93, bottom SZA 96. Colour marks the SZA in every panel (Okabe-Ito, fixed), markers and line styles repeat it.
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figure_style as style  # noqa: E402  (shared figure style, next to this file)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

CSV = HERE / "mystic_states.csv"
OUT = HERE / "mystic_states.png"

REP_STATE = "clear_87"       # state of panel (a): the largest ozone column of the eight, so the deepest Chappuis band
STATES = ["clear_37", "clear_43", "clear_62", "clear_66", "clear_68", "clear_80", "clear_86", "clear_87"]
PIXELS = [(1.35, 0.5), (1.35, 89.5), (10.35, 179.5), (20.25, 0.5), (30.15, 90.5), (60.75, 179.5)]
REP_PIXEL_INDEX = 3
# colour by SZA (fixed order, never cycled); marker and line style repeat it
SZA_STYLE = {93: dict(color=style.BLUE, marker="o", ls="-"),
             96: dict(color=style.VERMILLION, marker="s", ls=(0, (4, 1.6)))}
YLIM = {93: (0.985, 1.015), 96: (0.92, 1.08)}
YTICKS = {93: np.array([0.99, 1.00, 1.01]), 96: np.arange(0.92, 1.0801, 0.04)}


def load(path):
    with path.open(newline="") as f:
        lines = [ln for ln in f if ln.strip() and not ln.startswith("#")]
    out = []
    for r in csv.DictReader(lines):
        for k, v in r.items():
            if k not in ("quantity", "state"):
                r[k] = float(v) if v != "" else np.nan
        out.append(r)
    return out


def sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def col(rows, name):
    return np.array([r[name] for r in rows])


def main():
    rows = load(CSV)
    spec = [r for r in rows if r["quantity"] == "spectrum"]
    lum = [r for r in rows if r["quantity"] == "luminance"]

    style.apply()
    fig = plt.figure(figsize=(style.TWO_COL, 6.3))
    gs_a = GridSpec(1, 1, figure=fig, left=0.095, right=0.995, top=0.965, bottom=0.785)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.12, 1], hspace=0.11, wspace=0.07,
                  left=0.095, right=0.995, top=0.655, bottom=0.075)
    ax_a = fig.add_subplot(gs_a[0, 0])
    ax_b = {93: fig.add_subplot(gs[0, 0]), 96: None}
    ax_b[96] = fig.add_subplot(gs[1, 0], sharex=ax_b[93])
    ax_c = {93: fig.add_subplot(gs[0, 1], sharey=ax_b[93]), 96: None}
    ax_c[96] = fig.add_subplot(gs[1, 1], sharey=ax_b[96], sharex=ax_c[93])

    # ---------------------------------------------------------------- (a) spectra of one state
    for sza, st in SZA_STYLE.items():
        d = sel(spec, state=REP_STATE, sza_deg=sza)
        wl, eng, mys = col(d, "wavelength_nm"), col(d, "engine_I"), col(d, "mystic_I")
        ax_a.plot(wl, eng, color=st["color"], ls=st["ls"], lw=1.2, zorder=2)
        ax_a.plot(wl, mys, ls="none", marker=st["marker"], ms=3.6, mfc="white", mec=st["color"], mew=0.9, zorder=3)
    ax_a.set_yscale("log")
    ax_a.set_ylim(6e-6, 5e-3)
    ax_a.set_xlim(355, 835)
    ax_a.set_xlabel("wavelength (nm)", color=style.INK)
    ax_a.set_ylabel(r"radiance $L/E_0$ (sr$^{-1}$)", color=style.INK)
    handles = [Line2D([], [], color=SZA_STYLE[s]["color"], ls=SZA_STYLE[s]["ls"], lw=1.2, marker=SZA_STYLE[s]["marker"],
                      ms=3.6, mfc="white", mec=SZA_STYLE[s]["color"], mew=0.9, label=f"SZA {s}°") for s in SZA_STYLE]
    handles += [Line2D([], [], color=style.INK, lw=1.2, label="engine"),
                Line2D([], [], color=style.INK, ls="none", marker="o", ms=3.6, mfc="white", mec=style.INK, mew=0.9,
                       label="MYSTIC")]
    ax_a.legend(handles=handles, loc="center right", bbox_to_anchor=(1.0, 0.52), ncol=2, handlelength=2.6,
                columnspacing=1.6, labelcolor=style.INK, borderaxespad=0.2)
    style.panel_label(ax_a, "a", x=-0.075)
    ax_a.set_title(f"{REP_STATE}; pixel at 20.25° elevation, 0.5° azimuth from the Sun", loc="left", fontsize=7,
                   color=style.MUTED, pad=3)
    ax_a.text(603, 3.2e-4, "Chappuis band", ha="center", va="top", fontsize=7, color=style.MUTED)

    # ---------------------------------------------------------------- (b) spectral ratio, 8 states + mean
    for sza, st in SZA_STYLE.items():
        ax = ax_b[sza]
        for sid in STATES:
            d = sel(spec, state=sid, sza_deg=sza)
            ax.plot(col(d, "wavelength_nm"), col(d, "ratio"), color="0.68", lw=0.55, zorder=1)
        d = sel(spec, state="mean8", sza_deg=sza)
        wl, m, se = col(d, "wavelength_nm"), col(d, "ratio"), col(d, "ratio_se")
        ax.fill_between(wl, m - se, m + se, color=st["color"], alpha=0.25, lw=0, zorder=2)
        ax.plot(wl, m, color=st["color"], lw=1.1, marker=st["marker"], ms=2.8, zorder=3)
        ax.axhline(1.0, color=style.INK, lw=0.6, zorder=0)
        ax.set_ylim(*YLIM[sza])
        ax.set_yticks(YTICKS[sza])
        ax.set_xlim(355, 835)
        ax.set_ylabel("MYSTIC / engine", color=style.INK)
        ax.text(0.015, 0.95, f"SZA {sza}°", transform=ax.transAxes, va="top", ha="left", fontsize=8, color=style.INK,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=6)
    ax_b[93].tick_params(labelbottom=False)
    ax_b[96].set_xlabel("wavelength (nm)", color=style.INK)
    ax_b[93].legend(handles=[Line2D([], [], color="0.68", lw=0.8, label="single state"),
                             (Patch(facecolor="0.78", edgecolor="none"), Line2D([], [], color=style.INK, lw=1.0))],
                    labels=["single state", "8-state mean ± 1 SE"], loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2,
                    fontsize=7, handlelength=1.8, columnspacing=1.2, labelcolor=style.INK, borderaxespad=0.1,
                    handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})
    ax_b[93].text(0.0, 1.025, "Spectral ratio", transform=ax_b[93].transAxes, ha="left", va="bottom", fontsize=7,
                  color=style.MUTED)
    style.panel_label(ax_b[93], "b", x=-0.15)

    # ---------------------------------------------------------------- (c) luminance ratio, 6 pixels x 8 states
    offs = (np.arange(8) - 3.5) * 0.085
    for sza, st in SZA_STYLE.items():
        ax = ax_c[sza]
        ax.axvspan(REP_PIXEL_INDEX - 0.46, REP_PIXEL_INDEX + 0.46, color=style.BAND, lw=0, zorder=0)
        ax.axhline(1.0, color=style.INK, lw=0.6, zorder=1)
        for g, (pe, pa) in enumerate(PIXELS):
            for k, sid in enumerate(STATES):
                r = sel(lum, state=sid, sza_deg=sza, view_elevation_deg=pe, relative_azimuth_deg=pa)[0]
                x = g + offs[k]
                ax.plot([x, x], [r["ratio_ci_low"], r["ratio_ci_high"]], color=st["color"], lw=0.7, solid_capstyle="butt",
                        zorder=2)
                ax.plot(x, r["ratio"], ls="none", marker=st["marker"], ms=3.0, color=st["color"], mew=0, zorder=3)
            m = sel(lum, state="mean8", sza_deg=sza, view_elevation_deg=pe, relative_azimuth_deg=pa)[0]
            ax.plot([g - 0.40, g + 0.40], [m["ratio"]] * 2, color=style.INK, lw=1.0, solid_capstyle="butt", zorder=4)
        ax.set_xlim(-0.55, len(PIXELS) - 0.45)
        ax.set_ylim(*YLIM[sza])
        ax.tick_params(labelleft=False)
        ax.text(0.015, 0.95, f"SZA {sza}°", transform=ax.transAxes, va="top", ha="left", fontsize=8, color=style.INK,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=6)
        ax.set_xticks(range(len(PIXELS)))
    ax_c[93].tick_params(labelbottom=False)
    ax_c[96].set_xticklabels([f"{pe:g}°\n{pa:g}°" for pe, pa in PIXELS])
    ax_c[96].set_xlabel("pixel: elevation / azimuth from the Sun", color=style.INK)
    ax_c[93].text(0.085, 1.025, "Luminance ratio, 95 % intervals", transform=ax_c[93].transAxes, ha="left", va="bottom",
                  fontsize=7, color=style.MUTED)
    ax_c[93].text(REP_PIXEL_INDEX, YLIM[93][1] - 0.0005, "(a, b)", ha="center", va="top", fontsize=7, color=style.MUTED)
    ax_c[93].legend(handles=[Line2D([], [], color=style.INK, lw=1.0, label="8-state mean")], loc="lower right",
                    bbox_to_anchor=(1.0, 1.0), fontsize=7, handlelength=1.4, labelcolor=style.INK, borderaxespad=0.1)
    style.panel_label(ax_c[93], "c", x=-0.03)

    style.save(fig, OUT)
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
