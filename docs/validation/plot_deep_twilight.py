#!/usr/bin/env python3
"""Draw the deep-twilight figures of docs/VALIDATION.md and docs/MODEL.md from the CSVs next to this file.

    uv run --no-project --with matplotlib --with numpy python docs/validation/plot_deep_twilight.py

  mystic_noise.csv          -> mystic_noise.png            (VALIDATION Figure 2)
  second_order_direct.csv   -> second_order.png            (VALIDATION Figure 4)
                            -> second_order_grids.png      (VALIDATION Figure 5)
  strato_layer_colour.csv   -> strato_layer_colour.png     (VALIDATION Figure 7)
  scattering_orders.csv     -> scattering_orders.png       (MODEL Figure 1)
                            -> scattering_orders_map.png   (MODEL Figure 2)
  light_height.csv          -> light_height.png            (MODEL Figure 3)

The style is that of figure_style.py; the PNGs carry no date or software version.
"""
import csv
from pathlib import Path

import numpy as np

import figure_style as style  # noqa: E402  (shared figure style, next to this file)

import matplotlib.pyplot as plt  # noqa: E402  (after figure_style selects the Agg backend)

HERE = Path(__file__).resolve().parent
R_EARTH = 6371.0


def read(name):
    with (HERE / name).open(newline="") as f:
        return list(csv.DictReader(ln for ln in f if ln.strip() and not ln.startswith("#")))


def num(rows, key):
    return np.array([float(r[key]) if r[key] != "" else np.nan for r in rows])


def mystic_noise():
    rows = read("mystic_noise.csv")
    szas = [96, 99, 102, 105]
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(style.ONE_COL, 3.4),
                                   gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.1})
    sides = (("sun side, 20.25° elevation", style.VERMILLION, "o", -0.35, lambda a: a < 90),
             ("opposite side, 10.35° elevation", style.BLUE, "s", 0.35, lambda a: a > 90))
    for label, colour, marker, off, pick in sides:
        med, none = [], []
        for s in szas:
            sel = [r for r in rows if int(r["sza_deg"]) == s and pick(float(r["relative_azimuth_deg"]))]
            ok = [r for r in sel if r["status"] == "ok" and float(r["radiance"]) > 0]
            rse = np.array([100 * float(r["se"]) / float(r["radiance"]) for r in ok])
            jitter = np.linspace(-0.22, 0.22, rse.size) if rse.size > 1 else np.zeros(rse.size)
            ax1.plot(s + off + jitter, rse, linestyle="none", marker=".", ms=1.6, color=colour, alpha=0.35)
            med.append(np.median(rse) if rse.size else np.nan)
            none.append(100 * (1 - len(ok) / len(sel)))
        ax1.plot(np.array(szas) + off, med, "-", marker=marker, color=colour, ms=3.5, mfc="white", label=label)
        ax2.bar(np.array(szas) + off, none, width=0.6, color=colour)
    ax1.axhline(100, color=style.MUTED, lw=0.7, ls="--")
    ax1.text(95.2, 112, "100 %: one contributing photon path", fontsize=6.5, color=style.MUTED, va="bottom")
    ax1.set_yscale("log")
    ax1.set_ylim(3, 300)
    ax1.set_ylabel("relative standard error\nof one run (%)")
    ax1.legend(loc="lower right", fontsize=6.5)
    ax2.set_ylabel("runs without\nsignal (%)")
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("solar zenith angle (°)")
    ax2.set_xticks(szas)
    style.panel_label(ax1, "a", x=-0.24)
    style.panel_label(ax2, "b", x=-0.24, y=1.0)
    return style.save(fig, HERE / "mystic_noise.png")


DIRECTIONS = [((2, 0), "sun side, 2°"), ((10, 0), "sun side, 10°"), ((30, 0), "sun side, 30°"), ((89, 0), "sun side, 89°"),
              ((10, 180), "opposite, 10°"), ((30, 180), "opposite, 30°"), ((10, 90), "90° azimuth, 10°")]
DIR_COLOUR = [style.VERMILLION, style.ORANGE, style.YELLOW, style.INK, style.BLUE, style.SKY, style.GREEN]
DIR_MARKER = ["o", "s", "^", "D", "v", "<", ">"]
STATES = {"R": "R: molecules only", "T": "T: + tropospheric aerosol", "S": "S: + stratospheric layer"}


def second_order():
    rows = [r for r in read("second_order_direct.csv") if r["grid"] == "default"]
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(style.TWO_COL, 2.6), layout="constrained")
    for ax, (state, title) in zip(axes, STATES.items()):
        sel = [r for r in rows if r["state"] == state]
        step = {}
        for r in sel:
            s = 90 - int(r["solar_elevation_deg"])
            last = abs(float(r["reference_I2"]) / float(r["reference_I2_previous"]) - 1) * 100
            step[s] = max(step.get(s, 0.0), last)
        xs = sorted(step)
        ax.fill_between(xs, [-step[x] for x in xs], [step[x] for x in xs], color=style.BAND, lw=0, zorder=0)
        for (key, label), colour, marker in zip(DIRECTIONS, DIR_COLOUR, DIR_MARKER):
            pts = sorted((90 - int(r["solar_elevation_deg"]), 100 * (float(r["engine_I2"]) / float(r["reference_I2"]) - 1))
                         for r in sel if (int(r["view_elevation_deg"]), int(r["relative_azimuth_deg"])) == key)
            ax.plot(*zip(*pts), "-", marker=marker, ms=3, color=colour, mfc="white", lw=1.0, label=label)
        ax.axhline(0, color=style.INK, lw=0.6)
        ax.set_title(title)
        ax.set_xticks([96, 99, 102, 105, 108])
        ax.set_xlabel("solar zenith angle (°)")
    axes[0].set_ylabel("second order, engine / reference − 1 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=7, fontsize=6.5, handlelength=1.6, columnspacing=1.0)
    for ax, letter in zip(axes, "abc"):
        style.panel_label(ax, letter, x=-0.08)
    return style.save(fig, HERE / "second_order.png")


GRIDS = [("default", "default grid", style.INK, "o", "-"),
         ("h_coarse", "heights 44 → 39", style.VERMILLION, "v", "-"),
         ("h_fine", "heights 44 → 87", style.BLUE, "^", "-"),
         ("chi025", "grid in SZA 0.5° → 0.25°", style.GREEN, "s", "-"),
         ("fine_both", "87 heights and 0.25° step", style.PURPLE, "D", "-"),
         ("fine_all", "both, and 24 source nodes", style.SKY, "D", "--"),
         ("path24", "source nodes per piece 12 → 24", style.ORANGE, "<", ":"),
         ("nmu24", "nodes per polar interval 12 → 24", style.MUTED, ">", ":"),
         ("nphi32", "azimuth intervals 16 → 32", style.YELLOW, "x", ":")]


def second_order_grids():
    rows = [r for r in read("second_order_direct.csv") if r["state"] == "R"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(style.TWO_COL, 2.7), layout="constrained")
    for grid, label, colour, marker, ls in GRIDS:
        sel = [r for r in rows if r["grid"] == grid]
        szas = sorted({90 - int(r["solar_elevation_deg"]) for r in sel})
        diff = {s: [100 * (float(r["engine_I2"]) / float(r["reference_I2"]) - 1) for r in sel
                    if 90 - int(r["solar_elevation_deg"]) == s] for s in szas}
        kw = dict(color=colour, marker=marker, ms=3, mfc="white", ls=ls, lw=1.5 if grid == "default" else 1.0, label=label)
        ax1.plot(szas, [np.mean(diff[s]) for s in szas], **kw)
        ax2.plot(szas, [np.min(diff[s]) for s in szas], **kw)
    for ax, title in ((ax1, "mean over the seven directions"), (ax2, "most negative direction")):
        ax.axhline(0, color=style.INK, lw=0.6)
        ax.set_xticks([96, 99, 102, 105, 108])
        ax.set_xlabel("solar zenith angle (°)")
        ax.set_title(title)
    ax1.set_ylabel("second order, engine / reference − 1 (%)")
    ax2.legend(loc="lower left", fontsize=6.3, title="polar and azimuth changes lie on the default curve", title_fontsize=6)
    style.panel_label(ax1, "a")
    style.panel_label(ax2, "b")
    return style.save(fig, HERE / "second_order_grids.png")


def grazing_window(view_elevation, layer_km=21.0, observer_km=0.1):
    """Solar zenith angles at which the sunlight reaching the layer on this line of sight grazes 10 km and 0 km."""
    e = np.radians(view_elevation)
    r0, r1 = R_EARTH + observer_km, R_EARTH + layer_km
    s = -r0 * np.sin(e) + np.sqrt((r0 * np.sin(e)) ** 2 + r1 ** 2 - r0 ** 2)
    psi = np.degrees(np.arctan2(s * np.cos(e), r0 + s * np.sin(e)))
    return tuple(90 + np.degrees(np.arccos((R_EARTH + t) / r1)) + psi for t in (10.0, 0.0))


def strato_layer_colour():
    rows = read("strato_layer_colour.csv")
    fig, ax = plt.subplots(figsize=(style.ONE_COL, 2.4), layout="constrained")
    for el, colour, marker in ((30, style.GREEN, "^"), (10, style.VERMILLION, "s"), (2, style.BLUE, "o")):
        sel = sorted((r for r in rows if float(r["view_elevation_deg"]) == el), key=lambda r: -int(r["solar_elevation_deg"]))
        sza = [90 - int(r["solar_elevation_deg"]) for r in sel]
        d = [float(r["u_with_layer"]) - float(r["u_without_layer"]) for r in sel]
        lo, hi = grazing_window(el)
        ax.axvspan(lo, hi, color=colour, alpha=0.13, lw=0)
        ax.plot(sza, d, "-", marker=marker, ms=3, mfc="white", color=colour, label=f"view elevation {el}° (band: grazing window)")
    ax.axhline(0, color=style.INK, lw=0.6)
    ax.set_xlim(90, 108)
    ax.set_xticks(np.arange(90, 109, 3))
    ax.set_xlabel("solar zenith angle (°)")
    ax.set_ylabel("u′ with the layer − u′ without")
    ax.legend(loc="upper right", fontsize=6.5)
    return style.save(fig, HERE / "strato_layer_colour.png")


ORDER_DIRS = ["zenith", "sun side 10", "opposite side 10"]
ORDER_TITLES = {"zenith": "zenith", "sun side 10": "sun side, 10° elevation", "opposite side 10": "opposite side, 10° elevation"}


def scattering_orders(wavelength=553.875):
    rows = [r for r in read("scattering_orders.csv") if float(r["wavelength_nm"]) == wavelength]
    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(style.TWO_COL, 4.6), layout="constrained")
    colours = ("#F5D29A", style.ORANGE, "#7A1F00")
    for i, state in enumerate(STATES):
        for j, direction in enumerate(ORDER_DIRS):
            sel = sorted((r for r in rows if r["state"] == state and r["direction"] == direction),
                         key=lambda r: -int(r["solar_elevation_deg"]))
            sza = [90 - int(r["solar_elevation_deg"]) for r in sel]
            ax = axes[i, j]
            ax.stackplot(sza, num(sel, "first"), num(sel, "second"), num(sel, "higher"), colors=colours,
                         labels=("first order", "second order", "third and higher"))
            ax.set_xlim(90, 108)
            ax.set_ylim(0, 1)
            ax.set_xticks(np.arange(90, 109, 3))
            if i == 0:
                ax.set_title(ORDER_TITLES[direction])
            if j == 0:
                ax.set_ylabel(f"{STATES[state]}\nshare of radiance", fontsize=7)
            if i == 2:
                ax.set_xlabel("solar zenith angle (°)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=6.5)
    return style.save(fig, HERE / "scattering_orders.png")


def scattering_orders_map():
    rows = read("scattering_orders.csv")
    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(style.TWO_COL, 4.8), layout="constrained")
    for i, state in enumerate(STATES):
        for j, direction in enumerate(ORDER_DIRS):
            sel = [r for r in rows if r["state"] == state and r["direction"] == direction]
            wl = np.array(sorted({float(r["wavelength_nm"]) for r in sel}))
            sza = np.array(sorted({90 - int(r["solar_elevation_deg"]) for r in sel}))
            grid = np.full((sza.size, wl.size), np.nan)
            for r in sel:
                grid[np.searchsorted(sza, 90 - int(r["solar_elevation_deg"])), np.searchsorted(wl, float(r["wavelength_nm"]))] = (
                    float(r["higher"]))
            ax = axes[i, j]
            im = ax.pcolormesh(wl, sza, grid, vmin=0, vmax=1, cmap="magma", shading="nearest")
            cs = ax.contour(wl, sza, grid, levels=[0.1, 0.25, 0.5, 0.75], colors="white", linewidths=0.6)
            ax.clabel(cs, fmt="%.2f", fontsize=5.5)
            ax.set_yticks(np.arange(90, 109, 6))
            if not ax.yaxis_inverted():
                ax.invert_yaxis()
            if i == 0:
                ax.set_title(ORDER_TITLES[direction])
            if j == 0:
                ax.set_ylabel(f"{STATES[state]}\nsolar zenith angle (°)", fontsize=7)
            if i == 2:
                ax.set_xlabel("wavelength (nm)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="share of third and higher orders")
    return style.save(fig, HERE / "scattering_orders_map.png")


SHORT = {"zenith": "zenith", "sun side 10": "sun side, 10° elev.", "opposite side 10": "opposite, 10° elev."}


def light_height():
    rows = read("light_height.csv")
    pick = [90, 93, 96, 99, 102, 105, 108]
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 4, figsize=(style.TWO_COL, 2.7), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.3]})
    for ax, direction in zip(axes[:3], ORDER_DIRS):
        for n, s in enumerate(pick):
            sel = sorted((r for r in rows if r["direction"] == direction and 90 - int(r["solar_elevation_deg"]) == s),
                         key=lambda r: float(r["altitude_low_km"]))
            if not sel:
                continue
            c = num(sel, "total")
            edges = np.concatenate([[float(sel[0]["altitude_low_km"])], num(sel, "altitude_high_km")])
            ax.plot(np.concatenate([[0], np.cumsum(c) / c.sum()]), edges, color=cmap(n / (len(pick) - 1)), lw=1.0,
                    label=f"{s}°")
        ax.set_ylim(0, 120)
        ax.set_xlim(0, 1)
        ax.set_title(SHORT[direction], fontsize=7)
        ax.set_xlabel("cumulative share")
    axes[0].set_ylabel("altitude of the last scattering (km)")
    axes[0].legend(title="solar zenith angle", fontsize=6, title_fontsize=6, loc="upper left")
    ax = axes[3]
    for direction, colour in zip(ORDER_DIRS, (style.INK, style.VERMILLION, style.BLUE)):
        for key, ls in (("total", "-"), ("first", ":"), ("higher", "--")):
            szas, mean = [], []
            for s in range(90, 109):
                sel = [r for r in rows if r["direction"] == direction and 90 - int(r["solar_elevation_deg"]) == s]
                if not sel:
                    continue
                w, tot = num(sel, key), num(sel, "total").sum()
                mid = 0.5 * (num(sel, "altitude_low_km") + num(sel, "altitude_high_km"))
                if w.sum() > 1e-3 * tot:
                    szas.append(s)
                    mean.append((w * mid).sum() / w.sum())
            ax.plot(szas, mean, ls, color=colour, lw=1.2 if ls == "-" else 0.8,
                    label=SHORT[direction] if ls == "-" else None)
    sza = np.linspace(90, 108, 100)
    ax.plot(sza, R_EARTH * (1 / np.cos(np.radians(sza - 90)) - 1), color=style.MUTED, lw=0.7, ls="-.",
            label="Earth's shadow above the observer")
    ax.set_ylim(0, 120)
    ax.set_xlim(90, 108)
    ax.set_xticks(np.arange(90, 109, 6))
    ax.set_xlabel("solar zenith angle (°)")
    ax.set_ylabel("mean altitude (km)")
    ax.set_title("mean altitude", fontsize=7)
    ax.legend(loc="upper left", fontsize=5.8)
    for a, letter in zip(axes, "abcd"):
        style.panel_label(a, letter, x=-0.2)
    return style.save(fig, HERE / "light_height.png")


def main():
    style.apply()
    for draw in (mystic_noise, second_order, second_order_grids, strato_layer_colour, scattering_orders,
                 scattering_orders_map, light_height):
        path = draw()
        print(f"wrote {path.name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
