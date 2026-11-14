#!/usr/bin/env python3
"""Build a comparison CSV from a complete, verified set of records."""

import argparse
import csv
import hashlib
import math
from pathlib import Path

from common import DEFAULT_CONFIG, add_paths, read

ARM = "A0"
REP_PIXEL = (20.25, 0.5)
STATE_ORDER = ["clear_37", "clear_43", "clear_62", "clear_66", "clear_68", "clear_80", "clear_86", "clear_87"]
PIXEL_ORDER = [(1.35, 0.5), (1.35, 89.5), (10.35, 179.5), (20.25, 0.5), (30.15, 90.5), (60.75, 179.5)]
SZAS = (93, 96)
FIELDS = [
    "quantity",
    "state",
    "sza_deg",
    "view_elevation_deg",
    "relative_azimuth_deg",
    "wavelength_nm",
    "engine_I",
    "mystic_I",
    "mystic_I_se",
    "ratio",
    "ratio_se",
    "ratio_ci_low",
    "ratio_ci_high",
    "n_seeds",
]


HEADER_TEMPLATE = (
    '# Engine and MYSTIC on the same optical input: eight bundled states, six sky pixels, solar zeni'
    'th angles 93 and 96 degrees.\n'
    '# Data of Figure 2 of docs/VALIDATION.md (mystic_states.png, plot_mystic_states.py). It is the r'
    'ecord of the runs described\n'
    '# under Origin. The scripts in docs/validation/mystic rebuild it from the run records and can repe'
    'at the runs. Do not edit by hand.\n'
    '#\n'
    '# Rows (quantity)\n'
    '#   spectrum    L/E0 at the 30 wavelengths, pixel 20.25 deg / 0.5 deg only, 8 states x 2 SZA = '
    '480 rows; state "mean8" rows\n'
    '#               (60 more) hold the mean over the 8 states of the ratio at each wavelength\n'
    '#   luminance   Y_MYSTIC / Y_engine of one case, all 6 pixels x 8 states x 2 SZA = 96 rows; "me'
    'an8" rows (12 more) hold the\n'
    '#               mean over the 8 states\n'
    '#\n'
    '# Columns\n'
    '#   quantity              spectrum or luminance (see above)\n'
    '#   state                 bundled state of states_v2.json (parameters below), or mean8 = mean o'
    'ver the 8 states\n'
    '#   sza_deg               solar zenith angle at the observer, degrees (solar elevation = 90 - s'
    'za_deg: -3 and -6 degrees)\n'
    '#   view_elevation_deg    viewing elevation above the horizon, degrees\n'
    '#   relative_azimuth_deg  viewing azimuth relative to the Sun, degrees (0.5 = towards the Sun; '
    'table columns 0..179 = 0.5..179.5)\n'
    '#   wavelength_nm         wavelength, nm; centres of 30 equal bins over 360-830 nm, evaluated a'
    's single wavelengths ("point"\n'
    '#                         mode); empty for luminance rows\n'
    '#   engine_I              engine radiance divided by the solar irradiance E0 of the shared inpu'
    't at wavelength_nm, sr^-1\n'
    '#                         (E0 = solar irradiance on a plane normal to the beam); spectrum rows '
    'of single\n'
    '#                         states only\n'
    '#   mystic_I              MYSTIC L/E0, sr^-1: mean over the 16 seeds (the MYSTIC radiance of th'
    'e runs is transmittance x E0)\n'
    '#   mystic_I_se           standard error of mystic_I, sr^-1: sample standard deviation of the 1'
    '6 seed values / sqrt(16)\n'
    '#   ratio                 MYSTIC / engine. spectrum: mean over the seeds of (seed value / engin'
    'e_I); luminance: mean over the\n'
    '#                         seeds of Y_seed / Y_engine; mean8 rows: arithmetic mean of the 8 stat'
    'e values\n'
    '#   ratio_se              standard error of ratio: seed standard deviation / sqrt(16) (single s'
    'tates; the engine value is treated\n'
    '#                         as exact, so input, grid and model error are not in it); mean8 rows: '
    'sqrt(sum of squared state errors)\n'
    '#                         / 8, treating the states as independent Monte Carlo runs (the random '
    'seed of a run is its seed\n'
    '#                         label 11..26 plus 10,000 times the index of its state, pixel, SZA and'
    ' wavelength, so no two runs\n'
    '#                         share a seed)\n'
    '#   ratio_ci_low/high     95 % Student t interval of ratio, ratio -/+ 2.131449546 * ratio_se (1'
    '5 degrees of freedom); single\n'
    '#                         states only, empty for mean8\n'
    '#   n_seeds               independent MYSTIC runs (random seeds) combined in the value\n'
    '#\n'
    '# Definition of the ratio: MYSTIC divided by engine, so a value above 1 means MYSTIC is brighte'
    'r. Y is the photopic luminance of\n'
    '# the 30 radiances, Y = sum_i L_i w_i with w_i = 683 * integral of the CIE 1931 2-degree y-bar '
    'x E_sun over bin i / E_sun(lambda_i).\n'
    '# The ratio of Y is the figure\'s "Y ratio". There is no accuracy threshold; the intervals hold '
    'the Monte\n'
    '# Carlo sampling error of MYSTIC only.\n'
    '#\n'
    '# Origin\n'
    '#   Engine: sosfuse2 {engine_version}, path_n 12 source nodes per path segment, Legendre order '
    'L 16 in multiple scattering\n'
    '#   (the package defaults), plus SOSFUSE2_GROUND_SH_L=24 and SOSFUSE2_DETERMINISTIC=1, device profi'
    'le rig (its stage-1\n'
    '#   kernel options FMAD and FASTMATH), nr 44, 100 x 180\n'
    '#   directions, 25 solar elevations +6 to -18 degrees, point mode with ozone cross section at 2'
    '33 K, Anderson tolerance 0.001,\n'
    '#   density_profile usstd76, ozone_profile usstd76, rayleigh_column usstd, stratospheric sulfat'
    'e at 21 km, NO2 and O4 on. The\n'
    '#   table was read at the six pixels and the two SZA.\n'
    '#   MYSTIC: libRadtran 2.0.6, rte_solver mystic, 1D spherical geometry (earth_radius 6371 km), '
    'backward tracing (mc_backward,\n'
    '#   mc_escape on, mc_vroom on), polarised (mc_polarisation), mc_std, observer 0.1 km above the '
    'ground, Lambert albedo per state,\n'
    '#   no refraction, point Sun, layers of 0.1 km from the ground to 3 km and of 0.5 km above to 8'
    '0 km. The tropospheric aerosol\n'
    '#   phase function is Henyey-Greenstein expanded to 256 Legendre terms, the stratospheric one th'
    'e 33 Legendre moments of the\n'
    "#   engine's sulfate table. The optical input is exported from the engine (same Rayleigh, ozone"
    ', NO2, O4 and aerosol optical\n'
    '#   depths per layer); Rayleigh depolarisation 0.03, aerosol scattering fully depolarising.\n'
    '#   16 seeds (11..26) x 250,000 photons per wavelength; 46,080 runs of 8 x 6 x 2 x 30 x 16, all'
    ' successful, no seed discarded.\n'
    '#\n'
    '# States (id, aod550, Angstrom exponent, single-scattering albedo, aerosol scale height km, g, '
    'ozone DU, albedo, stratospheric\n'
    '#   tau550); the stratospheric sulfate layer is centred at 21 km, sigma 3 km\n'
    '#   state      aod550  angstrom  ssa    H_km   g      ozone_DU  albedo  strat_tau550\n'
    '{state_lines}\n'
    '#\n'
    '# Provenance (sha256) of the six research records the rows were taken from: the comparison tabl'
    'e, the engine table, the\n'
    '# full MYSTIC table, the shared optical input and two manifests, which also record the first fo'
    'ur hashes\n'
    '#   comparison            {compare_long_csv}\n'
    '#   engine                {engine_csv}\n'
    '#   mystic_full           {full_csv}\n'
    '#   optical_input         {optics_json}\n'
    '#   manifest_comparison   {manifest_compare_json}\n'
    '#   manifest_run          {manifest_json}\n'
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fnum(x, digits=10):
    return "" if x is None else f"{x:.{digits}g}"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    add_paths(ap)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = a.work.resolve() / "out"
    config = read(a.config)
    if config != read(DEFAULT_CONFIG):
        raise ValueError("E_PUBLIC_CONFIG")
    names = ["compare_long.csv", "engine.csv", "full.csv", "optics.json", "manifest_compare.json", "manifest.json"]
    sha = {n: sha256(d / n) for n in names}
    mc = read(d / "manifest_compare.json")
    mm = read(d / "manifest.json")
    if mm.get("complete") is not True or mm.get("counts", {}).get("success") != 46080:
        raise ValueError("E_MC_INCOMPLETE")
    if mm["config_sha256"] != sha256(a.config):
        raise ValueError("E_CONFIG_SHA")
    recorded = {
        "compare_long.csv": mc["outputs_sha256"]["compare_long.csv"],
        "engine.csv": mc["outputs_sha256"]["engine.csv"],
        "full.csv": mc["mystic_csv_sha256"],
        "optics.json": mc["optics_sha256"],
    }
    if mm["results_sha256"] != mc["mystic_csv_sha256"] or mc["mc_manifest_sha256"] != sha["manifest.json"]:
        raise ValueError("E_RESULT_SHA")
    for n, h in recorded.items():
        if sha[n] != h:
            raise ValueError("E_RESULT_SHA " + n)
    for n, h in mc["outputs_sha256"].items():
        if sha256(d / n) != h:
            raise ValueError("E_RESULT_SHA " + n)
    optics = read(d / "optics.json")
    states = optics["states"]
    engine_pixels = read(d / "engine_pixels.A0.json")
    engine_cfg = engine_pixels["contract"]
    engine_version = engine_cfg["engine_version"]
    if engine_cfg["engine_config_id"] != ARM or engine_cfg["optics_sha256"] != sha["optics.json"]:
        raise ValueError("E_ENGINE_INPUT_SHA")
    if mc["engine_pixels_sha256"][ARM] != sha256(d / "engine_pixels.A0.json"):
        raise ValueError("E_ENGINE_INPUT_SHA")
    expected_cases = {
        f'{sid}_i{p["i"]}_j{p["j"]}_sza{sza:g}'
        for sid in config['state_ids']
        for p in config['pixels']
        for sza in config['sza_deg']
    }
    if set(engine_pixels['cases']) != expected_cases:
        raise ValueError("E_ENGINE_CASE_SET")
    full_rows = read_csv(d / "full.csv")
    seen = {(r['case_id'], round(float(r['wavelength_nm']), 4), int(r['seed'])) for r in full_rows}
    wavelengths = {round(x, 4) for x in config['wavelength_nm']}
    expected = {(case, wave, seed) for case in expected_cases for wave in wavelengths for seed in config['seed_ids']}
    if len(full_rows) != len(expected) or seen != expected or any(r['status'] != 'ok' for r in full_rows):
        raise ValueError("E_MC_COVERAGE")
    eng = {}
    case_info = {}
    for r in read_csv(d / "engine.csv"):
        if r["engine_config_id"] != ARM or r["status"] != "ok":
            continue
        cid = r["case_id"]
        elev, azim = round(float(r["view_elev_deg"]), 2), float(r["view_azim_deg"])
        case_info[cid] = (r["state_id"], elev, azim, int(float(r["sza_deg"])))
        eng[(cid, round(float(r["wavelength_nm"]), 4))] = float(r["radiance"])
    assert len(case_info) == 96 and len(eng) == 96 * 30
    assert {c[1:3] for c in case_info.values()} == set(PIXEL_ORDER)
    assert {c[0] for c in case_info.values()} == set(STATE_ORDER)

    seeds = {}
    for r in full_rows:
        assert r["status"] == "ok"
        seeds.setdefault((r["case_id"], round(float(r["wavelength_nm"]), 4)), []).append(float(r["radiance"]))
    assert len(seeds) == 96 * 30 and all(len(v) == 16 for v in seeds.values())

    spec, lum = {}, {}
    for r in read_csv(d / "compare_long.csv"):
        if r["engine_config_id"] != ARM:
            continue
        if r["status"] != "OK" or r["n"] != "16" or r["discarded_seeds"] != "0":
            raise ValueError("E_INCOMPLETE_ESTIMATES")
        row = {k: float(r[k]) for k in ("mean", "se", "ci_low", "ci_high")}
        if r["metric"] == "spectral_ratio":
            spec[(r["case_id"], round(float(r["wavelength_nm"]), 4))] = row
        elif r["metric"] == "Y_ratio":
            lum[r["case_id"]] = row
    assert len(spec) == 96 * 30 and len(lum) == 96

    rows = []

    def add(**kw):
        rows.append({k: kw.get(k, "") for k in FIELDS})

    cases = {(c[0], c[1], c[2], c[3]): cid for cid, c in case_info.items()}

    elev, azim = REP_PIXEL
    max_dev = 0.0
    for sza in SZAS:
        for sid in STATE_ORDER:
            cid = cases[(sid, elev, azim, sza)]
            solar = states[sid]["solar"]
            wls = sorted(w for (c, w) in eng if c == cid)
            assert len(wls) == 30
            for wi, w in enumerate(wls):
                e_rad, ss = eng[(cid, w)], seeds[(cid, w)]
                mean_l = math.fsum(ss) / 16
                sd = math.sqrt(math.fsum((x - mean_l) ** 2 for x in ss) / 15)
                est = spec[(cid, w)]
                assert abs(est["mean"] - mean_l / e_rad) < 1e-9 * est["mean"]
                assert abs(est["se"] - sd / 4 / e_rad) < 1e-8 * est["se"]
                max_dev = max(max_dev, abs(est["mean"] - mean_l / e_rad) / est["mean"])
                add(
                    quantity="spectrum",
                    state=sid,
                    sza_deg=sza,
                    view_elevation_deg=elev,
                    relative_azimuth_deg=azim,
                    wavelength_nm=f"{w:.4f}",
                    engine_I=fnum(e_rad / solar[wi]),
                    mystic_I=fnum(mean_l / solar[wi]),
                    mystic_I_se=fnum(sd / 4 / solar[wi]),
                    ratio=fnum(est["mean"]),
                    ratio_se=fnum(est["se"]),
                    ratio_ci_low=fnum(est["ci_low"]),
                    ratio_ci_high=fnum(est["ci_high"]),
                    n_seeds=16,
                )
    print(
        "spectral ratios recomputed from the 16 seeds agree with compare_long.csv "
        f"(max relative deviation {max_dev:.1e})"
    )

    for sza in SZAS:
        sel = [r for r in rows if r["quantity"] == "spectrum" and int(r["sza_deg"]) == sza]
        for w in sorted({r["wavelength_nm"] for r in sel}):
            part = [r for r in sel if r["wavelength_nm"] == w]
            assert len(part) == 8
            m = math.fsum(float(r["ratio"]) for r in part) / 8
            se = math.sqrt(math.fsum(float(r["ratio_se"]) ** 2 for r in part)) / 8
            add(
                quantity="spectrum",
                state="mean8",
                sza_deg=sza,
                view_elevation_deg=elev,
                relative_azimuth_deg=azim,
                wavelength_nm=w,
                ratio=fnum(m),
                ratio_se=fnum(se),
                n_seeds=16,
            )

    for sza in SZAS:
        for pe, pa in PIXEL_ORDER:
            vals = []
            for sid in STATE_ORDER:
                est = lum[cases[(sid, pe, pa, sza)]]
                vals.append(est)
                add(
                    quantity="luminance",
                    state=sid,
                    sza_deg=sza,
                    view_elevation_deg=pe,
                    relative_azimuth_deg=pa,
                    ratio=fnum(est["mean"]),
                    ratio_se=fnum(est["se"]),
                    ratio_ci_low=fnum(est["ci_low"]),
                    ratio_ci_high=fnum(est["ci_high"]),
                    n_seeds=16,
                )
            add(
                quantity="luminance",
                state="mean8",
                sza_deg=sza,
                view_elevation_deg=pe,
                relative_azimuth_deg=pa,
                ratio=fnum(math.fsum(v["mean"] for v in vals) / 8),
                ratio_se=fnum(math.sqrt(math.fsum(v["se"] ** 2 for v in vals)) / 8),
                n_seeds=16,
            )

    def sp(sid):
        return states[sid]["state"]["parameters"]

    state_lines = "\n".join(
        f"#   {sid}   {sp(sid)['aod550']:.4f}   {sp(sid)['angstrom']:.3f}   {sp(sid)['ssa']:.3f}   "
        f"{sp(sid)['scale_height_km']:.3f}   {sp(sid)['g']:.3f}   {sp(sid)['ozone_du']:.1f}   "
        f"{sp(sid)['albedo']:.3f}   "
        f"{sp(sid)['stratosphere_tau550']:.4f}"
        for sid in STATE_ORDER
    )
    header = HEADER_TEMPLATE.format(
        engine_version=engine_version,
        state_lines=state_lines,
        **{name.replace('.', '_'): value for name, value in sha.items()},
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as f:
        f.write(header)
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV {a.out.name}: {len(rows)} rows, sha256 {sha256(a.out)}")


if __name__ == "__main__":
    main()
