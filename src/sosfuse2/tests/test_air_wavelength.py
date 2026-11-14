"""D1 air wavelength axis: air_wavelength.py against NIST's published Ciddor values,
the production builder's refusals, and independent recomputations of the relabelled ozone columns and of the TSIS-1 HSRS
solar column (0.1 nm cell means).

NIST reference: Engineering Metrology Toolbox, Refractive Index of Air Calculator, Documentation Table 1 (Ciddor,
CO2 450 µmol/mol), https://emtoolbox.nist.gov/Wavelength/Documentation.asp; dry rows (RH 0) only,
since air_wavelength.ciddor_dry has no water term. NIST uses the CODATA 1998 gas constant and its table has 9 decimals.
"""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from sosfuse2.atmosphere import air_wavelength as AW, spectral as SP
from sosfuse2.foundation import paths
from sosfuse2.checks import testkit  # noqa: E402

# (t °C, p kPa, vacuum λ nm, n Ciddor) — NIST Table 1 rows with RH 0 %
NIST_DRY = [
    (20, 101.325, 633, 1.000271800),
    (20, 60, 633, 1.000160924),
    (20, 120, 633, 1.000321916),
    (50, 100, 633, 1.000243285),
    (5, 100, 633, 1.000282756),
    (-40, 100, 633, 1.000337580),
    (20, 101.325, 1700, 1.000268479),
    (20, 101.325, 300, 1.000286581),
    (-40, 120, 300, 1.000427233),
]
# Research sources: the raw files are local copies (skip when absent), the builder and the source
# listings are tracked records (skip without SOSFUSE2_SKY_SOS_ROOT, fail when it is set and they are missing).
RAW = testkit.sky_sos_optional('results/sigma_tz_20261028/raw/serdyuchenkogorshelev5digits.dat')
NEEDS_RAW = pytest.mark.skipif(
    RAW is None, reason='raw Serdyuchenko file is a gitignored sky-sos local copy (SOSFUSE2_SKY_SOS_ROOT)'
)
HSRS = testkit.sky_sos_optional('results/d1_axis_20261031/raw/tsis1_hsrs_355_845.csv')
NEEDS_HSRS = pytest.mark.skipif(
    HSRS is None, reason='full-resolution TSIS-1 HSRS slice is a gitignored sky-sos local copy (SOSFUSE2_SKY_SOS_ROOT)'
)
FINE = paths.ASSETS / 'spectra_fine.json'


def builder():
    spec = importlib.util.spec_from_file_location(
        'build_spectra_fine', testkit.sky_sos('scripts/build_spectra_fine.py')
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('t, p, lam, n', NIST_DRY)
def test_ciddor_matches_nist(t, p, lam, n):
    assert abs(AW.ciddor_dry(lam, t, p * 1e3) - n) <= 1e-9  # measured |Δ| <= 0.4e-9


def test_standard_air_chain_and_edlen_record():
    lam = np.arange(360.0, 840.1, 1.0)
    np.testing.assert_allclose(
        AW.ciddor_dry(lam, 15.0, 101325.0), AW.standard_n(lam), rtol=0, atol=1e-15
    )  # density ratio 1
    gap = AW.standard_n(lam) - AW.edlen1966_n(lam)
    assert 1e-8 < gap.min() and gap.max() < 2e-8  # record: 1.10-1.44e-8
    air = AW.vacuum_to_air(lam)
    assert (
        np.all(np.diff(air) > 0)
        and np.all(np.diff(lam - air) > 0)
        and 0.101 < (lam - air)[0]
        and (lam - air)[-1] < 0.232
    )
    np.testing.assert_allclose(
        (lam - air)[[40, 440]], [0.11307, 0.21998], atol=1e-5
    )  # measured: 0.113 @400, 0.220 @800


def test_builder_refuses_other_sources(tmp_path):
    B = builder()
    p1nm = testkit.sky_sos('results/d1_axis_20261031/sources/hsrs_p1nm_355_845.csv')
    with pytest.raises(SystemExit, match='E_SPECTRA_SOURCE'):
        B.read_hsrs(p1nm)  # another HSRS product (p1nm) as the full-resolution source
    bad = tmp_path / 'hsrs.csv'
    bad.write_text(B.HSRS_HEADER + '\n355.000,1.0,0.01\n')
    with pytest.raises(SystemExit, match='E_SPECTRA_SOURCE'):
        B.read_hsrs(bad)  # right header, other bytes
    bad.write_bytes(b'\xff\xfe not ascii\n355.000,1.0,0.01\n')
    with pytest.raises(SystemExit, match='E_SPECTRA_SOURCE'):
        B.read_hsrs(bad)  # stable code, not UnicodeDecodeError (sha256 first)
    with pytest.raises(SystemExit, match='E_SPECTRA_SOURCE: TSIS-1 HSRS header'):
        B.read_hsrs(p1nm, B.P1NM_SHA)  # header path: bytes pinned, columns not the full-resolution ones
    assert B.read_hsrs(p1nm, B.P1NM_SHA, B.P1NM_HEADER).shape[1] == 4
    with pytest.raises(SystemExit, match='E_SPECTRA_SOURCE'):
        B.read_serdyuchenko(
            testkit.sky_sos('results/sigma_tz_20261028/out/aprime_p1/wavelength_axis_sources') / 'Spctrm_0.dat'
        )


def test_asset_declarations():
    fine, temp = (
        json.loads((paths.ASSETS / p).read_text()) for p in ('spectra_fine.json', 'spectra_fine_temperature.json')
    )
    assert (
        fine['wavelength_reference'].startswith('air') and temp['wavelength_reference'] == fine['wavelength_reference']
    )
    from sosfuse2.foundation.content_identity import asset_digest

    # The frozen reference pins published bytes; the loader exposes computation identity.
    assert asset_digest('spectra_fine.json', temp['frozen_source_sha256']) == SP.asset()['sha']
    assert temp['wavelength_nm'] == fine['wavelength_nm']
    assert all(temp[f'ozone_{t}K'] == fine[f'ozone_{t}K'] for t in (213, 233, 253)) and temp['solar'] == fine['solar']
    assert fine['generator']['air_wavelength']['script'] == 'sosfuse2.atmosphere.air_wavelength'  # Public module provenance.


@NEEDS_RAW
def test_independent_relabel():
    """Restates the relabel without the builder: grid columns within the 7-digit rounding, raw-point band means within the
    0.1 nm sampling error (measured max 3.9e-4 at 371.75-383.5 nm, median 2e-6; SPEC §4 v1 correction)."""
    O = np.loadtxt(RAW, skiprows=45)
    fine = json.loads(FINE.read_text())
    grid = np.asarray(fine['wavelength_nm'])
    x = O[:, 0] / AW.standard_n(O[:, 0])
    edges = SP.PRODUCT_EDGES
    for t, c in ((213, 9), (233, 7), (253, 5)):
        y = O[:, c] * 1e-4
        col = np.asarray(fine[f'ozone_{t}K'])
        np.testing.assert_allclose(col, np.interp(grid, x, y), rtol=6e-7, atol=1e-7 * np.abs(y).max())
        direct = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (x > lo) & (x < hi)
            xx = np.r_[lo, x[s], hi]
            direct.append(np.trapezoid(np.interp(xx, x, y), xx) / (hi - lo))
        asset = [SP.band_mean(col, lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]
        np.testing.assert_allclose(asset, direct, rtol=1e-3, atol=0)


@NEEDS_HSRS
def test_hsrs_jacobian_conserves_energy():
    """The builder's air relabel with the Jacobian keeps the source integral (measured 1.04e-14)."""
    B = builder()
    H = B.read_hsrs(HSRS)
    la, ea = B.to_air(H)
    assert np.all(np.diff(la) > 0)
    assert abs(np.trapezoid(ea, la) / np.trapezoid(H[:, 1], H[:, 0]) - 1) <= 1e-12


@NEEDS_HSRS
def test_independent_solar_cell_means():
    """Restates the solar column without the builder and without a Jacobian: each air cell [l - 0.05, l + 0.05] is mapped to
    the vacuum axis (fixed-point inverse of vacuum_to_air) and the vacuum source's linear interpolant is integrated there.
    Within the 7-digit rounding (measured max 4.70e-7, rtol 6e-7)."""
    d = np.loadtxt(HSRS, delimiter=',', skiprows=1)
    lv, ev = d[:, 0], d[:, 1]
    fine = json.loads(FINE.read_text())
    grid = np.asarray(fine['wavelength_nm'])
    edges = np.r_[grid - 0.05, grid[-1] + 0.05]
    vac = edges * 1.0003
    for _ in range(6):
        vac = edges * AW.standard_n(vac)
    assert np.max(np.abs(AW.vacuum_to_air(vac) - edges)) < 1e-12
    C = np.r_[0.0, np.cumsum(np.diff(lv) * (ev[:-1] + ev[1:]) / 2)]
    i = np.searchsorted(lv, vac, 'right') - 1
    F = C[i] + (vac - lv[i]) * (ev[i] + np.interp(vac, lv, ev)) / 2
    np.testing.assert_allclose(fine['solar'], np.diff(F) / 0.1, rtol=6e-7, atol=0)


@NEEDS_HSRS
@NEEDS_RAW
def test_builder_provenance_refuses_bad_solar_columns():
    """The builder's solar self-checks fail closed (E_SPECTRA_PROVENANCE) on a NaN, on point samples instead of cell
    means, and on an axis error that the to_air-based checks share (no relabel): only the vacuum-domain check sees the last."""
    B = builder()
    H = B.read_hsrs(HSRS)
    cols = B.columns(H, B.read_serdyuchenko(RAW))
    assert set(B.provenance(cols, H)['solar']) >= {
        'jacobian_energy_relative',
        'integral_360_840_relative',
        'vacuum_domain_max_relative',
    }
    nan = list(cols['solar'])
    nan[1000] = float('nan')
    point = B.sig(np.interp(B.GRID, *B.to_air(H)))
    no_relabel = B.sig(B.cell_means(H[:, 0], H[:, 1], B.GRID)[0])
    for bad in (nan, point, no_relabel):
        with pytest.raises(SystemExit, match='E_SPECTRA_PROVENANCE'):
            B.provenance(dict(cols, solar=bad), H)


@NEEDS_HSRS
def test_cpu_check_thresholds():
    """The filtered solar column vs exact source integrals — BM45 1e-3 (measured 7.3e-4),
    bins >= 5 nm 5e-3 (3.9e-3), Rayleigh 1e-4 (4.1e-5), sun |du'v'| 1e-5 (6.7e-7); a point-sampled column fails."""
    spec = importlib.util.spec_from_file_location('cpu_check', testkit.sky_sos('results/d1_axis_20261031/cpu_check.py'))
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    res, _ = C.measure()
    assert C.failures(res) == []
    point = dict(
        res,
        bins=res['point_sampled_bins'],
        rayleigh_relative=res['rayleigh_relative_point_sampled'],
        sun_duv=res['sun_duv_point_sampled'],
    )
    assert {'BM45', 'rayleigh', 'sun_duv'} <= set(C.failures(point))
