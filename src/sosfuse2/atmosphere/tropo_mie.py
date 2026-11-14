"""Tropospheric aerosol Mie phase function (opt-in; HG stays the default).

A state enables it with a top-level block
    "troposphere_phase": {"status": "enabled", "model": "mie", "optics_table": "default" | "<package-relative or absolute json>"}
An absent block or status "disabled" uses the historical Henyey-Greenstein path (parameters.g).
Other status values are rejected.

Only the angular distribution changes. Extinction, SSA and the Angstrom law stay the state's scalar
parameters (aod550, ssa, angstrom), so an HG/Mie pair differs in the phase function alone.
  * single scattering of the direct sun and of first-ground light (kernel.cu, -DTROPO_MIE): per-wavelength
    table of the full (untruncated) Mie phase function, per steradian, on the strato LUT grid
    u = -1 + i*2/TROPO_LUT_BINS (linear in u, same node formula as STRATO_MIE_LUT_FINE);
  * multiple scattering (source_solver.Solver.apply): SH eigenvalues chi_l/(2l+1), l <= 32, per wavelength,
    replacing g**l. Plain truncation at L32, no delta-M (see generate_tropo_optics.py checks: the
    eigenvalue at l=33 is ~1.6-2 %).
The kernel reads the table behind the asca pointer (asca[0:W] unchanged, table at asca+W), so the kernel
signature and launch argument lists do not change.
"""

import hashlib, json, math
from pathlib import Path
import numpy as np

from ..foundation import paths

ROOT = paths.PKG  # relative optics_table paths resolve inside the package
DEFAULT = paths.ASSETS / 'tropo_rural_sf79_rh70.json'
LUT_BINS = 16384  # == STRATO_LUT_BINS under STRATO_MIE_LUT_FINE (kernel.cu GF path shares the node indices)
L = 32


STATUS = ('enabled', 'disabled')
KEYS = {'status', 'model', 'optics_table'}
# Asset acceptance (load_table). For tropo_rural_sf79_rh70.json, the table mean over the sphere on its
# own theta grid |-1| <= 2.6e-5; LUT (16384 bins, linear in u) mean |-1| <= 2.0e-3 (forward first bin); SH eigenvalue
# chi_l/(2l+1) recomputed from the LUT vs the table <= 2.0e-3 (l <= 32). Tolerances = about 10x / 2.5x / 2.5x of these.
TABLE_MEAN_TOL = 2.5e-4
LUT_MEAN_TOL = 5e-3
EIGEN_TOL = 5e-3


def config(state):
    """None = HG: no key at all, or status "disabled". An enabled block must name model "mie" and an optics_table
    (a path, or the explicit word "default" for the shipped asset). Anything else (null included) is rejected, never
    read as HG."""
    if not isinstance(state, dict) or 'troposphere_phase' not in state:
        return None
    block = state['troposphere_phase']
    if not isinstance(block, dict):
        raise ValueError('E_TROPO_PHASE: troposphere_phase must be an object')
    status = block.get('status')
    if status == 'disabled':
        return None
    if status != 'enabled':
        raise ValueError('E_TROPO_PHASE_STATUS: status must be "enabled" or "disabled"')
    if set(block) - KEYS:
        raise ValueError('E_TROPO_PHASE_KEYS: unknown keys ' + ','.join(sorted(set(block) - KEYS)))
    if block.get('model') != 'mie':
        raise ValueError('E_TROPO_PHASE_MODEL: only model "mie" is supported')
    table = block.get('optics_table')
    if not isinstance(table, str) or not table.strip():
        raise ValueError('E_TROPO_PHASE_TABLE: optics_table required (path, or "default")')
    return block


def table_path(block):
    raw = block['optics_table']
    path = DEFAULT if raw == 'default' else Path(raw)
    return path if path.is_absolute() else ROOT / path


def path_label(path):
    return paths.rel(path) if Path(path).resolve().is_relative_to(ROOT) else Path(path).name


def load_table(path):
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise ValueError('E_TROPO_TABLE_READ') from e
    try:
        if d['version'] != 1:
            raise ValueError()
        wl = np.array(d['wavelengths_nm'], float)
        th = np.array(d['theta_deg'], float)
        rows = d['rows']
        chi = np.array([r['chi_l'][: L + 1] for r in rows], float)
        P = np.array([r['phase'] for r in rows], float)
        rwl = np.array([r['wavelength_nm'] for r in rows], float)
        g = np.array([r['g'] for r in rows], float)
        if not all(np.isfinite(v).all() for v in (wl, th, chi, P, rwl, g)):
            raise ValueError()
        if not np.array_equal(wl, rwl) or wl.ndim != 1 or len(wl) < 2 or np.any(wl <= 0) or np.any(np.diff(wl) <= 0):
            raise ValueError()
        if th.ndim != 1 or th[0] != 0 or th[-1] != 180 or np.any(np.diff(th) <= 0) or P.shape != (len(wl), len(th)):
            raise ValueError()
        if chi.shape != (len(wl), L + 1) or np.any(P <= 0):
            raise ValueError()
        if (
            np.max(abs(chi[:, 0] - 1)) > 1e-6
            or np.any(abs(chi) > 2 * np.arange(L + 1) + 1 + 1e-8)
            or np.max(abs(g - chi[:, 1] / 3)) > 1e-8
        ):
            raise ValueError()
    except (KeyError, TypeError, ValueError, IndexError) as e:
        raise ValueError('E_TROPO_TABLE_SCHEMA') from e
    mu = np.cos(np.deg2rad(th))
    mean = (-np.diff(mu) * (P[:, :-1] + P[:, 1:]) / 2).sum(1) / 2
    if np.max(abs(mean - 1)) > TABLE_MEAN_TOL:
        raise ValueError(
            f'E_TROPO_TABLE_NORMALIZATION: table mean {mean.min():.6f}..{mean.max():.6f}, tolerance {TABLE_MEAN_TOL}'
        )
    u = lut_nodes()
    rowsl = phase_on(np.rad2deg(np.arccos(np.clip(u, -1, 1))), th, P)
    w = np.full(len(u), 2.0 / LUT_BINS)
    w[[0, -1]] /= 2
    lmean = (rowsl @ w) / 2
    if np.max(abs(lmean - 1)) > LUT_MEAN_TOL:
        raise ValueError(
            f'E_TROPO_LUT_NORMALIZATION: LUT mean {lmean.min():.6f}..{lmean.max():.6f}, tolerance {LUT_MEAN_TOL}'
        )
    eig = ((rowsl * w) @ np.polynomial.legendre.legvander(u, L)) / 2  # eigenvalues chi_l/(2l+1) of the LUT
    diff = np.max(abs(eig - chi / (2 * np.arange(L + 1) + 1)))
    if diff > EIGEN_TOL:
        raise ValueError(f'E_TROPO_CHI_LUT: eigenvalue from LUT vs table {diff:.2e} > {EIGEN_TOL}')
    chi[:, 0] = 1.0
    return wl, th, chi, P


def lut_nodes(bins=LUT_BINS):
    """u nodes exactly as the kernel indexes them: x=(u+1)*(bins/2), node i at u=-1+i*(2/bins)."""
    return -1.0 + np.arange(bins + 1) * (2.0 / bins)


def phase_on(theta_deg, th, P):
    """Log-linear interpolation of the tabulated mean-one phase (rows) at angles theta_deg."""
    logp = np.log(P)
    return np.exp(np.stack([np.interp(theta_deg, th, row) for row in logp]))


def build_optics(state, wavelengths_nm, bins=LUT_BINS):
    block = config(state)
    if block is None:
        return None
    path = table_path(block)
    wl, th, chi, P = load_table(path)
    wanted = np.asarray(wavelengths_nm, float)
    if wanted.ndim != 1 or not np.isfinite(wanted).all() or np.any(wanted < wl[0]) or np.any(wanted > wl[-1]):
        raise ValueError('E_TROPO_WAVELENGTH_RANGE')
    coeff = np.stack([np.interp(wanted, wl, chi[:, l]) for l in range(L + 1)])  # [33,W]
    u = lut_nodes(bins)
    theta = np.rad2deg(np.arccos(np.clip(u, -1, 1)))
    rows = phase_on(theta, th, P)  # [nwl, bins+1]
    # Linear in wavelength between table rows (exact at the 15 band centres, which are table rows).
    j = np.searchsorted(wl, wanted, side='right').clip(1, len(wl) - 1)
    f = (wanted - wl[j - 1]) / (wl[j] - wl[j - 1])
    mean_one = rows[j - 1] * (1 - f)[:, None] + rows[j] * f[:, None]  # [W, bins+1]
    lut = np.ascontiguousarray((mean_one / (4 * math.pi)).T, 'f4')  # [bins+1, W], per sr
    du = 2.0 / bins
    energy = (2 * math.pi * du * (lut[:-1].astype('f8') + lut[1:].astype('f8')) / 2).sum(
        0
    )  # trapezoid of the linear LUT
    return dict(
        chi=coeff,
        eigen=coeff / (2 * np.arange(L + 1)[:, None] + 1),
        lut=lut,
        bins=bins,
        meta=dict(
            model='mie',
            optics_table=path_label(path),
            optics_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            single_scattering='full Mie phase, per-wavelength LUT linear in u, bins %d' % bins,
            multiple_scattering='SH eigenvalues chi_l/(2l+1), l<=32, plain truncation (no delta-M)',
            extinction_ssa_angstrom='state scalars (unchanged from HG path)',
            lut_energy_trapezoid=[float(x) for x in energy],
        ),
    )


def eigen_by_sh(optics, L=32):
    from ..backends.sh_layout import count

    count(L)
    levels = np.array([l for l in range(L + 1) for m in range(l + 1)])
    return optics['eigen'][levels]


def attach(s):
    """Idempotent: s.tropo = optics or None. With Mie on, s.af (HG g**l) is cleared so a path that still
    multiplies by it fails loudly instead of silently mixing HG into a Mie state."""
    if hasattr(s, 'tropo'):
        return s.tropo
    optics = build_optics(getattr(s, 'fuse_state', {}), s.wl)
    s.tropo = optics
    if optics is not None:
        s.af = None
    return optics
