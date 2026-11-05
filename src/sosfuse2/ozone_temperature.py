"""Opt-in A′ ozone scalars: band means followed by a profile-weighted layer mean (sky-sos research records).

Arithmetic follows results/sigma_tz_20261028/model.py:87-147 (BM45 bin mean of every 193..293 K column, linear in T,
vertical 64-point Gauss–Legendre on every knot interval). No research code is imported.

The q(z) shape is a property of the state, not of the settings: spectral.resolve(block, state) fills the block's
ozone_profile from state_shape(state), the shape the engine itself uses (density.build mask bit 2; analytic triangle
otherwise). An injected external ozone column (density.with_profile, mask bit 2) has no profile_mean identity.
"""
import hashlib
import json

import numpy as np

from . import density, spectral as SP

FINE = SP.paths.ASSETS / 'spectra_fine_temperature.json'
TEMP = SP.paths.ASSETS / 'temperature_usstd76.json'
TEMPS = np.arange(193., 294., 10.)
Z = np.arange(161) * .5
N = 64
SHAPES = ('triangle', 'usstd76')
# Definition descriptor signed in the block and block_key: a different averaging rule is a different spectral definition.
RULE = ('v1: BM45 bin mean of each 193-293 K column; linear in T, outside 193-293 K refused (no clamp); '
        'q-weighted GL64 on knots = 0.5 km grid + {10,25,40} km + T-column crossings of T(z)')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile_identity(shape):
    if shape == 'triangle':
        return dict(name='triangle')  # analytic max(0, 1 - abs(z - 25) / 15)
    if shape == 'usstd76':
        return dict(name=shape, asset_sha256=sha(density.ASSET))
    raise ValueError('E_OZONE_SHAPE: ozone_profile in ' + ', '.join(SHAPES))


def identity(shape):
    """profile_mean block fields; shape None = not yet resolved for a state (spectral.resolve)."""
    return dict(temperature_asset_sha256=sha(TEMP), ozone_sigma_asset_sha256=sha(FINE), ozone_temperature_rule=RULE,
                ozone_profile=None if shape is None else profile_identity(shape))


def same_profile_identity(a, b):
    """True when two ozone_profile identities name the same logical profile: the same shape, and for usstd76 asset digests
    that are equal or both in {density.ASSET_SHA256} | density.ASSET_SHA256_SUPERSEDED (an earlier generation of the same
    asset; docs/PROVENANCE_V3.md "Superseded asset digests"). Any other digest is another asset and the callers refuse it."""
    if not (isinstance(a, dict) and isinstance(b, dict)) or a.get('name') != b.get('name'): return False
    if a.get('name') != 'usstd76': return a == b
    known = {density.ASSET_SHA256, *density.ASSET_SHA256_SUPERSEDED}
    da, db = a.get('asset_sha256'), b.get('asset_sha256')
    return da == db or (da in known and db in known)


def identity_matches(block, expected):
    """spectral.check_asset rule: the fixed fields of identity() equal, the ozone_profile equal up to superseded digests."""
    return (all(block.get(k) == v for k, v in expected.items() if k != 'ozone_profile')
            and same_profile_identity(block.get('ozone_profile'), expected['ozone_profile']))


def asset_items(shape):
    """(provenance key, file) of the inputs of a profile_mean block (signed_contract code_hashes, gate code_identity).
    The ozone profile asset keeps its logical key wherever it is materialised (density.ASSET_KEY)."""
    items = [(SP.paths.rel(FINE), FINE), (SP.paths.rel(TEMP), TEMP)]
    return items + ([(density.ASSET_KEY, density.asset_path())] if shape == 'usstd76' else [])


def asset_files(shape):
    return [p for _, p in asset_items(shape)]


def asset_paths(shape):
    """Provenance keys of asset_files(shape)."""
    return [k for k, _ in asset_items(shape)]


def state_shape(state):
    """The ozone q(z) shape the engine uses for this state ('triangle' or 'usstd76').

    density.build: only mask bit 2 selects the tabulated ozone column; without it the kernel keeps the analytic
    triangle (an external table without bit 2 included). An external ozone column is refused (no identity here)."""
    ext = density.external(state)
    if ext is not None:
        if ext['mask'] & 2:
            raise ValueError('E_SPECTRAL_OZONE_PROFILE_MISMATCH: an external ozone table (density.external_profile, '
                             'mask bit 2) has no profile_mean q identity')
        return 'triangle'
    return density.options(state)['ozone_profile']


class ProfileMean:
    def __init__(self, bin_edges_nm, shape, *, band_mean_of='band', wavelengths=None):
        self.shape = shape
        raw_fine, raw_temp = FINE.read_bytes(), TEMP.read_bytes()
        # The SHAs of the bytes actually parsed, so a block check against them cannot race a file change.
        self.meta = dict(temperature_asset_sha256=hashlib.sha256(raw_temp).hexdigest(),
                         ozone_sigma_asset_sha256=hashlib.sha256(raw_fine).hexdigest(), ozone_temperature_rule=RULE,
                         ozone_profile=profile_identity(shape))
        a = json.loads(raw_fine); t = json.loads(raw_temp)
        if a['frozen_source_sha256'] != SP.asset()['sha']:
            raise ValueError('E_OZONE_T_ASSET_SHA: frozen spectral source')
        if (not np.array_equal(a['wavelength_nm'], SP.asset()['wl']) or
                not np.array_equal(a['temperatures_K'], TEMPS) or
                t['units'] != dict(altitude='geometric km', temperature='K') or
                not np.array_equal(t['altitude_km'], Z)):
            raise ValueError('E_OZONE_T_ASSET_GRID')
        self.t = np.asarray(t['temperature_K'], float)
        if self.t.shape != Z.shape:
            raise ValueError('E_OZONE_T_ASSET_GRID')
        self.check_temperature(self.t)
        edges = np.asarray(bin_edges_nm, float)
        if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or
                np.any(np.diff(edges) <= 0) or edges[0] < SP.asset()['wl'][0] or edges[-1] > SP.asset()['wl'][-1]):
            raise ValueError('E_SPECTRAL_EDGES')
        columns = np.asarray([a[f'ozone_{int(k)}K'] for k in TEMPS], float)
        # Preserve the measured columns, including small negative source samples;
        # the research definition neither rejects nor clamps them.
        if columns.shape != (len(TEMPS), len(a['wavelength_nm'])) or not np.isfinite(columns).all():
            raise ValueError('E_OZONE_T_ASSET_GRID: sigma columns')
        if band_mean_of == 'band':
            self.sigmas = np.array([[SP.band_mean(column, lo, hi)
                                    for lo, hi in zip(edges[:-1], edges[1:])] for column in columns])
        elif band_mean_of == 'point':
            wl = np.asarray(wavelengths, float)
            if (wl.shape != (len(edges)-1,) or not np.isfinite(wl).all() or
                    np.any(wl < edges[:-1]) or np.any(wl >= edges[1:])):
                raise ValueError('E_SPECTRAL_EDGES: point wavelengths must lie in their bins')
            self.sigmas = np.array([np.interp(wl, a['wavelength_nm'], column) for column in columns])
        else:
            raise ValueError('E_SPECTRAL_CONTRACT: band_mean_of in band, point')
        if shape == 'usstd76':   # the analytic triangle never touches the usstd76 asset (offline triangle runs, 2026-11-03 release review)
            _, _, _, oz = density.load_asset()
            self.q_usstd = oz*15/np.trapezoid(oz, Z)
        else: self.q_usstd = None
        self.meta.update(knot_count=len(self.knots()), quadrature_n=N, band_mean_of=band_mean_of)

    @staticmethod
    def check_temperature(t):
        if not np.isfinite(t).all() or np.any(t < 193) or np.any(t > 293):
            raise ValueError('E_OZONE_T_RANGE')

    def sigma(self, t):
        t = np.asarray(t, float)
        self.check_temperature(t)
        i = np.minimum(np.searchsorted(TEMPS, t, side='right')-1, 9)
        f = (t-TEMPS[i])/10
        return self.sigmas[i]*(1-f[..., None])+self.sigmas[i+1]*f[..., None]

    def q(self, z):
        if self.shape == 'triangle':
            return np.maximum(0, 1-np.abs(np.asarray(z)-25)/15)
        return np.interp(z, Z, self.q_usstd)

    def temp(self, z):
        return np.interp(z, Z, self.t)

    def knots(self):
        knots = list(Z)+[10., 25., 40.]
        for z0, z1, t0, t1 in zip(Z[:-1], Z[1:], self.t[:-1], self.t[1:]):
            if t1 != t0:
                f = (TEMPS-t0)/(t1-t0)
                knots += (z0+(z1-z0)*f[(f > 0) & (f < 1)]).tolist()
        return np.unique(knots)

    def quadrature(self):
        edges = np.unique(np.r_[self.knots(), ()])
        gx, gw = np.polynomial.legendre.leggauss(N)
        z = ((edges[1:]+edges[:-1])[:, None]+np.diff(edges)[:, None]*gx)/2
        w = np.diff(edges)[:, None]*gw/2
        return z.ravel(), w.ravel()

    def average(self):
        z, w = self.quadrature()
        qw = self.q(z)*w
        return qw@self.sigma(self.temp(z))/qw.sum()


def sigma_bar(bin_edges_nm, shape, *, band_mean_of='band', wavelengths=None):
    """Return (sigma_bar [m²], metadata); point sampling is comparison-only (spec_table always uses band)."""
    model = ProfileMean(bin_edges_nm, shape, band_mean_of=band_mean_of, wavelengths=wavelengths)
    return model.average(), model.meta
