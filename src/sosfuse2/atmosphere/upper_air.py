"""Pinned upper-atmosphere effective Rayleigh density; no generator dependencies."""

import hashlib
import json

import numpy as np

from ..foundation import paths
from ..foundation.top import UPPER_AIR_ASSET_SHA256 as ASSET_SHA256

ASSET_NAME = 'upper_air_us76_neff.json'


def load():
    raw = (paths.ASSETS / ASSET_NAME).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ASSET_SHA256:
        raise ValueError('E_UPPER_AIR_ASSET_SHA256: upper air asset differs from the pin')
    data = json.loads(raw)
    z, ratio = np.asarray(data['z_km'], 'f8'), np.asarray(data['neff_ratio'], 'f8')
    if (data['version'] != 1 or z.shape != (961,) or ratio.shape != z.shape
            or not np.array_equal(z, 80. + np.arange(961) * .125)
            or not np.isfinite(ratio).all() or np.any(ratio <= 0) or ratio[0] != 1.
            or np.any(np.diff(ratio) >= 0)):
        raise ValueError('E_UPPER_AIR_ASSET: invalid effective number-density grid')
    return data, z, ratio


def sample(heights):
    """PROFILE grids are exact subsets of the asset's geometric-height grid."""
    data, z, values = load()
    h = np.asarray(heights, 'f8')
    if not np.isfinite(h).all() or np.any(h < z[0]) or np.any(h > z[-1]):
        raise ValueError('E_ASSET_RANGE: upper_air_us76_neff.json: requested height outside asset')
    index = (h - z[0]) / .125
    if np.any(index != np.rint(index)):
        raise ValueError('E_PROFILE_GRID: upper-air samples must lie on the asset grid')
    return values[index.astype('i8')], data['metadata']


def connection_metadata(s, optics):
    """Measure the final, already scaled coefficient; do not apply normalization again."""
    from . import rayleigh
    from .density import load_asset

    step = optics['domain'].step_km
    boundary_value = optics['table'][int(80. / step), 0]
    beta = np.asarray(s.ext[0], 'f8') * boundary_value
    _, meta = sample([80.])
    _, _, lower_air, _ = load_asset()
    # sigma() is cm²/molecule; cm² -> m² and m^-1 -> km^-1.
    absolute = rayleigh.sigma(rayleigh.vacuum_nm(s.wl)) * 1e-4 * meta['n_tot_80_m3'] * 1e3
    return dict(policy='continuous Rayleigh coefficient at lower support boundary',
                wavelength_nm=np.asarray(s.wl, 'f8').tolist(), beta_rayleigh_80_minus_per_km=beta.tolist(),
                lower_profile_80=float(boundary_value), normalization_scale=float(optics['scale']),
                lower_profile_normalization_factor=float(boundary_value / lower_air[-1]),
                absolute_us76_ratio=(beta / absolute).tolist(),
                absolute_cross_section='Bodhaine (1999), engine air-to-vacuum wavelength conversion',
                n_tot_80_m3=meta['n_tot_80_m3'], asset_sha256=ASSET_SHA256)
