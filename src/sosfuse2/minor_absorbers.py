"""Offline cross-section loader. NO2 is packaged; O4 is explicitly prepared."""
import hashlib
import json

import numpy as np

from . import paths
from .generate_minor_absorbers import CIA_SHA, RULES_VERSION

NO2 = paths.ASSETS/'no2_vandaele1998_fine.json'
O4_NAME = 'o4_hitran2024_fine.json'
PREPARE = 'python -m sosfuse2.prepare_minor_absorbers (--cia FILE | --download)'
UNITS = {'no2_220K': 'm^2 molecule^-1 (0.1 nm cell mean)',
         'o4': 'm^5 molecule^-2 (0.1 nm cell mean; absorption coefficient = o4 * n_O2^2)'}


def o4_path():
    return paths.work_root()/'assets'/O4_NAME


def load(kind, *, required=True, path=None):
    """Validated FP64 values, exact byte SHA, and metadata. Never downloads."""
    path = path or (NO2 if kind == 'no2' else o4_path())
    if not path.is_file() and kind == 'o4':
        if not required:
            return None
        raise ValueError(f'E_O4_ASSET_MISSING: run {PREPARE}; expected {path}')
    try:
        raw = path.read_bytes()
        doc = json.loads(raw)
        field = 'no2_220K' if kind == 'no2' else 'o4'
        if kind == 'o4' and (doc.get('generator', {}).get('rules_version') != RULES_VERSION or
                             doc.get('source', {}).get('sha256') != CIA_SHA):
            raise ValueError('E_O4_ASSET_STALE: source SHA or rules version differs; run '+PREPARE)
        grid = np.asarray(doc['wavelength_nm'], 'f8')
        fine = json.loads((paths.ASSETS/'spectra_fine.json').read_bytes())
        values = np.asarray(doc[field], 'f8')
        if (doc['format'] != f'{kind}-fine-v1' or not np.array_equal(grid, fine['wavelength_nm']) or
                values.shape != grid.shape or not np.isfinite(values).all() or np.any(values < 0) or
                doc['units'].get(field) != UNITS[field] or doc['units'].get('wavelength_nm') != 'nm (air)'):
            raise ValueError('format, grid, finite nonnegative values or units')
    except (OSError, KeyError, TypeError, ValueError) as exc:
        if str(exc).startswith('E_O4_ASSET_STALE:'):
            raise
        raise ValueError(f'E_{kind.upper()}_ASSET_INVALID: {exc}') from exc
    return dict(wavelength_nm=grid, values=values, sha256=hashlib.sha256(raw).hexdigest(), meta=doc)


def hashes():
    o4 = load('o4', required=False)
    return dict(no2=load('no2')['sha256'], o4=o4['sha256'] if o4 else None)


def representative(block, wavelengths, *, o4=True):
    """Band means or the interpolant of 0.1 nm cell means; all arithmetic FP64."""
    from .spectral import band_mean
    result = {}
    for kind in ('no2', 'o4'):
        if kind == 'o4' and not o4:
            result[kind] = np.zeros_like(wavelengths, dtype='f8')
            continue
        asset = load(kind)
        if block is not None and block.get('minor_absorber_sha256', {}).get(kind) != asset['sha256']:
            raise ValueError('E_SPECTRAL_CONTRACT: minor absorber asset changed since block creation')
        if block is not None and block['mode'] == 'band':
            edges = block['bin_edges_nm']
            result[kind] = np.array([band_mean(asset['values'], a, b) for a, b in zip(edges[:-1], edges[1:])])
        else:
            result[kind] = np.interp(wavelengths, asset['wavelength_nm'], asset['values'])
    return result
