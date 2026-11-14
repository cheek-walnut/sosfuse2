"""Offline cross-section loader. NO2 is packaged; O4 is explicitly prepared."""

import hashlib
import json

import numpy as np

from ..foundation import paths, content_identity as source_identity

# The O4 rules version and the SHA-256 of the HITRAN CIA source file. load() checks both on a prepared O4 asset;
# asset_tools/generate_minor_absorbers.py imports them from here and writes them into the asset it builds.
RULES_VERSION = 'b2-s2-v2'   # v2: missing-marker policy, interior gaps and unknown markers refused
CIA_SHA = '0146acfe5c1414b8deaa56be9a63159154479d5804d1615855888a2e1efd4961'

NO2 = paths.ASSETS / 'no2_vandaele1998_fine.json'
O4_NAME = 'o4_hitran2024_fine.json'
PREPARE = 'python -m sosfuse2.asset_tools.prepare_minor_absorbers (--cia FILE | --download)'
UNITS = {
    'no2_220K': 'm^2 molecule^-1 (0.1 nm cell mean)',
    'o4': 'm^5 molecule^-2 (0.1 nm cell mean; absorption coefficient = o4 * n_O2^2)',
}


def o4_path():
    """Where the prepared O4 asset lives. Locating it never creates the work root: a reader only needs to know
    whether the file is there, and prepare() creates the directory before writing."""
    return paths.work_root(create=False) / 'assets' / O4_NAME


def load(kind, *, required=True, path=None):
    """Validated FP64 values, content identity, byte-integrity SHA, and metadata. Never downloads."""
    path = path or (NO2 if kind == 'no2' else o4_path())
    try:
        present = path.is_file()
    except OSError:  # an unreadable parent: Python before 3.14 raises here instead of returning False
        present = False
    if not present and kind == 'o4':
        if not required:
            return None
        raise ValueError(f'E_O4_ASSET_MISSING: run {PREPARE}; expected {path}')
    try:
        raw = path.read_bytes()
        doc = json.loads(raw)
        field = 'no2_220K' if kind == 'no2' else 'o4'
        if kind == 'o4' and (
            doc.get('generator', {}).get('rules_version') != RULES_VERSION
            or doc.get('source', {}).get('sha256') != CIA_SHA
        ):
            raise ValueError('E_O4_ASSET_STALE: source SHA or rules version differs; run ' + PREPARE)
        grid = np.asarray(doc['wavelength_nm'], 'f8')
        fine = json.loads((paths.ASSETS / 'spectra_fine.json').read_bytes())
        values = np.asarray(doc[field], 'f8')
        if (
            doc['format'] != f'{kind}-fine-v1'
            or not np.array_equal(grid, fine['wavelength_nm'])
            or values.shape != grid.shape
            or not np.isfinite(values).all()
            or np.any(values < 0)
            or doc['units'].get(field) != UNITS[field]
            or doc['units'].get('wavelength_nm') != 'nm (air)'
        ):
            raise ValueError('format, grid, finite nonnegative values or units')
    except (OSError, AttributeError, KeyError, TypeError, ValueError) as exc:  # AttributeError: not a JSON object
        if str(exc).startswith('E_O4_ASSET_STALE:'):
            raise
        raise ValueError(f'E_{kind.upper()}_ASSET_INVALID: {exc}') from exc
    name = NO2.name if kind == 'no2' else O4_NAME
    return dict(wavelength_nm=grid, values=values, sha256=source_identity.asset_identity(raw, name),
                raw_sha256=hashlib.sha256(raw).hexdigest(), meta=doc)


def canonical_hashes(value):
    if not isinstance(value, dict):
        return value
    names = dict(no2=NO2.name, o4=O4_NAME)
    result = dict(value)
    for kind, digest in value.items():
        if kind not in names or not isinstance(digest, str):
            continue
        result[kind] = source_identity.asset_digest(names[kind], digest)
        if result[kind] == digest:
            asset = load(kind, required=False)
            if asset is not None and digest == asset['raw_sha256']:
                result[kind] = asset['sha256']
    return result


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
        if block is not None and canonical_hashes(block.get('minor_absorber_sha256', {})).get(kind) != asset['sha256']:
            raise ValueError('E_SPECTRAL_CONTRACT: minor absorber asset changed since block creation')
        if block is not None and block['mode'] == 'band':
            edges = block['bin_edges_nm']
            result[kind] = np.array([band_mean(asset['values'], a, b) for a, b in zip(edges[:-1], edges[1:])])
        else:
            result[kind] = np.interp(wavelengths, asset['wavelength_nm'], asset['values'])
    return result
