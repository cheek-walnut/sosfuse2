"""Portable previews and an atomic archive of four products plus optional first-order radiance."""

import io
import json
import os
import shutil
import subprocess
import tarfile
import time
from functools import partial
from pathlib import PurePosixPath, PureWindowsPath

import numpy as np

from sosfuse2.foundation import paths
from sosfuse2.outputs import preview_functions

ROOT = paths.PKG
PREVIEW_PNG_LEVEL = 6


def render_functions():
    """Renderer mapping shared by the packer and the runner's spectral preview path."""
    weights = preview_functions.Y.copy()
    return dict(
        Y=weights,
        film=preview_functions.film,
        tone=preview_functions.tone,
        encode=preview_functions.encode,
        save_png=partial(preview_functions.save_png, compress_level=PREVIEW_PNG_LEVEL),
        projection=preview_functions.projection,
        cubic_weights=preview_functions.cubic_weights,
        sample=preview_functions.sample,
        project=preview_functions.project,
        central=partial(preview_functions.central, weights=weights),
    )


def preview_band_widths(meta, wl):
    """Band widths (nm) of a table for the preview weights: the resolved bin_edges_nm of meta['settings']['spectral'],
    else equal widths over 360-830 nm (legacy tables, whose edges are linspace). Edges that do not bracket the table
    wavelengths raise E_PREVIEW_EDGES."""
    block = (meta.get('settings') or {}).get('spectral') or {}
    edges = block.get('bin_edges_nm')
    if edges is None:
        return np.full(len(wl), 470 / len(wl))
    edges = np.asarray(edges, float)
    if edges.shape != (len(wl) + 1,) or np.any(wl < edges[:-1]) or np.any(wl >= edges[1:]):
        raise RuntimeError(f'E_PREVIEW_EDGES: {len(edges)} edges do not bracket {len(wl)} table wavelengths')
    return np.diff(edges)


def preview_weights(wl, widths, base_weights):
    """RGB weights for a table with band centres `wl` and widths `widths` (nm) from the 15 equal-band set (31.33 nm bands
    over 360-830 nm): interpolated to the centres and scaled by width/31.33, so a band contributes in proportion to the
    wavelength range it covers. Up to 1.3.1 the scale was 15/len(wl), which for the BM45 product edges (35 bands of
    11.75 nm, 10 of 5.875 nm) gave the narrow bands 1.78x and the wide ones 0.89x their share (G about +30 %)."""
    base = 360 + (np.arange(15) + 0.5) * 470 / 15
    interp = np.column_stack([np.interp(wl, base, base_weights[:, k]) for k in range(3)])
    return interp * (np.asarray(widths) / (470 / 15))[:, None]


def table_weights(meta, wl, weights):
    widths = preview_band_widths(meta, wl)
    centres = 360 + (np.arange(15) + 0.5) * 470 / 15
    standard_widths = np.diff(np.linspace(360, 830, 16))
    if (np.array_equal(wl, centres.astype(wl.dtype))
            and (np.array_equal(widths, standard_widths) or np.all(widths == 470 / 15))):
        return weights
    return preview_weights(wl, widths, weights)


def previews(out):
    from sosfuse2.atmosphere.spectral import check_meta_contract

    meta = json.loads((out / 'meta.json').read_text())
    check_meta_contract(meta)
    funcs = render_functions()
    weights = np.load(ROOT / 'assets/rgb_weights.npy')
    with np.load(out / 'table.npz') as f:
        table = f['radiance']
        wl = f['wavelength_nm']
        hs = f['solar_elevation_deg']
    if len(wl) == 1:
        rgb = np.repeat(table, 3, axis=-1)  # validation preview is explicitly monochrome
    elif len(wl) > 0:
        rgb = table @ table_weights(meta, wl, weights)
    else:
        raise RuntimeError('E_PREVIEW_SPECTRUM: unsupported bin count')
    for h in (-3, -6):
        hit = np.flatnonzero(hs == h)
        if len(hit) != 1:
            raise RuntimeError(f'E_PREVIEW_ALTITUDE: missing {h}')
        folded = rgb[hit[0]]
        # Renderer sample expects azimuth -179.5..179.5 (sun at central columns).
        pano = np.concatenate([folded[:, ::-1], folded], axis=1)
        projected = funcs['project'](pano, funcs['projection'](960, 540, 100, 0))
        if (projected @ funcs['Y']).max() > 0:
            _, k = funcs['central'](projected)
        else:
            k = 1.0
        funcs['save_png'](out / f'preview_m{abs(h):02d}.png', funcs['tone'](projected, k))


def _anonymous(info):
    info.uid = info.gid = 0
    info.uname = info.gname = ''
    return info


def _write_products(tar, out, names, meta):
    # Leave diagnostic details in the loose record for local monitoring.
    for timing in meta.get('timings') or ():
        worker = timing.get('solve_worker') if isinstance(timing, dict) else None
        if isinstance(worker, dict):
            worker.pop('pid', None)
            file = worker.get('file')
            if isinstance(file, str):
                for path in (PurePosixPath(file), PureWindowsPath(file)):
                    if path.is_absolute():
                        worker['file'] = path.name
                        break
    meta.pop('gpu', None)
    data = (json.dumps(meta, indent=2, allow_nan=False) + '\n').encode()
    for name in names:
        if name == 'meta.json':
            info = _anonymous(tar.gettarinfo(str(out / name), arcname=name))
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        else:
            tar.add(out / name, arcname=name, recursive=False, filter=_anonymous)


def pack_state(out, pack, compression='auto', started=None, *, reuse_previews=False):
    """Reuse previews only after a successful pack in the same attempt."""
    meta = json.loads((out / 'meta.json').read_text())
    if not meta.get('complete'):
        raise RuntimeError('E_PACK_INCOMPLETE')
    from sosfuse2.atmosphere.spectral import check_meta_contract
    from sosfuse2.outputs.radiance_table import exports_first_order

    exported = exports_first_order(meta)
    check_meta_contract(meta)
    if not reuse_previews:
        previews(out)
    pack.mkdir(parents=True, exist_ok=True)
    from sosfuse2.foundation.host_limits import peak_rss_bytes

    peak = peak_rss_bytes()
    meta['peak_rss_bytes'] = max(meta.get('peak_rss_bytes', 0), peak)
    if started is not None:
        meta['elapsed_seconds'] = time.monotonic() - started
        meta['elapsed_scope'] = 'current successful attempt through previews; archive compression excluded'
    meta_tmp = (out / 'meta.json').with_suffix('.json.tmp')
    meta_tmp.write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
    os.replace(meta_tmp, out / 'meta.json')
    zstd = shutil.which('zstd')
    if compression == 'zstd' and not zstd:
        raise RuntimeError('E_ZSTD_MISSING: install zstd (required by pack and verify)')
    use_zstd = bool(zstd) and compression != 'gzip'
    suffix = '.tar.zst' if use_zstd else '.tar.gz'
    target = pack / (out.name + suffix)
    tmp = target.with_suffix(target.suffix + '.tmp')
    names = ('table.npz', 'meta.json', 'preview_m03.png', 'preview_m06.png')
    if exported:
        first = meta.get('first_order')
        if not (out / 'first_order.npz').is_file() or not isinstance(first, dict) or first.get('file') != 'first_order.npz':
            raise RuntimeError('E_FIRST_ORDER_MISSING: first-order file or metadata missing')
        names += ('first_order.npz',)
    try:
        if use_zstd:
            with tmp.open('wb') as f:
                process = subprocess.Popen(
                    [zstd, '-q', '-T' + os.environ.get('OMP_NUM_THREADS', str(os.cpu_count() or 1)), '-c'],
                    stdin=subprocess.PIPE,
                    stdout=f,
                )
                try:
                    with tarfile.open(fileobj=process.stdin, mode='w|') as tar:
                        _write_products(tar, out, names, meta)
                finally:
                    process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError('E_PACK_ZSTD_FAILED')
        else:
            with tarfile.open(tmp, 'w:gz') as tar:
                _write_products(tar, out, names, meta)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    print(f'PACK {target.name} {target.stat().st_size} bytes', flush=True)
    return target
