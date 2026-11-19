#!/usr/bin/env python3
"""Fast folded radiance table from a stored fused source (source.npz + state -> table.npz).

Same rays, same endpoint packs, same CUDA kernel (phase 2), same host
post-processing as `s.query` (Solver.query):

* capture: `s.query` is run with a recording transport, so directions and the
  64-ray endpoint packs are produced by the unchanged accepted code;
* launch: all captured rays of many solar elevations go into ONE multi-block
  Bundle per `first` flag (the original launched one CUDA block per 64 rays,
  i.e. one SM of the GPU, and re-packed/uploaded the whole source per launch);
* replay: `s.query` is run again with a transport that returns the computed
  per-ray values, so reshape, [...,:3].sum(-1) and the solar factor are the
  accepted operations.

Per-ray kernel arithmetic does not depend on the block partition (phase 2 has
no cross-warp reduction; kernel.cu:117-239), so no approximation is introduced.
"""

import argparse, json, os, sys, time
import zipfile
import zlib
from pathlib import Path

if __name__ == '__main__':
    sys.dont_write_bytecode = True
from sosfuse2.foundation import paths, content_identity
from sosfuse2.backends import vram_trace
import numpy as np

UNITS = 'W m^-2 sr^-1 nm^-1'


def axes(reduced=False):
    wl = np.array([550.0]) if reduced else 360 + (np.arange(15) + 0.5) * 470 / 15
    hs = np.array([-3.0, -6.0]) if reduced else np.arange(6.0, -19.0, -1.0)
    nr, nc = (4, 6) if reduced else (100, 180)
    el = 90 - (np.arange(nr) + 0.5) * 90 / nr
    az = (np.arange(nc) + 0.5) * 180 / nc
    return wl, hs, el, az


def _capture(s, h, ee, aa, dummy):
    """Run the accepted query once with a recording transport; returns raw packs."""
    raws = []
    kernel = s.kernel

    def record(raw, solver, field, first=False):
        raws.append(raw)
        return np.zeros((len(raw['d']), solver.W, 2, 4), 'f8')

    kernel.transport = record
    try:
        s.query(float(h), ee, aa, field=dummy, first=True)
    finally:
        del kernel.transport
    return raws


def _replay(s, h, ee, aa, dummy, raws, values, first):
    """Run the accepted query again, returning precomputed per-chunk values."""
    kernel = s.kernel
    geo = s.geo
    index = iter(range(len(raws)))
    pack = geo.pack

    def token(p, d, n):
        i = next(index)
        if len(d) != len(raws[i]['d']) or not np.array_equal(d, raws[i]['d']):
            raise RuntimeError('E_REPLAY_DRIFT: ray order changed')
        return i

    def fetch(i, solver, field, first_flag=False):
        if bool(first_flag) != bool(first):
            raise RuntimeError('E_REPLAY_FLAG')
        value = values[i]
        out = np.zeros((len(value), solver.W, 2, 4), 'f8')
        out[:, :, 0] = value
        return out

    geo.pack = token
    kernel.transport = fetch
    try:
        result = s.query(float(h), ee, aa, field=dummy, first=first)
    finally:
        geo.pack = pack
        del kernel.transport
    if next(index, None) is not None:
        raise RuntimeError('E_REPLAY_COUNT')
    return result


def _groups(raws, group):
    """Block partition: consecutive `group`-ray slices of every captured pack."""
    from sosfuse2.solve.ray_geometry import subset

    groups = []
    sizes = []
    for g in raws:
        n = len(g['d'])
        sizes.append(n)
        for k in range(0, n, group):
            groups.append(subset(g, slice(k, min(k + group, n))))
    return groups, sizes


def _bundle(s, raws, group):
    """A5: one Bundle per captured batch, shared by the first and multiple launches
    (the ray set/order is identical; only the source field and FIRST differ)."""
    from sosfuse2.solve.ray_geometry import Bundle
    from sosfuse2.foundation.timers import stage

    if s.kernel.xp is np:
        return None
    with stage('table_bundle'):
        groups, sizes = _groups(raws, group)
        bundle = Bundle(groups, s.kernel.xp)
        bundle.sizes = sizes
    return bundle


def _evaluate(s, raws, field, first, group, bundle=None):
    """Per-chunk phase-2 values (n,W,4) float32 for every captured raw pack."""
    from sosfuse2.backends import numpy_reference
    from sosfuse2.foundation.timers import stage

    kernel = s.kernel
    xp = kernel.xp
    if xp is np:
        # NumPy has no block partition: `group` cannot change these values.
        packed = np.ascontiguousarray(field, 'f4')
        with stage('table_launch'):
            return [
                numpy_reference.transport(g, kernel.setup(s, g['nq']), packed, first, statistics=kernel.metrics) for g in raws
            ]
    if bundle is None:
        bundle = _bundle(s, raws, group)
    sizes = bundle.sizes
    with stage('table_launch', device=True):
        _, _, query = kernel.launch(bundle, s, field, first, 2)
    with stage('table_d2h', device=True):
        value = kernel.host(query)
    if value.shape != (sum(sizes), s.W, 4):
        raise RuntimeError('E_QUERY_SHAPE')
    return np.split(value, np.cumsum(sizes)[:-1])


def make_table(s, source, hs, el, az, group=32, batch=25, log=print):
    """Return (table[len(hs),nr,nc,W] float32, per-batch seconds).

    export_first_order keeps a separate host array in s.first_order_table before addition.
    This observed first-order term includes reflection of direct sunlight by the ground.
    """
    from sosfuse2.foundation.timers import stage

    if batch < 1:
        raise ValueError('E_TABLE_BATCH: batch must be positive')
    kernel = s.kernel
    xp = kernel.xp
    source = np.ascontiguousarray(source, 'f4')
    if source.shape != (s.geo.ns, 10 + s.kernel.K * (2 if getattr(s, "strato", None) is not None else 1), s.W):
        raise ValueError(f'E_SOURCE_SHAPE: {source.shape}')
    with vram_trace.item('query.source'):
        field = source if xp is np else xp.asarray(source)
    dummy = np.empty(0, 'f4')
    ee, aa = np.meshgrid(el, az, indexing='ij')
    table = np.empty((len(hs), len(el), len(az), s.W), 'f4')
    first_table = first_order_buffer(s, table.shape)
    seconds = []
    for b0 in range(0, len(hs), batch):
        t = time.monotonic()
        part = list(range(b0, min(b0 + batch, len(hs))))
        with stage('table_capture'):
            captured = [_capture(s, hs[j], ee, aa, dummy) for j in part]
        flat = [g for raws in captured for g in raws]
        results = {}
        bundle = _bundle(s, flat, group)
        for first in (True, False):
            results[first] = _evaluate(s, flat, field, first, group, bundle)
        kernel.sync()
        del bundle
        offset = 0
        for j, raws in zip(part, captured):
            with stage('table_replay'):
                one = _replay(s, hs[j], ee, aa, dummy, raws, results[True][offset : offset + len(raws)], True)
                multiple = _replay(s, hs[j], ee, aa, dummy, raws, results[False][offset : offset + len(raws)], False)
                offset += len(raws)
                spec = (one + multiple)[..., 0].astype('f4')
                if not np.isfinite(spec).all() or (spec < 0).any():
                    raise RuntimeError('E_RADIANCE_INVALID: nonfinite/negative')
                table[j] = spec
                if first_table is not None:
                    first_table[j] = one[..., 0]
        seconds.append(time.monotonic() - t)
        log(
            f'TABLE_FAST sun={hs[part[0]]:+g}..{hs[part[-1]]:+g} rays={sum(len(g["d"]) for g in flat)} seconds={seconds[-1]:.2f}'
        )
    return table, seconds


def first_order_buffer(s, shape):
    """Allocate only host storage for the optional observed term, including direct-ground reflection."""
    if not getattr(s, 'export_first_order', False):
        return None
    s.first_order_table = np.empty(shape, 'f4')
    return s.first_order_table


def save_first_order(out, parts, wl, hs, el, az, *, filename='first_order.npz'):
    """Save the observed first-order term, including direct-ground reflection, with the table axes."""
    if not parts:
        raise RuntimeError('E_FIRST_ORDER_MISSING: no first-order arrays collected')
    table = np.concatenate(parts, axis=-1)
    if table.shape != (len(hs), len(el), len(az), len(wl)) or table.dtype != np.dtype('f4') or not np.isfinite(table).all():
        raise RuntimeError('E_FIRST_ORDER_ARRAY: invalid shape, dtype or values')
    save_table(Path(out) / filename, table, wl, hs, el, az)
    return dict(first_order=dict(file=filename, shape=list(table.shape), dtype='float32', includes_ground_reflection=True))


def exports_first_order(meta):
    settings = meta.get('settings')
    if settings is None:
        return False
    if not isinstance(settings, dict):
        raise ValueError('E_FIRST_ORDER_SETTINGS: expected a settings object or null')
    return bool(settings.get('export_first_order'))


def first_order_complete(out, meta=None):
    """Check the stored first-order array, axes and units before reusing an export.

    Without readable metadata the export setting is unknown and the table/pack rule of the caller decides.
    """
    if meta is None:
        try:
            meta = json.loads((Path(out) / 'meta.json').read_text())
        except (OSError, ValueError):
            return True
    if not isinstance(meta, dict):
        return False
    try:
        exported = exports_first_order(meta)
    except ValueError:
        return False
    if not exported:
        return True
    first = meta.get('first_order')
    if not isinstance(first, dict) or first.get('file') != 'first_order.npz':
        return False
    try:
        with (np.load(Path(out) / first['file'], allow_pickle=False) as saved,
              np.load(Path(out) / 'table.npz', allow_pickle=False) as total):
            radiance = saved['radiance']
            shape = []
            for key in ('solar_elevation_deg', 'view_elevation_deg', 'relative_azimuth_deg', 'wavelength_nm'):
                axis = saved[key]
                if axis.ndim != 1 or not np.isfinite(axis).all() or not np.array_equal(axis, total[key]):
                    return False
                shape.append(len(axis))
            return (radiance.shape == tuple(shape)
                    and list(radiance.shape) == first.get('shape') and radiance.dtype == np.dtype('f4')
                    and np.isfinite(radiance).all() and saved['units'].item() == UNITS)
    except (OSError, ValueError, TypeError, RuntimeError, EOFError, KeyError, zipfile.BadZipFile, zlib.error):
        return False


def table_meta_path(path):
    path = Path(path)
    return path.with_name('meta.json' if path.name == 'table.npz' else path.name + '.meta.json')


def save_table(path, table, wl, hs, el, az):
    from sosfuse2.foundation.timers import stage

    with stage('table_save'):
        _save_table(path, table, wl, hs, el, az)


def _save_table(path, table, wl, hs, el, az):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(
            f,
            radiance=table,
            wavelength_nm=wl.astype('f4'),
            solar_elevation_deg=hs.astype('f4'),
            view_elevation_deg=el.astype('f4'),
            relative_azimuth_deg=az.astype('f4'),
            units=np.array(UNITS),
        )
    os.replace(tmp, path)


def check_source_density(contract, state):
    """The density configuration a source records in its contract state (canonical_state) must be the reader's."""
    from ..atmosphere.components import canonical_state

    def brief(d):
        keys = ('density_profile', 'ozone_profile', 'rayleigh_column', 'profile_step_km')
        return json.dumps(dict({k: d[k] for k in keys if k in d}, external_profile='external_profile' in d), sort_keys=True)

    saved = contract.get('state') if isinstance(contract, dict) else None
    density = saved.get('density') if isinstance(saved, dict) else None
    if not isinstance(density, dict):
        raise ValueError('E_SOURCE_DENSITY: the source records no density configuration')
    want = canonical_state(state)['density']
    if density != want:
        raise ValueError(f'E_SOURCE_DENSITY: source {brief(density)} vs requested {brief(want)}')


def check_source_tol(contract, requested):
    from ..jobs.run_state import effective_tol

    if not isinstance(contract, dict):
        raise ValueError('E_SOURCE_TOL: source contract must be an object')
    saved = contract.get('anderson_tol')
    cfg = contract.get('settings')
    if not isinstance(cfg, dict) or not isinstance(requested, dict):
        raise ValueError('E_SOURCE_TOL: source or requested settings missing')
    if saved != effective_tol(cfg) or saved != effective_tol(requested):
        raise ValueError('E_SOURCE_TOL: source tolerance missing or different')
    return saved


def load_source(path, s, wl, *, settings=None, backend=None, full_settings=False):
    with np.load(path) as f:
        data = {k: f[k] for k in f.files}
    # A source is re-queried by this solver: its physics must be this code's (a source without the field is legacy, so
    # E_PHYSICS_MIXED; components.check_current).
    from ..atmosphere.components import check_current, check_optics, optics_state

    try:
        physical = json.loads(str(data['physics'])) if 'physics' in data else None
    except ValueError as exc:
        raise ValueError('E_PHYSICS_CONTRACT: source physics unreadable') from exc
    check_current(physical, what='source')
    if 'source_contract' not in data:
        raise ValueError('E_SOURCE_CONTRACT: source contract missing')
    try:
        contract = json.loads(str(data['source_contract']))
    except (TypeError, ValueError) as exc:
        raise ValueError('E_SOURCE_CONTRACT: source contract unreadable') from exc
    if not (isinstance(contract, dict) and isinstance(contract.get('state'), dict)):
        raise ValueError('E_SOURCE_CONTRACT: source contract has no state')
    check_source_density(contract, s.fuse_state)
    try:
        amounts = json.loads(str(data['optics_state'])) if 'optics_state' in data else None
    except ValueError as exc:
        raise ValueError('E_OPTICS_STATE_MIXED: source optics_state unreadable') from exc
    check_optics(amounts, optics_state(s.fuse_state))
    source_wl = np.asarray(data['wavelength_nm'])
    wl = np.asarray(wl)
    if source_wl.shape != wl.shape or not np.allclose(source_wl, wl, rtol=0, atol=1e-9):
        raise ValueError('E_SOURCE_WAVELENGTH')
    if not (np.array_equal(data['h'], s.geo.h) and np.array_equal(data['chi'], s.geo.chi)):
        raise ValueError('E_SOURCE_GRID')
    from ..atmosphere.components import canonical_state
    from ..jobs.run_state import kernel_settings

    try:
        contract = json.loads(str(data['source_contract']))
        source_settings = contract['settings']
        if not isinstance(source_settings, dict) or not isinstance(settings, dict):
            raise ValueError('settings missing')
        if content_identity.state_identity(contract['state']) != canonical_state(s.fuse_state):
            raise ValueError('state differs')
        if full_settings:
            from ..atmosphere.spectral import same_settings

            if not same_settings(source_settings, settings):
                raise ValueError('solver settings differ')
        else:
            keys = ('nr', 'nchi', 'nmu', 'nphi', 'path_n', 'L', 'interpolation', 'observer_km')
            if any(
                source_settings.get(k) != settings.get(k) or k not in source_settings or k not in settings for k in keys
            ):
                raise ValueError('solver settings differ')
            if source_settings.get('spectral') is not None:
                raise ValueError('spectral settings unsupported')
            if source_settings.get('kernel') != kernel_settings():
                raise ValueError('kernel settings differ')
        if contract['backend'] != backend or contract['polarization_model'] != 'depolarizing':
            raise ValueError('backend or polarization differs')
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('E_SOURCE_CONTRACT: ' + str(exc)) from exc
    check_source_tol(contract, settings)
    return data['source']


def restored_settings(path, state, reduced, backend):
    """Require a matching recorded solve tolerance before restoring settings or selecting the legacy grid."""
    from types import SimpleNamespace
    from ..atmosphere.components import canonical_state
    from ..jobs.run_state import effective_tol, resolve_settings, settings_for

    with np.load(path) as data:
        wl = np.asarray(data['wavelength_nm'])
        if 'source_contract' not in data.files:
            raise ValueError('E_SOURCE_CONTRACT: source contract missing')
        try:
            contract = json.loads(str(data['source_contract']))
            cfg = contract['settings']
            if not isinstance(cfg, dict):
                raise ValueError('settings missing')
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('E_SOURCE_CONTRACT: ' + str(exc)) from exc
    check_source_density(contract, state)
    check_source_tol(contract, cfg)
    legacy_wl = axes(reduced)[0]
    recorded = np.asarray(cfg.get('wavelengths_nm', legacy_wl))
    if (
        cfg.get('spectral') is None
        and not any(k in cfg for k in ('top_km', 'export_first_order'))
        and effective_tol(cfg) == effective_tol({})
        and cfg.get('L') == 32
        and cfg.get('wavelengths', len(legacy_wl)) == len(legacy_wl)
        and wl.shape == recorded.shape == legacy_wl.shape
        and np.allclose(wl, recorded, rtol=0, atol=1e-9)
        and np.allclose(wl, legacy_wl, rtol=0, atol=1e-9)
    ):
        return None
    try:
        if content_identity.state_identity(contract['state']) != canonical_state(state):
            raise ValueError('state differs')
        if contract['backend'] != backend or contract['polarization_model'] != 'depolarizing':
            raise ValueError('backend or polarization differs')
        raw = {k: v for k, v in cfg.items() if k not in ('spectral', 'kernel')}
        block = cfg.get('spectral')
        if block is None:
            raw['spectral_mode'] = 'legacy'
        else:
            raw.update(spectral_mode=block['mode'], bin_edges_nm=block['bin_edges_nm'])
            if block.get('ozone_temperature') == 'profile_mean':
                raw.update(ozone_temperature='profile_mean', ozone_profile=block['ozone_profile']['name'])
            else:
                raw['ozone_temperature_K'] = {'ozone_213K': 213, 'ozone_233K': 233, 'ozone_253K': 253}[
                    block['ozone_column']
                ]

        class SettingsFile:
            def read_text(self):
                return json.dumps(raw, allow_nan=False)

        expected = resolve_settings(settings_for(SimpleNamespace(reduced=reduced, settings_json=SettingsFile())), state)
        from ..atmosphere.spectral import same_settings

        if not same_settings(cfg, expected):
            raise ValueError('solver settings differ from current code')
        expected_wl = np.asarray(cfg['wavelengths_nm'])
        if wl.shape != expected_wl.shape or not np.allclose(wl, expected_wl, rtol=0, atol=1e-9):
            raise ValueError('E_SOURCE_WAVELENGTH')
    except (KeyError, TypeError, ValueError) as exc:
        if str(exc) == 'E_SOURCE_WAVELENGTH':
            raise
        raise ValueError('E_SOURCE_CONTRACT: ' + str(exc)) from exc
    return cfg


def restored_axes(cfg):
    wl = np.asarray(cfg['wavelengths_nm'])
    hs = np.asarray(cfg['solar_elevation_deg'])
    nr, nc = cfg['view_shape']
    return wl, hs, 90 - (np.arange(nr) + 0.5) * 90 / nr, (np.arange(nc) + 0.5) * 180 / nc


def restored_table(path, state, cfg, prepare, backend, group, batch, log, *, first_order_out=None, first_order_name='first_order.npz'):
    """Replay the production source in the same FW=15 chunks as run_extra()."""
    from ..solve import ray_geometry as gm
    from ..jobs.run_state import create_configured

    if cfg.get('export_first_order') and first_order_out is None:
        raise ValueError('E_FIRST_ORDER_MISSING: restored output requires first_order_out')

    if backend == 'numpy':

        class NumpyScalarKernel(gm.Kernel):
            def host(self, array):
                return np.asarray(array)

        gm.Kernel = NumpyScalarKernel
    wl, hs, el, az = restored_axes(cfg)
    chunks = [wl] if len(wl) in (1, 15) else [wl[i : i + 15] for i in range(0, len(wl), 15)]
    pad = lambda c: c if len(c) == 15 or (len(c) == 1 and len(chunks) == 1) else np.r_[c, np.repeat(c[-1], 15 - len(c))]
    block_size = (cfg.get('kernel') or {}).get('block_size', 128)
    t = time.monotonic()
    s, scope = create_configured(prepare / 'prepare_0', cfg, pad(chunks[0]), backend, state, block_size)
    create_seconds = time.monotonic() - t
    full = load_source(path, s, wl, settings=cfg, backend=backend, full_settings=True)
    expected = (s.geo.ns, 10 + s.kernel.K * (2 if getattr(s, 'strato', None) is not None else 1), len(wl))
    if full.shape != expected:
        raise ValueError(f'E_SOURCE_SHAPE: {full.shape}')
    tables = []
    first_tables = []
    seconds = []
    start = 0
    table_seconds = 0.0
    from ..atmosphere.density import chunk_metadata

    table_density = None
    for index, chunk in enumerate(chunks):
        count = len(chunk)
        if index:
            t = time.monotonic()
            s, scope = create_configured(prepare / f'prepare_{index}', cfg, pad(chunk), backend, state, block_size)
            create_seconds += time.monotonic() - t
        table_density = chunk_metadata(s, chunk, table_density)
        field = full[..., start : start + count]
        start += count
        if count < s.W:
            field = np.concatenate([field, np.repeat(field[..., -1:], s.W - count, -1)], -1)
        s.ms = np.zeros((*field.shape[:2], 2 * s.W), 'f4')
        s.ms[..., ::2] = field
        t = time.monotonic()
        part, times = make_table(s, field, hs, el, az, group, batch, log)
        table_seconds += time.monotonic() - t
        tables.append(part[..., :count])
        if cfg.get('export_first_order'):
            first_tables.append(s.first_order_table[..., :count])
        seconds.extend(times)
        if len(chunks) > 1:
            s.kernel.constants.clear()
            s.kernel.device_constants.clear()
    from ..atmosphere.stratosphere import computed_metadata

    first_meta = save_first_order(first_order_out, first_tables, wl, hs, el, az, filename=first_order_name) if cfg.get('export_first_order') else {}

    return (
        np.concatenate(tables, -1),
        seconds,
        create_seconds,
        table_seconds,
        dict(computed_metadata(state, getattr(s, 'strato', None)), density=table_density, **first_meta),
    )


def build_solver(state, prepare, reduced, backend):
    """Same constructor call as run_state (optics, ground pool, sun columns)."""
    from sosfuse2.solve import ray_geometry as geometry_module

    if backend == 'numpy':

        class NumpyScalarKernel(geometry_module.Kernel):
            def host(self, array):
                return np.asarray(array)

        geometry_module.Kernel = NumpyScalarKernel
    wl, _, _, _ = axes(reduced)
    return geometry_module.create(prepare, 'selftest' if reduced else 'full', wl, backend, 128, state)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--id', required=True)
    p.add_argument('--states', type=Path, default=paths.ASSETS / 'states_v2.json')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--prepare', type=Path, required=True, help='solver scratch folder (sun columns cache)')
    p.add_argument('--out', type=Path, required=True, help='table.npz path')
    p.add_argument('--backend', choices=['cupy', 'numpy'], default='cupy')
    p.add_argument('--reduced', action='store_true')
    p.add_argument(
        '--group',
        type=int,
        default=32,
        help='rays per CUDA block',
    )
    p.add_argument('--batch', type=int, default=25)
    a = p.parse_args()
    if a.batch < 1:
        p.error('E_TABLE_BATCH: batch must be positive')
    from ..jobs.run_state import read_states

    state = next((s for s in read_states(a.states, True) if s['id'] == a.id), None)
    if state is None:
        p.error(f'E_STATE_ID: no state {a.id!r} in {a.states}')
    settings = restored_settings(a.source, state, a.reduced, a.backend)
    first_order_name = 'first_order.npz'
    meta_path = table_meta_path(a.out)
    if a.out.name != 'table.npz':
        first_order_name = a.out.name + '.first_order.npz'
    names = (a.out.name.casefold(), a.out.resolve().name.casefold())
    if (any(n in ('first_order.npz', 'meta.json') or n.endswith(('.first_order.npz', '.meta.json')) for n in names)
            or a.out.resolve() in (a.out.with_name(first_order_name).resolve(), meta_path.resolve())):
        p.error('E_FIRST_ORDER_OUTPUT_CONFLICT: choose a different --out')
    optical_meta = None
    if settings is None:
        t = time.monotonic()
        s, scope, settings = build_solver(state, a.prepare, a.reduced, a.backend)
        create_seconds = time.monotonic() - t
        wl, hs, el, az = axes(a.reduced)
        source = load_source(a.source, s, wl, settings=settings, backend=a.backend)
        t = time.monotonic()
        table, seconds = make_table(s, source, hs, el, az, a.group, a.batch, lambda m: print(m, flush=True))
        total = time.monotonic() - t
        from ..atmosphere.stratosphere import computed_metadata

        from ..atmosphere.density import metadata as density_metadata

        optical_meta = dict(computed_metadata(state, getattr(s, 'strato', None)), density=density_metadata(s))
    else:
        wl, hs, el, az = restored_axes(settings)
        table, seconds, create_seconds, total, optical_meta = restored_table(
            a.source, state, settings, a.prepare, a.backend, a.group, a.batch, lambda m: print(m, flush=True),
            first_order_out=a.out.parent, first_order_name=first_order_name,
        )
    save_table(a.out, table, wl, hs, el, az)
    from ..atmosphere.components import physics, optics_state
    from ..jobs.run_state import CONTRACT_VERSION, effective_tol

    meta = dict(
        complete=True,
        contract=CONTRACT_VERSION,
        physics=physics(),
        optics_state=optics_state(state),
        id=a.id,
        settings=settings,
        anderson_tol=effective_tol(settings),
        backend=a.backend,
        shape=list(table.shape),
        dtype=str(table.dtype),
    )
    if optical_meta is not None:
        meta.update(optical_meta, unsupported_parameters=[])
    tmp = meta_path.with_name(meta_path.name + '.tmp')
    tmp.write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, meta_path)
    print(
        f'TABLE_FAST_DONE id={a.id} create_seconds={create_seconds:.2f} table_seconds={total:.2f} shape={list(table.shape)}',
        flush=True,
    )


if __name__ == '__main__':
    main()
