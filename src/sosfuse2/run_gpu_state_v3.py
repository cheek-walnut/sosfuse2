#!/usr/bin/env python3
"""v2 numerical path with v3 identities, cooperative cache leases and phase events."""

import argparse, contextlib, copy, fcntl, hashlib, json, os, re, shutil
import subprocess, sys, tarfile, threading, time
from pathlib import Path

sys.dont_write_bytecode = True
from sosfuse2 import paths

# Solar column table resolution (sos_sky.SunGeometry scale). 4 = production (measured:
# scale 1 gives -2.38 % first order at -10 deg AOD .6, scale 4 +0.20 %); 1 = the accepted engine table.
SUN_SCALE = 4
# Production kernel SH order. 16 = production with SH_COMBINE in configs/C3.json (measured:
# 5060 Ti V14 L16+SH_COMBINE+nphi16+sun4 vs L32+nphi32+sun1 3.3-5.2x faster, order 2 vs direct L8 equal or 0.03 %p
# better, full radiance within 0.40 %). settings_json may still choose 24 or 32.
SH_L = 16
# Production Anderson fixed-point tolerance (observer and source relative residual). .001 = production (measured:
# .005 left up to 0.75 % iteration error at -9 deg, .001 -> .0002
# changes <= 0.16 %); was .005. The persistent service (pipeline_execution.py) solves with the same constant.
ANDERSON_TOLERANCE = 0.001


def atomic_json(path, data):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def phase(identifier, name, **data):
    print(
        json.dumps(
            dict(
                event='PHASE',
                id=identifier,
                phase=name,
                utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                timestamp_ns=time.time_ns(),
                monotonic_ns=time.monotonic_ns(),
                pid=os.getpid(),
                **data,
            ),
            allow_nan=False,
        ),
        flush=True,
    )


def attach_lock_name(state):
    """Lock group of a state's STRATO_ATTACH_CACHE writer, from the state dict alone (before any solver).
    stratosphere._geometry_key hashes grids/geo.cfg (settings cfg), sun columns (cfg, scale height,
    density), the profile [center,sigma,norm], parameters minus stratosphere.ATTACH_OPTICS_PARAMS and
    the density asset/mask.
    The name uses a subset of those inputs, normalized the same way (profile_config; same filter and
    json.dumps), so equal keys always give equal names: never finer than the key, and one key file is
    never computed by two lock groups at once (_ground_geometry writes a fixed '<key>.tmp.npz').
    Density and cfg are left out (coarser): cfg is shared by a queue and its mapping to the grids would
    have to be copied exactly. None = layer off: attach returns before _ground_geometry (tau 0)."""
    from sosfuse2.stratosphere import ATTACH_OPTICS_PARAMS, tau_of, profile_config

    if tau_of(state) == 0:
        return None
    p = profile_config(state)
    params = {k: v for k, v in state.get('parameters', {}).items() if k not in ATTACH_OPTICS_PARAMS}
    text = json.dumps(dict(profile=[p['center'], p['sigma']], params=params), sort_keys=True, default=str)
    return 'attach_' + hashlib.sha256(text.encode()).hexdigest()[:16]


@contextlib.contextmanager
def attach_cache_event(name, index, state, record=True):
    """Per-state PHASE record: hit, miss, unused (no stratosphere) or disabled.
    stratosphere._ground_geometry writes '<key>.tmp.npz' with a fixed name, so two writers of one key
    must never share a folder. Each call therefore runs against a private overlay folder (hard links
    to the published files): reads hit through the links, a miss is written privately and then
    published with an atomic rename. That alone is safe without any lock.
    The attach_lock_name group lock only avoids duplicate work: it is taken while the group has not
    published its key yet ('<group>.done' absent), so the first state computes and the rest wait and
    then hit. Once published, states of the group run in parallel and never wait.
    The caller appends (solver,scope); hit = this state's own key file was already published.
    record=False keeps the overlay and publish but writes no PHASE record (the geometry-only prefetch, whose
    attach work the worker's own attach_cache record then sees as a hit; pipeline and gate count those records)."""
    root = os.environ.get('STRATO_ATTACH_CACHE', '').strip()
    t = time.monotonic()
    made = []
    if not root:
        yield made
        if record:
            phase(name, 'attach_cache', chunk=index, result='disabled', seconds=time.monotonic() - t)
        return
    import tempfile

    folder = Path(root)
    folder.mkdir(parents=True, exist_ok=True)
    group = attach_lock_name(state)
    published = lambda: {n for n in os.listdir(folder) if re.fullmatch('attach_geometry_[0-9a-f]{24}[.]npz', n)}
    if group is None:
        yield made
        if not record:
            return
        phase(
            name,
            'attach_cache',
            chunk=index,
            result='unused',
            files=len(published()),
            lock_wait_seconds=0.0,
            seconds=time.monotonic() - t,
            lock=None,
        )
        return
    private = Path(tempfile.mkdtemp(prefix='overlay-', dir=folder))
    result = 'unused'
    waited = 0.0
    locked = False
    try:
        with contextlib.ExitStack() as stack:
            if not (folder / (group + '.done')).exists():
                fcntl.flock(stack.enter_context((folder / (group + '.lock')).open('a')), fcntl.LOCK_EX)
                locked = True
                # Published while this state waited: it will hit, so let the other waiters go too.
                if (folder / (group + '.done')).exists():
                    stack.close()
                    locked = False
            waited = time.monotonic() - t
            before = published()
            for n in before:
                os.link(folder / n, private / n)
            os.environ['STRATO_ATTACH_CACHE'] = str(private)
            try:
                yield made
            finally:
                os.environ['STRATO_ATTACH_CACHE'] = root
            for n in os.listdir(private):
                if re.fullmatch('attach_geometry_[0-9a-f]{24}[.]npz', n) and n not in before:
                    os.replace(private / n, folder / n)
            s = made[0][0] if made else None
            optics = getattr(s, 'strato', None)
            if optics is not None:
                from sosfuse2.stratosphere import _geometry_key

                own = f'attach_geometry_{_geometry_key(s, made[0][1], optics["profile"])}.npz'
                result = 'hit' if own in before else 'miss'
                if (folder / own).exists():
                    (folder / (group + '.done')).touch()
    finally:
        shutil.rmtree(private, ignore_errors=True)
    if not record:
        return
    phase(
        name,
        'attach_cache',
        chunk=index,
        result=result,
        files=len(published()),
        lock_wait_seconds=waited,
        seconds=time.monotonic() - t,
        lock=group,
        locked=locked,
    )


@contextlib.contextmanager
def lease_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.v3-leases.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (ValueError, OverflowError):
        return False


def register_geometry(s, scope, root, layers=1):
    # sosfuse2: local helper.
    from sosfuse2.geometry_cpu import host_geometry

    # A stable parent lock protects constructor mkdir, marker registration and deletion.
    with lease_lock(root):
        g = host_geometry(s, scope, root, layers)
        users = g.cache / 'users'
        users.mkdir(exist_ok=True)
        (users / str(os.getpid())).write_text('v3\n')
    return g


def release_geometry(root, cache, success, keep=False):
    with lease_lock(root):
        # Only the exact registered endpoint directory may be removed.
        if (
            cache.parent.resolve() != root.resolve()
            or not re.fullmatch('endpoints_[0-9a-f]{20}', cache.name)
            or cache.is_symlink()
        ):
            raise RuntimeError('E_CACHE_TARGET: refusing unowned cache path')
        if not cache.exists():
            return dict(geometry_cache_removed=False, reason='already_absent')
        users = cache / 'users'
        (users / str(os.getpid())).unlink(missing_ok=True)
        live = []
        for marker in users.iterdir():
            if marker.name.isdecimal() and alive(marker.name):
                live.append(marker.name)
            elif marker.name.isdecimal():
                marker.unlink()
            else:
                live.append(marker.name)  # unknown marker: fail closed
        reason = 'failed_attempt' if not success else 'keep_geometry' if keep else 'active_users' if live else 'removed'
        if reason == 'removed':
            try:
                shutil.rmtree(cache)
            except OSError:
                return dict(geometry_cache_removed=False, reason='E_CACHE_REMOVE_FAILED')
        return dict(geometry_cache_removed=reason == 'removed', reason=reason)


AUTO_FLAGS = ('STRATO', 'TROPO_MIE', 'AEROSOL_2L')


def kernel_settings():
    """Kernel variant from the environment only (the queue exports it and workers and the prefetch helper inherit
    os.environ), so scheduler, prefetch and runner derive the same settings and signature.  None selects default transport with block 128; physical flags are attached from the state."""
    from sosfuse2.kernel_config import KERNEL_FLAGS, resolve_config

    flags = sorted({f for f in os.environ.get('SOSFUSE2_KERNEL_FLAGS', '').replace(' ', '').split(',') if f})
    flags = list(resolve_config(flags).flags)
    allowed = [f for f in KERNEL_FLAGS if f not in AUTO_FLAGS]
    if set(flags) - set(allowed):
        raise ValueError(
            'E_KERNEL_FLAGS: SOSFUSE2_KERNEL_FLAGS from ' + ','.join(allowed) + ' (physical selectors are automatic)'
        )
    text = os.environ.get('SOSFUSE2_BLOCK_SIZE', '').strip()
    if text not in ('', '128', '256'):
        raise ValueError('E_KERNEL_BLOCK_SIZE: SOSFUSE2_BLOCK_SIZE must be 128 or 256')
    block = int(text or 128)
    from sosfuse2.ground_sh import configured_order, GRID_INTERVALS

    ground_order = configured_order()
    if not flags and block == 128 and not ground_order:
        return None
    settings = dict(flags=flags, block_size=block, source='env SOSFUSE2_KERNEL_FLAGS/SOSFUSE2_BLOCK_SIZE')
    if ground_order:
        settings['ground_sh'] = dict(order=ground_order, grid_intervals=GRID_INTERVALS, interpolation='cubic')
    return settings


def settings_for(a):
    import numpy as np

    # V14 production defaults; overrides are checked before any expensive work.
    cfg = dict(
        nr=4 if a.reduced else 44,
        h_refine='none' if a.reduced else 'below13mid_above13thirds',
        chi_step_deg=None if a.reduced else 0.5,
        nchi=10 if a.reduced else 75,
        nmu=1 if a.reduced else 12,
        nphi=2 if a.reduced else 16,
        path_n=4 if a.reduced else 12,
        L=SH_L,
        interpolation='log',
        observer_km=0.1,
        wavelengths=1 if a.reduced else 15,
        wavelengths_nm=[550.0] if a.reduced else (360 + (np.arange(15) + 0.5) * 470 / 15).tolist(),
        solar_elevation_deg=[-3.0, -6.0] if a.reduced else np.arange(6.0, -19.0, -1.0).tolist(),
        view_shape=[4, 6] if a.reduced else [100, 180],
    )
    raw = json.loads(a.settings_json.read_text()) if a.settings_json else {}
    from sosfuse2 import spectral

    # Spectral keys leave raw before the key check; legacy adds nothing, so explicit-legacy settings bytes are unchanged.
    spectral_raw = {k: raw.pop(k) for k in spectral.KEYS if isinstance(raw, dict) and k in raw}
    if not isinstance(raw, dict) or set(raw) - set(cfg):
        raise ValueError('E_SETTINGS_KEYS: unknown settings')
    # Product grid default band BM45; the reduced 550 nm smoke grid stays legacy unless a mode is named.
    if not a.reduced:
        spectral.product_default(spectral_raw, raw)
    cfg.update(raw)
    pairs = {20: 'none', 39: 'mid', 44: 'below13mid_above13thirds'}
    for k in ('nr', 'nchi', 'nmu', 'nphi', 'path_n', 'L', 'wavelengths'):
        if type(cfg[k]) is not int or cfg[k] < 1:
            raise ValueError('E_SETTINGS_VALUE: ' + k)
    if 'nr' in raw and 'h_refine' not in raw:
        cfg['h_refine'] = 'none' if a.reduced else pairs.get(cfg['nr'])
    if 'wavelengths' in raw and 'wavelengths_nm' not in raw:
        n = cfg['wavelengths']
        cfg['wavelengths_nm'] = (360 + (np.arange(n) + 0.5) * 470 / n).tolist()
    if len(cfg['wavelengths_nm']) != cfg['wavelengths']:
        raise ValueError('E_SETTINGS_WAVELENGTHS: inconsistent count')
    wl = np.asarray(cfg['wavelengths_nm'], float)
    if not np.isfinite(wl).all() or np.any(wl <= 0) or np.any(np.diff(wl) <= 0):
        raise ValueError('E_SETTINGS_WAVELENGTHS: positive ordered wavelengths required')
    if cfg['L'] not in (16, 24, 32) or cfg['interpolation'] != 'log':
        raise ValueError('E_SETTINGS_FIXED: L in {16,24,32} and log interpolation required')
    allowed = [(4, 'none')] if a.reduced else list(pairs.items())
    if (cfg['nr'], cfg['h_refine']) not in allowed:
        raise ValueError('E_SETTINGS_GRID: received=%r; allowed=%r' % ((cfg['nr'], cfg['h_refine']), allowed))
    if cfg['chi_step_deg'] not in ((None,) if a.reduced else (0.5, 0.25)):
        raise ValueError('E_SETTINGS_GRID: chi_step_deg must be 0.5 or 0.25')
    if not 0 <= cfg['observer_km'] <= 80:
        raise ValueError('E_SETTINGS_OBSERVER')
    if (
        not isinstance(cfg['view_shape'], list)
        or len(cfg['view_shape']) != 2
        or any(type(x) is not int or x < 1 for x in cfg['view_shape'])
    ):
        raise ValueError('E_SETTINGS_VIEW')
    hs = np.asarray(cfg['solar_elevation_deg'], float)
    if hs.ndim != 1 or not np.isfinite(hs).all() or len(set(hs)) != len(hs) or -3 not in hs or -6 not in hs:
        raise ValueError('E_SETTINGS_SOLAR: unique finite elevations including -3,-6 required')
    # After the settings_json key check, so a settings file can never set or override it.
    kernel = kernel_settings()
    if kernel is not None:
        cfg['kernel'] = kernel
    block = spectral.block(spectral_raw, wl, 'wavelengths_nm' in raw)
    if block is not None:
        cfg['spectral'] = block
    json.dumps(cfg, allow_nan=False)
    return cfg


def resolve_settings(cfg, state):
    """Per-state settings (A′ resolver): a profile_mean spectral block takes
    its q identity from this (effective) state; the shared cfg is never changed (a new copy is returned). Without a
    profile_mean block the same cfg object comes back. Callers: run, run_extra, geometry_only, plan_queue.probe, gate worker."""
    from sosfuse2 import spectral

    block = cfg.get('spectral')
    resolved = spectral.resolve(block, state)
    if resolved is block:
        return cfg
    out = copy.deepcopy(cfg)
    out['spectral'] = resolved
    return out


def grid_metadata(cfg, h=None):
    """Sign the height bytes without constructing solar tables or a solver."""
    import numpy as np

    if h is None:
        from sosfuse2.solver_columns import grids

        h = grids(1)[0] if cfg['nr'] != 4 else np.array([0.0, 0.1, 8.0, 80.0])
        if cfg['h_refine'] == 'mid':
            h = np.sort(np.r_[h, (h[:-1] + h[1:]) / 2])
        elif cfg['h_refine'] == 'below13mid_above13thirds':
            h = np.sort(
                np.r_[
                    h,
                    [(u + v) / 2 for u, v in zip(h[:-1], h[1:]) if v <= 13 + 1e-9],
                    [u + (v - u) * f for u, v in zip(h[:-1], h[1:]) if (u >= 13 - 1e-9) for f in (1 / 3, 2 / 3)],
                ]
            )
    return dict(h_refine=cfg['h_refine'], h_sha256=hashlib.sha256(np.ascontiguousarray(h).tobytes()).hexdigest())


def create_configured(output, cfg, wavelengths, backend, state, block_size=128, geometry_cache=None):
    """Configure the accepted engine's create() for the production settings through its keyword hooks (observer, kernel,
    grid, spectrum, sun geometry, deferred ground, aerosol profile). Returns the solver and its scope; the solver is
    built over `wavelengths` (a second Solver in the same scope when more than one)."""
    from sosfuse2 import spectral

    spectral.check_state(
        cfg.get('spectral'), state
    )  # A′: a block resolved for another (or no) state never reaches SPEC
    if cfg.get('spectral') is not None:
        from .components import check_current

        check_current(cfg['spectral'].get('physics'), what='solver spectral block')
    import numpy as np
    from sosfuse2 import geometry as gm
    from sosfuse2.fused import load_engine

    engine = load_engine()
    threads_budget = int(os.environ.get('OMP_NUM_THREADS', str(os.cpu_count() or 1)))
    from sosfuse2.density import profile_step_km, kernel_kwargs

    kw = kernel_kwargs(profile_step_km(state))  # ②b: compiled PROFILE node count follows the state (0.5 km = no option)

    def kernel_factory(threads, H, g):
        return gm.Kernel(threads_budget, H, g, backend, block_size, L=cfg['L'], **kw)

    def grid(mode, h, ch, nmu, nphi, path_n):
        # The settings own the grid: full mode refines the accepted h/chi, every mode takes the quadrature counts from cfg.
        if mode == 'full':
            if cfg['h_refine'] == 'mid':
                h = np.sort(np.r_[h, (h[:-1] + h[1:]) / 2])
            elif cfg['h_refine'] == 'below13mid_above13thirds':
                h = np.sort(
                    np.r_[
                        h,
                        [(u + v) / 2 for u, v in zip(h[:-1], h[1:]) if v <= 13 + 1e-9],
                        [u + (v - u) * f for u, v in zip(h[:-1], h[1:]) if (u >= 13 - 1e-9) for f in (1 / 3, 2 / 3)],
                    ]
                )
            if cfg['chi_step_deg'] == 0.25:
                degrees = np.round(np.rad2deg(ch), 12)
                ch = np.deg2rad(
                    np.unique(np.r_[degrees[(degrees < 85) | (degrees > 110)], np.arange(85, 110 + 0.125, 0.25)])
                )
        if len(h) != cfg['nr'] or len(ch) != cfg['nchi']:
            raise ValueError('E_SETTINGS_GRID_COUNT: actual grid disagrees with nr/nchi')
        return h, ch, cfg['nmu'], cfg['nphi'], cfg['path_n']

    from sosfuse2 import host_cache

    fast_host = host_cache.enabled()
    sun_geometry = None
    if fast_host or SUN_SCALE != 1:
        # host_cache.sun_geometry keys its folder by height and scale; the accepted constructor keys neither.
        def sun_geometry(old, cache, H):
            return host_cache.sun_geometry(old, SUN_SCALE, cache, H, shared=geometry_cache)

    spec = None
    if cfg.get('spectral') is not None:
        # The engine's only spectral input is this SPEC dict (a fresh table per call; create() completes it in place).
        # Representative values belong to the settings bins; a solver over a subset (geometry-only passes the first
        # wavelength) reads its own rows of the full table.
        if any(float(w) not in cfg['wavelengths_nm'] for w in wavelengths):
            raise ValueError('E_SPECTRAL_CONTRACT: solver wavelengths outside the settings')
        spec = spectral.spec_table(cfg['spectral'], cfg['wavelengths_nm'], state)
    from sosfuse2 import tropo_profile

    output.mkdir(parents=True, exist_ok=True)
    # prepare_ground=False: the ground is prepared once below (host cache or density.attach), after the wavelength Solver.
    initial, scope, _ = engine.create(
        state,
        output,
        'selftest' if cfg['nr'] == 4 else 'full',
        threads_budget,
        float(wavelengths[0]),
        observer_km=cfg['observer_km'],
        kernel_factory=kernel_factory,
        grid=grid,
        spec=spec,
        sun_geometry=sun_geometry,
        prepare_ground=not fast_host,
        aerosol=tropo_profile.engine_hooks(state),
    )
    s = initial if len(wavelengths) == 1 else scope.Solver(initial.geo, initial.kernel, wavelengths, 1, 0.03)
    s.levels = s.levels[: s.kernel.K]
    s.af = s.af[: s.kernel.K]
    s.active[1::2] = 0
    s.fuse_state = state
    tropo_profile.attach(s)
    if fast_host:
        scope.prepare_ground = True
        from sosfuse2.density import build

        if build(state) is None:
            s.prepare_ground()
    if geometry_cache is not None:
        s.host_cache_root = Path(geometry_cache)
    from sosfuse2.density import attach as attach_density
    from sosfuse2.stratosphere import tau_of

    # attach() immediately replaces all first-ground arrays when the layer is on.
    attach_density(s, scope, defer_ground=tau_of(state) > 0)
    from sosfuse2.stratosphere import attach

    attach(s, scope)
    from sosfuse2.tropo_mie import attach as attach_tropo

    attach_tropo(s)
    s.geo.pack = lambda p, d, n: gm.pack(p, d, n, s, scope, ground=False)
    return s, scope


def effective_state(state, a):
    state = copy.deepcopy(state)
    if getattr(a, 'stratosphere', None) == 'on':
        state.setdefault('stratosphere', {})['status'] = 'enabled'
    from sosfuse2.density import with_options

    return with_options(state, a)


def verify_pack(path, signature):
    # Read every archived member through zstd; verify both CRC/content and the identity.
    # tarfile's stream mode stops at the end-of-archive marker and leaves the record padding (10240 bytes measured) unread;
    # closing the pipe then can kill a zstd still writing it (SIGPIPE, nonzero exit) when the pipe buffer is small, which
    # failed a valid pack in an eda gate run (2026-11-03, FX-N; the same file decodes with rc 0). So the rest of the stream
    # is drained before close, and a real decode failure reports zstd's exit code and stderr.
    import tempfile

    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(['zstd', '-q', '-dc', str(path)], stdout=subprocess.PIPE, stderr=err)
        failed = False
        try:
            with tarfile.open(fileobj=proc.stdout, mode='r|') as archive:
                names = []
                meta = None
                for member in archive:
                    if not member.isfile():
                        raise RuntimeError('E_PACK_VERIFY: unexpected member')
                    names.append(member.name)
                    f = archive.extractfile(member)
                    if member.name == 'meta.json':
                        meta = json.load(f)
                    else:
                        while f.read(1024 * 1024):
                            pass
                if (
                    sorted(names) != sorted(['table.npz', 'meta.json', 'preview_m03.png', 'preview_m06.png'])
                    or not isinstance(meta, dict)
                    or meta.get('complete') is not True
                    or meta.get('signature') != signature
                ):
                    raise RuntimeError('E_PACK_VERIFY: members or identity')
        except BaseException:
            failed = True
            raise
        finally:
            try:
                while proc.stdout.read(1024 * 1024):
                    pass
            except OSError:
                pass
            proc.stdout.close()
            rc = proc.wait()
            if rc != 0 and not failed:  # an earlier E_PACK_VERIFY keeps its own reason
                err.seek(0)
                detail = err.read().decode(errors='replace').strip()[-300:]
                raise RuntimeError(f'E_PACK_VERIFY: zstd decode failed (exit {rc}{": " + detail if detail else ""})')


def enable_spectral_previews(module):
    import numpy as np

    original = module.previews

    def previews(out):
        from .spectral import check_meta_contract

        meta = json.loads((out / 'meta.json').read_text())
        check_meta_contract(meta)
        with np.load(out / 'table.npz') as f:
            table = f['radiance']
            wl = f['wavelength_nm']
            hs = f['solar_elevation_deg']
        if len(wl) in (1, 15):
            return original(out)
        weights = module.preview_weights(
            wl, module.preview_band_widths(meta, wl), np.load(paths.ASSETS / 'rgb_weights.npy')
        )
        rgb = table @ weights
        funcs = module.render_functions()
        for h in (-3, -6):
            folded = rgb[np.flatnonzero(hs == h)[0]]
            pano = np.concatenate([folded[:, ::-1], folded], axis=1)
            projected = funcs['project'](pano, funcs['projection'](960, 540, 100, 0))
            k = funcs['central'](projected)[1] if (projected @ funcs['Y']).max() > 0 else 1.0
            funcs['save_png'](out / f'preview_m{abs(h):02d}.png', funcs['tone'](projected, k))

    module.previews = previews


def package_hashes():
    """Provenance v4 code hashes (docs/PROVENANCE_V3.md): package-relative keys ('sosfuse2/...') over the
    production modules (paths.PRODUCTION_MODULES, a fixed list), the CUDA kernel, the engine module (v3 never
    hashed them), the host helpers, the native CPU kernel source and the data assets the engine reads. Sorted keys; every
    entry is a fixed name and a missing file is E_PACKAGE_FILE_MISSING, not a gap. package_version/package_commit are
    recorded beside the `contract` field but are NOT signature inputs: the same bytes sign the same in-tree and
    installed."""
    P = paths.PKG
    files = paths.production_files()
    files += [
        P / 'kernel.cu',
        *[P / 'accepted' / n for n in ACCEPTED_SOURCES],
        *[P / 'host' / n for n in HOST_MODULES],
        P / 'native/sh_cpu.cpp',
    ]
    files += [P / 'assets' / n for n in DATA_ASSETS]
    missing = [paths.rel(p) for p in files if not p.is_file()]
    if missing:
        raise RuntimeError(f'E_PACKAGE_FILE_MISSING: provenance inputs absent from the package: {missing}')
    return {paths.rel(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


# Fixed provenance inputs beside the production modules: the engine module, the host helpers and
# the data the engine reads. Adding a file here changes every v4 signature (intended); a listed file that is missing is an error.
ACCEPTED_SOURCES = ('__init__.py', 'engine.py')
HOST_MODULES = ('__init__.py', 'host_cores.py', 'pack_state.py')
DATA_ASSETS = (
    'spectra.json',
    'spectra_fine.json',
    'spectra_fine_temperature.json',
    'lef_cie.json',
    'cmf_1nm.json',
    'states_v1.json',
    'states_v2.json',
    'states_v1_usstd.json',
    'rgb_weights.npy',
    'temperature_usstd76.json',
    'no2_afglus_profile.json',
    'no2_vandaele1998_fine.json',
)


def package_info():
    """Release identity beside the content hashes: version from the installed metadata (None in-tree without install); commit from
    SOSFUSE2_PACKAGE_COMMIT, else the installer's direct_url.json (pip/uv VCS install: vcs_info.commit_id; used only when the
    distribution's sosfuse2/__init__.py is this package's own), else the git checkout whose
    src/sosfuse2 IS this package (never a consumer checkout that merely holds an installed copy). None otherwise, never an error."""
    dist = None
    ver = None
    try:
        from importlib.metadata import distribution

        dist = distribution('sosfuse2')
        ver = dist.version
    except Exception:
        pass
    commit = os.environ.get('SOSFUSE2_PACKAGE_COMMIT') or None
    if commit is None and dist is not None:
        try:
            # Only when the metadata describes THIS copy of the package: an in-tree run (PYTHONPATH=src) next to an installed
            # sosfuse2 must not record the installed copy's commit.
            if Path(dist.locate_file('sosfuse2/__init__.py')).resolve() == (Path(paths.PKG) / '__init__.py').resolve():
                du = dist.read_text('direct_url.json')
                commit = ((json.loads(du).get('vcs_info') or {}).get('commit_id') or None) if du else None
        except Exception:
            commit = None
    if commit is None:
        try:
            top = subprocess.run(
                ['git', 'rev-parse', '--show-toplevel'], cwd=paths.PKG, capture_output=True, text=True, timeout=10
            )
            if (
                top.returncode == 0
                and (Path(top.stdout.strip()) / 'src/sosfuse2').resolve() == Path(paths.PKG).resolve()
            ):
                r = subprocess.run(
                    ['git', 'rev-parse', 'HEAD'], cwd=paths.PKG, capture_output=True, text=True, timeout=10
                )
                commit = r.stdout.strip() or None if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            commit = None
    return dict(package_version=ver, package_commit=commit)


from .components import physics, optics_state, canonical_state, check_optics

CONTRACT_VERSION = (
    'v4'  # v3 tables (sky-sos repository-relative code_hashes) keep their meaning; readers must accept both
)


def signed_contract(state, cfg, backend, reduced, polarization_model, derived=None):
    state = canonical_state(state)
    amounts = optics_state(state)
    from .components import reference_active

    if amounts['o4'] and not reference_active():
        from .minor_absorbers import load

        load('o4')
    hashes = package_hashes()
    from sosfuse2.stratosphere import tau_of, table_path, path_label

    if tau_of(state) > 0:
        path = table_path(state)
        hashes[path_label(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    from sosfuse2 import density

    if density.build(state)['meta'].get('asset_sha256') is not None:
        hashes[density.ASSET_KEY] = hashlib.sha256(density.asset_path().read_bytes()).hexdigest()  # logical key
    from sosfuse2.tropo_mie import config as tropo_config, table_path as tropo_path, path_label as tropo_label

    if tropo_config(state) is not None:
        path = tropo_path(tropo_config(state))
        hashes[tropo_label(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if cfg.get('spectral') is not None:
        from sosfuse2 import spectral
        from sosfuse2.components import check_current  # a legacy block (copied from an old meta) never signs a new run

        check_current(cfg['spectral'].get('physics'), what='spectral settings block')
        spectral.check_asset(cfg['spectral'])
        hashes[cfg['spectral']['asset']] = spectral.asset()['sha']
        spectral.check_state(cfg['spectral'], state)
        hashes.update(spectral.asset_hashes(cfg['spectral']))  # A′ inputs; {} otherwise
    # package_version/package_commit are NOT signature inputs: run() records them in meta beside the `contract` field.
    contract = dict(
        contract=CONTRACT_VERSION,
        physics=physics(),
        optics_state=amounts,
        state=state,
        backend=backend,
        reduced=reduced,
        settings=cfg,
        polarization_model=polarization_model,
        tolerance=ANDERSON_TOLERANCE,
        code_hashes=hashes,
        grid=grid_metadata(cfg),
    )
    # M3: a table derived from a stored source signs its own mode; ordinary jobs keep the old signed input bytes.
    if derived is not None:
        contract['derived'] = derived
    signature = hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return signature, hashes


def parse_elevations(text):
    values = [float(x) for x in str(text).split(',') if x.strip()]
    if not values or len(set(values)) != len(values) or not all(abs(v) < 90 for v in values):
        raise ValueError('E_EXTRA_ELEVATIONS: unique finite |h|<90 list required')
    return values


def extra_contract(cfg, extra, source_out_name):
    """M3 (shared by scheduler and runner): rows = extra elevations + the -3/-6
    anchors (previews, pack format and a bit-level reuse check against the
    source job's table).  cfg stays the SOURCE job's settings unchanged."""
    if not re.fullmatch('[A-Za-z0-9_-]+', str(source_out_name or '')):
        raise ValueError('E_EXTRA_SOURCE: safe source_out_name required')
    extra = parse_elevations(extra) if isinstance(extra, str) else [float(x) for x in extra]
    if set(extra) & set(float(x) for x in cfg['solar_elevation_deg']):
        raise ValueError('E_EXTRA_ELEVATIONS: already a source table row')
    rows = sorted(set(extra) | {-3.0, -6.0}, reverse=True)
    return dict(
        mode='extra_elevations',
        extra_elevations=sorted(extra, reverse=True),
        table_elevations=rows,
        source_out_name=source_out_name,
    )


def read_states(path, include_parent=False):
    doc = json.loads(path.read_text())
    states = list(doc['states'])
    if include_parent and doc.get('parent'):
        parent = json.loads((path.parent / doc['parent']).read_text())['states']
        own = {s['id'] for s in states}
        states.extend(s for s in parent if s['id'] not in own)
    return states


def geometry_only(a):
    # This entry point never selects a CUDA backend or constructs a fused Solver.
    for k in (
        'OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
        'NUMEXPR_NUM_THREADS',
        'BLIS_NUM_THREADS',
    ):
        os.environ[k] = str(a.threads)
    import numpy as np
    from sosfuse2.geometry_cpu import build_cells, cache_complete, column_ready, ensure_column
    from sosfuse2.run_fuse import monitor, peak

    for k in (
        'OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
        'NUMEXPR_NUM_THREADS',
        'BLIS_NUM_THREADS',
    ):
        os.environ[k] = str(a.threads)
    if a.polarization_model != 'depolarizing':
        raise RuntimeError(
            'E_UNSUPPORTED_POL_MODEL: sosfuse2 fused.py/kernel.cu have no preserving aerosol tensor transport'
        )
    if a.lease_pid is not None and (a.lease_pid < 1 or a.lease_pid == os.getpid() or not alive(str(a.lease_pid))):
        raise RuntimeError('E_PREFETCH_LEASE_PID')
    found = [s for s in read_states(a.states, True) if s['id'] == a.id and s.get('runnable')]
    if len(found) != 1:
        raise ValueError('E_STATE_ID: exactly one runnable state required')
    cfg = settings_for(a)
    state = effective_state(found[0], a)
    cfg = resolve_settings(cfg, state)
    name = a.out_name or a.id
    output = a.work_root / 'prefetch' / name
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.geometry-only.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('E_PREFETCH_BUSY')
    ready = output / 'geometry_ready.json'
    ready.unlink(missing_ok=True)
    stop = threading.Event()
    threading.Thread(target=monitor, args=(stop, a.rss_gb * 1e9), daemon=True).start()
    leased = None
    transferred = False
    started = time.monotonic()
    try:
        phase(name, 'geometry_start', geometry_only=True)
        # Geometry columns depend on height/grid/quadrature, not wavelength count. The stratosphere attach runs in
        # the same private overlay as a worker's (attach_cache_event), so a prefetch and a worker of one key never
        # write the same '<key>.tmp.npz' in a shared STRATO_ATTACH_CACHE.
        with attach_cache_event(name, 0, state, record=False) as made:
            made.append(
                create_configured(
                    output / 'prepare',
                    cfg,
                    np.array([cfg['wavelengths_nm'][0]]),
                    'numpy',
                    state,
                    geometry_cache=a.geometry_cache,
                )
            )
        s, scope = made[0]
        leased = register_geometry(s, scope, a.geometry_cache, a.geometry_layers)
        with (a.geometry_cache / (leased.stats['signature'] + '.lock')).open('a') as builder:
            fcntl.flock(builder, fcntl.LOCK_EX)
            common = cache_complete(leased)
            build_cells(leased)
        # M1: also the per-state aerosol column, so the worker takes the cached-column path.
        had_column = common and column_ready(leased)
        column = ensure_column(leased)
        hit = common and had_column
        if 'cupy' in sys.modules:
            raise RuntimeError('E_PREFETCH_CUDA_IMPORTED')
        with lease_lock(a.geometry_cache):
            if a.lease_pid is not None:
                if not alive(str(a.lease_pid)):
                    raise RuntimeError('E_PREFETCH_PARENT_GONE')
                (leased.cache / 'users' / str(a.lease_pid)).write_text('v3-prefetch\n')
            try:
                atomic_json(
                    ready,
                    dict(
                        id=a.id,
                        out_name=name,
                        cache=str(leased.cache),
                        signature=leased.stats['signature'],
                        lease_pid=a.lease_pid,
                        cuda_imported=False,
                        settings=cfg,
                        seconds=time.monotonic() - started,
                        layers=a.geometry_layers,
                        common='hit' if common else 'miss',
                        column=column,
                        column_path=str(leased.column_path(leased.s.kernel.height)) if column else None,
                    ),
                )
            except BaseException:
                if a.lease_pid is not None:
                    (leased.cache / 'users' / str(a.lease_pid)).unlink(missing_ok=True)
                raise
            (leased.cache / 'users' / str(os.getpid())).unlink(missing_ok=True)
            transferred = True
        phase(
            name,
            'geometry_done',
            geometry_only=True,
            cache='hit' if hit else 'miss',
            seconds=time.monotonic() - started,
            signature=leased.stats['signature'],
            layers=a.geometry_layers,
            common='hit' if common else 'miss',
            column=column,
        )
    finally:
        if leased is not None and not transferred:
            release_geometry(a.geometry_cache, leased.cache, False, True)
        stop.set()


def run(a):
    from .components import production_only

    production_only()
    if a.polarization_model == 'preserving':
        raise RuntimeError(
            'E_UNSUPPORTED_POL_MODEL: fused depolarizing HG only; sosfuse2 fused.py/kernel.cu need a preserving tensor transport'
        )
    if not shutil.which('zstd'):
        raise RuntimeError('E_ZSTD_MISSING: install zstd (required by pack and verify)')
    if (
        os.environ.get('SOS_PERSISTENT_ROOT')
        and getattr(a, 'gpu_lock', None)
        and os.environ.get('SOSFUSE2_EXPORT_MOMENT') == '1'
    ):
        # persistent_chunk() never reaches the moment export further down: refuse instead of silently writing no moment.npz
        raise RuntimeError(
            'E_EXPORT_MOMENT_PERSISTENT: SOSFUSE2_EXPORT_MOMENT=1 is not supported by the persistent queue; run without --persistent'
        )
    for k in (
        'OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
        'NUMEXPR_NUM_THREADS',
        'BLIS_NUM_THREADS',
    ):
        os.environ[k] = str(a.threads)
    import numpy as np
    from sosfuse2 import geometry as gm
    from sosfuse2.fused import Solver
    from sosfuse2.geometry_cpu import build_cells, cache_complete, column_ready, ensure_column
    from sosfuse2.host import pack_state as pm
    from sosfuse2.run_fuse import monitor, peak

    for k in (
        'OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
        'NUMEXPR_NUM_THREADS',
        'BLIS_NUM_THREADS',
    ):
        os.environ[k] = str(a.threads)
    if a.backend == 'numpy':

        class ScalarKernel(gm.Kernel):
            def host(self, x):
                return np.asarray(x)

        gm.Kernel = ScalarKernel
    states = read_states(a.states, True)
    found = [s for s in states if s['id'] == a.id and s.get('runnable')]
    if len(found) != 1:
        raise ValueError('E_STATE_ID: exactly one runnable state required')
    state = effective_state(found[0], a)
    cfg = resolve_settings(settings_for(a), state)
    name = a.out_name or a.id
    if a.backend == 'cupy':
        # Refuse a first light the device profile cannot split, before any geometry or CPU preparation (eda C2, 8.6 s).
        from sosfuse2.device_profile import select, smi_device_name, first_light_split_error
        from sosfuse2.fused import default_flags
        from sosfuse2.stratosphere import tau_of

        pname, profile, _ = select(smi_device_name())
        err = first_light_split_error(default_flags(), tau_of(state) > 0, pname, profile)
        if err:
            raise RuntimeError(err)
    signature, hashes = signed_contract(state, cfg, a.backend, a.reduced, a.polarization_model)
    out = a.work_root / 'out' / name
    prepare = a.work_root / 'tmp' / name
    out.mkdir(parents=True, exist_ok=True)
    output_lock = (out / '.worker.lock').open('a')
    try:
        fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('E_OUTPUT_BUSY: out_name already owned by another worker')
    prepare.mkdir(parents=True, exist_ok=True)
    meta_path = out / 'meta.json'
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        check_optics(previous.get('optics_state'), optics_state(state))
        if previous.get('signature') != signature:
            raise RuntimeError('E_OUTPUT_IDENTITY: use a fresh work-root/out_name')
    # Recompute an interrupted attempt rather than treating a loose table as validated success.
    started = time.monotonic()
    stop = threading.Event()
    geometry = None
    leased = None
    released = False
    threading.Thread(target=monitor, args=(stop, a.rss_gb * 1e9), daemon=True).start()
    from sosfuse2.compile_cache import configure

    pipeline_lock = getattr(a, 'gpu_lock', None)
    token = None
    pending_tables = []
    solver = field = None
    persistent = bool(os.environ.get('SOS_PERSISTENT_ROOT'))
    if not pipeline_lock:
        configure(a.work_root.parent / 'nvrtc_cache_shared', a.backend, a.block_size)
    os.environ['CUDA_CACHE_PATH'] = str(prepare / 'cuda_driver_cache')
    wl = np.asarray(cfg['wavelengths_nm'])
    hs = np.asarray(cfg['solar_elevation_deg'])
    nr, nc = cfg['view_shape']
    el = 90 - (np.arange(nr) + 0.5) * 90 / nr
    az = (np.arange(nc) + 0.5) * 180 / nc
    tables = []
    fields = []
    moments = []
    stats_all = []
    timings = []
    source_seconds = query_seconds = 0.0
    ksplit_decision = None
    try:
        phase(name, 'geometry_start')
        geometry_start = time.monotonic()
        # Independent wavelengths: use unchanged FW=15 kernel, padding the final chunk.
        chunks = [wl] if len(wl) in (1, 15) else [wl[i : i + 15] for i in range(0, len(wl), 15)]
        for index, chunk in enumerate(chunks):
            count = len(chunk)
            padded = (
                chunk
                if count == 15 or (count == 1 and len(chunks) == 1)
                else np.r_[chunk, np.repeat(chunk[-1], 15 - count)]
            )
            if pipeline_lock:
                phase(name, 'cpu_prepare_start', chunk=index)
            with attach_cache_event(name, index, state) as made:
                made.append(
                    create_configured(
                        prepare / f'prepare_{index}',
                        cfg,
                        padded,
                        'numpy' if pipeline_lock else a.backend,
                        state,
                        a.block_size,
                        geometry_cache=a.geometry_cache,
                    )
                )
            s, scope = made[0]
            if ksplit_decision is not None:
                s.ksplit_decision = ksplit_decision
            s.kernel.ksplit_emit = lambda decision: phase(name, 'KSPLIT_FALLBACK', **decision)
            from sosfuse2.density import evidence as profile_evidence

            phase(
                name,
                'profile',
                chunk=index,
                where='host',
                **profile_evidence(getattr(s, 'density_optics', None), s.kernel.profile_n),
            )
            xp = s.kernel.xp
            if xp is not np:
                free, _ = xp.cuda.runtime.memGetInfo()
                if free < 4 * 1024**3:
                    raise RuntimeError('E_VRAM_FREE: need at least 4 GiB free')
                from sosfuse2.device_profile import pool_limit_bytes

                xp.get_default_memory_pool().set_limit(size=pool_limit_bytes(free))
            if geometry is None:
                leased = register_geometry(s, scope, a.geometry_cache, a.geometry_layers)
                with (a.geometry_cache / (leased.stats['signature'] + '.lock')).open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    common = cache_complete(leased)
                    build_cells(leased)
                # layers=2: column written here (per-height lock) so prepare() below only loads it.
                had_column = common and column_ready(leased)
                column = ensure_column(leased)
                hit = common and had_column
                geometry = gm.TransientGeometry(s, scope) if a.reduced else leased
                if not a.reduced:
                    # host_geometry uses a shallow CPU view: bind the actual CUDA solver.
                    geometry.s = s
                    geometry.prepare()
                else:
                    # Populate transient groups now so geometry phase excludes source work.
                    for i in range(s.geo.ns):
                        geometry.cell(i)
                phase(
                    name,
                    'geometry_done',
                    cache='hit' if hit else 'miss',
                    seconds=time.monotonic() - geometry_start,
                    signature=leased.stats['signature'],
                    layers=a.geometry_layers,
                    common='hit' if common else 'miss',
                    column=column,
                    column_load=leased.stats.get('aerosol_column') if not a.reduced else None,
                )
            else:
                geometry.s = s
                geometry.scope = scope
            if pipeline_lock:
                from sosfuse2.pipeline_execution import (
                    GpuToken,
                    prepare_geometry,
                    prepare_tables,
                    promote_solver,
                    evaluate_tables,
                    replay_tables,
                    release_device,
                )

                prepare_geometry(s, geometry, a.reduced)
                prepared_table = prepare_tables(s, hs, el, az, a.group, a.batch)
                if persistent:
                    from sosfuse2.pipeline_execution import persistent_chunk

                    host, stats, stiming, collected, sec, qsec = persistent_chunk(
                        s, geometry, prepared_table, a, name, phase
                    )
                    ksplit_decision = stiming.get('ksplit', ksplit_decision)
                    fields.append(host[..., :count])
                    stats_all.append(stats)
                    timings.append(stiming)
                    source_seconds += sec
                    query_seconds += qsec
                    pending_tables.append((s, prepared_table, collected, count, index))
                    phase(name, 'source_done', chunk=index, evals=stats['evaluations'], seconds=sec)
                    phase(name, 'table_transport_done', chunk=index, seconds=qsec)
                    continue
                phase(name, 'cpu_prepare_done', chunk=index, cuda_imported='cupy' in sys.modules)
                token = GpuToken(pipeline_lock, name, phase)
                token.acquire()
                configure(a.work_root.parent / 'nvrtc_cache_shared', a.backend, a.block_size)
                promote_solver(s, geometry, a.backend, a.block_size, a.threads)
                xp = s.kernel.xp
                if xp is not np:
                    free, _ = xp.cuda.runtime.memGetInfo()
                    if free < 4 * 1024**3:
                        raise RuntimeError('E_VRAM_FREE: need at least 4 GiB free')
                    from sosfuse2.device_profile import pool_limit_bytes

                    xp.get_default_memory_pool().set_limit(size=pool_limit_bytes(free))
            phase(name, 'source_start', chunk=index, wavelengths=count)
            t = time.monotonic()
            solver = Solver(s, geometry, fixed_boxes=a.fixed_boxes)
            field, stats = solver.solve(a.max_eval, ANDERSON_TOLERANCE)
            s.kernel.sync()
            sec = time.monotonic() - t
            ksplit_decision = solver.timings.get('ksplit', ksplit_decision)
            source_seconds += sec
            stats_all.append(stats)
            timings.append(solver.timings)
            phase(name, 'source_done', chunk=index, evals=stats['evaluations'], seconds=sec)
            host = s.kernel.host(field)
            fields.append(host[..., :count])
            if os.environ.get('SOSFUSE2_EXPORT_MOMENT') == '1':
                # One extra apply() on the converged source: its radiance moments (4 coherency + K scalar SH) are consistent with
                # the source that table.npz is made from (the loop's last apply saw the pre-mixing iterate). Opt-in, saddle CV.
                solver.apply(field)
                moments.append(s.kernel.host(solver.last_moment)[..., :count].astype('f4'))
            s.ms = np.zeros((*host.shape[:2], 2 * s.W), 'f4')
            s.ms[..., ::2] = host
            from sosfuse2.table_fast import make_table

            t = time.monotonic()
            if pipeline_lock:
                collected = evaluate_tables(s, field, prepared_table)
                sec = time.monotonic() - t
                query_seconds += sec
                pending_tables.append((s, prepared_table, collected, count, index))
                phase(name, 'table_transport_done', chunk=index, seconds=sec)
                # Drop every source-owned array before relinquishing the GPU.
                del solver, field, host
                solver = field = None
                release_device(s, geometry)
                token.release()
                token = None
            else:
                table, _ = make_table(s, host, hs, el, az, a.group, a.batch)
                sec = time.monotonic() - t
                query_seconds += sec
                tables.append(table[..., :count])
                phase(name, 'table_done', chunk=index, seconds=sec)
                # Device buffers belong to the chunk kernel, not to geometry.
                s.kernel.constants.clear()
                s.kernel.device_constants.clear()
                del solver, field, host
                solver = field = None
        if pipeline_lock:
            phase(name, 'cpu_finalize_start')
            for table_s, prepared_table, collected, count, index in pending_tables:
                t = time.monotonic()
                table = replay_tables(table_s, prepared_table, collected)
                sec = time.monotonic() - t
                query_seconds += sec
                tables.append(table[..., :count])
                phase(name, 'table_done', chunk=index, seconds=sec, deferred_replay=True)
            pending_tables.clear()
        table = np.concatenate(tables, axis=-1)
        if not np.isfinite(table).all():
            raise RuntimeError('E_TABLE_FINITE')
        # Uncompressed (2026-10-25): a per-state diagnostic read only through np.load (gate.py); zlib saved 6 % in
        # 0.5-1 s of single-core time per state. table.npz too (2026-10-25: numeric identity is judged on
        # arrays, not file SHA); the pack is tar.zst either way, and same arrays still give the same file bytes.
        np.savez(
            prepare / 'source.npz',
            source=np.concatenate(fields, axis=-1),
            h=s.geo.h,
            chi=s.geo.chi,
            wavelength_nm=wl,
            physics=np.array(json.dumps(physics(), sort_keys=True)),
            optics_state=np.array(json.dumps(optics_state(state), sort_keys=True)),
            source_contract=np.array(
                json.dumps(
                    dict(
                        state=canonical_state(state),
                        settings=cfg,
                        backend=a.backend,
                        polarization_model=a.polarization_model,
                    ),
                    sort_keys=True,
                    allow_nan=False,
                )
            ),
        )
        if moments:
            np.savez(
                prepare / 'moment.npz',
                moment=np.concatenate(moments, axis=-1),
                h=s.geo.h,
                chi=s.geo.chi,
                wavelength_nm=wl,
                L=np.int32(s.kernel.L),
                layout=np.array(
                    '(cell=ih*n_chi+ic, slot, channel); slot [0:4] Rayleigh coherency, [4:4+K] scalar radiance SH k=l(l+1)/2+m (m>=0, cos m phi), frame up=radial ex=sun-ward horizontal'
                ),
            )
        with (out / 'table.npz.tmp').open('wb') as f:
            np.savez(
                f,
                radiance=table,
                wavelength_nm=wl.astype('f4'),
                solar_elevation_deg=hs.astype('f4'),
                view_elevation_deg=el.astype('f4'),
                relative_azimuth_deg=az.astype('f4'),
                units=np.array('W m^-2 sr^-1 nm^-1'),
            )
        os.replace(out / 'table.npz.tmp', out / 'table.npz')
        from sosfuse2.stratosphere import computed_metadata

        optical_meta = computed_metadata(state, getattr(s, 'strato', None))
        from sosfuse2.density import metadata
        from sosfuse2.density import evidence as profile_evidence

        # ②b evidence: compiled/uploaded N, consumed mask, effective f8/f4 SHA, identity, policy,
        # origin. consumed = this process's GPU upload (fused.constants_device); persistent runs log it from the service.
        profile_meta = dict(
            host=profile_evidence(getattr(s, 'density_optics', None), s.kernel.profile_n),
            consumed=getattr(s.kernel, 'profile_consumed', None),
            persistent_service=bool(persistent and pipeline_lock),
            sun_method=getattr(s, 'density_sun_method', None),
        )  # ③ 0(b): gl/exact PROFILE columns of sun table+ground, None = no PROFILE
        meta = dict(
            density=metadata(s),
            profile=profile_meta,
            complete=True,
            contract=CONTRACT_VERSION,
            physics=physics(),
            optics_state=optics_state(state),
            **package_info(),
            signature=signature,
            id=a.id,
            out_name=name,
            parameters=state['parameters'],
            parameters_requested=state['parameters'],
            parameters_computed=optical_meta['parameters_computed'],
            stratosphere=optical_meta['stratosphere'],
            stratosphere_computed=optical_meta['stratosphere_computed'],
            settings=cfg,
            grid=grid_metadata(cfg, s.geo.h),
            mode='reduced-validation' if a.reduced else 'full',
            shape=list(table.shape),
            dtype='float32',
            code_hashes=hashes,
            backend=a.backend,
            polarization_model=a.polarization_model,
            source_solve=stats_all[0] if len(stats_all) == 1 else stats_all,
            source_seconds=source_seconds,
            query_seconds=query_seconds,
            elapsed_seconds=time.monotonic() - started,
            geometry=geometry.stats,
            timings=timings,
            threads=a.threads,
            peak_rss_bytes=peak(),
            attempt=a.attempt,
            gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
            unsupported_parameters=[],
            units='W m^-2 sr^-1 nm^-1',
            folding='mirror azimuth around solar plane',
            table_method=dict(runner='run_gpu_state_v3', table='table_fast.make_table', group=a.group, batch=a.batch),
            spectral_chunks=[len(c) for c in chunks],
            ksplit_effective=(ksplit_decision or {}).get('effective', 0),
            ksplit=ksplit_decision,
            geometry_cache_removed=False,
            reason='pack_not_verified',
            kernel=dict(
                deterministic=s.kernel.deterministic,
                requested=cfg.get('kernel'),
                effective_flags=list(s.kernel.flags),
                block_size=s.kernel.block_size,
                launch=getattr(s.kernel, 'launch_log', None),
                profile_n=s.kernel.profile_n,
            ),
        )
        if getattr(s, 'tropo', None) is not None:
            meta['troposphere_phase_computed'] = s.tropo['meta']
        if getattr(s, 'aerosol_profile', None) is not None:
            meta['troposphere_profile_computed'] = s.aerosol_profile['meta']
        atomic_json(meta_path, meta)
        enable_spectral_previews(pm)
        pack = pm.pack_state(out, a.work_root / 'pack', 'zstd', started=started)
        verify_pack(pack, signature)
        # layers=2: the common layer is shared by every state; never removed by a worker.
        result = release_geometry(a.geometry_cache, leased.cache, True, a.keep_geometry or a.geometry_layers == 2)
        released = True
        meta = json.loads(meta_path.read_text())
        meta.update(result)
        atomic_json(meta_path, meta)
        phase(name, 'geometry_cleanup', **result)
        # Repack once so archived meta records the actual post-verification deletion result.
        # This attempt's first pack and verification completed; its previews are ready.
        pack = pm.pack_state(out, a.work_root / 'pack', 'zstd', started=started, reuse_previews=True)
        verify_pack(pack, signature)
        phase(name, 'pack_done', bytes=pack.stat().st_size, **result)
        if pipeline_lock:
            phase(name, 'cpu_finalize_done')
    finally:
        if token is not None:
            # Exceptions must not unlock while this process still owns queued
            # CUDA work. Driver failures remain failures, never successful packs.
            solver = field = None
            try:
                release_device(s, geometry)
            finally:
                token.release()
        if leased is not None and not released:
            release_geometry(a.geometry_cache, leased.cache, False, a.keep_geometry)
        stop.set()


def run_extra(a):
    from .components import production_only

    production_only()
    """M3: table rows at extra solar elevations from a stored source
    (<work-root>/tmp/<source_out_name>/source.npz written by run()); no solve, no geometry cache."""
    if not shutil.which('zstd'):
        raise RuntimeError('E_ZSTD_MISSING: install zstd (required by pack and verify)')
    for k in (
        'OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
        'NUMEXPR_NUM_THREADS',
        'BLIS_NUM_THREADS',
    ):
        os.environ[k] = str(a.threads)
    import numpy as np
    from sosfuse2 import geometry as gm
    from sosfuse2 import table_fast as tf
    from sosfuse2.host import pack_state as pm
    from sosfuse2.run_fuse import monitor, peak

    if a.polarization_model != 'depolarizing':
        raise RuntimeError('E_UNSUPPORTED_POL_MODEL')
    if a.backend == 'numpy':

        class ScalarKernel(gm.Kernel):
            def host(self, x):
                return np.asarray(x)

        gm.Kernel = ScalarKernel
    found = [s for s in read_states(a.states, True) if s['id'] == a.id and s.get('runnable')]
    if len(found) != 1:
        raise ValueError('E_STATE_ID: exactly one runnable state required')
    state = effective_state(found[0], a)
    cfg = resolve_settings(settings_for(a), state)
    name = a.out_name or a.id
    derived = extra_contract(cfg, a.extra_elevations, a.source_out_name)
    signature, hashes = signed_contract(state, cfg, a.backend, a.reduced, a.polarization_model, derived)
    src_out = a.work_root / 'out' / a.source_out_name
    src = a.work_root / 'tmp' / a.source_out_name / 'source.npz'
    out = a.work_root / 'out' / name
    prepare = a.work_root / 'tmp' / name
    if name == a.source_out_name:
        raise ValueError('E_EXTRA_SOURCE: output must differ from source')
    out.mkdir(parents=True, exist_ok=True)
    prepare.mkdir(parents=True, exist_ok=True)
    output_lock = (out / '.worker.lock').open('a')
    try:
        fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('E_OUTPUT_BUSY: out_name already owned by another worker')
    meta_path = out / 'meta.json'
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        check_optics(previous.get('optics_state'), optics_state(state))
        if previous.get('signature') != signature:
            raise RuntimeError('E_OUTPUT_IDENTITY: use a fresh work-root/out_name')
    started = time.monotonic()
    stop = threading.Event()
    threading.Thread(target=monitor, args=(stop, a.rss_gb * 1e9), daemon=True).start()
    from sosfuse2.compile_cache import configure

    configure(a.work_root.parent / 'nvrtc_cache_shared', a.backend, a.block_size)
    os.environ['CUDA_CACHE_PATH'] = str(prepare / 'cuda_driver_cache')
    try:
        phase(name, 'source_reuse_start', source_out_name=a.source_out_name)
        try:
            source_meta = json.loads((src_out / 'meta.json').read_text())
        except (OSError, ValueError):
            raise RuntimeError('E_EXTRA_SOURCE: source job meta missing')
        if not src.is_file():
            raise RuntimeError('E_EXTRA_SOURCE: source.npz missing')
        # package_version/package_commit are recorded in meta but are never a reuse condition.
        from sosfuse2.components import check_current  # E_PHYSICS_MIXED / _CONTRACT before the generic comparison

        check_current(source_meta.get('physics'), what='source job')
        check_optics(source_meta.get('optics_state'), optics_state(state))
        same = dict(
            complete=True,
            contract=CONTRACT_VERSION,
            physics=physics(),
            id=a.id,
            settings=cfg,
            backend=a.backend,
            polarization_model=a.polarization_model,
            mode='reduced-validation' if a.reduced else 'full',
            code_hashes=hashes,
        )
        bad = [k for k, v in same.items() if source_meta.get(k) != v]
        if bad:
            raise RuntimeError('E_EXTRA_SOURCE: source meta differs: ' + ','.join(bad))
        wl = np.asarray(cfg['wavelengths_nm'])
        hs = np.asarray(derived['table_elevations'])
        nr, nc = cfg['view_shape']
        el = 90 - (np.arange(nr) + 0.5) * 90 / nr
        az = (np.arange(nc) + 0.5) * 180 / nc
        # Same FW=15 chunks as run(); a short final chunk is padded with its last wavelength (lanes are independent).
        chunks = [wl] if len(wl) in (1, 15) else [wl[i : i + 15] for i in range(0, len(wl), 15)]
        pad = lambda c: (
            c if len(c) == 15 or (len(c) == 1 and len(chunks) == 1) else np.r_[c, np.repeat(c[-1], 15 - len(c))]
        )
        t = time.monotonic()
        s, scope = create_configured(
            prepare / 'prepare_0', cfg, pad(chunks[0]), a.backend, state, a.block_size, geometry_cache=a.geometry_cache
        )
        create_seconds = time.monotonic() - t
        from sosfuse2.stratosphere import computed_metadata

        optical_meta = computed_metadata(state, getattr(s, 'strato', None))
        from sosfuse2.density import metadata, DEFAULTS

        if source_meta.get('density', dict(DEFAULTS)) != metadata(s):
            raise RuntimeError('E_EXTRA_SOURCE: density profile differs')
        if any(
            source_meta.get(k) != optical_meta[k]
            for k in ('parameters_requested', 'parameters_computed', 'stratosphere')
        ):
            raise RuntimeError('E_EXTRA_SOURCE: requested/computed optical state differs')
        full = tf.load_source(src, s, wl, settings=cfg, backend=a.backend, full_settings=True)
        source_sha = hashlib.sha256(src.read_bytes()).hexdigest()
        phase(name, 'source_reuse_done', seconds=time.monotonic() - started, source_sha256=source_sha)
        tables = []
        query_seconds = 0.0
        start = 0
        for index, chunk in enumerate(chunks):
            count = len(chunk)
            if index:
                t = time.monotonic()
                s, scope = create_configured(
                    prepare / f'prepare_{index}',
                    cfg,
                    pad(chunk),
                    a.backend,
                    state,
                    a.block_size,
                    geometry_cache=a.geometry_cache,
                )
                create_seconds += time.monotonic() - t
            field = full[..., start : start + count]
            start += count
            if count < s.W:
                field = np.concatenate([field, np.repeat(field[..., -1:], s.W - count, -1)], -1)
            s.ms = np.zeros((*field.shape[:2], 2 * s.W), 'f4')
            s.ms[..., ::2] = field
            t = time.monotonic()
            part, _ = tf.make_table(s, field, hs, el, az, a.group, a.batch)
            sec = time.monotonic() - t
            query_seconds += sec
            tables.append(part[..., :count])
            phase(name, 'table_done', chunk=index, seconds=sec, rows=len(hs))
            if len(chunks) > 1:
                s.kernel.constants.clear()
                s.kernel.device_constants.clear()
        table = np.concatenate(tables, -1)
        if not np.isfinite(table).all():
            raise RuntimeError('E_TABLE_FINITE')
        # Anchor rows (-3,-6) against the source job's own table: reuse check.
        with np.load(src_out / 'table.npz') as f:
            ref = f['radiance']
            ref_hs = f['solar_elevation_deg']
        anchors = {}
        for h in (-3.0, -6.0):
            a_row = table[int(np.flatnonzero(hs == h)[0])]
            b_row = ref[int(np.flatnonzero(ref_hs == h)[0])]
            anchors[str(h)] = dict(
                max_rel=float(np.max(np.abs(a_row - b_row)) / max(float(np.max(np.abs(b_row))), 1e-30))
            )
        for row in anchors.values():
            row['within_tol'] = row['max_rel'] <= 1e-6
        if max(v['max_rel'] for v in anchors.values()) > 1e-6:
            raise RuntimeError('E_EXTRA_ANCHOR: anchor rows differ from source table by >1e-6')
        with (out / 'table.npz.tmp').open('wb') as f:
            np.savez(
                f,
                radiance=table,
                wavelength_nm=wl.astype('f4'),
                solar_elevation_deg=hs.astype('f4'),
                view_elevation_deg=el.astype('f4'),
                relative_azimuth_deg=az.astype('f4'),
                units=np.array('W m^-2 sr^-1 nm^-1'),
            )
        os.replace(out / 'table.npz.tmp', out / 'table.npz')
        from sosfuse2.density import metadata

        meta = dict(
            density=metadata(s),
            complete=True,
            contract=CONTRACT_VERSION,
            physics=physics(),
            optics_state=optics_state(state),
            **package_info(),
            signature=signature,
            id=a.id,
            out_name=name,
            derived=derived,
            parameters=state['parameters'],
            **optical_meta,
            settings=cfg,
            grid=grid_metadata(cfg, s.geo.h),
            mode='reduced-validation' if a.reduced else 'full',
            shape=list(table.shape),
            dtype='float32',
            code_hashes=hashes,
            backend=a.backend,
            polarization_model=a.polarization_model,
            source_file=str(src.relative_to(a.work_root)),
            source_sha256=source_sha,
            source_signature=source_meta.get('signature'),
            anchor_check=anchors,
            solver_create_seconds=create_seconds,
            query_seconds=query_seconds,
            elapsed_seconds=time.monotonic() - started,
            threads=a.threads,
            peak_rss_bytes=peak(),
            attempt=a.attempt,
            gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
            unsupported_parameters=[],
            units='W m^-2 sr^-1 nm^-1',
            folding='mirror azimuth around solar plane',
            table_method=dict(
                runner='run_gpu_state_v3 --extra-elevations',
                table='table_fast.make_table',
                group=a.group,
                batch=a.batch,
            ),
            geometry_cache_removed=False,
            reason='no_geometry',
            kernel=dict(
                deterministic=s.kernel.deterministic,
                requested=cfg.get('kernel'),
                effective_flags=list(s.kernel.flags),
                block_size=s.kernel.block_size,
                launch=getattr(s.kernel, 'launch_log', None),
                profile_n=s.kernel.profile_n,
            ),
        )
        if getattr(s, 'tropo', None) is not None:
            meta['troposphere_phase_computed'] = s.tropo['meta']
        if getattr(s, 'aerosol_profile', None) is not None:
            meta['troposphere_profile_computed'] = s.aerosol_profile['meta']
        atomic_json(meta_path, meta)
        enable_spectral_previews(pm)
        pack = pm.pack_state(out, a.work_root / 'pack', 'zstd', started=started)
        verify_pack(pack, signature)
        phase(name, 'pack_done', bytes=pack.stat().st_size, geometry_cache_removed=False, reason='no_geometry')
    finally:
        stop.set()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--id', required=True)
    p.add_argument('--out-name')
    p.add_argument('--states', type=Path, default=paths.ASSETS / 'states_v2.json')
    p.add_argument('--work-root', type=Path, required=True)
    p.add_argument('--geometry-cache', type=Path)
    p.add_argument('--backend', choices=['cupy', 'numpy'], default='cupy')
    p.add_argument('--reduced', action='store_true')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--max-eval', type=int, default=64)
    p.add_argument('--rss-gb', type=float)
    p.add_argument(
        '--block-size',
        type=int,
        choices=(128, 256),
        default=None,
        help='A6(d): 256 threads/block (97,872 B shared at 15 wavelengths); default SOSFUSE2_BLOCK_SIZE or 128',
    )
    p.add_argument(
        '--fixed-boxes', action='store_true', help='A5: observer boxes from one fixed Bundle on the device source'
    )
    p.add_argument(
        '--geometry-layers',
        type=int,
        choices=(1, 2),
        default=2,
        help='2 (default): common endpoint layer (geometry_contract signature, no scale height) + per-height aerosol column; a worker never removes the common layer',
    )
    p.add_argument(
        '--extra-elevations',
        help='M3: comma list, e.g. --extra-elevations=-3.5,-4.5,-6.5; table rows from a stored source, no solve',
    )
    p.add_argument('--source-out-name', help='M3: out_name whose <work-root>/tmp/<name>/source.npz is reused')
    p.add_argument('--attempt', type=int, default=1)
    p.add_argument('--group', type=int, default=32)
    p.add_argument('--batch', type=int, default=25)
    p.add_argument('--geometry-only', action='store_true')
    p.add_argument('--lease-pid', type=int)
    p.add_argument('--column-cache', type=Path)
    p.add_argument('--gpu-lock', type=Path, help='CPU prepare first, exclusive GPU token, CPU finalize after release')
    p.add_argument('--keep-geometry', action='store_true')
    p.add_argument('--settings-json', type=Path)
    p.add_argument('--polarization-model', choices=['depolarizing', 'preserving'], default='depolarizing')
    from sosfuse2.density import add_arguments

    add_arguments(p)
    p.add_argument(
        '--stratosphere',
        choices=['on'],
        help='explicitly enable the state stratospheric layer; default honors state.status',
    )
    a = p.parse_args()
    a.work_root = a.work_root.resolve()
    a.geometry_cache = (a.geometry_cache or a.work_root / 'geometry_cache').resolve()
    if a.column_cache is not None:
        os.environ['SOSFUSE2_COLUMN_CACHE'] = str(a.column_cache.resolve())
    # Column-cache job scope (column_cache.py first-use comparison): one work-root = one job, shared by
    # every state of a pipeline queue. Its markers live outside out/ and pack/, never archived.
    os.environ['SOSFUSE2_COLUMN_CACHE_JOB'] = str(a.work_root)
    os.environ['SOSFUSE2_STATE_ID'] = a.out_name or a.id
    # Attach cache is on by default; only an explicit 'off' disables it.
    attach = os.environ.get('STRATO_ATTACH_CACHE', '').strip()
    if attach.lower() == 'off':
        os.environ.pop('STRATO_ATTACH_CACHE', None)
    elif not attach:
        os.environ['STRATO_ATTACH_CACHE'] = str(a.geometry_cache / 'attach_cache')
    if any(not re.fullmatch('[A-Za-z0-9_-]+', s) for s in (a.id, a.out_name or a.id)):
        p.error('unsafe state/output id')
    try:
        kernel = kernel_settings()
    except ValueError as e:
        p.error(str(e))
    if a.block_size is None:
        a.block_size = (kernel or {}).get('block_size', 128)
    elif os.environ.get('SOSFUSE2_BLOCK_SIZE', '').strip() and a.block_size != (kernel or {}).get('block_size', 128):
        p.error('E_KERNEL_BLOCK_SIZE: --block-size disagrees with SOSFUSE2_BLOCK_SIZE (recorded in settings.kernel)')
    if a.gpu_lock and (a.geometry_only or a.extra_elevations):
        p.error('E_PIPELINE_MODE: gpu-lock requires ordinary state mode')
    if a.geometry_only:
        a.backend = 'numpy'
    if a.backend == 'numpy' and not a.reduced and not a.geometry_only:
        p.error('numpy requires --reduced unless geometry-only')
    if sys.platform == 'darwin' and (a.backend != 'numpy' or not a.reduced):
        p.error('Mac requires numpy + reduced')
    from sosfuse2.host_limits import default_rss_gb, rss_cap_gb

    if a.rss_gb is None:
        a.rss_gb = default_rss_gb(1)  # one state process
    if not 0 < a.rss_gb <= rss_cap_gb():
        p.error('E_RSS_BUDGET: 0 < rss-gb <= %g required on this host' % rss_cap_gb())
    if not 1 <= a.threads <= (os.cpu_count() or 1) or a.max_eval < 2 or a.group < 1 or a.batch < 1:
        p.error('invalid budgets')
    if a.lease_pid is not None and not a.geometry_only:
        p.error('--lease-pid requires --geometry-only')
    if bool(a.extra_elevations) != bool(a.source_out_name):
        p.error('--extra-elevations and --source-out-name go together')
    if a.extra_elevations and a.geometry_only:
        p.error('--extra-elevations has no geometry phase')
    if a.geometry_only:
        geometry_only(a)
    elif a.extra_elevations:
        run_extra(a)
    else:
        run(a)


if __name__ == '__main__':
    main()
