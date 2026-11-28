#!/usr/bin/env python3
"""v2 numerical path with v3 identities, cooperative cache leases and phase events."""

import argparse, contextlib, fcntl, hashlib, json, os, re, shutil
import sys, threading, time
from pathlib import Path
from functools import wraps
from sosfuse2.backends import vram_trace

_THREAD_ENV = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'BLIS_NUM_THREADS',
)


def _thread_env_before_numpy(argv, environ):
    """Write the --threads value of argv (default 2) to the thread variables, as every mode of main() does.

    BLAS libraries read these variables once, when NumPy is first imported, and this module imports NumPy further
    down, so a direct run has to set them before that import. The last --threads N or --threads=N before a bare
    '--' wins, as in argparse, and so do its prefixes from --t: argparse reads a prefix as --threads or, when another
    option shares it, rejects the command line, so no mode runs with a value other than the one written here
    (test_threads_blas_env pins that no other option name is itself such a prefix). A value that is not a positive
    integer writes nothing; main() then rejects it. Returns the value written, or None.
    """
    value = '2'
    for i, arg in enumerate(argv):
        if arg == '--':
            break
        name, eq, rest = arg.partition('=')
        if not (len(name) >= 3 and '--threads'.startswith(name)):
            continue
        if eq:
            value = rest
        elif i + 1 < len(argv):
            value = argv[i + 1]
    try:
        n = int(value)
    except ValueError:
        return None
    if n < 1:
        return None
    for k in _THREAD_ENV:
        environ[k] = str(n)
    return n


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    _thread_env_before_numpy(sys.argv[1:], os.environ)
from sosfuse2.foundation import paths
from sosfuse2.solve import build, configuration
from sosfuse2.solve.build import SUN_SCALE
# Names that moved to solve/configuration.py stay importable from run_state for existing callers.
from sosfuse2.solve.configuration import (  # noqa: F401
    ANDERSON_TOLERANCE, AUTO_FLAGS, CHI_STEPS_DEG, H_REFINE_ALIASES, SH_L,
    _H_RULE, _chi_window, effective_state, effective_tol, execution_domain, extra_contract,
    grid_metadata, height_rule, kernel_settings, parse_elevations, read_states,
    refined_chi, refined_heights, resolve_settings, settings_for,
)
from . import run_storage as storage
# Names that moved to jobs/run_storage.py stay importable from run_state for existing callers.
from .run_storage import FirstOrderConflict, atomic_json, enable_spectral_previews, output_identity_message, verify_pack  # noqa: F401


def create_configured(
    output, cfg, wavelengths, backend, state, block_size=128, geometry_cache=None, *, share_sun_geometry=True
):
    """Build a state solver using the runner's current solar column resolution."""
    return build.create_configured(
        output, cfg, wavelengths, backend, state, block_size, geometry_cache,
        share_sun_geometry=share_sun_geometry, sun_scale=SUN_SCALE,
    )


def solve_chunk(solver, max_eval, tol):
    from ..foundation.version import __version__

    engine_file = str(paths.PKG / '__init__.py')
    try:
        field, stats = solver.solve(max_eval, tol)
    except Exception as exc:
        phase(os.environ.get('SOSFUSE2_STATE_ID', 'GPU_SERVICE'), 'solve_failed',
              tolerance=tol, max_eval=max_eval, termination=str(exc), engine_file=engine_file,
              engine_version=__version__)
        raise
    if stats['tolerance'] != tol:
        raise RuntimeError('E_SOLVE_TOL: solver returned a different tolerance')
    stats.update(max_eval=max_eval, termination='converged')
    solver.timings['solve_worker'] = dict(pid=os.getpid(), file=engine_file, version=__version__,
                                        L=solver.kernel.L, top_km=solver.kernel.domain.top_km)
    return field, stats


def chunk_summary(stats, worker):
    final = stats['history'][-1]
    return dict(worker=worker, tolerance=stats['tolerance'], evaluations=stats['evaluations'],
                max_eval=stats['max_eval'], final_relative=max(max(final['observer_relative']), max(final['source_relative'])),
                converged=all(stats['converged']), termination=stats['termination'], cache='fresh_solve')


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
    never computed by two lock groups at once (_ground_geometry's temporary file is per process).
    Density and cfg are left out (coarser): cfg is shared by a queue and its mapping to the grids would
    have to be copied exactly. None = layer off: attach returns before _ground_geometry (tau 0)."""
    from sosfuse2.atmosphere.stratosphere import ATTACH_OPTICS_PARAMS, tau_of, profile_config

    if tau_of(state) == 0:
        return None
    p = profile_config(state)
    params = {k: v for k, v in state.get('parameters', {}).items() if k not in ATTACH_OPTICS_PARAMS}
    text = json.dumps(dict(profile=[p['center'], p['sigma']], params=params), sort_keys=True, default=str)
    return 'attach_' + hashlib.sha256(text.encode()).hexdigest()[:16]


@contextlib.contextmanager
def attach_cache_event(name, index, state, record=True):
    """Per-state PHASE record: hit, miss, unused (no stratosphere) or disabled.
    stratosphere._ground_geometry publishes '<key>.npz' through a per-process temporary file, and two writers of
    one key would compute it twice, so they never share a folder. Each call therefore runs against a private
    overlay folder (hard links to the published files): reads hit through the links, a miss is written
    privately and then published with an atomic rename. That alone is safe without any lock.
    The attach_lock_name group lock only avoids duplicate work: it is taken while the group has not
    published its key yet ('<group>.done' absent), so the first state computes and the rest wait and
    then hit. Once published, states of the group run in parallel and never wait.
    The caller appends (solver,scope); hit = this state's own key file was already published.
    record=False keeps the overlay and publish but writes no PHASE record (the geometry-only prefetch, whose
    attach work the worker's own attach_cache record then sees as a hit; the pipeline and reference checks count those records)."""
    original_root = os.environ.get('STRATO_ATTACH_CACHE', '')
    root = original_root.strip()
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
                os.environ['STRATO_ATTACH_CACHE'] = original_root
            for n in os.listdir(private):
                if re.fullmatch('attach_geometry_[0-9a-f]{24}[.]npz', n) and n not in before:
                    os.replace(private / n, folder / n)
            s = made[0][0] if made else None
            optics = getattr(s, 'strato', None)
            if optics is not None:
                from sosfuse2.atmosphere.stratosphere import _geometry_key

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
    from sosfuse2.solve.ray_geometry_cpu import host_geometry

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


from ..foundation.content_identity import package_hashes, package_info
from ..foundation.content_identity import ACCEPTED_SOURCES as ACCEPTED_SOURCES, HOST_MODULES as HOST_MODULES, DATA_ASSETS as DATA_ASSETS


from ..atmosphere.components import physics, optics_state, canonical_state

CONTRACT_VERSION = 'v5'  # Readers also accept v3 and v4 tables.


def signed_contract(state, cfg, backend, reduced, polarization_model, derived=None):
    state = canonical_state(state)
    amounts = optics_state(state)
    from ..atmosphere.components import reference_active

    if amounts['o4'] and not reference_active():
        from ..atmosphere.minor_absorbers import load

        load('o4')
    hashes = package_hashes()
    from sosfuse2.atmosphere.stratosphere import tau_of, table_path, path_label

    if tau_of(state) > 0:
        path = table_path(state)
        key = path_label(path) if path.resolve().is_relative_to(paths.PKG) else f'external/stratosphere/{path.name}'
        hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    from sosfuse2.atmosphere import density

    if density.build(state)['meta'].get('asset_sha256') is not None:
        hashes[density.ASSET_KEY] = hashlib.sha256(density.asset_path().read_bytes()).hexdigest()  # logical key
    from sosfuse2.atmosphere.tropo_mie import config as tropo_config, table_path as tropo_path, path_label as tropo_label

    if tropo_config(state) is not None:
        path = tropo_path(tropo_config(state))
        key = tropo_label(path) if path.resolve().is_relative_to(paths.PKG) else f'external/troposphere/{path.name}'
        hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    if cfg.get('spectral') is not None:
        from sosfuse2.atmosphere import spectral
        from sosfuse2.atmosphere.components import check_current  # a legacy block (copied from an old meta) never signs a new run

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
        tolerance=effective_tol(cfg),
        code_hashes=hashes,
        grid=grid_metadata(cfg),
    )
    # M3: a table derived from a stored source signs its own mode; ordinary jobs keep the old signed input bytes.
    if derived is not None:
        contract['derived'] = derived
    signature = hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return signature, hashes


def geometry_only(a):
    # This entry point never selects a CUDA backend or constructs a fused Solver.
    _set_run_threads(a)
    import numpy as np
    from sosfuse2.solve.ray_geometry_cpu import build_cells, cache_complete, column_ready, ensure_column
    from sosfuse2.foundation.host_limits import monitor

    _set_run_threads(a)
    if a.polarization_model != 'depolarizing':
        raise RuntimeError(
            'E_UNSUPPORTED_POL_MODEL: sosfuse2 solve/source_solver.py and backends/kernel.cu have no preserving aerosol tensor transport'
        )
    if a.lease_pid is not None and (a.lease_pid < 1 or a.lease_pid == os.getpid() or not alive(str(a.lease_pid))):
        raise RuntimeError('E_PREFETCH_LEASE_PID')
    state, cfg = configuration.state_settings(a, geometry_only=True, resolve=resolve_settings)
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
        # compute it side by side in a shared STRATO_ATTACH_CACHE.
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


def trace_state(func):
    @wraps(func)
    def wrapped(a):
        if not vram_trace.enabled(a.backend):
            return func(a)
        with vram_trace.session(a.out_name or a.id, phase):
            return func(a)

    return wrapped


#: The four preparation pools; the same names as jobs/pipeline.py PREP_WORKER_ENV (tests/test_threads_pools.py).
POOL_WORKER_ENV = (
    'SOSFUSE2_COLUMN_WORKERS',
    'SOSFUSE2_SUN_WORKERS',
    'SOSFUSE2_GROUND_WORKERS',
    'SOSFUSE2_GEOMETRY_WORKERS',
)


def _bound_pool_workers(threads, environ):
    """An explicit --threads also sizes each preparation pool whose variable is unset or blank; nonblank values are kept."""
    for name in POOL_WORKER_ENV:
        if not environ.get(name, '').strip():
            environ[name] = str(threads)


@contextlib.contextmanager
def _pool_limit(threads, environ):
    """Apply _bound_pool_workers for one mode call and put the previous values back afterwards, also after an error."""
    saved = {name: environ.get(name) for name in POOL_WORKER_ENV}
    _bound_pool_workers(threads, environ)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                environ.pop(name, None)
            else:
                environ[name] = value


class _StateRun:
    """Per-run ownership of buffers, leases and the active device token."""

    def __init__(self):
        self.token = None
        self.geometry = None
        self.leased = None
        self.released = False


def _check_run_options(a):
    from ..atmosphere.components import production_only

    production_only()
    if a.polarization_model == 'preserving':
        raise RuntimeError(
            'E_UNSUPPORTED_POL_MODEL: fused depolarizing HG only; sosfuse2 solve/source_solver.py and backends/kernel.cu need a preserving tensor transport'
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


def _set_run_threads(a):
    for k in _THREAD_ENV:
        os.environ[k] = str(a.threads)


def _preflight_device(state):
    # Refuse a first light the device profile cannot split, before any geometry or CPU preparation (split profile, C2 8.6 s).
    from sosfuse2.backends.device_profile import select, smi_device_name, first_light_split_error
    from sosfuse2.solve.source_solver import default_flags
    from sosfuse2.atmosphere.stratosphere import tau_of

    pname, profile, _ = select(smi_device_name())
    err = first_light_split_error(default_flags(), tau_of(state) > 0, pname, profile)
    if err:
        raise RuntimeError(err)


def _start_run_monitor(job, a, monitor):
    job.started = time.monotonic()
    job.stop = threading.Event()
    threading.Thread(target=monitor, args=(job.stop, a.rss_gb * 1e9), daemon=True).start()


def _prepare_run_grid(job, a, np, configure, default_root):
    job.pipeline_lock = getattr(a, 'gpu_lock', None)
    job.pending_tables = []
    job.solver = job.field = None
    job.persistent = bool(os.environ.get('SOS_PERSISTENT_ROOT'))
    if not job.pipeline_lock:
        configure(default_root(a.work_root), a.backend, a.block_size)
    os.environ['CUDA_CACHE_PATH'] = str(job.prepare / 'cuda_driver_cache')
    job.wl = np.asarray(job.cfg['wavelengths_nm'])
    job.hs = np.asarray(job.cfg['solar_elevation_deg'])
    nr, nc = job.cfg['view_shape']
    job.el = 90 - (np.arange(nr) + 0.5) * 90 / nr
    job.az = (np.arange(nc) + 0.5) * 180 / nc
    job.tables = []
    job.first_tables = []
    job.fields = []
    job.moments = []
    job.stats_all = []
    job.timings = []
    job.trace = vram_trace.current.get()
    job.service_traces = []
    job.source_seconds = job.query_seconds = 0.0
    job.ksplit_decision = None


def _build_chunk_inputs(job, a, np, index, chunk, *, prepare_table=False):
    """Build an independent solver, scope and query bundle using only fixed run inputs."""
    count = len(chunk)
    padded = (
        chunk
        if count == 15 or (count == 1 and len(job.chunks) == 1)
        else np.r_[chunk, np.repeat(chunk[-1], 15 - count)]
    )
    if job.pipeline_lock:
        phase(job.name, 'cpu_prepare_start', chunk=index)
    with attach_cache_event(job.name, index, job.state) as made:
        made.append(
            create_configured(
                job.prepare / f'prepare_{index}',
                job.cfg,
                padded,
                'numpy' if job.pipeline_lock else a.backend,
                job.state,
                a.block_size,
                geometry_cache=a.geometry_cache,
            )
        )
    s, scope = made[0]
    table = None
    if prepare_table:
        from .pipeline_execution import prepare_tables

        table = prepare_tables(s, job.hs, job.el, job.az, a.group, a.batch)
    return s, scope, table


def _prepare_wavelength_chunk(job, a, np, chunk_metadata, inputs=None):
    if job.trace is not None:
        job.trace.chunk, job.trace.stage, job.trace.iteration = job.index, 'setup', None
    job.s, job.scope, job.prepared_table = (
        _build_chunk_inputs(job, a, np, job.index, job.chunk) if inputs is None else inputs.result()
    )
    job.table_density = chunk_metadata(job.s, job.chunk, job.table_density)
    if job.ksplit_decision is not None:
        job.s.ksplit_decision = job.ksplit_decision

    name = job.name

    def ksplit_fallback(decision):
        from sosfuse2.foundation.host_limits import ksplit_fallback_warning

        phase(name, 'KSPLIT_FALLBACK', **decision)
        print(ksplit_fallback_warning(decision, name), file=sys.stderr, flush=True)

    job.s.kernel.ksplit_emit = ksplit_fallback
    from sosfuse2.atmosphere.density import evidence as profile_evidence

    phase(
        job.name,
        'profile',
        chunk=job.index,
        where='host',
        **profile_evidence(getattr(job.s, 'density_optics', None), job.s.kernel.profile_n),
    )
    job.xp = job.s.kernel.xp
    if job.xp is not np:
        from sosfuse2.backends.device_profile import apply_pool_limit

        apply_pool_limit(job.xp)
    return len(job.chunk)


def _prepare_chunk_geometry(job, a, gm, cache_complete, build_cells, column_ready, ensure_column):
    if job.geometry is None:
        job.leased = register_geometry(job.s, job.scope, a.geometry_cache, a.geometry_layers)
        with (a.geometry_cache / (job.leased.stats['signature'] + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            common = cache_complete(job.leased)
            build_cells(job.leased)
        # layers=2: column written here (per-height lock) so prepare() below only loads it.
        had_column = common and column_ready(job.leased)
        column = ensure_column(job.leased)
        hit = common and had_column
        job.geometry = gm.TransientGeometry(job.s, job.scope) if a.reduced else job.leased
        if not a.reduced:
            # host_geometry uses a shallow CPU view: bind the actual CUDA solver.
            job.geometry.s = job.s
            job.geometry.prepare()
        else:
            # Populate transient groups now so geometry phase excludes source work.
            for i in range(job.s.geo.ns):
                job.geometry.cell(i)
        phase(
            job.name,
            'geometry_done',
            cache='hit' if hit else 'miss',
            seconds=time.monotonic() - job.geometry_start,
            signature=job.leased.stats['signature'],
            layers=a.geometry_layers,
            common='hit' if common else 'miss',
            column=column,
            column_load=job.leased.stats.get('aerosol_column') if not a.reduced else None,
        )
    else:
        job.geometry.s = job.s
        job.geometry.scope = job.scope


def _solve_chunk_source(job, a, np, Solver, count):
    phase(job.name, 'source_start', chunk=job.index, wavelengths=count)
    if job.trace is not None:
        job.trace.stage = 'source'
    t = time.monotonic()
    job.solver = Solver(job.s, job.geometry, fixed_boxes=a.fixed_boxes)
    job.field, job.stats = solve_chunk(job.solver, a.max_eval, job.cfg['anderson_tol'])
    job.s.kernel.sync()
    sec = time.monotonic() - t
    job.ksplit_decision = job.solver.timings.get('ksplit', job.ksplit_decision)
    job.source_seconds += sec
    job.stats_all.append(job.stats)
    job.timings.append(job.solver.timings)
    phase(
        job.name,
        'source_done',
        chunk=job.index,
        evals=job.stats['evaluations'],
        seconds=sec,
        **chunk_summary(job.stats, job.solver.timings['solve_worker']),
    )
    job.host = job.s.kernel.host(job.field)
    job.fields.append(job.host[..., :count])
    if os.environ.get('SOSFUSE2_EXPORT_MOMENT') == '1':
        # One extra apply() on the converged source: its radiance moments (4 coherency + K scalar SH) are consistent with
        # the source that table.npz is made from (the loop's last apply saw the pre-mixing iterate). Enabled by SOSFUSE2_EXPORT_MOMENT.
        job.solver.apply(job.field)
        job.moments.append(job.s.kernel.host(job.solver.last_moment)[..., :count].astype('f4'))
    job.s.ms = np.zeros((*job.host.shape[:2], 2 * job.s.W), 'f4')
    job.s.ms[..., ::2] = job.host


def _collect_chunk_table(job, a, np, count):
    from sosfuse2.outputs.radiance_table import make_table

    if job.trace is not None:
        job.trace.stage, job.trace.iteration = 'table_transport', None
    t = time.monotonic()
    if job.pipeline_lock:
        job.collected = job.evaluate_tables(job.s, job.field, job.prepared_table)
        sec = time.monotonic() - t
        job.query_seconds += sec
        job.pending_tables.append((job.s, job.prepared_table, job.collected, count, job.index))
        phase(job.name, 'table_transport_done', chunk=job.index, seconds=sec)
        # Drop every source-owned array before relinquishing the GPU.
        del job.solver, job.field, job.host
        job.solver = job.field = None
        job.release_device(job.s, job.geometry)
        job.token.release()
        job.token = None
    else:
        job.table, job.table_aux = make_table(job.s, job.host, job.hs, job.el, job.az, a.group, a.batch)
        sec = time.monotonic() - t
        job.query_seconds += sec
        job.tables.append(job.table[..., :count])
        if job.cfg.get('export_first_order'):
            job.first_tables.append(job.s.first_order_table[..., :count])
        phase(job.name, 'table_done', chunk=job.index, seconds=sec)
        # Device buffers belong to the chunk kernel, not to geometry.
        job.s.kernel.constants.clear()
        job.s.kernel.device_constants.clear()
        del job.solver, job.field, job.host
        job.solver = job.field = None
        if job.xp is not np and job.index + 1 < len(job.chunks):
            # The endpoints are shared, but split records and plans belong to this solve.
            job.s.kernel.sync()
            bundles = iter((job.geometry.bundle, job.geometry.ground_bundle))
            for job.bundle in bundles:
                if job.bundle is not None:
                    job.bundle.ksplit = None
            job.s.kernel.ksplit_aero = None
            job.s.kernel.ground_sh_tables.clear()
            job.xp.get_default_memory_pool().free_all_blocks()


def _prepare_pipeline_chunk(job, a, np, count, configure, default_root, on_sent=None):
    from sosfuse2.jobs.pipeline_execution import (
        GpuToken,
        prepare_geometry,
        prepare_tables,
        promote_solver,
        evaluate_tables,
        replay_tables,
        release_device,
    )

    job.release_device = release_device
    job.evaluate_tables = evaluate_tables
    job.replay_tables = replay_tables
    prepare_geometry(job.s, job.geometry, a.reduced)
    if job.prepared_table is None:
        job.prepared_table = prepare_tables(job.s, job.hs, job.el, job.az, a.group, a.batch)
    if job.persistent:
        from sosfuse2.jobs.pipeline_execution import persistent_chunk

        # every event of the persistent exchange carries the chunk index (benchmark segments are joined by chunk, not by order)
        job.host, job.stats, job.stiming, job.collected, sec, qsec = persistent_chunk(
            job.s, job.geometry, job.prepared_table, a, job.name,
            lambda name, event, **kw: phase(name, event, **{'chunk': job.index, **kw}), tol=job.cfg['anderson_tol'],
            shared=job.geometry_buffers, on_sent=on_sent,
        )
        if job.trace is not None:
            job.service_traces.append(job.stiming.pop('vram_trace'))
        job.ksplit_decision = job.stiming.get('ksplit', job.ksplit_decision)
        job.fields.append(job.host[..., :count])
        job.stats_all.append(job.stats)
        job.timings.append(job.stiming)
        job.source_seconds += sec
        job.query_seconds += qsec
        for batch in job.prepared_table['batches']:
            batch.pop('bundle')
            batch.pop('flat')
            batch['captured'] = [[{'d': raw['d']} for raw in raws] for raws in batch['captured']]
        job.s.kernel.constants.clear()
        job.pending_tables.append((job.s, job.prepared_table, job.collected, count, job.index))
        phase(
            job.name,
            'source_done',
            chunk=job.index,
            evals=job.stats['evaluations'],
            seconds=sec,
            **chunk_summary(job.stats, job.stiming['solve_worker']),
        )
        phase(job.name, 'table_transport_done', chunk=job.index, seconds=qsec)
        return True
    phase(job.name, 'cpu_prepare_done', chunk=job.index, cuda_imported='cupy' in sys.modules)
    job.token = GpuToken(job.pipeline_lock, job.name, phase)
    job.token.acquire()
    configure(default_root(a.work_root), a.backend, a.block_size)
    promote_solver(job.s, job.geometry, a.backend, a.block_size, a.threads)
    job.xp = job.s.kernel.xp
    if job.xp is not np:
        from sosfuse2.backends.device_profile import apply_pool_limit

        apply_pool_limit(job.xp)
    return False


def _solve_wavelength_chunks(
    job, a, np, gm, Solver, build_cells, cache_complete, column_ready, ensure_column, configure, default_root
):
    phase(job.name, 'geometry_start')
    job.geometry_start = time.monotonic()
    # Independent wavelengths: use unchanged FW=15 kernel, padding the final chunk.
    job.chunks = [job.wl] if len(job.wl) in (1, 15) else [job.wl[i : i + 15] for i in range(0, len(job.wl), 15)]
    from sosfuse2.atmosphere.density import chunk_metadata

    job.table_density = None
    from .pipeline_execution import geometry_buffers
    from concurrent.futures import ThreadPoolExecutor

    share = job.persistent and a.backend == 'cupy' and not a.reduced and len(job.chunks) > 1
    with contextlib.ExitStack() as stack:
        job.geometry_buffers = stack.enter_context(geometry_buffers()) if share else None
        pool = ThreadPoolExecutor(max_workers=1) if share else None
        if pool is not None:
            stack.callback(pool.shutdown, wait=True, cancel_futures=True)
        pending = None

        def prepare_next():
            nonlocal pending
            index = job.index + 1
            if index < len(job.chunks):
                pending = pool.submit(_build_chunk_inputs, job, a, np, index, job.chunks[index], prepare_table=True)

        for job.index, job.chunk in enumerate(job.chunks):
            count = _prepare_wavelength_chunk(job, a, np, chunk_metadata, pending)
            pending = None
            _prepare_chunk_geometry(job, a, gm, cache_complete, build_cells, column_ready, ensure_column)
            if job.pipeline_lock and _prepare_pipeline_chunk(
                job, a, np, count, configure, default_root, prepare_next if pool is not None else None
            ):
                continue
            _solve_chunk_source(job, a, np, Solver, count)
            _collect_chunk_table(job, a, np, count)


def _finish_chunk_tables(job, np):
    if job.ksplit_decision is not None:
        phase(job.name, 'ksplit', decision=job.ksplit_decision)
    if job.pipeline_lock:
        phase(job.name, 'cpu_finalize_start')
        pending = iter(job.pending_tables)
        for job.table_s, job.prepared_table, job.collected, count, job.index in pending:
            t = time.monotonic()
            job.table = job.replay_tables(job.table_s, job.prepared_table, job.collected)
            sec = time.monotonic() - t
            job.query_seconds += sec
            job.tables.append(job.table[..., :count])
            if job.cfg.get('export_first_order'):
                job.first_tables.append(job.table_s.first_order_table[..., :count])
            phase(job.name, 'table_done', chunk=job.index, seconds=sec, deferred_replay=True)
        job.pending_tables.clear()
    job.table = np.concatenate(job.tables, axis=-1)
    if not np.isfinite(job.table).all():
        raise RuntimeError('E_TABLE_FINITE')


def _run_metadata(job, a, peak, first_meta, optics):
    """Snapshot the live solver and geometry before their leases are released."""
    from sosfuse2.atmosphere.stratosphere import computed_metadata

    optical_meta = computed_metadata(job.state, getattr(job.s, 'strato', None))
    from sosfuse2.atmosphere.density import evidence as profile_evidence

    # ②b evidence: compiled/uploaded N, consumed mask, effective f8/f4 SHA, identity, policy,
    # origin. consumed = this process's GPU upload (source_solver.constants_device); persistent runs log it from the service.
    profile_meta = dict(
        host=profile_evidence(getattr(job.s, 'density_optics', None), job.s.kernel.profile_n),
        consumed=getattr(job.s.kernel, 'profile_consumed', None),
        persistent_service=bool(job.persistent and job.pipeline_lock),
        sun_method=getattr(job.s, 'density_sun_method', None),
    )  # ③ 0(b): gl/exact PROFILE columns of sun table+ground, None = no PROFILE
    meta = dict(
        **first_meta,
        **vram_trace.metadata(job.service_traces),
        density=job.table_density,
        profile=profile_meta,
        complete=True,
        contract=CONTRACT_VERSION,
        physics=physics(),
        optics_state=optics,
        **package_info(),
        signature=job.signature,
        id=a.id,
        out_name=job.name,
        parameters=job.state['parameters'],
        parameters_requested=job.state['parameters'],
        parameters_computed=optical_meta['parameters_computed'],
        stratosphere=optical_meta['stratosphere'],
        stratosphere_computed=optical_meta['stratosphere_computed'],
        settings=job.cfg,
        grid=grid_metadata(job.cfg, job.s.geo.h),
        mode='reduced-validation' if a.reduced else 'full',
        shape=list(job.table.shape),
        dtype='float32',
        code_hashes=job.hashes,
        backend=a.backend,
        polarization_model=a.polarization_model,
        anderson_tol=job.stats_all[0]['tolerance'],
        solve_grid=dict(h_km=job.s.geo.h.tolist(), chi_rad=job.s.geo.chi.tolist(), L=job.s.kernel.L),
        source_solve=job.stats_all[0] if len(job.stats_all) == 1 else job.stats_all,
        source_seconds=job.source_seconds,
        query_seconds=job.query_seconds,
        elapsed_seconds=time.monotonic() - job.started,
        geometry=job.geometry.stats,
        timings=job.timings,
        threads=a.threads,
        peak_rss_bytes=peak(),
        attempt=a.attempt,
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        unsupported_parameters=[],
        units='W m^-2 sr^-1 nm^-1',
        folding='mirror azimuth around solar plane',
        table_method=dict(runner='run_gpu_state_v3', table='table_fast.make_table', group=a.group, batch=a.batch),
        spectral_chunks=[len(c) for c in job.chunks],
        ksplit_effective=(job.ksplit_decision or {}).get('effective', 0),
        ksplit=job.ksplit_decision,
        geometry_cache_removed=False,
        reason='pack_not_verified',
        kernel=dict(
            deterministic=job.s.kernel.deterministic,
            requested=job.cfg.get('kernel'),
            effective_flags=list(job.s.kernel.flags),
            block_size=job.s.kernel.block_size,
            launch=getattr(job.s.kernel, 'launch_log', None),
            profile_n=job.s.kernel.profile_n,
            domain=job.s.kernel.domain.identity(),
        ),
    )
    if job.s.kernel.domain.monitors()['record']:
        meta.update(monitor_sets=job.s.kernel.domain.monitors(), convergence_path='extended-top-observer')
    if getattr(job.s, 'tropo', None) is not None:
        meta['troposphere_phase_computed'] = job.s.tropo['meta']
    if getattr(job.s, 'aerosol_profile', None) is not None:
        meta['troposphere_profile_computed'] = job.s.aerosol_profile['meta']
    return meta


def _pack_verified_output(job, a):
    storage.pack_output(job.out, a.work_root / 'pack', job.started, job.signature, verify=verify_pack)
    # layers=2: the common layer is shared by every state; never removed by a worker.
    result = release_geometry(a.geometry_cache, job.leased.cache, True, a.keep_geometry or a.geometry_layers == 2)
    job.released = True
    storage.update_metadata(job.meta_path, result)
    phase(job.name, 'geometry_cleanup', **result)
    # Repack once so archived meta records the actual post-verification deletion result.
    # This attempt's first pack and verification completed; its previews are ready.
    pack = storage.pack_output(job.out, a.work_root / 'pack', job.started, job.signature,
                               verify=verify_pack, reuse_previews=True)
    phase(job.name, 'pack_done', bytes=pack.stat().st_size, **result)
    if job.pipeline_lock:
        phase(job.name, 'cpu_finalize_done')


def _release_run_resources(job, a):
    if job.token is not None:
        # Exceptions must not unlock while this process still owns queued
        # CUDA work. Driver failures remain failures, never successful packs.
        job.solver = job.field = None
        try:
            job.release_device(job.s, job.geometry)
        finally:
            job.token.release()
    if job.leased is not None and not job.released:
        release_geometry(a.geometry_cache, job.leased.cache, False, a.keep_geometry)
    job.stop.set()


@trace_state
def run(a):
    _check_run_options(a)
    _set_run_threads(a)
    import numpy as np
    from sosfuse2.solve import ray_geometry as gm
    from sosfuse2.solve.source_solver import Solver
    from sosfuse2.solve.ray_geometry_cpu import build_cells, cache_complete, column_ready, ensure_column
    from sosfuse2.foundation.host_limits import monitor, peak_rss_bytes as peak

    _set_run_threads(a)
    job = _StateRun()
    job.state, job.cfg = configuration.state_settings(a, resolve=resolve_settings)
    job.name = a.out_name or a.id
    if a.backend == 'cupy':
        _preflight_device(job.state)
    job.signature, job.hashes = signed_contract(job.state, job.cfg, a.backend, a.reduced, a.polarization_model)
    job.out = a.work_root / 'out' / job.name
    job.prepare = a.work_root / 'tmp' / job.name
    job.meta_path = job.out / 'meta.json'
    job.out.mkdir(parents=True, exist_ok=True)
    with storage.output_lock(job.out):
        job.prepare.mkdir(parents=True, exist_ok=True)
        optics = optics_state(job.state)
        storage.check_output(job.out, job.signature, optics)
        _start_run_monitor(job, a, monitor)
        from sosfuse2.backends.compile_cache import configure, default_root

        try:
            _prepare_run_grid(job, a, np, configure, default_root)
            _solve_wavelength_chunks(
                job, a, np, gm, Solver, build_cells, cache_complete, column_ready, ensure_column, configure, default_root
            )
            _finish_chunk_tables(job, np)
            contract = dict(
                state=canonical_state(job.state), anderson_tol=job.stats_all[0]['tolerance'],
                settings=job.cfg, backend=a.backend, polarization_model=a.polarization_model,
            )
            first_meta = storage.write_run_arrays(
                job.out, job.prepare, job.table, job.wl, job.hs, job.el, job.az,
                h=job.s.geo.h, chi=job.s.geo.chi, fields=job.fields, moments=job.moments, order=job.s.kernel.L,
                physical=physics(), optics=optics, contract=contract,
                first_tables=job.first_tables if job.cfg.get('export_first_order') else None,
            )
            atomic_json(job.meta_path, _run_metadata(job, a, peak, first_meta, optics))
            _pack_verified_output(job, a)
        finally:
            _release_run_resources(job, a)


@trace_state
def run_extra(a):
    """M3: table rows at extra solar elevations from a stored source
    (<work-root>/tmp/<source_out_name>/source.npz written by run()); no solve, no geometry cache."""
    from ..atmosphere.components import production_only

    production_only()
    if not shutil.which('zstd'):
        raise RuntimeError('E_ZSTD_MISSING: install zstd (required by pack and verify)')
    _set_run_threads(a)
    import numpy as np
    from sosfuse2.outputs import radiance_table as tf
    from sosfuse2.foundation.host_limits import monitor, peak_rss_bytes as peak

    if a.polarization_model != 'depolarizing':
        raise RuntimeError('E_UNSUPPORTED_POL_MODEL')
    state, cfg = configuration.state_settings(a, resolve=resolve_settings)
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
    with contextlib.ExitStack() as resources:
        resources.enter_context(storage.output_lock(out))
        optics = optics_state(state)
        storage.check_output(out, signature, optics)
        started = time.monotonic()
        stop = threading.Event()
        threading.Thread(target=monitor, args=(stop, a.rss_gb * 1e9), daemon=True).start()
        resources.callback(stop.set)
        from sosfuse2.backends.compile_cache import configure, default_root

        configure(default_root(a.work_root), a.backend, a.block_size)
        os.environ['CUDA_CACHE_PATH'] = str(prepare / 'cuda_driver_cache')
        resources.enter_context(storage.source_lock(src_out))
        phase(name, 'source_reuse_start', source_out_name=a.source_out_name)
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
        source_meta = storage.read_source_metadata(src_out, src, optics, same)
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
        from sosfuse2.atmosphere.stratosphere import computed_metadata

        optical_meta = computed_metadata(state, getattr(s, 'strato', None))

        if any(
            source_meta.get(k) != optical_meta[k]
            for k in ('parameters_requested', 'parameters_computed', 'stratosphere')
        ):
            raise RuntimeError('E_EXTRA_SOURCE: requested/computed optical state differs')
        full = tf.load_source(src, s, wl, settings=cfg, backend=a.backend, full_settings=True)
        source_sha = hashlib.sha256(src.read_bytes()).hexdigest()
        phase(name, 'source_reuse_done', seconds=time.monotonic() - started, source_sha256=source_sha)
        tables = []
        first_tables = []
        query_seconds = 0.0
        start = 0
        from sosfuse2.atmosphere.density import chunk_metadata

        table_density = None
        for index, chunk in enumerate(chunks):
            trace = vram_trace.current.get()
            if trace is not None:
                trace.chunk, trace.stage = index, 'table_transport'
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
            table_density = chunk_metadata(s, chunk, table_density)
            if index == len(chunks) - 1 and source_meta.get('density') != table_density:
                raise RuntimeError('E_EXTRA_SOURCE: density profile differs')
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
            if cfg.get('export_first_order'):
                first_tables.append(s.first_order_table[..., :count])
            phase(name, 'table_done', chunk=index, seconds=sec, rows=len(hs))
            if len(chunks) > 1:
                s.kernel.constants.clear()
                s.kernel.device_constants.clear()
        table = np.concatenate(tables, -1)
        if not np.isfinite(table).all():
            raise RuntimeError('E_TABLE_FINITE')
        anchors = storage.check_anchor_rows(src_out, table, hs)
        # Extra-elevation reruns keep the previous completion until the new table replaces the old one, as before the split.
        first_meta = storage.write_table(out, table, wl, hs, el, az,
                                         first_tables=first_tables if cfg.get('export_first_order') else None)

        meta = dict(
            **first_meta,
            **vram_trace.metadata(),
            density=table_density,
            complete=True,
            contract=CONTRACT_VERSION,
            physics=physics(),
            optics_state=optics,
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
            anderson_tol=cfg['anderson_tol'],
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
        atomic_json(out / 'meta.json', meta)
        pack = storage.pack_output(out, a.work_root / 'pack', started, signature, verify=verify_pack)
        phase(name, 'pack_done', bytes=pack.stat().st_size, geometry_cache_removed=False, reason='no_geometry')


DEFAULT_ID = 'clear_34'  # the state of the README quick start
DEFAULT_WORK_ROOT = 'work'


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--id', default=DEFAULT_ID, help=f'state id in --states (default {DEFAULT_ID})')
    p.add_argument('--out-name')
    p.add_argument('--states', type=Path, default=paths.ASSETS / 'states_v2.json')
    p.add_argument(
        '--work-root',
        type=Path,
        default=Path(DEFAULT_WORK_ROOT),
        help=f'folder of out/, tmp/, pack/ and the caches of the run (default ./{DEFAULT_WORK_ROOT})',
    )
    p.add_argument('--geometry-cache', type=Path)
    p.add_argument(
        '--backend',
        choices=['auto', 'cupy', 'numpy'],
        default='auto',
        help='auto (default): cupy when nvidia-smi names a visible device, numpy on a machine without CUDA',
    )
    p.add_argument(
        '--reduced', action='store_true', help='reduced grid; always on with the numpy backend, except --geometry-only'
    )
    p.add_argument(
        '--threads',
        type=int,
        help='value for OMP_NUM_THREADS and the related thread variables, default 2; given explicitly, it also sets each '
        'unset or blank preparation-pool variable (POOL_WORKER_ENV) to N for the run in every mode, and nonblank values '
        'take precedence. Omitted: the pool variables are left unchanged',
    )
    p.add_argument('--max-eval', type=int, default=64)
    p.add_argument('--rss-gb', type=float)
    p.add_argument(
        '--block-size',
        type=int,
        choices=(128, 256),
        default=None,
        help='256 threads/block (97,872 B shared at 15 wavelengths); default SOSFUSE2_BLOCK_SIZE or 128',
    )
    p.add_argument(
        '--fixed-boxes', action='store_true', help='observer boxes from one fixed Bundle on the device source'
    )
    p.add_argument(
        '--geometry-layers',
        type=int,
        choices=(1, 2),
        default=2,
        help='2 (default): common endpoint layer (endpoint_contract signature, no scale height) + per-height aerosol column; a worker never removes the common layer',
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
    from sosfuse2.atmosphere.density import add_arguments

    add_arguments(p)
    from sosfuse2.backends.flags_env import add_run_options

    add_run_options(p)
    p.add_argument(
        '--stratosphere',
        choices=['on'],
        help='explicitly enable the state stratospheric layer; default honors state.status',
    )
    return p


def main():
    p = build_parser()
    a = p.parse_args()
    a.threads_explicit = a.threads is not None
    if a.threads is None:
        a.threads = 2
    if a.threads < 1:
        p.error('--threads must be a positive integer')
    a.work_root = a.work_root.resolve()
    a.geometry_cache = (a.geometry_cache or a.work_root / 'geometry_cache').resolve()
    from sosfuse2.backends.flags_env import apply_run_options

    try:
        apply_run_options(a, os.environ)  # before the kernel settings, the profile and the caches are read below
    except ValueError as e:
        p.error(str(e))
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
        kernel = kernel_settings(a.block_size)
    except ValueError as e:
        p.error(str(e))
    if a.block_size is None:
        a.block_size = (kernel or {}).get('block_size', 128)
    if a.gpu_lock and (a.geometry_only or a.extra_elevations):
        p.error('E_PIPELINE_MODE: gpu-lock requires ordinary state mode')
    if a.geometry_only:
        a.backend = 'numpy'
    from sosfuse2.backends.flags_env import resolve_backend

    try:
        a.backend, a.reduced, notes = resolve_backend(a.backend, a.reduced, full_numpy=a.geometry_only)
    except ValueError as e:
        p.error(str(e))
    for note in notes:
        print('run_state: ' + note, file=sys.stderr, flush=True)
    if a.backend == 'numpy' and not a.reduced and not a.geometry_only:
        p.error('numpy requires --reduced unless geometry-only')
    if sys.platform == 'darwin' and (a.backend != 'numpy' or not a.reduced):
        p.error('Mac requires numpy + reduced')
    from sosfuse2.foundation.host_limits import default_rss_gb, rss_cap_gb

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
    # An explicit limit holds in every mode (geometry only, extra elevations, solve) for this call only; without one the
    # pool variables are left as they are.
    with _pool_limit(a.threads, os.environ) if a.threads_explicit else contextlib.nullcontext():
        if a.geometry_only:
            geometry_only(a)
        elif a.extra_elevations:
            run_extra(a)
        else:
            run(a)


if __name__ == '__main__':
    main()
