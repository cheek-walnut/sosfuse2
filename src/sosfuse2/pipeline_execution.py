"""CPU-only preparation and an opt-in exclusive GPU execution boundary.

Ephemeral objects stay in their state process. Existing persistent geometry and
output formats are untouched; CuPy is imported only after acquiring the token.
"""

import contextlib
import copy
import fcntl
import hashlib
import os
import time
import traceback
from pathlib import Path

KSPLIT_STATE_VRAM_GB = 5.1  # = pipeline.KSPLIT_STATE_VRAM_GB (the controller stays import-free of this module)


class GpuToken:
    def __init__(self, path, identifier, emit):
        self.path = Path(path)
        self.identifier = identifier
        self.emit = emit
        self.file = None

    def acquire(self):
        self.emit(self.identifier, 'gpu_wait', lock=str(self.path))
        try:
            self.file = self.path.open('a')
            fcntl.flock(self.file, fcntl.LOCK_EX)
        except OSError as exc:
            self.release(emit=False)
            raise RuntimeError('E_PIPELINE_LOCK: cannot acquire GPU token') from exc
        self.emit(self.identifier, 'gpu_acquired', lock=str(self.path))

    def release(self, emit=True):
        if self.file is not None:
            fcntl.flock(self.file, fcntl.LOCK_UN)
            self.file.close()
            self.file = None
            if emit:
                self.emit(self.identifier, 'gpu_released', lock=str(self.path))

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_):
        self.release()


def prepare_bundle(kernel, s, bundle):
    """Compute the same lazy optical sidecars while every array is on the CPU."""
    c = kernel.setup(s, bundle.nq)
    if 'density_optics' in c:
        kernel.density_end_columns(bundle, c)
    if 'strato_ext' in c:
        prepared = bundle.__dict__.pop('prepared_strato', None)
        if prepared is not None:
            from .column_cache import cached_columns

            key = tuple(c['strato_profile'][k] for k in ('center', 'sigma', 'norm'))
            value = cached_columns(kernel, bundle, c, 'strato', lambda: prepared)
            bundle.strato_columns[key] = kernel.xp.asarray(value)
        kernel.strato_end_columns(bundle, c)
    return bundle


def prepare_geometry(s, geometry, reduced):
    from .geometry import Bundle
    import numpy as np

    if reduced:
        geometry.bundle = Bundle([geometry.cell(i) for i in range(s.geo.ns)], np)
        geometry.ground_bundle = Bundle([geometry.cell(i) for i in range(len(s.geo.chi))], np)
        geometry.stats['compact_bytes'] = geometry.bundle.bytes
    else:
        geometry.prepare()
    for bundle in (geometry.bundle, geometry.ground_bundle):
        prepare_bundle(s.kernel, s, bundle)


def promote_bundle(bundle, xp):
    """New device allocation with the exact host bytes and state-private sidecars."""
    result = copy.copy(bundle)
    result.arrays = {k: xp.asarray(v) for k, v in bundle.arrays.items()}
    for attr in ('density_columns', 'strato_columns'):
        if hasattr(bundle, attr):
            setattr(result, attr, {k: xp.asarray(v) for k, v in getattr(bundle, attr).items()})
    return result


def promote_solver(s, geometry, backend, block_size, threads):
    if backend == 'numpy':
        return
    from .fused import Kernel

    old = s.kernel
    from .density import kernel_kwargs

    kernel = Kernel(
        threads,
        old.height,
        old.g,
        backend,
        block_size,
        flags=old.flags,
        L=old.L,
        **kernel_kwargs(getattr(old, 'profile_step_km', 0.5)),
    )  # ②b: same PROFILE_N as the CPU kernel
    # setup() holds NumPy constants only; the state identity is unchanged.
    kernel.constants = old.constants
    s.kernel = kernel
    geometry.s = s
    geometry.bundle = promote_bundle(geometry.bundle, kernel.xp)
    geometry.ground_bundle = promote_bundle(geometry.ground_bundle, kernel.xp)
    kernel.sync()


def captured_queries(s, hs, part, ee, aa, dummy):
    """table_fast._capture raws for the sun elevations hs[part], through host_cache.

    The query packs are pure geometry (observer, sun elevation, view grid, endpoint geometry;
    pack(ground=False) writes no columns), so one copy serves every state of a grid. Replay still
    checks every chunk's directions against these raws (E_REPLAY_DRIFT).
    """
    from .table_fast import _capture
    from . import host_cache
    import numpy as np

    capture = lambda: [_capture(s, hs[j], ee, aa, dummy) for j in part]
    root = getattr(s, 'host_cache_root', None)
    if root is None or not host_cache.enabled():
        return capture()
    from .geometry_contract import identity

    common, signature = identity(s)
    b = s.ctx.b
    key = dict(
        kind='query01',
        common=common,
        observer_km=float(b.sh.OBS),
        radius_km=float(b.R),
        query_n=getattr(s.geo, 'query_n', 24),
        sun_elevation_deg=[float(hs[j]) for j in part],
        view=host_cache.array_hash(np.stack([ee, aa])),
        sources={
            'sosfuse2/' + name: hashlib.sha256((Path(__file__).resolve().parent / name).read_bytes()).hexdigest()
            for name in ('table_fast.py', 'geometry.py', 'pipeline_execution.py')
        },
    )
    computed = []
    started = time.monotonic()

    def event(result):
        import json
        from .column_parallel import event_fields

        print(
            json.dumps(
                dict(
                    event='PHASE',
                    phase='host_cache',
                    **event_fields(),
                    kind='query',
                    result=result,
                    seconds=time.monotonic() - started,
                )
            ),
            flush=True,
        )

    def build():
        captured = capture()
        computed.append(captured)
        flat = [g for raws in captured for g in raws]
        names = [k for k in flat[0] if k != 'nq']
        if any(list(g) != names + ['nq'] or g['nq'] != flat[0]['nq'] for g in flat) or any(
            g[k].shape[1:] != flat[0][k].shape[1:] for g in flat for k in names
        ):
            raise RuntimeError('E_QUERY_CACHE_LAYOUT: packs differ in fields or trailing shapes')
        return dict(
            {k: np.concatenate([g[k] for g in flat]) for k in names},
            query_rays=np.array([len(g['d']) for g in flat], 'i8'),
            query_packs=np.array([len(raws) for raws in captured], 'i8'),
            query_nq=np.array([flat[0]['nq']], 'i8'),
        )

    try:
        z = host_cache.cached(Path(root) / ('endpoints_' + signature) / 'query_cache', key, build)
    except host_cache.CacheUnavailable:
        out = computed[0] if computed else capture()
        event('unavailable')
        return out
    except RuntimeError as exc:
        if computed and str(exc).startswith('E_QUERY_CACHE_LAYOUT'):
            event('layout')
            return computed[0]
        raise
    event('miss' if computed else 'hit')
    if computed:
        return computed[0]
    names = [k for k in z if not k.startswith('query_')]
    cut = np.cumsum(z['query_rays'])[:-1]
    nq = int(z['query_nq'][0])
    parts = {k: np.split(z[k], cut) for k in names}
    flat = [dict({k: parts[k][i] for k in names}, nq=nq) for i in range(len(z['query_rays']))]
    edges = np.r_[0, np.cumsum(z['query_packs'])]
    return [flat[lo:hi] for lo, hi in zip(edges[:-1], edges[1:])]


def prepare_tables(s, hs, el, az, group, batch):
    from .table_fast import _groups
    from .geometry import Bundle
    import numpy as np

    ee, aa = np.meshgrid(el, az, indexing='ij')
    dummy = np.empty(0, 'f4')
    prepared = []
    for b0 in range(0, len(hs), batch):
        part = list(range(b0, min(b0 + batch, len(hs))))
        captured = captured_queries(s, hs, part, ee, aa, dummy)
        flat = [g for raws in captured for g in raws]
        groups, sizes = _groups(flat, group)
        bundle = Bundle(groups, np)
        bundle.sizes = sizes
        prepare_bundle(s.kernel, s, bundle)
        prepared.append(dict(part=part, captured=captured, flat=flat, bundle=bundle))
    return dict(batches=prepared, ee=ee, aa=aa, hs=hs, group=group, shape=(len(hs), len(el), len(az), s.W))


def evaluate_tables(s, field, prepared):
    """GPU transport + completed host collection; replay is deferred to CPU."""
    from .table_fast import _evaluate

    xp = s.kernel.xp
    import numpy as np

    device_field = np.ascontiguousarray(field, 'f4') if xp is np else field
    results = []
    for b in prepared['batches']:
        bundle = None if xp is np else promote_bundle(b['bundle'], xp)
        one = _evaluate(s, b['flat'], device_field, True, prepared['group'], bundle)
        multiple = _evaluate(s, b['flat'], device_field, False, prepared['group'], bundle)
        s.kernel.sync()
        results.append((one, multiple))
        del bundle
    return results


def replay_tables(s, prepared, results):
    from .table_fast import _replay
    import numpy as np

    table = np.empty(prepared['shape'], 'f4')
    dummy = np.empty(0, 'f4')
    for b, (one_values, multiple_values) in zip(prepared['batches'], results):
        offset = 0
        for j, raws in zip(b['part'], b['captured']):
            n = len(raws)
            args = (s, prepared['hs'][j], prepared['ee'], prepared['aa'], dummy, raws)
            one = _replay(*args, one_values[offset : offset + n], True)
            multiple = _replay(*args, multiple_values[offset : offset + n], False)
            offset += n
            spec = (one + multiple)[..., 0].astype('f4')
            if not np.isfinite(spec).all() or (spec < 0).any():
                raise RuntimeError('E_RADIANCE_INVALID: nonfinite/negative')
            table[j] = spec
    return table


def release_device(s, geometry, retain_pool=False):
    """Caller drops Solver/field references first, then frees state allocations."""
    import numpy as np

    kernel = s.kernel
    if kernel.xp is np:
        return
    kernel.sync()
    geometry.bundle = None
    geometry.ground_bundle = None
    kernel.device_constants.clear()
    kernel.constants.clear()
    # Diagnostic Event objects contain no state arrays. Synchronize before the
    # pool release so no queued operation still refers to the buffers.
    kernel.sync()
    if not retain_pool:
        kernel.xp.get_default_memory_pool().free_all_blocks()


def device_payload(s, geometry):
    """Transient array transport for a prepared GPU apply, not a cache schema.

    Accepted solver closures and CPU quadrature modules stay with the producer.
    Only the fields consumed by the prepared Solver.apply path cross the queue.
    All optical sidecars must already exist, so GPU consumers never run CPU
    column integration using the omitted module references.
    """
    from types import SimpleNamespace

    constants = {}
    for (_, nq), original in s.kernel.constants.items():
        c = {k: v for k, v in original.items() if k != 'strato_old'}
        if 'density_optics' in c:
            c['density_optics'] = {k: v for k, v in c['density_optics'].items() if k != 'old'}
        for bundle in (geometry.bundle, geometry.ground_bundle):
            if 'strato_ext' in c:
                key = tuple(c['strato_profile'][k] for k in ('center', 'sigma', 'norm'))
                if key not in getattr(bundle, 'strato_columns', {}):
                    raise RuntimeError('E_PREPARED_STRATO_COLUMNS')
            if 'density_optics' in c:
                from .density import profile_identity

                optics = c['density_optics']
                s.kernel.check_profile(optics, 'payload')
                key = profile_identity(optics)['sha256']  # fused.density_end_columns key
                if key not in getattr(bundle, 'density_columns', {}):
                    raise RuntimeError('E_PREPARED_DENSITY_COLUMNS')
        constants[nq] = c
    fields = {k: getattr(s, k) for k in ('W', 'ext', 'rf', 'af', 'asca')}
    if hasattr(s, 'ksplit_decision'):
        fields['ksplit_decision'] = s.ksplit_decision
    fields['geo'] = SimpleNamespace(ns=s.geo.ns, h=s.geo.h, chi=s.geo.chi)
    fields['strato'] = None if getattr(s, 'strato', None) is None else {k: v for k, v in s.strato.items() if k != 'old'}
    if getattr(s, 'tropo', None) is not None:
        fields['tropo'] = s.tropo
    return dict(
        solver=SimpleNamespace(**fields),
        constants=constants,
        bundle=geometry.bundle,
        ground_bundle=geometry.ground_bundle,
        flags=s.kernel.flags,
        height=s.kernel.height,
        g=s.kernel.g,
        block_size=s.kernel.block_size,
        L=s.kernel.L,
        profile_step_km=getattr(s.kernel, 'profile_step_km', 0.5),
    )


def install_constants(kernel, s, constants):
    """Prepared constants installed without setup() (payload restore, persistent service): each PROFILE table is checked
    against the kernel's PROFILE_N first; setup() checks again on every later read."""
    for c in constants.values():
        if 'density_optics' in c:
            kernel.check_profile(c['density_optics'], 'install')
    kernel.constants = {(id(s), nq): c for nq, c in constants.items()}


def profile_service_event(kernel):
    """Fields of the service 'profile' PHASE. CuPy: the upload record of fused.constants_device. NumPy (no
    upload, uploaded_n None): consumed mask, identity and N from the host table the reference transport reads."""
    from .density import evidence

    consumed = getattr(kernel, 'profile_consumed', None)
    if consumed is None:
        optics = next((c['density_optics'] for c in kernel.constants.values() if 'density_optics' in c), None)
        consumed = evidence(optics, kernel.profile_n)
        if optics is not None:
            kernel.check_profile(optics, 'service')
            consumed.update(uploaded_n=None, consumed_via='host table (NumPy reference transport, FP32 view)')
    return dict(
        where='service', compiled_n=kernel.profile_n, **{k: v for k, v in consumed.items() if k != 'compiled_n'}
    )


def restore_device_payload(payload, compiled):
    """Fresh state arrays/constants; process-local compiled code is shared."""
    from .fused import Kernel
    from .geometry import TransientGeometry

    s = payload['solver']
    from .density import kernel_kwargs

    kernel = Kernel(
        2,
        payload['height'],
        payload['g'],
        'cupy',
        payload['block_size'],
        flags=payload['flags'],
        L=payload['L'],
        **kernel_kwargs(payload.get('profile_step_km', 0.5)),
    )  # consumers re-check N (fused.check_profile)
    kernel.raw, kernel.attributes = compiled
    install_constants(kernel, s, payload['constants'])
    s.kernel = kernel
    geometry = TransientGeometry(s, None)
    geometry.bundle = promote_bundle(payload['bundle'], kernel.xp)
    geometry.ground_bundle = promote_bundle(payload['ground_bundle'], kernel.xp)
    return s, geometry


# Production IPC uses protocol-5 out-of-band buffers in anonymous session files.
# Only small descriptors cross the socket; consumers mmap arrays without pickle
# array copies. The sender retains ownership until the matching reply arrives.
@contextlib.contextmanager
def wire_send(conn, value, root, emit, name):
    import pickle
    import tempfile

    start = time.monotonic()
    with tempfile.TemporaryDirectory(dir=root) as directory:
        offsets = []
        with open(Path(directory) / 'arrays', 'wb') as f:

            def buffer(b):
                view = b.raw()
                offsets.append((f.tell(), view.nbytes))
                f.write(view)

            data = pickle.dumps(value, protocol=5, buffer_callback=buffer)
        (Path(directory) / 'object').write_bytes(data)
        size = sum(n for _, n in offsets) + len(data)
        conn.send((directory, offsets, size))
        emit(name, 'ipc_send', bytes=size, seconds=time.monotonic() - start)
        yield


def wire_recv(conn, emit, name):
    import mmap
    import pickle

    start = time.monotonic()
    message = conn.recv()
    # The service reports a failure as one 'E_PERSISTENT_STATE: <type>: <detail>' string in place of the descriptor;
    # unpacking it as (directory, offsets, size) hid the reason (R3 train0 4090: ENOSPC became a ValueError).
    if isinstance(message, str):
        raise RuntimeError(message if message.startswith('E_') else 'E_PERSISTENT_STATE: ' + message)
    directory, offsets, size = message
    path = Path(directory)
    # ndarray bases retain the mmap until every loaded array is dropped.
    with (path / 'arrays').open('rb') as f:
        mapped = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY) if offsets else None
    value = pickle.loads((path / 'object').read_bytes(), buffers=[memoryview(mapped)[i : i + n] for i, n in offsets])
    emit(name, 'ipc_receive', bytes=size, seconds=time.monotonic() - start)
    return value


class PreparedSegments:
    """NumPy reference boundary quadrature, generated by its CPU state owner."""

    def __init__(self, old, groups):
        self.rows = {
            self.key(g['p'], g['d'], 24): list(old.geometry_segments(g['p'], g['d'], 24)) for g in groups.values()
        }

    @staticmethod
    def key(p, d, n):
        import hashlib

        return (n, hashlib.sha256(p.tobytes() + d.tobytes()).digest())

    def geometry_segments(self, p, d, n):
        return iter(self.rows[self.key(p, d, n)])


def persistent_chunk(s, geometry, tables, a, name, emit):
    """CPU closure owner. Service calls back once for every observer evaluation."""
    import json
    import numpy as np
    from multiprocessing.connection import Client
    from .fused import Solver
    from .geometry import Bundle
    from .table_fast import _groups

    root = Path(os.environ['SOS_PERSISTENT_ROOT'])
    # macOS refuses a Unix-socket connect at once when the accept backlog is full (Linux blocks), which
    # failed test_3_prepare_failure 1 of 3 on 2026-10-05 (P1 ConnectionRefusedError while the service was
    # authenticating P0). Retry refusals for a bounded time; the service also listens with a deeper backlog.
    deadline = time.monotonic() + 30
    while True:
        try:
            conn = Client(str(root / 'socket'), authkey=bytes.fromhex(os.environ['SOS_PERSISTENT_KEY']))
            break
        except ConnectionRefusedError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)
    try:
        if os.environ.get('SOS_PERSISTENT_TEST_PREP_FAIL') == name:
            raise RuntimeError('E_PERSISTENT_PREPARE_INJECTED')
        prepared = {}

        def capture(_s, raws, _field, _first, _group, _bundle):
            groups, sizes = _groups(raws, 32 if a.fixed_boxes else 64)
            bundle = Bundle(groups, np)
            bundle.sizes = sizes
            prepare_bundle(s.kernel, s, bundle)
            prepared.update(raws=raws, bundle=bundle)
            return [np.zeros((len(g['d']), s.W, 4), 'f4') for g in raws]

        observer = Solver(s, geometry, prepared=True, box_evaluate=capture)
        observer.boxes_fixed(np.empty(0, 'f4'))
        payload = device_payload(s, geometry)
        if a.backend == 'numpy':
            for (_, nq), original in s.kernel.constants.items():
                target = payload['constants'][nq]
                if 'strato_old' in original:
                    target['strato_old'] = PreparedSegments(original['strato_old'], geometry.groups)
                if 'density_optics' in original:
                    target['density_optics']['old'] = PreparedSegments(
                        original['density_optics']['old'], geometry.groups
                    )
        payload.update(groups=geometry.groups if a.backend == 'numpy' else {}, tables=tables, observer=prepared)
        emit(name, 'cpu_prepare_done', cuda_imported='cupy' in __import__('sys').modules)
        emit(name, 'gpu_wait')
        conn.send((name, a.max_eval, a.block_size, a.threads))
        with wire_send(conn, payload, root, emit, name):
            while True:
                message = conn.recv()
                if message == 'observer':
                    values = wire_recv(conn, emit, name)
                    observer.box_evaluate = lambda *_: values
                    delay = float(os.environ.get('SOS_PERSISTENT_TEST_OBSERVER_DELAY', '0'))
                    if delay:
                        time.sleep(delay)
                    result = observer.boxes_fixed(np.empty(0, 'f4'))
                    conn.send(result)
                elif message == 'done':
                    result = wire_recv(conn, emit, name)
                    conn.send('ack')
                    return result
                else:
                    raise RuntimeError(
                        message
                        if isinstance(message, str) and message.startswith('E_')
                        else 'E_PERSISTENT_STATE: ' + str(message)
                    )
    except (EOFError, ConnectionError, OSError) as exc:
        raise RuntimeError('E_PERSISTENT_SERVICE_LOST') from exc
    finally:
        conn.close()


def persistent_service(root, backend, lock, active_limit, rss_gb):
    """One process/context; threads yield the compute gate only for CPU replay.

    GPU arithmetic is serialized between callbacks, avoiding concurrent scratch
    peaks and RawKernel compilation races. Live states remain device-resident.
    """
    import json
    import sys
    import threading
    from multiprocessing.connection import Listener
    from . import timers

    class LocalTimers:
        def __init__(self):
            self.local = threading.local()

        def __getattr__(self, key):
            if not hasattr(self.local, 'value'):
                self.local.value = timers.Timers()
            return getattr(self.local.value, key)

    # Install before importing any module that binds TIMERS/stage by value.
    timers.TIMERS = LocalTimers()
    timers.stage = lambda *a, **kw: timers.TIMERS.stage(*a, **kw)
    from .fused import Kernel, Solver, ksplit_setting
    from .density import kernel_kwargs
    from .geometry import TransientGeometry
    from .table_fast import _evaluate
    from .run_fuse import monitor
    from .host_limits import rss_cap_gb
    from .compile_cache import configure
    from .run_gpu_state_v3 import ANDERSON_TOLERANCE  # same solve tolerance as the gpu-lock path (byte parity)

    if not 0 < rss_gb <= rss_cap_gb():
        raise RuntimeError('E_PIPELINE_RSS_BUDGET')
    stop = threading.Event()
    threading.Thread(target=monitor, args=(stop, rss_gb * 1e9), daemon=True).start()
    root = Path(root)
    output_lock = threading.Lock()

    def emit(name, phase, **data):
        with output_lock:
            print(
                json.dumps(
                    dict(
                        event='PHASE',
                        id=name,
                        phase=phase,
                        timestamp_ns=time.time_ns(),
                        monotonic_ns=time.monotonic_ns(),
                        pid=os.getpid(),
                        **data,
                    )
                ),
                flush=True,
            )

    gate = threading.Condition(threading.RLock())
    reservations = {}
    allocations = {}  # actual live pool bytes per resident state, at each gate handoff
    compiled = {}
    token = GpuToken(lock, 'GPU_SERVICE', emit)
    xp = None
    pool_limit = None
    ksplit_waiting = set()  # states whose KSPLIT allocation waits for room (fused.vprod_columns)

    def handle(conn):
        nonlocal xp, pool_limit
        name = 'UNKNOWN'
        admitted = False
        s = geometry = solver = field = None
        observer_bundle = bundle = values = kernel = None
        start = time.monotonic()
        cpu_start = time.thread_time()
        wait_seconds = 0.0
        try:
            name, max_eval, block_size, threads = conn.recv()
            payload = wire_recv(conn, emit, name)
            # Reservation is deliberately conservative; no active state is evicted.
            # Includes resident geometry/constants plus Anderson vectors and scratch.
            import pickle

            arrays = []
            pickle.dumps(payload, protocol=5, buffer_callback=lambda b: arrays.append(b.raw().nbytes))
            state = payload['solver']
            field_bytes = state.geo.ns * (1132 if state.strato is not None else 571) * state.W * 4
            # Measured on a 4090 (2026-10-03): two full-grid strato states resident together used 7.5 GB
            # (payload arrays 1.2 GB + field 0.1 GB each), i.e. about 3.7 GB per state at peak.
            required = sum(arrays) + 30 * field_bytes
            # KSPLIT (default on CuPy, fused.ksplit_setting) sizes its columns, plans and aerosol buffer from free VRAM. With
            # several resident states it stays out of the other states' reservations (ksplit_room) and adds what it holds to
            # its own (ksplit_hold), so a later state is admitted only into memory nobody claims.
            ksplit = (
                active_limit > 1
                and backend == 'cupy'  # one resident state keeps sizing from free VRAM as before
                and ksplit_setting(backend, payload['flags'], payload['L']) in ('1', '2', '3')
            )
            # Admit a KSPLIT state only with room for its smallest aerosol option too, and keep that floor reserved: base
            # reservations alone let two states in whose KSPLIT then fit beside neither (L4 2026-10-26: base 7.64 GB each,
            # 4.7 GB left < 5.07 GB minimum, both waited, E_KSPLIT_MEMORY; pipeline.KSPLIT_STATE_VRAM_GB).
            floor = int(KSPLIT_STATE_VRAM_GB * 1e9) if ksplit else 0
            with gate:
                while True:
                    if not reservations and token.file is None:
                        token.acquire()
                    if xp is None:
                        configure(root / 'nvrtc_cache', backend, block_size)
                        prototype = Kernel(
                            threads,
                            payload['height'],
                            payload['g'],
                            backend,
                            block_size,
                            flags=payload['flags'],
                            L=payload['L'],
                            **kernel_kwargs(payload.get('profile_step_km', 0.5)),
                        )
                        xp = prototype.xp
                        if backend == 'cupy':
                            free, _ = xp.cuda.runtime.memGetInfo()
                            if free < 4 * 1024**3:
                                raise RuntimeError('E_VRAM_FREE')
                            pool_limit = min(20 * 1024**3, int(free) - 1024**3)
                            xp.get_default_memory_pool().set_limit(size=pool_limit)
                    available = float('inf')
                    if backend == 'cupy':
                        pool = xp.get_default_memory_pool()
                        free, _ = xp.cuda.runtime.memGetInfo()
                        capacity = min(pool_limit, int(free) + pool.total_bytes() - 1024**3)
                        available = capacity - sum(reservations.values())
                    # A lone state's unsplit base may fit while its KSPLIT floor
                    # does not. Admit it for the exact preflight/fallback check.
                    if len(reservations) < active_limit and (
                        required + floor <= available or not reservations and required <= available
                    ):
                        break
                    if not reservations:
                        raise RuntimeError('E_PERSISTENT_VRAM_ADMISSION')
                    emit(name, 'vram_wait', required_bytes=required + floor, available_bytes=available)
                    gate.wait(0.1)
                reservations[name] = required + floor
                admitted = True
                start = time.monotonic()
                emit(name, 'gpu_acquired', reserved_bytes=required + floor)
                kernel = Kernel(
                    threads,
                    payload['height'],
                    payload['g'],
                    backend,
                    block_size,
                    flags=payload['flags'],
                    L=payload['L'],
                    **kernel_kwargs(payload.get('profile_step_km', 0.5)),
                )

                def account():
                    if backend == 'cupy':
                        allocations[name] = max(
                            0,
                            int(xp.get_default_memory_pool().used_bytes())
                            - sum(v for k, v in allocations.items() if k != name),
                        )

                kernel.ksplit_other_bytes = lambda: sum(v for k, v in allocations.items() if k != name)
                kernel.ksplit_emit = lambda decision: emit(name, 'KSPLIT_FALLBACK', **decision)
                if ksplit:

                    def ksplit_room(name=name, base=required):
                        # None alone (size from free VRAM as before); else the pool not reserved by the other states
                        others = sum(v for k, v in reservations.items() if k != name)
                        if not others:
                            return None
                        pool = xp.get_default_memory_pool()
                        free_now, _ = xp.cuda.runtime.memGetInfo()
                        return min(pool_limit, int(free_now) + pool.total_bytes() - 1024**3) - others - base

                    def ksplit_hold(held, name=name, base=required):
                        reservations[name] = base + (
                            max(int(held), floor, getattr(kernel, 'ksplit_planned_bytes', 0)) if kernel.ksplit else 0
                        )
                        emit(name, 'ksplit_reserved', reserved_bytes=reservations[name], ksplit_bytes=int(held))

                    def ksplit_wait(name=name):
                        # too little room for KSPLIT beside the others: wait for a release unless every other state waits too
                        others = [k for k in reservations if k != name]
                        if not others or all(k in ksplit_waiting for k in others):
                            return False
                        account()
                        ksplit_waiting.add(name)
                        emit(name, 'ksplit_wait', waiting_for=others)
                        try:
                            gate.wait(0.5)
                        finally:
                            ksplit_waiting.discard(name)
                        return True

                    kernel.ksplit_room, kernel.ksplit_hold, kernel.ksplit_wait = ksplit_room, ksplit_hold, ksplit_wait
                # RawKernel keys include flags, W, L and PROFILE_N (fused.kernel_key), but block size is per Kernel; attributes
                # are keyed without L (with a _profileN suffix for N != 161). Consumers re-check compiled/optics/table N.
                kernel.raw, kernel.attributes = compiled.setdefault((block_size, payload['L']), ({}, {}))
                s = payload['solver']
                s.kernel = kernel
                install_constants(kernel, s, payload['constants'])
                geometry = TransientGeometry(s, None)
                geometry.groups = payload['groups']
                geometry.bundle = promote_bundle(payload['bundle'], xp)
                geometry.ground_bundle = promote_bundle(payload['ground_bundle'], xp)
                rays = payload['observer']['raws']
                observer_bundle = promote_bundle(payload['observer']['bundle'], xp)
                emit(
                    name,
                    'service_setup_done',
                    seconds=time.monotonic() - start,
                    cpu_seconds=time.thread_time() - cpu_start,
                )
                sequence = 0
                current = None

                def begin(kind):
                    nonlocal sequence, current
                    sequence += 1
                    current = (sequence, kind, time.monotonic(), time.thread_time())
                    emit(name, 'gpu_piece_start', piece=sequence, kind=kind)

                def end():
                    nonlocal current
                    if current:
                        kernel.sync()
                        seq, kind, wall, cpu = current
                        emit(
                            name,
                            'gpu_piece_done',
                            piece=seq,
                            kind=kind,
                            seconds=time.monotonic() - wall,
                            cpu_seconds=time.thread_time() - cpu,
                        )
                        current = None

                def observer(field):
                    nonlocal wait_seconds
                    end()
                    begin('observer_transport')
                    values = _evaluate(s, rays, field, False, 32, observer_bundle)
                    end()
                    conn.send('observer')
                    # Other state threads can perform first-light/apply/table here.
                    before = time.monotonic()
                    account()
                    gate.release()
                    try:
                        with wire_send(conn, values, root, emit, name):
                            result = conn.recv()
                    finally:
                        gate.acquire()
                        wait_seconds += time.monotonic() - before
                    begin('sos_iteration')
                    return result

                class ServiceSolver(Solver):
                    def initialize(self):
                        if self.first_moments is not None:
                            return
                        end()
                        begin('first_light')
                        super().initialize()
                        end()
                        begin('sos_iteration')

                solver = ServiceSolver(s, geometry, prepared=True, observer=observer)
                solve_start = time.monotonic()
                field, stats = solver.solve(max_eval, ANDERSON_TOLERANCE)
                end()
                # ②b: what this service kernel compiled and consumed (pipeline.py collects it into pipeline_summary).
                emit(name, 'profile', **profile_service_event(kernel))
                solve_seconds = time.monotonic() - solve_start
                begin('table_transport')
                table_start = time.monotonic()
                collected = evaluate_tables(s, field, payload['tables'])
                table_seconds = time.monotonic() - table_start
                result = (kernel.host(field), stats, solver.timings, collected, solve_seconds, table_seconds)
                end()
                conn.send('done')
                with wire_send(conn, result, root, emit, name):
                    conn.recv()
                del observer_bundle, result
        except BaseException as exc:
            # The traceback names the failing call (str(exc) alone hid the KSPLIT=3 rig-profile c0+None).
            emit(
                name,
                'persistent_error',
                reason='E_PERSISTENT_STATE',
                detail=f'{type(exc).__name__}: {exc}',
                traceback=traceback.format_exc(limit=12),
            )
            try:
                conn.send(f'E_PERSISTENT_STATE: {type(exc).__name__}: {exc}')
            except (OSError, EOFError):
                pass
        finally:
            with gate:
                solver = field = None
                if s is not None:
                    release_device(s, geometry, retain_pool=True)
                    # First-light caches also own state-private device arrays.
                    s.kernel.__dict__.pop('_fl_gtab', None)
                    s.kernel.__dict__.pop('_fl_segoff', None)
                observer_bundle = bundle = values = kernel = None
                s = geometry = None
                if admitted:
                    reservations.pop(name, None)
                    allocations.pop(name, None)
                    emit(
                        name,
                        'gpu_released',
                        seconds=time.monotonic() - start,
                        roundtrip_seconds=wait_seconds,
                        cpu_seconds=time.thread_time() - cpu_start,
                    )
                if not reservations:
                    token.release()
                gate.notify_all()
            conn.close()

    listener = Listener(str(root / 'socket'), backlog=64, authkey=bytes.fromhex(os.environ['SOS_PERSISTENT_KEY']))
    (root / 'ready').write_text(str(os.getpid()))
    emit('GPU_SERVICE', 'gpu_service_ready')
    from multiprocessing import AuthenticationError

    while True:
        try:
            conn = listener.accept()
        except (OSError, EOFError, AuthenticationError) as exc:
            # A worker that dies inside the handshake must not take the service (and every live state) down.
            emit('GPU_SERVICE', 'accept_failed', reason='E_PERSISTENT_ACCEPT', detail=f'{type(exc).__name__}: {exc}')
            continue
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


if __name__ == '__main__':
    import sys

    persistent_service(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]))
