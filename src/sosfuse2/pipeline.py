#!/usr/bin/env python3
"""Bounded, independent state processes sharing the runner's GPU token.

The controller owns no CUDA objects and imports only the standard library.
--runner is a subprocess testing seam; exit zero delegates pack verification to
the production runner, and is not independent authentication of arbitrary jobs.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from sosfuse2 import paths
from sosfuse2.host_limits import RIG_MIN_RAM_GB, default_rss_gb, default_workers, physical_ram_gb, rss_cap_gb

THREAD_ENV = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'BLIS_NUM_THREADS',
)
# Process pools of one state's CPU preparation (column_parallel, sun_parallel, ground_parallel, geometry_parallel).
# Unset, each pool takes every usable CPU, so W states prepared at once started W x cores processes
# (eda 2026-10-21: 4 states x 12 workers, solar table 52 s).
PREP_WORKER_ENV = (
    'SOSFUSE2_COLUMN_WORKERS',
    'SOSFUSE2_SUN_WORKERS',
    'SOSFUSE2_GROUND_WORKERS',
    'SOSFUSE2_GEOMETRY_WORKERS',
)
SAFE_ID = re.compile(r'[A-Za-z0-9_-]+\Z')
SAMPLE_INTERVAL = 0.1
TERMINATE_GRACE = 2.0


def clocks():
    return dict(timestamp_ns=time.time_ns(), monotonic_ns=time.monotonic_ns())


CORES_LEFT_FREE = 4  # for the resident GPU service, the queue, telemetry and the OS
WORKER_RSS_GB = 3.0  # one state's process, measured 1.9-2.6 GB on the rigs of 2026-10-03 and 10-05
RAM_SHARE = 0.8


def slot_workers(states, persistent):
    """States a memory-budgeted GPU slot holds (device profile slot_pss_gb, e.g. eda 5 GB): a persistent queue pays
    one service plus a worker per state, separate runners a full state process each. None if the profile sets no
    slot budget (rig). eda 2026-10-07 p1: service PSS 2.99 GB, workers 1.46 and 1.01 GB, so 2 workers = 5.47 GB."""
    from sosfuse2.device_profile import select

    name, _ = gpu_memory()
    _, p, _ = select(name)
    slot = p.get('slot_pss_gb')
    if not slot:
        return None
    if persistent:
        n = int((slot - p['service_pss_gb']) // p['worker_pss_gb'])
    else:
        n = int(slot // p['state_process_pss_gb'])
    return max(1, min(states, n))


def auto_workers(states, threads, backend='numpy', persistent=False):
    """States prepared at once when --workers is not given: as many as the cores read right now allow with
    [CORES_LEFT_FREE] kept back, never more than the states or than [RAM_SHARE] of the RAM holds. A state's CPU
    preparation is single-threaded, so the cores are filled with states, not with threads (2026-10-05: 8 threads on one
    state at a time used 1.08 of 64 cores and left the GPU idle 43 % of the queue). The Mac and eda keep 2."""
    if backend == 'cupy':
        slot = slot_workers(states, persistent)
        if slot is not None:
            return max(1, min(slot, cpu_budget() // threads))
    if rss_cap_gb() < RIG_MIN_RAM_GB:  # Mac, eda and the mid-size VMs keep 2
        return max(1, min(states, 2, cpu_budget() // threads))
    by_cores = max(1, (cpu_budget() - CORES_LEFT_FREE) // threads)
    by_ram = max(1, int(physical_ram_gb() * RAM_SHARE // WORKER_RSS_GB))
    return max(1, min(states, by_cores, by_ram))


VRAM_SHARE = 0.85  # of the GPU memory free at queue start; the rest for the context, LUTs and fragmentation
# A KSPLIT state also holds its aerosol chunk buffers: the smallest option for the product grid is 5.07 GB (L4
# 2026-10-26 repjob: 5 resident states on 22 GB left 3.83 GB, 35 of 39 failed E_KSPLIT_MEMORY while every state waited).
# B2 SINV_F=7 stores 28 bytes per point; retain this reservation until the 4090 pilot measures it.
KSPLIT_STATE_VRAM_GB = 5.1


def gpu_memory():
    """(name, free GB) of the visible GPU from nvidia-smi; the controller never initializes CUDA itself."""
    import subprocess

    dev = (os.environ.get('CUDA_VISIBLE_DEVICES') or '0').split(',')[0]
    try:
        out = (
            subprocess.run(
                ['nvidia-smi', '-i', dev, '--query-gpu=name,memory.free', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        name, free = [x.strip() for x in out.rsplit(',', 1)]
        return name, float(free) / 1024.0
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None, None


def auto_gpu_active(workers, backend):
    """Persistent-service states resident on the GPU at once when --gpu-active is not given: as many as [VRAM_SHARE]
    of the free GPU memory holds at the device profile's state_vram_gb (configs/devices.json: eda 4.5, rig 3.7),
    never more than the workers, plus [KSPLIT_STATE_VRAM_GB] unless SOSFUSE2_KSPLIT=0. The NumPy path keeps the old
    default 2. (2026-10-03 rig: 3 on a 4090 ran the GPU 94 %; a 10 GB 3080 holds 1.)"""
    if backend != 'cupy':
        return max(1, min(workers, 2))
    from sosfuse2.device_profile import select

    name, free = gpu_memory()
    _, profile, _ = select(name)
    per = profile.get('state_vram_gb') or 3.7
    if os.environ.get('SOSFUSE2_KSPLIT', '').strip() != '0':  # KSPLIT is the CuPy default (fused.ksplit_setting)
        per += KSPLIT_STATE_VRAM_GB
    if free is None:
        return 1
    return max(1, min(workers, int(free * VRAM_SHARE // per)))


def prep_workers(workers, threads, host_cores, environ, column_budget=None):
    """Per-state pool sizes for the PREP_WORKER_ENV names the caller left unset: max(1, host_cores // workers), and
    the column builds max(--threads, column_budget // workers) (its help; column_budget = cpu_budget(), which keeps the
    Mac at 4: the RSS guard counts the state process only, not these pool children).
    A name with a nonempty value is kept as given (each module's workers() treats empty as unset the same way).
    Pool sizes preserve prepared values within rtol 1e-6 (parallel module tests).
    Returns (per_state, explicit names, {name: value} to export)."""
    per_state = max(1, host_cores // max(1, workers))
    column = max(threads, (host_cores if column_budget is None else column_budget) // max(1, workers))
    explicit = [k for k in PREP_WORKER_ENV if environ.get(k, '').strip()]
    updates = {
        k: str(column if k == 'SOSFUSE2_COLUMN_WORKERS' else per_state) for k in PREP_WORKER_ENV if k not in explicit
    }
    return per_state, explicit, updates


def cpu_budget():
    if sys.platform == 'darwin':
        return min(4, os.cpu_count() or 1)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


class PipelineParser(argparse.ArgumentParser):
    def error(self, message):
        super().error(message if 'E_PIPELINE_' in message else 'E_PIPELINE_ARGUMENTS: ' + message)


def parser():
    p = PipelineParser(description=__doc__)
    p.add_argument('--states', type=Path, required=True)
    p.add_argument('--ids', help='comma list in queue order; required unless --plan')
    p.add_argument(
        '--plan', type=Path, help='plan_queue.py output: run the --plan-gpu list in its order (excludes --ids)'
    )
    p.add_argument('--plan-gpu', type=int, default=0, help='which GPU list of --plan (default 0)')
    p.add_argument('--work-root', type=Path)
    p.add_argument('--column-cache', type=Path, default=os.environ.get('SOSFUSE2_COLUMN_CACHE'))
    p.add_argument('--column-cache-plan', action='store_true')
    p.add_argument('--geometry-cache', type=Path)
    p.add_argument(
        '--workers', default='auto', help="States prepared at once; 'auto' fills the host's cores ([auto_workers])"
    )
    p.add_argument(
        '--threads',
        type=int,
        default=1,
        help='Per state; its CPU preparation is single-threaded (measured 2026-10-05) '
        'except the cold column builds, which use this many processes (SOSFUSE2_COLUMN_WORKERS)',
    )
    p.add_argument(
        '--rss-gb', type=float, help='Per state process; default from the host RAM and --workers ([default_rss_gb])'
    )
    p.add_argument('--gpu-lock', type=Path)
    p.add_argument('--persistent', action='store_true', help='Resident service with interleaved state solves')
    p.add_argument(
        '--gpu-active',
        default='auto',
        help="Device-resident states; 'auto' from free VRAM and the device profile ([auto_gpu_active])",
    )
    p.add_argument('--legacy', action='store_true', help='Sequential baseline without GPU token; requires workers=1')
    p.add_argument(
        '--runner',
        type=Path,
        default=None,
        help='Python runner script path (default: python -m sosfuse2.run_gpu_state_v3); public subprocess testing seam',
    )
    p.add_argument('--backend', choices=('numpy', 'cupy'), default='cupy')
    p.add_argument('--reduced', action='store_true')
    p.add_argument('--block-size', type=int, choices=(128, 256))
    p.add_argument('--max-eval', type=int, default=64)
    p.add_argument('--group', type=int, default=32)
    p.add_argument('--batch', type=int, default=25)
    p.add_argument('--fixed-boxes', action='store_true')
    p.add_argument('--settings-json', type=Path)
    p.add_argument('--geometry-layers', type=int, choices=(1, 2), default=2)
    p.add_argument('--keep-geometry', action='store_true')
    p.add_argument('--attempt', type=int, default=1)
    p.add_argument('--polarization-model', choices=('depolarizing', 'preserving'), default='depolarizing')
    p.add_argument('--stratosphere', choices=('on',))
    # Keep preflight free of numpy/CUDA; these choices match density.add_arguments.
    p.add_argument('--density-profile', choices=('exp8', 'usstd76'))
    p.add_argument('--ozone-profile', choices=('triangle', 'usstd76'))
    p.add_argument('--rayleigh-column', choices=('keep', 'usstd'))
    return p


def validate(a):
    """Complete read-only preflight before starting any state process."""
    if a.work_root is None:
        raise ValueError('E_PIPELINE_ARGUMENTS: --work-root required')
    ids = a.ids.split(',')
    if not ids or any(not SAFE_ID.fullmatch(i) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('E_PIPELINE_IDS: nonempty, unique safe IDs required')
    a.workers = auto_workers(len(ids), a.threads, a.backend, a.persistent) if a.workers == 'auto' else int(a.workers)
    if a.workers < 1 or not 1 <= a.threads <= (os.cpu_count() or 1) or a.workers * a.threads > cpu_budget():
        raise ValueError(
            'E_PIPELINE_CPU_BUDGET: workers*threads exceeds available CPU budget or invalid per-child threads'
        )
    cap = rss_cap_gb()
    if a.rss_gb is None:
        a.rss_gb = default_rss_gb(a.workers + a.persistent)  # the service is one more process
    if not math.isfinite(a.rss_gb) or not 0 < a.rss_gb <= cap:
        raise ValueError('E_PIPELINE_RSS_BUDGET: finite 0 < rss-gb <= %g required on this host' % cap)
    if a.max_eval < 2 or a.group < 1 or a.batch < 1 or a.attempt < 1:
        raise ValueError('E_PIPELINE_BUDGET: invalid runner budget')
    a.gpu_active = auto_gpu_active(a.workers, a.backend) if a.gpu_active == 'auto' else int(a.gpu_active)
    if a.gpu_active < 1 or (a.persistent and a.legacy):
        raise ValueError('E_PIPELINE_MODE: positive gpu-active; persistent excludes legacy')
    if a.legacy and (a.workers != 1 or a.gpu_lock is not None):
        raise ValueError('E_PIPELINE_MODE: legacy requires workers=1 and no gpu-lock')
    if a.backend == 'numpy' and not a.reduced:
        raise ValueError('E_PIPELINE_MODE: numpy requires reduced')
    if sys.platform == 'darwin' and (a.backend != 'numpy' or not a.reduced):
        raise ValueError('E_PIPELINE_MODE: Mac requires numpy + reduced')
    try:
        doc = json.loads(a.states.read_text())
        states = list(doc['states'])
        if doc.get('parent'):
            inherited = json.loads((a.states.parent / doc['parent']).read_text())['states']
            own = {s['id'] for s in states}
            states.extend(s for s in inherited if s['id'] not in own)
        for i in ids:
            if sum(s['id'] == i and bool(s.get('runnable')) for s in states) != 1:
                raise ValueError('E_PIPELINE_STATE_ID: exactly one runnable state required for ' + i)
    except (OSError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as e:
        raise ValueError('E_PIPELINE_STATES_INPUT: invalid states or parent input') from e
    if a.backend == 'cupy':
        # Same first-light split rule as the runner, for every state before any process starts (eda: C2 8.6 s launches).
        from sosfuse2.device_profile import first_light_split_error, select, smi_device_name

        pname, profile, _ = select(smi_device_name())
        flags = [f for f in os.environ.get('SOSFUSE2_KERNEL_FLAGS', '').replace(' ', '').split(',') if f]
        for i in ids:
            st = next(s for s in states if s['id'] == i and s.get('runnable'))
            layer = st.get('stratosphere') if isinstance(st.get('stratosphere'), dict) else {}
            on = a.stratosphere == 'on' or layer.get('status') == 'enabled'
            try:
                tau = float(st.get('parameters', st).get('stratosphere_tau550', 0.0) or 0.0)
            except (TypeError, ValueError):
                tau = 0.0
            err = first_light_split_error(flags, on and tau > 0, pname, profile)
            if err:
                raise ValueError(err.replace('E_LAUNCH_BUDGET_CONFIG', 'E_LAUNCH_BUDGET_CONFIG ' + i, 1))
    if a.runner is not None and not a.runner.is_file():
        raise ValueError('E_PIPELINE_RUNNER: runner must be a Python file')
    if a.settings_json is not None and not a.settings_json.is_file():
        raise ValueError('E_PIPELINE_SETTINGS: settings-json file does not exist')
    a.ids = ids
    a.states = a.states.resolve()
    a.work_root = a.work_root.resolve()
    a.geometry_cache = (a.geometry_cache or a.work_root / 'geometry_cache').resolve()
    # Column cache is on by default for full-grid runs; '--column-cache off' disables it.
    if a.column_cache is not None and str(a.column_cache) == 'off':
        a.column_cache = None
    elif a.column_cache is None and not a.reduced:
        a.column_cache = a.geometry_cache / 'column_cache'
    if a.runner is not None:
        a.runner = a.runner.resolve()
    if a.settings_json is not None:
        a.settings_json = a.settings_json.resolve()
    a.gpu_lock = None if a.legacy else (a.gpu_lock or Path(tempfile.gettempdir()) / 'sosfuse2_gpu.lock').resolve()
    return a


def child_command(a, state_id):
    cmd = [sys.executable, '-B', str(a.runner)] if a.runner is not None else paths.module_command('run_gpu_state_v3')
    cmd += ['--id', state_id, '--out-name', state_id]
    for key in (
        'states',
        'work_root',
        'geometry_cache',
        'threads',
        'rss_gb',
        'backend',
        'block_size',
        'max_eval',
        'group',
        'batch',
        'settings_json',
        'geometry_layers',
        'attempt',
        'polarization_model',
        'stratosphere',
        'density_profile',
        'ozone_profile',
        'rayleigh_column',
        'gpu_lock',
        'column_cache',
    ):
        value = getattr(a, key)
        if value is not None:
            cmd.extend(['--' + key.replace('_', '-'), str(value)])
    for key in ('reduced', 'fixed_boxes', 'keep_geometry'):
        if getattr(a, key):
            cmd.append('--' + key.replace('_', '-'))
    return cmd


def linux_sample(pid):
    """Only this child's accumulated CPU time and resident bytes, not wall time."""
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        fields = raw[raw.rfind(')') + 2 :].split()
        return dict(
            cpu_seconds=(int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK'),
            rss_bytes=int(fields[21]) * os.sysconf('SC_PAGE_SIZE'),
        )
    except (OSError, ValueError, IndexError):
        return None


def failure_reason(log, code):
    try:
        with log.open('rb') as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 32768))
            codes = re.findall(r'\bE_[A-Z][A-Z0-9_]*\b', f.read().decode('utf-8', errors='replace'))
        if codes:
            return codes[-1]
    except OSError:
        pass
    return 'E_CHILD_SIGNAL' if code < 0 else 'E_CHILD_EXIT'


def kill_group(pid, signum):
    # start_new_session makes each child PID the process-group identifier.
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass
    except PermissionError:
        if sys.platform != 'darwin':
            raise  # macOS: EPERM for an all-zombie group (see cleanup)


def run_queue(a):
    """Run an already validated queue, always retaining each child's outcome."""
    from .components import production_only

    production_only()
    a.work_root.mkdir(parents=True, exist_ok=True)
    logs = a.work_root / 'logs' / str(time.time_ns())
    logs.mkdir(parents=True, exist_ok=False)
    env = paths.child_env(dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))  # children import sosfuse2 in-tree too
    env.update({key: str(a.threads) for key in THREAD_ENV})
    # Each state's preparation pools share the host's usable CPUs with the other states prepared at once (bytes
    # unchanged by construction); --workers 1 gives every pool all of them, the children's own default.
    host_cores = default_workers()
    per_state, explicit, updates = prep_workers(a.workers, a.threads, host_cores, env, cpu_budget())
    env.update(updates)
    prep = dict(
        host_cores=host_cores,
        states_at_once=a.workers,
        per_state=per_state,
        explicit=explicit,
        env={k: env.get(k) for k in PREP_WORKER_ENV},
    )
    if a.column_cache is not None:
        env['SOSFUSE2_COLUMN_CACHE'] = str(a.column_cache.resolve())
    else:
        env.pop(
            'SOSFUSE2_COLUMN_CACHE', None
        )  # off (incl. the literal 'off' of the environment) must not reach a child as a folder name
    start = clocks()
    rows = [
        dict(id=i, out_name=i, status='pending', exit_code=None, reason=None, log=str(logs / (i + '.log')))
        for i in a.ids
    ]
    active = {}
    index = 0
    interrupted = None
    controller_error = None
    controller_detail = None
    sample_count = 0
    sampled = sys.platform.startswith('linux') and Path('/proc/self/stat').is_file()
    sample_path = logs / 'cpu_samples.jsonl'
    service = service_log = session = None
    old_handlers = {}

    def on_signal(signum, _frame):
        nonlocal interrupted
        if interrupted is None:
            interrupted = signum

    def finish(pid, code, reason=None):
        proc, stream, row = active.pop(pid)
        stream.close()
        row.update(
            end=clocks(),
            exit_code=code,
            status='succeeded' if code == 0 and reason is None else 'failed',
            reason=reason if reason is not None else (None if code == 0 else failure_reason(Path(row['log']), code)),
        )

    def cleanup(reason):
        owned = list(active)
        for pid in owned:
            kill_group(pid, signal.SIGTERM)
        # Retain group identities until escalation: the leader may exit while a
        # descendant ignores TERM, and polling the leader alone would miss it.
        deadline = time.monotonic() + TERMINATE_GRACE
        while time.monotonic() < deadline:
            live = False
            for pid in owned:
                try:
                    os.killpg(pid, 0)
                    live = True
                except ProcessLookupError:
                    pass
                except PermissionError:
                    # macOS answers EPERM for a group whose members are all zombies (seen 2026-10-05/07 on a
                    # timed-out test run): nothing left to signal; the leader is reaped below.
                    pass
                active[pid][0].poll()
            if not live:
                break
            time.sleep(0.05)
        for pid in owned:
            kill_group(pid, signal.SIGKILL)
        for pid in owned:
            finish(pid, active[pid][0].wait(), reason)

    samples = None
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, on_signal)
        # Create an empty file on unsupported platforms too, preventing stale
        # samples from a previous run from being mistaken for this execution.
        samples = sample_path.open('w')
        if a.persistent:
            session = tempfile.TemporaryDirectory(prefix='sos-persistent-')
            env.update(SOS_PERSISTENT_ROOT=session.name, SOS_PERSISTENT_KEY=os.urandom(32).hex())
            service_log = (logs / 'GPU_SERVICE.log').open('w')
            service = subprocess.Popen(
                [
                    sys.executable,
                    '-B',
                    '-m',
                    'sosfuse2.pipeline_execution',
                    session.name,
                    a.backend,
                    str(a.gpu_lock),
                    str(a.gpu_active),
                    str(a.rss_gb),
                ],
                env=env,
                stdout=service_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + 30
            while not (Path(session.name) / 'ready').exists():
                if service.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('E_PERSISTENT_SERVICE_START')
                time.sleep(0.05)
        while index < len(rows) or active:
            if service is not None and service.poll() is not None:
                controller_error = 'E_PERSISTENT_SERVICE_LOST'
                break
            if interrupted is not None:
                break
            while index < len(rows) and len(active) < a.workers and interrupted is None:
                row = rows[index]
                index += 1
                row['start'] = clocks()
                stream = Path(row['log']).open('w')
                try:
                    proc = subprocess.Popen(
                        child_command(a, row['id']),
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except OSError as exc:
                    stream.close()
                    row.update(
                        end=clocks(), status='failed', reason='E_CHILD_LAUNCH', detail=f'{type(exc).__name__}: {exc}'
                    )
                    continue
                row.update(pid=proc.pid, status='running')
                active[proc.pid] = (proc, stream, row)
            for pid, (proc, _stream, row) in list(active.items()):
                if sampled:
                    sample = linux_sample(pid)
                    if sample is not None:
                        samples.write(json.dumps(dict(id=row['id'], pid=pid, **clocks(), **sample)) + '\n')
                        sample_count += 1
                code = proc.poll()
                if code is not None:
                    finish(pid, code)
            samples.flush()
            if active and interrupted is None:
                time.sleep(SAMPLE_INTERVAL)
    except Exception as exc:
        controller_error = str(exc) if str(exc).startswith('E_PERSISTENT_') else 'E_PIPELINE_CONTROLLER'
        controller_detail = (
            f'{type(exc).__name__}: {exc}'  # the code alone hides the cause; the traceback goes to stderr
        )
        traceback.print_exc(file=sys.stderr)
    finally:
        reason = 'E_PIPELINE_INTERRUPTED' if interrupted is not None else controller_error or 'E_PIPELINE_CONTROLLER'
        if active:
            cleanup(reason)
        for row in rows:
            if row['status'] == 'pending':
                row.update(
                    status='failed'
                    if controller_error and controller_error.startswith('E_PERSISTENT_')
                    else 'not_started',
                    reason=controller_error
                    or ('E_NOT_STARTED_SIGNAL' if interrupted is not None else 'E_NOT_STARTED_CONTROLLER'),
                )
        if service is not None:
            kill_group(service.pid, signal.SIGTERM)
            try:
                service.wait(timeout=TERMINATE_GRACE)
            except subprocess.TimeoutExpired:
                kill_group(service.pid, signal.SIGKILL)
                service.wait()
            service_log.close()
        if session is not None:
            session.cleanup()
        if samples is not None:
            samples.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    success = interrupted is None and controller_error is None and all(r['status'] == 'succeeded' for r in rows)
    code = 128 + interrupted if interrupted is not None else (0 if success else 1)
    summary = dict(
        complete=success,
        exit_code=code,
        signal=interrupted,
        reason='E_PIPELINE_INTERRUPTED' if interrupted is not None else controller_error,
        controller_detail=controller_detail,
        controller=dict(pid=os.getpid(), start=start, end=clocks()),
        workers=a.workers,
        threads=a.threads,
        prep_workers=prep,
        rss_gb=a.rss_gb,
        gpu_lock=str(a.gpu_lock) if a.gpu_lock is not None else None,
        legacy=a.legacy,
        persistent=a.persistent,
        gpu_active=a.gpu_active,
        logs=str(logs),
        service_pid=service.pid if service else None,
        cpu_sampling=dict(
            status='available' if sample_count else 'absent',
            reason=None if sample_count else ('E_CPU_SAMPLES_UNAVAILABLE' if sampled else 'E_CPU_SAMPLES_PLATFORM'),
            path=str(sample_path),
            count=sample_count,
            interval_seconds=SAMPLE_INTERVAL,
        ),
        states=rows,
    )
    counts = {r['id']: dict(hit=0, miss=0, invalid=0, disabled=0) for r in rows}
    profiles = {r['id']: [] for r in rows}  # ②b PROFILE evidence (run_gpu_state_v3 host, persistent service upload)
    attach = {r['id']: dict(hit=0, miss=0, unused=0, disabled=0) for r in rows}
    for path in [*(Path(r['log']) for r in rows), logs / 'GPU_SERVICE.log']:
        if not path.is_file() or (path.name == 'GPU_SERVICE.log' and not a.persistent):
            continue
        with path.open() as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and event.get('phase') == 'column_cache' and event.get('id') in counts:
                    if event.get('result') in counts[event['id']]:
                        counts[event['id']][event['result']] += 1
                if isinstance(event, dict) and event.get('phase') == 'profile' and event.get('id') in profiles:
                    profiles[event['id']].append(
                        {
                            k: v
                            for k, v in event.items()
                            if k not in ('event', 'id', 'phase', 'utc', 'timestamp_ns', 'monotonic_ns', 'pid')
                        }
                    )
                if isinstance(event, dict) and event.get('phase') == 'attach_cache' and event.get('id') in attach:
                    if event.get('result') in attach[event['id']]:
                        attach[event['id']][event['result']] += 1
                if isinstance(event, dict) and event.get('phase') == 'KSPLIT_FALLBACK':
                    for row in rows:
                        if row['id'] == event.get('id'):
                            row.setdefault('ksplit_fallback', []).append(event)
    for row in rows:
        row['column_cache'] = counts[row['id']]
    for row in rows:
        row['attach_cache'] = attach[row['id']]
    for row in rows:
        row['profile'] = profiles[row['id']]
    summary['attach_cache'] = {k: sum(v[k] for v in attach.values()) for k in ('hit', 'miss', 'unused', 'disabled')}
    summary['column_cache'] = {k: sum(v[k] for v in counts.values()) for k in ('hit', 'miss', 'invalid', 'disabled')}
    (a.work_root / 'pipeline_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(
        json.dumps(dict(complete=success, exit_code=code, summary=str(a.work_root / 'pipeline_summary.json'))),
        flush=True,
    )
    return code


def column_cache_plan(a):
    # Same profile defaults as stratosphere.profile_config; no NumPy/CUDA imports.
    doc = json.loads(a.states.read_text())
    states = doc['states']
    ids = a.ids.split(',')
    if doc.get('parent'):
        own = {s['id'] for s in states}
        states += [s for s in json.loads((a.states.parent / doc['parent']).read_text())['states'] if s['id'] not in own]
    if len(set(ids)) != len(ids) or any(not SAFE_ID.fullmatch(i) for i in ids):
        raise ValueError('E_PIPELINE_IDS')
    groups = {}
    inactive = []
    for identifier in ids:
        found = [s for s in states if s['id'] == identifier and s.get('runnable')]
        if len(found) != 1:
            raise ValueError('E_PIPELINE_STATE_ID: ' + identifier)
        state = found[0]
        layer = state.get('stratosphere', {})
        if (
            not (a.stratosphere == 'on' or layer.get('status') == 'enabled')
            or float(state['parameters'].get('stratosphere_tau550', 0)) == 0
        ):
            inactive.append(identifier)
            continue
        center, sigma = float(layer.get('center_km', 21)), float(layer.get('sigma_km', 3))
        if (
            not math.isfinite(center + sigma)
            or not 0 <= center <= 80
            or sigma < 1
            or layer.get('profile', 'gaussian') != 'gaussian'
        ):
            raise ValueError('E_PIPELINE_PROFILE')
        groups.setdefault((center, sigma), []).append(identifier)
    print(
        json.dumps(
            dict(
                profile_count=len(groups),
                profiles=[dict(center=c, sigma=s, ids=v) for (c, s), v in groups.items()],
                inactive_ids=inactive,
                assumption='Same grid settings assumed; ray hashes not computed; actual cache hits require identical input bytes.',
            )
        )
    )
    return 0


def resolve_plan(a):
    """--plan: the ids of one GPU list of a plan_queue.py plan, in its order, if the plan was made from these states."""
    if a.plan is None:
        if not a.ids:
            raise ValueError('E_PIPELINE_ARGUMENTS: --ids or --plan required')
        return a
    if a.ids:
        raise ValueError('E_PIPELINE_PLAN: --ids and --plan are exclusive')
    try:
        plan = json.loads(a.plan.read_text())
        inputs = plan['inputs']
        ids = {g['gpu']: g['ids'] for g in plan['gpus']}.get(a.plan_gpu)
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ValueError('E_PIPELINE_PLAN: unreadable plan') from e
    if plan.get('version') != 1 or ids is None or not ids:
        raise ValueError('E_PIPELINE_PLAN: version 1 plan with a nonempty list for --plan-gpu required')
    import hashlib

    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if sha(a.states) != inputs.get('states_sha256'):
        raise ValueError('E_PIPELINE_PLAN_STALE: --states differs from the planned states file')
    doc = json.loads(a.states.read_text())
    parent = doc.get('parent')
    states = list(doc['states'])
    if parent:
        if sha(a.states.parent / parent) != inputs.get('parent_sha256'):
            raise ValueError('E_PIPELINE_PLAN_STALE: parent states file differs from the planned one')
        own = {s['id'] for s in states}
        states += [s for s in json.loads((a.states.parent / parent).read_text())['states'] if s['id'] not in own]
    # The keys and groups also depend on the grid settings and the optical options the runner applies.
    settings = (
        sha(a.settings_json)
        if a.settings_json is not None and a.settings_json.is_file()
        else (None if a.settings_json is None else 'missing')
    )
    run = dict(
        settings_sha256=settings,
        reduced=bool(a.reduced),
        geometry_layers=a.geometry_layers,
        **{k: getattr(a, k) for k in ('stratosphere', 'density_profile', 'ozone_profile', 'rayleigh_column')},
    )
    planned = dict(
        settings_sha256=inputs.get('settings_sha256'),
        reduced=bool(inputs.get('reduced', False)),
        geometry_layers=inputs.get('geometry_layers'),
        **inputs.get('options', {}),
    )
    differ = sorted(k for k in run if run[k] != planned.get(k))
    if differ:
        raise ValueError('E_PIPELINE_PLAN_STALE: run arguments differ from the plan: ' + ','.join(differ))
    # Every GPU list together: each planned state exactly once, and (planned without --ids) every runnable state.
    every = [i for g in plan['gpus'] for i in g['ids']]
    runnable = [s['id'] for s in states if s.get('runnable')]
    if (
        len(every) != len(set(every))
        or set(every) != set(inputs.get('ids', ()))
        or any(runnable.count(i) != 1 for i in every)
        or (not inputs.get('ids_explicit') and set(every) != set(runnable))
    ):
        raise ValueError('E_PIPELINE_PLAN: GPU lists must hold every planned runnable state exactly once')
    a.ids = ','.join(ids)
    return a


def main(argv=None):
    p = parser()
    try:
        a = p.parse_args(argv)
        a = resolve_plan(a)
        if a.column_cache_plan:
            return column_cache_plan(a)
        a = validate(a)
    except (ValueError, OSError, KeyError, TypeError) as e:
        p.error(str(e))
    return run_queue(a)


if __name__ == '__main__':
    sys.exit(main())
