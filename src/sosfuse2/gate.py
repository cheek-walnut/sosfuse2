"""sosfuse2 numerical gate v2. python -m sosfuse2.gate ...

G-R CUDA bytes (informational), G-P CUDA vs NumPy golden (rtol 1e-3 + atol), G-T ratchet against fixed anchors, G-A S2
vs same-run exact, G-H reduced selftest vs record (old fixture delta informational), G-PS persistent test, G-F full-grid cold/warm (--gf, eda deploys).
Exit 0 pass, 1 numerical failure, 2 configuration/environment/exception; out/gate.json is always written.
CUDA work only through the site's GPU queue: <python> -B -m sosfuse2.gate --all --out DIR.
"""

import argparse, copy, datetime as dt, hashlib, importlib.util, json, os, re, shutil, subprocess, sys, traceback
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from .kernel_config import deterministic
from . import paths

HERE = Path(__file__).resolve().parent
GATE = HERE / 'gate'  # fixtures.json, anchors.json, baselines/{manifest.json,numpy/,<toolkey>/}
STAGE1 = ('FMAD', 'FASTMATH')
CONTROLS = ('SOSFUSE2_KERNEL_FLAGS', 'SOSFUSE2_BLOCK_SIZE', 'SOSFUSE2_STRATO_ASYNC') + tuple(
    'SOSFUSE2_' + f for f in STAGE1
)
RTOL, RATCHET_FACTOR, RATCHET_FLOOR, ANCHOR_STOP = 1e-3, 3.0, 1e-6, 3e-4
PS_TESTS = ('test_1_parity', 'test_2_no_state_mixing', 'test_5_interleaving')


class GateError(Exception):
    """Configuration / environment failure: exit 2 with a stable code."""

    def __init__(self, code, detail=''):
        super().__init__(f'{code}: {detail}')
        self.code = code


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def jdump(path, data):
    Path(path).write_text(json.dumps(data, indent=1, default=str, allow_nan=True) + '\n')


def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha(a):
    """Integrity fingerprint of an array: dtype, shape and raw bytes (+0/-0 and NaN payloads differ)."""
    a = np.ascontiguousarray(a)
    return hashlib.sha256(f'{a.dtype.str}{a.shape}'.encode() + a.tobytes()).hexdigest()


def digest(x, h=None):
    """Input fingerprint of host-prepared geometry/constants (dicts in key order, arrays by bytes)."""
    top = h is None
    h = h or hashlib.sha256()
    if isinstance(x, dict):
        for k in sorted(x, key=str):
            h.update(str(k).encode())
            digest(x[k], h)
    elif isinstance(x, (list, tuple)):
        for v in x:
            digest(v, h)
    elif isinstance(x, np.ndarray) or type(x).__module__.startswith('numpy'):
        h.update(sha(np.asarray(x)).encode())
    elif isinstance(x, (int, float, str, bool, type(None))):
        h.update(repr(x).encode())
    else:
        h.update(type(x).__name__.encode())
    return h.hexdigest() if top else None


def numeric(v):
    if isinstance(v, dict):
        return {k: numeric(x) for k, x in v.items() if 'seconds' not in k}
    if isinstance(v, list):
        return [numeric(x) for x in v]
    return v


def code_identity():
    """Hash of everything that decides fixture numbers (gate.py and tests excluded): a staged run is blessable only by it."""
    from sosfuse2 import density
    from sosfuse2.run_gpu_state_v3 import package_hashes

    # One manifest for both provenance versions v3/v4: the v4 package_hashes() set (fixed lists: production modules,
    # kernel, accepted sources + loader, host helpers, native kernel, data assets) minus gate.py itself, plus the gate's own
    # fixtures/configs and the ozone profile asset under its logical key. A missing listed file is E_PACKAGE_FILE_MISSING.
    try:
        hashes = package_hashes()
    except RuntimeError as e:
        raise GateError('E_PACKAGE_FILE_MISSING', str(e)) from e
    hashes.pop('sosfuse2/gate.py', None)
    extra = [GATE / 'fixtures.json', *[HERE / 'configs' / f'{n}.json' for n in ('C3', 'C3_nostrato', 'devices')]]
    missing = [paths.rel(p) for p in extra if not p.is_file()]
    if missing:
        raise GateError('E_PACKAGE_FILE_MISSING', f'gate identity inputs absent from the package: {missing}')
    hashes.update({paths.rel(p): file_sha(p) for p in extra})
    hashes[density.ASSET_KEY] = file_sha(density.asset_path())  # ozone q (ozone_profile usstd76), logical key
    return hashlib.sha256(json.dumps(sorted(hashes.items())).encode()).hexdigest()


def load_fixtures():
    return json.loads((GATE / 'fixtures.json').read_text())


def config(name):
    return json.loads((HERE / 'configs' / f'{name}.json').read_text())


def state_id(fx):
    return 'GATE_' + fx.replace('-', '_')


def fixture_state(F, fx, solve=False):
    """② uses the product state; ① (reduced solve) the same state with F['solve_parameters'] (the product AOD .4 does not
    converge on the reduced grid: 8/15 after 64 evaluations, IMPL_REPORT.md). A fixture's 'parameters' override the product
    values (FX-A: albedo 0.3, so the boundary term that alone consumes main-ray endpoint columns is not zero)."""
    s = copy.deepcopy(F['state'])
    s['id'] = state_id(fx)
    s['runnable'] = True
    if not F['fixtures'][fx]['stratosphere']:
        s['stratosphere']['status'] = 'disabled'
    for key in ('parameters', 'parameters_requested', 'parameters_computed'):
        if key in s:
            s[key].update(F['fixtures'][fx].get('parameters', {}))
    for key in ('parameters', 'parameters_requested', 'parameters_computed') if solve else ():
        if key in s:
            s[key].update(F.get('solve_parameters', {}))
    return s


def fixture_settings(F, fx):
    """The shared settings with the fixture's own 'settings' keys on top (FX-AP: ozone_temperature profile_mean, the A′
    reference added by P2 next to the fixed-233 K fixtures)."""
    return dict(F['settings'], **F['fixtures'][fx].get('settings', {}))


def fixture_env(F, fx, base=None):
    """The caller's env (profile, LAUNCH_CELLS, caches) with every kernel control written explicitly."""
    spec = F['fixtures'][fx]
    c = config(spec['config'])
    e = {k: v for k, v in (os.environ if base is None else base).items() if k not in CONTROLS}
    e.update(
        SOSFUSE2_KERNEL_FLAGS=','.join(c['kernel_flags']),
        SOSFUSE2_BLOCK_SIZE=str(c['block_size']),
        SOSFUSE2_STRATO_ASYNC=str(spec['async']),
        PYTHONDONTWRITEBYTECODE='1',
        PYTHONUNBUFFERED='1',
        **{'SOSFUSE2_' + f: str(spec['stage1'][f]) for f in STAGE1},
    )
    return e


def expected_config(F, fx):
    spec = F['fixtures'][fx]
    c = config(spec['config'])
    from .kernel_config import resolve_config

    flags = list(resolve_config(set(c['kernel_flags']) | ({'STRATO'} if spec['stratosphere'] else set())).flags)
    from .density import profile_n, profile_step_km

    return dict(
        flags=flags,
        stage1=[f for f in STAGE1 if spec['stage1'][f]],
        async_=bool(spec['async']),
        L=F['L'],
        block=c['block_size'],
        profile_n=profile_n(profile_step_km(fixture_state(F, fx))),
    )


def ozone_mismatch(F, fx, spectral):
    """P2: the ozone treatment a fixture asks for (fixture_settings) vs the spectral block a run consumed; '' when they agree.
    Only runs that record their block are judged (records from before P2 carry none)."""
    asked = fixture_settings(F, fx).get('ozone_temperature', '233K')
    used = (spectral or {}).get('ozone_temperature', '233K')
    if asked == used and (used != 'profile_mean' or isinstance(spectral.get('ozone_profile'), dict)):
        return ''
    return f"ozone_temperature requested {asked} measured {used} (q {(spectral or {}).get('ozone_profile')})"


def check_config(F, fx, meta):
    """Requested fixture controls vs what the worker measured, including the NVRTC options it compiled."""
    want = expected_config(F, fx)
    got = meta['config']
    bad = []
    for k in ('flags', 'L', 'block') + (('stage1', 'async_') if meta['backend'] == 'cupy' else ()):
        if got[k] != want[k]:
            bad.append(f'{k} requested {want[k]} measured {got[k]}')
    # ②b: PROFILE node count (records from before ②b carry none: 161)
    if got.get('profile_n', 161) != want['profile_n']:
        bad.append(f"profile_n requested {want['profile_n']} measured {got.get('profile_n', 161)}")
    if 'spectral' in meta and (why := ozone_mismatch(F, fx, meta['spectral'])):
        bad.append(why)
    if meta['backend'] == 'cupy':
        if not got['options']:
            bad.append('no NVRTC options recorded')
        s1 = set(want['stage1'])
        for o in got['options']:
            seen = dict(
                fmad='--fmad=true' in o,
                div='--prec-div=false' in o,
                sqrt='--prec-sqrt=false' in o,
                fast='-DSOSFUSE2_FASTMATH' in o,
            )
            need = dict(fmad='FMAD' in s1, div='FASTMATH' in s1, sqrt='FASTMATH' in s1, fast='FASTMATH' in s1)
            if seen != need:
                bad.append(f'NVRTC options {o} disagree with stage1 {sorted(s1)}')
                break
    if bad:
        raise GateError('E_GATE_CONFIG', f'{fx}: ' + '; '.join(bad))


def toolkey(meta):
    from .kernel_config import resolve_config

    c = meta['config']
    fields = dict(
        cc=meta['device'].get('cc'),
        nvrtc=meta['device'].get('nvrtc'),
        profile=meta['device'].get('profile'),
        flags=list(resolve_config(c['flags']).flags),
        stage1=c['stage1'],
        async_=c['async_'],
        L=c['L'],
        block=c['block'],
    )
    # ②b: N joins the toolkey only off the default, so every 0.5 km toolkey (and its recorded baselines) is unchanged.
    if c.get('profile_n', 161) != 161:
        fields['profile_n'] = c['profile_n']
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16], fields


# ---------------------------------------------------------------- worker (subprocess; only it imports sosfuse2.*)


def worker(kind, fx, out, backend):
    os.environ.setdefault('SOSFUSE2_KSPLIT_FALLBACK', '0')
    from sosfuse2 import run_gpu_state_v3 as v3
    from sosfuse2.compile_cache import configure

    F = load_fixtures()
    out.mkdir(parents=True, exist_ok=True)
    os.environ.pop('STRATO_ATTACH_CACHE', None)  # cold attach inside every fixture run
    block = int(os.environ['SOSFUSE2_BLOCK_SIZE'])
    if backend == 'cupy':
        configure(out / 'nvrtc_cache', 'cupy', block)
        os.environ['CUDA_CACHE_PATH'] = str(out / 'cuda_driver_cache')
    if kind == 'historical':
        from sosfuse2.validate import reduced

        report, _ = reduced(out, backend, block)
        jdump(out / 'historical.json', report)
        return 0
    from sosfuse2.fused import Solver
    from sosfuse2.geometry import TransientGeometry
    from sosfuse2.stratosphere import source_size
    from sosfuse2.strato_parallel import _async_enabled

    (out / 'settings.json').write_text(json.dumps(fixture_settings(F, fx)))
    cfg = v3.settings_for(NS(reduced=True, settings_json=out / 'settings.json'))
    state = v3.effective_state(fixture_state(F, fx), NS())
    cfg = v3.resolve_settings(cfg, state)  # A′ fixtures: FX-AP
    el, az = (np.asarray(F['directions'][n], float) for n in ('elevation_deg', 'azimuth_deg'))
    # The kernel specializes to 1 or 15 wavelengths: n-wavelength fixtures (band BM45, 45) run FW=15 chunks like
    # run_gpu_state_v3 (last chunk padded, padding dropped); a single chunk keeps the paths and meta of the 15-wavelength gate.
    wl = np.asarray(cfg['wavelengths_nm'])
    chunks = [wl] if len(wl) in (1, 15) else [wl[i : i + 15] for i in range(0, len(wl), 15)]
    sources, radiance, host_inputs, launch_max, launches = (
        [[] for _ in range(F['applies'])],
        [[] for _ in range(F['applies'])],
        [],
        0.0,
        [],
    )
    for index, chunk in enumerate(chunks):
        count = len(chunk)
        padded = (
            chunk
            if count == 15 or (count == 1 and len(chunks) == 1)
            else np.r_[chunk, np.repeat(chunk[-1], 15 - count)]
        )
        s, scope = v3.create_configured(
            out / ('prepare' if len(chunks) == 1 else f'prepare_{index}'),
            cfg,
            padded,
            backend,
            state,
            block,
            geometry_cache=out / 'geometry_cache',
        )  # host sidecars cold and private per run
        geometry = TransientGeometry(s, scope)
        solver = Solver(s, geometry)
        k = s.kernel
        field = k.xp.zeros((s.geo.ns, source_size(s), s.W), 'f4')
        for i in range(F['applies']):
            field = solver.apply(field)
            host = k.host(field)
            sources[i].append(host[..., :count])
            full = np.zeros((*host.shape[:2], 2 * s.W), 'f4')
            full[..., ::2] = host
            r = np.stack(
                [
                    (s.query(h, el, az, full, True) + s.query(h, el, az, full, False))[..., ::2]
                    for h in F['observer_solar_deg']
                ]
            )
            if r.shape[-2] != s.W:
                raise GateError('E_GATE_CONFIG', f'radiance wavelength axis -2 expected {s.W}, shape {r.shape}')
            radiance[i].append(r[..., :count, :])  # (solar, direction, wavelength, 1)
        nq = sorted({g['nq'] for g in geometry.groups.values()})
        host_inputs.append(
            dict(
                groups={str(c): geometry.groups[c] for c in sorted(geometry.groups)},
                constants={str(n): k.setup(s, n) for n in nq},
                h=s.geo.h,
                chi=s.geo.chi,
            )
        )
        launch_max = max(
            launch_max,
            getattr(k, 'launch_log', {}).get('max_seconds', 0.0),
            k.metrics.get('strato_micro_max_seconds', 0.0),
        )
        launches.append(getattr(k, 'launch_log', {}).get('launches'))
    sources = [np.concatenate(c, axis=-1) for c in sources]
    radiance = [np.concatenate(c, axis=-2) for c in radiance]
    device = dict(profile=k.device()[0])
    if backend == 'cupy':
        import cupy

        p = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.runtime.getDevice())
        name = p['name'].decode() if isinstance(p['name'], bytes) else str(p['name'])
        device.update(
            cc=f"{p['major']}.{p['minor']}",
            nvrtc='.'.join(map(str, cupy.cuda.nvrtc.getVersion())),
            gpu=name,
            driver=cupy.cuda.runtime.driverGetVersion(),
            runtime=cupy.cuda.runtime.runtimeGetVersion(),
            cupy=cupy.__version__,
        )
    meta = dict(
        kind=kind,
        fixture=fx,
        backend=backend,
        device=device,
        numpy=np.__version__,
        code_identity=code_identity(),
        config=dict(
            deterministic=k.deterministic,
            flags=sorted(k.flags),
            stage1=list(k.stage1_flags()) if backend == 'cupy' else [],
            async_=_async_enabled(k),
            L=k.L,
            block=k.block_size,
            profile_n=k.profile_n,
            options=[list(o) for _, o in getattr(k, 'raw_options', {}).values()],
        ),
        spectral=cfg.get('spectral'),  # the resolved block (FX-AP: A′), check_config
        host_inputs_sha256=digest(host_inputs[0] if len(chunks) == 1 else host_inputs),
        launch=dict(max_seconds=launch_max, launches=launches[0] if len(chunks) == 1 else launches),
        env={n: v for n, v in os.environ.items() if n.startswith('SOSFUSE2_')},
    )
    np.savez(out / 'fixed.npz', radiance=np.stack(radiance), source=np.stack(sources))
    jdump(out / 'fixed.json', meta)
    return 0


def child(cmd, log, env, cwd=paths.SRC, timeout=3600):
    """Same process group as the gate (a safety stop of the gate reaches it); TERM then KILL on timeout or error."""
    with Path(log).open('w') as f:
        p = subprocess.Popen(cmd, cwd=str(cwd), env=paths.child_env(env), stdout=f, stderr=subprocess.STDOUT)
        try:
            rc = p.wait(timeout=timeout)
        except BaseException:
            p.terminate()
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            raise
    tail = Path(log).read_text(errors='replace').splitlines()[-40:]
    return rc, tail, (re.findall(r'\bE_[A-Z][A-Z0-9_]+', '\n'.join(tail)) or [None])[-1]


def log_launch(paths):
    m = 0.0
    for p in paths:
        for line in Path(p).read_text(errors='replace').splitlines():
            if line.startswith(('STRATO_ROUNDS ', 'LAUNCH_REPORT ')):
                m = max(m, json.loads(line.split(' ', 1)[1]).get('max_seconds', 0.0))
    return m


def run_fixture(F, fx, d, backend, solve, base=None):
    """② fixed applies in a worker process, then ① pipeline.py --reduced solve (cupy). Writes d/{fixed,solve}.*"""
    d.mkdir(parents=True, exist_ok=True)
    env = fixture_env(F, fx, base)
    rc, tail, code = child(
        [
            sys.executable,
            '-B',
            '-m',
            'sosfuse2.gate',
            '--worker',
            'fixed',
            '--fixture',
            fx,
            '--backend',
            backend,
            '--out',
            str(d),
        ],
        d / 'fixed.log',
        env,
    )
    if rc:
        raise GateError('E_GATE_WORKER', f'{fx} fixed rc={rc} last={code} tail={tail[-3:]}')
    if not solve:
        return
    sid = state_id(fx)
    w = d / 'solve'
    (d / 'states.json').write_text(json.dumps(dict(states=[fixture_state(F, fx, solve=True)])))
    (d / 'settings.json').write_text(json.dumps(fixture_settings(F, fx)))
    cmd = [
        *paths.module_command('pipeline'),
        '--states',
        str(d / 'states.json'),
        '--ids',
        sid,
        '--work-root',
        str(w),
        '--workers',
        '1',
        '--threads',
        '1',
        '--backend',
        'cupy',
        '--reduced',
        '--settings-json',
        str(d / 'settings.json'),
        '--gpu-lock',
        str(w / 'gpu.lock'),
        '--geometry-cache',
        str(w / 'geometry'),
    ]
    rc, tail, code = child(cmd, d / 'solve.log', env)
    row = {}
    try:
        row = json.loads((w / 'pipeline_summary.json').read_text())['states'][0]
    except (OSError, ValueError, KeyError, IndexError):
        pass
    meta = dict(rc=rc, reason=row.get('reason') or code)
    if rc == 0:
        m = json.loads((w / 'out' / sid / 'meta.json').read_text())
        solve_ = m['source_solve']
        with np.load(w / 'out' / sid / 'table.npz') as t, np.load(w / 'tmp' / sid / 'source.npz') as s:
            np.savez(d / 'solve.npz', table=t['radiance'], source=s['source'])
        meta.update(
            evaluations=solve_evaluations(solve_),
            converged=solve_converged(solve_),
            source_solve_sha256=hashlib.sha256(json.dumps(numeric(solve_), sort_keys=True).encode()).hexdigest(),
            flags=sorted(m['kernel']['effective_flags']),
            block=m['kernel']['block_size'],
            spectral=(m.get('settings') or {}).get('spectral'),
            launch_max_seconds=max(
                (m['kernel'].get('launch') or {}).get('max_seconds', 0.0), log_launch([Path(row['log'])])
            ),
        )
    jdump(d / 'solve.json', meta)


def solve_evaluations(ss):
    """meta source_solve is one dict (1 or 15 wavelengths) or one per FW=15 chunk (n-wavelength, run_gpu_state_v3): int or list."""
    return ss['evaluations'] if isinstance(ss, dict) else [c['evaluations'] for c in ss]


def solve_converged(ss):
    return all(all(c['converged']) for c in ([ss] if isinstance(ss, dict) else ss))


def more_iterations(a, e):
    """G-A: the approximate path needs more evaluations than the exact one (per chunk for chunked solves)."""
    return a > e if isinstance(a, int) else len(a) != len(e) or any(x > y for x, y in zip(a, e))


def load_run(d, fx):
    d = Path(d) / fx
    r = dict(meta=json.loads((d / 'fixed.json').read_text()))
    with np.load(d / 'fixed.npz') as z:
        r['radiance'], r['source'] = z['radiance'], z['source']
    if (d / 'solve.json').is_file():
        r['solve'] = json.loads((d / 'solve.json').read_text())
        if r['solve']['rc'] == 0:
            with np.load(d / 'solve.npz') as z:
                r['table'], r['solve_source'] = z['table'], z['source']
    return r


def outputs(r):
    """name -> byte sha of everything G-R pins (② every apply, ① table/source/numeric solve history)."""
    o = {f'fixed.radiance.{i}': sha(x) for i, x in enumerate(r['radiance'])}
    o.update({f'fixed.source.{i}': sha(x) for i, x in enumerate(r['source'])})
    if 'table' in r:
        o.update(
            {
                'solve.table': sha(r['table']),
                'solve.source': sha(r['solve_source']),
                'solve.source_solve': r['solve']['source_solve_sha256'],
            }
        )
    return o


def repeat_arrays(a, b, rtol=1e-5):
    """Repeat comparison with exact layout and finite values; zeros have no absolute allowance."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return dict(pass_=False, max_rel=None, reason='layout')
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return dict(pass_=False, max_rel=None, reason='nonfinite')
    delta = np.abs(a.astype('f8') - b.astype('f8'))
    denom = np.abs(b.astype('f8'))
    rel = np.divide(delta, denom, out=np.where(delta == 0, 0.0, np.inf), where=denom != 0)
    return dict(pass_=bool(np.all(delta <= rtol * denom)), max_rel=float(np.max(rel, initial=0.0)))


def repeat_run(a, b):
    mode = deterministic()
    fields = ('radiance', 'source', 'table', 'solve_source')
    comparisons = {
        k: repeat_arrays(a[k], b[k]) if k in a and k in b else dict(pass_=False, max_rel=None, reason='missing')
        for k in fields
        if k in a or k in b
    }
    same = outputs(a) == outputs(b)
    maxima = [v['max_rel'] for v in comparisons.values() if v['max_rel'] is not None]
    evaluations = [(r.get('solve') or {}).get('evaluations') for r in (a, b)]
    histories = [(r.get('solve') or {}).get('source_solve_sha256') for r in (a, b)]
    return dict(
        pass_=same if mode else all(v['pass_'] for v in comparisons.values()),
        deterministic=mode,
        repeat_max_rel=max(maxima, default=0.0),
        comparisons=comparisons,
        outputs_equal=same,
        evaluations=evaluations,
        source_solve=histories,
        evaluations_warning=None if evaluations[0] == evaluations[1] else 'W_REPEAT_EVALUATIONS_DIFFER',
    )


# ---------------------------------------------------------------- comparisons (pure)


def regress(base_out, run_out, base_rad=None, run_rad=None):
    names = sorted(set(base_out) | set(run_out))
    diff = [n for n in names if base_out.get(n) != run_out.get(n)]
    row = dict(pass_=not diff, compared=len(names), differing=diff[:12])
    if diff and base_rad is not None and run_rad is not None:
        t = tolerance(run_rad, base_rad, 0.0)
        row.update(first=diff[0], radiance_max_abs=t['max_abs'], radiance_max_rel=t['max_rel'], worst=t['worst'])
    return row


def tolerance(actual, ref, atol):
    a, r = np.asarray(actual, 'f8'), np.asarray(ref, 'f8')
    if a.shape != r.shape:
        return dict(pass_=False, reason=f'shape {a.shape} vs {r.shape}', max_abs=None, max_rel=None, worst=None)
    d = np.abs(a - r)
    bound = RTOL * np.abs(r) + atol
    excess = d - bound
    finite = bool(np.isfinite(a).all() and np.isfinite(r).all())
    normalized = np.divide(d, bound, out=np.where(d == 0, 0.0, np.inf), where=bound > 0)
    i = np.unravel_index(int(np.nanargmax(np.where(np.isnan(excess), np.inf, excess))), d.shape) if d.size else ()
    return dict(
        pass_=finite and bool((excess <= 0).all()),
        finite=finite,
        E_inf=float(normalized.max(initial=0.0)) if finite else float('inf'),
        max_abs=float(np.nanmax(d)) if d.size else 0.0,
        max_rel=float(np.nanmax(d / np.maximum(np.abs(r), 1e-30))) if d.size else 0.0,
        worst=[int(x) for x in i],
        minimum_atol_required=float(np.nanmax(np.maximum(d - RTOL * np.abs(r), 0))) if d.size else 0.0,
        rtol=RTOL,
        atol=atol,
    )


def l2_by_wavelength(c, r):
    c, r = np.asarray(c, 'f8').reshape(-1, c.shape[-1]), np.asarray(r, 'f8').reshape(-1, r.shape[-1])
    dl, rl = np.linalg.norm(c - r, axis=0), np.linalg.norm(r, axis=0)
    return np.divide(dl, rl, out=np.where(dl == 0, 0.0, np.inf), where=rl > 0)


def elevation_stats(c_rad, r_rad, hs):
    """S_max = validate.relative per elevation; S_rms over elements with |I_ref| >= 1e-3 max|I_ref| of that elevation."""
    out = dict(S_max={}, S_rms={})
    for i, h in enumerate(hs):
        c, r = np.asarray(c_rad[i], 'f8'), np.asarray(r_rad[i], 'f8')
        rel = np.abs(c - r) / np.maximum(np.abs(r), 1e-30)
        m = np.abs(r) >= 1e-3 * np.abs(r).max()
        key = f'{float(h):g}'
        out['S_max'][key] = float(rel.max())
        out['S_rms'][key] = float(np.sqrt(np.mean(rel[m] ** 2)))
    return out


def ratchet_stats(c_rad, r_rad, c_src, r_src, hs):
    out = elevation_stats(c_rad, r_rad, hs)
    out['S_src'] = float(l2_by_wavelength(c_src, r_src).max())  # last source, max over wavelengths
    return out


def flat(stats):
    for name, v in stats.items():
        if isinstance(v, dict):
            for h, x in v.items():
                yield f'{name}[{h}]', x
        else:
            yield name, v


def ratchet_check(values, anchors):
    rows, ok = {}, True
    for pair, st in values.items():
        if pair not in anchors:
            raise GateError('E_GATE_ANCHOR_MISSING', f'no anchor for {pair}')
        a = dict(flat(anchors[pair]))
        rows[pair] = {}
        for name, v in flat(st):
            if name not in a:
                raise GateError('E_GATE_ANCHOR_MISSING', f'{pair} {name}')
            limit = max(RATCHET_FACTOR * a[name], RATCHET_FLOOR)
            good = bool(v <= limit)
            rows[pair][name] = dict(value=v, anchor=a[name], limit=limit, pass_=good)
            ok &= good
    return ok, rows


def pairs(F, runs, golden):
    """Ratchet data: K-th apply radiance and last source per pair present in this run."""
    got = {}
    for cand, ref in F['ratchet_pairs']:
        if cand not in runs:
            continue
        if ref == 'golden':
            R = golden.get(F['fixtures'][cand]['golden'])
        else:
            R = runs.get(ref) and dict(radiance=runs[ref]['radiance'], source=runs[ref]['source'][-1])
        if R is None:
            raise GateError('E_GATE_RATCHET_MISSING', f'{cand}|{ref}: required reference absent')
        got[f'{cand}|{ref}'] = ratchet_stats(
            runs[cand]['radiance'][-1],
            R['radiance'][-1],
            runs[cand]['source'][-1],
            R['source'] if R['source'].ndim == 3 else R['source'][-1],
            F['observer_solar_deg'],
        )
    return got


# ---------------------------------------------------------------- baselines

RECORDS = None  # main(): None = package baselines (bless modes, --record-baselines); else <out>/records


def record(rel):
    """Where an execution record goes. Baseline updates (--bless, --bless-numpy, or --record-baselines in a writable checkout)
    write the package's own files; any other run writes copies under --out/records, so a gate run against an installed, possibly
    read-only package never modifies site-packages. rel is 'baselines/manifest.json', 'baselines/BASELINE_LOG.md' or 'anchors.json'."""
    p = GATE / rel if RECORDS is None else RECORDS / Path(rel).name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def manifest():
    p = GATE / 'baselines/manifest.json'
    return json.loads(p.read_text()) if p.is_file() else dict(schema=1, golden={}, cuda={}, gf=None)


def baseline_npz(rel, digest_):
    p = GATE / 'baselines' / rel
    if not p.is_file():
        raise GateError('E_GATE_BASELINE_MISSING', rel)
    if file_sha(p) != digest_:
        raise GateError('E_GATE_MANIFEST_SHA', rel)
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def goldens(F, names):
    """NumPy goldens the fixtures need; 'golden': null = fixed canonical CUDA tolerance, no G-P."""
    return sorted({F['fixtures'][fx]['golden'] for fx in names} - {None})


def golden_set(M, names):
    out = {}
    for n in names:
        e = M['golden'].get(n)
        if e is None:
            raise GateError('E_GATE_BASELINE_MISSING', f'NumPy golden {n}')
        out[n] = baseline_npz(e['file'], e['sha256'])
    return out


def log_line(text):
    with record('baselines/BASELINE_LOG.md').open('a') as f:
        f.write(f'- {utc()} {text}\n')


def commit():
    """Source commit recorded with a run: GATE_SOURCE_COMMIT, else the package's own commit (run_gpu_state_v3.package_info:
    SOSFUSE2_PACKAGE_COMMIT, direct_url.json, or the git checkout that IS this package; never a consumer checkout that merely
    holds an installed copy). None is a valid answer."""
    from sosfuse2.run_gpu_state_v3 import package_info

    return os.environ.get('GATE_SOURCE_COMMIT') or package_info()['package_commit']


def portable(path):
    """Record form of a reference file: '~/<rest>' under the home directory, else the bare file name (no host path in records)."""
    p = Path(path)
    try:
        return '~/' + p.expanduser().relative_to(Path.home()).as_posix()
    except (ValueError, RuntimeError):
        return p.name  # not under home, or no home directory at all


def load_anchors():
    p = GATE / 'anchors.json'
    if not p.is_file():
        raise GateError(
            'E_GATE_ANCHOR_MISSING',
            'gate/anchors.json absent; measure once with --bless --measure-anchors (never auto-initialized)',
        )
    return json.loads(p.read_text())['anchors']


def inject(kind, runs, golden, F, anchors):
    """Negative controls. An injected run is never blessed."""
    if kind == 'bytes':
        runs['FX-C3']['radiance'].reshape(-1).view(np.uint8)[0] ^= 1  # lowest mantissa bit of one element
    elif kind == 'golden':
        g = golden['FX-C3']['radiance']
        h = list(F['observer_solar_deg']).index(-6)
        last = g[-1, h]
        a = anchors.get('FX-C3|golden', {}).get('S_max', {}).get('-6', 0.0)
        delta = (max(RATCHET_FACTOR * a, RATCHET_FLOOR) + RTOL) / 2  # above the ratchet, below G-P's 1e-3
        i = np.unravel_index(int(np.argmax(np.abs(last))), last.shape)
        last[i] = last[i] * (1 + delta)
    elif kind == 'raise':
        raise RuntimeError('injected exception (negative control)')


# ---------------------------------------------------------------- gate


def evaluate(F, runs, golden, M, anchors, atol, report, regress_=True):
    """G-R/G-P/G-T/G-A on loaded runs; fills report['fixtures'] and report['gates']; returns ok."""
    G = report['gates']
    ok = True
    keys = {}
    for fx, r in runs.items():
        check_config(F, fx, r['meta'])
        s = r.get('solve')
        if s and s['rc'] == 0:
            want = expected_config(F, fx)
            if s['flags'] != want['flags'] or s['block'] != want['block']:
                raise GateError('E_GATE_CONFIG', f'{fx} solve kernel {s["flags"]}/{s["block"]}')
            if 'spectral' in s and (why := ozone_mismatch(F, fx, s['spectral'])):
                raise GateError('E_GATE_CONFIG', f'{fx} solve: {why}')
        keys[fx], fields = toolkey(r['meta'])
        report['fixtures'][fx] = dict(
            toolkey=keys[fx], toolkey_fields=fields, meta=r['meta'], solve=s, outputs=outputs(r)
        )
    G['R'], G['CUDA'] = {}, {}
    for fx, r in runs.items():
        if r['meta']['backend'] != 'cupy':
            continue
        entries = M['cuda'].get(fx, [])
        same = [e for e in entries if e['toolkey'] == keys[fx]]
        canonical = [e for e in entries if e.get('canonical')]
        # A numerical reference is never selected from the mutable execution records.
        if F['fixtures'][fx]['golden'] is None:
            if len(canonical) != 1:
                raise GateError('E_GATE_BASELINE_MISSING', f'{fx}: exactly one fixed canonical CUDA baseline required')
            e = canonical[0]
            b = baseline_npz(e['file'], e['sha256'])
            tol = dict(fixed=tolerance(r['radiance'], b['fixed_radiance'], atol))
            if 'table' in r:
                tol['table'] = tolerance(r['table'], b['solve_table'], atol)
            good = all(t['pass_'] for t in tol.values())
            ok &= good
            G['CUDA'][fx] = dict(
                pass_=good, blocking=True, tolerance=tol, baseline=e['file'], code=None if good else 'E_GATE_CROSS_TOL'
            )
        if not regress_:
            continue
        e = (same or canonical or [None])[0]
        if e is None:
            raise GateError(
                'E_GATE_BASELINE_MISSING', f'{fx}: no CUDA baseline for toolkey {keys[fx]} and no canonical one'
            )
        b = baseline_npz(e['file'], e['sha256'])
        row = regress(e['outputs'], outputs(r), b['fixed_radiance'], r['radiance'])
        deltas = dict(radiance=tolerance(r['radiance'], b['fixed_radiance'], 0.0))
        if 'table' in r and 'solve_table' in b:
            deltas['table'] = tolerance(r['table'], b['solve_table'], 0.0)
        row.update(
            mode='same-toolkey' if same else 'cross',
            baseline_toolkey=e['toolkey'],
            blocking=False,
            result='identical' if row['pass_'] else 'differs',
            differences=deltas,
            max_abs=max((d['max_abs'] for d in deltas.values() if d['max_abs'] is not None), default=None),
            max_rel=max((d['max_rel'] for d in deltas.values() if d['max_rel'] is not None), default=None),
            code=None if row['pass_'] else 'E_GATE_REGRESS_BYTES',
        )
        G['R'][fx] = row
    G['P'] = {}
    for fx, r in runs.items():
        g = golden.get(F['fixtures'][fx]['golden'])
        if g is None:
            continue
        rad = tolerance(r['radiance'], g['radiance'], atol)
        l2 = l2_by_wavelength(r['source'][-1], g['source'])
        good = rad['pass_'] and bool(np.all(l2 <= RTOL))
        blocking = F['fixtures'][fx].get('physics_blocking', True)
        G['P'][fx] = dict(
            radiance=rad,
            source_l2_max=float(l2.max()),
            pass_=good,
            blocking=blocking,
            code=None if good else 'E_GATE_NONFINITE' if not rad.get('finite', True) else 'E_GATE_PHYSICS_TOL',
        )
        if blocking:
            ok &= good
    values = pairs(F, runs, golden)
    if values:
        good, rows = ratchet_check(values, anchors)
        G['T'] = dict(pass_=good, pairs=rows, code=None if good else 'E_GATE_RATCHET')
        ok &= good
    for fx, r in runs.items():
        ref = F['fixtures'][fx].get('approximates')
        if not ref or ref not in runs:
            continue
        exact = G['P'].get(ref, {}).get('pass_') is True or G['CUDA'].get(ref, {}).get('pass_') is True
        row = dict(exact_gate_passed=bool(exact), fixed=tolerance(r['radiance'], runs[ref]['radiance'], atol))
        good = bool(exact) and row['fixed']['pass_']
        code = None if good else 'E_GATE_APPROX_TOL'
        if 'solve' in r and 'solve' in runs[ref]:
            a, e = r['solve'], runs[ref]['solve']
            if a['rc'] or e['rc']:
                good, code = False, 'E_GATE_APPROX_ITER' if 'CONVERGED' in str(a.get('reason')) else 'E_GATE_SOLVE'
            else:
                row.update(
                    table=tolerance(r['table'], runs[ref]['table'], atol),
                    evaluations=[a['evaluations'], e['evaluations']],
                )
                if not row['table']['pass_']:
                    good, code = False, 'E_GATE_APPROX_TOL'
                elif more_iterations(a['evaluations'], e['evaluations']) or not a['converged']:
                    good, code = False, 'E_GATE_APPROX_ITER'
        row.update(pass_=good, code=code)
        G.setdefault('A', {})[fx] = row
        ok &= good
    for fx, r in runs.items():
        s = r.get('solve')
        if s and s['rc'] and not F['fixtures'][fx].get('approximates'):
            G.setdefault('solve', {})[fx] = dict(pass_=False, code='E_GATE_SOLVE', reason=s.get('reason'))
            ok = False
    report['max_launch_seconds'] = max(
        [report.get('max_launch_seconds', 0.0)]
        + [r['meta']['launch']['max_seconds'] or 0.0 for r in runs.values()]
        + [(r.get('solve') or {}).get('launch_max_seconds') or 0.0 for r in runs.values()]
    )
    return ok


def run_all(F, names, out, backend, solve, base=None, reuse=None):
    if reuse is None:
        for fx in names:
            run_fixture(F, fx, out / fx, backend, solve, base)
    return {fx: load_run(reuse or out, fx) for fx in names}


def historical(out):
    """G-H: cupy reduced selftest vs the record fixtures/selftest_ref_simpson.npz (1e-5);
    the old CPU-engine fixture is only the informational historical_delta (validate.reduced)."""
    if not (paths.FIXTURES / 'selftest_ref_simpson.npz').is_file():
        raise GateError('E_GATE_CONFIG', 'fixtures/selftest_ref_simpson.npz (G-H record) missing')
    # the old CPU-engine fixture (outside git) is optional: without it the informational historical_delta is omitted
    rc, _, code = child(
        [
            sys.executable,
            '-B',
            '-m',
            'sosfuse2.gate',
            '--worker',
            'historical',
            '--fixture',
            'FX-N',
            '--backend',
            'cupy',
            '--out',
            str(out / 'historical'),
        ],
        out / 'historical.log',
        fixture_env(load_fixtures(), 'FX-N'),
    )
    if rc:
        raise GateError('E_GATE_WORKER', f'historical rc={rc} last={code}')
    r = json.loads((out / 'historical/historical.json').read_text())
    return dict(
        pass_=bool(r['pass_']),
        max_relative=r['max_relative'],
        identity_match=r['identity_match'],
        record=r.get('record'),
        historical_delta=r.get('historical_delta'),
        code=None if r['pass_'] else 'E_GATE_HISTORICAL',
    )


def persistent(out):
    env = {k: v for k, v in os.environ.items() if k not in CONTROLS}
    env.update(PERSISTENT_TEST_BACKEND='cupy', PERSISTENT_TEST_DEADLINE='900', PYTHONDONTWRITEBYTECODE='1')
    cmd = [sys.executable, '-B', '-m', 'unittest', '-v'] + [
        f'sosfuse2.tests.test_pipeline_persistent.PersistentTests.{t}' for t in PS_TESTS
    ]
    rc, tail, _ = child(cmd, out / 'persistent.log', env, cwd=paths.SRC, timeout=1800)
    return dict(pass_=rc == 0, rc=rc, tail=tail[-6:], code=None if rc == 0 else 'E_GATE_PERSISTENT')


def gf(args, F, report):
    """Product state on the full grid with bake common() arguments and one cache root: cold, then warm."""
    # G-F reuses the sky-sos bake template: a research-side dependency, reachable only via SOSFUSE2_SKY_SOS_ROOT
    sky = paths.sky_sos_root()
    if sky is None or not (sky / 'rig/bake/bake_run.py').is_file():
        raise GateError('E_GATE_CONFIG', 'G-F needs the sky-sos bake template: set SOSFUSE2_SKY_SOS_ROOT')
    spec = importlib.util.spec_from_file_location('bake_run', sky / 'rig/bake/bake_run.py')
    br = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(br)
    root = args.out / 'gf'
    P = NS(plan=root / 'plan', work=root / 'work')
    if P.work.exists():
        raise GateError('E_GATE_CONFIG', f'{P.work} exists: the cold run needs fresh caches')
    P.plan.mkdir(parents=True, exist_ok=True)
    st = copy.deepcopy(F['state'])
    sid = st['id']
    (P.plan / 'states.json').write_text(json.dumps(dict(states=[st])))
    (P.plan / 'settings.json').write_text('{}')
    env = fixture_env(F, 'FX-C3')
    env.update(STRATO_ATTACH_CACHE=str(P.work / 'cache/attach'), SOSFUSE2_COMPILE_CACHE=str(root / 'cache_compile'))
    M = manifest()
    fixed = M.get('gf') or {}
    reference = args.reference or (Path(fixed['reference']) if fixed.get('reference') else None)
    if reference is not None:
        reference = Path(reference).expanduser()  # the manifest keeps '~/…' (portable())
    if reference is None or not reference.is_file():
        raise GateError(
            'E_GATE_BASELINE_MISSING',
            f'G-F needs the fixed table (--reference); tried {reference if reference is not None else "none (no --reference, manifest gf.reference unset)"}',
        )
    with np.load(reference) as z:
        ref_array = z['radiance']
    ref = sha(ref_array)
    if fixed.get('radiance_sha256') and fixed['radiance_sha256'] != ref:
        raise GateError('E_GATE_MANIFEST_SHA', 'reference radiance is not the manifest gf table')
    if not fixed and not args.decision:
        raise GateError('E_GATE_CONFIG', 'initial G-F reference requires --decision')
    rows = {}
    for label in ('cold', 'warm'):
        cmd = [
            *paths.module_command('pipeline'),
            '--ids',
            sid,
            *br.common(P),
            '--persistent',
            '--workers',
            '1',
            '--gpu-active',
            '1',
        ]
        rc, _, code = child(cmd, root / f'{label}.log', env, timeout=1800)
        if rc:
            raise GateError('E_GATE_WORKER', f'G-F {label} rc={rc} last={code}')
        keep = root / label
        keep.mkdir()
        for p in (
            P.work / 'out' / sid / 'table.npz',
            P.work / 'out' / sid / 'meta.json',
            P.work / 'tmp' / sid / 'source.npz',
        ):
            shutil.copy2(p, keep / p.name)
        summary = json.loads((P.work / 'pipeline_summary.json').read_text())
        logs = sorted(Path(summary['logs']).glob('*.log'))
        m = json.loads((keep / 'meta.json').read_text())
        ev = [
            json.loads(x)
            for p in logs
            for x in p.read_text(errors='replace').splitlines()
            if x.startswith('{"event": "PHASE"')
        ]
        with np.load(keep / 'table.npz') as t, np.load(keep / 'source.npz') as s:
            rows[label] = dict(
                radiance=sha(t['radiance']),
                source=sha(s['source']),
                evaluations=solve_evaluations(m['source_solve']),
                source_solve=hashlib.sha256(
                    json.dumps(numeric(m['source_solve']), sort_keys=True).encode()
                ).hexdigest(),
                launch_max_seconds=max((m['kernel'].get('launch') or {}).get('max_seconds', 0.0), log_launch(logs)),
                attach=[e.get('result') for e in ev if e.get('phase') == 'attach_cache'],
                geometry=[e.get('cache') for e in ev if e.get('phase') == 'geometry_done'],
                column_cache=sorted(
                    {f"{e.get('kind')}|{e.get('result')}" for e in ev if e.get('phase') == 'column_cache'}
                ),
                host_cache=sorted({f"{e.get('kind')}|{e.get('result')}" for e in ev if e.get('phase') == 'host_cache'}),
            )
    c, w = rows['cold'], rows['warm']
    mode = deterministic()
    comparisons = {}
    for filename, key in (('table.npz', 'radiance'), ('source.npz', 'source')):
        with np.load(root / 'cold' / filename) as a, np.load(root / 'warm' / filename) as b:
            comparisons[key] = repeat_arrays(a[key], b[key])
    same = (
        all(c[k] == w[k] for k in ('radiance', 'source', 'source_solve'))
        if mode
        else all(v['pass_'] for v in comparisons.values())
    )
    repeat_max = max((v['max_rel'] for v in comparisons.values() if v['max_rel'] is not None), default=0.0)
    report.update(deterministic=mode, repeat_max_rel=repeat_max)
    with np.load(root / 'cold/table.npz') as t:
        change = tolerance(t['radiance'], ref_array, 0.0)
    good = same and change['pass_']  # --decision cannot waive numerical failure
    report['gates']['F'] = dict(
        pass_=good,
        reference=portable(reference),
        reference_radiance=ref,
        runs=rows,
        change=change,
        bytes_identical=c['radiance'] == ref,
        deterministic=mode,
        repeat_max_rel=repeat_max,
        repeat_comparisons=comparisons,
        evaluations_warning=None if c['evaluations'] == w['evaluations'] else 'W_GF_EVALUATIONS_DIFFER',
        code=None if good else ('E_GATE_GF_BYTES' if mode else 'E_GATE_GF_REPEAT_TOL') if not same else 'E_GATE_GF_TOL',
    )
    report['max_launch_seconds'] = max(
        report.get('max_launch_seconds', 0.0), c['launch_max_seconds'], w['launch_max_seconds']
    )
    if good:
        if not fixed:
            M['gf'] = dict(
                state=sid,
                radiance_sha256=ref,
                reference=portable(reference),
                commit=commit(),
                recorded_utc=utc(),
                decision=args.decision,
            )
        M.setdefault('records', {})['gf'] = dict(
            outputs=c, comparison=report['gates']['F'], commit=commit(), recorded_utc=utc()
        )
        jdump(record('baselines/manifest.json'), M)
        log_line(f'gf run {sid} radiance {c["radiance"][:16]}; fixed {ref[:16]}; reason: {args.reason}')
    return good


def bless(F, names, args, report):
    """Two repeat runs passing fixed numerical gates update execution records only."""
    base = args.staged or args.out
    runs = run_all(F, names, base / 'run1', 'cupy', True, reuse=args.staged and base / 'run1')
    again = run_all(F, names, base / 'run2', 'cupy', True, reuse=args.staged and base / 'run2')
    if args.staged and any(
        r['meta'].get('code_identity') != code_identity() for r in (*runs.values(), *again.values())
    ):
        raise GateError('E_GATE_CONFIG', f'staged runs in {base} were made by other code (code_identity)')
    report['staged'] = str(args.staged) if args.staged else None
    report['determinism'] = {fx: repeat_run(runs[fx], again[fx]) for fx in names}
    report['repeat_max_rel'] = max((v['repeat_max_rel'] for v in report['determinism'].values()), default=0.0)
    if not all(v['pass_'] for v in report['determinism'].values()):
        report['code'] = 'E_GATE_DETERMINISM'
        return False
    M = manifest()
    golden = golden_set(M, goldens(F, names))
    doc = None
    if args.measure_anchors:
        if not args.decision:
            raise GateError('E_GATE_CONFIG', 'measuring fixed anchors needs --decision <user decision ID>')
        values = pairs(F, runs, golden)
        high = {p: n for p, st in values.items() for n, v in flat(st) if not v <= ANCHOR_STOP}
        report['anchors_measured'] = values
        if high:
            report.update(code='E_GATE_ANCHOR_HIGH', anchor_high=high)
            return False
        anchors = values
        doc = dict(
            schema=1,
            measured_utc=utc(),
            commit=commit(),
            decision=args.decision or 'initial (user 2026-10-21, (a) plan)',
            anchors=values,
            rule='block if S > max(3*anchor, 1e-6) (SPEC_V2 3.3); raise only with a user decision ID',
        )
    else:
        anchors = load_anchors()
    if not evaluate(F, runs, golden, M, anchors, args.atol, report, regress_=False):
        return False
    if args.canonical:
        raise GateError(
            'E_GATE_CONFIG',
            '--bless records runs only; fixed canonical replacement needs a separate user-reviewed update',
        )
    for fx in names:
        key, fields = toolkey(runs[fx]['meta'])
        out = outputs(runs[fx])
        entry = dict(
            toolkey=key,
            toolkey_fields=fields,
            outputs=out,
            commit=commit(),
            recorded_utc=utc(),
            pass_=True,
            comparisons={n: g.get(fx) for n, g in report['gates'].items() if n != 'T' and isinstance(g, dict)},
            ratchet={p: v for p, v in report['gates'].get('T', {}).get('pairs', {}).items() if p.startswith(fx + '|')},
            metadata=dict(
                device=runs[fx]['meta']['device'],
                numpy=runs[fx]['meta']['numpy'],
                host_inputs=runs[fx]['meta']['host_inputs_sha256'],
            ),
        )
        M.setdefault('records', {}).setdefault('cuda', {}).setdefault(fx, {})[key] = entry
        log_line(f'cuda run {fx} {key} commit {entry["commit"]}; reason: {args.reason}')
    jdump(record('baselines/manifest.json'), M)
    if doc:
        jdump(record('anchors.json'), doc)
        log_line(f'anchors measured commit {doc["commit"]}; decision: {doc["decision"]}')
    return True


def bless_numpy(F, args, report):
    """Mac golden: E6 validate-mac PASS first; unchanged arrays keep their files."""
    e6 = json.loads(Path(args.e6).read_text()) if args.e6 else {}
    if e6.get('pass') is not True:
        raise GateError('E_GATE_CONFIG', 'E6 validate-mac PASS (--e6 validation.json) is required before a golden')
    names = goldens(F, F['fixtures'])
    runs = run_all(F, names, args.out / 'numpy', 'numpy', False)
    M = manifest()
    (GATE / 'baselines/numpy').mkdir(parents=True, exist_ok=True)
    report['golden'] = {}
    for fx in names:
        check_config(F, fx, runs[fx]['meta'])
        r = runs[fx]
        out = dict(radiance=sha(r['radiance']), source=sha(r['source'][-1]))
        old = M['golden'].get(fx)
        same = bool(old and old['outputs'] == out)
        report['golden'][fx] = dict(outputs=out, same_as_previous=same)
        if same:
            continue
        if not args.decision:
            raise GateError('E_GATE_CONFIG', f'{fx}: changing fixed golden requires --decision')
        np.savez_compressed(GATE / f'baselines/numpy/{fx}.npz', radiance=r['radiance'], source=r['source'][-1])
        M['golden'][fx] = dict(
            file=f'numpy/{fx}.npz',
            sha256=file_sha(GATE / f'baselines/numpy/{fx}.npz'),
            outputs=out,
            commit=commit(),
            numpy=np.__version__,
            e6=dict(sha256=file_sha(args.e6), pass_=True),
            host_inputs=r['meta']['host_inputs_sha256'],
            blessed_utc=utc(),
        )
        log_line(
            f'numpy golden {fx} commit {M["golden"][fx]["commit"]}: {"new" if not old else "CHANGED"}; reason: {args.reason}'
        )
    jdump(record('baselines/manifest.json'), M)
    return True


# ---------------------------------------------------------------- rig bake hooks


def plan_check(order, states, settings, reference):
    """Before MPS/precompile: the first state is the product fixture state, settings {} and the eda reference table."""
    F = load_fixtures()
    M = manifest()
    want = F['state']['id']
    bad = []
    if not order or order[0] != want:
        bad.append(f'first state {order[:1]} != product {want}')
    elif {k: v for k, v in states.get(want, {}).items() if k != 'runnable'} != {
        k: v for k, v in F['state'].items() if k != 'runnable'
    }:
        bad.append('product state definition differs from gate/fixtures.json')
    if settings != {}:
        bad.append('settings.json must be {} (the reference table uses code defaults)')
    ref = (M.get('gf') or {}).get('radiance_sha256')
    if not ref:
        bad.append('manifest gf.radiance_sha256 missing')
    elif not Path(reference).is_file():
        bad.append(f'{reference} missing (eda G-F table, R2)')
    else:
        with np.load(reference) as z:
            if sha(z['radiance']) != ref:
                bad.append('reference radiance sha256 differs from manifest gf')
    if bad:
        raise SystemExit('E_GATE_FIRST_STATE: ' + '; '.join(bad))


def first_state(table, reference, atol=0.0):
    """After the first state: radiance vs the eda table, rtol 1e-3 + atol; S_max/S_rms per elevation recorded."""
    with np.load(table) as a, np.load(reference) as b:
        x, r, hs = a['radiance'], b['radiance'], b['solar_elevation_deg']
    t = tolerance(x, r, atol)
    st = elevation_stats(x, r, hs) if x.shape == r.shape else {}
    return dict(
        pass_=t['pass_'],
        tolerance=t,
        S_max=st.get('S_max'),
        S_rms=st.get('S_rms'),
        shape=list(x.shape),
        code=None if t['pass_'] else 'E_GATE_FIRST_STATE',
    )


# ---------------------------------------------------------------- CLI


def selected(F, args):
    if args.flagsets is not None:
        sets = set(args.flagsets.split(','))
        unknown = sets - {v['config'] for v in F['fixtures'].values()}
        if unknown:
            raise GateError('E_GATE_CONFIG', 'unknown flagsets: ' + ','.join(sorted(unknown)))
        names = [fx for fx, v in F['fixtures'].items() if v['config'] in sets]
    else:
        names = list(args.fixtures.split(',') if args.fixtures is not None else F['fixtures'])
        unknown = set(names) - set(F['fixtures'])
        if unknown:
            raise GateError('E_GATE_CONFIG', 'unknown fixtures: ' + ','.join(sorted(unknown)))
    if not names:
        raise GateError('E_GATE_CONFIG', 'no fixtures selected')
    return names


def first_code(G):
    for row in G.values():
        rows = row.values() if isinstance(row, dict) and 'pass_' not in row else [row]
        for x in rows:
            if isinstance(x, dict) and x.get('code') and x.get('blocking', True):
                return x['code']
    return None


def dispatch(args, report):
    F = load_fixtures()
    names = selected(F, args)
    M = manifest()
    report['selected'] = names
    if args.inject == 'raise':
        inject('raise', None, None, F, None)
    if args.bless_numpy:
        return bless_numpy(F, args, report)
    if args.bless:
        if args.inject or args.reuse:
            raise GateError('E_GATE_CONFIG', 'an injected or reused run cannot be blessed')
        return bless(F, names, args, report)
    if args.stage:  # fresh runs for a later --bless --staged (no golden needed)
        rs = [run_all(F, names, args.out / f'run{i + 1}', 'cupy', True) for i in range(args.runs)]
        a = rs[0]
        report['determinism'] = {fx: [repeat_run(a[fx], r[fx]) for r in rs[1:]] for fx in names}
        report['repeat_max_rel'] = max(
            (v['repeat_max_rel'] for rows in report['determinism'].values() for v in rows), default=0.0
        )
        ok = all(v['pass_'] for rows in report['determinism'].values() for v in rows)
        report['fixtures'] = {
            fx: dict(
                toolkey=toolkey(a[fx]['meta']), outputs=outputs(a[fx]), solve=a[fx].get('solve'), meta=a[fx]['meta']
            )
            for fx in names
        }
        report['max_launch_seconds'] = max(
            [r['meta']['launch']['max_seconds'] for r in a.values()]
            + [(r.get('solve') or {}).get('launch_max_seconds') or 0.0 for r in a.values()]
        )
        if not ok:
            report['code'] = 'E_GATE_DETERMINISM'
        if not args.no_historical:
            report['gates']['H'] = historical(args.out)
            ok &= report['gates']['H']['pass_']
        if not args.no_persistent:
            report['gates']['PS'] = persistent(args.out)
            ok &= report['gates']['PS']['pass_']
        return ok
    if args.gf:
        return gf(args, F, report)
    golden = golden_set(M, goldens(F, names))
    anchors = load_anchors()
    runs = run_all(F, names, args.out, 'cupy', True, reuse=args.reuse)
    if args.inject:
        inject(args.inject, runs, golden, F, anchors)
    ok = evaluate(F, runs, golden, M, anchors, args.atol, report)
    if args.all and not args.reuse:
        if not args.no_historical:
            report['gates']['H'] = historical(args.out)
            ok &= report['gates']['H']['pass_']
        if not args.no_persistent:
            report['gates']['PS'] = persistent(args.out)
            ok &= report['gates']['PS']['pass_']
    return ok


def baselines_writable():
    """True when GATE/baselines (or, if it does not exist yet, its nearest existing parent) accepts new files."""
    d = GATE / 'baselines'
    while not d.exists() and d != d.parent:
        d = d.parent
    return os.access(d, os.W_OK)


def main(argv=None):
    os.environ.setdefault('SOSFUSE2_KSPLIT_FALLBACK', '0')
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    for m in ('all', 'pregate', 'stage', 'bless', 'bless_numpy', 'gf'):
        mode.add_argument('--' + m.replace('_', '-'), action='store_true')
    mode.add_argument('--worker', choices=('fixed', 'historical'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--fixture')
    p.add_argument('--backend', choices=('cupy', 'numpy'), default='cupy')
    p.add_argument('--fixtures', help='comma list (default all)')
    p.add_argument('--flagsets', help='pregate: plan flag sets, e.g. C3,C3_nostrato')
    p.add_argument(
        '--reuse', type=Path, help='compare stored fixture outputs instead of running them (negative controls)'
    )
    p.add_argument('--staged', type=Path, help='--bless from a --stage output (run1, run2) made by identical code')
    p.add_argument(
        '--runs', type=int, default=2, help='--stage: fresh runs (tol by default, bytes in deterministic mode)'
    )
    p.add_argument('--inject', choices=('bytes', 'golden', 'raise'))
    p.add_argument('--atol', type=float, default=0.0)
    p.add_argument('--measure-anchors', action='store_true')
    p.add_argument('--decision')
    p.add_argument('--canonical', action='store_true')
    p.add_argument('--reason', default='')
    p.add_argument('--e6', type=Path)
    p.add_argument('--reference', type=Path)
    p.add_argument('--no-historical', action='store_true')
    p.add_argument('--no-persistent', action='store_true')
    p.add_argument(
        '--record-baselines',
        action='store_true',
        help='write manifest/anchors/log updates into the package (maintenance in a writable checkout); default: <out>/records',
    )
    args = p.parse_args(argv)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    global RECORDS
    previous = (
        RECORDS  # restored below: the setting must not outlive this call (in-process callers, pytest -k / random order)
    )
    RECORDS = None if (args.bless or args.bless_numpy or args.record_baselines) else args.out / 'records'
    try:
        return run_main(p, args, argv)
    finally:
        RECORDS = previous


def run_main(p, args, argv):
    if args.worker:
        return worker(args.worker, args.fixture, args.out, args.backend)
    if not (np.isfinite(args.atol) and args.atol >= 0):
        p.error('--atol must be finite and nonnegative')
    report = dict(
        schema=2,
        utc=utc(),
        commit=commit(),
        argv=sys.argv[1:] if argv is None else list(argv),
        injected=args.inject,
        reused=str(args.reuse) if args.reuse else None,
        rtol=RTOL,
        atol=args.atol,
        fixtures={},
        gates={},
        code=None,
        code_sha256={n: file_sha(HERE / n) for n in ('kernel.cu', 'fused.py', 'ksplit.py', 'reference.py', 'gate.py')},
        env={k: v for k, v in os.environ.items() if k.startswith(('SOSFUSE2_', 'CUDA_VISIBLE'))},
    )
    try:
        try:
            report['deterministic'] = deterministic()
        except ValueError as e:
            raise GateError('E_DETERMINISTIC', str(e)) from e
        if (
            RECORDS is None and not baselines_writable()
        ):  # --bless/--bless-numpy/--record-baselines write the package's own files
            raise GateError(
                'E_GATE_CONFIG',
                f'package baselines not writable: {GATE / "baselines"} (use a writable checkout, or drop --bless/--bless-numpy/--record-baselines)',
            )
        ok = dispatch(args, report)
        report['code'] = report['code'] or first_code(report['gates'])
        rc = 0 if ok else 1
    except GateError as e:
        report.update(code=e.code, error=str(e))
        rc = 2
    except Exception as e:  # noqa: BLE001 - reason preserved, exit 2
        report.update(
            code='E_GATE_EXCEPTION', error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc()[-4000:]
        )
        rc = 2
    report['exit'] = rc
    jdump(args.out / 'gate.json', report)
    print(
        'GATE '
        + json.dumps(
            dict(
                exit=rc,
                code=report['code'],
                error=report.get('error'),
                max_launch_seconds=report.get('max_launch_seconds'),
            )
        ),
        flush=True,
    )
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
