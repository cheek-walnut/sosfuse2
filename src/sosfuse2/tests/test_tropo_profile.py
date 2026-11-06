#!/usr/bin/env python3
"""Two-layer tropospheric aerosol (tropo_profile.py, kernel.cu -DAEROSOL_2L, rig/sh_cpu.cpp): Mac checks.

  --static    clang++ -E -P token identity of kernel.cu with AEROSOL_2L off (merge-base vs this tree) over the production
              flag sets and launched specializations; AEROSOL_2L on preprocesses and passes the stubbed syntax check;
              point() compiled for the host with -DAEROSOL_2L: s.prof[1] vs reference.sample (FP32 mirror).
  --host      block validation codes; profile integral = aod550 (closed form, and through the patched sos_sky.profiles).
  --paths     subprocess probes (in parallel) of the small CPU paths: NumPy reference (selftest grid), create_configured
              host arrays, aerosol_column, the C++ engine (rig/sh_cpu.cpp, tiny grid, built in <out>):
                identity    no block and status "disabled": this tree == merge-base tree (git archive), bytes and cache keys
                equivalence a two-layer block equal to exp(-h/1.2) in exact arithmetic, on a state whose scale_height_km is
                            the decoy 3.0, vs the exp 1.2 state; the exp 3.0 state is the control (what a missed path gives)
                default     the default block computed on every small path, compared with exp 1.2
usage: python -B -m sosfuse2.tests.test_tropo_profile --static --host --paths --out <dir> [--jobs 6]
"""

from sosfuse2 import paths
import argparse, copy, hashlib, itertools, json, math, os, re, shutil, subprocess, sys, tarfile, io, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
HERE = paths.PKG  # the package directory (kernel, configs, assets)
from sosfuse2 import testing as T

ROOT = (
    T.sky_sos_root()
)  # sky-sos checkout: this harness compares against its git baseline tree and legacy rig/ entry points (pre-S4 layout)
TREE_FILES = (
    'scripts',
    'rig/engine.py',
    'rig/sh_cpu.cpp',
    'rig/states_v1.json',
    'rig/docker/geometry_cpu.py',
    'rig/pack_state.py',
    'results/sos_full/a2_driver.py',
    'results/sos_full/b2_driver.py',
    'assets',
)
WL = (450.0, 550.0, 650.0)
CELLS = (0, 7, 12, 25, 39)


def base_ref():
    if ROOT is None:
        return 'main'
    try:
        return subprocess.check_output(['git', '-C', str(ROOT), 'merge-base', 'HEAD', 'main'], text=True).strip()[:12]
    except (OSError, subprocess.CalledProcessError):
        return 'main'


# ---- static: kernel tokens and the host-compiled point() --------------------------------------------------------


def static(out):
    from sosfuse2.tests.test_tropo_mie import C3, C2OFF, SPECS, SYNTAX_STUB_EXTRA, tokens, defines, preprocess
    from sosfuse2.tests.test_kernel_static import STUB

    base = out / f'kernel_{base_ref()}.cu'
    base.write_text(
        subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{base_ref()}:scripts/sosfuse2/kernel.cu'], text=True)
    )
    new = HERE / 'kernel.cu'
    sets = {
        'C3_strato': C3 + ('STRATO', 'PROFILE'),
        'C3_strato_raylimit': C3 + ('STRATO', 'PROFILE', 'LAUNCH_RAY_LIMIT'),
        'C2off': C2OFF + ('PROFILE',),
        'C3_strato_mie': C3 + ('STRATO', 'PROFILE', 'TROPO_MIE'),
    }
    jobs = [
        (n, f, sp)
        for (n, f), sp in itertools.product(sets.items(), SPECS)
        if n != 'C3_strato_raylimit' or sp == (1, 0, 0)
    ]
    stub = out / 'cuda_stub.h'
    stub.write_text(STUB + SYNTAX_STUB_EXTRA)

    def one(job):
        name, flags, spec = job
        ra, a, _ = preprocess(base, flags, spec)
        rb, b, _ = preprocess(new, flags, spec)
        ron, on, eon = preprocess(new, flags + ('AEROSOL_2L',), spec)
        syn = subprocess.run(
            [
                'clang++',
                '-std=c++17',
                '-ffp-contract=off',
                '-fsyntax-only',
                '-x',
                'c++',
                '-include',
                str(stub),
                *defines(flags + ('AEROSOL_2L',), spec),
                str(new),
            ],
            capture_output=True,
            text=True,
        )
        return dict(
            set=name,
            spec=list(spec),
            off_tokens_identical=ra == 0 and rb == 0 and tokens(a) == tokens(b),
            on_rc=ron,
            on_override='s.prof[1]=aerosol_two_layer(s.h,sun_h+nsh)' in on.replace(' ', ''),
            on_syntax_rc=syn.returncode,
            on_error=(eon + syn.stderr).strip()[-300:],
        )

    with ThreadPoolExecutor(4) as pool:
        rows = list(pool.map(one, jobs))
    lut = ('STRATO', 'STRATO_MIE_LUT', 'STRATO_MIE_LUT_FINE')
    ra, a, _ = preprocess(base, lut, None)
    rb, b, _ = preprocess(new, lut, None)
    rows.append(
        dict(
            set='strato_lut_build',
            spec=None,
            off_tokens_identical=ra == 0 and rb == 0 and tokens(a) == tokens(b),
            on_rc=0,
            on_override=True,
            on_syntax_rc=0,
            on_error='',
        )
    )
    host = host_point(out, STUB)
    ok = all(
        r['off_tokens_identical'] and r['on_rc'] == 0 and r['on_override'] and r['on_syntax_rc'] == 0 for r in rows
    )
    return dict(
        pass_=ok and host['pass_'],
        baseline=f'git show {base_ref()}:scripts/sosfuse2/kernel.cu',
        kernels=rows,
        host_point=host,
        scope='clang preprocessing + stubbed host syntax; point() host build for numerics; NVRTC/CUDA is tested on GPU hosts',
    )


def host_point(out, STUB):
    """kernel.cu point() built for the host with -DAEROSOL_2L vs reference.sample on a two-layer selftest state."""
    import ctypes, numpy as np
    from sosfuse2 import reference, tropo_profile as tp
    from sosfuse2.geometry import create, TransientGeometry, normal_state

    kernel = (HERE / 'kernel.cu').resolve()
    cpp = out / 'aer2_host.cpp'
    lib = out / 'aer2_host.dylib'
    # <math.h> first: the stub's later expf/sqrtf/... prototypes then keep C linkage and link against libm.
    cpp.write_text(
        '#include <math.h>\n'
        + STUB
        + '\nDim blockIdx{},threadIdx{},blockDim{};extern "C" { double shared[50000]; }\n'
        + f'#include "{kernel}"\n'
        + '''
extern "C" void sample_c(const double*p,const double*d,const float*B,const float*dl,double a,double b,int lower,double gu,double gw,
 const float*h,int nh,const float*chi,int nc,float height,const float*sun_h,int nsh,const float*sun_col,int nsx,float*out){
 Sample s{};point(s,p,d,B,dl,0.f,a,b,lower,gu,gw,h,nh,chi,nc,height,sun_h,nsh,sun_col,nsx,false);out[0]=s.h;out[1]=s.prof[1];}
'''
    )
    link = (
        ['-dynamiclib'] if sys.platform == 'darwin' else ['-shared', '-fPIC', '-nostdinc++']
    )  # glibc: libstdc++'s <math.h> pulls in int64_t=long, which clashes with kernel.cu's NVRTC typedef (long long); the C header does not
    subprocess.run(
        [
            'clang++',
            '-std=c++17',
            '-O2',
            '-ffp-contract=off',
            '-DFW=1',
            '-DAEROSOL_2L',
            *link,
            str(cpp),
            '-o',
            str(lib),
        ],
        check=True,
    )
    native = ctypes.CDLL(str(lib))
    P = ctypes.c_void_p
    I = ctypes.c_int
    D = ctypes.c_double
    F = ctypes.c_float
    native.sample_c.argtypes = [P, P, P, P, D, D, I, D, D, P, I, P, I, F, P, I, P, I, P]
    state = copy.deepcopy(normal_state())
    state['troposphere_profile'] = dict(status='enabled', model='two_layer')
    s, scope, _ = create(out / 'host_point', 'selftest', (550.0,), 'numpy', 128, state=state)
    g = TransientGeometry(s, scope).cell(12)
    c = s.kernel.setup(s, g['nq'])
    sun_h = np.ascontiguousarray(np.r_[c['sun_h'], c['aer2']], 'f4')
    maxabs = maxrel = 0.0
    n = 0
    hmax = 0.0
    for seg in range(g['edges'].shape[1] - 1):
        for step in range(len(c['u'])):
            q = reference.sample(g, seg, step, c)
            for row in np.flatnonzero(q['valid']):
                arrays = [
                    np.ascontiguousarray(g[k][row], t)
                    for k, t in (('p', 'f8'), ('d', 'f8'), ('target', 'f4'), ('dloc', 'f4'))
                ]
                o = np.empty(2, 'f4')
                native.sample_c(
                    *[v.ctypes.data for v in arrays],
                    g['edges'][row, seg],
                    g['edges'][row, seg + 1],
                    int(g['lower'][row, seg]),
                    c['u'][step],
                    c['wu'][step],
                    c['h'].ctypes.data,
                    len(c['h']),
                    c['chi'].ctypes.data,
                    len(c['chi']),
                    float(c['height']),
                    sun_h.ctypes.data,
                    len(c['sun_h']),
                    c['sun_col'].ctypes.data,
                    c['sun_col'].shape[1],
                    o.ctypes.data,
                )
                hmax = max(hmax, float(abs(o[0] - q['h'][row])))
                ref = float(q['prof'][row, 1])
                maxabs = max(maxabs, abs(float(o[1]) - ref))
                maxrel = max(maxrel, abs(float(o[1]) / ref - 1))
                n += 1
    f64 = float(
        np.max(
            abs(
                tp.profile32(np.linspace(0, 80, 4001), c['aer2'])
                / tp.profile(np.linspace(0, 80, 4001), s.aerosol_profile['vector'])
                - 1
            )
        )
    )
    return dict(
        samples=n,
        height_max_abs=hmax,
        prof1_max_abs=maxabs,
        prof1_max_rel=maxrel,
        fp32_mirror_vs_fp64_max_rel=f64,
        flags=list(s.kernel.flags),
        pass_=n > 0 and hmax <= 1e-5 and maxrel <= 1e-6 and 'AEROSOL_2L' in s.kernel.flags,
        rule='host point() prof[1] within 1e-6 relative of the FP32 NumPy mirror (libm expf vs NumPy exp), s.h within 1e-5 km',
    )


# ---- host: validation and integrals ------------------------------------------------------------------------------


def host(out):
    import numpy as np
    from sosfuse2 import tropo_profile as tp
    from sosfuse2.geometry import create, normal_state

    base = copy.deepcopy(normal_state())
    good = dict(status='enabled', model='two_layer')
    rows = []
    for name, block, aod, code in (
        ('not an object', 'two', 0.15, 'E_TROPO_PROFILE'),
        ('no status', dict(model='two_layer'), 0.15, 'E_TROPO_PROFILE_STATUS'),
        ('status on', dict(good, status='on'), 0.15, 'E_TROPO_PROFILE_STATUS'),
        ('model exp', dict(good, model='exp'), 0.15, 'E_TROPO_PROFILE_MODEL'),
        ('unknown key', dict(good, bl_top=2), 0.15, 'E_TROPO_PROFILE_KEYS'),
        ('shape', dict(good, bl_shape='linear'), 0.15, 'E_TROPO_PROFILE_SHAPE'),
        ('uniform with H', dict(good, bl_shape='uniform', bl_scale_height_km=1.0), 0.15, 'E_TROPO_PROFILE_SHAPE'),
        ('top 0', dict(good, bl_top_km=0), 0.15, 'E_TROPO_PROFILE_VALUE'),
        ('top 80', dict(good, bl_top_km=80), 0.15, 'E_TROPO_PROFILE_VALUE'),
        ('bool', dict(good, ft_aod550=True), 0.15, 'E_TROPO_PROFILE_VALUE'),
        ('nan', dict(good, ft_scale_height_km=float('nan')), 0.15, 'E_TROPO_PROFILE_VALUE'),
        ('ft H 0', dict(good, ft_scale_height_km=0.0), 0.15, 'E_TROPO_PROFILE_VALUE'),
        ('ft aod < 0', dict(good, ft_aod550=-0.01), 0.15, 'E_TROPO_PROFILE_VALUE'),
        ('ft aod > aod', good, 0.03, 'E_TROPO_PROFILE_AOD'),
        ('width 0', dict(good, bl_transition_km=0), 0.15, 'E_TROPO_PROFILE_VALUE'),
        ('width below ground', dict(good, bl_top_km=0.1, bl_transition_km=0.3), 0.15, 'E_TROPO_PROFILE_VALUE'),
        ('explicit null', None, 0.15, 'E_TROPO_PROFILE'),
        ('empty object', {}, 0.15, 'E_TROPO_PROFILE_STATUS'),
    ):
        st = copy.deepcopy(base)
        st['troposphere_profile'] = block
        st['parameters']['aod550'] = aod
        try:
            tp.config(st)
            rows.append(dict(case=name, expected=code, got=None, pass_=False))
        except ValueError as e:
            rows.append(dict(case=name, expected=code, got=str(e)[:100], pass_=str(e).split(':')[0] == code))
    from sosfuse2 import tropo_mie

    for name, block, code in (
        ('phase explicit null', None, 'E_TROPO_PHASE'),
        ('phase empty object', {}, 'E_TROPO_PHASE_STATUS'),
    ):
        try:
            tropo_mie.config(dict(base, troposphere_phase=block))
            rows.append(dict(case=name, expected=code, got=None, pass_=False))
        except ValueError as e:
            rows.append(dict(case=name, expected=code, got=str(e)[:100], pass_=str(e).split(':')[0] == code))
    accepted = dict(
        absent=tp.config(base) is None,
        disabled=tp.config(dict(base, troposphere_profile=dict(status='disabled', junk=1))) is None,
        uniform=tp.config(dict(base, troposphere_profile=dict(good, bl_shape='uniform')))['bl_scale_height_km'] is None,
        ft_equal_aod=tp.config(dict(base, troposphere_profile=dict(good, ft_aod550=0.15))) is not None,
    )
    integrals = []
    for name, block in (
        ('default', good),
        ('uniform', dict(good, bl_shape='uniform', bl_top_km=1.5)),
        ('ft only', dict(good, ft_aod550=0.15)),
        ('thin ft', dict(good, ft_aod550=0.0, ft_scale_height_km=8.0)),
    ):
        st = copy.deepcopy(base)
        st['troposphere_profile'] = block
        cfg = tp.config(st)
        v = tp.parameters(cfg)['vector']
        z, hw = cfg['bl_top_km'], cfg['bl_transition_km'] / 2
        cuts = (0, z - hw, z + hw, 10, 30, 80)
        closed = tp.layer_integrals(cfg)
        x, w = np.polynomial.legendre.leggauss(200)

        def quad(f, a, b):
            return float(np.sum(f((a + b) / 2 + (b - a) / 2 * x) * w) * (b - a) / 2)

        whole = lambda f: sum(quad(f, a, b) for a, b in zip(cuts[:-1], cuts[1:]))
        num = cfg['aod550'] * whole(lambda h: tp.profile(h, v))
        blc = lambda h: (
            v[2] * np.exp(-h * v[3]) * (1 - tp.smoothstep(np.clip((h - v[0]) * v[1], 0, 1)))
        )  # boundary-layer component alone
        integrals.append(
            dict(
                case=name,
                aod550=cfg['aod550'],
                bl=closed['bl'],
                ft=closed['ft'],
                bl_expected=cfg['aod550'] - cfg['ft_aod550'],
                closed_sum_rel=abs(closed['bl'] + closed['ft'] - cfg['aod550']) / cfg['aod550'],
                gl_rel=abs(num - cfg['aod550']) / cfg['aod550'],
                bl_gl_abs=abs(cfg['aod550'] * whole(blc) - (cfg['aod550'] - cfg['ft_aod550'])),
            )
        )
    # The patched sos_sky.profiles (engine source patch) and the solver's own column quadrature.
    st = copy.deepcopy(base)
    st['troposphere_profile'] = good
    s, scope, _ = create(out / 'host_integral', 'selftest', (550.0,), 'numpy', 128, state=st)
    old = scope['b'].sh.old
    aer = old.NORMAL['aer']
    c = s.aerosol_profile['config']
    z, hw = c['bl_top_km'], c['bl_transition_km'] / 2
    x, w = np.polynomial.legendre.leggauss(200)
    quad = lambda a, b: float(np.sum(old.profiles((a + b) / 2 + (b - a) / 2 * x)[:, 1] * w) * (b - a) / 2)
    patched = aer * sum(quad(a, b) for a, b in ((0, z - hw), (z - hw, z + hw), (z + hw, 10), (10, 80)))

    def columns(old, aer):
        rows = []
        for zen in (0.0, 60.0, 80.0, 85.0):
            mu = math.cos(math.radians(zen))
            p = np.array([[0.0, 0.0, 6371.0]])
            d = np.array([[math.sin(math.radians(zen)), 0.0, mu]])
            solver = float(old.total_columns(p, d, 24)[0, 1]) * aer
            edges = old.probe.boundaries(p, d)[0]
            t = np.linspace(0, edges[-1], 400001)
            h = np.linalg.norm(p[0] + t[:, None] * d[0], axis=1) - 6371.0
            dense = float(np.trapezoid(old.profiles(h)[:, 1], t)) * aer
            rows.append(dict(zenith_deg=zen, solver_gl24=solver, dense_trapezoid=dense, solver_rel=solver / dense - 1))
        return rows

    _, scope_exp, _ = create(out / 'host_integral_exp', 'selftest', (550.0,), 'numpy', 128, state=copy.deepcopy(base))
    by_width = {}
    for width in (0.1, 0.2, 0.3, 2.0):  # the default (1.0 km) is solver_columns
        sw = copy.deepcopy(base)
        sw['troposphere_profile'] = dict(good, bl_transition_km=width)
        _, scope_w, _ = create(out / f'host_integral_w{width}', 'selftest', (550.0,), 'numpy', 128, state=sw)
        by_width[str(width)] = [
            round(r['solver_rel'], 6) for r in columns(scope_w['b'].sh.old, scope_w['b'].sh.NORMAL['aer'])
        ]
    integ = dict(
        cases=integrals,
        patched_sos_sky_profiles_rel=abs(patched / st['parameters']['aod550'] - 1),
        normal_aer=aer,
        solver_columns=columns(old, aer),
        solver_columns_exp12=columns(scope_exp['b'].sh.old, scope_exp['b'].sh.NORMAL['aer']),
        solver_rel_by_transition_width=dict(by_width, zenith_deg=[0, 60, 80, 85]),
        note='solver_columns: GL-24 per segment across the default 1 km blend (no shell at bl_top); reported, not '
        'gated. exp12 = the same quadrature on the exponential 1.2 km profile',
    )
    # LOWTRAN class rule: defaults = V <= 23 km class; the explicit Lee-comparison states carry the class of their V.
    lt = {str(v): [round(x, 4) for x in tp.lowtran_free_troposphere(v)] for v in (10.0, 23.0, 27.0, 50.0)}
    states = json.loads((ROOT / 'rig/states_tropo2l_g065.json').read_text())['states']
    explicit = []
    for st2 in states:
        cfg = tp.config(st2)
        vis = 3.912 / (st2['parameters']['aod550'] / 1.2 + 0.0116)
        tau, H = tp.lowtran_free_troposphere(vis)
        explicit.append(
            dict(
                id=st2['id'],
                vis_km=round(vis, 1),
                ft_aod550=cfg['ft_aod550'],
                ft_scale_height_km=cfg['ft_scale_height_km'],
                pass_=cfg['ft_aod550'] == round(tau, 4) and cfg['ft_scale_height_km'] == round(H, 2),
            )
        )
    lowtran = dict(
        by_visibility_km=lt,
        states=explicit,
        defaults_match_v23=[tp.DEFAULTS['ft_aod550'], tp.DEFAULTS['ft_scale_height_km']]
        == [lt['23.0'][0], round(lt['23.0'][1], 2)],
    )
    lowtran['pass_'] = (
        lowtran['defaults_match_v23']
        and all(r['pass_'] for r in explicit)
        and len(explicit) == 2
        and lt['50.0'] == [0.0429, 2.3877]
    )
    # Gate (coordinator 2026-10-07): with the default 1 km blend the GL-24 column error is within 0.5 % at every zenith.
    integ['solver_columns_gate'] = dict(
        rule='|solver_rel| <= 0.005 at zenith 0/60/80/85 deg, default block',
        pass_=all(abs(r['solver_rel']) <= 0.005 for r in integ['solver_columns']),
    )
    ok = (
        all(r['pass_'] for r in rows)
        and all(accepted.values())
        and all(
            r['closed_sum_rel'] < 1e-14
            and r['gl_rel'] < 1e-12
            and abs(r['bl'] - r['bl_expected']) < 1e-15
            and r['bl_gl_abs'] < 1e-14
            for r in integrals
        )
        and integ['patched_sos_sky_profiles_rel'] < 1e-12
        and lowtran['pass_']
        and integ['solver_columns_gate']['pass_']
    )
    guard = legacy(out)
    return dict(
        pass_=ok and guard['pass_'], rejects=rows, accepted=accepted, integrals=integ, lowtran=lowtran, legacy=guard
    )


# Entry points that call a create() which ignores troposphere_profile / troposphere_phase (rig/engine.py and its
# sosfast/sosfuse relatives): each must refuse such a state (E_LEGACY_UNSUPPORTED_BLOCK) before computing anything.
LEGACY_READERS = ('rig/run_state.py', 'rig/run_state_v2.py', 'rig/run_state_v3.py')  # read_state(path, id)
LEGACY_CLIS = (
    'scripts/sosfuse/run_gpu_state.py',
    'scripts/sosfuse/run_gpu_state_v2.py',
    'scripts/sosfuse/run_gpu_state_v3.py',
)
LEGACY_GUARDED = (
    LEGACY_READERS
    + LEGACY_CLIS
    + ('scripts/sosfuse/run_fuse.py', 'rig/docker/geometry_cpu.py', 'results/column_parallel_20261007/measure.py')
)


def legacy(out):
    from sosfuse2 import tropo_profile as tp
    from sosfuse2.geometry import normal_state

    folder = out / 'legacy'
    folder.mkdir(parents=True, exist_ok=True)
    base = dict(copy.deepcopy(normal_state()), runnable=True)
    states = {
        'p2l': dict(troposphere_profile=dict(status='enabled', model='two_layer')),
        'pmie': dict(troposphere_phase=dict(status='enabled', model='mie', optics_table='default')),
        'pnull': dict(troposphere_profile=None),
        'pdisabled': dict(troposphere_profile=dict(status='disabled'), troposphere_phase=dict(status='disabled')),
    }
    sf = folder / 'states.json'
    sf.write_text(json.dumps(dict(states=[dict(base, id=k, **v) for k, v in states.items()])))
    code = 'E_LEGACY_UNSUPPORTED_BLOCK'

    def reader(job):
        path, sid = job
        prog = (
            f"import sys,pathlib;sys.path.insert(0,{str((ROOT / path).parent)!r});import {Path(path).stem} as m\n"
            f"try:\n m.read_state(pathlib.Path({str(sf)!r}),{sid!r});print('NO_ERROR')\nexcept Exception as e:print(str(e)[:120])"
        )
        r = subprocess.run(
            [sys.executable, '-B', '-c', prog], capture_output=True, text=True, cwd=str(ROOT / path).rsplit('/', 1)[0]
        )
        return dict(entry=path, state=sid, result=(r.stdout.strip() or r.stderr.strip()[-160:]))

    def cli(job):
        path, sid = job
        r = subprocess.run(
            [
                sys.executable,
                '-B',
                str(ROOT / path),
                '--id',
                sid,
                '--states',
                str(sf),
                '--work-root',
                str(folder / 'work'),
                '--backend',
                'numpy',
                '--reduced',
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        return dict(entry=path, state=sid, result=(r.stderr.strip().splitlines() or ['rc %d' % r.returncode])[-1][:160])

    jobs = [(reader, (p, s)) for p in LEGACY_READERS for s in states] + [
        (cli, (p, s)) for p in LEGACY_CLIS for s in ('p2l', 'pmie')
    ]
    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(lambda j: j[0](j[1]), jobs))
    for r in rows:
        r['pass_'] = (code not in r['result']) if r['state'] == 'pdisabled' else (code in r['result'])
    static_calls = {p: 'reject_legacy(' in (ROOT / p).read_text() for p in LEGACY_GUARDED}
    unit = []
    for name, st, want in (
        ('absent', {}, None),
        ('disabled', states['pdisabled'], None),
        ('enabled', states['p2l'], code),
        ('phase', states['pmie'], code),
        ('null', states['pnull'], code),
        ('empty', dict(troposphere_phase={}), code),
    ):
        try:
            tp.reject_legacy(st, 'test')
            got = None
        except ValueError as e:
            got = str(e).split(':')[0]
        unit.append(dict(case=name, expected=want, got=got, pass_=got == want))
    return dict(
        pass_=all(r['pass_'] for r in rows) and all(static_calls.values()) and all(u['pass_'] for u in unit),
        entry_points=rows,
        guarded_sources=static_calls,
        unit=unit,
        note='disabled blocks may still fail later for other reasons (e.g. E_LAYER_SUPPORT); only the guard code is judged',
    )


# ---- paths: subprocess probes -------------------------------------------------------------------------------------


def extract_base(out):
    folder = out / f'tree_{base_ref()}'
    if not (folder / 'scripts/sosfuse2/kernel.cu').exists():
        data = subprocess.check_output(['git', '-C', str(ROOT), 'archive', base_ref(), *TREE_FILES])
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            t.extractall(folder, filter='data')
    return folder


def build_native(tree, folder):
    """rig/engine.py compile_kernel's command for <tree>/rig/sh_cpu.cpp, output in <folder> (never the tree's rig/build)."""
    compiler = shutil.which('g++') or shutil.which('clang++')
    target = folder / 'kernel_t2.so'
    if not target.exists():
        subprocess.run(
            [
                compiler,
                '-std=c++17',
                '-O3',
                '-march=native',
                '-fPIC',
                '-shared',
                '-pthread',
                '-DRIG_THREADS=2',
                str(Path(tree) / 'rig/sh_cpu.cpp'),
                '-o',
                str(target),
            ],
            check=True,
        )
    return target


def digest(a):
    import numpy as np

    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes() + str((a.dtype.str, a.shape)).encode()).hexdigest()[:16]


def probe(tree, state_file, folder):
    """Subprocess body: sosfuse2/engine from <tree>; outputs to <folder>/arrays.npz and keys.json."""
    sys.path[:0] = [str(Path(tree) / 'scripts'), str(Path(tree) / 'rig'), str(Path(tree) / 'rig/docker')]
    import numpy as np
    from types import SimpleNamespace
    from sosfuse2 import reference, geometry as gm, geometry_contract as gc
    from sosfuse2.fused import load_engine
    from sosfuse2.run_gpu_state_v3 import create_configured, settings_for, signed_contract

    try:
        from sosfuse2 import tropo_profile as tp
    except ImportError:
        tp = None
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    state = json.loads(Path(state_file).read_text())
    A = {}
    P = {}
    K = {}
    rng = np.random.default_rng(20261007)
    # A: raw arrays (numerical equivalence). P: physical arrays (equivalence): radiances, ground terms, and aerosol
    # extinction/optical depth (the exponential and two-layer paths factor NORMAL['aer'] x profile differently).

    def physical(prefix, s_, aer):
        P[prefix + 'beta'] = np.outer(s_.ha, s_.ext[1])
        P[prefix + 'tau_sun'] = s_.geo.sun.col[..., 1] * aer
        P[prefix + 'gm'] = s_.gm
        P[prefix + 'gval'] = s_.gval

    # 1. NumPy reference (selftest grid, 3 wavelengths)
    s, scope, _ = gm.create(folder / 'np', 'selftest', WL, 'numpy', 128, state=state)
    aer = scope['b'].sh.NORMAL['aer']
    for k in ('ext', 'asca', 'ha', 'hr', 'gm', 'gval', 'gdir', 'goff'):
        A['np_' + k] = getattr(s, k)
    A['np_sun_col'] = s.geo.sun.col
    A['np_normal_aer'] = np.array(aer)
    physical('np_', s, aer)
    geo = gm.TransientGeometry(s, scope)
    groups = []
    field = rng.normal(0, 1e-4, (s.geo.ns, 571, s.W)).astype('f4')
    field[:, :4] = abs(field[:, :4])
    # Aerosol SH rows are per unit profile value, as Solver.apply builds them (moments x asca): scale by asca so a
    # two-layer state and its exponential twin carry the same physical source.
    field[:, 10:] *= np.asarray(s.asca, 'f4')[None, None, :]
    for i in CELLS:
        g = geo.cell(i)
        groups.append(g)
        c = s.kernel.setup(s, g['nq'])
        for first in (True, False):
            v, Y = reference.transport(g, c, field, first, return_basis=True)
            m, f = reference.project(g, v, c, Y)
            for name, x in (('ray', v), ('mom', m), ('flux', f)):
                A[f'np_c{i}_{int(first)}_{name}'] = P[f'np_c{i}_{int(first)}_{name}'] = x
        A[f'np_c{i}_endcol'] = g['end_columns']
        P[f'np_c{i}_tau_end'] = g['end_columns'][:, 1] * aer
    K['flags'] = list(s.kernel.flags)
    K['constants'] = sorted(c)
    bundle = gm.Bundle(groups, np)
    host = {k: np.asarray(v) for k, v in bundle.arrays.items()}
    A['col_aerosol_column'] = (
        gm.aerosol_column(host, s.kernel.height)
        if tp is None
        else gm.aerosol_column(host, s.kernel.height, profile=tp.column_function(s))
    )
    P['col_tau'] = A['col_aerosol_column'] * aer
    K['identity'] = {str(v): gm.identity(s, v)[1] for v in (1, 2, 3)}
    K['contract_identity'] = gc.identity(s)[1]
    K['identity_content'] = {str(v): {k: x for k, x in gm.identity(s, v)[0].items() if k != 'files'} for v in (1, 2, 3)}
    K['identity_files'] = {str(v): gm.identity(s, v)[0]['files'] for v in (1, 2, 3)}
    K['column_path'] = gm.TwoLayerGeometry.column_path(SimpleNamespace(cache=Path('X'), s=s), s.kernel.height).name
    # 2. create_configured (production solver constructor, reduced settings, host cache on)
    cfg = settings_for(SimpleNamespace(reduced=True, settings_json=None))
    s2, scope2 = create_configured(folder / 'cc', cfg, np.array([550.0]), 'numpy', state, geometry_cache=folder / 'geo')
    for k in ('ext', 'asca', 'ha', 'gm', 'gval'):
        A['cc_' + k] = getattr(s2, k)
    A['cc_sun_col'] = s2.geo.sun.col
    physical('cc_', s2, scope2['b'].sh.NORMAL['aer'])
    sig, hashes = signed_contract(state, cfg, 'numpy', True, 'depolarizing')
    K['settings'] = cfg
    K['code_hashes'] = hashes
    K['signature'] = sig
    # 3. C++ engine (rig/sh_cpu.cpp), tiny grid
    engine = load_engine()
    lib = build_native(tree, folder)
    engine.compile_kernel = lambda threads, force=False, lib=lib: lib
    create = engine.create if tp is None else (lambda *a, **k: engine.create(*a, aerosol=tp.engine_hooks(state), **k))
    s3, scope3, _ = create(state, folder / 'cpp', 'selftest', 2, 550.0)
    A['cpp_ha'] = s3.ha
    A['cpp_sun_col'] = s3.geo.sun.col
    A['cpp_gm'] = s3.gm
    physical('cpp_', s3, scope3['b'].sh.NORMAL['aer'])
    src = rng.normal(0, 1e-4, (s3.geo.ns, 571, 2 * s3.W)).astype('f4')
    src[:, :4] = abs(src[:, :4])
    src[:, 10:] *= np.repeat(np.asarray(s3.asca, 'f4'), 2)[None, None, :]  # lanes v = 2w+lane
    for i in range(0, s3.geo.ns, 7):
        z, L = s3.radiance(i, src)
        A[f'cpp_c{i}_first'] = P[f'cpp_c{i}_first'] = z['first']
        A[f'cpp_c{i}_rad'] = P[f'cpp_c{i}_rad'] = L
    np.savez(folder / 'arrays.npz', **{'raw__' + k: v for k, v in A.items()}, **{'phys__' + k: v for k, v in P.items()})
    K['digests'] = {k: digest(v) for k, v in A.items()}
    (folder / 'keys.json').write_text(json.dumps(K, indent=1, sort_keys=True, default=str) + '\n')


def equivalent_block(aod, H, z=2.0, w=1.0):
    """Two-layer parameters that equal aod*exp(-h/H)/(H(1-exp(-80/H))) in exact arithmetic: both components are that
    exponential, so the smoothstep blend is the identity, and ft_aod550 = aod*c*integral of s*exp(-h/H)."""
    from sosfuse2.tropo_profile import basis_integrals

    c = 1 / (H * (-math.expm1(-80.0 / H)))
    _, jf = basis_integrals(z, w, 1 / H, 1 / H)
    return dict(
        status='enabled',
        model='two_layer',
        bl_top_km=z,
        bl_transition_km=w,
        bl_shape='exponential',
        bl_scale_height_km=H,
        ft_aod550=aod * c * math.exp(-z / H) * jf,
        ft_scale_height_km=H,
    )


def paths(out, jobs):
    import numpy as np
    from sosfuse2.geometry import normal_state

    base_tree = extract_base(out)
    normal = copy.deepcopy(normal_state())
    cases = {
        'base_absent': (base_tree, normal),
        'absent': (ROOT, normal),
        'disabled': (ROOT, dict(normal, troposphere_profile=dict(status='disabled'))),
        'equivalent': (
            ROOT,
            dict(
                normal,
                parameters=dict(normal['parameters'], scale_height_km=3.0),
                troposphere_profile=equivalent_block(normal['parameters']['aod550'], 1.2),
            ),
        ),
        'decoy_exp3': (ROOT, dict(normal, parameters=dict(normal['parameters'], scale_height_km=3.0))),
        'default': (ROOT, dict(normal, troposphere_profile=dict(status='enabled', model='two_layer'))),
    }

    def run(name):
        tree, state = cases[name]
        folder = out / 'paths' / name
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        sf = folder / 'state.json'
        sf.write_text(json.dumps(state))
        t = time.monotonic()
        r = subprocess.run(
            [sys.executable, '-B', str(Path(__file__).resolve()), '--probe', str(tree), str(sf), str(folder)],
            capture_output=True,
            text=True,
            env=dict(os.environ, OMP_NUM_THREADS='1'),
        )
        print(f'PROBE {name} rc={r.returncode} {time.monotonic() - t:.1f}s', flush=True)
        if r.returncode:
            raise RuntimeError(f'probe {name} failed: {r.stderr[-2000:]}')
        return name

    with ThreadPoolExecutor(jobs) as pool:
        list(pool.map(run, cases))

    def load(n):
        with np.load(out / 'paths' / n / 'arrays.npz') as z:
            phys = {k[6:]: z[k] for k in z.files if k.startswith('phys__')}
        return phys, json.loads((out / 'paths' / n / 'keys.json').read_text())

    (ba, bk), (na, nk), (da, dk) = load('base_absent'), load('absent'), load('disabled')
    from sosfuse2.numerical_checks import within_tol

    def raw(n):
        with np.load(out / 'paths' / n / 'arrays.npz') as z:
            return {k: z[k] for k in z.files if k.startswith('raw__')}

    br, nr, dr = raw('base_absent'), raw('absent'), raw('disabled')
    ident = dict(
        arrays=len(br),
        absent_outside_tol=[k for k in br if k not in nr or not within_tol(nr[k], br[k])],
        disabled_outside_tol=[k for k in br if k not in dr or not within_tol(dr[k], br[k])],
        array_keys_equal=set(br) == set(nr) == set(dr),
    )
    for key in ('identity_content', 'contract_identity', 'column_path', 'settings', 'flags', 'constants'):
        ident[key] = bk[key] == nk[key] == dk[key]
    # v2/v3 (layers=1 and the two-layer class without geometry_contract) hash scripts/sosfuse2/geometry.py itself; v1 and the
    # production common-layer identity (v4) do not.
    ident['identity_signatures'] = dict(base=bk['identity'], new=nk['identity'])
    ident['identity_files_changed'] = sorted(
        {
            f
            for v in bk['identity_files']
            for f in bk['identity_files'][v]
            if bk['identity_files'][v][f] != nk['identity_files'][v].get(f)
        }
    )
    changed = sorted(
        k for k in set(bk['code_hashes']) & set(nk['code_hashes']) if bk['code_hashes'][k] != nk['code_hashes'][k]
    )
    ident.update(
        code_hashes_changed=changed,
        code_hashes_added=sorted(set(nk['code_hashes']) - set(bk['code_hashes'])),
        code_hashes_removed=sorted(set(bk['code_hashes']) - set(nk['code_hashes'])),
        signature_base=bk['signature'][:16],
        signature_new=nk['signature'][:16],
        keys_base=dict(identity=bk['identity'], contract=bk['contract_identity'], column=bk['column_path']),
    )
    ident['pass_'] = (
        ident['array_keys_equal']
        and not ident['absent_outside_tol']
        and not ident['disabled_outside_tol']
        and all(
            ident[k] for k in ('identity_content', 'contract_identity', 'column_path', 'settings', 'flags', 'constants')
        )
        and not ident['code_hashes_removed']
        and set(ident['identity_files_changed']) <= {'scripts/sosfuse2/geometry.py'}
        and bk['identity']['1'] == nk['identity']['1']
    )
    ident['absent_outside_tol'] = ident['absent_outside_tol'] or 'all equal'
    ident['disabled_outside_tol'] = ident['disabled_outside_tol'] or 'all equal'
    (ea, ek), (xa, xk), (fa, fk) = load('equivalent'), load('decoy_exp3'), load('default')

    def rel(a, b):
        a = np.asarray(a, 'f8')
        b = np.asarray(b, 'f8')
        scale = max(float(np.max(abs(b))), 1e-300)
        return float(np.max(abs(a - b))) / scale

    groups = {
        'numpy reference (rays, moments, flux, end tau)': 'np_c',
        'numpy solver (beta, sun tau, ground)': 'np_',
        'create_configured (beta, sun tau, ground)': 'cc_',
        'aerosol_column (tau)': 'col_',
        'C++ engine (first light, radiance, ground)': 'cpp_',
    }
    equiv = {}
    for label, prefix in groups.items():
        names = [k for k in na if k.startswith(prefix) and (prefix != 'np_' or not k.startswith('np_c'))]
        e = max(rel(ea[k], na[k]) for k in names)
        x = max(rel(xa[k], na[k]) for k in names)
        f = max(rel(fa[k], na[k]) for k in names)
        equiv[label] = dict(
            arrays=len(names),
            equivalent_vs_exp12=e,
            decoy_exp3_vs_exp12=x,
            default_vs_exp12=f,
            pass_=e <= 1e-4 * x and x > 1e-3 and f > 1e-3 and all(np.isfinite(fa[k]).all() for k in names),
        )
    equiv['flags'] = dict(equivalent=ek['flags'], default=fk['flags'], absent=nk['flags'])
    equiv['column_path'] = dict(default=fk['column_path'], equivalent=ek['column_path'], absent=nk['column_path'])
    equiv['pass_'] = (
        all(v['pass_'] for v in equiv.values() if isinstance(v, dict) and 'pass_' in v)
        and 'AEROSOL_2L' in fk['flags']
        and 'AEROSOL_2L' not in nk['flags']
        and fk['column_path'].startswith('A2L_')
        and nk['column_path'].startswith('H_')
    )
    return dict(
        pass_=ident['pass_'] and equiv['pass_'],
        identity=ident,
        equivalence=equiv,
        rule='identity: every array within tolerance and keys equal; equivalence: |two-layer==exp1.2 - exp1.2| <= 1e-4 x |exp3 - exp1.2|',
        grids='NumPy selftest grid (4 heights x 10 chi, nmu 1, nphi 2, path 4) cells '
        + str(CELLS)
        + '; create_configured reduced '
        'settings; C++ engine tiny grid (h 0/.1/8/80 km); 3 wavelengths 450/550/650 nm (C++ 550 only)',
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path)
    p.add_argument('--jobs', type=int, default=6)
    for k in ('static', 'host', 'paths'):
        p.add_argument('--' + k, action='store_true')
    p.add_argument('--probe', nargs=3, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.probe:
        return probe(*a.probe)
    if ROOT is None:
        raise SystemExit(
            f'E_SKY_SOS_ROOT: this harness needs the sky-sos checkout ({T.ENV}) for its git baseline and legacy entry points'
        )
    a.out.mkdir(parents=True, exist_ok=True)
    rep = {}
    with ThreadPoolExecutor(1) as pool:  # paths = subprocesses only; static/host run here meanwhile
        background = pool.submit(paths, a.out, a.jobs) if a.paths else None
        for k, fn in (('static', static), ('host', host)):
            if getattr(a, k):
                t = time.monotonic()
                rep[k] = fn(a.out)
                print(f'{k.upper()} done {time.monotonic() - t:.1f}s pass={rep[k]["pass_"]}', flush=True)
        if background is not None:
            rep['paths'] = background.result()
    rep['pass_'] = all(v.get('pass_') is True for v in rep.values())
    (a.out / 'tropo_profile_tests.json').write_text(json.dumps(rep, indent=1, default=str) + '\n')
    print(json.dumps({k: (v.get('pass_') if isinstance(v, dict) else v) for k, v in rep.items()}))
    print('TROPO_PROFILE TESTS', 'PASS' if rep['pass_'] else 'FAIL')
    raise SystemExit(0 if rep['pass_'] else 1)


if __name__ == '__main__':
    main()
