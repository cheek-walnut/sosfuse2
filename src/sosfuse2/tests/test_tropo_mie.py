#!/usr/bin/env python3
"""Tropospheric Mie phase (tropo_mie.py, kernel.cu -DTROPO_MIE): Mac static/host checks and the split-profile GPU parity.

  --static  clang++ -E -P token identity of the HG kernel for the production flag sets
            (C3 + STRATO + PROFILE, with and without LAUNCH_RAY_LIMIT; the strato-off C2 set + PROFILE; the
            strato_lut_build options) over every launched specialization; TROPO_MIE preprocesses without #error for
            the same sets; the unsupported ground paths stop at their #error.
  --host    optics table (normalization, LUT energy, LUT vs table, eigenvalue order) and the NumPy reference with a
            Mie table filled from hg() reproducing the HG reference; Kernel.setup of a Mie state.
  --gpu     CUDA vs NumPy reference with TROPO_MIE: strato off (C2 set) and on (C3 set),
            15 lanes, first light (phase 2 rays, phase 0 moments, ground flux) and warm; HG control on the C3 set.
usage: python -B -m sosfuse2.tests.test_tropo_mie --static --host --out <dir>   |   ... --gpu --out <dir>
"""

from sosfuse2.foundation import paths
import argparse, copy, itertools, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = paths.PKG  # the package directory (kernel, configs, assets)
import numpy as np
from sosfuse2.checks import testkit

ROOT = testkit.sky_sos_root()  # The HG-baseline comparison needs the research data root.
from sosfuse2.checks.numerical_checks import within_tol


def base_ref():
    """HG baseline kernel = merge-base of this branch with main (falls back to 'main')."""
    if ROOT is None:
        return 'main'
    try:
        return subprocess.check_output(['git', '-C', str(ROOT), 'merge-base', 'HEAD', 'main'], text=True).strip()[:12]
    except (OSError, subprocess.CalledProcessError):
        return 'main'


C3 = tuple(json.loads((HERE / 'backends/configs/C3.json').read_text())['kernel_flags'])
C2OFF = (
    'GROUND_PHASE_BATCH',
    'LANE_ATTEN',
    'LANE_GROUND',
    'LM_LUT',
    'PHASE1_TRACE3',
    'PHASE2_NOBARRIER',
    'POINT_NOINLINE',
    'RAY_CONST_SHARED',
    'SH_AUX',
    'SH_AUX_PAIR',
    'SPECIALIZE',
    'TRANSPORT_WAVE_LANES',
)  # task A strato-off set
SPECS = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0), (1, 2, 0), (0, 2, 0))  # precompile.SPECS
WL15 = (360 + (np.arange(15) + 0.5) * 470 / 15).tolist()
MIE = dict(
    status='enabled', model='mie', optics_table='assets/tropo_rural_sf79_rh70.json'
)  # package-relative (tropo_mie.ROOT)


SYNTAX_STUB_EXTRA = r'''
struct int2{int x,y;}; inline int2 make_int2(int a,int b){return int2{a,b};}
inline int __float_as_int(float){return 0;} inline float __int_as_float(int){return 0;}
template<class T> T __ldcg(const T*p){return *p;}
typedef long long int64_t; typedef unsigned int uint32_t; typedef unsigned short uint16_t; typedef unsigned char uint8_t;
typedef int int32_t; typedef unsigned long long uint64_t;
inline double atomicAdd(double*,double){return 0;} inline float atomicAdd(float*,float){return 0;} inline int atomicAdd(int*,int){return 0;}
inline unsigned atomicAdd(unsigned*,unsigned){return 0;} inline int __syncthreads_or(int p){return p;}
inline void __threadfence(){} inline int __popc(unsigned){return 0;} inline int __ffs(int){return 0;}
template<class T>T __shfl_up_sync(unsigned,T v,int){return v;} template<class T>T __shfl_xor_sync(unsigned,T v,int){return v;}
double exp(double);double log(double);float powf(float,float);double cos(double);double sin(double);double acos(double);
float acosf(float);float rsqrtf(float);
#define __launch_bounds__(...)
'''


def tokens(text):
    return re.findall(r'\w+|\S', text)


def defines(flags, spec, W=15):
    out = [f'-DFW={W}'] + [f'-D{f}' for f in sorted(set(flags))]
    if spec is not None:
        out += [f'-DK_FIRST={spec[0]}', f'-DK_PHASE={spec[1]}', f'-DK_BOUNDARY={spec[2]}']
    return out


def preprocess(path, flags, spec):
    r = subprocess.run(
        ['clang++', '-E', '-P', '-x', 'c++', *defines(flags, spec), str(path)], capture_output=True, text=True
    )
    return r.returncode, r.stdout, r.stderr


def static(out):
    if shutil.which('clang++') is None or shutil.which('git') is None or ROOT is None:
        return dict(pass_=None, reason=f'clang++/git/sky-sos checkout ({testkit.ENV}) missing')
    BASE = base_ref()
    base = out / f'kernel_{BASE}.cu'
    base.write_text(
        subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{BASE}:scripts/sosfuse2/kernel.cu'], text=True)
    )
    new = HERE / 'backends/kernel.cu'
    sets = {
        'C3_strato': C3 + ('STRATO', 'PROFILE'),
        'C3_strato_raylimit': C3 + ('STRATO', 'PROFILE', 'LAUNCH_RAY_LIMIT'),
        'C2off': C2OFF + ('PROFILE',),
    }
    hg_rows, mie_rows = [], []
    for (name, flags), spec in itertools.product(sets.items(), SPECS):
        if name == 'C3_strato_raylimit' and spec != (1, 0, 0):
            continue
        ra, a, ea = preprocess(base, flags, spec)
        rb, b, eb = preprocess(new, flags, spec)
        hg_rows.append(
            dict(
                set=name,
                spec=list(spec),
                base_rc=ra,
                new_rc=rb,
                tokens=len(tokens(b)),
                tokens_identical=ra == 0 and rb == 0 and tokens(a) == tokens(b),
            )
        )
        rm, m, em = preprocess(new, flags + ('TROPO_MIE',), spec)
        mie_rows.append(
            dict(
                set=name,
                spec=list(spec),
                rc=rm,
                error=em.strip()[-200:] if rm else '',
                tropo_phase_used=('tropo_phase(' in m.split('tropo_pair(const')[-1]) if rm == 0 else None,
                token_delta=len(tokens(m)) - len(tokens(b)) if rm == 0 else None,
            )
        )
    lut = ('STRATO', 'STRATO_MIE_LUT', 'STRATO_MIE_LUT_FINE')
    ra, a, _ = preprocess(base, lut, None)
    rb, b, _ = preprocess(new, lut, None)
    hg_rows.append(
        dict(set='strato_lut_build', spec=None, tokens_identical=ra == 0 and rb == 0 and tokens(a) == tokens(b))
    )
    errors = []
    for flags, spec, want in (
        (
            tuple(f for f in C2OFF if f != 'LANE_GROUND') + ('PROFILE', 'TROPO_MIE'),
            (1, 2, 0),
            'E_TROPO_MIE_UNSUPPORTED_GROUND_PATH',
        ),
        (
            tuple(f for f in C3 if f != 'STRATO_MIE_LUT_FINE') + ('STRATO', 'PROFILE', 'TROPO_MIE'),
            (1, 0, 0),
            'E_TROPO_MIE_GF_REQUIRES_STRATO_MIE_LUT_FINE',
        ),
    ):
        rc, _, err = preprocess(new, flags, spec)
        errors.append(dict(expected=want, rc=rc, pass_=rc != 0 and want in err))
    from sosfuse2.tests.test_kernel_static import STUB

    stub = out / 'cuda_stub.h'
    stub.write_text(STUB + SYNTAX_STUB_EXTRA)
    syntax = []
    for (name, flags), spec, mie in itertools.product(sets.items(), SPECS, (False, True)):
        if name == 'C3_strato_raylimit' and spec != (1, 0, 0):
            continue
        rs = subprocess.run(
            [
                'clang++',
                '-std=c++17',
                '-ffp-contract=off',
                '-fsyntax-only',
                '-x',
                'c++',
                '-include',
                str(stub),
                *defines(flags + (('TROPO_MIE',) if mie else ()), spec),
                str(new),
            ],
            capture_output=True,
            text=True,
        )
        syntax.append(
            dict(
                set=name,
                spec=list(spec),
                tropo_mie=mie,
                rc=rs.returncode,
                error=rs.stderr.strip()[-300:] if rs.returncode else '',
            )
        )
    ok = (
        all(r['tokens_identical'] for r in hg_rows)
        and all(r['rc'] == 0 for r in mie_rows)
        and all(e['pass_'] for e in errors)
    )
    ok = ok and all(r['rc'] == 0 for r in syntax)
    first_mie = [r for r in mie_rows if r['spec'][0] == 1]
    ok = ok and all(r['tropo_phase_used'] for r in first_mie)
    return dict(
        pass_=ok,
        baseline=f'git show {BASE}:scripts/sosfuse2/kernel.cu',
        hg_token_identity=hg_rows,
        mie_preprocess=mie_rows,
        expected_errors=errors,
        syntax=syntax,
        scope='clang preprocessing + host C++ syntax with stubbed CUDA builtins; NVRTC compile and numerics are the --gpu part',
    )


def hg_lut(g, W, bins=16384):
    from sosfuse2.atmosphere.tropo_mie import lut_nodes
    from sosfuse2.backends.numpy_reference import hg

    u = lut_nodes(bins).astype('f4')
    return np.ascontiguousarray(np.repeat(hg(u, np.float32(g))[:, None], W, 1), 'f4')


def rejects(out):
    """Every malformed block or asset is refused with its code; the shipped asset, "default" and "disabled" pass."""
    from sosfuse2.atmosphere import tropo_mie as tm

    rows = []

    def expect(name, code, fn):
        try:
            fn()
            rows.append(dict(case=name, expected=code, got=None, pass_=False))
        except ValueError as e:
            rows.append(dict(case=name, expected=code, got=str(e)[:120], pass_=str(e).startswith(code)))

    for name, block, code in (
        ('empty block', {}, 'E_TROPO_PHASE_STATUS'),
        ('status enable', dict(MIE, status='enable'), 'E_TROPO_PHASE_STATUS'),
        ('no status', {k: v for k, v in MIE.items() if k != 'status'}, 'E_TROPO_PHASE_STATUS'),
        ('not an object', 'mie', 'E_TROPO_PHASE'),
        ('model hg', dict(MIE, model='hg'), 'E_TROPO_PHASE_MODEL'),
        ('no optics_table', {k: v for k, v in MIE.items() if k != 'optics_table'}, 'E_TROPO_PHASE_TABLE'),
        ('empty optics_table', dict(MIE, optics_table=''), 'E_TROPO_PHASE_TABLE'),
        ('unknown key', dict(MIE, g=0.7), 'E_TROPO_PHASE_KEYS'),
    ):
        expect(name, code, lambda b=block: tm.config(dict(troposphere_phase=b)))
    base = json.loads(tm.DEFAULT.read_text())

    def bad(edit):
        d = copy.deepcopy(base)
        edit(d)
        f = out / 'bad_asset.json'
        f.write_text(json.dumps(d))
        return f

    def setp(i, v):
        def e(d):
            d['rows'][3]['phase'][i] = v

        return e

    def scale(d):
        d['rows'][3]['phase'] = [x * 1.01 for x in d['rows'][3]['phase']]

    def chi5(d):
        d['rows'][3]['chi_l'][5] += 0.2

    def chi0(d):
        d['rows'][3]['chi_l'][0] = 1.1

    def theta(d):
        d['theta_deg'][10], d['theta_deg'][11] = d['theta_deg'][11], d['theta_deg'][10]

    def swap(d):
        d['rows'][0]['phase'], d['rows'][-1]['phase'] = d['rows'][-1]['phase'], d['rows'][0]['phase']

    for name, edit, code in (
        ('NaN phase', setp(100, float('nan')), 'E_TROPO_TABLE_SCHEMA'),
        ('negative phase', setp(100, -1.0), 'E_TROPO_TABLE_SCHEMA'),
        ('theta not monotone', theta, 'E_TROPO_TABLE_SCHEMA'),
        ('chi0 1.1', chi0, 'E_TROPO_TABLE_SCHEMA'),
        ('phase x1.01', scale, 'E_TROPO_TABLE_NORMALIZATION'),
        ('chi5 +0.2', chi5, 'E_TROPO_CHI_LUT'),
        ('phase rows 360/830 swapped', swap, 'E_TROPO_CHI_LUT'),
    ):
        expect(name, code, lambda f=bad(edit): tm.load_table(f))
    ok = dict(
        asset=tm.load_table(tm.DEFAULT) is not None,
        default_word=tm.build_optics(dict(troposphere_phase=dict(MIE, optics_table='default')), WL15) is not None,
        disabled=tm.build_optics(dict(troposphere_phase=dict(status='disabled')), WL15) is None,
    )
    return dict(
        pass_=all(r['pass_'] for r in rows) and all(ok.values()),
        cases=rows,
        accepted=ok,
        tolerances=dict(table_mean=tm.TABLE_MEAN_TOL, lut_mean=tm.LUT_MEAN_TOL, eigen=tm.EIGEN_TOL),
    )


def host(out):
    from sosfuse2.atmosphere import tropo_mie as tm
    from sosfuse2.solve.ray_geometry import create, TransientGeometry, normal_state
    from sosfuse2.backends import numpy_reference

    r = {}
    # 1. table/LUT
    wl, th, chi, P = tm.load_table(tm.DEFAULT)
    o = tm.build_optics(dict(troposphere_phase=MIE), WL15)
    u20 = np.cos(np.deg2rad(20.0))
    i20 = int((u20 + 1) * 8192)
    lut20 = o['lut'][i20] + ((u20 + 1) * 8192 - i20) * (o['lut'][i20 + 1] - o['lut'][i20])
    tab20 = np.exp(np.interp(20.0, th, np.log(P[1]))) / (4 * np.pi)  # row 1 = 375.67 nm = WL15[0]
    lev = np.array([l for l in range(33) for m in range(l + 1)])
    r['table'] = dict(
        lut_shape=list(o['lut'].shape),
        lut_dtype=str(o['lut'].dtype),
        lut_energy_min_max=[min(o['meta']['lut_energy_trapezoid']), max(o['meta']['lut_energy_trapezoid'])],
        lut20_vs_table20_rel=float(abs(lut20[0] / tab20 - 1)),
        chi0_max_abs=float(abs(o['chi'][0] - 1).max()),
        eigen_order_ok=bool(within_tol(tm.eigen_by_sh(o), o['eigen'][lev])),
        hg_state_none=tm.build_optics(dict(parameters={}), WL15) is None,
        disabled_none=tm.build_optics(dict(troposphere_phase=dict(MIE, status='disabled')), WL15) is None,
    )
    try:
        tm.config(dict(troposphere_phase=dict(MIE, model='hg')))
        r['table']['bad_model_rejected'] = False
    except ValueError as e:
        r['table']['bad_model_rejected'] = 'E_TROPO_PHASE_MODEL' in str(e)
    t = r['table']
    t['pass_'] = (
        t['lut_shape'] == [16385, 15]
        and t['lut_dtype'] == 'float32'
        and abs(t['lut_energy_min_max'][0] - 1) < 5e-3
        and abs(t['lut_energy_min_max'][1] - 1) < 5e-3
        and t['lut20_vs_table20_rel'] < 1e-4
        and t['chi0_max_abs'] < 1e-6
        and t['eigen_order_ok']
        and t['hg_state_none']
        and t['disabled_none']
        and t['bad_model_rejected']
    )
    # 2. reference with an HG table == HG reference; Kernel.setup of a Mie state
    state = copy.deepcopy(normal_state())
    s, scope, _ = create(out / 'host_geometry', 'selftest', WL15, 'numpy', 128, state=state)
    geo = TransientGeometry(s, scope)
    g = geo.cell(12)
    c = s.kernel.setup(s, g['nq'])
    field = np.zeros((s.geo.ns, 571, s.W), 'f4')
    a = numpy_reference.transport(g, c, field, True)
    cm = dict(c, tropo_lut=hg_lut(float(c['g']), s.W).ravel(), tropo_bins=16384)
    b = numpy_reference.transport(g, cm, field, True)
    rel = float(np.max(abs(b - a)) / np.max(abs(a)))
    mstate = copy.deepcopy(normal_state())
    mstate['troposphere_phase'] = dict(MIE)
    sm, scope_m, _ = create(out / 'host_geometry_mie', 'selftest', WL15, 'numpy', 128, state=mstate)
    gm_ = TransientGeometry(sm, scope_m).cell(12)
    cmie = sm.kernel.setup(sm, gm_['nq'])
    mie = numpy_reference.transport(gm_, cmie, field, True)
    r['reference'] = dict(
        hg_table_vs_hg_max_rel_to_peak=rel,
        mie_state_flags_has_TROPO_MIE='TROPO_MIE' in sm.kernel.flags,
        mie_state_af_cleared=sm.af is None,
        mie_state_lut=list(np.shape(cmie.get('tropo_lut', ()))),
        hg_state_flags_has_TROPO_MIE='TROPO_MIE' in s.kernel.flags,
        hg_state_tropo=getattr(s, 'tropo', 'missing'),
        mie_vs_hg_first_light_ratio_median=float(np.median(mie[..., 0][a[..., 0] > 0] / a[..., 0][a[..., 0] > 0])),
    )
    # 3. source_solver.Solver.apply multiplies the aerosol SH moments by the per-wavelength Mie eigenvalues (HG: g**l)
    from sosfuse2.solve.source_solver import Solver
    from sosfuse2.atmosphere.tropo_mie import eigen_by_sh

    class Unit:
        def evaluate(self, s_, field_, first=False, phase=0, boundary=None, ground_only=False):
            return np.ones((s_.geo.ns, 565, s_.W)), np.ones((len(s_.geo.chi) if ground_only else s_.geo.ns, s_.W))

    apply_ok = {}
    for label, solver in (('mie', sm), ('hg', s)):
        fs = Solver(solver, Unit(), prepared=True)
        got = fs.apply(np.zeros((solver.geo.ns, 571, solver.W), 'f4'))[:, 10:571]
        eig = eigen_by_sh(solver.tropo) if label == 'mie' else solver.af[:, None]
        want = (np.float32(2) * eig.astype('f4') * solver.asca.astype('f4')[None, :]).astype('f4')
        apply_ok[label] = bool(within_tol(got, np.broadcast_to(want[None], got.shape)))
    r['reference']['apply_eigen_exact'] = apply_ok
    rr = r['reference']
    rr['pass_'] = (
        all(apply_ok.values())
        and rel < 1e-5
        and rr['mie_state_flags_has_TROPO_MIE']
        and rr['mie_state_af_cleared']
        and rr['mie_state_lut'] == [16385 * 15]
        and not rr['hg_state_flags_has_TROPO_MIE']
        and rr['hg_state_tropo'] is None
    )
    r['rejects'] = rejects(out)
    r['pass_'] = t['pass_'] and rr['pass_'] and r['rejects']['pass_']
    return r


C2OFF_RAY = C2OFF + (
    'ACC_GLOBAL',
    'FL_RAYSPLIT',
)  # strato-off with ray-range first light (split profile: C2 first light 8.6 s > 5 s)


def gpu(out, only=None):
    from sosfuse2.solve.ray_geometry import create, TransientGeometry, normal_state, Bundle
    from sosfuse2.backends import numpy_reference
    from sosfuse2.checks.validate import relative, moment_comparison
    from sosfuse2.atmosphere.stratosphere import source_size

    rows = []
    cases = [
        ('mie_strato_off_C2', False, True, C2OFF),
        ('mie_strato_on_C3', True, True, C3),
        ('hg_strato_on_C3', True, False, C3),
        ('mie_strato_off_C2R', False, True, C2OFF_RAY),
        ('hg_strato_off_C2R', False, False, C2OFF_RAY),
    ]
    if only:
        cases = [c for c in cases if c[0] in only]
    for name, strato, mie, flags in cases:
        os.environ['SOSFUSE2_KERNEL_FLAGS'] = ','.join(flags)
        state = copy.deepcopy(normal_state())
        if strato:
            state['stratosphere'].update(status='enabled')
            state['parameters']['stratosphere_tau550'] = 0.01
        if mie:
            state['troposphere_phase'] = dict(MIE)
        t0 = time.perf_counter()
        s, scope, _ = create(out / f'gpu_{name}', 'selftest', WL15, 'cupy', 256, state=state)
        g = TransientGeometry(s, scope).cell(12)
        c = s.kernel.setup(s, g['nq'])
        rng = np.random.default_rng(7)
        src = np.zeros((s.geo.ns, source_size(s), s.W), 'f4')
        src[:, :571] = rng.random((s.geo.ns, 571, s.W), dtype=np.float32) * np.float32(1e-3)
        if strato:
            src[:, 571:] = src[:, 10:571] * np.float32(0.1)
        bundle = Bundle([g], s.kernel.xp)
        for first in (True, False):
            ref, Y = numpy_reference.transport(g, c, src, first, return_basis=True)
            moment, flux = numpy_reference.project(g, ref, c, Y)
            _, _, gq = s.kernel.launch(bundle, s, s.kernel.xp.asarray(src), first, phase=2)
            qerr = relative(s.kernel.host(gq), ref)
            gmom, gf, _ = s.kernel.launch(bundle, s, s.kernel.xp.asarray(src), first, phase=0)
            comp = moment_comparison(s.kernel.host(gmom)[0], moment)
            ferr = relative(s.kernel.host(gf)[0], flux)
            rows.append(
                dict(
                    case=name,
                    first=first,
                    flags_effective=list(s.kernel.flags),
                    ray_max_relative=qerr,
                    moment_l2_relative=comp['moment_l2_relative'],
                    moment_scaled_relative=comp['moment_scaled_relative'],
                    ground_flux_max_relative=ferr,
                    seconds=round(time.perf_counter() - t0, 1),
                    max_launch_seconds=(getattr(s.kernel, 'launch_log', None) or {}).get('max_seconds'),
                )
            )
            print(json.dumps(rows[-1]), flush=True)
    return judge(rows)


def judge(rows):
    """A Mie case passes when its ray error is <= 2x the HG control's with the same first-light flag,
    and its moment L2 and ground flux errors are <= 1e-5. The ray bound accounts for the float32 error floor."""
    hg = {(r['case'], r['first']): r for r in rows if r['case'].startswith('hg_')}
    checks = []
    for r in rows:
        if r['case'].startswith('hg_'):
            continue
        twin = 'hg_' + r['case'][4:]  # same flag set if present, else the C3 control
        h = hg.get((twin, r['first'])) or hg[('hg_strato_on_C3', r['first'])]
        checks.append(
            dict(
                case=r['case'],
                first=r['first'],
                ray_ratio_to_hg=r['ray_max_relative'] / max(h['ray_max_relative'], 1e-30),
                moment_l2=r['moment_l2_relative'],
                ground=r['ground_flux_max_relative'],
                pass_=r['ray_max_relative'] <= 2 * h['ray_max_relative']
                and r['moment_l2_relative'] <= 1e-5
                and r['ground_flux_max_relative'] <= 1e-5,
            )
        )
    worst = max(
        (max(r['ray_max_relative'], r['moment_l2_relative'], r['ground_flux_max_relative']) for r in rows), default=None
    )
    return dict(
        pass_=bool(checks) and all(c['pass_'] for c in checks),
        worst=worst,
        rule='ray <= 2x HG control (same first), moment l2 and ground <= 1e-5',
        checks=checks,
        cases=rows,
        note='HG control row gives the float32 parity floor of the same flag set',
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    for k in ('static', 'host', 'gpu'):
        p.add_argument('--' + k, action='store_true')
    p.add_argument('--gpu-cases', help='comma list of gpu case names (default all)')
    p.add_argument('--recheck', type=Path, help='re-judge the gpu rows of an earlier tropo_mie_tests.json (no GPU)')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rep = {}
    if a.static:
        rep['static'] = static(a.out)
    if a.host:
        rep['host'] = host(a.out)
    if a.gpu:
        rep['gpu'] = gpu(a.out, set(a.gpu_cases.split(',')) if a.gpu_cases else None)
    if a.recheck:
        rep['gpu'] = judge(json.loads(a.recheck.read_text())['gpu']['cases'])
        rep['gpu']['rechecked_from'] = str(a.recheck)
    rep['pass_'] = bool(rep) and all(v.get('pass_') is True for v in rep.values())
    (a.out / 'tropo_mie_tests.json').write_text(json.dumps(rep, indent=1, default=str) + '\n')
    print(json.dumps({k: (v.get('pass_') if isinstance(v, dict) else v) for k, v in rep.items()}))
    print('TROPO_MIE TESTS', 'PASS' if rep['pass_'] else 'FAIL')
    raise SystemExit(0 if rep['pass_'] else 1)


if __name__ == '__main__':
    main()
