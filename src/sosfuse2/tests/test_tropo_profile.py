#!/usr/bin/env python3
"""Two-layer tropospheric aerosol (tropo_profile.py, kernel.cu -DAEROSOL_2L): CPU checks.

  --host      block validation codes; profile integral = aod550 (closed form and engine integration);
              legacy entry points refuse a state that carries the block.
The former --static and --paths modes are retired. Nothing replaces the host-built point() check of the -DAEROSOL_2L
branch: test_kernel_static and test_tropo_mie build
the kernel without that flag. test_tropo_patches covers how an enabled block reaches the CPU engine.
usage: python -B -m sosfuse2.tests.test_tropo_profile --host --out <dir>
"""

import argparse, copy, json, math, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
from sosfuse2.checks import testkit as T

ROOT = T.sky_sos_root()  # Optional research data for two-layer states and legacy entry points.

# ---- host: validation and integrals ------------------------------------------------------------------------------


def host(out):
    import numpy as np
    from sosfuse2.atmosphere import tropo_profile as tp
    from sosfuse2.solve.ray_geometry import create, normal_state

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
    from sosfuse2.atmosphere import tropo_mie

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
    # The engine profile integral and the solver's own column quadrature.
    st = copy.deepcopy(base)
    st['troposphere_profile'] = good
    s, scope, _ = create(out / 'host_integral', 'selftest', (550.0,), 'numpy', 128, state=st)
    old = scope.b.sh.old
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
            round(r['solver_rel'], 6) for r in columns(scope_w.b.sh.old, scope_w.b.sh.NORMAL['aer'])
        ]
    integ = dict(
        cases=integrals,
        patched_sos_sky_profiles_rel=abs(patched / st['parameters']['aod550'] - 1),
        normal_aer=aer,
        solver_columns=columns(old, aer),
        solver_columns_exp12=columns(scope_exp.b.sh.old, scope_exp.b.sh.NORMAL['aer']),
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
    # With the default 1 km blend the GL-24 column error is within 0.5 % at every zenith.
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
# rig/run_state_v2.py and rig/run_state_v3.py import the rig/engine.py that left that checkout, so they cannot run at all;
# their source is still checked for the guard call below.
LEGACY_READERS = ('rig/run_state.py',)  # read_state(path, id)
LEGACY_CLIS = (
    'scripts/sosfuse/run_gpu_state.py',
    'scripts/sosfuse/run_gpu_state_v2.py',
    'scripts/sosfuse/run_gpu_state_v3.py',
)
LEGACY_GUARDED = (
    LEGACY_READERS
    + ('rig/run_state_v2.py', 'rig/run_state_v3.py')
    + LEGACY_CLIS
    + ('scripts/sosfuse/run_fuse.py', 'rig/docker/geometry_cpu.py', 'results/column_parallel_20261007/measure.py')
)


def legacy(out):
    from sosfuse2.atmosphere import tropo_profile as tp
    from sosfuse2.solve.ray_geometry import normal_state

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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path)
    p.add_argument('--host', action='store_true')
    a = p.parse_args()
    if not a.host or a.out is None:
        p.error('E_NO_CHECK: pass --host and --out <dir>')
    if ROOT is None:
        raise SystemExit(
            f'E_SKY_SOS_ROOT: this harness needs the sky-sos checkout ({T.ENV}) for its states and legacy entry points'
        )
    a.out.mkdir(parents=True, exist_ok=True)
    rep = {}
    t = time.monotonic()
    rep['host'] = host(a.out)
    print(f'HOST done {time.monotonic() - t:.1f}s pass={rep["host"]["pass_"]}', flush=True)
    rep['pass_'] = all(v.get('pass_') is True for v in rep.values())
    (a.out / 'tropo_profile_tests.json').write_text(json.dumps(rep, indent=1, default=str) + '\n')
    print(json.dumps({k: (v.get('pass_') if isinstance(v, dict) else v) for k, v in rep.items()}))
    print('TROPO_PROFILE TESTS', 'PASS' if rep['pass_'] else 'FAIL')
    raise SystemExit(0 if rep['pass_'] else 1)


if __name__ == '__main__':
    main()
