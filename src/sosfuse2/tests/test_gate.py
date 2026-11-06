"""gate.py without a GPU: synthetic fixture runs (--reuse) against a temporary gate home.
Run: python -m pytest -q src/sosfuse2/test_gate.py"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from sosfuse2 import gate, testing  # noqa: E402

REAL = gate.GATE
K, H, D, W = 4, 4, 15, 15


def opts(stage1):
    fast = 'FASTMATH' in stage1
    return [
        [
            '-std=c++17',
            '--fmad=true' if 'FMAD' in stage1 else '--fmad=false',
            f'--prec-div={str(not fast).lower()}',
            f'--prec-sqrt={str(not fast).lower()}',
            '-DFW=15',
        ]
        + (['-DSOSFUSE2_FASTMATH'] if fast else [])
    ]


def data(seed, rel=0.0):
    rng = np.random.default_rng(seed)
    rad = rng.uniform(1, 2, (K, H, D, W, 1))
    src = rng.uniform(-1, 1, (K, 6, 5, W)).astype('f4')
    return (
        rad * (1 + rel * rng.uniform(-1, 1, rad.shape)),
        src * np.float32(1 + rel),
        rng.uniform(1, 2, (H, 4, 6, W)).astype('f4'),
    )


def write_run(root, fx, rad, src, table, cc='8.6', evaluations=7, converged=True, options=None, stage1=None):
    F = json.loads((REAL / 'fixtures.json').read_text())
    want = gate.expected_config(F, fx)
    d = root / fx
    d.mkdir(parents=True)
    s1 = want['stage1'] if stage1 is None else stage1
    np.savez(d / 'fixed.npz', radiance=rad, source=src)
    np.savez(d / 'solve.npz', table=table, source=src[-1])
    meta = dict(
        kind='fixed',
        fixture=fx,
        backend='cupy',
        device=dict(profile='eda', cc=cc, nvrtc='12.9'),
        numpy='2.5.3',
        config=dict(
            flags=want['flags'],
            stage1=s1,
            async_=want['async_'],
            L=16,
            block=want['block'],
            options=options or opts(s1),
        ),
        host_inputs_sha256='h',
        launch=dict(max_seconds=0.01, launches=3),
        env={},
    )
    (d / 'fixed.json').write_text(json.dumps(meta))
    (d / 'solve.json').write_text(
        json.dumps(
            dict(
                rc=0,
                reason=None,
                evaluations=evaluations,
                converged=converged,
                source_solve_sha256='s' + str(evaluations),
                flags=want['flags'],
                block=want['block'],
                launch_max_seconds=0.02,
            )
        )
    )


def runs(root, cc='8.6', s2_evals=7, rel_s2=2e-6, **kw):
    g, gn = data(1), data(2)
    for fx, (rad, src, table) in (('FX-C3', g), ('FX-N', gn)):
        write_run(root, fx, rad * (1 + 1e-5), src, table, cc=cc, **kw.get(fx, {}))
    rad, src, table = g
    write_run(
        root,
        'FX-S2',
        rad * (1 + 1e-5) * (1 + rel_s2),
        src,
        table * np.float32(1 + rel_s2),
        cc=cc,
        evaluations=s2_evals,
        **kw.get('FX-S2', {}),
    )
    ra, sa, ta = data(3)
    write_run(root, 'FX-A', ra, sa, ta, cc=cc, **kw.get('FX-A', {}))
    rp, sp, tp = data(4)  # FX-AP (A′, P2): its own NumPy golden data(4)
    write_run(root, 'FX-AP', rp * (1 + 1e-5), sp, tp, cc=cc, **kw.get('FX-AP', {}))
    return root


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Gate home: fixtures copy, NumPy golden, canonical CUDA baselines from runs(), anchors from those runs."""
    g = tmp_path / 'gate'
    (g / 'baselines/numpy').mkdir(parents=True)
    shutil.copy(REAL / 'fixtures.json', g)
    monkeypatch.setattr(gate, 'GATE', g)
    monkeypatch.setattr(
        gate, 'RECORDS', None
    )  # main() sets it per call and restores it; a stale value must not leak between tests
    M = dict(schema=1, golden={}, cuda={}, gf=None)
    for name, seed in (('FX-C3', 1), ('FX-N', 2), ('FX-AP', 4)):
        rad, src, _ = data(seed)
        np.savez(g / f'baselines/numpy/{name}.npz', radiance=rad, source=src[-1])
        M['golden'][name] = dict(
            file=f'numpy/{name}.npz', sha256=gate.file_sha(g / f'baselines/numpy/{name}.npz'), outputs={}
        )
    base = runs(tmp_path / 'base')
    F = gate.load_fixtures()
    loaded = {fx: gate.load_run(base, fx) for fx in F['fixtures']}
    for fx, r in loaded.items():
        key, fields = gate.toolkey(r['meta'])
        (g / 'baselines' / key).mkdir(exist_ok=True)
        np.savez(g / f'baselines/{key}/{fx}.npz', fixed_radiance=r['radiance'], solve_table=r['table'])
        M['cuda'][fx] = [
            dict(
                toolkey=key,
                file=f'{key}/{fx}.npz',
                sha256=gate.file_sha(g / f'baselines/{key}/{fx}.npz'),
                outputs=gate.outputs(r),
                canonical=True,
            )
        ]
    (g / 'baselines/manifest.json').write_text(json.dumps(M))
    golden = gate.golden_set(M, ['FX-C3', 'FX-N', 'FX-AP'])
    (g / 'anchors.json').write_text(json.dumps(dict(anchors=gate.pairs(F, loaded, golden))))
    return tmp_path


def run_gate(tmp, reuse, *extra):
    out = tmp / ('out' + str(len(list(tmp.glob('out*')))))
    rc = gate.main(['--all', '--reuse', str(reuse), '--out', str(out), '--no-historical', '--no-persistent', *extra])
    return rc, json.loads((out / 'gate.json').read_text())


def test_bytes_signed_zero_nan_dtype_shape():
    a = np.array([0.0, 1.0], 'f4')
    assert gate.regress({'x': gate.sha(a)}, {'x': gate.sha(a.copy())})['pass_']
    nan2 = np.array([np.nan, 1.0], 'f4')
    nan2.view('u4')[0] ^= 1
    for b in (np.array([-0.0, 1.0], 'f4'), np.array([np.nan, 1.0], 'f4'), a.astype('f8'), a.reshape(1, 2)):
        assert not gate.regress({'x': gate.sha(a)}, {'x': gate.sha(b)})['pass_']
    assert gate.sha(np.array([np.nan], 'f4')) != gate.sha(nan2[:1])


def test_tolerance_rtol_atol_boundaries():
    r = np.array([1.0, 2.0])
    assert (
        gate.tolerance(r * (1 + 0.999e-3), r, 0.0)['pass_'] and not gate.tolerance(r * (1 + 1.001e-3), r, 0.0)['pass_']
    )
    assert not gate.tolerance(r + 0.0021, r, 0.0)['pass_'] and gate.tolerance(r + 0.0021, r, 0.0011)['pass_']
    bad = gate.tolerance(np.array([1.0, np.nan]), r, 1.0)
    assert not bad['pass_'] and bad['finite'] is False
    assert not gate.tolerance(r[:1], r, 0.0)['pass_']


def test_ratchet_statistics_and_limits():
    ref = np.array([[[1.0, 1e-6, 2.0]]])
    cand = ref * np.array([1 + 1e-5, 1 + 1e-2, 1 - 2e-5])
    st = gate.ratchet_stats(cand, ref, np.ones((2, 3)), np.ones((2, 3)), [-6.0])
    assert st['S_max']['-6'] == pytest.approx(1e-2) and st['S_rms']['-6'] == pytest.approx(np.sqrt((1e-10 + 4e-10) / 2))
    assert st['S_src'] == 0.0 and gate.l2_by_wavelength(np.ones((2, 1)), np.zeros((2, 1)))[0] == np.inf
    anchors = {'p': dict(S_max={'-6': 1e-5}, S_src=0.0)}
    ok, rows = gate.ratchet_check({'p': dict(S_max={'-6': 3e-5}, S_src=1e-6)}, anchors)
    assert ok
    for bad in (
        dict(S_max={'-6': 3.1e-5}, S_src=0.0),
        dict(S_max={'-6': 0.0}, S_src=1.1e-6),
        dict(S_max={'-6': np.nan}, S_src=0.0),
    ):
        assert not gate.ratchet_check({'p': bad}, anchors)[0]


def test_pass_cross_mode_and_negative_controls(home):
    rc, g = run_gate(home, runs(home / 'same'))
    assert rc == 0 and g['gates']['R']['FX-C3']['mode'] == 'same-toolkey', g.get('error')
    rc, g = run_gate(
        home, runs(home / 'other_gpu', cc='8.9', rel_s2=3e-6)
    )  # new toolkey, different bytes: L4 records only
    assert rc == 0 and g['gates']['R']['FX-S2']['result'] == 'differs' and g['gates']['R']['FX-S2']['blocking'] is False
    rc, g = run_gate(home, home / 'same', '--inject', 'bytes')
    assert rc == 0 and g['gates']['R']['FX-C3']['blocking'] is False
    rc, g = run_gate(home, home / 'same', '--inject', 'golden')
    assert (rc, g['code']) == (1, 'E_GATE_RATCHET') and g['gates']['P']['FX-C3']['pass_'] is True


def test_albedo_fixture_uses_fixed_cuda_tolerance(home):
    # FX-A (albedo 0.3): fixed canonical CUDA tolerance protects its boundary term; no G-P.
    F = gate.load_fixtures()
    assert (
        gate.fixture_state(F, 'FX-A')['parameters']['albedo'] == 0.3
        and gate.fixture_state(F, 'FX-C3')['parameters']['albedo'] == 0.0
    )
    assert gate.fixture_state(F, 'FX-A', solve=True)['parameters']['albedo'] == 0.3 and gate.goldens(F, ['FX-A']) == []
    rc, g = run_gate(home, home / 'same' if (home / 'same').exists() else runs(home / 'same'))
    assert rc == 0 and 'FX-A' not in g['gates']['P'] and g['gates']['R']['FX-A']['pass_'], g.get('error')
    a = runs(home / 'albedo_changed')
    r = np.load(a / 'FX-A/fixed.npz')
    rad = r['radiance'].copy()
    rad.reshape(-1)[0] *= 1 + 1e-7
    np.savez(a / 'FX-A/fixed.npz', radiance=rad, source=r['source'])
    rc, g = run_gate(home, a)
    assert rc == 0 and g['gates']['R']['FX-A']['pass_'] is False and g['gates']['CUDA']['FX-A']['pass_']


def test_albedo_fixture_cross_gpu_is_tolerance_checked(home):
    # Another toolkey: old fixtures record bytes only, FX-A (no golden) must stay within rtol 1e-3 of its canonical run.
    rc, g = run_gate(home, runs(home / 'other', cc='8.9', **{'FX-A': {}}))
    assert rc == 0 and g['gates']['R']['FX-A']['mode'] == 'cross' and g['gates']['CUDA']['FX-A']['code'] is None, g.get(
        'error'
    )
    a = runs(home / 'other_bad', cc='8.9')
    r = np.load(a / 'FX-A/fixed.npz')
    rad = r['radiance'].copy()
    rad.reshape(-1)[0] *= 1 + 2e-3
    np.savez(a / 'FX-A/fixed.npz', radiance=rad, source=r['source'])
    rc, g = run_gate(home, a)
    assert (rc, g['code']) == (1, 'E_GATE_CROSS_TOL') and g['gates']['R']['FX-C3']['blocking'] is False


def test_missing_anchor_is_a_config_error(home):
    (gate.GATE / 'anchors.json').unlink()
    rc, g = run_gate(home, runs(home / 'r'))
    assert (rc, g['code']) == (2, 'E_GATE_ANCHOR_MISSING')


def test_anchor_above_stop_is_refused(home, monkeypatch):
    (gate.GATE / 'anchors.json').unlink()
    bad = runs(home / 'bad')
    for fx in ('FX-C3', 'FX-S2'):
        z = dict(np.load(bad / fx / 'fixed.npz'))
        z['radiance'] = z['radiance'] * (1 + 5e-4)
        np.savez(bad / fx / 'fixed.npz', **z)
    F = gate.load_fixtures()
    monkeypatch.setattr(gate, 'run_all', lambda F_, names, *a, **k: {fx: gate.load_run(bad, fx) for fx in names})
    args = gate.argparse.Namespace(
        out=home / 'b',
        staged=None,
        measure_anchors=True,
        decision='test-anchor-measurement',
        atol=0.0,
        canonical=True,
        reason='t',
    )
    report = dict(fixtures={}, gates={}, code=None)
    assert gate.bless(F, list(F['fixtures']), args, report) is False and report['code'] == 'E_GATE_ANCHOR_HIGH'
    assert not (gate.GATE / 'anchors.json').exists() and 'FX-C3|golden' in report['anchor_high']


def test_tampered_baseline_is_a_config_error(home):
    M = json.loads((gate.GATE / 'baselines/manifest.json').read_text())
    with (gate.GATE / 'baselines' / M['cuda']['FX-N'][0]['file']).open('ab') as f:
        f.write(b'x')
    rc, g = run_gate(home, runs(home / 'r'))
    assert (rc, g['code']) == (2, 'E_GATE_MANIFEST_SHA')


def test_requested_vs_measured_controls_exit_2(home):
    bad = opts([])
    bad[0][1] = '--fmad=true'  # exact fixture compiled with FMA contraction
    rc, g = run_gate(home, runs(home / 'o', **{'FX-C3': dict(options=bad)}))
    assert (rc, g['code']) == (2, 'E_GATE_CONFIG')
    rc, g = run_gate(home, runs(home / 's', **{'FX-N': dict(stage1=['FMAD'])}))
    assert (rc, g['code']) == (2, 'E_GATE_CONFIG')


def test_more_s2_iterations_rejected(home):
    rc, g = run_gate(home, runs(home / 'r', s2_evals=8))
    a = g['gates']['A']['FX-S2']
    assert rc == 1 and (a['code'], a['evaluations']) == ('E_GATE_APPROX_ITER', [8, 7])


def test_exception_keeps_gate_json_and_exit_2(home):
    rc, g = run_gate(home, runs(home / 'r'), '--inject', 'raise')
    assert (rc, g['code']) == (2, 'E_GATE_EXCEPTION') and 'injected' in g['error'] and g['traceback']


def test_injected_or_reused_runs_cannot_bless(home):
    rc = gate.main(['--bless', '--inject', 'bytes', '--out', str(home / 'b')])
    g = json.loads((home / 'b/gate.json').read_text())
    assert (rc, g['code']) == (2, 'E_GATE_CONFIG')


def test_gate_flags_exception_writes_json(tmp_path, monkeypatch):
    from sosfuse2 import gate_flags

    def boom(*a, **k):
        raise RuntimeError('no GPU')

    monkeypatch.setattr(gate_flags, 'controls', lambda *a, **k: [])  # no os.environ writes in-process
    monkeypatch.setattr(gate_flags, 'reduced', boom)
    monkeypatch.setattr(sys, 'argv', ['g', '--out', str(tmp_path / 'a')])
    assert gate_flags.main() == 2 and json.loads((tmp_path / 'a/gate.json').read_text())['code'] == 'E_GATE_EXCEPTION'
    monkeypatch.setattr(gate_flags, 'reduced', lambda *a: (dict(pass_=True), None))
    monkeypatch.setattr(gate_flags, 'evaluate', boom)
    monkeypatch.setattr(sys, 'argv', ['g', '--out', str(tmp_path / 'b')])
    assert gate_flags.main() == 2  # was KeyError 'pass' (gate_flags.py:108-109)
    g = json.loads((tmp_path / 'b/gate.json').read_text())
    assert g['pass'] is False and g['exact']['C3']['code'] == 'E_GATE_EXCEPTION'


def test_boundary_ftz_numpy_matches_cuda_contract():
    """G-B: production reference.sun_columns lit at the FTZ boundary = the CUDA -ftz table
    (-FLT_MIN 0, -min subnormal 1, -0 1, +0 1, +min subnormal 1, +FLT_MIN 1)."""
    from sosfuse2 import reference as ref
    from sosfuse2.arithmetic import F, R

    tiny = np.finfo('f4').tiny
    sub = np.nextafter(F(0), F(1))
    top = np.nextafter(tiny, F(0))  # top = largest subnormal
    v = np.array([-tiny, -sub, -0.0, 0.0, sub, tiny, -top, top, -np.nextafter(tiny, F(1)), np.nan], 'f4')
    c = {'sun_h': np.array([0, 80], 'f4'), 'sun_col': np.ones((2, 2, 5), 'f4')}
    q = {
        'h': np.zeros(len(v), 'f4'),
        'r_DS': (np.full(len(v), R, 'f4'), np.zeros(len(v), 'f4')),
        'qz_DS': (v * R, np.zeros(len(v), 'f4')),
    }
    assert ref.sun_columns(q, c)[1].tolist() == [False, True, True, True, True, True, True, True, False, False]
    pair = lambda h, l: float(ref.ftz_value((np.array([h], 'f4'), np.array([l], 'f4')))[0])  # noqa: E731
    # DS hi/lo: an FTZ build drops the subnormal lo first, so (-FLT_MIN, +min sub) stays -FLT_MIN (lit false, as CUDA).
    assert (pair(-tiny, sub), pair(sub, 0.0), pair(tiny, -sub), pair(-0.0, 0.0)) == (
        -float(tiny),
        0.0,
        float(tiny),
        0.0,
    )
    assert np.isnan(pair(np.nan, 0.0))


def _log_blend_corners(traces, weights):
    """Exercise the reference's corner moments and scale, retaining its linear tensor accumulation."""
    from sosfuse2 import reference as ref

    traces, weights = np.broadcast_arrays(np.asarray(traces, 'f4'), np.asarray(weights, 'f4'))
    traces = traces.reshape(-1, 4)
    weights = weights.reshape(-1, 4)
    val = np.zeros((len(traces), 1, 4), 'f4')
    logs = np.zeros((len(traces), 1), 'f4')
    wg = np.zeros_like(logs)
    am = np.zeros_like(logs)
    for corner in range(4):
        tensor = np.zeros_like(val)
        tensor[:, 0, 0] = traces[:, corner]
        with np.errstate(invalid='ignore'):  # val intentionally retains 0*NaN, just as transport does
            val += weights[:, corner, None, None] * tensor
        ref.trace_accumulate(traces[:, corner, None], weights[:, corner, None], logs, wg, am)
    return val, ref.trace_scale(val, logs, wg, am), ref.trace_scale(np.ones_like(val), logs, wg, am)[:, 0, 0]


@pytest.mark.parametrize('pair', [(1.0, 9.0), (1e-2, 1e-5), (1e-2, 1e-8), (1.0, 1e-30)])
@pytest.mark.parametrize('axis', ['ac', 'ah'])
def test_log_blend_continuity(pair, axis):
    """Bad-corner weight b changes the result continuously on both bilinear axes, even at an exact grid node."""
    b = np.array([0.0, np.nextafter(np.float32(0), np.float32(1)), 1.4e-6, 1.4e-5, 1e-3, 0.3, 1 - 1.4e-5, 1.0], 'f4')
    weights = np.stack([0.5 * (1 - b), 0.5 * b, 0.5 * (1 - b), 0.5 * b], axis=-1)
    traces = np.array([pair[0], 0.0, pair[1], 0.0], 'f4')
    if axis == 'ah':
        weights = weights[:, [0, 2, 1, 3]]
        traces = traces[[0, 2, 1, 3]]
    val, result, scale = _log_blend_corners(traces, weights)
    assert np.isfinite(result).all() and np.isfinite(scale).all()
    assert (scale <= 1 + 1e-5).all()
    # Near the node (b <= 1e-3) the result itself moves by O(b). Over the whole sweep the log of the scale moves by
    # b*ln(A/G) <= 69*b; the result may move far for large b, since b -> 1 is the linear limit by design.
    small = b <= 1e-3
    relative = np.abs(result[:, 0, 0].astype('f8') / result[0, 0, 0] - 1)
    assert (relative[small] <= 50 * b[small].astype('f8')).all(), (b, relative)
    log_step = np.abs(np.log(scale.astype('f8')) - np.log(np.float64(scale[0])))
    assert (log_step <= 69 * b.astype('f8') + 1e-5).all(), (b, log_step)
    assert scale[-1] == 1 and np.array_equal(result[-1], val[-1])
    if pair == (1.0, 9.0):
        assert result[0, 0, 0] == pytest.approx(3.0, rel=1e-6)
    # All-good stencils must recover the original geometric/linear trace ratio without a special branch.
    good = np.array([pair[0], pair[0], pair[1], pair[1]], 'f4')
    if axis == 'ah':
        good = good[[0, 2, 1, 3]]
    val, result, scale = _log_blend_corners(good, weights)
    logs = np.sum(weights.astype('f8') * np.log(good.astype('f8')), axis=1)
    expected = np.exp(logs) / ((val[:, 0, 0] + val[:, 0, 1]) + val[:, 0, 2])
    np.testing.assert_allclose(scale, expected, rtol=1e-5)
    assert np.isfinite(result).all() and (scale <= 1 + 1e-5).all()


@pytest.mark.parametrize(
    'traces,weights',
    [
        ([0.0, 0.0, 0.0, 0.0], [0.25] * 4),
        ([-1.0, -2.0, -3.0, -4.0], [0.25] * 4),
        ([np.nextafter(np.float32(0), np.float32(1)), 0.0, -1.0, 0.0], [0.25] * 4),
        ([1.0, 9.0, -2.0, 0.0], [0.25] * 4),
        ([1.0, 9.0, np.nan, 0.0], [0.25] * 4),
        ([1.0, 9.0, np.nan, 0.0], [0.5, 0.5, 0.0, 0.0]),
        ([np.nan] * 4, [0.25] * 4),
        ([np.finfo('f4').tiny, 0.0, 0.0, 0.0], [0.25] * 4),
    ],
)
def test_log_blend_adversarial(traces, weights):
    """Invalid and subnormal traces cannot poison the scale; only the original val may propagate NaN."""
    from sosfuse2 import reference as ref

    with np.errstate(invalid='raise', divide='raise', over='raise'):
        val, result, scale = _log_blend_corners(traces, weights)
    assert np.isfinite(scale).all() and (scale <= 1 + 1e-5).all()
    np.testing.assert_array_equal(np.isnan(result), np.isnan(val))
    assert np.isfinite(result[~np.isnan(val)]).all()
    tr = np.asarray(traces, 'f8')
    w = np.asarray(weights, 'f8')
    good = tr >= ref.TRACE_MIN
    wg = w[good].sum()
    am = (w[good] * tr[good]).sum()
    expected = (
        np.exp(np.sum(w[good] * np.log(tr[good])) - wg * np.log(am / wg)) if wg > 0 and am >= ref.TRACE_MIN else 1.0
    )
    np.testing.assert_allclose(scale, expected, rtol=1e-5)


@pytest.mark.parametrize('perturbation', ['atan2-bit0', 'on-grid'])
def test_log_blend_removes_one_ulp_sensitivity(tmp_path, monkeypatch, perturbation):
    """The log-space blend keeps FX-N radiance at -9/-12 deg within 2e-4 under bit-selected or exact-grid +1 ULP perturbations."""
    from sosfuse2 import reference as ref

    F = gate.load_fixtures()
    env = gate.fixture_env(F, 'FX-N')
    for k in ('SOSFUSE2_KERNEL_FLAGS', 'SOSFUSE2_BLOCK_SIZE', 'SOSFUSE2_STRATO_ASYNC') + tuple(
        'SOSFUSE2_' + f for f in gate.STAGE1
    ):
        monkeypatch.setenv(k, env[k])
    for k in ('COLUMN', 'SUN', 'GROUND', 'GEOMETRY'):
        monkeypatch.setenv(f'SOSFUSE2_{k}_WORKERS', '2')
    monkeypatch.delenv('STRATO_ATTACH_CACHE', raising=False)
    gate.worker('fixed', 'FX-N', tmp_path / 'base', 'numpy')

    class Ulp:
        def __init__(self, mask=None):
            self.mask = mask

        def __getattr__(self, name):
            return getattr(np, name)

        def arctan2(self, y, x):
            a = np.arctan2(y, x)
            mask = (np.asarray(y, 'f4').view('u4') & 1).astype(bool) if self.mask is None else self.mask
            return np.where(mask, np.nextafter(a, np.float32(np.inf)), a)

    changed = 0
    if perturbation == 'atan2-bit0':
        monkeypatch.setattr(ref, 'np', Ulp())
    else:
        original_sample = ref.sample

        def on_grid(*args, **kwargs):
            nonlocal changed
            q = original_sample(*args, **kwargs)
            mask = (q['sw'][:, 1] == 0) & (q['sw'][:, 3] == 0)  # exactly ac == 0, including ah endpoints
            if mask.any():
                with monkeypatch.context() as m:
                    m.setattr(ref, 'np', Ulp(mask))
                    perturbed = original_sample(*args, **kwargs)
                changed += np.count_nonzero(q['valid'] & mask & np.any(perturbed['sw'] != q['sw'], axis=1))
                return perturbed
            return q

        monkeypatch.setattr(ref, 'sample', on_grid)
    gate.worker('fixed', 'FX-N', tmp_path / 'ulp', 'numpy')
    if perturbation == 'on-grid':
        assert changed > 0, 'no valid on-grid sample was perturbed'
    a, b = (np.load(tmp_path / d / 'fixed.npz')['radiance'] for d in ('ulp', 'base'))
    st = gate.ratchet_stats(a[-1], b[-1], np.zeros((1, 1)), np.zeros((1, 1)), F['observer_solar_deg'])
    assert max(st['S_max']['-9'], st['S_max']['-12']) <= 2e-4, st


def test_chunked_source_solve_meta():
    """BM45 (45 wavelengths): run_gpu_state_v3 writes one source_solve per FW=15 chunk; a single chunk keeps the dict form."""
    one = dict(evaluations=8, converged=[True] * 15)
    many = [
        dict(evaluations=8, converged=[True] * 15),
        dict(evaluations=9, converged=[True] * 15),
        dict(evaluations=7, converged=[True, False]),
    ]
    assert gate.solve_evaluations(one) == 8 and gate.solve_converged(one)
    assert (
        gate.solve_evaluations(many) == [8, 9, 7]
        and not gate.solve_converged(many[:3])
        and gate.solve_converged(many[:2])
    )
    assert not gate.more_iterations(8, 8) and gate.more_iterations(9, 8)
    assert (
        not gate.more_iterations([8, 9], [8, 9])
        and gate.more_iterations([8, 10], [9, 9])
        and gate.more_iterations([8], [8, 9])
    )


@pytest.mark.parametrize('cc', ['8.6', '8.9'])
def test_automatic_bless_cannot_move_fixed_baseline(home, monkeypatch, cc):
    F = gate.load_fixtures()
    base = runs(home / 'drift', cc=cc)
    loaded = {'FX-A': gate.load_run(base, 'FX-A')}
    original = loaded['FX-A']['radiance'].copy()
    monkeypatch.setattr(gate, 'run_all', lambda *a, **k: loaded)
    args = gate.NS(
        out=home / 'b',
        staged=None,
        measure_anchors=False,
        decision=None,
        atol=0.0,
        canonical=False,
        reason='drift test',
    )
    fixed = gate.manifest()
    canonical = fixed['cuda']['FX-A'][0]
    fixed_sha = gate.file_sha(gate.GATE / 'baselines' / canonical['file'])
    passed = []
    for n in range(1, 101):
        loaded['FX-A']['radiance'] = original * 1.0005**n
        report = dict(fixtures={}, gates={})
        passed.append(gate.bless(F, ['FX-A'], args, report))
        if not passed[-1]:
            assert report['gates']['CUDA']['FX-A']['code'] == 'E_GATE_CROSS_TOL'
    assert passed[0] and not any(passed[1:])
    current = gate.manifest()
    assert current['cuda'] == fixed['cuda'] and current['golden'] == fixed['golden']
    assert gate.file_sha(gate.GATE / 'baselines' / canonical['file']) == fixed_sha
    assert current['records']['cuda']['FX-A']


def test_required_ratchet_pair_missing(home):
    F = gate.load_fixtures()
    loaded = {'FX-S2': gate.load_run(runs(home / 'partial'), 'FX-S2')}
    with pytest.raises(gate.GateError, match='E_GATE_RATCHET_MISSING'):
        gate.pairs(F, loaded, {})
    rc, report = run_gate(home, home / 'partial', '--fixtures', 'FX-S2')
    assert rc == 2 and report['code'] == 'E_GATE_RATCHET_MISSING'


@pytest.mark.parametrize(
    'delta,decision,expected', [(5e-4, None, True), (2e-3, 'approved-id', False), (2e-3, None, False)]
)
def test_gf_fixed_tolerance_and_decision_no_bypass(home, monkeypatch, delta, decision, expected):
    testing.sky_sos(
        'rig/bake/bake_run.py'
    )  # G-F loads the sky-sos bake template (research consumer): skip without the checkout
    F = gate.load_fixtures()
    sid = F['state']['id']
    out = home / 'gf_test'
    ref = home / 'reference.npz'
    np.savez(ref, radiance=np.ones((2, 3)))
    M = gate.manifest()
    M['gf'] = dict(radiance_sha256=gate.sha(np.ones((2, 3))), reference=str(ref))
    gate.jdump(gate.GATE / 'baselines/manifest.json', M)

    def fake_child(*a, **kw):
        work = out / 'gf/work'
        d = work / 'out' / sid
        d.mkdir(parents=True, exist_ok=True)
        t = work / 'tmp' / sid
        t.mkdir(parents=True, exist_ok=True)
        np.savez(d / 'table.npz', radiance=np.ones((2, 3)) * (1 + delta))
        np.savez(t / 'source.npz', source=np.ones((2, 3)))
        gate.jdump(d / 'meta.json', dict(source_solve=dict(evaluations=3, converged=[True]), kernel={}))
        logs = work / 'logs' / 'fake'
        logs.mkdir(parents=True, exist_ok=True)
        (logs / 'worker.log').write_text(json.dumps(dict(event='PHASE', phase='attach_cache', result='hit')) + '\n')
        gate.jdump(work / 'pipeline_summary.json', dict(logs=str(logs)))
        return 0, [], None

    monkeypatch.setattr(gate, 'child', fake_child)
    report = dict(gates={})
    args = gate.NS(out=out, reference=ref, decision=decision, reason='test')
    assert gate.gf(args, F, report) == expected
    assert report['gates']['F']['pass_'] == expected
    assert report['gates']['F']['change']['pass_'] == expected
    assert all(row['attach'] == ['hit'] for row in report['gates']['F']['runs'].values())
    assert gate.manifest()['gf'] == M['gf']
    if not expected:
        assert report['gates']['F']['code'] == 'E_GATE_GF_TOL'


def test_main_restores_records_and_checks_writable_baselines(home, monkeypatch):
    """C-07/C-08: RECORDS is per call (restored after main), and a bless mode on read-only package baselines is E_GATE_CONFIG."""
    assert gate.RECORDS is None
    run_gate(home, runs(home / 'r'))
    assert gate.RECORDS is None
    monkeypatch.setattr(gate.os, 'access', lambda path, mode: False)
    rc = gate.main(['--bless-numpy', '--out', str(home / 'ro')])
    g = json.loads((home / 'ro/gate.json').read_text())
    assert (rc, g['code']) == (2, 'E_GATE_CONFIG') and 'not writable' in g['error']
    assert gate.RECORDS is None


def test_commit_and_portable_reference(monkeypatch, tmp_path):
    """C-03/C-05: the recorded commit is GATE_SOURCE_COMMIT or the package's own; a reference path never records a host path."""
    from sosfuse2 import run_gpu_state_v3 as v3

    monkeypatch.setattr(v3, 'package_info', lambda: dict(package_version='9.9.9', package_commit='feedface'))
    monkeypatch.delenv('GATE_SOURCE_COMMIT', raising=False)
    assert gate.commit() == 'feedface'
    monkeypatch.setenv('GATE_SOURCE_COMMIT', 'cafe')
    assert gate.commit() == 'cafe'
    monkeypatch.setenv('HOME', str(tmp_path))
    assert gate.portable(tmp_path / 'a/b.npz') == '~/a/b.npz' and gate.portable('~/a/b.npz') == '~/a/b.npz'
    assert gate.portable('/somewhere/else/table.npz') == 'table.npz'


def test_record_paths_follow_records_or_the_package(tmp_path, monkeypatch):
    """gate.record(): RECORDS set -> every execution record under it; RECORDS None (bless modes) -> the package's own files."""
    pkg = tmp_path / 'gate'
    records = tmp_path / 'records'
    monkeypatch.setattr(gate, 'GATE', pkg)
    wanted = ('baselines/manifest.json', 'baselines/BASELINE_LOG.md', 'anchors.json')
    monkeypatch.setattr(gate, 'RECORDS', records)
    assert [gate.record(r) for r in wanted] == [
        records / 'manifest.json',
        records / 'BASELINE_LOG.md',
        records / 'anchors.json',
    ]
    monkeypatch.setattr(gate, 'RECORDS', None)
    assert [gate.record(r) for r in wanted] == [pkg / r for r in wanted]
    assert not any(p.is_file() for p in tmp_path.rglob('*'))  # paths only: no record file was written
