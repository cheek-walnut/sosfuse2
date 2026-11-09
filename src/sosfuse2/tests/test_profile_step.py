"""②b PROFILE node count 161/321/641 (0.5/0.25/0.125 km), CPU only.

Eight groups:
  (a) grids keep the asset nodes and integrals; 0.5 is the historical table byte for byte
  (b) E_PROFILE_* rejections                      (c) clang++ profile_value == interpolate32 per N, profile_at masks
  (d) injected q_eff used as given, 0.125 state round trip through effective_state/build/signature, S1 adapter
  (e) endpoint columns vs an independent knot-split GL integral (vertical, off-grid tangent, short)
  (f) Simpson prefix: constant/linear/triangle exact, kinked profile converges with more nodes
  (g) one RawKernel key rule, -DPROFILE_N only off 161, shared raw dict 161->641->161, consumption check without setup,
      payload/promotion, precompile --profile-step-km/--W, gate toolkey/config
  (h) profile identity separates tables/policies, host/column/device caches A->B->A cold/warm, outputs independent of N
The independent references are test-local; column_table.linear_columns is never used as one. About 30 s on a Mac.
"""

from sosfuse2 import paths
import copy
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = paths.PKG  # the package directory (kernel, configs, assets)
from sosfuse2 import column_table, density, fused, geometry, host_cache, reference, testing  # noqa: E402
from sosfuse2.arithmetic import F  # noqa: E402

try:
    from sosfuse2.numerical_checks import assert_equivalent, within_tol
except ImportError:  # standalone research harness
    from numerical_checks import assert_equivalent, within_tol

R = 6371.0
STEPS = ((0.5, 161), (0.25, 321), (0.125, 641))
ASSET = dict(density_profile='usstd76', ozone_profile='usstd76', rayleigh_column='usstd')


def state(**extra):
    s = copy.deepcopy(geometry.normal_state())
    s['density'] = dict(ASSET, **extra)
    return s


def kink(z, center=24.3, width=6.0):
    return np.exp(-np.abs(np.asarray(z) - center) / width)


def external(n, q=None, mask=2, base=None, **kw):
    z = density.grid(n)
    q = kink(z) if q is None else q
    return density.with_profile(
        state() if base is None else base,
        air_surface_number_density_m3=2.5469e25,
        z_km=z,
        table=np.column_stack([np.ones(n), q]),
        mask=mask,
        **kw,
    )


@pytest.fixture(scope='module')
def env():
    with pytest.MonkeyPatch.context() as m:
        for k, v in dict(
            SOSFUSE2_SUN_WORKERS='1',
            SOSFUSE2_GROUND_WORKERS='1',
            SOSFUSE2_COLUMN_WORKERS='1',
            SOSFUSE2_HOST_CACHE='0',
            SOSFUSE2_COLUMN_METHOD='table-v1',
        ).items():
            m.setenv(k, v)
        for k in (
            'SOSFUSE2_COLUMN_CACHE',
            'SOSFUSE2_COLUMN_TABLE_CACHE',
            'SOSFUSE2_COLUMN_CACHE_JOB',
            'STRATO_ATTACH_CACHE',
        ):
            m.delenv(k, raising=False)
        yield m


def create(tmp_path, st):
    return geometry.create(tmp_path, 'selftest', [550.0], 'numpy', state=st)


# ------------------------------------------------------------------------------------------------ independent reference


def ray_height(p, d, t):
    sa = float(p @ d)
    rho = math.sqrt(max(float(p @ p) - sa * sa, 0.0))
    return np.hypot(rho, sa + np.asarray(t, 'f8')) - R


def ref_integral(f, p, d, a, b, knots, n=64):
    """FP64 GL-n between the ray's crossings of every knot shell and the tangent point (test-local, independent)."""
    sa = float(p @ d)
    rho = math.sqrt(max(float(p @ p) - sa * sa, 0.0))
    cuts = [a, b] + ([-sa] if a < -sa < b else [])
    for hh in knots:
        c2 = (R + hh) ** 2 - rho * rho
        if c2 > 0:
            cuts += [t for t in (-sa - math.sqrt(c2), -sa + math.sqrt(c2)) if a < t < b]
    cuts = np.unique(cuts)
    x, w = np.polynomial.legendre.leggauss(n)
    x = (x + 1) / 2
    w = w / 2
    lo, hi = cuts[:-1], cuts[1:]
    tt = lo[:, None] + (hi - lo)[:, None] * x[None, :]
    return float(np.sum(f(ray_height(p, d, tt.ravel())).reshape(tt.shape) * w * (hi - lo)[:, None]))


def rays():
    out = {}
    p = np.array([0.0, 0.0, R])
    out['vertical'] = (p, np.array([0.0, 0.0, 1.0]), 80.0)
    r0, rt = R + 30.0, R + 17.3
    c = rt / r0
    d = np.array([c, 0.0, -math.sqrt(1 - c * c)])
    p = np.array([0.0, 0.0, r0])
    sa = float(p @ d)
    out['tangent17.3'] = (p, d, -sa + math.sqrt((R + 80.0) ** 2 - rt * rt))
    e = math.radians(30.0)
    out['short0.8'] = (np.array([0.0, 0.0, R + 24.93]), np.array([math.cos(e), 0.0, math.sin(e)]), 0.8)
    return out


# ------------------------------------------------------------------------------------------------ (a)


def test_a_grids_keep_asset_nodes_and_integrals(subtests):
    assert density.grid(161).tobytes() == (np.arange(161) * 0.5).tobytes()
    _, z, air, oz = density.load_asset()
    expected = np.column_stack([air * density.LEGACY_COLUMN / np.trapezoid(air, z), oz * 15 / np.trapezoid(oz, z)])
    half = density.build(state())
    assert (within_tol(half['table'][:, :2], expected)) and not half['table'].flags.writeable
    assert set(half['meta']) == {
        *ASSET,
        'asset_sha256',
        'rayleigh_tau550',
        'ozone_column',
        'interpolation',
        'no2',
        'o4',
        'profile_identity',
    }  # historical keys
    assert half['meta']['interpolation'] == 'linear 0.5 km'
    assert within_tol(density.build(state(profile_step_km=0.5))['table'][:, :2], expected)
    assert density.build(dict(density=dict(profile_step_km=0.125)))['table'].shape == (
        641,
        3,
    )  # exp8/triangle: nothing to refine, asset unopened
    for step, n in STEPS[1:]:
        with subtests.test(step=step):
            o = density.build(state(profile_step_km=step))
            zz = density.grid(n)
            assert o['table'].shape == (n, 3) and np.array_equal(zz, np.linspace(0, 80, n))
            assert within_tol(o['table'][:: round(0.5 / step), :2], expected)  # normalized on the 161 asset nodes first
            for k in (0, 1):
                assert np.trapezoid(o['table'][:, k], zz) == pytest.approx(np.trapezoid(expected[:, k], z), rel=1e-14)
            assert (o['meta']['profile_step_km'], o['meta']['origin']) == (step, 'asset')
            assert o['meta']['profile_identity'] == density.profile_identity(o)['sha256']


# ------------------------------------------------------------------------------------------------ (b)


def test_b_rejections_have_stable_codes(subtests):
    z, good = density.grid(321), np.column_stack([np.ones(321), kink(density.grid(321))])
    wp = lambda base=None, **kw: density.with_profile(
        state() if base is None else base,
        **dict(dict(air_surface_number_density_m3=2.5469e25, z_km=z, table=good, mask=2), **kw),
    )
    bad = dict(
        step_value=(lambda: density.build(state(profile_step_km=0.3)), 'E_PROFILE_STEP'),
        step_bool=(lambda: density.profile_n(True), 'E_PROFILE_STEP'),
        step_text=(lambda: density.build(state(profile_step_km='0.25')), 'E_PROFILE_STEP'),
        step_null=(lambda: density.profile_step_km(state(profile_step_km=None)), 'E_PROFILE_STEP'),
        unknown_key=(lambda: density.options(dict(density=dict(profile_nodes=161))), 'E_DENSITY_OPTIONS'),
        grid_count=(
            lambda: wp(air_surface_number_density_m3=2.5469e25, z_km=np.arange(160) * 0.5, table=good[:160]),
            'E_PROFILE_GRID',
        ),
        grid_values=(lambda: wp(air_surface_number_density_m3=2.5469e25, z_km=z + 1e-9), 'E_PROFILE_GRID'),
        table_shape=(lambda: wp(table=np.ones((321, 3))), 'E_PROFILE_TABLE'),
        table_nan=(lambda: wp(table=np.where(good == good.max(), np.nan, good)), 'E_PROFILE_TABLE'),
        table_negative=(lambda: wp(table=good - 2), 'E_PROFILE_TABLE'),
        table_complex=(lambda: wp(table=good.astype(complex)), 'E_PROFILE_TABLE'),
        table_fp32_overflow=(lambda: wp(table=good * 1e39), 'E_PROFILE_TABLE'),
        mask_bool=(lambda: wp(mask=True), 'E_PROFILE_MASK'),
        mask_range=(lambda: wp(mask=4), 'E_PROFILE_MASK'),
        mask_float=(lambda: wp(mask=2.0), 'E_PROFILE_MASK'),
        air_zero=(lambda: wp(table=np.column_stack([np.zeros(321), good[:, 1]]), mask=1), 'E_PROFILE_AIR'),
        normalize_zero=(
            lambda: wp(table=np.column_stack([good[:, 0], np.zeros(321)]), normalize_ozone=True),
            'E_PROFILE_OZONE_ZERO',
        ),
        normalize_type=(lambda: wp(normalize_ozone=1), 'E_PROFILE_NORMALIZE'),
        scale_zero=(lambda: wp(scale=0.0), 'E_PROFILE_SCALE'),
        scale_bool=(lambda: wp(scale=True), 'E_PROFILE_SCALE'),
        explicit_step_conflict=(lambda: wp(base=state(profile_step_km=0.5)), 'E_PROFILE_STEP_CONFLICT'),
        payload_keys=(
            lambda: density.build(
                state(external_profile=dict(version=1, air_surface_number_density_m3=2.5469e25, z_km=z.tolist()))
            ),
            'E_PROFILE_PAYLOAD',
        ),
        options_conflict=(
            lambda: density.with_options(state(profile_step_km=0.25), SimpleNamespace(profile_step_km=0.125)),
            'E_PROFILE_STEP_CONFLICT',
        ),
    )
    for name, (call, code) in bad.items():
        with subtests.test(name=name):
            with pytest.raises(ValueError, match='^' + code):
                call()


# ------------------------------------------------------------------------------------------------ (c)


@pytest.mark.parametrize('n', (161, 321, 641))
def test_c_native_profile_value_matches_interpolate32(n, tmp_path):
    """kernel.cu's own PROFILE_N block, clip() and profile_value (text cut from kernel.cu, not copied here) compiled by
    clang++ for the host (-DPROFILE_N only off 161) vs density.interpolate32, bit for bit."""
    testing.require_tool('clang++')
    src = (HERE / 'kernel.cu').read_text()
    clip = re.search(r'__device__ float clip\(.*\n', src)[0]
    start = src.index('#ifndef PROFILE_N')
    body = src[start : src.index('\n}\n', src.index('float profile_value(', start)) + 3]
    cpp = tmp_path / 'profile.cpp'
    lib = tmp_path / 'profile.so'
    cpp.write_text(
        '#include <algorithm>\n#include <cmath>\n#define __device__\nusing std::min; using std::fminf; using std::fmaxf;\n'
        + clip
        + body
        + 'extern "C" float profile_c(float h,const float*t,int k){return profile_value(h,t,k);}\n'
    )
    opts = [] if n == 161 else [f'-DPROFILE_N={n}']
    link = ['-dynamiclib'] if platform.system() == 'Darwin' else ['-shared']
    subprocess.run(
        ['clang++', '-std=c++17', '-O1', '-ffp-contract=off', '-fPIC', *link, *opts, str(cpp), '-o', str(lib)],
        check=True,
    )
    import ctypes

    native = ctypes.CDLL(str(lib))
    native.profile_c.argtypes = [ctypes.c_float, ctypes.c_void_p, ctypes.c_int]
    native.profile_c.restype = ctypes.c_float
    rng = np.random.default_rng(n)
    table = np.ascontiguousarray(rng.uniform(0.1, 2.0, (n, 3)), 'f4')
    nodes = density.grid(n).astype('f4')
    h = np.r_[
        nodes,
        np.nextafter(nodes, np.float32(-1)),
        np.nextafter(nodes, np.float32(100)),
        rng.uniform(-1, 81, 2000),
        [-0.5, 0.0, 80.0, 80.5, np.nextafter(np.float32(80), np.float32(0))],
    ].astype('f4')
    want = density.interpolate32(h, table)
    got = np.array([[native.profile_c(float(v), table.ctypes.data, k) for k in (0, 1, 2)] for v in h], 'f4')
    assert within_tol(got, want)


def test_c_profile_at_masks_and_bad_n(subtests):
    h = np.r_[np.linspace(0, 80, 777), [17.3, 24.93, 79.99]].astype('f4')
    for step, n in STEPS:
        table = np.ascontiguousarray(
            np.column_stack(
                [np.exp(-density.grid(n) / 8.3), kink(density.grid(n)), density.no2_profile(density.grid(n))]
            ),
            'f4',
        )
        tab = density.interpolate32(h, table)
        for mask in range(4):
            with subtests.test(n=n, mask=mask):
                prof = reference.profile_at(
                    h, dict(height=F(1.2), density_optics=dict(table=table, mask=mask, meta=dict(profile_step_km=step)))
                )
                air = tab[:, 0] if mask & 1 else np.exp(-h / F(8))
                oz = tab[:, 1] if mask & 2 else np.maximum(F(0), F(1) - abs(h - F(25)) / F(15))
                assert (within_tol(prof[:, 0], air)) and (within_tol(prof[:, 2], oz))
    testing.require_tool('clang')
    r = subprocess.run(
        ['clang', '-E', '-P', '-x', 'c++', '-DFW=15', '-DPROFILE_N=200', '-'],
        input=(HERE / 'kernel.cu').read_text(),
        text=True,
        capture_output=True,
    )
    assert r.returncode != 0 and 'PROFILE_N must be 161, 321 or 641' in r.stderr


# ------------------------------------------------------------------------------------------------ (d)


def test_d_injected_values_used_as_given():
    z = density.grid(321)
    t = np.column_stack([np.ones(321), kink(z) * 0.37])
    st = json.loads(
        json.dumps(density.with_profile(state(), air_surface_number_density_m3=2.5469e25, z_km=z, table=t, mask=2))
    )  # JSON-serializable state
    o = density.build(st)
    assert o['table'][:, :2].tobytes() == t.tobytes() and (o['mask'], o['scale']) == (2, 1.0)
    assert (
        o['meta']['normalization'].startswith('none')
        and o['meta']['origin'] == 'external'
        and o['meta']['asset_sha256'] is None
    )
    assert np.trapezoid(o['table'][:, 1], z) != pytest.approx(15.0)
    n = density.build(
        density.with_profile(
            state(), air_surface_number_density_m3=2.5469e25, z_km=z, table=t, mask=2, normalize_ozone=True
        )
    )
    assert (
        np.trapezoid(n['table'][:, 1], z) == pytest.approx(15.0, rel=1e-13)
        and n['table'][:, 0].tobytes() == t[:, 0].tobytes()
    )
    assert density.profile_identity(n)['sha256'] != density.profile_identity(o)['sha256']
    for step, _ in STEPS:  # production asset path keeps 15 at every N
        a = density.build(state(profile_step_km=step))
        assert np.trapezoid(a['table'][:, 1], density.grid(len(a['table']))) == pytest.approx(15.0, rel=1e-13)


def test_d_0125_state_round_trip(tmp_path, env):
    """with_profile -> effective_state -> build / signed_contract keeps the external table and step."""
    from sosfuse2 import run_gpu_state_v3 as v3

    st = external(641, mask=2)
    eff = v3.effective_state(st, SimpleNamespace())
    assert (
        eff['density']['profile_step_km'] == 0.125
        and eff['density']['external_profile'] == st['density']['external_profile']
    )
    assert within_tol(density.build(eff)['table'], density.build(st)['table'])
    cli = v3.effective_state(
        st, SimpleNamespace(density_profile='exp8')
    )  # CLI strings are metadata; the payload mask decides
    assert cli['density']['density_profile'] == 'exp8' and density.build(cli)['mask'] == 2
    plain = v3.effective_state(state(), SimpleNamespace())
    assert (
        'profile_step_km' not in plain['density'] and 'external_profile' not in plain['density']
    )  # omitted stays omitted
    cfg = v3.settings_for(SimpleNamespace(reduced=True, settings_json=None))
    sig = lambda s: v3.signed_contract(s, cfg, 'numpy', True, 'depolarizing')[0]
    other = v3.effective_state(external(641, q=kink(density.grid(641), 30.0)), SimpleNamespace())
    assert sig(eff) == sig(copy.deepcopy(eff)) and len({sig(eff), sig(other), sig(plain)}) == 3
    s, _, _ = create(tmp_path, eff)
    assert s.kernel.profile_n == 641 and (within_tol(s.density_optics['table'], density.build(st)['table']))


def test_d_s1_qeff_adapter_keeps_air_mask_scale(tmp_path, env):
    """Fix #12: S1 form (air kept, ozone column = q_eff, mask|2, normalize_ozone=False, original scale) and its consumption."""
    base_state = state(profile_step_km=0.125)
    base = density.build(base_state)
    z = density.grid(641)
    q_eff = base['table'][:, 1] * (
        1 + 0.02 * np.sin(z / 3.0)
    )  # evaluated on the 0.125 km nodes (a stand-in for Model.effective)
    st = density.with_profile(
        state(),
        air_surface_number_density_m3=2.5469e25,
        z_km=z,
        table=np.column_stack([base['table'][:, 0], q_eff]),
        mask=base['mask'] | 2,
        normalize_ozone=False,
        scale=base['scale'],
    )
    o = density.build(st)
    assert (within_tol(o['table'][:, 0], base['table'][:, 0])) and (within_tol(o['table'][:, 1], q_eff))
    assert (o['mask'], o['scale']) == (3, base['scale'])
    s, _, _ = create(tmp_path / 's1', st)
    a, _, _ = create(tmp_path / 'asset', base_state)
    assert (within_tol(s.ext[0], a.ext[0])) and (within_tol(s.hr, a.hr))  # Rayleigh scale and air consumed as before
    c = s.kernel.setup(s, 4)
    h = np.linspace(0, 80, 999).astype('f4')
    assert within_tol(reference.profile_at(h, c)[:, 2], density.interpolate32(h, o['table'])[:, 1])
    ev = density.evidence(s.density_optics, s.kernel.profile_n)
    assert (ev['compiled_n'], ev['uploaded_n'], ev['mask'], ev['origin']) == (641, 641, 3, 'external')
    assert ev['table_f8_sha256'] == hashlib.sha256(np.ascontiguousarray(o['table']).tobytes()).hexdigest()
    assert ev['table_f4_sha256'] == hashlib.sha256(np.ascontiguousarray(o['table'], 'f4').tobytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ (e)


def test_e_columns_vs_independent_reference(subtests):
    for step, n in STEPS:
        z = density.grid(n)
        o = density.build(external(n))
        f = lambda hh, z=z, o=o: np.interp(hh, z, o['table'][:, 1])
        spec = column_table.molecular(o, 1)
        for name, (p, d, end) in rays().items():
            with subtests.test(n=n, ray=name):
                want = ref_integral(f, p, d, 0.0, end, z)
                assert want == pytest.approx(ref_integral(f, p, d, 0.0, end, z, 128), rel=1e-13)
                exact = column_table.exact_columns(spec, p[None], d[None], np.array([end]))[0]
                table = column_table.column(spec, p[None], d[None], np.array([end]))[0]
                # closed form: arcsinh primitives cancel to ~1e-10 km on sub-km pieces at r ~ 6400 km
                # table-v1: difference of two cubic lookups in the 1024^2 cumulative table (gate B accuracy class)
                assert exact == pytest.approx(want, rel=1e-10, abs=1e-9) and table == pytest.approx(
                    want, rel=2e-6, abs=5e-6
                )


# ------------------------------------------------------------------------------------------------ (f)


def simpson_prefix(table, nq, edges=(0.0, 10.0, 25.0, 40.0, 80.0)):
    """reference.SimpsonPrefix driven like transport() on a vertical ray from the ground, unit ozone extinction."""
    table = np.column_stack([table, density.no2_profile(density.grid(len(table)))])
    p, d = np.array([[0.0, 0.0, R]]), np.array([[0.0, 0.0, 1.0]])
    e = np.array([edges])
    g = dict(p=p, d=d, edges=e, lower=np.ones((1, len(edges) - 1), bool), target=np.eye(3)[None], dloc=d.copy())
    u, wu = np.polynomial.legendre.leggauss(nq)
    c = dict(
        W=1,
        u=(u + 1) / 2,
        wu=wu / 2,
        h=np.linspace(0, 80, 21).astype('f4'),
        chi=np.linspace(0, math.pi, 13).astype('f4'),
        height=F(1.2),
        ext=np.zeros((5, 1), 'f4'),
        density_optics=dict(table=table, mask=2, meta=dict(profile_step_km=density.STEP_OF_N[len(table)])),
    )
    c['ext'][2] = 1.0
    P = reference.SimpsonPrefix(g, c)
    out = []
    for seg in range(len(edges) - 1):
        P.begin(seg)
        a, b = edges[seg], edges[seg + 1]
        for step in range(nq):
            q = reference.sample(g, seg, step, c)
            out.append((a + (b - a) * c['u'][step] ** 2, float(P.node(q)[0, 0])))
        P.finish()
    return out, float((P.tau - P.tc)[0, 0])


def test_f_simpson_prefix_on_refined_grids(subtests):
    for step, n in STEPS:
        z = density.grid(n)
        cases = dict(constant=np.ones(n), linear=z / 80.0, triangle=np.maximum(0, 1 - np.abs(z - 25) / 15))
        for name, col in cases.items():
            with subtests.test(n=n, profile=name):
                nodes, total = simpson_prefix(np.column_stack([np.ones(n), col]), 12)
                prim = {
                    'constant': lambda t: t,
                    'linear': lambda t: t * t / 160.0,
                    'triangle': lambda t: ref_integral(
                        lambda hh: np.interp(hh, z, col), np.array([0, 0, R]), np.array([0, 0, 1.0]), 0.0, t, z
                    ),
                }[name]
                for t, v in nodes:
                    assert v == pytest.approx(prim(t), rel=2e-6, abs=1e-7)
                assert total == pytest.approx(prim(80.0), rel=2e-6)
        with subtests.test(n=n, profile='kinked: more nodes converge'):
            col = kink(z)
            f = lambda hh: np.interp(hh, z, col)
            p0, d0 = np.array([0, 0, R]), np.array([0, 0, 1.0])
            err = {
                nq: abs(
                    simpson_prefix(np.column_stack([np.ones(n), col]), nq)[1] - ref_integral(f, p0, d0, 0.0, 80.0, z)
                )
                for nq in (12, 48)
            }
            assert err[48] < err[12] / 4


# ------------------------------------------------------------------------------------------------ (g)


class Raw:
    made = []

    def __init__(self, code, symbol, options, **kw):
        self.code, self.symbol, self.options, self.attributes = code, symbol, options, {}
        Raw.made.append(self)

    def compile(self):
        pass


def capture_kernel(step, W=15):
    from sosfuse2.kernel_config import resolve_config

    runtime = SimpleNamespace(
        getDevice=lambda: 0,
        getDeviceProperties=lambda d: dict(sharedMemPerBlockOptin=1 << 30, sharedMemPerBlock=1 << 30),
    )
    k = fused.Kernel(flags=resolve_config({'STRATO'}, W=W).flags, backend='numpy', L=16, **density.kernel_kwargs(step))
    k.xp = SimpleNamespace(RawKernel=Raw, cuda=SimpleNamespace(runtime=runtime))
    k.device = lambda: ('capture', {'stage1_defaults': ()}, 'test')
    return k


def test_g_one_key_rule_and_profile_option(subtests, monkeypatch):
    for name in (
        'SOSFUSE2_KSPLIT',
        'SOSFUSE2_KSPLIT_VERIFY',
        'SOSFUSE2_GROUND_SH_L',
        'SOSFUSE2_FMAD',
        'SOSFUSE2_FASTMATH',
    ):
        monkeypatch.setenv(name, '0')
    for W in (1, 15):
        for step, n in STEPS:
            with subtests.test(W=W, n=n):
                k = capture_kernel(step, W)
                raw = k.raw_kernel(W, 0, 0, 0)
                flagged = [o for o in raw.options if o.startswith('-DPROFILE_N')]
                assert flagged == ([] if n == 161 else [f'-DPROFILE_N={n}'])
                base = (tuple(k.config(W).defines), W, W, 0, 0, 0, ('L', 16))
                assert list(k.raw) == [base if n == 161 else base + (('PROFILE_N', n),)]  # 161: the historical key
                assert set(k.attributes) == {
                    f'{"+".join(k.config(W).defines) or "default"}:{W}_first0_phase0_boundary0_tile{W}'
                    + k.attribute_suffix(),
                    str(W),
                }
                k.raw_kernel(W, 1, 0, 0, ray_limit=True)
                project = k.fl_project_kernel(W, 0)
                assert project.options == k.raw_kernel(W, 1, 0, 0, ray_limit=True).options
                if W == 15:
                    vy, vp = k.vprod_kernels(15, 0, 0)
                    sp = k.slim_proj_kernel(15, 0)
                    assert all(
                        o.options[-1] == (f'-DPROFILE_N={n}' if n != 161 else o.options[-1]) for o in (vy, vp, sp)
                    )
                    keys = [
                        key
                        for key in k.raw
                        if isinstance(key, tuple)
                        and key[-1] in ('sos_vprod', 'sos_vprod_y', 'sos_slim_proj', 'fl_project')
                    ]
                    assert len(keys) == 4 and all((('PROFILE_N', n) in key) == (n != 161) for key in keys)


def test_g_shared_raw_dict_161_641_161(monkeypatch):
    """At CPU level: one compiled (raw, attributes) pair shared like the persistent service."""
    for name in ('SOSFUSE2_KSPLIT', 'SOSFUSE2_FMAD', 'SOSFUSE2_FASTMATH'):
        monkeypatch.setenv(name, '0')
    shared = ({}, {})
    a, b, c = capture_kernel(0.5), capture_kernel(0.125), capture_kernel(0.5)
    for k in (a, b, c):
        k.raw, k.attributes = shared
    ra = a.raw_kernel(15, 0, 0, 0)
    count = len(Raw.made)
    rb = b.raw_kernel(15, 0, 0, 0)
    assert len(Raw.made) == count + 1 and rb is not ra and '-DPROFILE_N=641' in rb.options
    rc = c.raw_kernel(15, 0, 0, 0)
    assert rc is ra and len(Raw.made) == count + 1  # memo hit, nothing recompiled
    assert len(shared[0]) == 2 and any(key.endswith('_profileN641') for key in shared[1])


def minimal_constants(table, mask=2):
    z = np.zeros(2, 'f4')
    c = {
        name: z
        for name in ('h', 'chi', 'hr', 'ha', 'ext', 'asca', 'sun_h', 'sun_col', 'gm', 'goff', 'gdir', 'gval', 'u', 'wu')
    }
    c.update(
        W=15,
        density_optics=dict(
            table=table, mask=mask, scale=1.0, meta=dict(profile_step_km=density.STEP_OF_N[len(table)])
        ),
    )
    return c


def test_g_consumption_checks_without_setup():
    """Fix #6: a restored/persistent kernel never runs setup(); upload and launch still check compiled == optics == table N."""
    s = SimpleNamespace()
    table = np.column_stack([np.ones(641), kink(density.grid(641)), density.no2_profile(density.grid(641))])
    k = fused.Kernel(backend='numpy')
    k.constants[(id(s), 4)] = minimal_constants(table)
    with pytest.raises(RuntimeError, match='E_PROFILE_N_MISMATCH: setup'):  # constants_device reads setup() first
        k.constants_device(s, 4)
    with pytest.raises(RuntimeError, match='E_PROFILE_N_MISMATCH: upload'):
        k.check_profile(k.constants[(id(s), 4)]['density_optics'], 'upload', 641)
    ok = fused.Kernel(backend='numpy', profile_step_km=0.125)
    ok.constants[(id(s), 4)] = minimal_constants(table)
    dc = ok.constants_device(s, 4)
    assert dc['profile_table'].shape == (641, 3) and ok.profile_consumed['uploaded_n'] == 641
    assert (
        ok.profile_consumed['table_f4_sha256']
        == hashlib.sha256(np.ascontiguousarray(table, 'f4').tobytes()).hexdigest()
    )
    with pytest.raises(RuntimeError, match='E_PROFILE_N_MISMATCH: launch'):
        ok.check_profile(ok.constants[(id(s), 4)]['density_optics'], 'launch', 321)
    lying = dict(table=table, mask=2, meta=dict(profile_step_km=0.25))  # declared step disagrees with the table
    with pytest.raises(RuntimeError, match='E_PROFILE_N_MISMATCH'):
        ok.check_profile(lying, 'setup')


def test_g_promotion_payload_restore_carry_step():
    from sosfuse2 import pipeline_execution as pe
    from unittest import mock

    calls = []

    class Recorder:
        def __init__(self, threads, height, g, backend, block_size, flags=None, L=32, **kw):
            self.height, self.g, self.block_size, self.flags, self.L = height, g, block_size, flags, L
            self.profile_step_km = kw.get('profile_step_km', 0.5)
            self.xp = np
            self.raw = {}
            self.attributes = {}
            self.constants = {}
            calls.append(kw)

        def sync(self):
            pass

    for step in (0.5, 0.125):
        old = SimpleNamespace(
            height=1.2, g=0.8, flags=('STRATO',), block_size=256, L=16, constants={}, profile_step_km=step
        )
        s = SimpleNamespace(kernel=old, W=15, ext=0, rf=0, af=0, asca=0, geo=SimpleNamespace(ns=1, h=0, chi=0))
        geo = SimpleNamespace(s=None, bundle=None, ground_bundle=None)
        calls.clear()
        with mock.patch.object(fused, 'Kernel', Recorder), mock.patch.object(pe, 'promote_bundle', lambda b, xp: b):
            pe.promote_solver(s, geo, 'cupy', 256, 2)
            payload = pe.device_payload(s, geo)
            restored, _ = pe.restore_device_payload(payload, ({}, {}))
        assert s.kernel.profile_step_km == payload['profile_step_km'] == restored.kernel.profile_step_km == step
        assert calls == (
            [{}, {}] if step == 0.5 else [dict(profile_step_km=step)] * 2
        )  # 0.5: the historical constructor call


def test_g_precompile_jobs_and_gate_keys(monkeypatch, tmp_path):
    from sosfuse2 import gate, precompile

    jobs = []

    class Pool:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def map(self, function, scheduled):
            jobs.extend(scheduled)
            return []

    monkeypatch.setattr(precompile, 'ProcessPoolExecutor', Pool)
    monkeypatch.setenv('SOSFUSE2_KERNEL_FLAGS', '')
    for argv, tail in (([], (15, 0.5)), (['--profile-step-km', '0.125', '--W', '1'], (1, 0.125))):
        jobs.clear()
        monkeypatch.setattr('sys.argv', ['precompile', '--cache', str(tmp_path), '--ray-limit', 'off', *argv])
        precompile.main()
        assert (
            jobs
            and all(j[-2:] == tail for j in jobs)
            and jobs[-1][0] in ('strato_lut_build',) + (() if tail[0] == 1 else (('ksplit', 1, 0),))
        )
        assert (tail[0] == 1) == (not any(isinstance(j[0], tuple) and j[0][0] == 'ksplit' for j in jobs))
    meta = dict(device={}, config=dict(flags=['STRATO'], stage1=[], async_=False, L=16, block=256))
    plain = gate.toolkey(meta)
    assert gate.toolkey(dict(meta, config=dict(meta['config'], profile_n=161))) == plain  # 0.5 toolkeys unchanged
    assert gate.toolkey(dict(meta, config=dict(meta['config'], profile_n=641)))[0] != plain[0]
    F = gate.load_fixtures()
    want = gate.expected_config(F, 'FX-C3')
    assert want['profile_n'] == 161
    with pytest.raises(gate.GateError, match='profile_n requested 161 measured 641'):
        gate.check_config(F, 'FX-C3', dict(backend='numpy', config=dict(want, profile_n=641)))


# ------------------------------------------------------------------------------------------------ (h)


def test_h_identity_separates_tables_and_policies():
    z = density.grid(321)
    A = kink(z)
    B = np.maximum(0, 1 - np.abs(z - 24.3) / 7.0)
    B *= np.trapezoid(A, z) / np.trapezoid(B, z)
    build = lambda **kw: density.build(external(321, **kw))
    a = build(q=A)
    variants = dict(
        shape=build(q=B),
        mask=build(q=A, mask=3),
        scale=build(q=A, scale=1.5),
        normalized=build(q=A, normalize_ozone=True),
        wavelength=build(q=A * 1.0001),
        grid=density.build(external(641, q=np.interp(density.grid(641), z, A))),
        asset_05=density.build(state()),
        asset_025=density.build(state(profile_step_km=0.25)),
    )
    ids = {k: density.profile_identity(v)['sha256'] for k, v in variants.items()}
    assert (
        density.profile_identity(build(q=A))['sha256'] == density.profile_identity(a)['sha256']
    )  # repeat: same identity
    assert len(set(ids.values()) | {density.profile_identity(a)['sha256']}) == len(ids) + 1
    assert np.trapezoid(variants['shape']['table'][:, 1], z) == pytest.approx(
        np.trapezoid(a['table'][:, 1], z), rel=1e-14
    )
    assert column_table.table_key(column_table.molecular(a, 1)) != column_table.table_key(
        column_table.molecular(variants['shape'], 1)
    )
    from sosfuse2.stratosphere import _geometry_key

    sun = SimpleNamespace(h=np.zeros(2), x=np.zeros(2), col=np.zeros((2, 2, 3)))
    s = lambda o: SimpleNamespace(
        geo=SimpleNamespace(h=np.zeros(2), chi=np.zeros(2), sun=sun, cfg={}), density_optics=o, fuse_state={}
    )
    prof = dict(center=19.0, sigma=3.0, norm=1.0)
    assert _geometry_key(s(a), None, prof) != _geometry_key(s(variants['shape']), None, prof)


def test_h_caches_A_B_A_cold_warm(tmp_path, env, monkeypatch):
    """Same N/mask/integral, different shape; host endpoint cache, column disk cache and device memo."""
    z = density.grid(321)
    qa = kink(z)
    qb = np.maximum(0, 1 - np.abs(z - 24.3) / 7.0)
    qb *= np.trapezoid(qa, z) / np.trapezoid(qb, z)
    sa = external(321, q=qa)
    sb = external(321, q=qb)
    s, scope, _ = create(tmp_path / 'solver', sa)
    old = scope.b.sh.old
    tg = geometry.TransientGeometry(s, scope)
    cells = [tg.cell(i) for i in (0, 9)]
    starts = np.stack([np.asarray(g['p'], 'f8').reshape(-1, 3)[0] for g in cells])  # one start per cell
    dirs = np.concatenate([g['d'] for g in cells])
    off = np.r_[0, np.cumsum([len(g['d']) for g in cells])]
    host = dict(starts=starts, cell_off=off, dirs=dirs, end_col=np.zeros((len(dirs), 5), 'f4'))
    oa, ob = density.build(sa), density.build(sb)
    for o in (oa, ob):
        o['old'] = old
    monkeypatch.setenv('SOSFUSE2_HOST_CACHE', '1')
    seq = [
        (k, host_cache.endpoint_columns(s, host, o, tmp_path / 'hc', 'sig'))
        for k, o in (('a', oa), ('b', ob), ('a', oa), ('b', ob))
    ]
    assert (within_tol(seq[0][1], seq[2][1])) and (within_tol(seq[1][1], seq[3][1]))
    assert not np.array_equal(seq[0][1], seq[1][1])  # (diagnostic only; equality would not fail the test)
    assert len(list((tmp_path / 'hc' / 'host_cache').glob('*.hcache'))) == 2
    monkeypatch.setenv('SOSFUSE2_COLUMN_CACHE', str(tmp_path / 'cc'))
    k = s.kernel
    results = []
    for o in (oa, ob, oa):
        bundle = SimpleNamespace(arrays=dict(host), rays=len(dirs))  # fresh Bundle per state; cache files shared
        results.append(k.density_end_columns(bundle, dict(density_optics=o)))
        assert list(bundle.density_columns) == [density.profile_identity(o)['sha256']]
    monkeypatch.delenv('SOSFUSE2_COLUMN_CACHE')
    for o, r in zip((oa, ob, oa), results):
        fresh = k.density_end_columns(
            SimpleNamespace(arrays=dict(host), rays=len(dirs)), dict(density_optics=o)
        )  # no disk cache
        assert within_tol(r, fresh)
    assert len(list((tmp_path / 'cc').glob('*.npy'))) == 2


def test_h_outputs_independent_of_N(tmp_path, env):
    """Fix #10: (h, chi) values/order/dtype, cell order, source slots and the COMMON geometry identity do not depend on N."""
    from sosfuse2.geometry_contract import identity
    from sosfuse2.stratosphere import source_size

    made = {step: create(tmp_path / str(step), state(profile_step_km=step)) for step in (0.5, 0.125)}
    (a, sa, _), (b, sb, _) = made[0.5], made[0.125]
    assert (a.kernel.profile_n, b.kernel.profile_n) == (161, 641)
    for name in ('h', 'chi'):
        x, y = getattr(a.geo, name), getattr(b.geo, name)
        assert (x.dtype.str, x.shape, x.tobytes()) == (y.dtype.str, y.shape, y.tobytes())
    assert (a.geo.ns, source_size(a), a.kernel.K, identity(a)) == (b.geo.ns, source_size(b), b.kernel.K, identity(b))
    ta, tb = geometry.TransientGeometry(a, sa), geometry.TransientGeometry(b, sb)
    for i in (0, 13, a.geo.ns - 1):
        ga, gb = ta.cell(i), tb.cell(i)
        assert all(
            (within_tol(np.asarray(ga[k]), np.asarray(gb[k]))) for k in ('p', 'd', 'edges', 'weight', 'end_index')
        )
    src = re.sub(
        r'\s+', '', (HERE / 'run_gpu_state_v3.py').read_text()
    )  # whitespace-free: the call may be formatted over lines
    assert (
        src.count("np.savez(prepare/'moment.npz',moment=np.concatenate(moments,axis=-1),h=s.geo.h,chi=s.geo.chi") == 1
    )


# ------------------------------------------------------------------------------------------------ review


def review_bad_optics():
    """Declared 0.5 km (Kernel N 161) with a 321-row table whose ozone row j is min(j*.5, 80)."""
    table = np.column_stack([np.ones(321), np.minimum(np.arange(321) * 0.5, 80.0)])
    return dict(table=table, mask=2, scale=1.0, meta=dict(density.DEFAULTS, asset_sha256=None, profile_step_km=0.5))


def test_r1_declared_step_mismatch_rejected_on_every_path(subtests):
    """Fix #1: cached setup (NumPy transport, geometry, table_fast read it), reference.profile_at, profiles, identity,
    payload restore / persistent install: all refuse Kernel 161 + meta 0.5 + table 321x2 (no silent 0.25 reinterpretation)."""
    from sosfuse2 import pipeline_execution as pe

    bad = review_bad_optics()
    s = SimpleNamespace()
    h = np.array([20.0], 'f4')
    calls = dict(
        cached_setup=lambda: (
            k.constants.__setitem__((id(s), 4), minimal_constants(bad['table'])),
            k.constants[(id(s), 4)].update(density_optics=bad),
            k.setup(s, 4),
        ),
        reference_profile_at=lambda: reference.profile_at(h, dict(height=F(1.2), density_optics=bad)),
        reference_transport_sample=lambda: reference.SimpsonPrefix(
            dict(p=np.array([[0, 0, R]]), d=np.array([[0, 0, 1.0]])),
            dict(W=1, height=F(1.2), ext=np.zeros((5, 1), 'f4'), density_optics=bad),
        ).beta((np.array([20.0], 'f4'), np.zeros(1, 'f4'))),
        profiles=lambda: density.profiles(np.array([20.0]), bad),
        profile_identity=lambda: density.profile_identity(bad),
        install=lambda: pe.install_constants(k, s, {4: dict(minimal_constants(bad['table']), density_optics=bad)}),
        explicit_N_321_kernel=lambda: fused.Kernel(backend='numpy', profile_step_km=0.25).check_profile(bad, 'setup'),
    )
    for name, call in calls.items():
        k = fused.Kernel(backend='numpy')
        with subtests.test(path=name):
            with pytest.raises(RuntimeError, match='E_PROFILE_N_MISMATCH'):
                call()


def test_r2_rebound_table_rejected_as_stale():
    """A->B rebinding (same N, same s/nq) after the upload is E_PROFILE_STALE on reuse and at launch."""
    z = density.grid(161)
    s = SimpleNamespace()
    a = density.build(
        density.with_profile(
            state(), air_surface_number_density_m3=2.5469e25, z_km=z, table=np.column_stack([np.ones(161), z]), mask=2
        )
    )
    optics = dict(a)  # attach-style: consumers hold this dict
    k = fused.Kernel(backend='numpy')
    k.constants[(id(s), 4)] = dict(minimal_constants(a['table']), density_optics=optics)
    dc = k.constants_device(s, 4)
    assert k.constants_device(s, 4) is dc  # unchanged optics: the upload is reused
    optics['table'] = np.column_stack([np.ones(161), 2 * z, density.no2_profile(z)])  # rebinding forbidden after attach
    with pytest.raises(RuntimeError, match='E_PROFILE_STALE: device_constants'):
        k.constants_device(s, 4)
    with pytest.raises(RuntimeError, match='E_PROFILE_STALE: launch'):
        k.check_upload(k.constants[(id(s), 4)], dc, 'launch')
    optics['table'] = a['table']
    optics['mask'] = 3  # a rebound mask is stale as well
    with pytest.raises(RuntimeError, match='E_PROFILE_STALE'):
        k.check_upload(k.constants[(id(s), 4)], dict(dc), 'launch')  # launch-local dict copies share the buffer key


def test_r5_numpy_service_event_records_consumed_profile():
    """Fix #5: a NumPy persistent service consumed the host table: mask, identity and N recorded, uploaded_n None."""
    from sosfuse2 import pipeline_execution as pe

    o = density.build(external(321, mask=2))
    s = SimpleNamespace()
    k = fused.Kernel(backend='numpy', profile_step_km=0.25)
    pe.install_constants(k, s, {4: dict(minimal_constants(o['table']), density_optics=o)})
    ev = pe.profile_service_event(k)
    assert (ev['where'], ev['compiled_n'], ev['optics_n'], ev['uploaded_n'], ev['mask']) == (
        'service',
        321,
        321,
        None,
        2,
    )
    assert ev['profile_identity'] == density.profile_identity(o)['sha256'] and ev['origin'] == 'external'
    none = pe.profile_service_event(fused.Kernel(backend='numpy'))
    assert (none['mask'], none['uploaded_n'], none['profile_identity']) == (0, None, None)
