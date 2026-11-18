"""Run the KSPLIT record writer and both readers on the same host buffer."""

import ctypes
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from sosfuse2.atmosphere.components import SINV_F
from sosfuse2.backends.flags import KSPLIT_VARIANTS
from sosfuse2.checks import testkit
from sosfuse2.foundation import paths
from sosfuse2.tests.ground_native import host_warp_stub
from sosfuse2.tests.test_kernel_static import defines

WAVES = 15
RTOL = 1e-3
HEIGHTS = np.array([0, 4, 9, 17, 30, 80], 'f4')
ANGLES = np.array([0, 0.67, 0.73, 1.38, 1.45, 3.15], 'f4')


def build_kernels(compiler, build):
    (build / 'cuda_stub.h').write_text(host_warp_stub())
    source = Path(__file__).with_name('ksplit_host.cpp')
    functions = {}
    for role in ('direct', 'record', 'slim', 'slimray'):
        flags = ('STRATO', 'SH_COMBINE', 'SH_L=16') + (() if role == 'direct' else KSPLIT_VARIANTS[role])
        if role == 'slimray':
            flags += ('LAUNCH_RAY_LIMIT',)
        library = build / f'{role}.so'
        cmd = [
            compiler,
            '-std=c++17',
            '-O2',
            '-ffp-contract=off',
            '-pthread',
            '-fPIC',
            '-fvisibility=hidden',
            '-dynamiclib' if platform.system() == 'Darwin' else '-shared',
            *defines(WAVES, flags, (0, 0, 1)),
            '-I',
            str(build),
            '-I',
            str(paths.KERNEL_CU.parent),
            str(source),
            '-o',
            str(library),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, f'{role}: {result.stderr}'
        run = ctypes.CDLL(str(library)).run_segment
        f64 = np.ctypeslib.ndpointer(dtype='f8', flags='C_CONTIGUOUS')
        f32 = np.ctypeslib.ndpointer(dtype='f4', flags='C_CONTIGUOUS')
        integer = ctypes.c_int
        run.argtypes = (
            [integer] * 4
            + [ctypes.c_double] * 2
            + [f64, f64, integer, f64, f64]
            + [integer, f32, integer, f32]
            + [f32] * 4
        )
        run.restype = None
        functions[role] = run
    return functions


def ray_geometry(angle, height, sign):
    radial = np.array([0.8 * np.sin(angle), 0.6 * np.sin(angle), np.cos(angle)])
    tangent = np.array([0.8 * np.cos(angle), 0.6 * np.cos(angle), -np.sin(angle)])
    direction = sign * (0.55 * radial + 0.75 * tangent) + np.sqrt(1 - 0.55**2 - 0.75**2) * np.array([-0.6, 0.8, 0])
    return (6371 + height) * radial, direction


def expected_columns(start, direction, a, t):
    """Integrate the six fixture profiles on each Simpson panel in double precision."""

    def profiles(x):
        height = np.linalg.norm(start + x[:, None] * direction, axis=1) - 6371
        ones = np.ones_like(height)
        return np.column_stack(
            (
                0.82 * ones,
                np.exp(-height / 5.5),
                0.55 * ones,
                0.23 * ones,
                (0.82 / 1.2) ** 2 * ones,
                np.exp(-0.5 * ((height - 12) / 7) ** 2) / 1.7,
            )
        )

    prev = np.r_[a, t[:-1]]
    return np.cumsum((t - prev)[:, None] / 6 * (profiles(prev) + 4 * profiles((prev + t) / 2) + profiles(t)), axis=0)


def check_close(actual, expected, context):
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=0, equal_nan=False, err_msg=context)


def check_interpolation(points, positions):
    height = np.linalg.norm(positions, axis=1) - 6371
    angle = np.arctan2(np.hypot(positions[:, 0], positions[:, 1]), positions[:, 2])
    ir = HEIGHTS.searchsorted(height, side='right') - 1
    j = ANGLES.searchsorted(angle, side='right') - 1
    ih = (height - HEIGHTS[ir]) / (HEIGHTS[ir + 1] - HEIGHTS[ir])
    ic = (angle - ANGLES[j]) / (ANGLES[j + 1] - ANGLES[j])
    weights = np.column_stack(((1 - ih) * (1 - ic), (1 - ih) * ic, ih * (1 - ic), ih * ic))
    base = ir * len(ANGLES) + j
    cells = base[:, None] + [0, 1, len(ANGLES), len(ANGLES) + 1]
    check_close(points[:, 16:20], weights, 'bilinear weights')
    np.testing.assert_array_equal(points[:, 20:24], cells)
    np.testing.assert_array_equal(points[:, 24], ir)
    check_close(points[:, 25], height, 'sample height')


def check_segment_pairs(kernels, nq):
    nodes, weights = np.polynomial.legendre.leggauss(nq)
    nodes, weights = (1 + nodes) / 2, weights / 2
    basis = np.eye(SINV_F, WAVES, dtype='f4')
    # A transparent wave exposes ds; each remaining wave mixes a different spectrum.
    coeff = np.random.default_rng(17).uniform(0.003, 0.07, (SINV_F, WAVES)).astype('f4')
    coeff[:, 0] = 0
    records = np.full((8, nq, SINV_F), np.nan, 'f4')
    expected = np.full(records.shape, np.nan)
    segments = [(1, 2, True, *ray_geometry(0.69, 6, 1)), (3, 6, False, *ray_geometry(1.435, 21, -1))]
    a, b = 1.25, 28.75

    def run(role, mode, segment, extinction):
        ray, seg, lower, start, direction = segment
        points = np.full((nq + 1, 27), np.nan, 'f4')
        att = np.full((nq + 1, WAVES), np.nan, 'f4')
        kernels[role](
            mode,
            ray,
            seg,
            lower,
            a,
            b,
            start,
            direction,
            nq,
            nodes,
            weights,
            len(HEIGHTS),
            HEIGHTS,
            len(ANGLES),
            ANGLES,
            extinction,
            records,
            points,
            att,
        )
        assert np.isnan(points[nq]).all(), f'{role} exported an absent node'
        if role == 'slimray' and nq % 2:
            check_close(att[nq], 0, 'absent odd node contributed to attenuation')
        return points[:nq], att[:nq]

    # Finish all writes before any consumer runs. Unused segments guard both sides of each block.
    recorded_points = []
    for segment in segments:
        ray, seg, lower, start, direction = segment
        points, _ = run('record', 1, segment, basis)
        recorded_points.append(points)
        t = a + (b - a) * nodes**2 if lower else b - (b - a) * (1 - nodes) ** 2
        check_interpolation(points, start + t[:, None] * direction)
        expected[seg - ray] = expected_columns(start, direction, a, t)
    active = np.isfinite(expected)
    np.testing.assert_array_equal(np.isnan(records), ~active, err_msg='record row address or stride')

    for segment, written in zip(segments, recorded_points, strict=True):
        ray, seg, lower, _, _ = segment
        check_close(records[seg - ray], expected[seg - ray], f'record columns, ray {ray}')
        ds = 2 * (b - a) * (nodes if lower else 1 - nodes) * weights
        full, direct = run('direct', 0, segment, coeff)
        check_close(direct, ds[:, None] * np.exp(-(expected[seg - ray] @ coeff)), 'mode 0 attenuation')
        for role in ('record', 'slim', 'slimray'):
            if role == 'record':
                points = written
            else:
                points, consumed = run(role, 2, segment, coeff)
                check_close(consumed, direct, f'{role} mode 2 vs mode 0, ray {ray}')
            check_close(points[:, :16], full[:, :16], f'{role} T')
            check_close(points[:, 16:20], full[:, 16:20], f'{role} sw')
            np.testing.assert_array_equal(points[:, 20:25], full[:, 20:25], err_msg=f'{role} cell/ir')
            check_close(points[:, 25], full[:, 25], f'{role} h')
            check_close(points[:, 26], full[:, 26], f'{role} ds')
        check_close(full[:, 26], ds, 'direct ds')


def test_ksplit_column_roundtrip(tmp_path):
    compiler = testkit.require_tool('clang++')
    # A divergent warp or an invalid record address must fail within the test budget.
    result = subprocess.run(
        [sys.executable, '-B', __file__, compiler, str(tmp_path)], capture_output=True, text=True, timeout=110
    )
    assert result.returncode == 0, f'KSPLIT host exit {result.returncode}\n{result.stdout}{result.stderr}'


if __name__ == '__main__':
    kernels = build_kernels(sys.argv[1], Path(sys.argv[2]))
    for nq in (1, 16, 19, 33):
        print(f'nq={nq}', flush=True)
        check_segment_pairs(kernels, nq)
