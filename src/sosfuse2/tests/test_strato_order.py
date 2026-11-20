#!/usr/bin/env python3
"""Exact new-phase C++/NumPy arithmetic and 1-vs-15-lane transport evidence.

Clang executes mie_phase from kernel.cu as a host helper with CUDA builtins
stubbed. No GPU transport is executed. Per-wave NumPy compares first, warm,
SH_AUX and boundary operations including tensor/SH/ground projections.
"""

from sosfuse2.foundation import paths
import argparse, ctypes, json, subprocess, sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
from sosfuse2.backends import numpy_reference
from sosfuse2.atmosphere.stratosphere import source_size
from sosfuse2.backends.flags import resolve_config
from sosfuse2.checks import testkit
from sosfuse2.solve.ray_geometry import create, TransientGeometry
from sosfuse2.solve.source_solver import Solver, sh_aux
from sosfuse2.asset_tools.strato_states import build
from sosfuse2.tests.test_kernel_static import STUB, defines
from sosfuse2.tests.test_strato_precision import numpy_lut
from sosfuse2.tests.test_tropo_mie import SYNTAX_STUB_EXTRA

from sosfuse2.checks.numerical_checks import within_tol


def run(out):
    testkit.require_tool(
        'clang++'
    )  # host build of the CUDA phase helper; SkipTest under pytest (D15), a readable stop in the CLI
    out.mkdir(parents=True, exist_ok=True)
    kernel = (paths.PKG / 'backends/kernel.cu').resolve()
    cpp = out / 'phase_host.cpp'
    lib = out / ('phase_host.dylib' if sys.platform == 'darwin' else 'phase_host.so')
    cpp.write_text(
        STUB
        + SYNTAX_STUB_EXTRA
        + '\nDim blockIdx{},threadIdx{},blockDim{};extern "C" { double shared[50000]; }\n'
        + f'#include "{kernel}"\n'
        + 'extern "C" float phase_c(float mu,int w,const float*chi){return mie_phase(mu,w,chi);}\n'
        + 'extern "C" int phase_bins(){return STRATO_LUT_BINS;}\n'
        + '''extern "C" void build_phase_lut(float* chi) {
    blockDim.x = 1;
    for (blockIdx.x = 0; blockIdx.x < (STRATO_LUT_BINS + 1) * FW; ++blockIdx.x)
        strato_lut_build(chi);
}
'''
        + 'extern "C" int sample_size(){return sizeof(Sample);}\n'
    )
    cmd = [
        'clang++',
        '-std=c++17',
        '-O2',
        '-ffp-contract=off',
        *defines(15, resolve_config(('STRATO',), W=15).defines, spec=(1, 0, 0)),
        '-dynamiclib' if sys.platform == 'darwin' else '-shared',
        '-fPIC',
        str(cpp),
        '-o',
        str(lib),
        '-lm',
    ]
    subprocess.run(cmd, check=True)
    native = ctypes.CDLL(str(lib))
    native.build_phase_lut.argtypes = [ctypes.c_void_p]
    native.build_phase_lut.restype = None
    native.phase_c.argtypes = [ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
    native.phase_c.restype = ctypes.c_float
    wl = 360 + (np.arange(15) + 0.5) * 470 / 15
    state = build()['states'][5]
    s, scope, _ = create(out / 'geometry', 'selftest', wl, state=state)
    sv = Solver(s, TransientGeometry(s, scope))
    g = sv.geometry.cell(12)
    c = s.kernel.setup(s, g['nq'])
    chi = c['strato_chi']
    mu = np.linspace(-1, 1, 133, dtype='f4')
    bins = native.phase_bins()
    lut = np.empty((chi.shape[0] + bins + 1, 15), dtype='f4')
    lut[: chi.shape[0]] = chi
    native.build_phase_lut(lut.ctypes.data)
    actual = np.array([[native.phase_c(float(v), w, lut.ctypes.data) for w in range(15)] for v in mu], 'f4')
    expected = numpy_lut(mu, chi, bins)
    rows = dict(
        phase_native_within_tol=(within_tol(actual, expected)),
        phase_max_relative=float(np.max(abs(actual.astype('f8') - expected) / np.maximum(abs(expected), 1e-30))),
        phase_samples=actual.size,
        STRATO_sample_bytes=int(native.sample_size()),
        expected_sample_bytes=304 + 40 * 15 + 8 + 16 * 15,
    )
    # Seed every coefficient/component with distinct values: invalid traces
    # intentionally exercise linear interpolation fallback without changing rules.
    rng = np.random.default_rng(20261001)
    field = rng.normal(0, 1e-5, (s.geo.ns, source_size(s), 15)).astype('f4')
    field[:, :3] = abs(field[:, :3])
    boundary = rng.random((len(s.geo.chi), 15)) * 1e-4
    cases = []
    for first, bd, aux in (
        (True, None, False),
        (False, None, False),
        (False, boundary, False),
        (False, boundary, True),
    ):
        full, Y = numpy_reference.transport(
            g, c, field, first, boundary=bd, return_basis=True, sh_source=sh_aux(field, L=s.kernel.L) if aux else None
        )
        m, f = numpy_reference.project(g, full, c, Y)
        lanes = []
        lm = []
        lf = []
        for w in range(15):
            one = dict(c, W=1)
            for name in ('ext', 'asca', 'gm', 'gv', 'gval', 'strato_ext', 'strato_sca', 'strato_chi'):
                one[name] = np.ascontiguousarray(c[name][..., w : w + 1])
            src = np.ascontiguousarray(field[..., w : w + 1])
            b = None if bd is None else bd[:, w : w + 1]
            value, y = numpy_reference.transport(
                g, one, src, first, boundary=b, return_basis=True, sh_source=sh_aux(src, L=s.kernel.L) if aux else None
            )
            mm, ff = numpy_reference.project(g, value, one, y)
            lanes.append(value)
            lm.append(mm)
            lf.append(ff)
        per = np.concatenate(lanes, axis=1)
        mom = np.concatenate(lm, axis=1)
        flux = np.concatenate(lf)
        cases.append(
            dict(
                first=first,
                boundary=bd is not None,
                SH_AUX=aux,
                ray_within_tol=(within_tol(full, per)),
                moment_within_tol=(within_tol(m, mom)),
                moment_max_absolute=float(np.max(abs(m - mom))),
                moment_l2_relative=float(np.linalg.norm(m - mom) / max(np.linalg.norm(mom), 1e-30)),
                moment_absolute_pass_=bool(np.max(abs(m - mom)) <= 1e-12),
                ground_within_tol=(within_tol(f, flux)),
            )
        )
    report = dict(
        **rows,
        cases=cases,
        pass_=rows['phase_native_within_tol']
        and rows['STRATO_sample_bytes'] == rows['expected_sample_bytes']
        and all(all(r[k] for k in ('ray_within_tol', 'moment_absolute_pass_', 'ground_within_tol')) for r in cases),
        scope='actual CUDA phase helper compiled as host C++; per-wave NumPy operator; not CUDA/NVRTC transport parity',
    )
    (out / 'order.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    raise SystemExit(0 if run(a.out)['pass_'] else 1)
