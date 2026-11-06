#!/usr/bin/env python3
"""Native CPU tests of the actual old/new CUDA Mie function bodies.

Compiles only the extracted phase functions, not the transport kernel. No GPU
claim is made. Requires clang++ and NumPy, works on macOS/Linux. Values are
normal FP32 in the supplied optical fixture; CPU FTZ/CUDA expf are not tested.
"""

import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

try:
    from sosfuse2.numerical_checks import assert_equivalent, within_tol
except ImportError:  # standalone research harness
    from numerical_checks import assert_equivalent, within_tol

F = np.float32


def extract_function(text, name):
    start = text.index('__device__ float ' + name + '(')
    opening = text.index('{', start)
    depth = 1
    i = opening + 1
    while depth:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    return text[start:i]


def numpy_phase(mu, coeff):
    mu = np.asarray(mu, 'f4')[:, None]
    coeff = np.asarray(coeff, 'f4')
    p0 = np.ones_like(mu)
    p1 = mu
    value = p0 * coeff[0]
    value = value + p1 * coeff[1]
    for l in range(2, 33):
        num = F(2 * l - 1) * mu * p1 - F(l - 1) * p0
        pn = num / F(l)
        value = value + pn * coeff[l]
        p0, p1 = p1, pn
    return value / F(4 * math.pi)


def numpy_lut(mu, coeff, bins):
    nodes = numpy_phase(np.linspace(-1, 1, bins + 1, dtype='f4'), coeff)
    x = (np.asarray(mu, 'f4') + F(1)) * F(bins // 2)
    index = np.minimum(x.astype('i4'), bins - 1)
    fraction = x - index.astype('f4')
    return nodes[index] + fraction[:, None] * (nodes[index + 1] - nodes[index])


class NativePhase:
    def __init__(self, root, W, variant, out):
        compiler = shutil.which('clang++')
        if not compiler:
            raise RuntimeError('E_CLANG: clang++ required')
        self.W = W
        self.variant = variant
        self.bins = 16384 if variant == 'lut' else 0
        base = (root / 'assets/kernel_C2_strato_base.cu').read_text()
        current = (root / 'kernel.cu').read_text()
        if variant == 'base':
            body = extract_function(base, 'mie_phase')
            flags = []
        else:
            start = current.index('#ifdef STRATO\n#define STRATO_LUT_BINS')
            end = current.index('__device__ void kah(', start)
            body = current[start:end]
            flags = ['-DSTRATO']
        function = 'strato_phase_node' if variant == 'exact' else 'mie_phase'
        source = (
            '''#include <cmath>
#include <algorithm>
#define __device__
#define __global__
#define __constant__
#define PI 3.14159265358979323846
using std::min;using std::max;
struct Dim {int x,y,z;};Dim blockIdx,blockDim,threadIdx;
float clip(float x,float lo,float hi){return fminf(hi,fmaxf(lo,x));}
'''
            + body
            + '''
extern "C" void phase_many(const float*u,const float*c,int n,float*out){
 for(int i=0;i<n;i++)for(int w=0;w<FW;w++)out[i*FW+w]=PHASE_FUNCTION(u[i],w,c);
}
#ifdef STRATO
extern "C" void phase_nodes(float*c){
 for(int i=0;i<=STRATO_LUT_BINS;i++)for(int w=0;w<FW;w++){
  float u=-1.f+(float)i*(2.f/(float)STRATO_LUT_BINS);
  c[(33+i)*FW+w]=strato_phase_node(u,w,c);
 }
}
#endif
'''
        )
        source = source.replace('PHASE_FUNCTION', function)
        cpp = out / f'phase_W{W}_{variant}.cpp'
        cpp.write_text(source)
        library = cpp.with_suffix('.dylib' if sys.platform == 'darwin' else '.so')
        command = [
            compiler,
            '-std=c++17',
            '-O2',
            '-ffp-contract=off',
            '-fno-fast-math',
            '-fPIC',
            '-dynamiclib' if sys.platform == 'darwin' else '-shared',
            f'-DFW={W}',
            *flags,
            str(cpp),
            '-o',
            str(library),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        self.library = ctypes.CDLL(str(library.resolve()))
        array = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
        self.library.phase_many.argtypes = [array, array, ctypes.c_int, array]
        self.library.phase_many.restype = None
        if self.bins:
            self.library.phase_nodes.argtypes = [array]
            self.library.phase_nodes.restype = None
        self.command = command
        self.source_sha = hashlib.sha256(source.encode()).hexdigest()

    def evaluate(self, mu, coeff):
        mu = np.ascontiguousarray(mu, 'f4')
        coeff = np.ascontiguousarray(coeff, 'f4')
        if coeff.shape != (33, self.W):
            raise ValueError('E_NATIVE_COEFF')
        if self.bins:
            data = np.empty((33 + self.bins + 1) * self.W, 'f4')
            data[: 33 * self.W] = coeff.ravel()
            self.library.phase_nodes(data)
        else:
            data = coeff.ravel()
        out = np.empty((len(mu), self.W), 'f4')
        self.library.phase_many(mu, data, len(mu), out)
        return out


def metrics(got, reference):
    a = np.asarray(got, 'f8')
    b = np.asarray(reference, 'f8')
    err = np.abs(a - b)
    nonzero = b != 0
    relative = np.zeros_like(b)
    relative[nonzero] = err[nonzero] / np.abs(b[nonzero])
    relative[(~nonzero) & (err != 0)] = np.inf
    return dict(
        max_absolute=float(err.max()),
        max_relative=float(relative.max()),
        l2_relative=float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)),
        max_relative_by_wave=relative.max(axis=0).tolist(),
        values_within_tol=bool((within_tol(np.asarray(got), np.asarray(reference)))),
    )


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).parent
    asset = root / 'assets/strato_sulfate_rg0.08_sg1.6_n1.43.json'
    data = json.loads(asset.read_text())
    wl = np.array(data['wavelengths_nm'])
    table = np.array([row['chi_l'] for row in data['rows']])
    rng = np.random.default_rng(20261002)
    mu = np.unique(
        np.r_[
            np.linspace(-1, 1, 65537, dtype='f4'),
            rng.uniform(-1, 1, 100000).astype('f4'),
            F(-1),
            np.nextafter(F(-1), F(0)),
            F(0),
            np.nextafter(F(1), F(0)),
            F(1),
        ]
    )
    records = []
    native_specs = []
    exact_values = 0
    q, weights = np.polynomial.legendre.leggauss(256)
    qf = q.astype('f4')
    basis = np.polynomial.legendre.legvander(qf.astype('f8'), 32)
    for W in (1, 15):
        wanted = np.array([550.0]) if W == 1 else 360 + (np.arange(15) + 0.5) * 470 / 15
        coeff = np.stack([np.interp(wanted, wl, table[:, l]) for l in range(33)]).astype('f4')
        functions = {name: NativePhase(root, W, name, out) for name in ('base', 'exact', 'lut')}
        base = functions['base'].evaluate(mu, coeff)
        assert within_tol(base, numpy_phase(mu, coeff)), 'native/Python fixture disagreement'
        old_q = functions['base'].evaluate(qf, coeff)
        old_mom = 2 * math.pi * basis.T @ (weights[:, None] * old_q.astype('f8'))
        truth = np.polynomial.legendre.legval(mu.astype('f8'), coeff.astype('f8')).T / (4 * math.pi)
        # Absolute interpolation bound for the real polynomial before FP32 rounding.
        degree = np.arange(33, dtype='f8')
        second = (degree - 1) * degree * (degree + 1) * (degree + 2) / 8
        curvature = np.sum(abs(coeff.astype('f8')) * second[:, None], axis=0) / (4 * math.pi)
        for name, fn in functions.items():
            got = fn.evaluate(mu, coeff)
            expected = numpy_lut(mu, coeff, fn.bins) if fn.bins else numpy_phase(mu, coeff)
            assert within_tol(got, expected), (W, name, 'native/NumPy mismatch')
            if name in ('base', 'exact'):
                assert within_tol(got, base)
                exact_values += got.size
            got_q = fn.evaluate(qf, coeff)
            got_mom = 2 * math.pi * basis.T @ (weights[:, None] * got_q.astype('f8'))
            row = dict(
                W=W,
                variant=name,
                wavelengths_nm=wanted.tolist(),
                mu_count=len(mu),
                vs_old_fp32=metrics(got, base),
                vs_float64_polynomial=metrics(got, truth),
                phase_legendre_moment_l2=float(np.linalg.norm(got_mom - old_mom) / np.linalg.norm(old_mom)),
                phase_min=float(got.min()),
                phase_max=float(got.max()),
                native_vs_numpy_within_tol=True,
                meaning='Phase function and local angular coefficient tests ONLY; not transported field moments or final radiance.',
            )
            if fn.bins:
                row.update(
                    intervals=fn.bins,
                    device_allocation_bytes=(33 + fn.bins + 1) * W * 4,
                    analytic_interpolation_absolute_bound_by_wave=((2 / fn.bins) ** 2 / 8 * curvature).tolist(),
                    bound_scope='Exact polynomial linear interpolation only. FP32 node/coordinate/lookup errors are additional.',
                )
            records.append(row)
            native_specs.append(dict(W=W, variant=name, command=fn.command, source_sha256=fn.source_sha))
        # Local phase tolerances do not control a near-cancelling downstream sum.
        approx = functions['lut'].evaluate(mu, coeff)
        error = approx.astype('f8') - base.astype('f8')
        i, w = np.unravel_index(np.argmax(abs(error)), error.shape)
        epsilon = max(abs(float(base[i, w])) * 1e-10, 1e-30)
        cancellation = dict(
            W=W,
            mu=float(mu[i]),
            wave_index=int(w),
            constructed_reference_difference=epsilon,
            candidate_difference=epsilon + float(error[i, w]),
            constructed_relative_error=abs(float(error[i, w])) / epsilon,
            scope='Synthetic A-B cancellation counterexample, NOT a measured atmospheric output.',
        )
        records.append(dict(cancellation_counterexample=cancellation))
    report = dict(
        pass_=True,
        exact_scalar_comparisons=exact_values,
        modified_vs_base_scalar_comparisons=exact_values // 2,
        asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
        records=records,
        native_builds=native_specs,
        scope='CPU-native extracted CUDA phase bodies, -O2 -ffp-contract=off -fno-fast-math. No CUDA execution, no FTZ assertion, no full-grid precision gate.',
    )
    (out / 'precision.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('NATIVE PHASE PASS', exact_values, 'exact scalar comparisons')
    for row in records:
        if row.get('W') == 15:
            print(row['variant'], row['vs_old_fp32']['max_relative'], row['vs_old_fp32']['l2_relative'])
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    run(a.out)
