"""GPU projection checks: tails, empty cells/parts, offsets, all output slots, repeatability.

The production function bodies are compiled unchanged. The independent oracle sums
per-ray float contributions in NumPy FP64, without production batch boundaries.
"""


import numpy as np
from sosfuse2.checks.gate import repeat_arrays
import unittest
from sosfuse2.backends.arithmetic import basis, cosine, norm_constants
from sosfuse2.backends.sh_layout import kernel_source, count

try:
    import cupy as cp
except ImportError:
    cp = None


def function(source, name):
    if name == 'sh_basis':
        start = source.index('__device__ void ' + name)
    elif name == 'sh_basis_fast':
        start = source.index('__device__ __forceinline__ void ' + name)
    elif name == 'sh_project':
        start = source.rindex('extern "C" __global__', 0, source.index('    sh_project('))
    elif name == 'projection_radiance':
        declaration = source.index('projection_radiance(')
        start = source.rindex('template <int Threads, int Rays>', 0, declaration)
    else:
        start = source.index('extern "C" __global__', source.index('#define ' + name))
    end = source.index('{', start)
    depth = 1
    while depth:
        end += 1
        depth += (source[end] == '{') - (source[end] == '}')
    return source[start : end + 1] + '\n'


def check_projection_batches(L, W, deterministic):
    source = kernel_source(L)
    K = count(L)
    lm = source[source.index('struct LMTable') : source.index('#endif', source.index('struct LMTable'))]
    fl_start = source.index('#define FLP_THREADS')
    # SP_THREADS is chosen under #ifdef FAST_VPROD_ON; start at that directive so the #else/#endif stay paired.
    sp_start = source.rindex('#ifdef FAST_VPROD_ON', 0, source.index('#define SP_THREADS'))
    header = (
        f'''typedef long long int64_t; typedef unsigned int uint32_t;
#define K {K}
#define NMOM (4+K)
#define TW {W}
#define FW {W}
#define SH_L {L}
#define GW(w) (w)
#define PI 3.14159265358979323846f
#define SP_PARTS 9
__device__ int ix(int l,int m){{return l*(l+1)/2+m;}}
__device__ float clip(float x,float a,float b){{return fminf(b,fmaxf(a,x));}}
'''
        + source[source.index('struct ProjectionSum {'):source.index('struct PointData {')]
        + lm
        + function(source, 'sh_basis')
        + function(source, 'sh_basis_fast')  # fl_project builds its basis rows with the fast helper
    )
    code = (
        header
        + function(source, 'projection_radiance')
        + source[fl_start : source.index('extern "C"', fl_start)]
        + function(source, 'FLP_THREADS')
        + function(source, 'sh_project')
        + source[sp_start : source.index('extern "C"', sp_start)]
        + function(source, 'SP_THREADS')
    )
    module = cp.RawModule(
        code=code, options=('--std=c++17', '--fmad=false', f'-DSOSFUSE2_DETERMINISTIC={deterministic}')
    )
    rng = np.random.default_rng(744)
    lengths = np.array([0, 1, 7, 8, 9, 31, 32, 33, 279, 288, 297], dtype='i8')
    base = 5
    offsets = np.r_[base, base + np.cumsum(lengths)].astype('i8')
    n = offsets[-1]
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    dirs[base] = [0, 0, 1]  # pole branch
    weights = rng.uniform(0.01, 1, n).astype('f4')
    targets = np.broadcast_to(np.eye(3), (len(lengths), 3, 3)).copy()
    radiance = rng.uniform(0, 1, (n - base, W, 4)).astype('f4')
    y = basis(dirs[base:, 2], L)
    st = np.sqrt(np.maximum(0, 1 - dirs[base:, 2] ** 2))
    az = np.ones_like(st)
    np.divide(dirs[base:, 0], st, out=az, where=st > 1e-12)
    cd = cosine(az.astype('f4'))
    m = np.array([m for l in range(L + 1) for m in range(l + 1)])
    tr = (radiance[:, :, 0] + radiance[:, :, 1]) + radiance[:, :, 2]
    weighted = tr * weights[base:, None]
    terms = np.concatenate(
        ((radiance * weights[base:, None, None]).transpose(0, 2, 1), weighted[:, None, :] * (y * cd[:, m])[:, :, None]),
        axis=1,
    )
    flux_terms = tr * (weights[base:] * np.maximum(0, dirs[base:, 2].astype('f4')))[:, None]
    want_m = np.zeros((len(lengths), 4 + K, W))
    want_f = np.zeros((len(lengths), W))
    for c, (a, b) in enumerate(zip(offsets[:-1] - base, offsets[1:] - base)):
        want_m[c] = terms[a:b].sum(axis=0, dtype='f8')
        want_f[c] = flux_terms[a:b].sum(axis=0, dtype='f8') * (0.2 / np.pi)
    dev = [cp.asarray(x) for x in (offsets, targets.astype('f4'), dirs.astype('f4'), weights)]
    rr = cp.asarray(radiance)
    yy = cp.asarray(y)
    coeff = tuple(cp.asarray(x, dtype='f4') for x in norm_constants(L))
    edge = cp.arange(n + 1, dtype='uint32')
    for name, block, grid, args in (
        (
            'fl_project',
            256,
            (len(lengths),),
            (np.int32(len(lengths)), np.int64(base), *dev, *coeff, rr, np.float32(0.2)),
        ),
        (
            'sh_project',
            256,
            (((K + 127) // 128 + 1) * len(lengths),),
            (np.int32(len(lengths)), np.int64(base), *dev, *coeff, rr, np.float32(0.2)),
        ),
        (
            'sos_slim_proj',
            128,
            (len(lengths), (5 + K + 63) // 64),
            (np.int32(len(lengths)), np.int32(4), np.int64(base), *dev, edge, yy, rr, np.float32(0.2)),
        ),
    ):
        kernel = module.get_function(name)
        outputs = []
        for _ in range(2):
            out = cp.full(want_m.shape, cp.nan, dtype='f4')
            flux = cp.full(want_f.shape, cp.nan, dtype='f4')
            kernel(grid, (block,), args + (out, flux))
            actual = (cp.asnumpy(out), cp.asnumpy(flux))
            # FP32 basis and compensated FP32 sums against the FP64 truth: rtol 1e-3, with near-zero
            # moments judged against the array's largest magnitude (as twilight differences are).
            for got, want in zip(actual, (want_m, want_f)):
                np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3 * np.abs(want).max(), err_msg=name)
            outputs.append(actual)
        for a, b in zip(*outputs):
            if deterministic:
                assert a.tobytes() == b.tobytes(), name
            else:
                result = repeat_arrays(a, b)
                assert result['pass_'], (name, result)


@unittest.skipIf(cp is None, 'CuPy/GPU required')
class ProjectionBatches(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            if not cp.is_available():
                raise unittest.SkipTest('No CUDA device available')
        except cp.cuda.runtime.CUDARuntimeError as exc:
            if exc.status in (35, 100):
                raise unittest.SkipTest(f'CUDA device unavailable ({exc.status})') from exc
            raise

    def test_contract(self):
        for L, W in ((16, 1), (16, 15), (24, 15), (32, 1), (32, 15)):
            for deterministic in (0, 1):
                with self.subTest(L=L, W=W, deterministic=deterministic):
                    check_projection_batches(L, W, deterministic)


if __name__ == '__main__':
    unittest.main()
