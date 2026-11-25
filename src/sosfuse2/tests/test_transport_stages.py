"""Native checks of Fourier corner contraction and full/compact sample views.

These exercise the actual C++ stages. Warp shuffles and GPU scheduling are not
simulated; their numerical validation belongs to the CUDA fixed-fixture gate.
"""

import ctypes
import platform
import subprocess

import numpy as np
import pytest

from sosfuse2.checks import testkit
from sosfuse2.foundation import paths
from sosfuse2.tests.test_kernel_static import STUB
from sosfuse2.tests.test_tropo_mie import SYNTAX_STUB_EXTRA

CASES = (
    (1, False, False, 16),
    (15, False, False, 24),
    (1, True, False, 32),
    (15, True, False, 16),
    (15, True, False, 24),
    (15, True, True, 16),
    (15, True, True, 32),
)


@pytest.mark.parametrize('waves,strato,combined,order', CASES)
def test_native_fourier_and_views(tmp_path, waves, strato, combined, order):
    testkit.require_tool('clang++')
    source = tmp_path / 'stages.cpp'
    library = tmp_path / 'stages.so'
    source.write_text(
        '#include <math.h>\n'
        + STUB
        + SYNTAX_STUB_EXTRA
        + '\nDim blockIdx{},threadIdx{},blockDim{};extern "C" { float4 shared_vectors[25000]; }\n'
        + f'#include "{paths.KERNEL_CU}"\n'
        + r'''
extern "C" void contract(const float *y, const float *coeff, int live_a, int live_s,
                         float *out, float *strato_out, float *last, float *strato_last, int *tags_out) {
    Sample s{};
    for (int c=0;c<4;c++) {
        s.cell[c]=10+c;
        for (int w=0;w<TW;w++) {
            s.c32[c][w]=-17.f;
#ifdef STRATO
            s.st_c32[c][w]=-19.f;
#endif
        }
    }
    for (int lane=0;lane<32;lane++) {
        float a[4][TW], b[4][TW];
        int tags[4]={-1,-1,-1,-1};
        for (int c=0;c<4;c++)for(int w=0;w<TW;w++){a[c][w]=-11.f;b[c][w]=-13.f;}
        contract_corner(s,a,b,tags,y,coeff,lane,0,1,3,live_a,live_s);
        reuse_corner<1,0>(s,a,b,tags,lane);
        for (int c=0;c<4;c++)for(int w=0;w<TW;w++){
            out[(lane*4+c)*TW+w]=a[c][w];
            strato_out[(lane*4+c)*TW+w]=b[c][w];
        }
        for(int c=0;c<4;c++)tags_out[lane*4+c]=tags[c];
    }
    for(int c=0;c<4;c++)for(int w=0;w<TW;w++){
        last[c*TW+w]=s.c32[c][w];
#ifdef STRATO
        strato_last[c*TW+w]=s.st_c32[c][w];
#else
        strato_last[c*TW+w]=-19.f;
#endif
    }
}
extern "C" void views(const float *tensor, const float *weights, const int *cells,
                      const float *moment, float *out) {
    Sample s{}; PointM2 compact{};
    for(int k=0;k<16;k++)s.T[k]=compact.T[k]=tensor[k];
    for(int c=0;c<4;c++){s.sw[c]=compact.sw[c]=weights[c];s.cell[c]=compact.cell[c]=cells[c];}
    s.ir=compact.ir=7;s.h=compact.h=83.25f;
    FullSampleView<Sample> full(s); CompactSampleView small(compact);
    for(int c=0;c<4;c++){
        out[c]=full.rotate(c,moment);out[4+c]=small.rotate(c,moment);
        out[8+c]=full.weight(c);out[12+c]=small.weight(c);
        out[16+c]=full.cell(c);out[20+c]=small.cell(c);
    }
    out[24]=full.ir();out[25]=small.ir();out[26]=full.height();out[27]=small.height();
}
'''
    )
    flags = [f'-DFW={waves}', f'-DSH_L={order}', '-DK_FIRST=0', '-DK_PHASE=0', '-DK_BOUNDARY=1']
    if strato:
        flags.append('-DSTRATO')
    if combined:
        flags.append('-DSH_COMBINE')
    link = ['-dynamiclib'] if platform.system() == 'Darwin' else ['-shared', '-fPIC', '-nostdinc++']
    result = subprocess.run(
        ['clang++', '-std=c++17', '-O2', '-ffp-contract=off', *flags, *link, str(source), '-o', str(library)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    native = ctypes.CDLL(str(library))
    pointer = ctypes.c_void_p
    native.contract.argtypes = [pointer, pointer, ctypes.c_int, ctypes.c_int, pointer, pointer, pointer, pointer, pointer]
    native.contract.restype = None
    native.views.argtypes = [pointer] * 5
    native.views.restype = None

    rng = np.random.default_rng(waves * 100 + order)
    count = (order + 1) * (order + 2) // 2
    padded = (count + 31) // 32 * 32
    planes = 2 if strato else 1
    coeff = rng.uniform(0.1, 0.9, (planes * 3, (waves + 1) // 2, padded, 2)).astype('f4')
    # Signed wavelength rows with well-conditioned sums: the independent FP64
    # dot oracle tests addressing and l/m selection without cancellation noise.
    coeff[..., 1] *= -1
    basis = rng.uniform(0.1, 0.9, count).astype('f4')
    masks = ((False, False), (False, True), (True, False), (True, True)) if strato and waves == 15 else ((True, True),)
    for live_a, live_s in masks:
        actual = np.empty((32, 4, waves), 'f4')
        st_actual = np.empty_like(actual)
        last = np.empty((4, waves), 'f4')
        st_last = np.empty_like(last)
        tags = np.empty((32, 4), 'i4')
        native.contract(basis.ctypes.data, coeff.ctypes.data, live_a, live_s, actual.ctypes.data,
                        st_actual.ctypes.data, last.ctypes.data, st_last.ctypes.data, tags.ctypes.data)
        expected = np.full_like(actual, -11)
        st_expected = np.full_like(actual, -13)
        expected[:, :2] = 0
        if strato:
            st_expected[:, :2] = 0
        last_expected = np.full_like(last, -17)
        st_last_expected = np.full_like(last, -19)
        for m in range(32):
            indices = np.array([degree * (degree + 1) // 2 + m for degree in range(m, order + 1)], dtype=int)
            for w in range(waves):
                if live_a or (combined and live_s):
                    expected[m, :2, w] = basis[indices].astype('f8') @ coeff[1, w // 2, indices, w % 2].astype('f8')
                if strato and not combined and live_s:
                    st_expected[m, :2, w] = basis[indices].astype('f8') @ coeff[4, w // 2, indices, w % 2].astype('f8')
        if order == 32:
            for w in range(waves):
                if live_a or (combined and live_s):
                    last_expected[:2, w] = float(basis[560]) * float(coeff[1, w // 2, 560, w % 2])
                if strato and combined:
                    st_last_expected[:2, w] = 0
                elif strato and live_s:
                    st_last_expected[:2, w] = float(basis[560]) * float(coeff[4, w // 2, 560, w % 2])
        else:
            last_expected[:2] = 0
            if strato:
                st_last_expected[:2] = 0
        for observed, expected_values in ((actual, expected), (st_actual, st_expected), (last, last_expected), (st_last, st_last_expected)):
            np.testing.assert_allclose(observed, expected_values, rtol=1e-3, atol=0)
        np.testing.assert_array_equal(tags, np.tile([1, 11, -1, -1], (32, 1)))

    tensor = (np.arange(1, 17, dtype='f4') / 8).reshape(4, 4)
    tensor[1::2] *= -1
    weights = np.array([0.25, 0.125, 0.5, 0.125], 'f4')
    cells = np.array([7, 8, 17, 18], 'i4')
    moment = np.array([0.5, 2, 1, 0.25], 'f4')
    observed = np.empty(28, 'f4')
    native.views(tensor.ctypes.data, weights.ctypes.data, cells.ctypes.data, moment.ctypes.data, observed.ctypes.data)
    rotation = tensor.astype('f8') @ moment.astype('f8')
    expected = np.r_[rotation, rotation, weights, weights, cells, cells, 7, 7, 83.25, 83.25]
    np.testing.assert_allclose(observed, expected, rtol=1e-3, atol=0)
