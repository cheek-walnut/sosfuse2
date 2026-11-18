"""Native support/profile checks and syntax coverage of split transport roles."""

import ctypes
import itertools
import platform
import subprocess

import numpy as np
import pytest

from sosfuse2.atmosphere import components
from sosfuse2.backends import numpy_reference as reference
from sosfuse2.checks import testkit
from sosfuse2.foundation import paths
from sosfuse2.foundation.top import Domain
from sosfuse2.tests.test_kernel_static import STUB, defines
from sosfuse2.tests.test_tropo_mie import SYNTAX_STUB_EXTRA

DOMAINS = (Domain(), Domain(120.0, 0.5, (120.0,), 'vacuum'), Domain(120.0, 0.5, (120.0,), 'us76_neff'))
DOMAIN_IDS = ('default', 'vacuum', 'us76_neff')


def test_native_column_record_layout(tmp_path):
    """Write a nonzero row in C++ and compare direct, preloaded and dumped values."""
    compiler = testkit.require_tool('clang++')
    buf, hdr32, hdr64 = components.allocate_column_records(np, 8)
    header = components.column_record_bytes(0)
    stride = components.column_record_bytes(1) - header
    assert buf.nbytes == components.column_record_bytes(8)
    assert header == hdr64.nbytes == hdr32.nbytes and not hdr64.any()
    assert np.shares_memory(buf, hdr32) and np.shares_memory(buf, hdr64)
    cpp = tmp_path / 'columns.cpp'
    cpp.write_text(
        '#include <math.h>\n' + STUB + SYNTAX_STUB_EXTRA
        + '\nDim blockIdx{},threadIdx{},blockDim{};extern "C" { double shared[50000]; }\n'
        + f'#include "{paths.KERNEL_CU}"\n'
        + f'static_assert(ColumnRecord::width == {components.SINV_F});\n'
        + f'static_assert(sizeof(ColumnRecord) == {stride});\n'
        + f'static_assert(ColumnRecord::header_words * sizeof(float) == {header});\n'
        + f'static_assert(ColumnRecord::strato_slot == {components.STRATO_COLUMN_SLOT});\n'
        + r'''
extern "C" void columns(float *buffer, float *direct, float *preloaded) {
    Transport<Sample> transport{};
    transport.nq = 3;
    transport.column_rows = ColumnRecord::rows(buffer);
    ColumnRecord &row = transport.column_record(8, 7, 1);
    for (int w = 0; w < FW; ++w)
        row.write_basis(w, w + 0.5f);
    ColumnRecord cached{};
    cached.preload_from(row);
    for (int k = 0; k < NCOMP; ++k) {
        direct[k] = row.component(k);
        preloaded[k] = cached.component(k);
    }
    direct[NCOMP] = row.stratosphere();
    preloaded[NCOMP] = cached.stratosphere();
}
'''
    )
    library = tmp_path / 'columns.so'
    link = ['-dynamiclib'] if platform.system() == 'Darwin' else ['-shared', '-fPIC', '-nostdinc++']
    subprocess.run(
        [compiler, '-std=c++17', '-O2', *defines(15, ('SINV', 'RECORD'), (0, 0, 1)), *link, str(cpp), '-o', str(library)],
        check=True, capture_output=True,
    )
    columns = ctypes.CDLL(str(library)).columns
    columns.argtypes, columns.restype = [ctypes.c_void_p] * 3, None
    buf.fill(-1)
    direct = np.full(components.NCOMP + 1, -1, 'f4')
    preloaded = direct.copy()
    columns(buf.ctypes.data, direct.ctypes.data, preloaded.ctypes.data)
    values = np.arange(components.SINV_F, dtype='f4') + 0.5
    for observed in (direct, preloaded, components.column_record_chunk(buf, 4, 1)):
        np.testing.assert_allclose(observed, values, rtol=1e-3, atol=0)
    assert np.all(buf[:header // buf.itemsize] == -1)
    assert np.all(components.column_record_chunk(buf, 0, 4) == -1)
    assert np.all(components.column_record_chunk(buf, 5, 3) == -1)
    assert components.column_record_chunk(buf, 8, 0).size == 0


@pytest.mark.parametrize('domain', DOMAINS, ids=DOMAIN_IDS)
@pytest.mark.parametrize('layers', (False, True), ids=('single', 'two_layer_strato'))
def test_native_support_and_profiles(tmp_path, domain, layers):
    """Run the actual C++ functions at both support ends and their neighboring floats."""
    testkit.require_tool('clang++')
    source = paths.KERNEL_CU.read_text()
    excerpt = source[source.index('struct ProfilePolicy {') : source.index('__device__ int ix(')]
    excerpt += source[source.index('struct ProfileParams {') : source.index('#define POINT_INLINE')]
    wrapper = r'''
extern "C" void profiles(float h, const float *table, const float *aer2, int mask, float *out) {
    ProfileParams p{1.2f, aer2, table, mask, 2.3f
#ifdef STRATO
        ,80.f, 40.f, 1.f
#endif
    };
    profile_at(h, p, out, out[5]);
}
extern "C" float supported(float h, float value, int air) {
    return support_value(h, value, air);
}
extern "C" float sided(float h, float inside) { return support_height(h, inside); }
'''
    cpp = tmp_path / 'support.cpp'
    cpp.write_text(
        '#include <cmath>\n#include <algorithm>\n#define __device__\n'
        '#define __forceinline__ inline\n#define TRACE_MIN 1.17549435e-38f\n'
        '#define FW 1\n#define NCOMP 5\nusing std::min;using std::max;\n' + excerpt + wrapper
    )
    library = tmp_path / 'support.so'
    flags = ['-DSTRATO', '-DAEROSOL_2L'] if layers else []
    subprocess.run(
        [
            'clang++',
            '-std=c++17',
            '-O2',
            '-ffp-contract=off',
            '-fPIC',
            '-dynamiclib' if platform.system() == 'Darwin' else '-shared',
            *flags,
            *domain.defines(),
            str(cpp),
            '-o',
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    native = ctypes.CDLL(str(library))
    pointer, float_, int_ = ctypes.c_void_p, ctypes.c_float, ctypes.c_int
    native.profiles.argtypes = [float_, pointer, pointer, int_, pointer]
    native.profiles.restype = None
    native.supported.argtypes, native.supported.restype = [float_, float_, int_], float_
    native.sided.argtypes, native.sided.restype = [float_, float_], float_
    ends = np.array([80.0, domain.top_km], 'f4')
    heights = np.r_[
        -1.0,
        0.0,
        1.3,
        19.1,
        41.0,
        ends,
        np.nextafter(ends, np.float32(-np.inf)),
        np.nextafter(ends, np.float32(np.inf)),
        domain.top_km + 1.0,
    ].astype('f4')
    z = np.arange(domain.n) * domain.step_km
    table = np.ascontiguousarray(np.column_stack([2.3 * np.exp(-z / 8), 1 + z / 100, 0.3 + z / 200]), 'f4')
    aer2 = np.array([1.0, 1.0, 1.0, 0.7, 1.0, 0.1, 2.0], 'f4')
    for mask in range(4):
        optics = {
            'table': table,
            'mask': mask,
            'domain': domain,
            'air_surface': 2.3,
            'meta': {'profile_step_km': domain.step_km},
        }
        constants = {'domain': domain, 'height': np.float32(1.2), 'density_optics': optics}
        if layers:
            constants.update(aer2=aer2, strato_profile={'center': 80.0, 'sigma': 40.0, 'norm': 1.0})
        actual = np.empty((len(heights), 6), 'f4')
        for i, h in enumerate(heights):
            native.profiles(h, table.ctypes.data, aer2.ctypes.data, mask, actual[i].ctypes.data)
        expected = reference.profile_at(heights, constants)
        np.testing.assert_allclose(actual[:, :5], expected, rtol=1e-3, atol=0)
        expected_strato = reference.strato_profile(heights, constants) if layers else np.zeros(len(heights))
        np.testing.assert_allclose(actual[:, 5], expected_strato, rtol=1e-3, atol=0)

    # Support must select zero, even when an interpolated source is negative or NaN.
    for air, value in itertools.product((False, True), (2.0, -3.0, np.nan)):
        actual = np.array([native.supported(h, value, air) for h in heights], 'f4')
        expected = reference.support_value(heights, value, domain, air=air)
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0)
        if domain.top_km > 80:
            end = domain.air_end if air else 80.0
            assert np.all(actual[heights > end] == 0)

    for inside in (79.0, 81.0, 119.0, 121.0):
        actual = np.array([native.sided(h, inside) for h in heights], 'f4')
        expected = reference.support_height(heights, np.float32(inside), domain)
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0)
        # A tolerance on heights alone would miss a one-ULP side-selection error.
        for end in (domain.lower_end, domain.air_end):
            assert np.all((actual > end) == (expected > end))


@pytest.mark.parametrize('domain', DOMAINS, ids=DOMAIN_IDS)
@pytest.mark.parametrize('order', (16, 24, 32))
def test_split_role_syntax(tmp_path, domain, order):
    """Exercise each point representation, path precision, and output role."""
    testkit.require_tool('clang++')
    stub = tmp_path / 'stub.h'
    stub.write_text(STUB + SYNTAX_STUB_EXTRA)
    variants = [
        (1, ('AEROSOL_2L',), (1, 2, 1)),
        (15, ('STRATO', 'AEROSOL_2L'), (1, 0, 1)),
        (15, ('STRATO', 'AEROSOL_2L'), (0, 0, 1)),
        (15, ('STRATO', 'SH_COMBINE', 'WARM_RAYSPLIT'), (0, 0, 1)),
        (15, ('STRATO', 'SH_COMBINE', 'SINV', 'RECORD'), (0, 0, 1)),
        (15, ('STRATO', 'SH_COMBINE', 'SINV', 'PROD'), (0, 0, 1)),
        (15, ('STRATO', 'SH_COMBINE', 'SINV', 'SLIM'), (0, 0, 1)),
        (15, ('STRATO', 'SH_COMBINE', 'SINV', 'SLIM', 'SLIM_RAYS', 'LAUNCH_RAY_LIMIT'), (0, 0, 1)),
        (15, ('STRATO', 'TROPO_MIE'), (1, 2, 1)),
        (15, ('STRATO', 'GROUND_SH_L=16', 'GROUND_SH_GRID=32'), (1, 2, 1)),
        (15, ('STRATO', 'LUT_ONLY'), (1, 2, 1)),
    ]
    for waves, flags, spec in variants:
        result = subprocess.run(
            [
                'clang++',
                '-std=c++17',
                '-fsyntax-only',
                '-x',
                'c++',
                '-include',
                str(stub),
                *defines(waves, flags, spec),
                f'-DSH_L={order}',
                *domain.defines(),
                str(paths.KERNEL_CU),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (waves, flags, spec, result.stderr)
    print(f'SPLIT {len(variants)} syntax: L={order}, domain={domain.upper_air or 'default'} PASS')
