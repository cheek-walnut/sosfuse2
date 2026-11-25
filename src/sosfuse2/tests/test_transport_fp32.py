"""Exercise the transport arithmetic with host-compiled kernel helpers and host CSR inputs."""

import ctypes
import platform
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from sosfuse2.backends.arithmetic import basis, norm_constants
from sosfuse2.checks import testkit
from sosfuse2.foundation import paths
from sosfuse2.jobs.pipeline_execution import _transport_payload, promote_bundle
from sosfuse2.solve.ray_geometry import Bundle, ResidentGeometry, device_geometry, device_paths
from sosfuse2.solve.source_iteration import _anderson_weights
from sosfuse2.tests.test_kernel_static import STUB, defines
from sosfuse2.tests.test_tropo_mie import SYNTAX_STUB_EXTRA


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    compiler = testkit.require_tool('clang++')
    folder = tmp_path_factory.mktemp('fp32')
    cpp, library = folder / 'geometry.cpp', folder / 'geometry.so'
    cpp.write_text(
        '#include <math.h>\n' + STUB + SYNTAX_STUB_EXTRA +
        '\nDim blockIdx{}, threadIdx{}, blockDim{}; extern "C" { float4 shared_vectors[25000]; }\n' +
        f'#include "{paths.KERNEL_CU}"\n' + r'''
extern "C" void panels(const float *p, const RayGeometry *r, const PathEdge *edges, int low,
                       int n, const float *u, float *height, float *width, int *lit) {
    const auto path = path_segment(p, *r, edges, low);
    float prev = path.start();
    for (int i=0; i<n; ++i) {
        const float x = path.node(u[i]);
        height[i] = path.height(x);
        width[i] = path.width(prev,x);
        lit[i] = path.lit(x);
        prev = x;
    }
}
extern "C" void harmonics(float mu, float st, const float *aa, const float *bb,
                          const float *dd, float *out) {
#if defined(__aarch64__)
    unsigned long save;
    asm volatile("mrs %0, fpcr" : "=r"(save));
    const unsigned long flush = save | (1ul << 24);
    asm volatile("msr fpcr, %0" :: "r"(flush) : "memory");
#elif defined(__SSE__)
    unsigned save, flush;
    asm volatile("stmxcsr %0" : "=m"(save));
    flush = save | (1u << 15);
    asm volatile("ldmxcsr %0" :: "m"(flush) : "memory");
#endif
    for (int lane=0; lane<32; ++lane)
        sh_basis(mu, st, lane, out, aa, bb, dd);
#if defined(__aarch64__)
    asm volatile("msr fpcr, %0" :: "r"(save) : "memory");
#elif defined(__SSE__)
    asm volatile("ldmxcsr %0" :: "m"(save) : "memory");
#endif
}
extern "C" float projected(float *values, int n) {
    ProjectionSum sum{};
    for (int i=0; i<n; ++i) sum.add(values[i]);
    return sum.value();
}
'''
    )
    link = ['-dynamiclib'] if platform.system() == 'Darwin' else ['-shared', '-fPIC', '-nostdinc++']
    subprocess.run([compiler, '-std=c++17', '-O2', '-ffp-contract=off', *link,
                    *defines(1, ('TOP_KM=200.f', 'LOWER_END_KM=80.f')), str(cpp), '-o', str(library)], check=True)
    dll = ctypes.CDLL(str(library))
    f = np.ctypeslib.ndpointer(dtype='f4', flags='C_CONTIGUOUS')
    i = np.ctypeslib.ndpointer(dtype='i4', flags='C_CONTIGUOUS')
    dll.panels.argtypes = [f, f, f, ctypes.c_int, ctypes.c_int, f, f, f, i]
    dll.harmonics.argtypes = [ctypes.c_float, ctypes.c_float, f, f, f, f]
    dll.projected.argtypes, dll.projected.restype = [f, ctypes.c_int], ctypes.c_float
    return dll


def host_ray(p, d, edges, lower):
    return dict(starts=np.array([p], 'f8'), dirs=np.array([d], 'f8'), targets=np.eye(3)[None, :, :],
                edges=np.asarray(edges, 'f8'), edge_off=np.array([0, len(edges)], 'u4'),
                cell_off=np.array([0, 1], 'i8'), lower=np.array([lower], 'u2'))


def panels(native, host, u, segment=0):
    a = device_geometry(host)
    nodes = np.asarray(u, 'f4')
    h, widths = np.empty_like(nodes), np.empty_like(nodes)
    lit = np.empty(len(nodes), 'i4')
    low = (int(host['lower'][0]) >> segment) & 1
    native.panels(a['starts'], a['ray_geometry'], a['edges'][segment:], low, len(u), nodes, h, widths, lit)
    return h, widths, lit


def test_shell_endpoint_and_tangent(native):
    R, h0, mu = 6371., 40., -.1
    p, d = np.array([R+h0, 0, 0]), np.array([mu, np.sqrt(1-mu*mu), 0])
    pd = p @ d
    crossing = -pd + np.sqrt(pd*pd + (80-h0)*(2*R+80+h0))
    host = host_ray(p, d, [-pd, crossing, crossing+5], 3)
    nodes = np.array([0., .1, .5, 1.])
    h, _, _ = panels(native, host, nodes)
    t = -pd + (crossing+pd)*nodes**2
    expected = np.linalg.norm(p + t[:, None]*d, axis=1) - R
    np.testing.assert_allclose(h, expected, rtol=1e-3, atol=0)
    outside, _, _ = panels(native, host, nodes, 1)
    assert h[-1] <= 80 < outside[0]
    # Both intervals share the exact same host intersection.
    assert abs(h[-1]-80) < .0001 and abs(outside[0]-80) < .0001


@pytest.mark.parametrize('lower', [False, True])
def test_local_panel_widths(native, lower):
    a, length = 1000., .01
    u = np.r_[0., (np.polynomial.legendre.leggauss(8)[0] + 1)/2, 1.]
    host = host_ray([6400, 0, 0], [0, 1, 0], [a, a+length], lower)
    _, w, _ = panels(native, host, u)
    x = u**2 if lower else -(1-u)**2
    assert np.all(w[1:] > 0)
    np.testing.assert_allclose(w[1:], length*np.diff(x), rtol=1e-3, atol=0)
    assert abs(w.sum(dtype='f8')-length) < length*1e-6


def test_shadow_crossings_and_direct_transmission(native):
    R, h = 6371., 10.
    r = R+h
    horizon = -np.sqrt(h*(2*R+h))/r
    offsets = np.r_[np.arange(-10, 11)*abs(np.spacing(np.float32(horizon))), -1e-10, 1e-10]
    mu = horizon + offsets
    # The naive local FP32 comparison can turn on a finite direct term.
    delta = mu.astype('f4') - np.float32(horizon)
    truth = offsets >= 0
    flips = (delta >= 0) != truth
    assert flips[-2] and np.max(abs(offsets[flips])) <= abs(np.spacing(np.float32(horizon)))
    transmission = np.exp(-1.)  # unit incident radiance, unit slant optical depth
    assert np.max(abs(np.where(delta >= 0, transmission, 0) - truth*transmission)) == transmission
    for m, dark in zip(mu[offsets != 0], ~truth[offsets != 0]):
        p = [r*np.sqrt(1-m*m), 0, r*m]
        _, _, lit = panels(native, host_ray(p, [0, 0, 1], [0, 1], False), [0])
        assert bool(lit[0]) != dark
    # A horizontal ray enters and exits the shadow. A vertical one crosses the terminator.
    for p, d, cuts, expected in [
        ([-R-2, 100, -500], [1, 0, 0], [0, 2*R+5], [1, 0, 1]),
        ([R-1, 0, -200], [0, 0, 1], [0, 400], [0, 0, 1]),
    ]:
        _, _, lit = panels(native, host_ray(p, d, cuts, True), [0, .5, 1])
        np.testing.assert_array_equal(lit, expected)


def test_actual_sh_coefficients_with_poles(native):
    theta = np.r_[0., 1e-4, .017, .06, .317, .93, 1.72, np.pi-.004, np.pi]
    mu, st = np.cos(theta), np.hypot(.6*np.sin(theta), .8*np.sin(theta))
    st[[0, -1]] = 0
    constants = [v.astype('f4') for v in norm_constants(32)]
    expected = basis(mu, 32)  # FP64 recurrence, rounded at output
    observed = np.empty_like(expected)
    for m, t, row in zip(mu, st, observed):
        native.harmonics(m, t, *constants, row)
    # Each order starts at its diagonal. Flushing that seed zeros the entire
    # recurrence, even when a later reference value grows into the normal range.
    m = np.array([order for degree in range(33) for order in range(degree + 1)])
    diagonal = np.array([order * (order + 1) // 2 + order for order in range(33)])
    seed = expected[:, diagonal].astype('f8') / np.where(np.arange(33) > 0, np.sqrt(2.), 1.)
    flushed = (np.abs(seed) < np.finfo('f4').tiny)[:, m]
    np.testing.assert_allclose(observed[~flushed], expected[~flushed], rtol=1e-3, atol=0)
    assert observed[1, -1] == 0  # sin(theta)**32 is below the FP32 normal range
    if platform.machine().lower() in ('arm64', 'aarch64', 'x86_64', 'amd64'):
        assert np.all(observed[flushed] == 0)
        assert 0 < expected[3, -1] < np.finfo('f4').tiny and observed[3, -1] == 0
    assert np.count_nonzero(observed[0]) == 33


def test_compensated_projection_and_host_gram(native):
    values = np.array([1e8, 1, -1e8] * 17, 'f4')
    assert native.projected(values, len(values)) == values.sum(dtype='f8')
    f = np.full((32, 4), 3, 'f4')
    gram = f.T @ f
    assert gram.dtype == np.float32
    np.testing.assert_array_equal(gram + np.eye(4, dtype='f4')*np.float32(gram.trace()*1e-10), gram)
    weights = _anderson_weights(gram)
    assert weights.dtype == np.float32
    np.testing.assert_allclose(weights, .25, rtol=1e-3, atol=0)


def test_prepared_path_batches_preserve_ray_order(monkeypatch):
    n = 32771
    row = host_ray([6371., 0., 0.], [1., 0., 0.], [0., 80., 200.], True)
    row.update(starts=np.array([[6371., 0., 0.], [6571., 0., 0.]]),
               dirs=np.tile(row['dirs'], (n, 1)), edges=np.tile(row['edges'], n),
               edge_off=np.arange(n + 1, dtype='u4') * 3, cell_off=np.array([0, n - 2, n]))
    monkeypatch.setenv('SOSFUSE2_GEOMETRY_WORKERS', '1')
    serial = device_paths(row)
    monkeypatch.setenv('SOSFUSE2_GEOMETRY_WORKERS', '2')
    parallel = device_paths(row)
    for key in serial:
        np.testing.assert_allclose(parallel[key], serial[key], rtol=1e-5, atol=0)
    np.testing.assert_array_equal(parallel['ray_geometry'][-3:, 0], [6371, 6571, 6571])
    np.testing.assert_array_equal(parallel['edges'][-9:, 1], [80, -120, 0, -80, -120, 0, -80, -120, 0])


def test_geometry_upload_and_prefix(monkeypatch):
    xp = SimpleNamespace(asarray=np.asarray)
    start = [6411., 0., 0.]
    mu = -.1
    direction = np.array([mu, np.sqrt(1-mu*mu), 0.])
    rays = np.array([direction, [1., 0., 0.]])
    # A non-representable position must survive column-cache misses on the host.
    start[0] += 1e-7
    g = dict(p=np.tile(start, (2, 1)), target=np.tile(np.eye(3), (2, 1, 1)), d=rays, weight=np.ones(2),
             edges=np.array([[0., 1000., 1000.01], [0., 10., 20.]]), lower=np.ones((2, 2), bool),
             end_fraction=np.zeros(2), end_index=np.zeros(2), end_columns=np.ones((2, 5)),
             hits_ground=np.array([True, False]), nq=8)
    host = Bundle([g, g], np)
    direct = Bundle([g, g], xp)
    promoted = promote_bundle(host, xp)
    wire = _transport_payload(host)
    next_group = _transport_payload(host)
    assert wire.arrays['ray_geometry'] is next_group.arrays['ray_geometry']
    assert wire.arrays['edges'] is next_group.arrays['edges']
    assert wire.arrays['starts'] is host.arrays['starts'] and wire.arrays['dirs'] is host.arrays['dirs']
    assert host.arrays['edges'].ndim == 1 and '_device_paths' not in wire.__dict__
    resident = ResidentGeometry.__new__(ResidentGeometry)
    resident.s = SimpleNamespace(kernel=SimpleNamespace(xp=xp, sync=lambda: None, host=np.asarray),
                                  geo=SimpleNamespace(cfg={'path_n': 8}, chi=[0]))
    resident.stats = {}
    resident.install(host.arrays, [dict(samples=32)]*2)
    for other in (promoted, promote_bundle(wire, xp), resident.bundle):
        assert direct.arrays.keys() == other.arrays.keys()
        for key, value in direct.arrays.items():
            np.testing.assert_array_equal(value, other.arrays[key])
            assert value.dtype.kind != 'f' or value.dtype == np.float32
        np.testing.assert_array_equal(other.host_coordinates['starts'], host.arrays['starts'])
        np.testing.assert_array_equal(other.host_coordinates['dirs'], host.arrays['dirs'])
    assert promoted.arrays['edges'][1, 1] == np.float32(.01)
    assert promoted.arrays['starts'][0, 0] != host.arrays['starts'][0, 0]
    prefix = resident.prefix(3)
    assert len(prefix.arrays['ray_geometry']) == len(prefix.host_coordinates['dirs']) == 3
    assert len(prefix.arrays['edges']) == int(prefix.arrays['edge_off'][-1])
    np.testing.assert_array_equal(prefix.arrays['hits'], [1, 0, 1])
    np.testing.assert_array_equal(prefix.host_coordinates['cell_off'], [0, 2, 3])

    from sosfuse2.precompute import column_cache, column_parallel
    from sosfuse2.solve.source_solver import Kernel

    used = []

    def integrate(out, kind, starts, offsets, dirs, old, optics):
        np.testing.assert_array_equal(starts, host.arrays['starts'])
        np.testing.assert_array_equal(dirs, host.arrays['dirs'])
        np.testing.assert_array_equal(offsets, host.arrays['cell_off'])
        used.append(kind)
        out.fill(0)
        return True

    monkeypatch.setattr(column_cache, 'cached_columns', lambda k, b, c, kind, build: build())
    monkeypatch.setattr(column_parallel, 'parallel_columns', integrate)
    monkeypatch.setattr('sosfuse2.atmosphere.density.profile_identity', lambda _: {'sha256': 'density'})
    kernel = SimpleNamespace(xp=xp, host=np.asarray)
    service = promote_bundle(wire, xp)
    strato = dict(strato_profile=dict(center=20., sigma=3., norm=1.), strato_old=None)
    Kernel.strato_end_columns(kernel, service, strato)
    Kernel.density_end_columns(kernel, service, dict(density_optics=dict(old=None)))
    assert used == ['strato', 'density']
