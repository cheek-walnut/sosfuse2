"""Host FP32 checks of the source-cell indices written and read by direct transport."""

import ctypes
import subprocess
import sys

import numpy as np

from sosfuse2.backends.sh_layout import FastTransport, kernel_source
from sosfuse2.checks import testkit
from sosfuse2.jobs.run_state import refined_chi
from sosfuse2.rays.shell_segments import grids
from sosfuse2.tests.test_kernel_static import STUB
from sosfuse2.tests.test_tropo_mie import SYNTAX_STUB_EXTRA


def test_direct_uses_shared_cell_selection():
    source = kernel_source(24, combine=True)
    for name in ('vp_c0cp_chunk', 'point_slim'):
        body = source.split(f' {name}(', 1)[1].split('\n}', 1)[0]
        assert 'path_cell(' in body, name


def test_direct_cells_on_chi_boundaries(tmp_path):
    compiler = testkit.require_tool('clang++')
    layout = FastTransport(24, 15, 'direct', strato=True)
    # Lane zero evaluates one node on a single ray with a live lower height.
    # Only its own vote/shuffle is needed to select the cell.
    stub = STUB + SYNTAX_STUB_EXTRA + r'''
#define __ballot_sync(mask, predicate) ((predicate) ? 1u : 0u)
Dim blockIdx{}, threadIdx{}, blockDim{128, 1, 1};
extern "C" { float4 shared_vectors[25000]; float2 staged_plane[1]; }
'''
    probe = r'''
extern "C" void direct_cells(const float *p, const float *chi, int nc, int *result) {
    const float d[3] = {0.f, 1.f, 0.f}, frame[9] = {}, dl[3] = {};
    const float h[] = {0.f, 80.f}, density[] = {1.f, 1.f}, u = 0.f;
    const int64_t cells[] = {0, 1};
    const uint32_t offsets[] = {0, 2};
    const uint16_t lower = 1;
    const RayGeometry ray{p[1], 0.f, 0.f};
    const PathEdge edges[] = {{0.f, 1.f}, {1.f, 0.f}};
    int2 record{};
    vp_c0cp_chunk(1, 2, nc, 1, 0, 1, 0, cells, p, d, offsets, edges, &lower, &ray,
                  &u, h, chi, density, density, &record, density);
    SlimPoint sample{};
    float height;
    point_slim(sample, p, d, frame, dl, path_segment(p, ray, edges, true), u, 1.f,
               h, 2, chi, nc, 0.f, 1.f, true, nullptr, 0, height);
    result[0] = record.x;
    result[1] = sample.cell[0];
}
'''
    cpp, lib = tmp_path / 'cells.cpp', tmp_path / 'cells.so'
    cpp.write_text('#include <math.h>\n' + stub + kernel_source(24, combine=True) + probe)
    link = ['-dynamiclib'] if sys.platform == 'darwin' else ['-shared', '-fPIC', '-nostdinc++']
    subprocess.run([compiler, '-std=c++17', '-O2', '-ffp-contract=off', *link,
                    '-DFW=15', '-DSTRATO', '-DK_FIRST=0', '-DK_PHASE=0', '-DK_BOUNDARY=0',
                    *layout.defines(), str(cpp), '-o', str(lib)], check=True)
    run = ctypes.CDLL(str(lib)).direct_cells
    floats = np.ctypeslib.ndpointer(dtype='f4', flags='C_CONTIGUOUS')
    run.argtypes = [floats, floats, ctypes.c_int, np.ctypeslib.ndpointer(dtype='i4', flags='C_CONTIGUOUS')]
    run.restype = None

    chi = refined_chi(grids(1)[1], 0.5, [[88, 104.5], [104, 104.25]]).astype('f4')
    rng = np.random.default_rng(4100)
    knots = np.repeat(np.arange(1, len(chi) - 1), 9)
    ulps = np.tile(np.arange(-4, 5), len(chi) - 2)
    angle = chi[knots] + ulps * np.spacing(chi[knots])
    phi = rng.uniform(-np.pi, np.pi, len(angle))
    radius = rng.uniform(6371., 6451., len(angle))
    rho = radius * np.sin(angle)
    points = np.column_stack((rho * np.cos(phi), rho * np.sin(phi), radius * np.cos(angle), radius - 6371., radius))
    witness = [4100.9838867, 2.4605908, 4887.3637695, 9., 6380.]
    points = np.vstack((witness, points)).astype('f4')
    cells = np.empty((len(points), 2), dtype='i4')
    for p, result in zip(points, cells, strict=True):
        run(p, chi, len(chi), result)
    np.testing.assert_array_equal(cells[:, 0], cells[:, 1])
    assert cells[0, 0] == 4
    np.testing.assert_array_equal(cells[1:, 0][ulps == -4], knots[ulps == -4] - 1)
    np.testing.assert_array_equal(cells[1:, 0][ulps == 4], knots[ulps == 4])
