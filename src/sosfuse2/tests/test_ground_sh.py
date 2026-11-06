"""Ground SH normalization/mirroring and actual-LUT quadrature tests (CPU only)."""

from sosfuse2 import paths
import numpy as np
import pytest
from sosfuse2 import testing
from sosfuse2.ground_sh import configured_order, lut_eigenvalues, project, real_basis

try:
    from sosfuse2.numerical_checks import assert_equivalent, within_tol
except ImportError:  # standalone research harness
    from numerical_checks import assert_equivalent, within_tol


@pytest.mark.parametrize('order', [16, 24, 32])
def test_mirrored_convolution_matches_direct_legendre(order):
    rng = np.random.default_rng(312)
    directions = rng.normal(size=(19, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    views = rng.normal(size=(23, 3))
    views /= np.linalg.norm(views, axis=1)[:, None]
    values = rng.uniform(0.1, 1, (19, 3))
    # A polynomial phase gives an independent exact direct-pair convolution.
    eigen = (0.37 ** np.arange(order + 1))[:, None] * np.array([1.0, 0.7, 0.4])
    out = project(np.array([0, 19, 19]), directions, values, 0.37, eigen, order)
    got = real_basis(views, order) @ out[0, :, :, 1].T
    plus = np.clip(views @ directions.T, -1, 1)
    mirror = directions.copy()
    mirror[:, 1] *= -1
    minus = np.clip(views @ mirror.T, -1, 1)
    coeff = (2 * np.arange(order + 1) + 1)[:, None] * eigen / (4 * np.pi)
    expected = np.stack(
        [
            ((np.polynomial.legendre.legval(plus, coeff[:, w]) + np.polynomial.legendre.legval(minus, coeff[:, w])) / 2)
            @ values[:, w]
            for w in range(3)
        ],
        axis=-1,
    )
    np.testing.assert_allclose(got, expected, rtol=8e-7, atol=1e-8)
    assert not np.any(out[1])


def test_lut_moments_preserve_nonunit_normalization():
    u = np.linspace(-1, 1, 18)
    lut = np.stack([2 + 0.6 * u, 1 - 0.3 * u], axis=-1) / (4 * np.pi)
    got = lut_eigenvalues(lut, 32)
    expected = np.zeros_like(got)
    expected[0] = [2, 1]
    expected[1] = [0.2, -0.1]
    np.testing.assert_allclose(got, expected, atol=2e-14)


def test_piecewise_linear_lut_moments():
    # |u| has a knot at zero: known integral against P0, P2 and P4.
    lut = np.array([[1.0], [0.0], [1.0]]) / (2 * np.pi)
    got = lut_eigenvalues(lut, 16)[:, 0]
    np.testing.assert_allclose(got[[0, 2, 4]], [1.0, 0.25, -1 / 24], atol=1e-14)
    np.testing.assert_allclose(got[1::2], 0, atol=1e-14)


@pytest.mark.parametrize('dtype,contiguous,block', [('f4', True, 1 << 18), ('f8', False, 1 << 18), ('f4', True, 64)])
def test_project_matches_cell_by_cell_evaluation(dtype, contiguous, block):
    # Directions repeat across cells (a fixed angular rule); the shared-basis
    # projection must agree with evaluating the basis cell by cell (tol, not
    # bytes), for either precision, a strided view and small row blocks.
    rng = np.random.default_rng(7)
    pool = rng.normal(size=(40, 3))
    pool /= np.linalg.norm(pool, axis=1)[:, None]
    counts = rng.integers(0, 30, 25)
    counts[[3, 11]] = 0
    goff = np.r_[0, np.cumsum(counts)]
    gdir = pool[rng.integers(0, 40, goff[-1])].astype(dtype)
    if not contiguous:
        gdir = np.concatenate([gdir, gdir], axis=1)[:, ::2]
    gval = rng.uniform(0, 1, (goff[-1], 4)).astype('f4')
    eigen = rng.uniform(0.1, 1, (17, 4))
    got = project(goff, gdir, gval, 0.5, eigen, 16, rows_per_block=block)
    levels = np.array([l for l in range(17) for m in range(l + 1)])
    expected = np.zeros_like(got)
    for cell, (a, b) in enumerate(zip(goff[:-1], goff[1:])):
        moments = real_basis(gdir[a:b], 16).T @ gval[a:b].astype('f8')
        expected[cell, :, :, 0] = (moments * 0.5 ** levels[:, None]).T
        expected[cell, :, :, 1] = (moments * eigen[levels]).T
    assert_equivalent(got, expected)
    assert not np.any(got[[3, 11]])


def test_empty_ground_and_poles():
    out = project(np.array([0, 0]), np.empty((0, 3)), np.empty((0, 1)), 0.8, np.ones((33, 1)), 32)
    assert np.isfinite(out).all() and not np.any(out)
    y = real_basis([[0, 0, 1], [0, 0, -1]], 32)
    assert np.isfinite(y).all()
    assert np.all(y[:, [l * (l + 1) // 2 + m for l in range(33) for m in range(1, l + 1)]] == 0)


def test_setting_is_opt_in_and_rejects_bad_order(monkeypatch):
    monkeypatch.delenv('SOSFUSE2_GROUND_SH_L', raising=False)
    assert configured_order() == 0
    for order in (16, 24, 32):
        monkeypatch.setenv('SOSFUSE2_GROUND_SH_L', str(order))
        assert configured_order() == order
    monkeypatch.setenv('SOSFUSE2_GROUND_SH_L', '17')
    with pytest.raises(ValueError, match='E_GROUND_SH_L'):
        configured_order()


def test_synthesized_grid_and_ghosts_against_direct_legendre():
    from sosfuse2.ground_sh import synthesize, GRID_INTERVALS

    d = np.array([[0.3, 0.4, np.sqrt(0.75)]])
    eigen = (0.4 ** np.arange(17))[:, None]
    coeff = project(np.array([0, 1, 1]), d, np.array([[0.7]]), 0.2, eigen, 16)
    grid = synthesize(coeff, 16, np)
    angle = (np.arange(GRID_INTERVALS + 3) - 1) * np.pi / GRID_INTERVALS
    theta, phi = np.meshgrid(angle, angle, indexing='ij')
    views = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1).reshape(-1, 3)
    mirror = d.copy()
    mirror[:, 1] *= -1
    for species, g in enumerate((0.2, 0.4)):
        # This tests synthesis/layout, independently of the finite-L truncation
        # error: compare to the exact degree-16 Legendre phase, not infinite HG.
        phase = (2 * np.arange(17) + 1) * g ** np.arange(17) / (4 * np.pi)
        expected = (
            0.7
            * (
                np.polynomial.legendre.legval(views @ d[0], phase)
                + np.polynomial.legendre.legval(views @ mirror[0], phase)
            )
            / 2
        )
        np.testing.assert_allclose(grid[0, :, species], expected, rtol=3e-6, atol=0)
    assert not np.any(grid[1])


def test_runtime_metadata_separates_opt_in_outputs(monkeypatch):
    from sosfuse2.run_gpu_state_v3 import kernel_settings

    monkeypatch.delenv('SOSFUSE2_KERNEL_FLAGS', raising=False)
    monkeypatch.delenv('SOSFUSE2_BLOCK_SIZE', raising=False)
    monkeypatch.delenv('SOSFUSE2_GROUND_SH_L', raising=False)
    assert kernel_settings() is None
    monkeypatch.setenv('SOSFUSE2_GROUND_SH_L', '16')
    assert kernel_settings()['ground_sh'] == dict(order=16, grid_intervals=32, interpolation='cubic')


def test_angular_cache_shared_across_quadratures_but_not_states(monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from sosfuse2.fused import Kernel

    monkeypatch.setenv('SOSFUSE2_GROUND_SH_L', '16')
    flags = json.loads(paths.CONFIGS.joinpath('C3.json').read_text())['kernel_flags']
    kernel = Kernel(backend='numpy', flags=flags + ['STRATO'], L=16)
    c = {
        name: np.ones(1, 'f4')
        for name in (
            'h',
            'chi',
            'hr',
            'ha',
            'ext',
            'asca',
            'sun_h',
            'sun_col',
            'gm',
            'u',
            'wu',
            'strato_ext',
            'strato_sca',
            'strato_density',
            'strato_sun_col',
        )
    }
    c.update(
        W=1,
        g=0.65,
        goff=np.array([0, 1]),
        gdir=np.array([[0.0, 0.0, 1.0]]),
        gval=np.array([[0.7]]),
        strato_chi=np.ones((33, 1), 'f4'),
    )
    kernel.setup = lambda state, nq: c
    # Replace only the CUDA LUT builder; the real projection/synthesis run on
    # NumPy. A constant per-steradian phase isolates cache ownership from optics.
    kernel.strato_lut_device = lambda chi, waves: np.full(33 + 16385, 1 / (4 * np.pi), 'f4')
    state, other = SimpleNamespace(), SimpleNamespace()
    one = kernel.constants_device(state, 2)['ground_sh']
    two = kernel.constants_device(state, 3)['ground_sh']
    three = kernel.constants_device(other, 2)['ground_sh']
    assert one is two and one is not three
    assert_equivalent(one, three)


def test_disabled_ground_sh_preserves_ground_convolution_tokens(tmp_path):
    testing.require_tool('clang++')
    import json
    import re
    import subprocess
    from pathlib import Path
    from unittest.mock import patch
    from sosfuse2 import sh_layout

    here = paths.PKG  # kernel.cu and configs/ live in the package directory
    flags = json.loads((here / 'configs/C3.json').read_text())['kernel_flags'] + ['PROFILE', 'STRATO']
    try:  # HEAD's kernel.cu of whichever checkout holds the package (sky-sos monorepo or the public repo)
        prefix = subprocess.check_output(
            ['git', '-C', str(here), 'rev-parse', '--show-prefix'], text=True, stderr=subprocess.DEVNULL
        ).strip()
        before = subprocess.check_output(
            ['git', '-C', str(here), 'show', f'HEAD:{prefix}kernel.cu'], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('kernel.cu HEAD baseline needs a git checkout of the package')
    with patch.object(sh_layout.Path, 'read_text', return_value=before):
        previous = sh_layout.kernel_source(16)
    files = [tmp_path / 'before.cu', tmp_path / 'after.cu']
    for p, code in zip(files, [previous, sh_layout.kernel_source(16)]):
        p.write_text(code)
    for phase in (0, 2):
        opts = ['-DFW=15', '-DK_FIRST=1', f'-DK_PHASE={phase}', '-DK_BOUNDARY=0'] + ['-D' + f for f in flags]
        tokens = [
            re.findall(
                r'\w+|[^\s\w]', subprocess.check_output(['clang++', '-E', '-P', '-x', 'c++', *opts, str(p)], text=True)
            )
            for p in files
        ]

        # The numerical tolerance permits unrelated first-light arithmetic/layout
        # changes. This test guards the disabled ground-SH path, not all HEAD tokens.
        def convolution(t):
            assert 'ground_sh_sums' not in t
            assert t.count('gf_sums') == 2  # definition and the direct-convolution call
            start = t.index('gf_sums')
            opening = t.index('{', start)
            depth = 1
            end = opening + 1
            while depth:
                depth += (t[end] == '{') - (t[end] == '}')
                end += 1
            call = t.index('gf_sums', end)
            close = t.index(';', call)
            return t[start:end], t[call:close]

        assert convolution(tokens[0]) == convolution(tokens[1])
