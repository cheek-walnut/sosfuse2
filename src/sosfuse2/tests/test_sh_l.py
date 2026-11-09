"""SH truncation invariants, storage and translation-unit compatibility."""

from sosfuse2 import paths
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from sosfuse2 import testing
from sosfuse2.sh_layout import count, padded, workspace, kernel_source
from sosfuse2.arithmetic import basis
from sosfuse2.fused import sh_aux, sh_aux_pair
from sosfuse2.strato_parallel import strato_sh_aux

try:
    from sosfuse2.numerical_checks import assert_equivalent, within_tol
except ImportError:  # standalone research harness
    from numerical_checks import assert_equivalent, within_tol


def test_default_source_is_current_kernel():
    # L=32 must not rewrite the current kernel; historical source bytes need not match.
    assert kernel_source() == (paths.PKG / 'kernel.cu').read_text()


@pytest.mark.parametrize('L', [16, 24, 32])
@pytest.mark.parametrize('combine', [False, True])
def test_rewrite_targets_present(L, combine):
    # Every rewrite counts its targets (sh_layout._replace); the current kernel.cu matches all of them.
    kernel_source(L, combine=combine)


@pytest.mark.parametrize(
    'L,combine,old,new',
    [
        # Reformatting the density factor the combined plane already carries (sh_layout.combined_coefficients) must
        # not leave it in the kernel: the plane would be scaled by ha twice.
        (16, True, 'if(!first)aero*=ha[ir];', 'if(!first) aero*=ha[ir];'),
        (32, True, 'if(!first)aero*=ha[ir];', 'if(!first) aero*=ha[ir];'),
        # A new source-row literal that the L<32 renumbering would rewrite unnoticed.
        (16, False, '#define K 561', '#define K 561\n// row 1132'),
        (24, True, 'if(m<32)', 'if (m<32)'),
    ],
)
def test_rewrite_drift_is_an_error(L, combine, old, new):
    code = (paths.PKG / 'kernel.cu').read_text()
    assert old in code
    with pytest.raises(RuntimeError, match='E_SH_LAYOUT_DRIFT'):
        kernel_source(L, combine=combine, code=code.replace(old, new, 1))


@pytest.mark.parametrize('L', [16, 24, 32])
def test_basis_and_layout(L):
    K = count(L)
    mu = np.linspace(-1, 1, 67)
    assert within_tol(basis(mu, L), basis(mu)[:, :K].copy())
    rng = np.random.default_rng(901)
    full = rng.normal(size=(3, 10 + 2 * K, 15)).astype('f4')
    src = full[:, : 10 + K].copy()
    plain = sh_aux(src, L=L)
    pair = sh_aux_pair(src, L=L)
    assert plain.shape == (3, 15, padded(L))
    assert np.array_equal(plain[:, :, :K], src[:, 10:].transpose(0, 2, 1))
    assert np.all(plain[:, :, K:] == 0)
    both = strato_sh_aux(full, pair=True, L=L)
    assert np.array_equal(both[:3], pair)
    assert np.array_equal(both[3:, :, :K, 0], full[:, 10 + K :, ::2].transpose(0, 2, 1))
    assert np.all(both[:, -1, :, 1] == 0)
    assert workspace(L) >= 162


@pytest.mark.parametrize('L', [16, 24, 32])
def test_lower_source_syntax(tmp_path, L):
    testing.require_tool('clang++')
    from sosfuse2.tests.test_kernel_static import STUB, defines
    from sosfuse2.tests.test_tropo_mie import SYNTAX_STUB_EXTRA
    import json

    flags = json.loads(paths.CONFIGS.joinpath('C3.json').read_text())['kernel_flags'] + ['STRATO', 'PROFILE']
    stub = tmp_path / 'stub.h'
    stub.write_text(STUB + SYNTAX_STUB_EXTRA)
    for combine in (False, True):
        code = tmp_path / 'lower.cpp'
        code.write_text(kernel_source(L, combine=combine))
        if L < 32:
            assert 'y[560]' not in code.read_text()
        for spec in ((1, 0, 0), (0, 0, 1), (0, 1, 0), (0, 2, 0)):
            result = subprocess.run(
                [
                    'clang++',
                    '-std=c++17',
                    '-fsyntax-only',
                    '-x',
                    'c++',
                    '-include',
                    str(stub),
                    *defines(15, flags + (['SH_COMBINE'] if combine else []), spec),
                    str(code),
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('L', [16, 24])
@pytest.mark.parametrize('strato', [False, True])
def test_configured_numpy_transport_matches_mask(tmp_path, monkeypatch, L, strato):
    import json
    from types import SimpleNamespace
    from sosfuse2.run_gpu_state_v3 import settings_for, create_configured
    from sosfuse2.geometry import TransientGeometry
    from sosfuse2 import reference, host_cache

    monkeypatch.setattr(host_cache, 'ROOT', tmp_path / 'host_cache')
    from sosfuse2 import paths

    path = paths.FIXTURES / 'states_dual_strato_usstd.json' if strato else paths.ASSETS / 'states_v1_usstd.json'
    states = json.loads(path.read_text())['states']
    state = next(s for s in states if s['id'] == ('dual_st0p01_aod0p15' if strato else 'normal'))
    settings = tmp_path / 'settings.json'
    settings.write_text(json.dumps(dict(L=L)))
    cfg = settings_for(SimpleNamespace(reduced=True, settings_json=settings))
    s, scope = create_configured(tmp_path / 'state', cfg, np.array([550.0]), 'numpy', state)
    assert len(s.af) == count(L)
    geo = TransientGeometry(s, scope)
    g = geo.cell(12)
    c = s.kernel.setup(s, g['nq'])
    K = count(L)
    rng = np.random.default_rng(907)
    full = rng.uniform(1e-9, 1e-8, (s.geo.ns, 1132 if strato else 571, 1)).astype('f4')
    full[:, 10 + K : 571] = 0
    if strato:
        full[:, 571 + K :] = 0
    small = np.concatenate((full[:, : 10 + K], full[:, 571 : 571 + K]), axis=1) if strato else full[:, : 10 + K].copy()
    x = reference.transport(g, c, small)
    y = reference.transport(g, dict(c, L=32), full)
    assert within_tol(x, y)
    if strato:
        combined = reference.transport(g, dict(c, sh_combine=True), small)
        assert np.all(abs(combined - x) <= 1e-5 * abs(x))


@pytest.mark.parametrize('nh', [3, 33, 44, 65])
def test_combine_species_masks_and_source_preservation(nh):
    from sosfuse2.sh_layout import combined_coefficients, combined_aux

    K = count(16)
    src = np.ones((nh * 2, 10 + 2 * K, 1), 'f4')
    src[:, 10 + K :] = 2
    saved = src.copy()
    tau = np.float32(1e-7)
    above = np.nextafter(tau, np.float32(1))
    ha = np.resize(np.array([above, tau, 0], 'f4'), nh)
    hs = ha[::-1].copy()
    c = dict(ha=ha, hr=np.ones(nh, 'f4'), strato_density=hs, chi=np.zeros(2))
    got = combined_coefficients(src, c, 16, skip_thin=True)
    expected = np.repeat((ha * (ha > tau) + 2 * hs * (hs > tau)) if nh <= 64 else ha + 2 * hs, 2)
    assert np.array_equal(got[:, :, 0], np.broadcast_to(expected[:, None], (2 * nh, K)))
    aux = combined_aux(src, c, 16, skip_thin=True)
    assert aux.shape == (2 * nh, 1, padded(16), 2)
    assert np.array_equal(aux[:, 0, :K, 0], got[:, :, 0]) and np.all(aux[:, :, :, 1] == 0)
    assert np.array_equal(src, saved)
