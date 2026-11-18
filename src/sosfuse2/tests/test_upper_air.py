"""Upper air asset, coefficient continuity and a CUDA reference comparison."""

import hashlib
import json

import numpy as np
import pytest

from sosfuse2.atmosphere import density, rayleigh, upper_air
from sosfuse2.foundation import top, content_identity
from sosfuse2.tests.test_extended_top import config, state, small_solver, serial  # noqa: F401


def test_upper_asset_and_default_settings(tmp_path):
    data, z, r = upper_air.load()
    meta = data['metadata']
    assert z.shape == r.shape == (961,) and (z[0], z[-1]) == (80., 200.)
    assert r[0] == 1 and np.all(np.diff(r) < 0)
    assert meta['source_version'] == '0.3.4' and meta['reference_wavelength_nm'] == 550.
    for h, value in ((100, .971), (120, .857), (200, .543)):
        assert abs(meta['f_at_km'][str(h)] - value) <= .002
    assert np.isfinite(meta['f_80']) and meta['f_80'] > 0
    assert meta['factor_spectral_max_rel'] < .002
    assert config(tmp_path, dict(top_km=120))['upper_air'] == 'us76_neff'
    assert config(tmp_path, dict(top_km=120, upper_air='us76_neff'))['upper_air'] == 'us76_neff'
    assert upper_air.ASSET_NAME in content_identity.DATA_ASSETS
    assert content_identity.package_hashes()['sosfuse2/assets/' + upper_air.ASSET_NAME] == upper_air.ASSET_SHA256
    with pytest.raises(ValueError, match='E_ASSET_RANGE'):
        upper_air.sample([200.125])
    with pytest.raises(ValueError, match='E_PROFILE_GRID'):
        upper_air.sample([80.01])


@pytest.mark.parametrize('step', [.5, .25, .125])
def test_upper_connection_and_lower_columns(tmp_path, step):
    from sosfuse2.backends import numpy_reference as nr

    d = top.resolve(dict(top_km=120), step)
    s, ctx = small_solver(tmp_path, upper_air='us76_neff', step=step)
    opt = s.density_optics
    base = density.build(state(step))
    nlower = len(base['table'])
    np.testing.assert_allclose(opt['table'][:nlower], base['table'], rtol=1e-6, atol=0)
    assert opt['scale'] == base['scale']
    lower_ext = ctx.b.sh.old.coefficients(s.wl, ctx.b.sh.NORMAL)
    np.testing.assert_allclose(s.ext[0], lower_ext[0] * base['scale'], rtol=1e-6, atol=0)
    np.testing.assert_allclose(s.ext[1:], lower_ext[1:], rtol=1e-6, atol=0)
    z = np.arange(nlower) * step
    np.testing.assert_allclose(np.trapezoid(opt['table'][:nlower], z, axis=0),
                               np.trapezoid(base['table'], z, axis=0), rtol=1e-6, atol=0)
    ratios, _ = upper_air.sample(np.arange(nlower - 1, d.n) * step)
    # The engine's final coefficient includes scale once, and the lower boundary is the join.
    boundary = base['table'][-1, 0]
    np.testing.assert_allclose(opt['table'][nlower - 1:, 0], boundary * ratios, rtol=1e-12, atol=0)
    assert np.all(opt['table'][nlower:, 1:] == 0)
    samples = np.array([np.nextafter(np.float32(80), np.float32(-np.inf)), 80.,
                        np.nextafter(np.float32(80), np.float32(np.inf)), 80.25, 120.], 'f4')
    profiles = density.interpolate32(samples, opt['table'], d)
    beta = profiles[:, 0] * s.ext[0, 0]
    np.testing.assert_allclose(beta[2] / beta[0], 1., rtol=1e-4, atol=0)
    assert beta[-1] > 0 and np.all(profiles[2:, 1:] == 0)
    c = dict(domain=d, density_optics=opt, height=np.float32(1.2))
    assembled = nr.profile_at(samples, c)
    assert np.all(assembled[2:, 1:] == 0)  # includes O4 derived from the extended air
    assert np.all(assembled[2:, 0] > 0)
    m = density.metadata(s)['upper_air_connection']
    actual = float(s.ext[0, 0] * boundary)
    np.testing.assert_allclose(m['beta_rayleigh_80_minus_per_km'], [actual], rtol=1e-12)
    absolute = rayleigh.sigma(rayleigh.vacuum_nm(s.wl)) * 1e-4 * m['n_tot_80_m3'] * 1e3
    np.testing.assert_allclose(m['absolute_us76_ratio'], actual / absolute, rtol=1e-12)
    assert m['normalization_scale'] == base['scale']
    _, _, lower_air, _ = density.load_asset()
    assert m['lower_profile_normalization_factor'] == pytest.approx(boundary / lower_air[-1])
    assert d.identity()['upper_asset_sha256'] == upper_air.ASSET_SHA256
    # Asset contents affect data identity, but not compiled machine code.
    from unittest.mock import patch
    from sosfuse2.solve.source_solver import Kernel
    definitions = d.defines()
    compile_key = Kernel(domain=d).kernel_key((), 1, 1, 0, 0, 0)
    profile_key = density.profile_identity(opt)['sha256']
    with patch.object(top, 'UPPER_AIR_ASSET_SHA256', 'different-asset'):
        assert d.identity()['upper_asset_sha256'] != m['asset_sha256']
        assert d.defines() == definitions
        assert Kernel(domain=d).kernel_key((), 1, 1, 0, 0, 0) == compile_key
        assert density.profile_identity(opt)['sha256'] != profile_key
    (tmp_path / 'connection.json').write_text(json.dumps(m, indent=2))


def reference_tables(path, backend='numpy', width=1):
    """Small 120 km grid with observed first order and three fixed apply evaluations."""
    from sosfuse2.solve import source_solver, ray_geometry
    from sosfuse2.atmosphere.stratosphere import source_size
    from sosfuse2.outputs import radiance_table

    wavelengths = np.linspace(450., 650., width) if width > 1 else None
    s, ctx = small_solver(path, upper_air='us76_neff', backend=backend, wavelengths=wavelengths)
    wrapper = source_solver.Solver(s, ray_geometry.TransientGeometry(s, ctx), prepared=True)
    x = s.kernel.xp.zeros((s.geo.ns, source_size(s), s.W), 'f4')
    values = []
    for iteration in range(4):
        table, _ = radiance_table.make_table(s, s.kernel.host(x), [-3., -6.], [10., 45.], [0., 180.], log=lambda _: None)
        values.append(table)
        if iteration < 3:
            x = wrapper.apply(x)
    if backend == 'cupy' and width == 15:
        assert wrapper.timings['ksplit']['effective'] == 3
    return np.array(values)


@pytest.mark.parametrize('width', [1, 15])
def test_numpy_upper_reference(tmp_path, width):
    value = reference_tables(tmp_path / 'prepare', width=width)
    assert value.shape == (4, 2, 2, 2, width)
    assert np.isfinite(value).all() and np.all(value > 0)
    np.savez(tmp_path / 'numpy_reference.npz', radiance=value)
    (tmp_path / 'reference.json').write_text(json.dumps(dict(
        domain=top.resolve(dict(top_km=120)).identity(), shape=list(value.shape),
        min=float(value.min()), max=float(value.max()),
        array_sha256=hashlib.sha256(value.tobytes()).hexdigest())))


@pytest.mark.parametrize('width', [1, 15])
def test_cuda_upper_reference(tmp_path, monkeypatch, width):
    cupy = pytest.importorskip('cupy', reason='CuPy is unavailable')
    if not cupy.is_available():
        pytest.skip('CUDA device is unavailable')
    monkeypatch.delenv('SOSFUSE2_KSPLIT', raising=False)
    reference = reference_tables(tmp_path / 'numpy', width=width)
    actual = reference_tables(tmp_path / 'cuda', 'cupy', width)
    np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=0)
    rel = np.abs(actual - reference) / np.maximum(np.abs(reference), 1e-30)
    (tmp_path / 'cuda_comparison.json').write_text(json.dumps(dict(max_rel=float(rel.max()),
        location=[int(i) for i in np.unravel_index(np.argmax(rel), rel.shape)])))
