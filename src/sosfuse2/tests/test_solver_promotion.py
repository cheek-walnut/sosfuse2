"""Per-state globals and file-level cache dependencies of promoted definitions."""

import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sosfuse2 import column_cache, density, geometry as gm, geometry_contract, host_cache, paths
from sosfuse2 import run_gpu_state_v3 as runner, spectral, sun_parallel


def prepare(tmp_path, monkeypatch, state=None, cfg=None):
    monkeypatch.setattr(runner, 'SUN_SCALE', 1)
    for key in ('SUN', 'GROUND', 'GEOMETRY'):
        monkeypatch.setenv('SOSFUSE2_' + key + '_WORKERS', '1')
    cfg = cfg or runner.settings_for(SimpleNamespace(reduced=True, settings_json=None))
    return runner.create_configured(tmp_path, cfg, cfg['wavelengths_nm'], 'numpy', state or gm.normal_state())


def test_live_state_survives_other_height_phase_observer_and_spectrum(tmp_path, monkeypatch):
    a = gm.normal_state()
    a['parameters'].update(scale_height_km=2.4, g=0.65)
    cfg = runner.settings_for(SimpleNamespace(reduced=True, settings_json=None))
    cfg['observer_km'] = 0.8
    sa, scope = prepare(tmp_path / 'a', monkeypatch, a, cfg)
    old = scope['b'].sh.old
    h = np.array([0.1, 2.0, 8.0, 25.0])
    p = np.array([[0.0, 0.0, 6371.8]])
    d = np.array([[0.0, 0.0, 1.0]])

    def values(o):
        return o.profiles(h), o.coefficients([550.0], o.NORMAL), o.total_columns(p, d, 24)

    before = values(old)
    b = copy.deepcopy(a)
    b['parameters'].update(scale_height_km=1.7, g=0.9)
    cfg_b = dict(cfg, observer_km=0.2, wavelengths=2, wavelengths_nm=[477.5, 712.5])
    cfg_b['spectral'] = spectral.block(
        dict(spectral_mode='band', bin_edges_nm=[360.0, 595.0, 830.0]), np.array(cfg_b['wavelengths_nm']), True
    )
    sb, scope_b = prepare(tmp_path / 'b', monkeypatch, b, cfg_b)
    other = scope_b['b'].sh.old
    assert old.profiles.__globals__ is not other.profiles.__globals__
    assert old.SPEC != other.SPEC
    assert scope['b'].sh.OBS == 0.8 and scope_b['b'].sh.OBS == 0.2
    for first, after, second in zip(before, values(old), values(other)):
        np.testing.assert_allclose(after, first, rtol=1e-12, atol=0)
        assert not np.allclose(first, second, rtol=1e-6, atol=0)
    assert not np.allclose(sa.af, sb.af)


def test_column_producers_invalidate_caches_but_coefficients_only_sign_products(tmp_path, monkeypatch):
    s, scope = prepare(tmp_path / 'prepare', monkeypatch)
    old = scope['b'].sh.old
    optics = density.build(
        density.with_profile(
            {}, air_surface_number_density_m3=2.5469e25, z_km=density.grid(161), table=np.ones((161, 2)), mask=3
        )
    )
    bundle = gm.Bundle([gm.TransientGeometry(s, scope).cell(11)], np)
    c = dict(strato_old=old, strato_profile=dict(center=21.0, sigma=3.0, norm=1.0))
    actual_read = Path.read_bytes

    def keys():
        # Capture the real shared-sun folder selected by sun_geometry, without rebuilding a solar table per mutation.
        folders = []

        def sun(scale, folder):
            folders.append(folder)
            return SimpleNamespace(h=np.empty(81), x=np.empty(401))

        with (
            patch.object(old, 'SunGeometry', side_effect=sun),
            patch.object(sun_parallel, 'prebuild', return_value=None),
        ):
            host_cache.sun_geometry(old, 1, tmp_path / 'local', 1.2, shared=tmp_path / 'shared')
        return dict(
            geometry=geometry_contract.identity(s)[1],
            legacy_geometry=gm.identity(s, 2)[1],
            host=host_cache.digest(host_cache.context(s, optics)[1]),
            sun=folders[0],
            column=column_cache.cache_key(s.kernel, bundle, c, 'strato'),
            product=runner.package_hashes(),
        )

    original = keys()
    for name, invalidates in (('solver_columns.py', True), ('solver_optics.py', False)):
        target = paths.PKG / name
        with patch.object(Path, 'read_bytes', lambda p: actual_read(p) + (b'\n# mutation\n' if p == target else b'')):
            changed = keys()
        for cache in ('geometry', 'legacy_geometry', 'host', 'sun', 'column'):
            assert (changed[cache] != original[cache]) is invalidates, (name, cache)
        assert changed['product'] != original['product']
