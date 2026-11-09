"""Per-state contexts and file-level cache dependencies of promoted definitions."""

import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sosfuse2 import column_cache, density, geometry_contract, host_cache, paths, spectral, sun_parallel
from sosfuse2 import geometry as gm
from sosfuse2 import run_gpu_state_v3 as runner


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
    old = scope.b.sh.old
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
    other = scope_b.b.sh.old
    assert old.AER_PROFILE is not other.AER_PROFILE
    assert old.SPEC != other.SPEC
    assert scope.b.sh.OBS == 0.8 and scope_b.b.sh.OBS == 0.2
    for first, after, second in zip(before, values(old), values(other)):
        np.testing.assert_allclose(after, first, rtol=1e-12, atol=0)
        assert not np.allclose(first, second, rtol=1e-6, atol=0)
    assert not np.allclose(sa.af, sb.af)


def test_column_producers_invalidate_caches_but_coefficients_only_sign_products(tmp_path, monkeypatch):
    s, scope = prepare(tmp_path / 'prepare', monkeypatch)
    old = scope.b.sh.old
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


def context_solver(tmp_path, *, prepare_ground=False, state=None):
    from sosfuse2 import tropo_profile
    from sosfuse2.accepted.engine import create

    state = state or gm.normal_state()
    state['parameters']['o4'] = False
    tmp_path.mkdir(parents=True, exist_ok=True)
    return create(
        state,
        tmp_path,
        'selftest',
        1,
        prepare_ground=prepare_ground,
        kernel_factory=lambda *args: None,
        sun_geometry=lambda *args: SimpleNamespace(
            h=np.empty(0),
            x=np.linspace(0, 1, 3),
            col=np.empty((0, 3, 5)),
            lookup=lambda q, r: (np.zeros((*r.shape, 5)), np.ones(r.shape, bool)),
        ),
        aerosol=tropo_profile.engine_hooks(state),
        grid=lambda mode, h, ch, *counts: (h[:1], ch[:4], 1, 2, 4),
    )


def test_scope_mapping_facade(tmp_path):
    import operator
    import warnings
    from collections.abc import Mapping

    import pytest

    s, ctx, _ = context_solver(tmp_path)
    assert s.ctx is ctx and isinstance(ctx, Mapping)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert ctx  # truthiness does not go through the deprecated mapping methods
    keys = ('b', 'Solver', 'basis_at', 'pack', 'C', 'AER_PROFILE', 'AER_HEIGHT', 'AER_G', 'R', 'Z')
    warning = 'sosfuse2 engine scope indexing'
    for key in keys:
        with pytest.warns(DeprecationWarning, match=warning):
            assert ctx[key] == getattr(ctx, key)
    for query, expected in (
        (lambda: 'b' in ctx, True),
        (lambda: 'np' in ctx, False),
        (lambda: ctx.get('b'), ctx.b),
        (lambda: ctx.get('absent', 7), 7),
        (lambda: ctx.get('absent'), None),
        (lambda: tuple(ctx.keys()), keys),
        (lambda: tuple(iter(ctx)), keys),
        (lambda: len(ctx), 10),
    ):
        with pytest.warns(DeprecationWarning, match=warning):
            assert query() == expected
    for key in ('np', 'math', 'time', 'json', 'Path', 'ROOT', 'build', 'SOURCE_PATH', 'HOST_ORIGINAL_GROUND'):
        with pytest.warns(DeprecationWarning, match=warning), pytest.raises(KeyError):
            ctx[key]
    for mutate in (
        lambda: operator.setitem(ctx, 'C', 0),
        lambda: operator.delitem(ctx, 'C'),
        lambda: ctx.pop('C'),
        lambda: ctx.update(C=0),
    ):
        with pytest.raises(TypeError, match='E_ENGINE_SCOPE_READONLY:'):
            mutate()
    with pytest.warns(DeprecationWarning, match=warning):
        other = ctx['Solver'](s.geo, s.kernel, [450.0, 650.0], 1, 0.03)
    assert other.ctx is ctx and other.W == 2


def test_ground_deferred_until_context_enabled(tmp_path):
    first, ctx, _ = context_solver(tmp_path)
    second = ctx.Solver(first.geo, first.kernel, [450.0, 650.0], 1, 0.03)
    for s in (first, second):
        assert all(not hasattr(s, name) for name in ('gm', 'goff', 'gdir', 'gval'))
    ctx.prepare_ground = True
    third = ctx.Solver(first.geo, first.kernel, [550.0], 1, 0.03)
    assert all(hasattr(third, name) for name in ('gm', 'goff', 'gdir', 'gval'))
    first.prepare_ground()
    assert all(hasattr(first, name) for name in ('gm', 'goff', 'gdir', 'gval'))
    assert not hasattr(second, 'gm')


def test_total_columns_replacement_is_late_only_outside_module(tmp_path, monkeypatch):
    import pytest

    from sosfuse2 import solver_columns

    s, ctx, _ = context_solver(tmp_path)
    old = ctx.b.sh.old
    original = old.total_columns

    def replaced(*args):
        raise RuntimeError('replacement reached')

    old.total_columns = replaced
    with pytest.raises(RuntimeError, match='replacement reached'):
        s.prepare_ground()
    with pytest.raises(RuntimeError, match='replacement reached'):
        s.geo.build(0, s)
    with patch.object(solver_columns, 'total_columns', wraps=solver_columns.total_columns) as direct:
        sun = old.SunGeometry(1, tmp_path)
    assert direct.call_count == len(sun.h)
    assert type(sun) is solver_columns.SunGeometry
    old.total_columns = original
    assert ctx.b.sh.NORMAL is old.NORMAL
    normal = ctx.b.sh.NORMAL
    old.NORMAL = dict(normal)
    assert ctx.b.sh.NORMAL is normal and old.NORMAL is not normal


def test_density_attach_restores_columns_on_ground_failure(tmp_path, monkeypatch):
    import pytest

    s, ctx, _ = context_solver(tmp_path)
    s.fuse_state = gm.normal_state()
    s.fuse_state['parameters']['o4'] = False
    old = ctx.b.sh.old
    original = old.total_columns
    monkeypatch.setattr(host_cache, 'enabled', lambda: False)
    monkeypatch.setenv('SOSFUSE2_SUN_METHOD', 'gl')

    def fail():
        assert old.total_columns is not original
        raise RuntimeError('ground failed')

    monkeypatch.setattr(s, 'prepare_ground', fail)
    with pytest.raises(RuntimeError, match='ground failed'):
        density.attach(s, ctx)
    assert old.total_columns is original


def test_geometry_worker_preserves_aerosol_descriptors(tmp_path, monkeypatch):
    from sosfuse2 import column_parallel, column_table, geometry_parallel

    monkeypatch.setattr(geometry_parallel, '_GEOMETRY', None)  # _init below sets worker globals in this process
    monkeypatch.setattr(geometry_parallel, '_SOURCE_DRIFT', None)
    descriptors = []
    for i, (height, two_layer) in enumerate(((1.2, False), (2.4, False), (1.2, True))):
        state = gm.normal_state()
        state['parameters']['scale_height_km'] = height
        if two_layer:
            state['troposphere_profile'] = {'status': 'enabled', 'model': 'two_layer'}
        s, ctx, _ = context_solver(tmp_path / str(i), state=state)
        frozen = geometry_parallel.freeze(SimpleNamespace(scope=ctx))
        geo = SimpleNamespace(h=s.geo.h, chi=s.geo.chi, cfg=s.geo.cfg)
        geometry_parallel._init(frozen, geo, False, None, None, column_parallel.SOURCE_HASHES)
        worker = geometry_parallel._GEOMETRY.scope.b.sh.old
        actual = column_table.aerosol(worker)
        assert actual == column_table.aerosol(ctx.b.sh.old)
        descriptors.append(actual)
    assert descriptors[0] == {'kind': 'exp', 'height': 1.2}
    assert descriptors[1] == {'kind': 'exp', 'height': 2.4}
    assert descriptors[2]['kind'] == 'two_layer' and len(descriptors[2]['vector']) == 7
