"""Host extension points used by the CPU preparation of a transport solve."""

from functools import partial
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from sosfuse2.atmosphere import density, minor_absorbers as minor, spectral, stratosphere, tropo_profile
from sosfuse2.foundation import content_identity as identity, top
from sosfuse2.jobs.pipeline_execution import PreparedSegments
from sosfuse2.precompute import column_parallel
from sosfuse2.rays import column_table as ct, ray_columns as rays, shell_segments as shell


def geometry(aerosol=None):
    domain = top.Domain()
    aerosol = aerosol if aerosol is not None else tropo_profile.ExponentialProfile(1.4, clamp=True)
    old = NS(
        domain=domain,
        profiles=partial(shell.profiles, aerosol, domain=domain),
        geometry_segments=partial(shell.geometry_segments, aerosol, domain=domain),
        probe=NS(gl=shell.gl, boundaries=partial(shell.boundaries, domain=domain)),
    )
    # Installed by engine.create when its geometry provider supplies a scalar integral.
    if hasattr(rays, '_strato_columns'):
        old._strato_quadrature = old.geometry_segments, rays._strato_columns
    return old


def ray():
    return np.array([[0., 0., 6372.]]), np.array([[0., 0., 1.]])


@pytest.fixture
def optics():
    z = np.arange(161) / 2
    return dict(table=np.stack((np.exp(-z / 9), np.ones_like(z), np.exp(-z / 5)), axis=-1),
                scale=1., mask=3, meta=dict(no2={}), domain=top.Domain())


def record(monkeypatch, owner, name, events):
    impl = getattr(owner, name)

    def observed(*args, **kwargs):
        events.append(name)
        return impl(*args, **kwargs)

    monkeypatch.setattr(owner, name, observed)


def test_public_absorber_dispatch(monkeypatch):
    events = []
    bands = []
    grid = np.array([420., 421.])

    def load(kind, *, required=True):
        events.append(('load', kind, required))
        value = 987. if kind == 'no2' else 654.
        return dict(values=np.full(2, value), wavelength_nm=grid, sha256=kind, raw_sha256='raw')

    def digest(name, raw):
        events.append(('digest', name, raw))
        return raw

    def mean(v, a, b, *_grid):
        bands.append((a, b, v[0]))
        return v[0]

    monkeypatch.setattr(minor, 'load', load)
    monkeypatch.setattr(identity, 'asset_digest', digest)
    monkeypatch.setattr(spectral, 'band_mean', mean)
    assert minor.hashes() == dict(no2='no2', o4='o4')
    assert events == [('load', 'o4', False), ('load', 'no2', True)]
    events.clear()
    assert minor.canonical_hashes(dict(o4='raw', no2='raw')) == dict(no2='no2', o4='o4')
    assert events == [('digest', minor.O4_NAME, 'raw'), ('load', 'o4', False),
                      ('digest', minor.NO2.name, 'raw'), ('load', 'no2', False)]
    events.clear()
    assert minor.representative(None, grid[:1], o4=False)['no2'].item() == 987
    assert events == [('load', 'no2', True)]
    events.clear()

    def canonical(_):
        events.append('canonical')
        return {'no2': 'no2', 'o4': 'o4'}

    monkeypatch.setattr(minor, 'canonical_hashes', canonical)
    values = minor.representative(dict(mode='band', bin_edges_nm=grid), grid[:1])
    assert (values['no2'].item(), values['o4'].item()) == (987., 654.)
    assert events == [('load', 'no2', True), 'canonical', ('load', 'o4', True), 'canonical']
    assert bands == [(420., 421., 987.), (420., 421., 654.)]


@pytest.fixture
def asset(tmp_path, monkeypatch):
    name = minor.NO2.name
    doc = dict(format='no2-fine-v1', wavelength_nm=[420., 421.], no2_220K=[2., 3.],
               units={'no2_220K': minor.UNITS['no2_220K'], 'wavelength_nm': 'nm (air)'},
               source={'name': 'source'}, definition='definition', frozen_source_sha256='original')
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    (tmp_path / 'spectra_fine.json').write_text(json.dumps(dict(wavelength_nm=doc['wavelength_nm'])))
    monkeypatch.setattr(minor, 'NO2', path)
    monkeypatch.setattr(minor.paths, 'ASSETS', tmp_path)
    return path


@pytest.mark.parametrize('policy', ['RULES', 'ASSET_NOTES', 'MINOR_ASSETS', 'ASSET_SUPERSEDED'])
def test_asset_policy_follows_live_identity(asset, monkeypatch, policy):
    first = minor.load('no2')['sha256']
    if policy == 'RULES':
        monkeypatch.setattr(identity, policy, 'different policy')
    elif policy == 'ASSET_NOTES':
        monkeypatch.setitem(identity.ASSET_NOTES, asset.name, ())
    elif policy == 'MINOR_ASSETS':
        monkeypatch.setattr(identity, policy, ())
    else:
        monkeypatch.setitem(identity.ASSET_SUPERSEDED['spectra_fine.json'], 'original', 'changed')
    expected = identity.asset_identity(asset.read_bytes(), asset.name)
    assert first != expected
    assert minor.load('no2')['sha256'] == expected


@pytest.mark.parametrize('owner,name', [
    (identity, 'asset_identity'), (identity, 'asset_digest'), (identity, '_json_document'),
    (identity, '_utf8'), (json, 'loads'), (json, 'dumps'),
])
def test_asset_callbacks_on_warm_reads(asset, monkeypatch, owner, name):
    result = minor.load('no2')
    events = []
    record(monkeypatch, owner, name, events)
    for _ in range(2):
        events.clear()
        match = minor.load('no2')
        assert result['sha256'] == match['sha256']
        # Asset and grid are decoded, then the identity is strictly decoded and serialized.
        assert events == [name] * (3 if name == 'loads' else 1)


def test_asset_bytes_units_and_ownership(asset, monkeypatch):
    first = minor.load('no2')
    first['values'][0] = 11
    first['meta']['units'].clear()
    second = minor.load('no2')
    assert second['values'][0] == 2
    assert second['meta']['units']['no2_220K'] == minor.UNITS['no2_220K']
    doc = json.loads(asset.read_bytes())
    doc['no2_220K'][0] = 23
    asset.write_text(json.dumps(doc))
    assert minor.load('no2')['values'][0] == 23
    monkeypatch.setitem(minor.UNITS, 'no2_220K', 'different')
    with pytest.raises(ValueError, match='E_NO2_ASSET_INVALID'):
        minor.load('no2')
    monkeypatch.setitem(minor.UNITS, 'no2_220K', doc['units']['no2_220K'])
    (asset.parent / 'spectra_fine.json').write_text('{"wavelength_nm":[421,422]}')
    with pytest.raises(ValueError, match='E_NO2_ASSET_INVALID'):
        minor.load('no2')


def test_o4_path_and_error_order(asset, monkeypatch):
    path = asset.parent / minor.O4_NAME
    events = []
    monkeypatch.setattr(minor, 'o4_path', lambda: events.append('o4_path') or path)
    assert minor.load('o4', required=False) is None
    doc = json.loads(asset.read_bytes())
    doc.update(format='o4-fine-v1', o4=[1, 2], generator=dict(rules_version=minor.RULES_VERSION),
               source=dict(sha256=minor.CIA_SHA))
    doc['units']['o4'] = minor.UNITS['o4']
    path.write_text(json.dumps(doc))
    assert minor.load('o4')['values'][0] == 1
    with monkeypatch.context() as m:
        m.setattr(minor, 'CIA_SHA', 'other')
        with pytest.raises(ValueError, match='E_O4_ASSET_STALE'):
            minor.load('o4')
    (asset.parent / 'spectra_fine.json').unlink()
    monkeypatch.setattr(minor, 'RULES_VERSION', 'other')
    monkeypatch.setattr(minor, 'PREPARE', 'prepare again')
    with pytest.raises(ValueError, match='E_O4_ASSET_STALE.*prepare again'):
        minor.load('o4')
    assert events == ['o4_path'] * 4


def test_identity_errors_propagate(asset, monkeypatch):
    def refused(*args):
        raise ValueError('E_IDENTITY_SOURCE: custom reason')

    monkeypatch.setattr(identity, 'asset_identity', refused)
    with pytest.raises(ValueError, match='^E_IDENTITY_SOURCE: custom reason$'):
        minor.load('no2')


@pytest.mark.parametrize('form', ['prepared', 'worker', 'custom'])
def test_strato_geometry_provider(monkeypatch, form):
    monkeypatch.delenv('SOSFUSE2_STRATO_HOST_NOPREFIX', raising=False)
    p, d = ray()
    old = geometry()
    profile = stratosphere.profile_config({})
    if form == 'prepared':
        old = PreparedSegments(old, {0: dict(p=p, d=d)})
    elif form == 'worker':
        old = column_parallel.thaw(column_parallel.freeze(old, ['geometry_segments']))
    else:
        def rows(p, d, n):
            yield np.array([0]), None, np.array([[6392.]]), np.array([[7.]]), None
        old.geometry_segments = rows
    expected = sum((stratosphere.density(r - 6371., profile) * ds).sum()
                   for _, _, r, ds, _ in old.geometry_segments(p, d, 24))
    events = []
    record(monkeypatch, old, 'geometry_segments', events)
    value = stratosphere.columns(p, d, old, profile)
    assert events == ['geometry_segments']
    np.testing.assert_allclose(value, [expected], rtol=1e-14)


@pytest.mark.parametrize('owner,name', [
    (shell, 'boundaries'), (shell, 'gl'), (shell, 'profiles'), (tropo_profile.ExponentialProfile, '__call__'),
    (density, 'no2_profile'), (density, 'no2_asset'), (density, 'require_asset_range'),
    (stratosphere, 'density'),
])
def test_strato_uses_geometry_bindings(monkeypatch, owner, name):
    p, d = ray()
    old = geometry()
    events = []
    record(monkeypatch, owner, name, events)
    profile = stratosphere.profile_config({})
    with monkeypatch.context() as m:
        m.setenv('SOSFUSE2_STRATO_HOST_NOPREFIX', '0')
        expected = stratosphere.columns(p, d, old, profile)
    expected_calls = events.copy()
    assert expected_calls
    events.clear()
    monkeypatch.delenv('SOSFUSE2_STRATO_HOST_NOPREFIX', raising=False)
    actual = stratosphere.columns(p, d, old, profile)
    assert events == expected_calls
    np.testing.assert_allclose(actual, expected, rtol=1e-14)


def test_strato_probe_is_distinct_from_shell_and_explicit_opt_in(monkeypatch):
    p, d = ray()
    old = geometry()
    profile = stratosphere.profile_config({})
    calls = []
    def gl(n):
        calls.append(n)
        u, w = shell.gl(n)
        return u, w * 2

    old.probe.gl = gl
    monkeypatch.setenv('SOSFUSE2_STRATO_HOST_NOPREFIX', '0')
    expected = stratosphere.columns(p, d, old, profile)
    monkeypatch.delenv('SOSFUSE2_STRATO_HOST_NOPREFIX')
    np.testing.assert_allclose(stratosphere.columns(p, d, old, profile), expected, rtol=1e-14)
    assert calls == []
    monkeypatch.setenv('SOSFUSE2_STRATO_HOST_NOPREFIX', '1')
    np.testing.assert_allclose(stratosphere.columns(p, d, old, profile), 2 * expected, rtol=1e-14)
    assert calls == [24]


def test_leggauss_replacement_after_warmup(monkeypatch):
    shell.gl(8)
    calls = []
    def rule(n):
        calls.append(n)
        return np.zeros(n), np.full(n, 2 * len(calls))
    monkeypatch.setattr(np.polynomial.legendre, 'leggauss', rule)
    for i in (1, 2):
        u, w = shell.gl(8)
        np.testing.assert_array_equal(u, 0.5)
        np.testing.assert_array_equal(w, i)
    assert calls == [8, 8]


@pytest.mark.parametrize('backend', ['available', 'unavailable'])
def test_quadrature_callback_once(monkeypatch, backend):
    p, d = ray()
    old = geometry()
    edges = np.array([[0., 2.]])
    if backend == 'unavailable':
        monkeypatch.setattr(rays, '_host_loops', lambda: None, raising=False)
    calls = []
    def rule(n):
        calls.append(n)
        return np.full(n, 0.5), np.full(n, len(calls) / n)
    old.probe.gl = rule
    base, _, _ = rays.integrate(p, d, old, 4, edges=edges, legacy=True)
    assert calls == [4]
    np.testing.assert_allclose(base[0], 2 * old.profiles(1.5), rtol=1e-14)


def test_segment_override_owns_edges_and_order(monkeypatch, optics):
    p, d = ray()
    old = geometry()
    events = []
    def segments(p, d, owner, n, edges):
        assert owner is old and edges is None
        events.append('segments')
        yield np.array([0]), np.array([[6372.]]), np.array([[2.]])
    def boundaries(p, d):
        events.append('boundaries')
        return np.array([[0., 9.]])
    def exact(p, d, end, optics, selected):
        events.append(('exact', end.item(), selected))
        return np.full((1, 4), 19.)
    monkeypatch.setattr(rays, 'segments', segments)
    old.probe.boundaries = boundaries
    monkeypatch.setattr(density, 'exact_profile_columns', exact)
    old.profiles = lambda h: events.append('base') or np.ones((*h.shape, 5))
    monkeypatch.setattr(density, 'molecular_profiles', lambda h, op: events.append('mol') or np.ones((*h.shape, 4)))
    monkeypatch.setattr(stratosphere, 'density', lambda h, cfg: events.append('strato') or np.ones_like(h))
    base, corr, st = rays.integrate(p, d, old, optics=optics, profile={}, legacy=True)
    assert events == ['segments', 'base', 'mol', 'strato', 'boundaries', ('exact', 9., (2, 3))]
    np.testing.assert_array_equal(base, 2)
    np.testing.assert_array_equal(corr, [[2., 2., 19., 19.]])
    np.testing.assert_array_equal(st, 2)


def test_molecular_end_reads_probe_again(monkeypatch, optics):
    old = geometry()
    p, d = ray()
    events = []
    def boundaries(p, d):
        events.append('boundary')
        return np.array([[0., float(len(events))]])
    def rule(n):
        events.append('gl')
        return shell.gl(n)
    def exact(p, d, end, op, selected):
        events.append(('exact', end.item()))
        return np.full((1, 4), end.item())
    old.probe.boundaries = boundaries
    old.probe.gl = rule
    monkeypatch.setattr(density, 'exact_profile_columns', exact)
    _, corr, _ = rays.integrate(p, d, old, 8, optics=optics)
    assert events == ['boundary', 'gl', 'boundary', ('exact', 3.)]
    assert corr[0, 2] == 3.


@pytest.mark.parametrize('owner,name,mode', [
    (shell, 'profiles', 'legacy'), (density, 'no2_asset', 'legacy'),
    (density, 'no2_profile', 'legacy'), (density, 'require_asset_range', 'legacy'),
    (tropo_profile.ExponentialProfile, '__call__', 'legacy'),
    (tropo_profile.TwoLayerProfile, '__call__', 'two'), (tropo_profile, 'profile', 'two'),
    (density, 'molecular_profiles', 'molecular'), (density, 'profiles', 'molecular'),
    (density, 'grid', 'molecular'), (density, 'domain_of', 'molecular'),
    (density, 'check_consumed', 'molecular'), (top, 'profile_n', 'molecular'),
    (np, 'interp', 'legacy'), (np, 'exp', 'molecular'),
])
def test_profile_calls_per_segment(monkeypatch, optics, owner, name, mode):
    p, d = ray()
    events = []
    record(monkeypatch, owner, name, events)
    old = geometry(tropo_profile.TwoLayerProfile(np.array([1., 1., 2., 0., 0.3, 1., 1.]))) if mode == 'two' else geometry()
    edges = np.array([[0., 1., 3.]])
    molecular = mode == 'molecular'
    if name == 'exp':
        optics['mask'] = 0
    # Isolate the segment reduction from the independently tested exact endpoint correction.
    monkeypatch.setattr(density, 'exact_profile_columns', lambda *a, **kw: np.zeros((1, 4)))
    expected = np.zeros((1, 4 if molecular else 5))
    for active, r, ds in rays.segments(p, d, old, 4, edges):
        h = r - 6371.
        v = density.molecular_profiles(h, optics) if molecular else old.profiles(h)
        expected[active] += np.sum(v * ds[..., None], axis=1)
    if molecular:
        expected[:, 2:] = 0
    expected_calls = events.copy()
    events.clear()
    base, corr, _ = rays.integrate(p, d, old, 4, legacy=not molecular, optics=optics if molecular else None, edges=edges)
    assert events == expected_calls
    np.testing.assert_allclose(corr if molecular else base, expected, rtol=1e-14)


@pytest.mark.parametrize('owner,name', [
    (ct, 'molecular'), (ct, '_molecular'), (ct, 'descriptor'), (density, 'profile_identity'),
    (ct, 'exact_columns'), (ct, 'direct_columns'), (ct, 'linear_columns'), (ct, 'pieces'),
    (ct, 'coordinates'), (ct, 'values'), (ct, '_values'), (density, 'grid'),
    (np.polynomial.legendre, 'leggauss'),
])
@pytest.mark.parametrize('selected', [(2, 3), (1, 3, 2, 3)])
def test_exact_selected_protocol(monkeypatch, optics, owner, name, selected):
    p, d = ray()
    end = np.array([1.])
    events = []
    record(monkeypatch, owner, name, events)
    # This ray lies entirely within the profile. The endpoint corrections are zero.
    density.molecular_profiles(np.array([0., 80.]), optics)
    expected = np.zeros((1, 4))
    for k in selected:
        expected[:, k] = ct.exact_columns(ct.molecular(optics, k), p, d, end)
    expected_calls = events.copy()
    events.clear()
    actual = density.exact_profile_columns(p, d, end, optics, selected=selected)
    assert events == expected_calls
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-12)


@pytest.mark.parametrize('name', ['values', '_values', 'grid', 'profile'])
def test_build_table_preserves_rows(monkeypatch, name):
    rows, cols = 9, 7
    monkeypatch.setattr(ct, 'NA', rows)
    monkeypatch.setattr(ct, 'NX', cols)
    if name == 'grid':
        spec = dict(kind='usstd76', values=np.ones(161))
        owner = density
    elif name == 'profile':
        spec = dict(kind='two_layer', vector=[1., 1., 2., 0., .3, 1., 1.])
        owner = tropo_profile
    else:
        spec = dict(kind='exp', height=8.)
        owner = ct
    impl = getattr(owner, name)
    calls = []
    def callback(arg, *args, **kwargs):
        calls.append(np.shape(arg))
        return impl(arg, *args, **kwargs)
    expected = ct.build_table(spec, False)
    monkeypatch.setattr(owner, name, callback)
    actual = ct.build_table(spec, False)
    assert calls == ([()] if name == 'grid' else [(cols - 1, 8)]) * rows
    np.testing.assert_allclose(actual, expected, rtol=1e-14)


def test_main_method_delegates_before_geometry(monkeypatch, optics):
    events = []
    monkeypatch.setattr(ct, 'method', lambda: events.append('method') or 'exact')
    def compute(p, d, old, **kwargs):
        events.append(('integrate', kwargs))
        return 'selected'
    monkeypatch.setattr(ct, 'integrate', compute)
    assert rays.integrate(*ray(), object(), main=True, optics=optics) == 'selected'
    assert events == ['method', ('integrate', dict(legacy=False, optics=optics, profile=None, edges=None))]


def test_squared_shell_calls_profile_on_each_intersection(monkeypatch):
    # Two half-km shells on the outward radial ray. Each stateful value is constant within that shell.
    events = []
    def values(h, spec):
        events.append(h.copy())
        return np.full_like(h, 2 * len(events))
    monkeypatch.setattr(ct, 'values', values)
    p = np.array([[0., 0., ct.R]])
    _, d = ray()
    spec = dict(kind='air_squared', values=np.ones(161))
    result = ct.direct_columns(spec, p, d, np.ones(1))
    assert [v.shape for v in events] == [(1, 8), (1, 8)]
    assert events[0].max() < events[1].min()
    np.testing.assert_allclose(result, [3.])


def test_piece_provider_owns_its_descriptor(monkeypatch):
    calls = []
    def pieces(spec):
        calls.append(spec)
        yield ct.R, ct.R + 2., 7., 0.
    monkeypatch.setattr(ct, 'pieces', pieces)
    spec = {}
    result = ct.linear_columns(spec, np.zeros(1), np.array([ct.R]), np.array([ct.R + 1]))
    assert calls == [spec]
    np.testing.assert_array_equal(result, [7.])


def test_selected_is_lazy_and_each_occurrence_is_evaluated(monkeypatch, optics):
    calls = []
    def selected():
        for k in (2, 3, 3, 2):
            calls.append(('select', k))
            yield k
    def descriptor(op, k):
        calls.append(('spec', k))
        return k
    def exact(k, p, d, end):
        calls.append(('exact', k))
        return np.full(len(d), len(calls))
    monkeypatch.setattr(ct, 'molecular', descriptor)
    monkeypatch.setattr(ct, 'exact_columns', exact)
    result = density.exact_profile_columns(*ray(), np.ones(1), optics, selected())
    assert calls == [(name, k) for k in (2, 3, 3, 2) for name in ('select', 'spec', 'exact')]
    np.testing.assert_array_equal(result, [[0, 0, 12, 9]])


def test_ray_uses_its_fixed_altitude_origin(monkeypatch):
    p, d = ray()
    old = geometry()
    edges = np.array([[0., 1.]])
    monkeypatch.setattr(shell, 'R', 6300.)
    base, _, _ = rays.integrate(p, d, old, 8, legacy=True, edges=edges)
    expected = np.zeros_like(base)
    for active, r, ds in rays.segments(p, d, old, 8, edges):
        expected[active] += np.sum(old.profiles(r - 6371.) * ds[..., None], axis=1)
    np.testing.assert_allclose(base, expected, rtol=1e-13)


def test_molecular_checks_consumed_grid(monkeypatch, optics):
    # The declared table row count is valid in isolation, but cannot stand on this execution grid.
    optics['domain'] = top.Domain(step_km=.25)
    monkeypatch.setattr(density, 'exact_profile_columns', lambda *args, **kwargs: np.zeros((1, 4)))
    with pytest.raises(ValueError, match='E_PROFILE_N_MISMATCH'):
        rays.integrate(*ray(), geometry(), 8, optics=optics, edges=np.array([[0., 1.]]))


def test_domain_factory_lookup_count(monkeypatch, optics):
    p, d = ray()
    old = geometry()
    calls = []
    record(monkeypatch, top, 'Domain', calls)
    monkeypatch.setattr(density, 'exact_profile_columns', lambda *a, **kw: np.zeros((1, 4)))
    edges = np.array([[0., 1., 3.]])
    expected = np.zeros((1, 4))
    for active, r, ds in rays.segments(p, d, old, 4, edges):
        expected[active] += np.sum(density.molecular_profiles(r - 6371., optics) * ds[..., None], axis=1)
    expected[:, 2:] = 0
    expected_calls = calls.copy()
    calls.clear()
    _, corr, _ = rays.integrate(p, d, old, 4, edges=edges, optics=optics)
    assert calls == expected_calls
    np.testing.assert_allclose(corr, expected, rtol=1e-14)


def test_geometry_prefix_validates_a_cold_profile(monkeypatch, tmp_path):
    old = geometry()
    density.no2_asset.cache_clear()
    monkeypatch.setattr(density.paths, 'ASSETS', tmp_path)
    monkeypatch.delenv('SOSFUSE2_STRATO_HOST_NOPREFIX', raising=False)
    try:
        with pytest.raises(FileNotFoundError):
            stratosphere.columns(*ray(), old, stratosphere.profile_config({}), 8)
    finally:
        density.no2_asset.cache_clear()


def test_engine_calls_grid_kernel_then_sun(monkeypatch, tmp_path):
    from sosfuse2.solve import ray_geometry
    from sosfuse2.tests.test_tropo_patches import stopping_engine, Stop

    calls = []
    captured = {}
    engine = stopping_engine(captured, monkeypatch)
    state = ray_geometry.normal_state()
    state['density'] = dict(density.LEGACY)
    state['parameters']['o4'] = False
    def grid(mode, h, chi, nmu, nphi, n):
        calls.append('grid')
        return h, chi, nmu, nphi, n
    def kernel(*args):
        calls.append('kernel')
        return 'custom kernel'
    def sun(old, path, height):
        calls.append('sun')
        return 'custom sun'
    with pytest.raises(Stop):
        engine.create(state, tmp_path, 'selftest', 1, grid=grid, kernel_factory=kernel,
                      sun_geometry=sun, prepare_ground=False)
    assert calls == ['grid', 'kernel', 'sun']
    assert captured['geo'].sun == 'custom sun' and captured['kernel'] == 'custom kernel'


@pytest.mark.parametrize('fn', ['exp', 'sqrt', 'sum', 'maximum', 'minimum', 'cumsum', 'interp', 'linspace'])
def test_table_numpy_operations_observe_original_row_shapes(monkeypatch, fn):
    monkeypatch.setattr(ct, 'NA', 5)
    monkeypatch.setattr(ct, 'NX', 6)
    spec = dict(kind='usstd76', values=np.arange(161, dtype='f8'), end_km=19.)
    if fn in ('exp', 'maximum'):
        spec = dict(kind='exp', height=2., end_km=19.)
    original = getattr(np, fn)
    events = []
    def traced(first, *args, **kwargs):
        events.append((np.shape(first), kwargs.get('axis')))
        return original(first, *args, **kwargs)
    monkeypatch.setattr(np, fn, traced)
    with monkeypatch.context() as m:
        # A user profile requires exactly one ordinary call on each row.
        values = ct.values
        m.setattr(ct, 'values', lambda h, desc: values(h, desc))
        expected = ct.build_table(spec, True)
    trace = events.copy()
    assert trace
    events.clear()
    got = ct.build_table(spec, True)
    assert events == trace
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize('kind', ['exp', 'triangle', 'gaussian', 'two_layer', 'usstd76', 'air_squared'])
@pytest.mark.parametrize('ground', [True, False])
def test_default_tables_match_row_callback(monkeypatch, kind, ground):
    monkeypatch.setattr(ct, 'NA', 21)
    monkeypatch.setattr(ct, 'NX', 26)
    z = np.arange(401) / 2
    spec = dict(kind=kind, domain=dict(top_km=200.), end_km=80., grid=dict(step_km=.5, top_km=200.),
                values=np.exp(-z / 13), height=1.7, center=22., sigma=3., norm=7.,
                vector=[1., 1., 2., 0., .3, 1., 1.])
    got = ct.build_table(spec, ground)
    orig = ct.values
    monkeypatch.setattr(ct, 'values', lambda h, s: orig(h, s))
    expected = ct.build_table(spec, ground)
    np.testing.assert_allclose(got, expected, atol=2e-12, rtol=2e-14)
    np.testing.assert_array_equal(got.astype('f4'), expected.astype('f4'))


@pytest.mark.parametrize('custom', ['aerosol', 'geometry', 'density', 'factory'])
def test_strato_stateful_hook_matches_prefix(monkeypatch, custom):
    old = geometry()
    p, d = ray()
    profile = stratosphere.profile_config({})
    calls = []
    def tapped(fn):
        def wrapped(*args, **kwargs):
            calls.append(len(calls) + 1)
            return fn(*args, **kwargs)
        return wrapped
    if custom == 'aerosol':
        old.geometry_segments = partial(shell.geometry_segments, tapped(old.profiles.args[0]), domain=old.domain)
    elif custom == 'geometry':
        old.geometry_segments = tapped(old.geometry_segments)
    elif custom == 'density':
        def profile_value(h, desc):
            calls.append(len(calls) + 1)
            return np.full_like(h, calls[-1])
        monkeypatch.setattr(stratosphere, 'density', profile_value)
    else:
        monkeypatch.setattr(np.polynomial.legendre, 'leggauss', tapped(np.polynomial.legendre.leggauss))
    with monkeypatch.context() as m:
        m.setenv('SOSFUSE2_STRATO_HOST_NOPREFIX', '0')
        expected = stratosphere.columns(p, d, old, profile)
    expected_calls = list(calls)
    calls.clear()
    monkeypatch.delenv('SOSFUSE2_STRATO_HOST_NOPREFIX', raising=False)
    got = stratosphere.columns(p, d, old, profile)
    assert calls == expected_calls and calls
    np.testing.assert_allclose(got, expected, rtol=1e-13)


def test_paired_walk_respects_live_table_radius_and_domain(monkeypatch, optics):
    monkeypatch.setattr(ct, 'R', 6400.)
    p, d = ray()
    p[0, 2] = ct.R + 1.
    end = np.array([1.])
    got = density.exact_profile_columns(p, d, end, optics, selected=(2, 3))
    exact = ct.exact_columns
    calls = []
    def recorded(spec, *args):
        calls.append(spec['kind'])
        return exact(spec, *args)
    monkeypatch.setattr(ct, 'exact_columns', recorded)
    expected = density.exact_profile_columns(p, d, end, optics, selected=(2, 3))
    assert calls == ['usstd76', 'air_squared']
    np.testing.assert_allclose(got, expected, rtol=2e-12, atol=1e-12)


@pytest.mark.parametrize('api', ['json', 'leggauss'])
def test_factory_replaced_before_consumer_import(monkeypatch, asset, api):
    import importlib.util

    owner, name, consumer = (json, 'loads', minor) if api == 'json' else (np.polynomial.legendre, 'leggauss', shell)
    events = []
    record(monkeypatch, owner, name, events)
    # Load just this consumer into a separate namespace, leaving the running engine intact.
    spec = importlib.util.spec_from_file_location(consumer.__name__ + '_alternate', consumer.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if api == 'json':
        monkeypatch.setattr(module, 'NO2', asset)
        first = module.load('no2')
        events.clear()
        assert module.load('no2')['sha256'] == first['sha256']
        assert events == ['loads'] * 3
    else:
        module.gl(8)
        module.gl(8)
        assert events == ['leggauss', 'leggauss']


@pytest.mark.parametrize('owner,name', [(shell, 'TOP_KM'), (top, 'NORMALIZATION_TOP_KM')])
def test_profile_cutoffs_read_from_active_module(monkeypatch, optics, owner, name):
    p, d = ray()
    p[0, 2] = 6449.
    old = geometry()
    edges = np.array([[0., 5.]])
    monkeypatch.setattr(owner, name, 70.)
    monkeypatch.setattr(density, 'exact_profile_columns', lambda *args, **kwargs: np.zeros((1, 4)))
    for legacy in (True, False):
        expected = np.zeros((1, 5 if legacy else 4))
        for active, r, ds in rays.segments(p, d, old, 8, edges):
            h = r - 6371.
            values = old.profiles(h) if legacy else density.molecular_profiles(h, optics)
            expected[active] += np.sum(values * ds[..., None], axis=1)
        base, corrected, _ = rays.integrate(p, d, old, 8, legacy=legacy, optics=None if legacy else optics, edges=edges)
        if not legacy:
            expected[:, 2:] = 0
        np.testing.assert_allclose(base if legacy else corrected, expected, rtol=1e-14)


def test_numeric_callback_installed_before_shell_import(monkeypatch):
    import importlib.util
    from sosfuse2 import rays as package

    original = np.exp
    calls = []
    def exponential(h):
        calls.append(np.shape(h))
        return 2 * original(h)
    monkeypatch.setattr(np, 'exp', exponential)
    spec = importlib.util.spec_from_file_location(shell.__name__ + '_alternate', shell.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(package, 'shell_segments', module)
    old = geometry()
    old.profiles = partial(module.profiles, old.profiles.args[0], domain=old.domain)
    old.probe = NS(gl=module.gl, boundaries=partial(module.boundaries, domain=old.domain))
    p, d = ray()
    edges = np.array([[0., 2.]])
    expected = np.zeros((1, 5))
    for active, r, ds in rays.segments(p, d, old, 4, edges):
        expected[active] += np.sum(old.profiles(r - 6371.) * ds[..., None], axis=1)
    trace = calls.copy()
    calls.clear()
    base, _, _ = rays.integrate(p, d, old, 4, edges=edges, legacy=True)
    assert calls == trace and calls
    np.testing.assert_allclose(base, expected, rtol=1e-14)


def test_table_accepts_a_profile_array_provider_per_row(monkeypatch):
    monkeypatch.setattr(ct, 'NA', 5)
    monkeypatch.setattr(ct, 'NX', 6)
    calls = []
    class Values:
        def __len__(self):
            return 161

        def __array__(self, dtype=None, copy=None):
            calls.append(len(calls) + 1)
            return np.full(161, float(calls[-1]), dtype=dtype)
    desc = dict(kind='usstd76', values=Values())
    with monkeypatch.context() as m:
        original = ct.values
        m.setattr(ct, 'values', lambda h, spec: original(h, spec))
        expected = ct.build_table(desc, False)
    reference = calls.copy()
    calls.clear()
    actual = ct.build_table(desc, False)
    assert calls == reference == [1, 2, 3, 4, 5]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('limit,step,linear_end,squared_end', [
    (79.8,   0.5,   80.,    79.5),
    (79.875, 0.25,  80.,    79.75),
    (79.625, 0.25,  79.5,   79.5),
    (79.95,  0.125, 80.,    79.875),
    (79.75,  0.125, 79.75,  79.75),
])
def test_exact_component_last_shell(monkeypatch, limit, step, linear_end, squared_end):
    monkeypatch.setattr(top, 'NORMALIZATION_TOP_KM', limit)
    domain = top.Domain(90., step, (90.,), 'us76_neff')
    optics = dict(table=np.ones((domain.n, 3)), scale=1., mask=3, domain=domain,
                  meta=dict(no2={}, profile_step_km=step))
    # Cross the same shells outwards and inwards. Unit profiles make each column a path length.
    p = np.array([[0., 0., ct.R + 79.4], [0., 0., ct.R + 80.]])
    d = np.array([[0., 0., 1.], [0., 0., -1.]])
    end = np.full(2, 0.6)
    lower = density.exact_profile_columns(p, d, end, optics, selected=(2, 3))
    other = density.exact_profile_columns(p, d, end, optics, selected=(0, 1))
    expected = np.array([0.6, linear_end - 79.4, linear_end - 79.4, squared_end - 79.4])
    np.testing.assert_allclose(other + lower, np.tile(expected, (2, 1)), rtol=0, atol=2e-12)


def test_sun_identity_tracks_column_table_source(monkeypatch):
    baseline = json.loads(shell.SunGeometry.identity(1, 1.2, None))
    target = shell.paths.PKG / 'rays/column_table.py'
    file_sha = identity.file_sha
    changed_sha = '0' * 64
    monkeypatch.setattr(identity, 'file_sha', lambda path: changed_sha if path == target else file_sha(path))
    changed = json.loads(shell.SunGeometry.identity(1, 1.2, None))
    assert changed != baseline
    assert changed['sources'][target.name] == changed_sha
    changed['sources'][target.name] = file_sha(target)
    assert changed == baseline
