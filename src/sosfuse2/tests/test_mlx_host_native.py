"""Cross-check native host quadrature against the no-compiler NumPy path."""

from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from sosfuse2.atmosphere import density, stratosphere, tropo_profile
from sosfuse2.foundation import top
from sosfuse2.precompute import sun_parallel
from sosfuse2.rays import column_table as ct, ray_columns as rays, shell_segments as shell


def model(domain):
    aerosol = tropo_profile.ExponentialProfile(1.2, clamp=True)
    geometry = partial(shell.geometry_segments, aerosol, domain=domain)
    old = SimpleNamespace(domain=domain, profiles=partial(shell.profiles, aerosol, domain=domain),
                          probe=SimpleNamespace(gl=shell.gl, boundaries=partial(shell.boundaries, domain=domain)),
                          geometry_segments=geometry, _strato_quadrature=(geometry, rays._strato_columns))
    z = np.arange(domain.n) * domain.step_km
    optics = dict(domain=domain, mask=3, scale=1., meta=dict(no2={}),
                  table=np.column_stack((1 + z / 100, 2 - z / 150, .3 + z / 400)))
    return old, optics


def native():
    import shutil
    if not any(shutil.which(cc) for cc in ('cc', 'gcc', 'clang')):
        pytest.skip('a compiler is required for the native comparison')
    lib = density._host_native()
    assert lib is not None
    return lib


def no_compiler(monkeypatch, root):
    monkeypatch.setenv('SOSFUSE2_WORK_ROOT', str(root))
    monkeypatch.setenv('PATH', str(root / 'missing-bin'))
    with pytest.warns(RuntimeWarning, match='E_HOST_NATIVE_COMPILER_MISSING'):
        assert density._host_native() is None


@pytest.mark.parametrize('top_km,limit,upper', [
    (80., 80., 'vacuum'), (80., 79.8, 'us76_neff'), (200., 79.8, 'vacuum'), (200., 80., 'us76_neff'),
])
def test_sun_segments_exact_and_layer(monkeypatch, tmp_path, top_km, limit, upper):
    lib = native()
    monkeypatch.setattr(top, 'NORMALIZATION_TOP_KM', limit)
    monkeypatch.setenv(sun_parallel.ENV, '1')
    domain = top.Domain(top_km, .5, (top_km,), upper)
    old, optics = model(domain)
    h = np.array([0., .1, 13., limit - .1, limit, top_km])
    x = np.linspace(0, 1, 14)
    layer = dict(center=21., sigma=5., norm=3., end_km=limit)
    sun = SimpleNamespace(h=h, x=x, col=np.zeros((len(h), len(x), 5)))
    solver = SimpleNamespace(geo=SimpleNamespace(sun=sun))
    # Include both signs of each near-tangent ray and both sides of the cutoff.
    angle = np.deg2rad([0., 27., 49., 88., 133.])
    tilt = np.deg2rad([0., 30., 45., 90.])
    rr = shell.R + np.array([limit, top_km])
    radius, a, b = (v.ravel() for v in np.meshgrid(
        np.r_[np.nextafter(rr, -np.inf), rr, np.nextafter(rr, np.inf)], angle, tilt))
    point = radius[:, None] * np.column_stack((np.sin(a)*np.cos(b), np.sin(b), np.cos(a)*np.cos(b)))
    d = np.column_stack((np.cos(a), np.zeros_like(a), -np.sin(a)))
    p, d = np.concatenate((point, point)), np.concatenate((d, -d))
    edges = np.tile([0., 1e-6], (len(p), 1))
    def calculate():
        base = sun_parallel.rows(old, h, x, shell.R, 32)
        corrected = sun_parallel.rows(old, h, x, shell.R, 32, base, optics)
        return (base, corrected, stratosphere.solar_grid(solver, old, layer),
                rays.integrate(p, d, old, optics=optics, edges=edges)[1],
                stratosphere.columns(p, d, old, layer),
                rays.integrate(p, d, old, profile=layer, edges=edges)[2],
                density.exact_profile_columns(p, d, edges[:, -1], optics, selected=(2, 3)))
    with monkeypatch.context() as observed:
        calls = Mock(wraps=lib)
        observed.setattr(density, '_host_native', lambda: calls)
        accelerated = calculate()
        assert calls.sun_rays_c.called and calls.profile_columns.called and calls.squared_columns.called
    no_compiler(monkeypatch, tmp_path / 'fallback')
    for got, reference in zip(accelerated, calculate()):
        np.testing.assert_allclose(got, reference, rtol=1e-12, atol=0.)


@pytest.mark.parametrize('ground', [True, False])
def test_profile_tables(monkeypatch, tmp_path, ground):
    native()
    monkeypatch.setattr(ct, 'NA', 19)
    monkeypatch.setattr(ct, 'NX', 27)
    z = np.arange(401) / 2
    specs = [dict(kind=k, domain=dict(top_km=200.), end_km=79.8, grid=dict(step_km=.5, top_km=200.),
                  values=np.exp(-z / 13), height=1.7, center=22., sigma=7., norm=3.,
                  vector=[1., 1., 2., .1, .3, 1., 1.])
             for k in ('exp', 'triangle', 'gaussian', 'two_layer', 'usstd76', 'air_squared')]
    with monkeypatch.context() as observed:
        calls = Mock(wraps=native())
        observed.setattr(density, '_host_native', lambda: calls)
        actual = [ct.build_table(spec, ground) for spec in specs]
        assert calls.build_table_c.call_count == len(specs)
    no_compiler(monkeypatch, tmp_path / 'fallback')
    for got, spec in zip(actual, specs):
        np.testing.assert_allclose(got, ct.build_table(spec, ground), rtol=1e-12, atol=0.)


@pytest.mark.parametrize('setting', ['source', 'flags'])
def test_native_keys(monkeypatch, setting):
    def keys():
        spec = ct.descriptor('exp', height=8.)
        return (Path(native()._name), ct.table_key(spec)['source'], shell.SunGeometry.identity(1, 1.2, None))
    before = keys()
    if setting == 'source':
        monkeypatch.setattr(density, '_HOST_NATIVE_SRC', density._HOST_NATIVE_SRC + '\n/* distinct input */\n')
    else:
        monkeypatch.setattr(density, '_HOST_NATIVE_CFLAGS', (*density._HOST_NATIVE_CFLAGS, '-DSOS_TEST=1'))
    after = keys()
    assert all(a != b for a, b in zip(before, after))
    assert before[0].parent == after[0].parent


@pytest.mark.parametrize('bad', [np.empty(4, 'f4'), np.empty(8)[::2], np.empty(3), np.empty((2, 2))])
def test_reject_wrong_native_buffer(bad):
    with pytest.raises(ValueError, match='E_HOST_NATIVE_BUFFER'):
        density._host_buffer(bad, (4,))


def test_disk_tables_invalidate_after_source_edit(tmp_path, monkeypatch):
    monkeypatch.setenv('SOSFUSE2_COLUMN_TABLE_CACHE', str(tmp_path))
    monkeypatch.setattr(ct, 'NA', 4)
    monkeypatch.setattr(ct, 'NX', 4)
    build = Mock(side_effect=lambda *_: np.full((4, 4), float(build.call_count)))
    monkeypatch.setattr(ct, 'build_table', build)
    monkeypatch.setattr(ct, '_MEMO', ct.OrderedDict())
    spec = ct.descriptor('exp', height=8.)
    first = ct.tables(spec)['ground']
    ct._MEMO.clear()
    np.testing.assert_array_equal(ct.tables(spec)['ground'], first)
    assert build.call_count == 2
    monkeypatch.setattr(density, '_HOST_NATIVE_SRC', density._HOST_NATIVE_SRC + '\n/* new source */\n')
    ct._MEMO.clear()
    assert not np.array_equal(ct.tables(spec)['ground'], first)
    assert build.call_count == 4


@pytest.mark.parametrize('which', ['build', 'load', 'work_root'])
def test_optional_library_failure(tmp_path, monkeypatch, which):
    import subprocess
    root = tmp_path / 'work'
    monkeypatch.setenv('SOSFUSE2_WORK_ROOT', str(root))
    if which == 'work_root':
        root.write_text('occupied')
    elif which == 'build':
        fail = subprocess.CalledProcessError(7, 'cc', stderr='compiler diagnostic')
        monkeypatch.setattr(subprocess, 'run', Mock(side_effect=fail))
        monkeypatch.setattr(density.shutil, 'which', lambda _: 'cc')
    else:
        def invalid_image(command, **kwargs):
            Path(command[command.index('-o') + 1]).write_bytes(b'invalid image')
        monkeypatch.setattr(subprocess, 'run', invalid_image)
        monkeypatch.setattr(density.shutil, 'which', lambda _: 'cc')
    monkeypatch.setattr(ct, 'NA', 5)
    monkeypatch.setattr(ct, 'NX', 9)
    reason = 'E_HOST_NATIVE_' + which.upper()
    if which == 'build':
        reason += '.*compiler diagnostic'
    with pytest.warns(RuntimeWarning, match=reason):
        table = ct.build_table(ct.descriptor('exp', height=8.), False)
    assert density._host_native() is None
    with monkeypatch.context() as m:
        values = ct.values
        m.setattr(ct, 'values', lambda h, spec: values(h, spec))
        np.testing.assert_array_equal(table, ct.build_table(ct.descriptor('exp', height=8.), False))


def test_native_guards_reject_mismatched_geometry(monkeypatch):
    old, _ = model(top.Domain())
    p = np.tile([0., 0., ct.R], (2, 1))
    d = np.tile([1., 0., 0.], (2, 1))
    with pytest.raises(ValueError, match='E_COLUMN_RAYS'):
        rays.integrate(p, d, old, legacy=True, edges=np.zeros((1, 2)))
    sun = SimpleNamespace(h=np.arange(2.), x=np.linspace(0, 1, 3), col=np.zeros((1, 3, 5)))
    with pytest.raises(ValueError, match='E_SUN_GRID'):
        stratosphere.solar_grid(SimpleNamespace(geo=SimpleNamespace(sun=sun)), old, stratosphere.profile_config({}))


def test_profile_grid_buffer_bounds(monkeypatch):
    monkeypatch.setattr(ct, 'NA', 4)
    monkeypatch.setattr(ct, 'NX', 4)
    spec = dict(kind='usstd76', values=np.arange(161.), grid=dict(n=161, step_km=.5, top_km=80.))
    for bad in (dict(spec, values=np.arange(160.)), dict(spec, values=np.ones((161,1))),
                dict(spec, grid=dict(n=160, step_km=.5, top_km=80.)),
                dict(spec, grid=dict(n=161, step_km=0., top_km=80.))):
        with pytest.raises(ValueError, match='E_(PROFILE|HOST_NATIVE_BUFFER)'):
            ct.build_table(bad, True)


@pytest.mark.parametrize('accelerated', [False, True], ids=['numpy', 'native'])
@pytest.mark.parametrize('kind', ['usstd76', 'air_squared'])
def test_profile_support_beyond_grid(monkeypatch, tmp_path, accelerated, kind):
    monkeypatch.setattr(ct, 'NA', 12)
    monkeypatch.setattr(ct, 'NX', 20)
    if accelerated:
        calls = Mock(wraps=native())
        monkeypatch.setattr(density, '_host_native', lambda: calls)
    else:
        no_compiler(monkeypatch, tmp_path)
    spec = dict(kind=kind, values=np.ones(161), grid=dict(n=161, step_km=.5, top_km=80.),
                domain=dict(top_km=200.), end_km=120.)
    for ground in (True, False):
        table = ct.build_table(spec, ground)
        assert table[0, -1] == pytest.approx(1242.3526069518266, rel=1e-12, abs=0.)
        # Beyond the last sample np.interp holds its value, including for a nonzero last slope.
        values = np.linspace(.25, 1., 161)
        extended = dict(spec, values=np.pad(values, (0, 80), mode='edge'),
                        grid=dict(n=241, step_km=.5, top_km=120.))
        np.testing.assert_allclose(ct.build_table(dict(spec, values=values), ground),
                                   ct.build_table(extended, ground), rtol=1e-12, atol=0.)
    if accelerated:
        assert calls.build_table_c.call_count == 6


def test_sun_identity_uses_integration_shells(tmp_path, monkeypatch):
    class ExtraShells(top.Domain):
        @property
        def extra_shells(self):
            return (*super().extra_shells, .05, .5, 2.)

    identity = partial(shell.SunGeometry.identity, 1, 1.2, None)
    default = identity(domain=top.Domain())
    assert default == identity() == identity(domain=top.Domain())
    modified = ExtraShells()
    assert identity(domain=modified) != default
    monkeypatch.setattr(shell, 'sun_grid', lambda *_: (np.array([78.0125]), np.array([0., .0025])))
    build = partial(shell.SunGeometry, 1, tmp_path, aerosol=tropo_profile.ExponentialProfile(1.2, clamp=True),
                    aerosol_height=1.2, descriptor=None)
    first = build(domain=top.Domain())
    assert build(domain=top.Domain()).build_seconds == 0.
    with pytest.warns(RuntimeWarning, match='SUN_TABLE_STALE'):
        changed = build(domain=modified)
    assert abs(changed.col[0, 1, 3] - first.col[0, 1, 3]) > 1e-6
    assert build(domain=ExtraShells()).build_seconds == 0.


def test_rebuild_unloadable_library(tmp_path, monkeypatch):
    lib = native()
    monkeypatch.setenv('SOSFUSE2_WORK_ROOT', str(tmp_path))
    target = tmp_path / 'build' / Path(lib._name).name
    target.parent.mkdir()
    target.write_bytes(b'incomplete library')
    with pytest.warns(RuntimeWarning, match='E_HOST_NATIVE_LOAD'):
        recovered = density._host_native()
    assert recovered is not None
    assert density._host_native() is recovered
    assert density.ctypes.CDLL(str(target)).build_table_c


def test_tiny_columns_with_absolute_floors(tmp_path, monkeypatch):
    native()
    monkeypatch.setattr(ct, 'NA', 12)
    monkeypatch.setattr(ct, 'NX', 20)
    gaussian = dict(kind='gaussian', center=42., sigma=1., norm=3.)
    squared = dict(kind='air_squared', grid=dict(step_km=.5, top_km=80.), end_km=80.,
                   values=np.r_[np.ones(159), 1e-12, 1.])
    p = np.array([[0., 0., ct.R + 79.5]])
    mu = np.array([-.002, -.0005, .0005, .002])
    d = np.column_stack((np.sqrt(1 - mu * mu), np.zeros_like(mu), mu))
    length = np.full(len(mu), 1e-6)
    o4 = ct.direct_columns(squared, p, d, length)
    table = ct.build_table(gaussian, True)
    no_compiler(monkeypatch, tmp_path)
    o4_reference = ct.direct_columns(squared, p, d, length)
    table_reference = ct.build_table(gaussian, True)
    assert np.all((o4_reference > 1e-25) & (o4_reference < 1e-23))
    assert np.any((table_reference > 0.) & (table_reference < np.finfo('f8').tiny))
    # sqrt and hypot can round differently on these short paths. At the bottom of
    # the Gaussian tail a single float64 step is already a large relative change.
    np.testing.assert_allclose(o4, o4_reference, rtol=1e-12, atol=2e-34)
    np.testing.assert_allclose(table, table_reference, rtol=1e-12, atol=np.nextafter(0., 1.))


def test_grazing_rows_keep_scalar_rounding(monkeypatch, tmp_path):
    native()
    monkeypatch.setenv(sun_parallel.ENV, '1')
    monkeypatch.setattr(top, 'NORMALIZATION_TOP_KM', 79.8)
    domain = top.Domain(200., .5, (200.,), 'us76_neff')
    old, optics = model(domain)
    z, no2, _ = density.no2_asset()
    upper_z = np.arange(domain.n)*domain.step_km
    optics['table'][:, 2] = np.interp(upper_z, z, no2)
    # At this height scalar ratio**2 and ratio*ratio differ by an ulp.
    heights = np.array([58.327126961464046])
    x = np.linspace(0, 1, 80)
    def rows():
        base = sun_parallel.rows(old, heights, x, ct.R, 32)
        return base, sun_parallel.rows(old, heights, x, ct.R, 32, base, optics)
    result = rows()
    no_compiler(monkeypatch, tmp_path)
    for got, reference in zip(result, rows()):
        np.testing.assert_allclose(got, reference, rtol=1e-12, atol=0.)


def test_linear_column_and_noncontiguous_rays(monkeypatch, tmp_path):
    calls = Mock(wraps=native())
    z = .5 * np.arange(401)
    spec = dict(kind='usstd76', grid=dict(step_km=.5, top_km=200.), end_km=79.8, values=.2 + np.exp(-z/14.))
    rho = (ct.R + np.arange(12.))[::3]
    a = np.array([-20., 0., -1., -100.])
    b = np.array([20., 100., 0., 40.])
    with monkeypatch.context() as m:
        m.setattr(density, '_host_native', lambda: calls)
        result = ct.linear_columns(spec, rho, a, b)
        assert calls.linear_columns_c.call_count == 1
    no_compiler(monkeypatch, tmp_path)
    np.testing.assert_allclose(result, ct.linear_columns(spec, rho, a, b), rtol=1e-12, atol=0.)


@pytest.mark.parametrize('hook', ['rows', 'profiles', 'exact', 'domain', 'base'])
def test_solar_hooks_keep_row_inputs(monkeypatch, hook):
    from dataclasses import dataclass

    monkeypatch.setenv(sun_parallel.ENV, '1')
    old, optics = model(top.Domain())
    h, x = np.array([4., 9.]), np.linspace(0, 1, 5)
    base = np.zeros((len(h), len(x), 5))
    calls = []
    if hook == 'rows':
        original = sun_parallel._row
        def row(*args):
            calls.append(args[1])
            return original(*args)
        monkeypatch.setattr(sun_parallel, '_row', row)
    elif hook == 'exact':
        def exact(p, d, end, optics, selected):
            calls.append(len(d))
            return np.full((len(d), 4), len(calls))
        monkeypatch.setattr(density, 'exact_profile_columns', exact)
    elif hook == 'domain':
        @dataclass(frozen=True)
        class CustomDomain(top.Domain):
            @property
            def extra_shells(self):
                calls.append('shells')
                return (2.25, 14.75)
        old.domain = CustomDomain()
    elif hook == 'base':
        class CustomBase(np.ndarray):
            def __getitem__(self, item):
                calls.append(item)
                return np.asarray(self)[item] + 7.
            def copy(self):
                return np.asarray(self).copy()
        base = base.view(CustomBase)
    else:
        original = old.profiles
        def profile(height):
            calls.append(height.shape)
            return original(height)
        old.profiles = profile
        base = optics = None

    with monkeypatch.context() as ordinary:
        ordinary.setattr(sun_parallel, '_native_rows', lambda *args: False)
        expected = sun_parallel.rows(old, h, x, ct.R, 8, base, optics)
        trace = calls.copy()
    calls.clear()
    actual = sun_parallel.rows(old, h, x, ct.R, 8, base, optics)
    assert calls == trace and calls
    np.testing.assert_array_equal(actual, expected)


def test_probe_and_shell_domains_are_independent(monkeypatch):
    native()
    monkeypatch.setenv(sun_parallel.ENV, '1')
    domain = top.Domain(200., .5, (200.,), 'us76_neff')
    old, _ = model(domain)
    old.probe.boundaries = partial(shell.boundaries, domain=top.Domain(120., .5, (120.,), 'vacuum'))
    h, x = np.array([0., 3., 12.]), np.linspace(0, 1, 21)
    got = sun_parallel.rows(old, h, x, ct.R, 16)
    want = [sun_parallel._row(old, y, x, ct.R, 16, None, None) for y in h]
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.)


def test_table_buffer_dimensions_checked_before_call(monkeypatch):
    lib = Mock()
    monkeypatch.setattr(rays, '_host_loops', lambda: lib)
    a = np.linspace(0, 1, 4)
    u, w = shell.gl(8)
    for x in (np.empty((4, 0)), np.empty(4, 'f4'), np.empty(8)[::2]):
        with pytest.raises(ValueError, match='E_HOST_NATIVE_BUFFER'):
            ct._fill_table(np.empty((4, 4)), dict(kind='exp', height=1.), True, a, x, u, w,
                           a + ct.R, np.zeros(4), 80.)
    lib.build_table_c.assert_not_called()


def test_compiler_recovery(tmp_path, monkeypatch):
    native()
    from types import SimpleNamespace
    clock = Mock(return_value=200.)
    monkeypatch.setattr(density, 'time', SimpleNamespace(monotonic=clock))
    monkeypatch.setenv('SOSFUSE2_WORK_ROOT', str(tmp_path))
    with monkeypatch.context() as missing:
        no_compiler(missing, tmp_path)
    assert density._host_native() is None
    clock.return_value += 61.
    assert density._host_native() is not None
