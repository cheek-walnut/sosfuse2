"""Cache source identities and host profile edge cases."""

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from sosfuse2 import column_cache, column_parallel, column_table, density, geometry, geometry_contract
from sosfuse2 import geometry_cpu, host_cache, paths, solver_columns, stratosphere, sun_parallel, tropo_profile
from sosfuse2 import identity as source_identity
from sosfuse2.accepted import engine


def solver():
    sun = SimpleNamespace(h=np.array([0.0, 80.0]), x=np.array([0.0, 1.0]), col=np.zeros((2, 2, 5)))
    geo = SimpleNamespace(h=np.array([0.0, 80.0]), chi=np.array([0.0, 1.0]), cfg={}, sun=sun)
    kernel = SimpleNamespace(height=1.2, host=lambda value: value)
    return SimpleNamespace(geo=geo, kernel=kernel, fuse_state={})


def source_edit(data, edit):
    if edit == 'comment':
        return data + b'\n# note\n'
    if edit == 'middle':
        anchor = b'R = 6371.0\n'
        assert anchor in data
        return data.replace(anchor, anchor + b'# note\n# another note\n', 1)
    if edit == 'docstring':
        anchor = b'Column integration and the sun-column table'
        assert anchor in data
        return data.replace(anchor, b'Column integration\nand the sun-column table\nAdditional documentation', 1)
    if edit == 'code':
        return data + b'\nIDENTITY_PROBE = 1\n'
    raise AssertionError(edit)


def patch_source(target, transform):
    actual_read = Path.read_bytes

    def read(path):
        data = actual_read(path)
        return transform(data) if path == target else data

    return patch.object(Path, 'read_bytes', read)


def sun_folder(tmp_path, monkeypatch):
    folders = []

    def sun(scale, folder):
        folders.append(folder)
        return SimpleNamespace(h=np.empty(81), x=np.empty(401))

    old = SimpleNamespace(R=6371.0, Z=80.0, SunGeometry=sun)
    with monkeypatch.context() as context:
        context.setattr(sun_parallel, 'prebuild', lambda *args: None)
        host_cache.sun_geometry(old, 1, tmp_path / 'local', 1.2, shared=tmp_path / 'shared')
    return folders[0]


def cache_keys(tmp_path, monkeypatch):
    s = solver()
    optics = density.build({})
    bundle = SimpleNamespace(arrays={
        'starts': np.zeros((1, 3)), 'dirs': np.array([[0.0, 0.0, 1.0]]), 'cell_off': np.array([0, 1]),
    })
    config = {'strato_profile': {'center': 21.0, 'sigma': 3.0, 'norm': 1.0}}
    return {
        'geometry': geometry_contract.identity(s)[1],
        'legacy_geometry': geometry.identity(s, 2)[1],
        'host': host_cache.digest(host_cache.context(s, optics)[1]),
        'sun': sun_folder(tmp_path, monkeypatch),
        'column': column_cache.cache_key(s.kernel, bundle, config, 'strato'),
    }


@pytest.mark.parametrize('edit', ['comment', 'middle', 'docstring', 'code'])
def test_shared_cache_keys_follow_source_identity(tmp_path, monkeypatch, edit):
    monkeypatch.delenv('SOSFUSE2_SKY_SOS_ROOT', raising=False)
    baseline = cache_keys(tmp_path, monkeypatch)
    target = paths.PKG / 'solver_columns.py'
    with patch_source(target, lambda data: source_edit(data, edit)):
        changed = cache_keys(tmp_path, monkeypatch)
    for name in baseline:
        assert (changed[name] != baseline[name]) is (edit == 'code'), name


def test_geometry_contract_upstream_comment_and_code(tmp_path, monkeypatch):
    for name in geometry_contract.UPSTREAM:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((geometry_contract.ROOT / name).read_bytes())
    monkeypatch.setattr(geometry_contract, 'ROOT', tmp_path)
    target = tmp_path / 'solver_columns.py'
    original = target.read_bytes()
    baseline = geometry_contract.identity(solver())[1]
    for edit in ('comment', 'middle', 'docstring', 'code'):
        target.write_bytes(source_edit(original, edit))
        assert (geometry_contract.identity(solver())[1] != baseline) is (edit == 'code')


@pytest.mark.parametrize('version', [1, 2, 3])
def test_geometry_versions_use_source_identity(monkeypatch, version):
    monkeypatch.delenv('SOSFUSE2_SKY_SOS_ROOT', raising=False)
    s = solver()
    content, baseline = geometry.identity(s, version)
    if version == 1:
        assert content['files']['sosfuse/geometry.py'] == 'unavailable'
    target = paths.PKG / 'solver_columns.py'
    for edit in ('comment', 'middle', 'docstring', 'code'):
        with patch_source(target, lambda data: source_edit(data, edit)):
            changed = geometry.identity(s, version)[1]
        assert (changed != baseline) is (edit == 'code')


def test_geometry_legacy_source_uses_source_identity(tmp_path, monkeypatch):
    target = tmp_path / 'scripts' / 'sosfuse' / 'geometry.py'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'"""Legacy geometry."""\nVALUE = 1\n')
    monkeypatch.setenv('SOSFUSE2_SKY_SOS_ROOT', str(tmp_path))
    baseline = geometry.identity(solver(), 1)[1]
    with patch_source(target, lambda data: data + b'\n# note\n'):
        assert geometry.identity(solver(), 1)[1] == baseline
    with patch_source(target, lambda data: data + b'\nIDENTITY_PROBE = 1\n'):
        assert geometry.identity(solver(), 1)[1] != baseline


@pytest.mark.parametrize('version', [2, 3])
def test_geometry_own_source(monkeypatch, version):
    monkeypatch.delenv('SOSFUSE2_SKY_SOS_ROOT', raising=False)
    target = paths.PKG / 'geometry.py'
    baseline = geometry.identity(solver(), version)[1]
    for suffix, different in ((b'\n# note\n', False), (b'\nIDENTITY_PROBE = 1\n', True)):
        with patch_source(target, lambda data: data + suffix):
            assert (geometry.identity(solver(), version)[1] != baseline) is different


def test_geometry_contract_own_source():
    target = paths.PKG / 'geometry_contract.py'
    baseline = geometry_contract.identity(solver())[1]
    for suffix, different in ((b'\n# note\n', False), (b'\nIDENTITY_PROBE = 1\n', True)):
        with patch_source(target, lambda data: data + suffix):
            assert (geometry_contract.identity(solver())[1] != baseline) is different


@pytest.mark.parametrize('name', ['column_table.py', 'tropo_profile.py'])
def test_column_table_sources(monkeypatch, name):
    target = paths.PKG / name
    column_table.source_hash.cache_clear()
    baseline = column_table.source_hash()
    try:
        for suffix, different in ((b'\n# note\n', False), (b'\nIDENTITY_PROBE = 1\n', True)):
            column_table.source_hash.cache_clear()
            with patch_source(target, lambda data: data + suffix):
                assert (column_table.source_hash() != baseline) is different
    finally:
        column_table.source_hash.cache_clear()


@pytest.mark.parametrize('name', ['density.py', 'components.py'])
def test_engine_fingerprint_and_cell_folder(tmp_path, name):
    target = paths.PKG / 'solver_columns.py'
    baseline = engine.fingerprint()
    for edit in ('comment', 'middle', 'docstring', 'code'):
        with patch_source(target, lambda data: source_edit(data, edit)):
            assert (engine.fingerprint() != baseline) is (edit == 'code')
    target = paths.PKG / name
    initial = engine.cell_folder(tmp_path, [0.0], [0.0], {})
    for suffix, different in ((b'\n# note\n', False), (b'\nIDENTITY_PROBE = 1\n', True)):
        with patch_source(target, lambda data: data + suffix):
            assert (engine.cell_folder(tmp_path, [0.0], [0.0], {}) != initial) is different


def test_engine_fingerprint_native_source_and_asset():
    native = paths.PKG / 'native' / 'sh_cpu.cpp'
    asset = paths.PKG / 'assets' / 'rgb_weights.npy'
    baseline = engine.fingerprint()
    with patch_source(native, lambda data: data + b'\n// note\n'):
        assert engine.fingerprint() == baseline
    with patch_source(native, lambda data: data + b'\nint identity_probe = 1;\n'):
        assert engine.fingerprint() != baseline
    with patch_source(asset, lambda data: data + b'\x00'):
        assert engine.fingerprint() != baseline


def test_worker_source_drift_uses_raw_bytes():
    target = paths.PKG / 'solver_columns.py'
    original = column_parallel.source_hashes()
    assert original[target.name] == hashlib.sha256(target.read_bytes()).hexdigest()
    with patch_source(target, lambda data: data + b'\n# note\n'):
        assert column_parallel.source_hashes()[target.name] != original[target.name]
        assert column_parallel.source_drift(original) == target.name


@pytest.mark.parametrize('kind,name', [
    ('strato', 'stratosphere.py'),
    ('strato', 'solver_columns.py'),
    ('strato', 'ray_columns.py'),
    ('strato', 'accepted/engine.py'),
    ('density', 'density.py'),
])
def test_column_cache_sources(kind, name):
    s = solver()
    bundle = SimpleNamespace(arrays={
        'starts': np.zeros((1, 3)), 'dirs': np.array([[0.0, 0.0, 1.0]]),
        'cell_off': np.array([0, 1]), 'end_col': np.zeros((1, 5)),
    })
    config = {'strato_profile': {'center': 21.0, 'sigma': 3.0, 'norm': 1.0}, 'density_optics': density.build({})}
    baseline = column_cache.cache_key(s.kernel, bundle, config, kind)
    for suffix, different in ((b'\n# note\n', False), (b'\nIDENTITY_PROBE = 1\n', True)):
        with patch_source(paths.PKG / name, lambda data: data + suffix):
            assert (column_cache.cache_key(s.kernel, bundle, config, kind) != baseline) is different


@pytest.mark.parametrize('name', ['host_cache.py', 'solver_sky.py', 'sun_parallel.py', 'ground_parallel.py'])
def test_host_context_additional_sources(name):
    s, optics = solver(), density.build({})
    baseline = host_cache.context(s, optics)[1]
    for suffix, different in ((b'\n# note\n', False), (b'\nIDENTITY_PROBE = 1\n', True)):
        with patch_source(paths.PKG / name, lambda data: data + suffix):
            assert (host_cache.context(s, optics)[1] != baseline) is different


def test_sun_source_and_host_geometry_keys(monkeypatch):
    s = solver()
    profile = {'center': 21.0, 'sigma': 3.0, 'norm': 1.0}
    target = paths.PKG / 'tropo_profile.py'
    baseline = solver_columns.SunGeometry.identity(1, 1.2, None)
    aerosol = host_cache.aerosol_identity(s)
    for suffix, different in ((b'\n# note\n', False), (b'\nIDENTITY_PROBE = 1\n', True)):
        with patch_source(target, lambda data: data + suffix):
            assert (solver_columns.SunGeometry.identity(1, 1.2, None) != baseline) is different
            assert (host_cache.aerosol_identity(s) != aerosol) is different
    target = paths.PKG / 'ground_parallel.py'
    baseline = stratosphere._geometry_key(s, None, profile)
    for suffix, different in ((b'\n# note\n', False), (b'\nIDENTITY_PROBE = 1\n', True)):
        with patch_source(target, lambda data: data + suffix):
            assert (stratosphere._geometry_key(s, None, profile) != baseline) is different


def test_shared_sun_asset_uses_content_identity(tmp_path, monkeypatch):
    asset = paths.PKG / 'assets' / 'no2_afglus_profile.json'
    baseline = sun_folder(tmp_path, monkeypatch)
    with patch_source(asset, lambda data: data + b'\n'):
        assert sun_folder(tmp_path, monkeypatch) == baseline
    def change_value(data):
        doc = json.loads(data)
        doc['no2_number_density_m3'][0] *= 1.01
        return json.dumps(doc).encode()
    with patch_source(asset, change_value):
        assert sun_folder(tmp_path, monkeypatch) != baseline


def test_source_digest_encodings(tmp_path, monkeypatch):
    column_table.source_hash.cache_clear()
    expected = hashlib.sha256()
    for path in (paths.PKG / 'column_table.py', paths.PKG / 'tropo_profile.py'):
        expected.update(path.name.encode())
        expected.update(source_identity.file_sha(path).encode())
    assert column_table.source_hash() == expected.hexdigest()

    names = (
        'solver_columns.py', 'accepted/engine.py', 'sun_parallel.py', 'column_parallel.py',
        'tropo_profile.py', 'ray_columns.py', 'density.py', 'assets/no2_afglus_profile.json',
    )
    expected = hashlib.sha256(json.dumps({'R': 6371.0, 'Z': 80.0}).encode())
    for name in sorted(str(paths.PKG / name) for name in names):
        p = Path(name)
        digest = source_identity.asset_sha(p) if p.name in source_identity.ASSET_NOTES else source_identity.file_sha(p)
        expected.update(digest.encode())
    assert sun_folder(tmp_path, monkeypatch).parent.name == expected.hexdigest()[:20]

    s = solver()
    profile = {'center': 21.0, 'sigma': 3.0, 'norm': 1.0}
    expected = hashlib.sha256()
    for array in (s.geo.h, s.geo.chi, s.geo.sun.h, s.geo.sun.x, s.geo.sun.col):
        expected.update(np.ascontiguousarray(array).tobytes())
    for name in ('stratosphere.py', 'ground_parallel.py', 'ray_columns.py', 'density.py', 'column_table.py', 'column_parallel.py'):
        expected.update(name.encode())
        expected.update(bytes.fromhex(source_identity.file_sha(paths.PKG / name)))
    expected.update(json.dumps({
        'profile': [21.0, 3.0, 1.0], 'cfg': {}, 'params': {}, 'density': None,
    }, sort_keys=True, default=str).encode())
    assert stratosphere._geometry_key(s, None, profile) == expected.hexdigest()[:24]


def test_column_method_selects_workers_after_table_path(monkeypatch):
    monkeypatch.setenv('SOSFUSE2_COLUMN_WORKERS', 'invalid')
    monkeypatch.setenv('SOSFUSE2_COLUMN_METHOD', 'table-v1')
    monkeypatch.setattr(column_table, 'fill', lambda *args: 'table')
    assert column_parallel.parallel_columns(None, 'strato', None, None, None, None, None) == 'table'
    monkeypatch.setenv('SOSFUSE2_COLUMN_METHOD', 'gl24')
    with pytest.raises(ValueError, match='^E_COLUMN_WORKERS:'):
        column_parallel.parallel_columns(None, 'strato', None, None, None, None, None)


def profile_state(**overrides):
    return {
        'parameters': {'aod550': 0.15},
        'troposphere_profile': {'status': 'enabled', 'model': 'two_layer', **overrides},
    }


def test_tropo_profile_small_boundary_scale_and_default_values():
    result = tropo_profile.parameters(tropo_profile.config(profile_state()))
    np.testing.assert_allclose(result['vector'], [
        1.5, 1.0, 0.3619283280304329, 0.4672897196261682,
        0.28511908516954365, 0.5405405405405405, 2.0,
    ], rtol=1e-12, atol=0)
    np.testing.assert_allclose(result['basis'], [1.2949156790712957, 1.8635488151113435], rtol=1e-12, atol=0)
    assert result['column'] == 1.0
    state = profile_state(bl_shape='exponential', bl_scale_height_km=1e-320)
    with pytest.raises(ValueError, match='^E_TROPO_PROFILE_FP32: boundary-layer scale height too small$'):
        tropo_profile.parameters(tropo_profile.config(state))


@pytest.mark.parametrize('basis', [(0.0, 1.0), (1.0, 0.0)])
def test_tropo_profile_rejects_zero_integrals(monkeypatch, basis):
    monkeypatch.setattr(tropo_profile, 'basis_integrals', lambda *args: basis)
    with pytest.raises(ValueError, match='^E_TROPO_PROFILE_FP32: boundary-layer scale height too small$'):
        tropo_profile.parameters(tropo_profile.config(profile_state()))


def test_geometry_prepare_publishes_process_specific_temp(tmp_path, monkeypatch):
    instance = geometry.TwoLayerGeometry.__new__(geometry.TwoLayerGeometry)
    instance.bundle = None
    instance.s = solver()
    instance.cache = tmp_path
    instance.stats = {}
    (tmp_path / 'source_height.json').write_text(json.dumps({'height': instance.s.kernel.height}))
    target = tmp_path / 'aerosol' / 'column.f4'
    host = {'end_col': np.array([[1, 2, 3, 4, 5]], dtype='f4'), 'dirs': np.zeros((1, 3))}
    instance.host_arrays = lambda: (host, [])
    instance.column_path = lambda height: target
    instance.install = lambda *args: None
    monkeypatch.setattr(host_cache, 'enabled', lambda: False)
    moves = []
    replace = os.replace

    def record(source, destination):
        moves.append((source, destination))
        replace(source, destination)

    monkeypatch.setattr(geometry.os, 'replace', record)
    instance.prepare()
    assert moves == [(target.with_name(f'{target.name}.{os.getpid()}.tmp'), target)]
    np.testing.assert_array_equal(np.fromfile(target, dtype='f4'), [2.0])


def test_geometry_cpu_import_does_not_change_bytecode_setting(monkeypatch):
    source = Path(geometry_cpu.__file__)
    namespace = {'__name__': 'sosfuse2.geometry_cpu', '__package__': 'sosfuse2', '__file__': str(source)}
    monkeypatch.setattr(sys, 'dont_write_bytecode', False)
    exec(compile(source.read_bytes(), str(source), 'exec'), namespace)
    assert sys.dont_write_bytecode is False
