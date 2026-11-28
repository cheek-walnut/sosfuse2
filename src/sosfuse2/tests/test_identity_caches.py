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

from sosfuse2.precompute import column_cache, column_parallel, endpoint_contract
from sosfuse2.rays import column_table
from sosfuse2.atmosphere import density
from sosfuse2.solve import ray_geometry
from sosfuse2.solve import ray_geometry_cpu
from sosfuse2.precompute import host_cache, sun_parallel
from sosfuse2.foundation import paths
from sosfuse2.rays import shell_segments
from sosfuse2.atmosphere import stratosphere, tropo_profile
from sosfuse2.foundation import content_identity as source_identity
from sosfuse2.solve import engine


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
        'geometry': endpoint_contract.identity(s)[1],
        'legacy_geometry': ray_geometry.identity(s, 2)[1],
        'host': host_cache.digest(host_cache.context(s, optics)[1]),
        'sun': sun_folder(tmp_path, monkeypatch),
        'column': column_cache.cache_key(s.kernel, bundle, config, 'strato'),
    }


@pytest.mark.parametrize('edit', ['comment', 'middle', 'docstring', 'code'])
def test_shared_cache_keys_follow_source_identity(tmp_path, monkeypatch, edit):
    monkeypatch.delenv('SOSFUSE2_SKY_SOS_ROOT', raising=False)
    baseline = cache_keys(tmp_path, monkeypatch)
    target = paths.PKG / 'rays/shell_segments.py'
    with patch_source(target, lambda data: source_edit(data, edit)):
        changed = cache_keys(tmp_path, monkeypatch)
    for name in baseline:
        assert (changed[name] != baseline[name]) is (edit == 'code'), name


def test_geometry_contract_upstream_comment_and_code(tmp_path, monkeypatch):
    for name in endpoint_contract.UPSTREAM:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((endpoint_contract.ROOT / name).read_bytes())
    monkeypatch.setattr(endpoint_contract, 'ROOT', tmp_path)
    target = tmp_path / 'rays/shell_segments.py'
    original = target.read_bytes()
    baseline = endpoint_contract.identity(solver())[1]
    for edit in ('comment', 'middle', 'docstring', 'code'):
        target.write_bytes(source_edit(original, edit))
        assert (endpoint_contract.identity(solver())[1] != baseline) is (edit == 'code')


@pytest.mark.parametrize('version', [1, 2, 3])
def test_geometry_versions_use_source_identity(monkeypatch, version):
    monkeypatch.delenv('SOSFUSE2_SKY_SOS_ROOT', raising=False)
    s = solver()
    content, baseline = ray_geometry.identity(s, version)
    if version == 1:
        assert content['files']['sosfuse/geometry.py'] == 'unavailable'
    target = paths.PKG / 'rays/shell_segments.py'
    for edit in ('comment', 'middle', 'docstring', 'code'):
        with patch_source(target, lambda data: source_edit(data, edit)):
            changed = ray_geometry.identity(s, version)[1]
        assert (changed != baseline) is (edit == 'code')


def test_geometry_legacy_source_uses_source_identity(tmp_path, monkeypatch):
    target = tmp_path / 'scripts/sosfuse/geometry.py'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'"""Legacy geometry."""\nVALUE = 1\n')
    monkeypatch.setenv('SOSFUSE2_SKY_SOS_ROOT', str(tmp_path))
    baseline = ray_geometry.identity(solver(), 1)[1]
    with patch_source(target, lambda data: data + b'\n# note\n'):
        assert ray_geometry.identity(solver(), 1)[1] == baseline
    with patch_source(target, lambda data: data + b'\nIDENTITY_PROBE = 1\n'):
        assert ray_geometry.identity(solver(), 1)[1] != baseline


@pytest.mark.parametrize('version', [2, 3])
def test_geometry_own_source(monkeypatch, version):
    monkeypatch.delenv('SOSFUSE2_SKY_SOS_ROOT', raising=False)
    target = paths.PKG / 'solve/ray_geometry.py'
    baseline = ray_geometry.identity(solver(), version)[1]
    for suffix, different in ((b'\n# note\n', False), (b'\nIDENTITY_PROBE = 1\n', True)):
        with patch_source(target, lambda data: data + suffix):
            assert (ray_geometry.identity(solver(), version)[1] != baseline) is different


def test_geometry_contract_own_source():
    target = paths.PKG / 'precompute/endpoint_contract.py'
    baseline = endpoint_contract.identity(solver())[1]
    for suffix, different in ((b'\n# note\n', False), (b'\nIDENTITY_PROBE = 1\n', True)):
        with patch_source(target, lambda data: data + suffix):
            assert (endpoint_contract.identity(solver())[1] != baseline) is different


@pytest.mark.parametrize('name', ['rays/column_table.py', 'atmosphere/tropo_profile.py'])
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


@pytest.mark.parametrize('name', ['atmosphere/density.py', 'atmosphere/components.py'])
def test_engine_fingerprint_and_cell_folder(tmp_path, name):
    target = paths.PKG / 'rays/shell_segments.py'
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
    native = paths.PKG / 'backends/native/sh_cpu.cpp'
    asset = paths.PKG / 'assets' / 'rgb_weights.npy'
    baseline = engine.fingerprint()
    with patch_source(native, lambda data: data + b'\n// note\n'):
        assert engine.fingerprint() == baseline
    with patch_source(native, lambda data: data + b'\nint identity_probe = 1;\n'):
        assert engine.fingerprint() != baseline
    with patch_source(asset, lambda data: data + b'\x00'):
        assert engine.fingerprint() != baseline


def test_worker_source_drift_uses_raw_bytes():
    target = paths.PKG / 'rays/shell_segments.py'
    original = column_parallel.source_hashes()
    name = target.relative_to(paths.PKG).as_posix()
    assert original[name] == hashlib.sha256(target.read_bytes()).hexdigest()
    with patch_source(target, lambda data: data + b'\n# note\n'):
        assert column_parallel.source_hashes()[name] != original[name]
        assert column_parallel.source_drift(original) == name


@pytest.mark.parametrize('kind,name', [
    ('strato', 'atmosphere/stratosphere.py'),
    ('strato', 'rays/shell_segments.py'),
    ('strato', 'rays/ray_columns.py'),
    ('strato', 'solve/engine.py'),
    ('density', 'atmosphere/density.py'),
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


@pytest.mark.parametrize('name', ['precompute/host_cache.py', 'rays/incoming_quadrature.py', 'precompute/sun_parallel.py', 'precompute/ground_parallel.py'])
def test_host_context_additional_sources(name):
    s, optics = solver(), density.build({})
    baseline = host_cache.context(s, optics)[1]
    for suffix, different in ((b'\n# note\n', False), (b'\nIDENTITY_PROBE = 1\n', True)):
        with patch_source(paths.PKG / name, lambda data: data + suffix):
            assert (host_cache.context(s, optics)[1] != baseline) is different


def test_sun_source_and_host_geometry_keys(monkeypatch):
    s = solver()
    profile = {'center': 21.0, 'sigma': 3.0, 'norm': 1.0}
    target = paths.PKG / 'atmosphere/tropo_profile.py'
    baseline = shell_segments.SunGeometry.identity(1, 1.2, None)
    aerosol = host_cache.aerosol_identity(s)
    for suffix, different in ((b'\n# note\n', False), (b'\nIDENTITY_PROBE = 1\n', True)):
        with patch_source(target, lambda data: data + suffix):
            assert (shell_segments.SunGeometry.identity(1, 1.2, None) != baseline) is different
            assert (host_cache.aerosol_identity(s) != aerosol) is different
    target = paths.PKG / 'precompute/ground_parallel.py'
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


def test_registered_source_order():
    """Check the historical inputs, allowing additional files to acquire their own slots."""
    from sosfuse2.foundation import dependencies

    expected = {
        'accepted': 'solve/__init__.py solve/engine.py',
        'host_module': 'foundation/__init__.py foundation/available_cpus.py outputs/pack_state.py',
        'asset': (
            'assets/spectra.json assets/spectra_fine.json assets/spectra_fine_temperature.json assets/lef_cie.json '
            'assets/cmf_1nm.json assets/states_v1.json assets/states_v2.json assets/states_v1_usstd.json '
            'assets/rgb_weights.npy assets/temperature_usstd76.json assets/no2_afglus_profile.json '
            'assets/no2_vandaele1998_fine.json assets/upper_air_us76_neff.json'
        ),
        'state_asset': 'assets/states_v1.json assets/states_v2.json assets/states_v1_usstd.json',
        'engine_source': (
            'foundation/top.py rays/shell_segments.py atmosphere/extinction.py rays/incoming_quadrature.py '
            'rays/incoming_rings.py rays/zenith_interpolation.py solve/engine_core.py outputs/preview_functions.py '
            'atmosphere/tropo_profile.py assets/spectra.json'
        ),
        'engine_extra': 'solve/engine.py outputs/pack_state.py backends/native/sh_cpu.cpp assets/rgb_weights.npy',
        'cell_extra': 'atmosphere/density.py atmosphere/components.py',
        'endpoint_upstream': (
            'foundation/top.py solve/engine.py rays/shell_segments.py rays/incoming_rings.py solve/engine_core.py '
            'atmosphere/tropo_profile.py rays/ray_columns.py atmosphere/density.py atmosphere/components.py '
            'solve/ray_geometry_parallel.py precompute/column_parallel.py jobs/run_state.py solve/configuration.py solve/build.py'
        ),
        'endpoint_contract': 'precompute/endpoint_contract.py',
        'column_integral': 'rays/column_table.py atmosphere/tropo_profile.py foundation/top.py',
        'strato_geometry': (
            'foundation/top.py atmosphere/stratosphere.py precompute/ground_parallel.py rays/ray_columns.py '
            'atmosphere/density.py rays/column_table.py precompute/column_parallel.py rays/incoming_quadrature.py '
            'rays/shell_segments.py'
        ),
        'shared_sun': (
            'assets/no2_afglus_profile.json atmosphere/density.py atmosphere/tropo_profile.py foundation/top.py '
            'precompute/column_parallel.py precompute/sun_parallel.py rays/ray_columns.py rays/shell_segments.py '
            'solve/engine.py'
        ),
        'sun_table': (
            'rays/shell_segments.py foundation/top.py atmosphere/density.py atmosphere/tropo_profile.py '
            'rays/ray_columns.py rays/column_table.py precompute/sun_parallel.py precompute/column_parallel.py'
        ),
        'host_aerosol': 'precompute/ground_parallel.py atmosphere/tropo_profile.py',
        'host_context': (
            'precompute/host_cache.py atmosphere/density.py rays/ray_columns.py rays/incoming_quadrature.py '
            'precompute/sun_parallel.py precompute/ground_parallel.py rays/column_table.py '
            'precompute/column_parallel.py'
        ),
        'worker': (
            'foundation/top.py rays/shell_segments.py rays/incoming_quadrature.py rays/incoming_rings.py '
            'rays/ray_columns.py solve/engine_core.py solve/ray_geometry_parallel.py atmosphere/density.py '
            'atmosphere/stratosphere.py precompute/column_parallel.py precompute/sun_parallel.py '
            'precompute/ground_parallel.py'
        ),
        'query': 'outputs/radiance_table.py solve/ray_geometry.py jobs/pipeline_execution.py',
        'gate_exclude': 'checks/gate.py',
        'gate_fixture': 'checks/gate_data/fixtures.json',
        'gate_config': 'backends/configs/C3.json backends/configs/C3_nostrato.json backends/configs/devices.json',
    }
    # Keep the original provenance order as well. Projecting onto these paths lets a contributor
    # add a module to any scope without editing this historical sample.
    production = (
        '__init__ __main__ foundation/__init__ atmosphere/__init__ rays/__init__ precompute/__init__ '
        'backends/__init__ solve/__init__ outputs/__init__ jobs/__init__ checks/__init__ asset_tools/__init__ '
        'foundation/available_cpus foundation/top atmosphere/upper_air asset_tools/generate_upper_air '
        'backends/flags_env solve/engine outputs/pack_state atmosphere/air_wavelength backends/arithmetic '
        'jobs/cache_store precompute/column_cache precompute/column_parallel rays/column_table '
        'backends/compile_cache atmosphere/components atmosphere/density backends/device_profile '
        'backends/vram_budget backends/vram_trace solve/source_solver solve/source_iteration checks/gate '
        'checks/kernel_diagnostics asset_tools/generate_density_profile asset_tools/generate_minor_absorbers '
        'asset_tools/generate_strato_optics asset_tools/generate_tropo_optics solve/ray_geometry '
        'precompute/endpoint_contract solve/ray_geometry_cpu solve/ray_geometry_parallel '
        'precompute/ground_parallel solve/ground_sh precompute/host_cache foundation/host_limits '
        'foundation/content_identity asset_tools/iprt backends/flags solve/ksplit asset_tools/mie '
        'atmosphere/minor_absorbers checks/numerical_checks atmosphere/ozone_temperature foundation/paths '
        'asset_tools/prepare_minor_absorbers rays/shell_segments atmosphere/extinction rays/incoming_quadrature '
        'rays/incoming_rings rays/zenith_interpolation solve/engine_core outputs/preview_functions jobs/pipeline '
        'jobs/pipeline_execution jobs/plan_queue jobs/precompile checks/validation_evidence rays/ray_columns '
        'atmosphere/rayleigh backends/numpy_reference checks/solver_diagnostics jobs/run_state solve/configuration solve/build '
        'jobs/run_storage jobs/walkthrough '
        'backends/sh_layout atmosphere/spectral solve/strato_sh_aux asset_tools/strato_states backends/mlx_backend '
        'backends/mlx_ops backends/mlx_query backends/mlx_query_kernel backends/mlx_transport backends/mlx_warm '
        'backends/mlx_warm_kernels atmosphere/stratosphere precompute/sun_parallel outputs/radiance_table '
        'outputs/quantized_table jobs/tables_from_sources foundation/timers foundation/version '
        'atmosphere/tropo_mie atmosphere/tropo_profile checks/validate checks/stratosphere_checks '
        'solve/ksplit_tolerance'
    ).split()
    production_set = set(production)
    assert [name for name in paths.PRODUCTION_MODULES if name in production_set] == production
    original = {f'{name}.py' for name in production} | {name for names in expected.values() for name in names.split()}
    original.update(('backends/kernel.cu', 'foundation/dependencies.py'))
    fields = {field for row in dependencies.FILES.values() for field in row}
    assert fields == expected.keys() | {'density_columns', 'strato_columns', 'column_name'}
    for scope, names in expected.items():
        assert [p for p in dependencies.files(scope) if p in original] == names.split(), scope

    for density_mode, first in ((True, 'atmosphere/density.py'), (False, 'atmosphere/stratosphere.py')):
        slots = [(key, path) for key, path in dependencies.column_slots(density_mode) if path in original]
        assert slots == [
            ('0', first), ('1', 'rays/shell_segments.py'), ('2', 'rays/shell_segments.py'),
            ('3', 'rays/shell_segments.py'), ('joint_columns', 'rays/ray_columns.py'), ('engine', 'solve/engine.py'),
        ]


def test_cache_scope_matrix(tmp_path, monkeypatch):
    from sosfuse2.checks import gate

    # Keep the optional external profile constant on installations without IPRT inputs.
    profile = tmp_path / 'ozone.json'
    profile.write_bytes(b'{}')
    monkeypatch.setattr(density, 'asset_path', lambda: profile)
    s = solver()
    bundle = SimpleNamespace(arrays={
        'starts': np.zeros((1, 3)), 'dirs': np.array([[0., 0., 1.]]),
        'cell_off': np.array([0, 1]), 'end_col': np.zeros((1, 5)),
    })
    config = dict(strato_profile=dict(center=21., sigma=3., norm=1.),
                  density_optics=density.build({'density': dict(density.LEGACY)}))

    def keys():
        column_table.source_hash.cache_clear()
        return {
            'endpoint': endpoint_contract.identity(s)[1],
            'integral': column_table.source_hash(),
            'density_columns': column_cache.cache_key(s.kernel, bundle, config, 'density'),
            'strato_columns': column_cache.cache_key(s.kernel, bundle, config, 'strato'),
            'sun': shell_segments.SunGeometry.identity(1, 1.2, None),
            'shared_sun': sun_folder(tmp_path, monkeypatch),
            'cell': engine.cell_folder(tmp_path, s.geo.h, s.geo.chi, s.geo.cfg),
            'gate': gate.code_identity(),
        }

    baseline = keys()
    try:
        for name, changed in (
            ('jobs/run_state.py', {'endpoint', 'gate'}),
            ('rays/column_table.py', {'endpoint', 'integral', 'density_columns', 'strato_columns', 'sun', 'gate'}),
            ('atmosphere/stratosphere.py', {'strato_columns', 'gate'}),
            ('precompute/sun_parallel.py', {'sun', 'shared_sun', 'gate'}),
            ('outputs/preview_functions.py', {'cell', 'gate'}),
            ('outputs/pack_state.py', {'cell', 'gate'}),
            ('foundation/version.py', {'gate'}),
            ('checks/gate.py', set()),
        ):
            with patch_source(paths.PKG / name, lambda data: source_edit(data, 'code')):
                after = keys()
            assert {key for key in after if after[key] != baseline[key]} == changed, name
    finally:
        column_table.source_hash.cache_clear()


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
    instance = ray_geometry.TwoLayerGeometry.__new__(ray_geometry.TwoLayerGeometry)
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

    monkeypatch.setattr(ray_geometry.os, 'replace', record)
    instance.prepare()
    scratch = target.with_name(f'{target.name}.{os.getpid()}.tmp')
    marker = target.with_suffix('.f4.sha256')
    assert moves == [(scratch, target), (scratch.with_suffix('.sha256.tmp'), marker)]
    assert marker.read_text().strip() == hashlib.sha256(target.read_bytes()).hexdigest()
    np.testing.assert_array_equal(np.fromfile(target, dtype='f4'), [2.0])


def test_geometry_cpu_import_does_not_change_bytecode_setting(monkeypatch):
    source = Path(ray_geometry_cpu.__file__)
    namespace = dict(__name__='sosfuse2.solve.ray_geometry_cpu', __package__=ray_geometry_cpu.__package__, __file__=str(source))
    monkeypatch.setattr(sys, 'dont_write_bytecode', False)
    exec(compile(source.read_bytes(), str(source), 'exec'), namespace)
    assert sys.dont_write_bytecode is False


def test_attach_disk_cache_temporary_name_is_per_process(monkeypatch, tmp_path):
    """Two processes sharing STRATO_ATTACH_CACHE never write the same temporary file (atomic replace per writer)."""
    from sosfuse2.precompute import ground_parallel

    geom = [tuple(np.zeros(2) for _ in stratosphere.GEOM_FIELDS)]
    monkeypatch.setenv('STRATO_ATTACH_CACHE', str(tmp_path))
    monkeypatch.setattr(stratosphere, '_geometry_key', lambda s, scope, profile: 'k')
    monkeypatch.setattr(stratosphere, 'solar_grid', lambda s, old, profile: np.zeros(3))
    monkeypatch.setattr(stratosphere, '_sun_method', lambda s: None)
    monkeypatch.setattr(ground_parallel, 'cells', lambda *a, **kw: geom)
    monkeypatch.setattr(stratosphere, '_GEOM_MEMO', {})
    moves = []
    replace = os.replace

    def record(source, destination):
        moves.append((Path(source).name, Path(destination).name))
        replace(source, destination)

    monkeypatch.setattr(os, 'replace', record)
    stratosphere._ground_geometry(SimpleNamespace(), None, None, None)
    assert moves == [(f'attach_geometry_k.{os.getpid()}.tmp.npz', 'attach_geometry_k.npz')]


@pytest.mark.parametrize(
    'module', ['generate_minor_absorbers', 'generate_strato_optics', 'generate_tropo_optics']
)
def test_asset_generator_import_does_not_change_bytecode_setting(monkeypatch, module):
    import importlib

    mod = importlib.import_module(f'sosfuse2.asset_tools.{module}')
    source = Path(mod.__file__)
    namespace = dict(__name__=mod.__name__, __package__=mod.__package__, __file__=str(source))
    monkeypatch.setattr(sys, 'dont_write_bytecode', False)
    exec(compile(source.read_bytes(), str(source), 'exec'), namespace)
    assert sys.dont_write_bytecode is False


@pytest.mark.parametrize('component', [2, 3])
def test_missing_optics_keeps_legacy_columns_offline(tmp_path, monkeypatch, component):
    monkeypatch.setenv('SOSFUSE2_WORK_ROOT', str(tmp_path))
    monkeypatch.setenv('SOSFUSE2_OFFLINE', '1')
    optics = density.build(dict(density=dict(density.LEGACY)))
    got = column_table.molecular(None, component)
    assert got == column_table.molecular(optics, component)
    if component == 3:
        assert got['kind'] == 'exp' and got['height'] == 4.0
    assert not (tmp_path / 'assets').exists()


def test_endpoint_identity_without_density_assets(tmp_path, monkeypatch):
    monkeypatch.setenv('SOSFUSE2_WORK_ROOT', str(tmp_path))
    monkeypatch.setenv('SOSFUSE2_OFFLINE', '1')
    common, signature = endpoint_contract.identity(solver())
    assert common['minor_profiles'] == [column_table.molecular(None, k) for k in (2, 3)]
    assert len(signature) == 20
    assert not (tmp_path / 'assets').exists()


@pytest.mark.parametrize('name,old,new,affects_endpoint', [
    ('solve/configuration.py', b'v <= 13 + 1e-9', b'v <= 14 + 1e-9', True),
    ('solve/build.py', b'gm.pack(p, d, n, s, scope, ground=False)', b'gm.pack(p, d, n, s, scope, ground=True)', True),
    ('jobs/run_storage.py', b'allow_nan=False', b'allow_nan=True', False),
])
def test_run_module_geometry_keys(name, old, new, affects_endpoint):
    from sosfuse2.checks import gate

    state = solver()

    def identities():
        return (endpoint_contract.identity(state)[1], ray_geometry.identity(state, 2)[1],
                source_identity.package_hashes(), gate.code_identity())

    target = paths.PKG / name
    assert old in target.read_bytes()
    original = identities()
    with patch_source(target, lambda data: data.replace(old, new, 1)):
        changed = identities()
    assert tuple(a != b for a, b in zip(changed, original)) == (affects_endpoint, affects_endpoint, True, True)
