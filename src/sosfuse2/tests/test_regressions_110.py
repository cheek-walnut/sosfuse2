"""Regression tests for defects fixed before the 1.1.0 release."""
import ast
import errno
import inspect
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sosfuse2 import cache_store, column_parallel, column_table, density, gate, table_fast, table_q, tropo_profile
from sosfuse2.tests import test_b2_components, test_cache_store, test_density_order, test_spectral, test_strato_precision, test_tropo_mie
from sosfuse2.components import canonical_state, optics_state, physics


def test_density_parallel_really_runs_and_matches_serial(monkeypatch):
    monkeypatch.setenv('SOSFUSE2_COLUMN_METHOD', 'gl24')
    monkeypatch.setattr(column_parallel, 'CHUNK', 2)
    monkeypatch.setattr(column_parallel, 'TASK_CHUNKS', 1)
    monkeypatch.setattr(column_parallel, 'MIN_RAYS', 1)
    old = test_b2_components.old_solver()
    optics = test_b2_components.atmosphere(True)
    starts = np.array([[0., 0., 6371.], [0., 0., 6381.]])
    dirs = np.array([[0., 0., 1.], [0.2, 0., np.sqrt(.96)], [0., 0., 1.], [0.6, 0., .8]])
    offsets = np.array([0, 2, 4])
    p = starts[np.repeat(np.arange(2), 2)]
    base = old.total_columns(p, dirs, 24)
    expected = density.correct_columns(p, dirs, base, old, optics)
    actual = base.copy()
    assert column_parallel.parallel_columns(actual, 'density', starts, offsets, dirs, old, optics, count=2)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)


def test_default_workers_and_column_method(monkeypatch):
    monkeypatch.delenv(column_parallel.ENV, raising=False)
    monkeypatch.delenv('SOSFUSE2_COLUMN_METHOD', raising=False)
    assert column_parallel.workers() == column_parallel.default_workers()
    assert column_table.method() == 'table-v1'


@pytest.mark.parametrize('name', ['identity.json', 'source_height.json'])
def test_endpoint_metadata_conflict_does_not_publish_packed(tmp_path, name):
    stage, root = tmp_path/'stage', tmp_path/'mirror'
    key = test_cache_store.endpoints(stage, 'a'*20)
    old = test_cache_store.put(root, key+'/'+name, 'different')
    unit = cache_store.Unit(key, 'endpoints', [])
    assert cache_store.publish(unit, stage, root) == 'E_CACHE_CONFLICT'
    assert old.read_text() == 'different'
    assert not (root/key/'packed').exists()


def test_endpoint_identical_existing_metadata_is_safe(tmp_path):
    stage, root = tmp_path/'stage', tmp_path/'mirror'
    key = test_cache_store.endpoints(stage, 'a'*20)
    for name in ('identity.json', 'source_height.json'):
        test_cache_store.put(root, key+'/'+name, (stage/key/name).read_bytes())
    assert cache_store.publish(cache_store.Unit(key, 'endpoints', []), stage, root) == 'published'
    assert (root/key/'packed/complete.json').is_file()


def test_link_error_never_overwrites_racing_writer(tmp_path, monkeypatch):
    source, dest = tmp_path/'source', tmp_path/'target'
    source.write_text('download')
    def race(src, dst):
        Path(dst).write_text('other writer')
        raise OSError(errno.EPERM, 'hard links unsupported')
    monkeypatch.setattr(cache_store.os, 'link', race)
    with pytest.raises(OSError) as error:
        cache_store._link(source, dest)
    assert error.value.errno == errno.EPERM
    assert dest.read_text() == 'other writer'
    assert source.read_text() == 'download'


@pytest.mark.parametrize('args', [('--flagsets', 'C3_typo'), ('--flagsets', 'C3,C3_typo'),
                                  ('--flagsets', ''), ('--fixtures', 'FX-C3,missing'), ('--fixtures', '')])
def test_gate_refuses_unknown_selection(tmp_path, args):
    out = tmp_path/'gate'
    assert gate.main(['--pregate', '--out', str(out), *args]) == 2
    report = json.loads((out/'gate.json').read_text())
    assert report['code'] == 'E_GATE_CONFIG'
    assert not report['fixtures']


def test_gate_refuses_empty_selection():
    with pytest.raises(gate.GateError, match='E_GATE_CONFIG'):
        gate.selected({'fixtures': {}}, SimpleNamespace(flagsets=None, fixtures=None))


def test_extra_source_file_is_work_root_relative():
    tree = ast.parse(inspect.getsource(__import__('sosfuse2.run_gpu_state_v3', fromlist=['run_extra']).run_extra))
    value = next(k.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                 for k in node.keywords if k.arg == 'source_file')
    code = compile(ast.Expression(value), '<source_file>', 'eval')
    work = Path('work')
    assert eval(code, {'src': work/'tmp'/'job'/'source.npz', 'a': SimpleNamespace(work_root=work)}) == 'tmp/job/source.npz'


def _source_fixture(path):
    state = {'id': 'one', 'parameters': {'ozone_du': 280., 'aod550': .15, 'o4': False}}
    settings = dict(nr=1, nchi=1, nmu=1, nphi=2, path_n=4, L=32, interpolation='log', observer_km=.1)
    solver = SimpleNamespace(fuse_state=state, geo=SimpleNamespace(h=np.array([0.]), chi=np.array([0.])))
    arrays = dict(source=np.ones((1, 571, 1), 'f4'), h=solver.geo.h, chi=solver.geo.chi, wavelength_nm=[550.],
                  physics=np.array(json.dumps(physics())), optics_state=np.array(json.dumps(optics_state(state))))
    contract = dict(state=canonical_state(state), settings=dict(settings), backend='numpy', polarization_model='depolarizing')
    np.savez(path, **arrays, source_contract=np.array(json.dumps(contract)))
    return solver, settings, arrays, contract


def test_source_contract_rejects_wrong_state_settings_and_missing(tmp_path, monkeypatch):
    for name in ('SOSFUSE2_KERNEL_FLAGS', 'SOSFUSE2_BLOCK_SIZE'):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path/'source.npz'
    solver, settings, arrays, contract = _source_fixture(path)
    assert table_fast.load_source(path, solver, [550.], settings=settings, backend='numpy').shape == (1, 571, 1)
    for key, value in [('ozone_du', 600.), ('aod550', .4)]:
        changed = json.loads(json.dumps(solver.fuse_state))
        changed['parameters'][key] = value
        with pytest.raises(ValueError, match='E_SOURCE_CONTRACT'):
            table_fast.load_source(path, SimpleNamespace(fuse_state=changed, geo=solver.geo), [550.], settings=settings, backend='numpy')
    with pytest.raises(ValueError, match='E_SOURCE_CONTRACT'):
        table_fast.load_source(path, solver, [550.], settings=dict(settings, nphi=64), backend='numpy')
    with pytest.raises(ValueError, match='E_SOURCE_CONTRACT'):
        table_fast.load_source(path, solver, [550.], settings=settings, backend='cupy')
    assert table_fast.load_source(path, solver, [550.], settings=settings, backend='numpy', full_settings=True).shape == (1, 571, 1)
    with pytest.raises(ValueError, match='E_SOURCE_CONTRACT'):
        table_fast.load_source(path, solver, [550.], settings=dict(settings, spectral={'mode': 'different'}),
                               backend='numpy', full_settings=True)
    contract['settings']['spectral'] = {'mode': 'different'}
    np.savez(path, **arrays, source_contract=np.array(json.dumps(contract)))
    with pytest.raises(ValueError, match='E_SOURCE_CONTRACT'):
        table_fast.load_source(path, solver, [550.], settings=settings, backend='numpy')
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='E_SOURCE_CONTRACT'):
        table_fast.load_source(path, solver, [550.], settings=settings, backend='numpy')


def test_table_fast_cli_writes_current_physics_for_q16(tmp_path, monkeypatch):
    source = tmp_path/'source.npz'
    solver, settings, _, _ = _source_fixture(source)
    states = tmp_path/'states.json'
    states.write_text(json.dumps({'states': [solver.fuse_state]}))
    monkeypatch.setattr(table_fast, 'build_solver', lambda *args: (solver, {}, settings))
    monkeypatch.setattr(table_fast, 'make_table', lambda s, f, hs, el, az, *args: (np.arange(1, 49, dtype='f4').reshape(2, 4, 6, 1), []))
    monkeypatch.setattr(sys, 'argv', ['table_fast', '--id', 'one', '--states', str(states), '--source', str(source),
                                    '--prepare', str(tmp_path/'prepare'), '--out', str(tmp_path/'table.npz'), '--backend', 'numpy', '--reduced'])
    table_fast.main()
    table_q.save(tmp_path/'table.npz', tmp_path/'q.npz')
    assert table_q.load(tmp_path/'q.npz')['header']['physics'] == physics()
    assert table_q.round_trip(tmp_path/'table.npz', tmp_path/'q.npz')['max_rel'] <= table_q.MAX_REL


def test_spectral_child_forwards_state():
    node = next(n for n in ast.walk(ast.parse(test_spectral.CHILD)) if isinstance(n, ast.FunctionDef) and n.name == 'patched')
    calls = []
    def original(b, w, state):
        calls.append((b, w, state))
        return {'solar': np.ones(3), 'ozone': np.ones(3)}
    scope = {'orig': original, 'kind': 'E', 'k': 1}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<child>', 'exec'), scope)
    result = scope['patched']('band', [550.], {'state': 1})
    assert calls == [('band', [550.], {'state': 1})] and result['solar'][1] == 2


def test_strato_host_occupancy_model():
    from sosfuse2.tests.test_strato_parallel_host import occupancy_tests
    report = occupancy_tests()
    assert report['pass_'] and report['remaining_C3_STRATO_bytes'] == 101376-74008


@pytest.mark.skipif(shutil.which('clang++') is None, reason='clang++ required')
def test_density_native_uses_all_three_profile_columns(tmp_path):
    native = test_density_order.compile_native(tmp_path)
    optics = test_b2_components.atmosphere(True)
    table = optics['table'].astype('f4')
    h = np.array([0., 12.3, 80.], 'f4')
    actual = np.array([[native.profile_c(float(x), table.ctypes.data, k) for k in range(3)] for x in h], 'f4')
    np.testing.assert_allclose(actual, density.interpolate32(h, table), rtol=1e-6, atol=0)
    p, d = np.array([0., 0., 6371.]), np.array([0., 0., 1.])
    basis, dlocal = np.eye(3, dtype='f4'), np.array([0., 0., 1.], 'f4')
    heights, chi = np.array([0., 80.], 'f4'), np.array([0., np.pi], 'f4')
    sun = np.zeros((2, 2, 5), 'f4'); output = np.empty(4, 'f4')
    native.sample_c(p.ctypes.data, d.ctypes.data, basis.ctypes.data, dlocal.ctypes.data,
                    0., 1., 1, .5, 1., heights.ctypes.data, len(heights), chi.ctypes.data, len(chi),
                    1.2, heights.ctypes.data, len(heights), sun.ctypes.data, 2, table.ctypes.data,
                    3, float(optics['air_surface']), output.ctypes.data)
    from sosfuse2.reference import profile_at
    expected = profile_at(output[:1], dict(height=np.float32(1.2), density_optics=optics))[0, :3]
    np.testing.assert_allclose(output[1:], expected, rtol=1e-6, atol=0)


@pytest.mark.skipif(shutil.which('clang++') is None, reason='clang++ required')
@pytest.mark.parametrize('waves', [1, 15])
def test_strato_native_lut_has_full_allocation_and_exact_path(tmp_path, waves):
    from sosfuse2 import paths
    root = paths.PKG                                   # the package directory: kernel.cu, assets/kernel_C2_strato_base.cu
    coeff = np.zeros((33, waves), 'f4'); coeff[0] = 1.; coeff[1] = .1
    mu = np.array([-1., -.5, 0., .5, 1.], 'f4')
    exact = test_strato_precision.NativePhase(root, waves, 'exact', tmp_path)
    lut = test_strato_precision.NativePhase(root, waves, 'lut', tmp_path)
    assert lut.bins == 16384 and exact.bins == 0
    np.testing.assert_allclose(exact.evaluate(mu, coeff), test_strato_precision.numpy_phase(mu, coeff), rtol=1e-6, atol=0)
    np.testing.assert_allclose(lut.evaluate(mu, coeff), test_strato_precision.numpy_lut(mu, coeff, 16384), rtol=1e-6, atol=0)


def test_tropo_fp32_overflow_rejected_and_normal_profile_unchanged():
    state = dict(parameters=dict(aod550=.15), troposphere_profile=dict(status='enabled', model='two_layer'))
    normal = tropo_profile.build(state)
    assert np.isfinite(tropo_profile.profile32([0., 1., 1.5, 2.], normal['kernel'])).all()
    state['troposphere_profile']['ft_scale_height_km'] = .005
    with pytest.raises(ValueError, match='E_TROPO_PROFILE_FP32'):
        tropo_profile.build(state)


@pytest.mark.parametrize('requested', [[], ['--static']])
def test_tropo_mie_empty_or_unavailable_check_fails(tmp_path, monkeypatch, requested):
    monkeypatch.setattr(test_tropo_mie, 'ROOT', None)
    monkeypatch.setattr(sys, 'argv', ['test_tropo_mie', '--out', str(tmp_path), *requested])
    with pytest.raises(SystemExit) as exc:
        test_tropo_mie.main()
    assert exc.value.code == 1
    assert json.loads((tmp_path/'tropo_mie_tests.json').read_text())['pass_'] is False
    assert test_tropo_mie.judge([])['pass_'] is False


def test_tropo_mie_static_uses_package_import(tmp_path):
    script = '''
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
from sosfuse2.tests import test_tropo_mie as t
t.ROOT = Path(sys.argv[2])
with patch.object(t.shutil, 'which', return_value='tool'), patch.object(t, 'base_ref', return_value='base'), \\
     patch.object(t.subprocess, 'check_output', return_value=''), patch.object(t, 'preprocess', return_value=(0, '', '')), \\
     patch.object(t.itertools, 'product', return_value=[]):
    assert 'expected_errors' in t.static(t.ROOT)
'''
    result = subprocess.run([sys.executable, '-B', '-I', '-c', script, str(Path(test_tropo_mie.__file__).parents[2]), str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('energy_ok', [False, True])
def test_strato_v3_affects_overall_verdict(tmp_path, monkeypatch, energy_ok):
    from sosfuse2 import validate_strato as v
    monkeypatch.setattr(v, 'zero_bytes', lambda *args: {'pass_': True})
    monkeypatch.setattr(v, 'thin_case', lambda *args: ({'pass_': True}, None, None, None))
    monkeypatch.setattr(v, 'energy', lambda: {'source_integral_pass_': energy_ok})
    monkeypatch.setattr(v, 'additivity', lambda *args: {'pass_': True})
    monkeypatch.setattr(v, 'profile_grid', lambda *args: {'pass_': True})
    monkeypatch.setattr(v.json, 'load', lambda f: {'checks': {'sample': {'pass_': True}}})
    monkeypatch.setattr(sys, 'argv', ['validate_strato', '--out', str(tmp_path/'out'), '--baseline', str(tmp_path), '--skip-timing'])
    assert v.main()['local_numeric_pass_'] is energy_ok


def test_common_geometry_identity_tracks_endpoint_sources(tmp_path, monkeypatch):
    from sosfuse2 import geometry_contract as gc
    monkeypatch.setenv('SOSFUSE2_COLUMN_METHOD', 'gl24')
    original_root = gc.ROOT
    for name in gc.UPSTREAM:
        target = tmp_path/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((original_root/name).read_bytes())
    monkeypatch.setattr(gc, 'ROOT', tmp_path)
    solver = SimpleNamespace(geo=SimpleNamespace(h=np.array([0., 80.]), chi=np.array([0., 1.]), cfg={}))
    content, baseline = gc.identity(solver)
    for name in ('ray_columns.py', 'density.py', 'components.py', 'geometry_parallel.py',
                 'column_parallel.py', 'run_gpu_state_v3.py'):
        target = tmp_path/name
        before = target.read_bytes()
        assert 'sosfuse2/'+name in content['files']
        target.write_bytes(before+b'\n# changed endpoint source\n')
        assert gc.identity(solver)[1] != baseline
        target.write_bytes(before)
    assert gc.identity(solver)[1] == baseline


@pytest.mark.parametrize('stored,requested', [
    (np.arange(45.), np.arange(15.)),
    (np.array([550.]), np.full(15, 550.)),
    (np.full((1, 15), 550.), np.full(15, 550.)),
    (np.arange(15.), np.arange(15.)+1),
])
def test_source_wavelength_shape_and_values_are_checked(tmp_path, stored, requested):
    path = tmp_path/'source.npz'
    solver, settings, arrays, contract = _source_fixture(path)
    arrays['wavelength_nm'] = stored
    np.savez(path, **arrays, source_contract=np.array(json.dumps(contract)))
    with pytest.raises(ValueError, match='^E_SOURCE_WAVELENGTH$'):
        table_fast.load_source(path, solver, requested, settings=settings, backend='numpy')


def _multispectral_source(tmp_path, monkeypatch, count, block_size=128, strato=False):
    from sosfuse2 import paths, run_gpu_state_v3 as run
    for name in ('SOSFUSE2_KERNEL_FLAGS', 'SOSFUSE2_BLOCK_SIZE'):
        monkeypatch.delenv(name, raising=False)
    if block_size != 128:monkeypatch.setenv('SOSFUSE2_BLOCK_SIZE', str(block_size))
    state = next(s for s in json.loads((paths.ASSETS/'states_v1.json').read_text())['states'] if s.get('runnable'))
    if strato:
        state['stratosphere']['status'] = 'enabled'
        state['parameters']['stratosphere_tau550'] = .01
    raw = dict(view_shape=[1, 2], solar_elevation_deg=[-3., -6.])
    if count != 45:raw.update(wavelengths=count, spectral_mode='band')
    class SettingsFile:
        def read_text(self):return json.dumps(raw)
    cfg = run.resolve_settings(run.settings_for(SimpleNamespace(reduced=False, settings_json=SettingsFile())), state)
    h, chi = np.array([0.]), np.array([0.])
    values = np.arange(1, count+1, dtype='f4')
    arrays = dict(source=np.tile(values, (1, 12 if strato else 11, 1)), h=h, chi=chi, wavelength_nm=cfg['wavelengths_nm'],
                  physics=np.array(json.dumps(physics())), optics_state=np.array(json.dumps(optics_state(state))))
    contract = dict(state=canonical_state(state), settings=cfg, backend='cupy', polarization_model='depolarizing')
    path = tmp_path/'source.npz'
    np.savez(path, **arrays, source_contract=np.array(json.dumps(contract)))
    return state, cfg, arrays, contract, path


def test_restored_source_rejects_changed_settings_and_shape(tmp_path, monkeypatch):
    state, cfg, arrays, contract, path = _multispectral_source(tmp_path, monkeypatch, 45)
    assert table_fast.restored_settings(path, state, False, 'cupy') == cfg
    with pytest.raises(ValueError, match='E_SOURCE_CONTRACT'):
        table_fast.restored_settings(path, state, False, 'numpy')
    bad_contract = json.loads(json.dumps(contract))
    bad_contract['settings']['spectral']['asset_sha256'] = 'changed'
    np.savez(path, **arrays, source_contract=np.array(json.dumps(bad_contract)))
    with pytest.raises(ValueError, match='E_SOURCE_CONTRACT'):
        table_fast.restored_settings(path, state, False, 'cupy')
    arrays['wavelength_nm'] = arrays['wavelength_nm'][:-1]
    np.savez(path, **arrays, source_contract=np.array(json.dumps(contract)))
    with pytest.raises(ValueError, match='E_SOURCE_WAVELENGTH'):
        table_fast.restored_settings(path, state, False, 'cupy')


def test_restored_source_rejects_field_shape_before_slicing(tmp_path, monkeypatch):
    from sosfuse2 import run_gpu_state_v3 as run
    state, cfg, arrays, contract, path = _multispectral_source(tmp_path, monkeypatch, 45)
    arrays['source'] = arrays['source'][..., :-1]
    np.savez(path, **arrays, source_contract=np.array(json.dumps(contract)))
    def create(output, settings, waves, backend, state, block_size):
        return SimpleNamespace(W=len(waves), geo=SimpleNamespace(h=arrays['h'], chi=arrays['chi'], ns=1),
                               kernel=SimpleNamespace(K=1), fuse_state=state), {}
    monkeypatch.setattr(run, 'create_configured', create)
    with pytest.raises(ValueError, match='E_SOURCE_SHAPE'):
        table_fast.restored_table(path, state, cfg, tmp_path/'prepare', 'cupy', 1, 1, lambda message: None)


@pytest.mark.parametrize('count,block_size,strato', [(45, 128, False), (31, 128, False), (15, 128, False),
                                                    (45, 256, False), (45, 128, True)])
def test_multispectral_sources_replay_and_batch_preview(tmp_path, monkeypatch, count, block_size, strato):
    from sosfuse2 import run_gpu_state_v3 as run, tables_from_sources as batch
    from sosfuse2.host import pack_state as pm
    state, cfg, arrays, _, source = _multispectral_source(tmp_path, monkeypatch, count, block_size, strato)
    seen = []
    def create(output, settings, waves, backend, state, requested_block):
        assert settings == cfg and backend == 'cupy' and requested_block == block_size
        seen.append(np.array(waves))
        kernel = SimpleNamespace(K=1, constants={}, device_constants={})
        optics = dict(meta=dict(stratosphere_tau550=.01, optics_table='test', optics_sha256='f'*64)) if strato else None
        return SimpleNamespace(W=len(waves), geo=SimpleNamespace(h=arrays['h'], chi=arrays['chi'], ns=1),
                               kernel=kernel, fuse_state=state, strato=optics), {}
    def make(solver, field, hs, el, az, *args):
        np.testing.assert_array_equal(solver.ms[..., ::2], field)
        assert np.all(solver.ms[..., 1::2] == 0)
        return np.broadcast_to(field[0, 0], (len(hs), len(el), len(az), solver.W)).copy(), []
    monkeypatch.setattr(run, 'create_configured', create)
    monkeypatch.setattr(table_fast, 'make_table', make)
    rendered = []
    def save(path, rgb):
        rendered.append(np.array(rgb))
        path.write_bytes(b'preview')
    monkeypatch.setattr(pm, 'render_functions', lambda: dict(Y=np.array([.2126, .7152, .0722]),
        projection=lambda *args: None, project=lambda pano, projection: pano,
        central=lambda projected: (None, 1.), tone=lambda rgb, k: rgb, save_png=save))
    real_pack = pm.pack_state
    monkeypatch.setattr(pm, 'pack_state', lambda out, pack, compression, started=None:
                        real_pack(out, pack, 'gzip', started=started))
    args = SimpleNamespace(reduced=False, backend='cupy', group=1, batch=2, threads=1)
    assert batch.process(tmp_path, 'state', source, state, args, lambda message: None).startswith('done ')
    meta = json.loads((tmp_path/'out/state/meta.json').read_text())
    assert meta['source_file'] == str(source.relative_to(tmp_path))
    with tarfile.open(tmp_path/'pack/state.tar.gz') as archive:
        assert json.load(archive.extractfile('meta.json'))['source_file'] == str(source.relative_to(tmp_path))
    assert meta['stratosphere']['status'] == ('enabled' if strato else 'disabled')
    assert meta['parameters_computed']['stratosphere_tau550'] == (.01 if strato else 0.)
    assert meta['unsupported_parameters'] == []
    with np.load(tmp_path/'out/state/table.npz') as table:
        np.testing.assert_array_equal(table['radiance'][0, 0, 0], arrays['source'][0, 0])
        np.testing.assert_array_equal(table['wavelength_nm'], np.asarray(cfg['wavelengths_nm'], 'f4'))
    assert len(seen) == (count+14)//15 and all(len(chunk) == 15 for chunk in seen)
    if count % 15:assert np.all(seen[-1][count % 15:] == seen[-1][count % 15-1])
    assert (tmp_path/'pack/state.tar.gz').is_file() and len(rendered) == 2
    common_previews = list(rendered)
    assert batch.process(tmp_path, 'state', source, state, args, lambda message: None) == 'packed-existing'
    for first, repacked in zip(common_previews, rendered[2:]):
        np.testing.assert_array_equal(first, repacked)
    monkeypatch.setattr(pm, 'previews', pm.previews)
    run.enable_spectral_previews(pm)
    pm.previews(tmp_path/'out/state')
    for common, production in zip(common_previews, rendered[4:]):
        np.testing.assert_array_equal(common, production)
    states = tmp_path/'states.json'
    states.write_text(json.dumps({'states': [state]}))
    out = tmp_path/'cli/table.npz';out.parent.mkdir()
    monkeypatch.setattr(sys, 'argv', ['table_fast', '--id', state['id'], '--states', str(states),
                                     '--source', str(source), '--prepare', str(tmp_path/'prepare'),
                                     '--out', str(out), '--backend', 'cupy'])
    table_fast.main()
    with np.load(out) as table:
        np.testing.assert_array_equal(table['radiance'][0, 0, 0], arrays['source'][0, 0])


@pytest.mark.parametrize('platform,flags', [('darwin', ['-dynamiclib']), ('linux', ['-shared', '-fPIC', '-nostdinc++'])])
@pytest.mark.parametrize('harness', ['density', 'tropo'])
def test_host_harness_link_flags(tmp_path, monkeypatch, platform, flags, harness):
    from sosfuse2.tests import test_tropo_profile
    commands = []
    class StopCompile(Exception): pass
    def capture(command, **kwargs):
        commands.append(command)
        raise StopCompile
    monkeypatch.setattr(sys, 'platform', platform)
    monkeypatch.setattr(subprocess, 'run', capture)
    with pytest.raises(StopCompile):
        if harness == 'density': test_density_order.compile_native(tmp_path)
        else: test_tropo_profile.host_point(tmp_path, '')
    for flag in ('-dynamiclib', '-shared', '-fPIC', '-nostdinc++'):
        assert (flag in commands[0]) == (flag in flags)


def test_batch_failure_status_and_continue(tmp_path, monkeypatch, capsys):
    from sosfuse2 import tables_from_sources as batch
    states = tmp_path/'states.json'
    states.write_text(json.dumps({'states': [dict(id=name, runnable=True) for name in ('bad1', 'bad2', 'good')]}))
    for name in ('bad1', 'bad2', 'good'):
        source = tmp_path/'tmp'/name/'source.npz'; source.parent.mkdir(parents=True); source.write_bytes(b'not an npz')
    seen = []
    def process(root, ident, source, state, args, log):
        seen.append(ident)
        if ident == 'bad1': raise ValueError('invalid')
        if ident == 'bad2': raise RuntimeError('failed')
        return 'done'
    monkeypatch.setattr(batch, 'process', process)
    monkeypatch.setattr(sys, 'argv', ['batch', '--work-roots', str(tmp_path), '--states', str(states)])
    assert batch.main() == 1
    assert seen == ['bad1', 'bad2', 'good']
    output = capsys.readouterr().out
    assert 'FAIL '+tmp_path.name+'/bad1: invalid' in output
    assert 'FAIL '+tmp_path.name+'/bad2: failed' in output
    assert 'TABLES '+tmp_path.name+'/good done' in output
    monkeypatch.setattr(batch, 'process', lambda *args: 'done')
    assert batch.main() == 0
    result = subprocess.run([sys.executable, '-B', '-m', 'sosfuse2.tables_from_sources', '--work-roots', str(tmp_path),
                             '--states', str(states)], capture_output=True, text=True)
    assert result.returncode == 1 and all(f'FAIL {tmp_path.name}/{name}:' in result.stdout for name in ('bad1', 'bad2', 'good'))


@pytest.mark.parametrize('remote', [
    ':s3,provider=Cloudflare,access_key_id=AKIAEXAMPLE,secret_access_key=SECRETEXAMPLE,endpoint=https://x:bucket/p',
    ':s3,secret_access_key="SECRET,WITH:COLON",access_key_id=AKIAEXAMPLE:bucket/p',
    "store,custom='SECRET''WITH,QUOTE',access_key_id=AKIAEXAMPLE:bucket/p",
])
def test_cache_remote_redacted_in_rows_and_errors(tmp_path, monkeypatch, capsys, remote):
    seen = []
    def listing(value):
        seen.append(value)
        return {}, set()
    monkeypatch.setattr(cache_store, 'remote_listing', listing)
    monkeypatch.setattr(cache_store, 'REMOTE', remote)
    (tmp_path/'source').mkdir()
    report = tmp_path/'report.json'
    assert cache_store.main(['push', str(tmp_path/'source'), '--report', str(report)]) == 0
    assert cache_store.main(['pull', '--all', str(tmp_path/'target')]) == 0
    text = capsys.readouterr().out + report.read_text()
    for secret in ('Cloudflare', 'AKIAEXAMPLE', 'SECRET', 'https://x'):
        assert secret not in text
    assert ':bucket/p' in text and ('":s3,' in text or '"store,' in text)
    assert seen == [remote, remote]
    monkeypatch.setattr(cache_store, 'remote_listing', lambda value: (_ for _ in ()).throw(
        cache_store.CacheStoreError('E_CACHE_RCLONE', f'lsjson rc=7 for {remote}')))
    assert cache_store.main(['list', '--json']) == 2
    captured = capsys.readouterr(); text = captured.out+captured.err
    assert 'E_CACHE_RCLONE' in text and 'rc=7' in text and ':bucket/p' in text
    assert 'AKIAEXAMPLE' not in text and 'SECRET' not in text


@pytest.mark.parametrize('label', ['access_key_id', 'client_key', 'my_key_id', 'client_secret', 'secret_access_key'])
def test_cache_clean_secret_names_and_quoted_values(label):
    result = cache_store.clean(f'problem {label}="SECRET, WITH SPACE" rc=7')
    assert result == f'problem {label}=<redacted> rc=7'
    assert cache_store.clean(f'problem {label}="SECRET, WITH SPACE') == f'problem {label}=<redacted>'


def test_cache_redaction_preserves_paths_with_commas_and_spaces():
    path = 'bucket/a,mode=keep path/folder,foo=bar'
    remote = ':s3,secret_access_key=SECRET:'+path
    assert cache_store.redact_connections(remote, remote=True) == ':s3,<redacted>:'+path
    assert cache_store.redact_connections('local/dir,mode=keep', remote=True) == 'local/dir,mode=keep'
    assert cache_store.clean('failure :s3,secret=SECRET:bucket/a,mode=keep reason=bad') == (
        'failure :s3,<redacted>:bucket/a,mode=keep reason=bad')
    assert cache_store.clean(f'failure "{remote}" reason=bad') == f'failure ":s3,<redacted>:{path}" reason=bad'


@pytest.mark.parametrize('binary', [False, True])
def test_rclone_failure_redacts_connection(monkeypatch, binary):
    remote = ":s3,secret_access_key='SECRET,ONE:TWO',access_key_id=AKIAEXAMPLE:bucket/p"
    message = f'failed to create filesystem for {remote}: cause rc=7'
    monkeypatch.setattr(cache_store, 'CONFIG', None)
    monkeypatch.setattr(cache_store, 'rclone_exe', lambda: 'fake-rclone')
    monkeypatch.setattr(cache_store.subprocess, 'run', lambda *a, **kw:
                        SimpleNamespace(returncode=7, stderr=message.encode() if binary else message, stdout=''))
    with pytest.raises(cache_store.CacheStoreError) as exc:
        cache_store.rclone(['cat', remote], binary=binary)
    assert exc.value.code == 'E_CACHE_RCLONE' and ':bucket/p' in exc.value.detail and 'cause rc=7' in exc.value.detail
    assert 'SECRET' not in exc.value.detail and 'AKIAEXAMPLE' not in exc.value.detail


@pytest.mark.parametrize('invalid', [b'', b'{broken', b'null', b'[]', b'\xff'])
def test_kernel_stamp_rebuild_and_atomic_publication(tmp_path, monkeypatch, invalid):
    import hashlib
    from sosfuse2.accepted import engine
    monkeypatch.setattr(engine.paths, 'work_root', lambda: tmp_path)
    monkeypatch.setattr(engine.shutil, 'which', lambda name: 'fake-compiler')
    digest = hashlib.sha256((engine.paths.NATIVE/'sh_cpu.cpp').read_bytes()+b'2').hexdigest()
    target = tmp_path/'build'/f'kernel_t2_{digest[:12]}.so'; target.parent.mkdir()
    target.write_bytes(b'old')
    stamp = target.with_suffix('.json'); stamp.write_bytes(invalid)
    builds = []
    def compile_fake(args, **kwargs):
        builds.append(args); Path(args[-1]).write_bytes(b'new')
    real_replace = engine.os.replace
    def check_replace(src, dst):
        if Path(dst) == stamp:
            assert stamp.read_bytes() == invalid
            assert json.loads(Path(src).read_text())['source_hash'] == digest
        real_replace(src, dst)
    monkeypatch.setattr(engine.subprocess, 'run', compile_fake)
    monkeypatch.setattr(engine.os, 'replace', check_replace)
    assert engine.compile_kernel() == target and target.read_bytes() == b'new'
    assert json.loads(stamp.read_text())['source_hash'] == digest
    assert engine.compile_kernel() == target and len(builds) == 1
    assert sorted(target.parent.iterdir()) == sorted([target, stamp])


def test_spectral_output_refuses_nonempty_without_removal(tmp_path, monkeypatch):
    monkeypatch.setattr(test_spectral, 'ROOT', tmp_path)
    out = tmp_path/'out'; out.mkdir(); marker = out/'keep'; marker.write_text('keep')
    with pytest.raises(SystemExit, match='E_SPECTRAL_OUT_NOT_EMPTY'):
        test_spectral.runs(out, 1)
    assert marker.read_text() == 'keep' and list(out.iterdir()) == [marker]
    with pytest.raises(SystemExit, match='E_SPECTRAL_OUT_NOT_DIRECTORY'):
        test_spectral.runs(marker, 1)
    assert marker.read_text() == 'keep'


@pytest.mark.parametrize('archive_rc', [0, 7])
def test_spectral_archive_uses_argument_arrays_and_checks_git(tmp_path, monkeypatch, archive_rc):
    root = tmp_path/'research ; $quoted'; root.mkdir()
    out = tmp_path/'output ; $quoted'; out.mkdir()
    monkeypatch.setattr(test_spectral, 'ROOT', root)
    calls = []
    class Archive:
        def __init__(self, args, **kwargs):
            calls.append(('git', args, kwargs)); self.args = args; self.stdout = io.BytesIO(); self.returncode = archive_rc
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def wait(self): return self.returncode
    class StopAfterArchive(Exception): pass
    def tar(args, **kwargs): calls.append(('tar', args, kwargs))
    def stop(jobs): raise StopAfterArchive
    monkeypatch.setattr(test_spectral.subprocess, 'Popen', Archive)
    monkeypatch.setattr(test_spectral.subprocess, 'run', tar)
    monkeypatch.setattr(test_spectral, 'ThreadPoolExecutor', stop)
    with pytest.raises(StopAfterArchive if archive_rc == 0 else subprocess.CalledProcessError):
        test_spectral.runs(out, 1)
    assert calls[0][1] == ['git', '-C', str(root), 'archive', test_spectral.LEGACY_BASELINE, *test_spectral.TREE_FILES]
    assert calls[0][2] == {'stdout': subprocess.PIPE}
    assert calls[1][1] == ['tar', '-x', '-C', str(out/'legacy_tree')]
    assert calls[1][2]['check'] is True and 'shell' not in calls[1][2]
