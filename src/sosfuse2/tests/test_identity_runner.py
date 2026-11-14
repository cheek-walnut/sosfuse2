"""Runner identities, stored-source compatibility and shared resource helpers."""

import argparse
import copy
import json
import resource
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sosfuse2.backends import device_profile
from sosfuse2.foundation import host_limits, content_identity, paths
from sosfuse2.checks import solver_diagnostics
from sosfuse2.outputs import radiance_table
from sosfuse2.jobs import tables_from_sources
from sosfuse2.jobs import run_state as runner
from sosfuse2.atmosphere.components import canonical_state, optics_state, physics


@pytest.fixture
def state():
    return json.loads((paths.ASSETS / 'states_v2.json').read_text())['states'][0]


def settings(state):
    return runner.resolve_settings(runner.settings_for(SimpleNamespace(reduced=True, settings_json=None)), state)


def test_shared_identity_and_comment_mutation(state, monkeypatch):
    assert runner.package_hashes is content_identity.package_hashes
    assert runner.CONTRACT_VERSION == 'v5'
    cfg = settings(state)
    baseline_hashes = runner.package_hashes()
    baseline = runner.signed_contract(state, cfg, 'numpy', True, 'depolarizing')
    original = Path.read_bytes
    suffix = b'\n# Additional explanation.\n'

    def read(path):
        raw = original(path)
        return raw + suffix if path == paths.PKG / 'foundation/host_limits.py' else raw

    monkeypatch.setattr(Path, 'read_bytes', read)
    assert runner.package_hashes() == baseline_hashes
    assert runner.signed_contract(state, cfg, 'numpy', True, 'depolarizing') == baseline
    suffix = b'\nIDENTITY_PROBE = 1\n'
    assert runner.package_hashes() != baseline_hashes
    assert runner.signed_contract(state, cfg, 'numpy', True, 'depolarizing')[0] != baseline[0]


def test_state_labels_and_parameters(state):
    cfg = settings(state)
    before = runner.signed_contract(state, cfg, 'numpy', True, 'depolarizing')[0]
    changed = copy.deepcopy(state)
    changed.update(description='Updated description', group='diagnostic')
    assert runner.signed_contract(changed, cfg, 'numpy', True, 'depolarizing')[0] == before
    changed['parameters']['aod550'] += 0.01
    assert runner.signed_contract(changed, cfg, 'numpy', True, 'depolarizing')[0] != before


def source(path, state, cfg, version):
    solver = SimpleNamespace(fuse_state=state, geo=SimpleNamespace(h=np.array([0.0]), chi=np.array([0.0])))
    contract = dict(contract=version, state=dict(canonical_state(state), description='Stored description', group='stored'),
                    settings=cfg, backend='numpy', polarization_model='depolarizing')
    np.savez(path, source=np.ones((1, 571, 1), 'f4'), h=solver.geo.h, chi=solver.geo.chi,
             wavelength_nm=cfg['wavelengths_nm'], physics=json.dumps(physics()),
             optics_state=json.dumps(optics_state(state)), source_contract=json.dumps(contract))
    return solver


@pytest.mark.parametrize('version', ['v3', 'v4', 'v5'])
@pytest.mark.parametrize('order', [16, 32])
def test_stored_labels_and_legacy_order(tmp_path, state, version, order):
    cfg = settings(state)
    cfg['L'] = order
    path = tmp_path / 'source.npz'
    solver = source(path, state, cfg, version)
    restored = radiance_table.restored_settings(path, state, True, 'numpy')
    assert restored == (None if order == 32 else cfg)
    assert radiance_table.load_source(path, solver, [550.0], settings=cfg, backend='numpy',
                                  full_settings=order == 16).shape == (1, 571, 1)
    changed = copy.deepcopy(state)
    changed['parameters']['aod550'] += 0.1
    solver.fuse_state = changed
    with pytest.raises(ValueError, match='E_SOURCE_CONTRACT: state differs'):
        radiance_table.load_source(path, solver, [550.0], settings=cfg, backend='numpy', full_settings=True)
    if order == 16:
        with pytest.raises(ValueError, match='E_SOURCE_CONTRACT: state differs'):
            radiance_table.restored_settings(path, changed, True, 'numpy')


@pytest.mark.parametrize('platform,scale', [('darwin', 1), ('linux', 1024)])
def test_rss_units(monkeypatch, platform, scale):
    monkeypatch.setattr(resource, 'getrusage', lambda who: SimpleNamespace(ru_maxrss=12345))
    monkeypatch.setattr(sys, 'platform', platform)
    assert host_limits.peak_rss_bytes() == 12345 * scale
    assert solver_diagnostics.peak() == 12345 * scale
    assert isinstance(host_limits.peak_rss_bytes(), int)


@pytest.mark.parametrize('free_gib', [3, 4, 8])
def test_apply_pool_limit(monkeypatch, free_gib):
    monkeypatch.setenv('SOSFUSE2_POOL_LIMIT_GIB', '5')
    limits = []
    free = free_gib * 1024**3
    pool = SimpleNamespace(set_limit=lambda *, size: limits.append(size))
    xp = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=lambda: (free, free))),
                         get_default_memory_pool=lambda: pool)
    if free_gib < 4:
        with pytest.raises(RuntimeError, match='E_VRAM_FREE: need at least 4 GiB free'):
            device_profile.apply_pool_limit(xp)
        assert limits == []
    else:
        limit = min(free - 1024**3, 5 * 1024**3)
        assert device_profile.apply_pool_limit(xp) == (limit, free)
        assert limits == [limit]


@pytest.mark.parametrize('module', [radiance_table, tables_from_sources])
def test_cli_states_default_and_parent(tmp_path, state, monkeypatch, module):
    parent = dict(state, id='parent')
    own = dict(state, id='own')
    (tmp_path / 'parent.json').write_text(json.dumps(dict(states=[parent])))
    (tmp_path / 'states_v2.json').write_text(json.dumps(dict(parent='parent.json', states=[own])))
    monkeypatch.setattr(paths, 'ASSETS', tmp_path)
    parse = argparse.ArgumentParser.parse_args
    observed = []

    def parse_args(parser):
        argv = (['--id', 'parent', '--source', str(tmp_path / 'source.npz'), '--prepare', str(tmp_path / 'prepare'),
                 '--out', str(tmp_path / 'table.npz')] if module is radiance_table else
                ['--work-roots', str(tmp_path), '--dry-run', '--threads', '2'])
        args = parse(parser, argv)
        assert args.states == tmp_path / 'states_v2.json'
        return args

    read = runner.read_states

    def read_states(path, include_parent=False):
        assert include_parent is True
        rows = read(path, include_parent)
        assert [s['id'] for s in rows] == ['own', 'parent']
        observed.append(rows)
        return rows

    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', parse_args)
    monkeypatch.setattr(runner, 'read_states', read_states)
    if module is radiance_table:
        class ReadComplete(Exception):
            pass

        def restored(path, selected, *args):
            assert selected == parent
            raise ReadComplete

        monkeypatch.setattr(radiance_table, 'restored_settings', restored)
        with pytest.raises(ReadComplete):
            module.main()
    else:
        assert module.main() == 0
    assert len(observed) == 1


@pytest.fixture
def run_args(tmp_path, state, monkeypatch):
    from sosfuse2.solve import ray_geometry
    # The runner's NumPy adapter changes this global; isolate the test's lifetime.
    monkeypatch.setattr(ray_geometry, 'Kernel', ray_geometry.Kernel)
    monkeypatch.setattr(runner.shutil, 'which', lambda name: name)
    monkeypatch.setattr(runner, 'package_hashes', lambda: {'module': 'current'})
    states = tmp_path / 'states.json'
    states.write_text(json.dumps(dict(states=[dict(state, runnable=True)])))
    return SimpleNamespace(states=states, id=state['id'], out_name='target', source_out_name='source',
                           work_root=tmp_path, backend='numpy', reduced=True, settings_json=None,
                           polarization_model='depolarizing', threads=2, stratosphere=None,
                           extra_elevations='-4', rss_gb=6, block_size=128)


@pytest.mark.parametrize('version', ['v3', 'v4'])
@pytest.mark.parametrize('entry', [runner.run, runner.run_extra])
def test_old_output_rejected_before_work(run_args, state, version, entry, monkeypatch):
    out = run_args.work_root / 'out' / 'target'
    out.mkdir(parents=True)
    effective = runner.effective_state(dict(state, runnable=True), run_args)
    cfg = settings(effective)
    derived = runner.extra_contract(cfg, run_args.extra_elevations, run_args.source_out_name) if entry is runner.run_extra else None
    current = runner.signed_contract(effective, cfg, 'numpy', True, 'depolarizing', derived)[0]
    with monkeypatch.context() as patch:
        patch.setattr(runner, 'CONTRACT_VERSION', version)
        old = runner.signed_contract(effective, cfg, 'numpy', True, 'depolarizing', derived)[0]
    assert old != current  # Only the contract version differs in these signed inputs.
    (out / 'meta.json').write_text(json.dumps(dict(contract=version, signature=old, optics_state=optics_state(state))))
    with pytest.raises(RuntimeError, match=f"E_OUTPUT_IDENTITY: .*stored contract='{version}'"):
        entry(run_args)
    assert not (out / 'table.npz').exists()


@pytest.mark.parametrize('version', ['v3', 'v4'])
def test_old_extra_source_rejected_with_reason(run_args, state, monkeypatch, version):
    from sosfuse2.backends import compile_cache
    from sosfuse2.atmosphere import density

    monkeypatch.setattr(compile_cache, 'configure', lambda *args: None)
    monkeypatch.setattr(runner.threading, 'Thread', lambda **kw: SimpleNamespace(start=lambda: None))
    out = run_args.work_root / 'out' / 'source'
    out.mkdir(parents=True)
    source_dir = run_args.work_root / 'tmp' / 'source'
    source_dir.mkdir(parents=True)
    (source_dir / 'source.npz').touch()
    meta = dict(complete=True, contract=version, physics=physics(), optics_state=optics_state(state),
                id=state['id'], settings=settings(state), backend='numpy', polarization_model='depolarizing',
                mode='reduced-validation', code_hashes={'module': 'old'}, density=density.build(state)['meta'])
    (out / 'meta.json').write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match='E_EXTRA_SOURCE: source meta differs: contract,code_hashes'):
        runner.run_extra(run_args)


def test_script_path_peak():
    import runpy

    namespace = runpy.run_path(str(paths.PKG / 'checks/solver_diagnostics.py'))
    assert namespace['peak']() > 0


def test_batch_tolerance(state, monkeypatch):
    captured = []
    monkeypatch.setattr(runner, 'package_hashes', lambda: {})
    original = tables_from_sources.json.dumps

    def dumps(value, **kw):
        captured.append(value)
        return original(value, **kw)

    monkeypatch.setattr(tables_from_sources.json, 'dumps', dumps)
    tables_from_sources.signature_of(state, 'numpy', True, np.array([550.0]), np.array([-3.0]), 4, 6)
    assert captured[-1]['tolerance'] == runner.ANDERSON_TOLERANCE


def test_persistent_disconnect_preserves_reason(tmp_path, monkeypatch):
    from multiprocessing import connection
    from sosfuse2.solve import source_solver
    from sosfuse2.jobs import pipeline_execution

    closed = []
    monkeypatch.setenv('SOS_PERSISTENT_ROOT', str(tmp_path))
    monkeypatch.setenv('SOS_PERSISTENT_KEY', '00')
    monkeypatch.delenv('SOS_PERSISTENT_TEST_PREP_FAIL', raising=False)
    monkeypatch.setattr(connection, 'Client', lambda *a, **kw: SimpleNamespace(close=lambda: closed.append(True)))

    def disconnected(*args, **kwargs):
        raise ConnectionResetError('connection reset')

    monkeypatch.setattr(source_solver, 'Solver', disconnected)
    with pytest.raises(RuntimeError, match='E_PERSISTENT_SERVICE_LOST: ConnectionResetError: connection reset'):
        pipeline_execution.persistent_chunk(None, None, None, SimpleNamespace(), 'one', lambda *a, **kw: None)
    assert closed == [True]


def test_query_cache_uses_source_identity(tmp_path, monkeypatch):
    from sosfuse2.precompute import endpoint_contract, host_cache
    from sosfuse2.jobs import pipeline_execution

    monkeypatch.setattr(host_cache, 'enabled', lambda: True)
    monkeypatch.setattr(endpoint_contract, 'identity', lambda solver: ({}, 'geometry'))
    solver = SimpleNamespace(host_cache_root=tmp_path, geo=SimpleNamespace(),
                             ctx=SimpleNamespace(b=SimpleNamespace(sh=SimpleNamespace(OBS=0.1), R=6371.0)))
    observed = []

    class KeyCaptured(Exception):
        pass

    def cached(path, key, build):
        observed.append(key)
        raise KeyCaptured

    monkeypatch.setattr(host_cache, 'cached', cached)
    with pytest.raises(KeyCaptured):
        pipeline_execution.captured_queries(solver, [-3.0], [0], np.array([1.0]), np.array([2.0]), None)
    assert observed[0]['sources'] == {
        'sosfuse2/' + name: content_identity.file_sha(paths.PKG / name)
        for name in ('outputs/radiance_table.py', 'solve/ray_geometry.py', 'jobs/pipeline_execution.py')
    }


def test_extra_source_without_density_is_rejected(run_args, state, monkeypatch):
    from sosfuse2.backends import compile_cache
    from sosfuse2.atmosphere import density

    monkeypatch.setattr(compile_cache, 'configure', lambda *args: None)
    monkeypatch.setattr(runner.threading, 'Thread', lambda **kw: SimpleNamespace(start=lambda: None))
    monkeypatch.setattr(runner, 'create_configured', lambda *args, **kw: (
        SimpleNamespace(fuse_state=state, density_optics=density.build(state)), {}))
    run_args.geometry_cache = run_args.work_root / 'geometry'
    out = run_args.work_root / 'out/source'
    out.mkdir(parents=True)
    source_dir = run_args.work_root / 'tmp/source'
    source_dir.mkdir(parents=True)
    (source_dir / 'source.npz').touch()
    meta = dict(complete=True, contract=runner.CONTRACT_VERSION, physics=physics(), optics_state=optics_state(state),
                id=state['id'], settings=settings(state), backend='numpy', polarization_model='depolarizing',
                mode='reduced-validation', code_hashes={'module': 'current'})
    (out / 'meta.json').write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match='E_EXTRA_SOURCE: density profile missing'):
        runner.run_extra(run_args)
