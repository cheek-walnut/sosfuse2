"""Runner identities, stored-source compatibility and shared resource helpers."""

import argparse
import copy
import errno
import json
import os
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
                    settings=cfg, anderson_tol=cfg['anderson_tol'], backend='numpy', polarization_model='depolarizing')
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


@pytest.mark.parametrize(
    'driver_gib,cached_gib,used_gib,cap_gib,limit_gib',
    [(3, 0, 1, '5', None), (4, 0, 0, '5', 3), (8, 0, 0, '5', 5), (2, 2, 1, '', 4)],
)
def test_apply_pool_limit(monkeypatch, driver_gib, cached_gib, used_gib, cap_gib, limit_gib):
    monkeypatch.setenv('SOSFUSE2_POOL_LIMIT_GIB', cap_gib)
    gib = 1024**3
    limits = []
    driver_free = driver_gib * gib
    pool = SimpleNamespace(
        free_bytes=lambda: cached_gib * gib, used_bytes=lambda: used_gib * gib,
        set_limit=lambda *, size: limits.append(size),
    )
    runtime = SimpleNamespace(memGetInfo=lambda: (driver_free, 10 * gib))
    xp = SimpleNamespace(cuda=SimpleNamespace(runtime=runtime), get_default_memory_pool=lambda: pool)
    if limit_gib is None:
        with pytest.raises(RuntimeError, match=rf'E_VRAM_FREE: .*free_bytes={3 * gib}, required_bytes={4 * gib}'):
            device_profile.apply_pool_limit(xp)
        assert limits == []
    else:
        limit = limit_gib * gib
        assert device_profile.apply_pool_limit(xp) == (limit, driver_free)
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


def test_run_frees_chunk_device_memory_before_next_admission(run_args, monkeypatch):
    from sosfuse2.backends import compile_cache
    from sosfuse2.solve import ray_geometry_cpu, source_solver

    run_args.backend = 'cupy'
    run_args.reduced = False
    run_args.geometry_cache = run_args.work_root
    run_args.geometry_layers = 2
    run_args.fixed_boxes = run_args.keep_geometry = False
    run_args.max_eval = 1
    run_args.group = run_args.batch = 1
    cfg = dict(wavelengths_nm=list(range(500, 516)), solar_elevation_deg=[0.0], view_shape=[1, 1],
               anderson_tol=runner.ANDERSON_TOLERANCE)
    geometry = SimpleNamespace(stats={'signature': 'mock'}, cache=run_args.work_root, prepare=lambda: None)
    geometry.bundle = SimpleNamespace(ksplit=object(), arrays={'edges': object()})
    geometry.ground_bundle = SimpleNamespace(ksplit=object())
    edges = geometry.bundle.arrays['edges']
    events = []
    gib = 1024**3

    def mem_info():
        previous_chunk_held = 'table_done' in events and 'free' not in events
        events.append('admit')
        return (2 if previous_chunk_held else 8) * gib, 10 * gib

    def free_blocks():
        assert geometry.bundle.ksplit is None
        assert geometry.ground_bundle.ksplit is None
        assert geometry.bundle.arrays['edges'] is edges
        assert geometry.s.kernel.ksplit_aero is None
        assert not geometry.s.kernel.ground_sh_tables
        events.append('free')

    pool = SimpleNamespace(free_bytes=lambda: 0, used_bytes=lambda: gib, set_limit=lambda **kw: None,
                           free_all_blocks=free_blocks)
    xp = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=mem_info)),
                         get_default_memory_pool=lambda: pool)

    def create(_path, _cfg, wavelengths, *_args, **_kwargs):
        kernel = SimpleNamespace(xp=xp, profile_n=161, sync=lambda: None, host=np.asarray, constants={},
                                 device_constants={}, ksplit_aero=object(), ground_sh_tables={'temp': object()})
        return SimpleNamespace(kernel=kernel, W=len(wavelengths)), None

    class SecondChunkStarted(Exception):
        pass

    def phase(_name, event, **info):
        if event == 'source_start' and info['chunk'] == 1:
            raise SecondChunkStarted
        if event == 'table_done':
            events.append('table_done')

    monkeypatch.setattr(runner, 'phase', phase)
    monkeypatch.setattr(runner, 'resolve_settings', lambda *_: cfg)
    monkeypatch.setattr(runner, 'signed_contract', lambda *_: ('mock', {}))
    monkeypatch.setattr(runner, 'create_configured', create)
    monkeypatch.setattr(runner, 'register_geometry', lambda *_: geometry)
    monkeypatch.setattr(runner, 'release_geometry', lambda *_: None)
    monkeypatch.setattr(runner.threading, 'Thread', lambda **_: SimpleNamespace(start=lambda: None))
    monkeypatch.setattr(ray_geometry_cpu, 'build_cells', lambda _: None)
    monkeypatch.setattr(ray_geometry_cpu, 'cache_complete', lambda _: False)
    monkeypatch.setattr(ray_geometry_cpu, 'ensure_column', lambda _: 'mock')
    monkeypatch.setattr(device_profile, 'smi_device_name', lambda: None)
    monkeypatch.setattr(compile_cache, 'configure', lambda *_: None)
    monkeypatch.setattr(source_solver, 'Solver', lambda *a, **kw: SimpleNamespace(timings={'solve_worker': {}}))
    monkeypatch.setattr(runner, 'solve_chunk', lambda *_: (np.zeros((1, 1, 15)), {'evaluations': 1}))
    monkeypatch.setattr(runner, 'chunk_summary', lambda *_: {})
    monkeypatch.setattr(radiance_table, 'make_table', lambda *_: (np.zeros((1, 1, 1, 15)), None))
    monkeypatch.setenv('SOSFUSE2_DEVICE_PROFILE', 'split')
    monkeypatch.setenv('SOSFUSE2_POOL_LIMIT_GIB', '')
    monkeypatch.setenv('STRATO_ATTACH_CACHE', '')
    monkeypatch.setenv('SOSFUSE2_VRAM_TRACE', '0')
    monkeypatch.delenv('SOSFUSE2_EXPORT_MOMENT', raising=False)
    with pytest.raises(SecondChunkStarted):
        runner.run(run_args)
    assert events == ['admit', 'table_done', 'free', 'admit']


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
    with pytest.raises(RuntimeError, match='E_OUTPUT_IDENTITY') as caught:
        entry(run_args)
    assert f"E_OUTPUT_IDENTITY: {out}: signature differs (stored format='{version}')." in str(caught.value)
    assert 'use a separate output root or inspect and move the old output.' in str(caught.value)
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


def stale_source(run_args, state, monkeypatch):
    """A finished source job without a density record, so run_extra stops at the first comparison after reading it."""
    from sosfuse2.backends import compile_cache

    monkeypatch.setattr(compile_cache, 'configure', lambda *args: None)
    monkeypatch.setattr(runner.threading, 'Thread', lambda **kw: SimpleNamespace(start=lambda: None))
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'CUDA_CACHE_PATH'):
        # run_extra sets them. setenv records the earlier state, also of a name that was not set, so the fixture
        # puts it back; delenv records nothing for a name that is absent.
        monkeypatch.setenv(name, os.environ.get(name, ''))
    out = run_args.work_root / 'out' / 'source'
    out.mkdir(parents=True)
    source_dir = run_args.work_root / 'tmp' / 'source'
    source_dir.mkdir(parents=True)
    (source_dir / 'source.npz').touch()
    meta = dict(complete=True, contract=runner.CONTRACT_VERSION, physics=physics(), optics_state=optics_state(state),
                id=state['id'], settings=settings(state), backend='numpy', polarization_model='depolarizing',
                mode='reduced-validation', code_hashes={'module': 'current'})
    (out / 'meta.json').write_text(json.dumps(meta))
    return out / '.worker.lock'


def test_extra_elevations_stop_while_a_worker_writes_the_source(run_args, state, monkeypatch):
    import fcntl

    lock = stale_source(run_args, state, monkeypatch)
    with lock.open('a') as writer:  # run() holds this lock while it rewrites source.npz, table.npz and meta.json
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match=r'E_EXTRA_SOURCE: source job is owned by a running worker'):
            runner.run_extra(run_args)
    opened = []
    real_open = type(lock).open

    def refuse_append(path, mode='r', *args, **kwargs):  # a lock file of another user, or a read-only mount
        opened.append((path.name, mode))
        if path == lock and mode == 'a':
            raise OSError(errno.EROFS, 'read-only file system')
        return real_open(path, mode, *args, **kwargs)

    with lock.open('r') as writer:
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(type(lock), 'open', refuse_append)
        with pytest.raises(RuntimeError, match=r'E_EXTRA_SOURCE: source job is owned by a running worker'):
            runner.run_extra(run_args)
    assert ('.worker.lock', 'r') in opened  # the shared hold was taken on a file opened for reading


def test_extra_elevations_hold_the_source_against_a_rerun(run_args, state, monkeypatch):
    import fcntl

    lock = stale_source(run_args, state, monkeypatch)
    seen = {}

    def phase(_name, event, **_info):
        if event != 'source_reuse_start':
            return
        with lock.open('a') as other:  # a second derived job reads the same source
            fcntl.flock(other, fcntl.LOCK_SH | fcntl.LOCK_NB)
        with lock.open('a') as rerun:  # the source job started again under the same name
            try:
                fcntl.flock(rerun, fcntl.LOCK_EX | fcntl.LOCK_NB)
                seen['rerun'] = 'admitted'
            except BlockingIOError:
                seen['rerun'] = 'refused'

    monkeypatch.setattr(runner, 'phase', phase)
    with lock.open('a') as reader:  # another derived job already holds the source
        fcntl.flock(reader, fcntl.LOCK_SH | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='E_EXTRA_SOURCE: density profile missing'):  # the first stop after the read
            runner.run_extra(run_args)
        assert seen == {'rerun': 'refused'}
    seen.clear()
    with pytest.raises(RuntimeError, match='E_EXTRA_SOURCE: density profile missing'):
        runner.run_extra(run_args)  # alone: only this job's own hold can refuse the rerun
    assert seen == {'rerun': 'refused'}
    with lock.open('a') as rerun:  # released when the derived job ends, also when it ends with an error
        fcntl.flock(rerun, fcntl.LOCK_EX | fcntl.LOCK_NB)


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


def test_reduced_legacy_view_shape(tmp_path, state):
    cfg = settings(state)
    cfg['L'] = 32
    path = tmp_path / 'source.npz'
    source(path, state, cfg, 'v5')
    assert radiance_table.restored_settings(path, state, True, 'numpy') is None
    cfg['view_shape'] = [1, 2]
    source(path, state, cfg, 'v5')
    restored = radiance_table.restored_settings(path, state, True, 'numpy')
    _, _, el, az = radiance_table.restored_axes(restored)
    assert [len(el), len(az)] == cfg['view_shape']


def test_worker_provenance_names_the_running_install():
    # The loose record keeps the full path so a run can be matched to its install; the archive strips it (pack_state).
    solver = SimpleNamespace(
        solve=lambda max_eval, tol: (None, {'tolerance': tol}), timings={},
        kernel=SimpleNamespace(L=16, domain=SimpleNamespace(top_km=80)),
    )
    runner.solve_chunk(solver, 2, 1e-3)
    worker = solver.timings['solve_worker']
    assert Path(worker['file']).resolve() == paths.PKG / '__init__.py'


def test_block_size_alone_is_signed(state, monkeypatch):
    for name in ('SOSFUSE2_BLOCK_SIZE', 'SOSFUSE2_KERNEL_FLAGS', 'SOSFUSE2_GROUND_SH_L'):
        monkeypatch.delenv(name, raising=False)
    parser = runner.build_parser()
    default_args = parser.parse_args(['--reduced'])
    explicit_args = parser.parse_args(['--reduced', '--block-size', '256'])
    default = runner.resolve_settings(runner.settings_for(default_args), state)
    explicit = runner.resolve_settings(runner.settings_for(explicit_args), state)
    assert 'kernel' not in default
    assert explicit['kernel']['block_size'] == 256
    monkeypatch.setenv('SOSFUSE2_BLOCK_SIZE', '256')
    assert explicit == runner.resolve_settings(runner.settings_for(default_args), state)
    assert runner.signed_contract(state, default, 'numpy', True, 'depolarizing')[0] != runner.signed_contract(
        state, explicit, 'numpy', True, 'depolarizing'
    )[0]


@pytest.mark.parametrize('failure', ['source', 'table'])
def test_interrupted_rerun_invalidates_meta_before_replacing_source(tmp_path, state, failure):
    out, prepare = tmp_path / 'out', tmp_path / 'prepare'
    out.mkdir()
    prepare.mkdir()
    meta = out / 'meta.json'
    meta.write_text(json.dumps({'complete': True, 'signature': 'original'}))
    source = prepare / 'source.npz'
    source.write_bytes(b'old source')
    job = SimpleNamespace(
        prepare=prepare, out=out, meta_path=meta, s=SimpleNamespace(geo=SimpleNamespace(h=[0], chi=[0])),
        moments=[], fields=[np.ones((1, 1, 1))], wl=np.array([550.]), state=state,
        stats_all=[{'tolerance': 1e-3}], cfg={}, table=np.ones((1, 1, 1, 1)),
        hs=np.array([-3.]), el=np.array([40.]), az=np.array([0.]),
    )
    a = SimpleNamespace(backend='numpy', polarization_model='depolarizing')

    def save(target, **arrays):
        if failure == 'source' or 'radiance' in arrays:
            if hasattr(target, 'write'):
                target.write(b'partial')
            else:
                Path(target).write_bytes(b'partial')
            raise OSError('interrupted')
        np.savez(target, **arrays)

    with pytest.raises(OSError, match='interrupted'):
        runner._write_run_arrays(job, SimpleNamespace(concatenate=np.concatenate, array=np.array, savez=save), a)
    assert json.loads(meta.read_text()) == {'complete': False, 'signature': 'original'}
    if failure == 'source':
        assert source.read_bytes() == b'old source'
    else:
        with np.load(source) as result:
            np.testing.assert_allclose(result['source'], job.fields[0], rtol=1e-3, atol=0)
    assert list(prepare.iterdir()) == [source]


def test_block_size_option_alone_restores_on_the_legacy_grid_and_odd_values_are_refused(tmp_path, state, monkeypatch):
    """A source recorded with --block-size 256 alone restores from its settings even when it looks like the legacy
    grid (L 32, default axes), whose rebuild would take the kernel of the environment; a recorded block size the runner
    cannot record, or a kernel entry that is not an object, is E_SOURCE_CONTRACT."""
    monkeypatch.delenv('SOSFUSE2_BLOCK_SIZE', raising=False)
    monkeypatch.delenv('SOSFUSE2_KERNEL_FLAGS', raising=False)
    cfg = runner.resolve_settings(runner.settings_for(SimpleNamespace(reduced=True, settings_json=None, block_size=256)),
                                  state)
    assert cfg['kernel']['block_size'] == 256
    cfg['L'] = 32
    path = tmp_path / 'source.npz'
    solver = source(path, state, cfg, 'v5')
    restored = radiance_table.restored_settings(path, state, True, 'numpy')
    assert restored == cfg
    assert radiance_table.load_source(path, solver, [550.0], settings=restored, backend='numpy').shape == (1, 571, 1)
    for kernel in (dict(cfg['kernel'], block_size=64), 'bad', dict(cfg['kernel'], block_size=True)):
        source(path, state, dict(cfg, kernel=kernel), 'v5')
        with pytest.raises(ValueError, match='E_SOURCE_CONTRACT'):
            radiance_table.restored_settings(path, state, True, 'numpy')
        with pytest.raises(ValueError, match='E_SOURCE_CONTRACT'):
            radiance_table.load_source(path, solver, [550.0], settings=dict(cfg, kernel=kernel), backend='numpy')
