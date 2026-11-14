"""Live-worker detection for old and current state runner commands."""

from pathlib import Path

from sosfuse2.jobs import tables_from_sources


def test_live_workers_accepts_old_and_current_runner_paths(tmp_path, monkeypatch):
    proc = tmp_path / 'proc'
    commands = (
        ('run_gpu_state_v3.py', True),
        ('sosfuse2.run_gpu_state_v3', True),
        ('sosfuse2.jobs.run_state', True),
        ('jobs/run_state.py', True),
        ('/package/sosfuse2/jobs/run_state.py', True),
        ('/other/notjobs/run_state.py', False),
        ('/other/jobs/run_state.py.backup', False),
    )
    for index, (command, _) in enumerate(commands):
        folder = proc / str(index + 1)
        folder.mkdir(parents=True)
        args = ['python', command, '--id', str(index), '--work-root', str(tmp_path)]
        (folder / 'cmdline').write_bytes(b'\0'.join(arg.encode() for arg in args) + b'\0')
    monkeypatch.setattr(tables_from_sources, 'Path', lambda path: proc if path == '/proc' else Path(path))
    assert tables_from_sources.live_workers() == {
        (str(tmp_path.resolve()), str(index)) for index, (_, matches) in enumerate(commands) if matches
    }


def test_default_source_round_trip_and_density_records(tmp_path, monkeypatch):
    """A block-free NumPy state is reusable by both table command entry points."""
    import json
    import shutil
    import subprocess
    import sys

    import numpy as np

    from sosfuse2.atmosphere import density
    from sosfuse2.foundation import paths
    from sosfuse2.jobs import run_state

    state = next(s for s in run_state.read_states(paths.ASSETS / 'states_v2.json', True) if s['id'] == 'clear_34')
    assert 'density' not in state
    state['parameters']['o4'] = False
    state['stratosphere'] = dict(status='disabled')
    states = tmp_path / 'states.json'
    states.write_text(json.dumps(dict(states=[state])))
    for name in ('SOSFUSE2_SUN_WORKERS', 'SOSFUSE2_GROUND_WORKERS', 'SOSFUSE2_COLUMN_WORKERS'):
        monkeypatch.setenv(name, '1')
    monkeypatch.setenv('SOSFUSE2_OFFLINE', '1')

    def run(module, *args, code=0):
        proc = subprocess.run(
            [sys.executable, '-B', '-m', module, *map(str, args)],
            capture_output=True, text=True, timeout=600, env=paths.child_env(),
        )
        assert proc.returncode == code, proc.stdout + proc.stderr
        return proc.stdout + proc.stderr

    run('sosfuse2.jobs.run_state', '--states', states, '--id', state['id'], '--reduced', '--backend', 'numpy', '--threads', '2', '--work-root', tmp_path / 'solve')
    source = tmp_path / 'solve/tmp' / state['id'] / 'source.npz'
    batch_root = tmp_path / 'batch'
    batch_source = batch_root / 'tmp' / state['id'] / 'source.npz'
    batch_source.parent.mkdir(parents=True)
    shutil.copyfile(source, batch_source)
    batch_args = ('--states', states, '--work-roots', batch_root, '--reduced', '--backend', 'numpy', '--threads', '2')
    run('sosfuse2.jobs.tables_from_sources', *batch_args)
    cli = tmp_path / 'single/table.npz'
    cli.parent.mkdir()
    single_args = ('--states', states, '--id', state['id'], '--source', source, '--prepare', tmp_path / 'prepare', '--out', cli, '--reduced', '--backend', 'numpy')
    run('sosfuse2.outputs.radiance_table', *single_args)
    expected = density.build(state)['meta']
    for path in (batch_root / 'out' / state['id'] / 'meta.json', cli.with_name('meta.json')):
        assert json.loads(path.read_text())['density'] == expected
    with np.load(source) as saved:
        arrays = dict(saved)
    contract = json.loads(str(arrays['source_contract']))
    assert contract['state']['density'] == density.DEFAULTS
    contract['state'].pop('density')
    arrays['source_contract'] = np.array(json.dumps(contract))
    np.savez(source, **arrays)
    rejected_root = tmp_path / 'rejected'
    rejected_source = rejected_root / 'tmp' / state['id'] / 'source.npz'
    rejected_source.parent.mkdir(parents=True)
    shutil.copyfile(source, rejected_source)
    rejected_args = tuple(rejected_root if value == batch_root else value for value in batch_args)
    assert 'E_SOURCE_DENSITY' in run('sosfuse2.jobs.tables_from_sources', *rejected_args, code=1)
    assert 'E_SOURCE_DENSITY' in run('sosfuse2.outputs.radiance_table', *single_args, code=1)
