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
