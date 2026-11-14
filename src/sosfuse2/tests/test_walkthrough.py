"""Walkthrough (jobs/walkthrough.py): the static map points at real code, a failed run keeps the old page, and the table
equals the one the production runner makes for the same state.

Run:
  python -B -m pytest -q -rs src/sosfuse2/tests/test_walkthrough.py
  SOSFUSE2_WALKTHROUGH_SLOW=1 python -B -m pytest -q src/sosfuse2/tests/test_walkthrough.py
      # adds the production comparison: the walkthrough and jobs/run_state.py, about 45 s each
"""

import ast
import json
import os
import shutil
import subprocess
import tempfile
from unittest import mock

import numpy as np
import pytest

from sosfuse2.foundation import paths
from sosfuse2.jobs import walkthrough


def defined(path, qualname):
    """True when `qualname` (a name or Class.method) is defined in `path`, read with ast (nothing is imported)."""
    body = ast.parse(path.read_text()).body
    for part in qualname.split('.'):
        kinds = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        node = next((n for n in body if isinstance(n, kinds) and n.name == part), None)
        if node is None:
            return False
        body = node.body
    return True


def test_static_map_points_at_real_code():
    assert len(walkthrough.STEPS) == 18
    refs = [r for _, _, rs, _ in walkthrough.STEPS for r in rs] + [r for _, rs, _ in walkthrough.NOT_RUN for r in rs]
    missing = []
    for ref in refs:
        file, _, name = ref.partition(':')
        if not (paths.PKG / file).is_file() or (name and not defined(paths.PKG / file, name)):
            missing.append(ref)
    assert not missing
    # The check itself can fail: a method that does not exist, a module-level name that is a method.
    assert not defined(paths.PKG / 'solve/source_solver.py', 'Solver.no_such_step')
    assert not defined(paths.PKG / 'solve/source_solver.py', 'initialize')


@pytest.mark.parametrize('where', ['run', 'render'])
def test_failed_run_keeps_the_page(tmp_path, monkeypatch, where):
    page = tmp_path / 'WALKTHROUGH.md'
    page.write_text('previous page\n')
    (tmp_path / 'tmp').mkdir()
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path / 'tmp'))

    def run(root):
        if where == 'run':
            raise RuntimeError('E_NOT_CONVERGED: injected')
        return dict(seconds={})  # incomplete record: render() fails

    monkeypatch.setattr(walkthrough, 'run_example', run)
    with mock.patch.dict(os.environ), pytest.raises((RuntimeError, KeyError)):
        walkthrough.main(['--write', str(page)])
    assert page.read_text() == 'previous page\n'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['WALKTHROUGH.md', 'tmp']
    assert not any((tmp_path / 'tmp').iterdir())  # the temporary work root is removed after a failure too


@pytest.mark.skipif(
    os.environ.get('SOSFUSE2_WALKTHROUGH_SLOW') != '1',
    reason='set SOSFUSE2_WALKTHROUGH_SLOW=1: the walkthrough and jobs/run_state.py take about 45 s each',
)
def test_table_matches_run_state(tmp_path):
    if shutil.which('zstd') is None:
        pytest.skip('zstd not found: jobs/run_state.py needs it for its archive')
    states = tmp_path / 'states.json'
    states.write_text(json.dumps(dict(version=2, states=[walkthrough.example_state()])))
    page, tool, prod = tmp_path / 'WALKTHROUGH.md', tmp_path / 'tool', tmp_path / 'prod'
    command = paths.module_command('jobs.walkthrough', '--work-root', tool, '--write', page)
    subprocess.run(command, env=paths.child_env(walkthrough.example_environment(tool)), check=True, timeout=600)
    prod.mkdir()
    args = ['--id', walkthrough.STATE_ID, '--states', states, '--reduced', '--backend', 'numpy', '--work-root', prod]
    command = paths.module_command('jobs.run_state', *args)  # same environment, its own empty work root
    subprocess.run(command, env=paths.child_env(walkthrough.example_environment(prod)), check=True, timeout=600)
    with np.load(tool / 'table.npz') as a, np.load(prod / 'out' / walkthrough.STATE_ID / 'table.npz') as b:
        assert a['radiance'].shape == b['radiance'].shape == (2, 4, 6, 1)
        for k in ('wavelength_nm', 'solar_elevation_deg', 'view_elevation_deg', 'relative_azimuth_deg'):
            np.testing.assert_array_equal(a[k], b[k])
        np.testing.assert_allclose(a['radiance'], b['radiance'], rtol=1e-3, atol=0)
    text = page.read_text()
    assert '(2, 4, 6, 1)' in text and str(tmp_path) not in text and str(paths.PKG) not in text
