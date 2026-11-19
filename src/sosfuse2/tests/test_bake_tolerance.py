"""Probe solver arguments in the real state and persistent-service processes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from sosfuse2.foundation import paths
from sosfuse2.jobs import run_state
from sosfuse2.outputs import radiance_table
from sosfuse2.tests.test_identity_caches import cache_keys


def test_effective_tolerance_and_cache_keys(tmp_path, monkeypatch):
    def cfg(raw):
        return run_state.settings_for(SimpleNamespace(reduced=True, settings_json=SimpleNamespace(read_text=lambda: json.dumps(raw))))
    state = run_state.read_states(paths.ASSETS/'states_v1.json')[0]
    sign = lambda settings: run_state.signed_contract(state, settings, 'numpy', True, 'depolarizing')[0]
    assert sign(cfg({})) == sign(cfg({'anderson_tol': .001})) != sign(cfg({'anderson_tol': .0002}))
    for value in (True, None, '0.001', float('inf'), float('nan'), 1e-7, .011):
        with pytest.raises(ValueError, match='E_SETTINGS_TOL'):
            cfg({'anderson_tol': value})
    for value in (1e-6, 1e-2):
        assert cfg({'anderson_tol': value})['anderson_tol'] == value
    monkeypatch.setenv('SOSFUSE2_COLUMN_METHOD', 'table-v1')
    keys = cache_keys(tmp_path, monkeypatch)
    assert keys == cache_keys(tmp_path, monkeypatch)
    monkeypatch.setenv('SOSFUSE2_COLUMN_METHOD', 'exact')
    assert keys['geometry'] != cache_keys(tmp_path, monkeypatch)['geometry']


def test_source_and_derived_cache_require_original_tolerance(tmp_path):
    from sosfuse2.tests.test_regressions_110 import _source_fixture
    from sosfuse2.jobs.tables_from_sources import output_complete
    path = tmp_path/'source.npz'
    solver, cfg, arrays, contract = _source_fixture(path)
    (tmp_path/'table.npz').touch()
    pack = tmp_path/'pack'
    pack.touch()
    (tmp_path/'meta.json').write_text(json.dumps(dict(complete=True, settings=cfg, anderson_tol=.0002)))
    for original in (None, .001):
        contract['settings'] = dict(cfg, anderson_tol=.0002)
        contract.pop('anderson_tol', None)
        if original is not None:
            contract['anderson_tol'] = original
        np.savez(path, **arrays, source_contract=json.dumps(contract))
        with pytest.raises(ValueError, match='E_SOURCE_TOL'):
            radiance_table.load_source(path, solver, [550.], settings=contract['settings'], backend='numpy')
        with pytest.raises(ValueError, match='E_SOURCE_TOL'):
            radiance_table.restored_settings(path, solver.fuse_state, True, 'numpy')
        assert not output_complete(tmp_path, pack, path, solver.fuse_state, SimpleNamespace(reduced=True, backend='numpy'))


# Replace only the numerical iteration, inside the process actually importing the solver.
SPY = """
import json, os, runpy, sys
from pathlib import Path
from sosfuse2.solve.source_solver import Solver, source_size
def spy(self, max_eval, tolerance):
    with open(os.environ['TOL_CALLS'], 'a') as f:
        print(json.dumps([os.getpid(), max_eval, tolerance]), file=f, flush=True)
    field = self.xp.zeros((self.s.geo.ns, source_size(self.s), self.s.W), 'f4')
    return field, dict(tolerance=tolerance, evaluations=2, converged=[True]*self.s.W,
                       history=[dict(source_relative=[0.], observer_relative=[0.])])
Solver.solve = spy
runpy.run_module(sys.argv.pop(1), run_name='__main__')
"""


@pytest.mark.parametrize('persistent', [False, True])
def test_solve_arguments_cross_process(tmp_path, persistent):
    import tempfile
    with tempfile.TemporaryDirectory(prefix='tol-') as d, (tmp_path/'service.log').open('w') as log:
        root = Path(d)
        env = dict(paths.child_env(), TOL_CALLS=str(tmp_path/'calls'))
        cmd = [sys.executable, '-B', '-c', SPY]
        service = None
        if persistent:
            env.update(SOS_PERSISTENT_ROOT=d, SOS_PERSISTENT_KEY=os.urandom(32).hex())
            service = subprocess.Popen([*cmd, 'sosfuse2.jobs.pipeline_execution', d, 'numpy', str(root/'lock'), '1', '4'],
                                       env=env, stdout=log, stderr=log)
        try:
            if service:
                deadline = time.monotonic()+20
                while not (root/'ready').exists():
                    assert service.poll() is None and time.monotonic()<deadline, (tmp_path/'service.log').read_text()
                    time.sleep(.03)
            expected = [.001,.0002,.001] if persistent else [.001,.0002]
            for i, tol in enumerate(expected):
                out = tmp_path/str(i)
                args = ['--states', str(paths.ASSETS/'states_v1_usstd.json'), '--id','clear_01',
                        '--reduced','--backend','numpy','--threads','1','--work-root',str(out),'--max-eval','16']
                if persistent:
                    args += ['--gpu-lock', str(root/'lock')]
                if tol != .001:
                    settings = tmp_path/'settings.json'
                    settings.write_text(json.dumps({'anderson_tol':tol}))
                    args += ['--settings-json',str(settings)]
                proc = subprocess.run([*cmd, 'sosfuse2.jobs.run_state', *args], env=env, capture_output=True, text=True, timeout=120)
                assert proc.returncode == 0, proc.stdout[-1200:]+proc.stderr
                meta = json.loads((out/'out/clear_01/meta.json').read_text())
                observed = [json.loads(line) for line in (tmp_path/'calls').read_text().splitlines()]
                assert [row[2] for row in observed] == expected[:i+1]
                worker = meta['timings'][0]['solve_worker']
                assert observed[-1] == [worker['pid'],16,tol]
                assert meta['anderson_tol'] == meta['source_solve']['tolerance'] == tol
                assert Path(worker['file']).resolve() == paths.PKG/'__init__.py'
                if service:
                    assert worker['pid'] == service.pid
        finally:
            if service:
                service.terminate()
                service.wait(timeout=10)


def test_mirror_publication_kill_boundaries(tmp_path):
    import shutil
    from sosfuse2.jobs import cache_store as cs
    from sosfuse2.tests.test_cache_store import tree
    from sosfuse2.solve import ray_geometry as geometry
    from sosfuse2.solve.ray_geometry_cpu import cache_complete
    # A two-layer endpoint becomes usable only after BOTH its packed marker and source height exist.
    # Exercise the production writer's last rename on either side of SIGKILL.
    for after in (False, True):
        root = tmp_path/str(after)
        tree(root)
        units, _ = cs.scan(root)
        endpoint = next(u for u in units.values() if u.kind == 'endpoints')
        folder = root/endpoint.key
        height = folder/'source_height.json'
        height.unlink()
        p = os.fork()
        if p == 0:
            replace = os.replace
            def stop(src, dst):
                if after:
                    replace(src, dst)
                os.kill(os.getpid(), signal.SIGKILL)
            os.replace = stop
            geometry.publish_json(height, {'height':1.2})
            os._exit(1)
        assert os.waitpid(p, 0)[1] == signal.SIGKILL
        geo = SimpleNamespace(packed=lambda:geometry.PackedCache(folder/'packed'), column_path=True, cache=folder)
        assert cache_complete(geo) is after
        assert (endpoint.key in cs.scan(root)[0]) is after
        # Store publication for every transferable shape. Endpoint dependencies already exist.
        for unit in cs.ordered(units):
            if unit.requires and unit.requires != endpoint.key:
                continue
            dst = root/('destination-'+unit.kind)
            stage = root/('stage-'+unit.kind)
            if unit.kind == 'endpoints' and not height.exists():
                geometry.publish_json(height, {'height':1.2})
            for rel in unit.all():
                (stage/rel).parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(root/rel, stage/rel)
            if unit.requires:
                shutil.copytree(folder, dst/unit.requires, ignore=shutil.ignore_patterns('aerosol','host_cache'))
            p = os.fork()
            if p == 0:
                for method in ('rename','replace','link'):
                    original = getattr(os,method)
                    def stop(src, dest, *args, original=original, **kwargs):
                        last = (Path(dest) == dst/unit.key/'packed' if unit.kind == 'endpoints' else
                                Path(dest) == dst/(unit.files[0] if unit.files else unit.markers[0]))
                        if last and not after:
                            os.kill(os.getpid(), signal.SIGKILL)
                        result = original(src,dest,*args,**kwargs)
                        if last:
                            os.kill(os.getpid(), signal.SIGKILL)
                        return result
                    setattr(os, method, stop)
                cs.publish(unit,stage,dst)
                os._exit(2)
            assert os.waitpid(p,0)[1] == signal.SIGKILL, unit.kind
            assert (unit.key in cs.scan(dst)[0]) is after, unit.kind
