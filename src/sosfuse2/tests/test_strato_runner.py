#!/usr/bin/env python3
"""Local enabled reduced solver -> table -> verified pack, plus source reuse.

No rig files are edited; all artifacts stay in the supplied module scratchpad.
"""

import argparse, copy, hashlib, io, json, os, subprocess, sys, tarfile, time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
from sosfuse2.strato_states import build
from sosfuse2.stratosphere import DEFAULT
from sosfuse2.run_gpu_state_v3 import SH_L
from sosfuse2.sh_layout import count

try:
    from sosfuse2.numerical_checks import assert_equivalent, within_tol
except ImportError:  # standalone research harness
    from numerical_checks import assert_equivalent, within_tol
from sosfuse2 import paths, testing


def run(out):
    out = testing.outside_package(out)
    out.mkdir(parents=True, exist_ok=True)
    state = build()['states'][1]
    states = out / 'states.json'
    states.write_text(json.dumps(dict(version=2, states=[state]), indent=2) + '\n')
    env = paths.child_env(dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    base = [
        *paths.module_command('run_gpu_state_v3'),
        '--id',
        state['id'],
        '--states',
        str(states),
        '--work-root',
        str(out / 'work'),
        '--backend',
        'numpy',
        '--reduced',
        '--threads',
        '2',
        '--geometry-layers',
        '2',
        '--max-eval',
        '64',
    ]
    rows = []
    start = time.monotonic()

    def command(name, args):
        log = out / (name + '.log')
        t = time.monotonic()
        with log.open('w') as f:
            p = subprocess.run([*base, *args], cwd=paths.SRC, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=240)
        print(name, 'exit', p.returncode, 'seconds', time.monotonic() - t, flush=True)
        if p.returncode:
            raise RuntimeError('E_RUNNER_TEST: ' + str(log))
        folder = out / 'work/out' / name
        meta = json.loads((folder / 'meta.json').read_text())
        assert meta['complete'] and meta['stratosphere']['status'] == 'enabled'
        assert meta['parameters_requested'] == state['parameters']
        assert meta['parameters_computed']['stratosphere_tau550'] == 0.003
        assert meta['stratosphere']['optics_sha256'] == hashlib.sha256(DEFAULT.read_bytes()).hexdigest()
        assert meta['stratosphere']['polarization'] == 'fully depolarizing'
        pack = out / 'work/pack' / (name + '.tar.zst')
        if not pack.exists():
            matches = list((out / 'work/pack').glob(name + '*'))
            assert len(matches) == 1
            pack = matches[0]
        data = subprocess.check_output(['zstd', '-q', '-dc', str(pack)])
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            archived = json.load(archive.extractfile('meta.json'))
        assert archived == meta
        rows.append(dict(name=name, pack=str(pack), bytes=pack.stat().st_size, signature=meta['signature'], pass_=True))
        return meta

    ordinary = command('enabled', ['--out-name', 'enabled'])
    source = out / 'work/tmp/enabled/source.npz'
    width = 10 + 2 * count(SH_L)  # stratosphere source: 10 + two SH blocks at the production L (1132 at L32)
    with np.load(source) as f:
        assert f['source'].shape == (40, width, 1) and np.isfinite(f['source']).all()
    derived = command('extra', ['--out-name', 'extra', '--source-out-name', 'enabled', '--extra-elevations=-3.5,-4.5'])
    for key in ('parameters_requested', 'parameters_computed', 'stratosphere', 'stratosphere_computed'):
        assert ordinary[key] == derived[key], key
    assert all(v['within_tol'] for v in derived['anchor_check'].values())
    # Same physical state enabled through the explicit CLI override.
    disabled = copy.deepcopy(state)
    disabled['stratosphere']['status'] = 'unsupported'
    states.write_text(json.dumps(dict(version=2, states=[disabled]), indent=2) + '\n')
    override = command('override', ['--out-name', 'override', '--stratosphere', 'on'])
    with np.load(out / 'work/out/enabled/table.npz') as a, np.load(out / 'work/out/override/table.npz') as b:
        assert within_tol(a['radiance'], b['radiance'])
    report = dict(
        pass_=True,
        cases=rows,
        enabled_source_shape=[40, width, 1],
        extra_anchors_within_tol=True,
        override_table_within_tol=True,
        seconds=time.monotonic() - start,
    )
    (out / 'runner_test.json').write_text(json.dumps(report, indent=2) + '\n')
    print('STRATO RUNNER PASS', flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=None)
    a = p.parse_args()
    run(a.out or testing.scratch('strato-runner'))
