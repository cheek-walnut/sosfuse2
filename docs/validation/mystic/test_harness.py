"""Offline checks for scheduling, inputs, statistics and state selection."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from common import DEFAULT_CONFIG, DENSITY, make_states, read, sha
from compare import T975, color, estimate
from inputs import text_input
from run_mystic import build_jobs


def fixture_data(config):
    return {'states': {sid: {'wavelengths_nm': config['wavelength_nm']} for sid in config['state_ids']}}


def key(job):
    return job['case'], job['wi'], job['seed_id']


def test_seed_formula_and_partitions():
    config = read(DEFAULT_CONFIG)
    data = fixture_data(config)
    jobs = build_jobs(config, data, 'identity', 'lrt')
    assert len(jobs) == 46080
    seeds = [j['seed'] for j in jobs]
    assert len(set(seeds)) == len(jobs)
    assert min(seeds) > 0 and max(seeds) + 1000 < 2147483647
    assert not set(seeds) & {seed + 1000 for seed in seeds}
    assert (
        next(j['seed'] for j in jobs if j['case'] == 'clear_43_i98_j89_sza93' and j['wi'] == 1 and j['seed_id'] == 11)
        == 10011
    )
    assert (
        next(j['seed'] for j in jobs if j['case'] == 'clear_80_i98_j89_sza93' and j['wi'] == 0 and j['seed_id'] == 11)
        == 3600011
    )
    all_keys = set(map(key, jobs))
    partitions = [set(map(key, build_jobs(config, data, 'identity', 'lrt', shard=(i, 6)))) for i in range(1, 7)]
    assert set.union(*partitions) == all_keys
    assert sum(map(len, partitions)) == len(all_keys)


def test_input_text(tmp_path):
    p = {'i': 77, 'j': 0, 'elevation_deg': 20.25, 'azimuth_deg': 0.5}
    data = {'states': {'x': {'wavelengths_nm': [550.0], 'state': {'parameters': {'albedo': 0.25}}}}}
    text = text_input(data, 'x', p, 93, 0, 1000, 10011, Path('lrt'), work=tmp_path)
    assert f'atmosphere_file {tmp_path}/inputs/x/wl_00/atmosphere.dat\n' in text
    assert 'wavelength 550 550\n' in text
    assert 'sza 93\n' in text and 'mc_randomseed 10011\n' in text and 'mc_photons 1000\n' in text
    assert 'mc_spherical 1D\n' in text and 'mc_polarisation\n' in text


def test_student_interval_and_negative_color():
    result = estimate([1.0, 3.0])
    assert result['status'] == 'UNRESOLVED' and result['n'] == 2
    assert result['se'] == pytest.approx(1.0)
    assert result['ci_low'] == pytest.approx(2 - T975[1])
    assert result['ci_high'] == pytest.approx(2 + T975[1])
    with pytest.raises(ValueError, match='E_COLOR'):
        color(np.array([-1.0, 2.0]), np.ones((2, 3)))


def test_bundled_states_and_density(tmp_path, monkeypatch):
    monkeypatch.setenv('SOSFUSE2_WORK_ROOT', str(tmp_path / 'root'))
    config = read(DEFAULT_CONFIG)
    source = make_states(tmp_path, config)
    bundle = read(source)
    generated = read(tmp_path / 'states.json')['states']
    expected = {s['id']: s for s in bundle['states']}
    assert {s['id'] for s in generated} == set(config['state_ids'])
    for state in generated:
        assert state['density'] == DENSITY
        assert {k: v for k, v in state.items() if k != 'density'} == {
            k: v for k, v in expected[state['id']].items() if k != 'density'
        }
    from sosfuse2.jobs.run_state import read_states

    # The engine reads the file together with its parent bundle.
    engine_ids = {s['id'] for s in read_states(tmp_path / 'states.json', True)}
    assert set(config['state_ids']) <= engine_ids


def test_partial_manifest_refusal_and_opt_in(tmp_path):
    work = tmp_path / 'work'
    out = work / 'out'
    out.mkdir(parents=True)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'state_ids': [], 'pixels': [], 'sza_deg': [], 'seed_ids': []}))
    (out / 'optics.json').write_text('{}')
    (out / 'full.csv').write_text('case_id\n')
    manifest = {
        'complete': False,
        'optics_sha256': sha(out / 'optics.json'),
        'results_sha256': sha(out / 'full.csv'),
        'config_sha256': sha(config),
    }
    (out / 'manifest.json').write_text(json.dumps(manifest))
    script = Path(__file__).with_name('compare.py')
    cmd = [sys.executable, '-B', str(script), '--work', str(work), '--config', str(config)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    rejected = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert rejected.returncode != 0 and 'E_MC_INCOMPLETE' in rejected.stderr
    subprocess.run(cmd + ['--allow-partial'], env=env, check=True, capture_output=True, text=True)
    assert read(out / 'compare.json')['status'] == 'PARTIAL'
