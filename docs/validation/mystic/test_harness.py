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


@pytest.mark.parametrize('complete', [False, True])
def test_incomplete_inputs_require_opt_in(tmp_path, complete):
    work = tmp_path / 'work'
    out = work / 'out'
    out.mkdir(parents=True)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'state_ids': [], 'pixels': [], 'sza_deg': [], 'seed_ids': []}))
    (out / 'optics.json').write_text('{}')
    (out / 'full.csv').write_text('case_id\n')
    manifest = {
        'complete': complete,
        'optics_sha256': sha(out / 'optics.json'),
        'results_sha256': sha(out / 'full.csv'),
        'config_sha256': sha(config),
    }
    (out / 'manifest.json').write_text(json.dumps(manifest))
    script = Path(__file__).with_name('compare.py')
    cmd = [sys.executable, '-B', str(script), '--work', str(work), '--config', str(config)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    rejected = subprocess.run(cmd, env=env, capture_output=True, text=True)
    reason = 'E_ENGINE_MISSING' if complete else 'E_MC_INCOMPLETE'
    assert rejected.returncode != 0 and reason in rejected.stderr
    assert not (out / 'compare.json').exists()
    subprocess.run(cmd + ['--allow-partial'], env=env, check=True, capture_output=True, text=True)
    assert read(out / 'compare.json')['status'] == ('WAITING_ENGINE' if complete else 'PARTIAL')


@pytest.mark.parametrize('top, reason', [(120, 'E_MYSTIC_TOP'), (80, 'E_O4_ASSET_MISSING'), (80, 'E_STATE_MISSING')])
def test_rejected_optics_preserves_previous_settings(tmp_path, monkeypatch, top, reason):
    import export_optics

    work = tmp_path / 'work'
    out = work / 'out'
    out.mkdir(parents=True)
    previous = {'top_km': 80, 'wavelengths': 15}
    (work / 'settings.json').write_text(json.dumps(previous))
    (out / 'optics.json').write_text(json.dumps({'settings': previous}))
    candidate = read(DEFAULT_CONFIG)
    candidate['settings']['top_km'] = top
    assets = tmp_path / 'root'
    if reason == 'E_STATE_MISSING':
        from sosfuse2.checks.testkit import synthetic_o4

        synthetic_o4(assets)
        candidate['state_ids'].append('missing')
    config = tmp_path / 'candidate.json'
    config.write_text(json.dumps(candidate))
    monkeypatch.setenv('SOSFUSE2_WORK_ROOT', str(assets))
    monkeypatch.setattr(sys, 'argv', ['export_optics', '--work', str(work), '--config', str(config)])
    with pytest.raises(ValueError, match=reason):
        export_optics.main()
    assert read(work / 'settings.json') == read(out / 'optics.json')['settings'] == previous


@pytest.mark.parametrize('fault', [None, 'missing_worker', 'missing_timing', 'extra_timing', 'wrong_order'])
def test_engine_checks_timing_for_each_solve(monkeypatch, fault):
    import extract_engine
    from sosfuse2.atmosphere import density, stratosphere
    from sosfuse2.jobs import run_state

    monkeypatch.setattr(density, 'build', lambda st: {'meta': {}})
    monkeypatch.setattr(stratosphere, 'table_path', lambda st: Path('optics'))
    monkeypatch.setattr(stratosphere, 'path_label', lambda path: path.name)
    monkeypatch.setattr(extract_engine, 'sha', lambda path: 'hash')
    monkeypatch.setattr(run_state, 'signed_contract', lambda *args: ('signature', {}))
    settings = dict(observer_km=0.1, kernel={'ground_sh': {'order': 24}})
    state = dict(parameters={}, stratosphere={})
    chunk = dict(converged=[True], tolerance=0.001)
    meta = dict(
        complete=True, parameters={}, density={}, settings=settings, package_version='1.9.2',
        polarization_model='depolarizing',
        stratosphere=dict(optics_table='optics', optics_sha256='hash', polarization='fully depolarizing', status='enabled'),
        kernel=dict(deterministic=True), backend='cupy', signature='signature', source_solve=[chunk, chunk],
        timings=[dict(solve_worker={}, ground_sh_order=24) for _ in range(2)],
    )
    contract = dict(engine_version='1.9.2', tolerance=0.001)
    if fault is None:
        assert extract_engine.validate(meta, state, settings, contract)['signature'] == 'signature'
        return
    if fault == 'missing_worker':
        meta['timings'][1] = dict(ground_sh_order=0)
    elif fault == 'missing_timing':
        meta['timings'].pop()
    elif fault == 'extra_timing':
        meta['timings'].append(dict(solve_worker={}, ground_sh_order=24))
    else:
        meta['timings'][1]['ground_sh_order'] = 0
    with pytest.raises(ValueError, match='E_ENGINE_GROUND_SH'):
        extract_engine.validate(meta, state, settings, contract)


@pytest.mark.parametrize('stage', ['prepare', 'extract'])
def test_engine_rejects_mismatched_observer(tmp_path, monkeypatch, stage):
    import extract_engine
    import prepare_arm
    from common import dump
    from sosfuse2.jobs import run_state

    config = read(DEFAULT_CONFIG)
    settings = dict(config['settings'], observer_km=1, path_n=12, L=16)
    overrides = {'A0': {'observer_km': 1}}
    monkeypatch.setattr(run_state, 'settings_for', lambda _: dict(settings, kernel={'ground_sh': {'order': 24}}))
    dest = tmp_path / 'engine/A0'
    opts = tmp_path / 'out/optics.json'
    dump(opts, {})
    dump(tmp_path / 'states.json', {})
    monkeypatch.setenv('SOSFUSE2_WORK_ROOT', str(tmp_path))
    if stage == 'prepare':
        arms = tmp_path / 'arms.json'
        dump(arms, overrides)
        monkeypatch.setattr(sys, 'argv', ['prepare_arm', '--work', str(tmp_path), '--arms-file', str(arms), '--arm', 'A0'])
        with pytest.raises(ValueError, match='E_MYSTIC_OBSERVER'):
            prepare_arm.main()
        assert not (dest / 'settings.json').exists()
    else:
        contract = dict(settings=dict(settings, kernel={}), optics_sha256='digest', states_sha256='digest',
                        states_parent_sha256='digest')
        dump(dest / 'contract.json', contract)
        monkeypatch.setattr(extract_engine, 'sha', lambda path: 'digest')
        with pytest.raises(ValueError, match='E_MYSTIC_OBSERVER'):
            extract_engine.extract(tmp_path, config, {'settings': settings}, overrides, 'A0', [])


@pytest.mark.parametrize('mode', ['short_solar_grid', 'optimized_missing_coverage', 'zero_standard_error'])
def test_public_csv_validates_records(tmp_path, monkeypatch, mode):
    import make_csv

    script = Path(make_csv.__file__)
    ns = {'__name__': make_csv.__name__}
    exec(compile(script.read_text(), str(script), 'exec', optimize=int(mode != 'zero_standard_error')), ns)
    config = read(DEFAULT_CONFIG)
    cases = {
        f'{sid}_i{p["i"]}_j{p["j"]}_sza{sza:g}': (sid, p, sza)
        for sid in config['state_ids'] for p in config['pixels'] for sza in config['sza_deg']
    }
    settings = dict(nr=44, view_shape=[100, 180], solar_elevation_deg=list(range(6, -19, -1)),
                    path_n=12, L=16, observer_km=0.1, anderson_tol=0.001,
                    spectral=dict(mode='point', ozone_column='ozone_233K'), kernel=dict(ground_sh=dict(order=24)))
    if mode == 'short_solar_grid':
        settings['solar_elevation_deg'] = [-3, -6]
    profile = dict(aod550=0.1, angstrom=1, ssa=0.9, scale_height_km=1, g=0.6, ozone_du=300, albedo=0.2, stratosphere_tau550=0.01)
    states = {sid: dict(solar=[1.0] * 30, state=dict(parameters=profile)) for sid in config['state_ids']}
    docs = {
        'config.json': config,
        'optics.json': dict(states=states),
        'manifest.json': dict(complete=True, counts=dict(success=46080), config_sha256='sha', results_sha256='sha'),
        'manifest_compare.json': dict(
            outputs_sha256={'compare_long.csv': 'sha', 'engine.csv': 'sha'}, mystic_csv_sha256='sha',
            optics_sha256='sha', mc_manifest_sha256='sha', engine_pixels_sha256={'A0': 'sha'},
        ),
        'engine_pixels.A0.json': dict(
            cases=cases, contract=dict(settings=settings, engine_version='1.9.2', engine_config_id='A0', optics_sha256='sha')
        ),
    }
    mc, engine, comparisons = [], [], []
    for cid, (sid, p, sza) in cases.items():
        entry = dict(case_id=cid, engine_config_id='A0', state_id=sid, sza_deg=sza,
                     view_elev_deg=p['elevation_deg'], view_azim_deg=p['azimuth_deg'])
        estimate_row = dict(entry, status='OK', n='16', discarded_seeds='0', mean='1', se='0', ci_low='1', ci_high='1')
        comparisons.append(dict(estimate_row, metric='Y_ratio'))
        for wave in config['wavelength_nm']:
            engine.append(dict(entry, wavelength_nm=wave, status='ok', radiance=1))
            comparisons.append(dict(estimate_row, metric='spectral_ratio', wavelength_nm=wave))
            mc.extend(dict(case_id=cid, wavelength_nm=wave, seed=seed, status='ok', radiance=1)
                      for seed in config['seed_ids'])
    if mode == 'optimized_missing_coverage':
        engine.pop()
    csvs = {'full.csv': mc, 'engine.csv': engine, 'compare_long.csv': comparisons}
    ns.update(read=lambda path: docs[path.name], sha256=lambda path: 'sha', read_csv=lambda path: csvs[path.name])
    output = tmp_path / 'result.csv'
    monkeypatch.setattr(sys, 'argv', ['make_csv', '--work', str(tmp_path), '--out', str(output)])
    if mode == 'zero_standard_error':
        ns['main']()
        assert output.is_file()
    else:
        reason = 'E_PUBLIC_GRID' if mode == 'short_solar_grid' else 'E_ENGINE_COVERAGE'
        with pytest.raises(ValueError, match=reason):
            ns['main']()
        assert not output.exists()


def test_scheduler_retries_failed_jobs_and_keeps_attempts(tmp_path, monkeypatch):
    import concurrent.futures
    from types import SimpleNamespace

    import run_mystic

    config = read(DEFAULT_CONFIG)
    data = fixture_data(config)
    job, successful = build_jobs(config, data, 'identity', 'lrt', smoke=True)
    monkeypatch.setattr(run_mystic, 'H', tmp_path)
    monkeypatch.setattr(run_mystic, 'text_input', lambda *a, **kw: 'input')
    monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', concurrent.futures.ThreadPoolExecutor)
    calls = []
    failed = True

    def run(command, *, cwd, **kwargs):
        calls.append(cwd)
        if cwd.parent.parent.name == f'w{job["wi"]:02d}' and failed:
            return SimpleNamespace(returncode=1)
        for name in ('mc.rad.spc', 'mc.rad.std.spc'):
            (cwd / name).write_text('0 0 0 0 1\n')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, 'run', run)
    run_mystic.schedule([job, successful], data, 1, None)
    assert run_mystic.read_result(job)['status'] == 'failed'
    assert len(calls) == 3
    folder = run_mystic.folder_for(job)
    before = [read(folder / f'attempt_{i}/attempt.json') for i in range(2)]
    failed = False
    run_mystic.schedule([job, successful], data, 1, None)
    assert len(calls) == 3
    run_mystic.schedule([job, successful], data, 1, None, retry_failed=True)
    result = run_mystic.read_result(job)
    assert [attempt['status'] for attempt in result['attempts']] == ['failed', 'failed', 'ok']
    assert [read(folder / f'attempt_{i}/attempt.json') for i in range(2)] == before
    assert len(calls) == 4 and calls[-1] == folder / 'attempt_2'


def test_rejects_more_shards_than_geometries():
    config = read(DEFAULT_CONFIG)
    data = fixture_data(config)
    limit = len(config['state_ids']) * len(config['pixels']) * len(config['sza_deg'])
    assert build_jobs(config, data, 'identity', 'lrt', shard=(limit, limit))
    with pytest.raises(ValueError, match='E_SHARD_COUNT'):
        build_jobs(config, data, 'identity', 'lrt', shard=(1, limit + 1))
