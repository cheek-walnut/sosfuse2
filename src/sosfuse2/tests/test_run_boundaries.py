"""Configuration and storage can be used without starting an execution."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from sosfuse2.atmosphere.components import physics
from sosfuse2.foundation import paths
from sosfuse2.jobs import run_state


def test_state_settings_without_output(monkeypatch):
    from sosfuse2.solve import configuration

    def unexpected(*args, **kwargs):
        pytest.fail('selecting a state started a thread or acquired a lock')
    monkeypatch.setattr(run_state.threading.Thread, 'start', unexpected)
    monkeypatch.setattr(run_state.fcntl, 'flock', unexpected)
    a = SimpleNamespace(states=paths.ASSETS / 'states_v2.json', id='clear_34', reduced=True, settings_json=None)
    state, cfg = configuration.state_settings(a)
    assert state['id'] == 'clear_34'
    assert cfg == run_state.resolve_settings(run_state.settings_for(a), state)


@pytest.mark.parametrize('entry', ['run', 'run_extra', 'geometry_only'])
def test_modes_use_configuration_before_output(entry, monkeypatch):
    from sosfuse2.solve import configuration

    class Selected(Exception):
        pass

    a = run_state.build_parser().parse_args(['--id', 'clear_34', '--reduced', '--backend', 'numpy', '--threads', '1'])
    del a.work_root
    select = configuration.state_settings

    def selected(*args, **kwargs):
        state, cfg = select(*args, **kwargs)
        assert state['id'] == a.id and cfg['nr'] == 4
        raise Selected

    monkeypatch.setattr(configuration, 'state_settings', selected)
    monkeypatch.setattr(run_state.shutil, 'which', lambda name: name)
    for key in run_state._THREAD_ENV:
        monkeypatch.setenv(key, '1')
    with pytest.raises(Selected):
        getattr(run_state, entry)(a)


def test_array_storage_and_source_reuse(tmp_path, monkeypatch):
    from sosfuse2.jobs import run_storage as storage

    out, prepare = tmp_path / 'out', tmp_path / 'tmp'
    out.mkdir()
    prepare.mkdir()
    h, chi, wl = np.array([0.1]), np.array([1.0]), np.array([550.0])
    field = np.arange(4, dtype='f4').reshape(1, 4, 1)[:, ::2]
    table = np.arange(4, dtype='f4').reshape(4, 1, 1, 1)[::2]
    hs, el, az = np.array([-3., -6.]), np.array([45.]), np.array([90.])
    optics = dict(no2_du=0.0, o4=False, n0=None)
    meta = dict(signature='same', complete=True, physics=physics(), optics_state=optics, density={})
    source = prepare / 'source.npz'

    def write():
        storage.write_run_arrays(
            out, prepare, table, wl, hs, el, az, h=h, chi=chi, fields=[field], moments=[field], order=16,
            physical=physics(), optics=optics, contract={'state': {}},
        )
        storage.atomic_json(out / 'meta.json', meta)

    with storage.output_lock(out):
        write()
        with pytest.raises(RuntimeError, match='E_EXTRA_SOURCE: source job is owned'):
            with storage.source_lock(out):
                pytest.fail('writer admitted a reader')
    with storage.source_lock(out), storage.source_lock(out):
        assert storage.read_source_metadata(out, source, optics, {'complete': True}) == meta
        with np.load(source) as data:
            np.testing.assert_array_equal(data['source'], field)
        with np.load(prepare / 'moment.npz') as data:
            np.testing.assert_array_equal(data['moment'], field)
            np.testing.assert_array_equal(data['h'], h)
            np.testing.assert_array_equal(data['chi'], chi)
            assert data['L'] == 16
        with np.load(out / 'table.npz') as data:
            np.testing.assert_array_equal(data['radiance'], table)
        with pytest.raises(RuntimeError, match='E_OUTPUT_BUSY'):
            with storage.output_lock(out):
                pytest.fail('readers admitted a writer')

    with storage.output_lock(out):
        storage.check_output(out, 'same', optics)
        with pytest.raises(RuntimeError, match='E_OUTPUT_IDENTITY'):
            storage.check_output(out, 'other', optics)
        with monkeypatch.context() as m:
            def interrupted(*args, **kwargs):
                raise OSError('interrupted')
            m.setattr(np, 'savez', interrupted)
            with pytest.raises(OSError, match='interrupted'):
                write()
        assert json.loads((out / 'meta.json').read_text()) == dict(meta, complete=False)
        storage.check_output(out, 'same', optics)
        write()
        assert json.loads((out / 'meta.json').read_text()) == meta


@pytest.mark.parametrize('entry,expected', [
    ('run', 'E_DENSITY_PARTIAL'), ('run_extra', 'E_DENSITY_PARTIAL'), ('geometry_only', 'E_SETTINGS_KEYS'),
])
def test_state_and_settings_error_priority(entry, expected, tmp_path, monkeypatch):
    invalid = tmp_path / 'settings.json'
    invalid.write_text('{"unknown_grid_option": 1}')
    a = run_state.build_parser().parse_args(['--reduced', '--backend', 'numpy', '--threads', '1'])
    a.settings_json = invalid
    a.density_profile = 'exp8'
    monkeypatch.setattr(run_state.shutil, 'which', lambda name: name)
    for key in run_state._THREAD_ENV:
        monkeypatch.setenv(key, '1')
    with pytest.raises(ValueError, match=expected):
        getattr(run_state, entry)(a)


def test_each_assembly_looks_up_the_engine_loader(tmp_path, monkeypatch):
    from sosfuse2.solve import source_solver

    a = run_state.build_parser().parse_args(['--reduced', '--backend', 'numpy'])
    state = next(s for s in run_state.read_states(a.states, True) if s['id'] == a.id)
    cfg = run_state.resolve_settings(run_state.settings_for(a), state)

    class EngineRequested(Exception):
        pass

    for attempt in (1, 2):
        def load_engine(index=attempt):
            raise EngineRequested(index)
        monkeypatch.setattr(source_solver, 'load_engine', load_engine)
        with pytest.raises(EngineRequested) as seen:
            run_state.create_configured(tmp_path, cfg, [550.], 'numpy', state)
        assert seen.value.args == (attempt,)
