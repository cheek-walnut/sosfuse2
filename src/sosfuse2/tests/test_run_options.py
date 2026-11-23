"""Options of jobs/run_state.py and jobs/pipeline.py that set environment variables (backends/flags_env.py).

Each option writes the variable the package reads, so the settings derived from an option equal the settings derived
from the variable typed by hand.

The defaults: a command with no argument, or with some of them, resolves to the same arguments as the command typed in
full.
"""

import argparse

import pytest

from sosfuse2.backends import flags_env

NAMES = (
    'SOSFUSE2_KERNEL_FLAGS',
    'SOSFUSE2_BLOCK_SIZE',
    'SOSFUSE2_GROUND_SH_L',
    'SOSFUSE2_DEVICE_PROFILE',
    'SOSFUSE2_COMPILE_CACHE',
    'STRATO_ATTACH_CACHE',
)


def parse(*argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--block-size', type=int, choices=(128, 256))
    flags_env.add_run_options(parser)
    return parser.parse_args(argv)


def test_no_option_leaves_the_environment_unchanged():
    environ = {'SOSFUSE2_KERNEL_FLAGS': 'X', 'OTHER': '1'}
    flags_env.apply_run_options(parse(), environ)
    assert environ == {'SOSFUSE2_KERNEL_FLAGS': 'X', 'OTHER': '1'}


@pytest.mark.parametrize('name', ['C3', 'C3_nostrato'])
def test_config_sets_flags_and_block_size_like_the_export_line(name):
    config = flags_env.load(name)
    environ = {'SOSFUSE2_KERNEL_FLAGS': 'OLD', 'SOSFUSE2_BLOCK_SIZE': '128'}
    flags_env.apply_run_options(parse('--config', name), environ)
    assert environ['SOSFUSE2_KERNEL_FLAGS'] == ','.join(sorted(config['kernel_flags']))
    assert environ['SOSFUSE2_BLOCK_SIZE'] == str(config['block_size'])


def test_unknown_config_and_block_size_conflict_are_errors():
    with pytest.raises(ValueError, match='E_CONFIG_UNKNOWN'):
        flags_env.apply_run_options(parse('--config', 'no_such_config'), {})
    other = 128 if flags_env.load('C3')['block_size'] == 256 else 256
    environ = {}
    with pytest.raises(ValueError, match='E_KERNEL_BLOCK_SIZE'):
        flags_env.apply_run_options(parse('--config', 'C3', '--block-size', str(other)), environ)
    assert environ == {}


def test_each_option_overrides_its_variable(tmp_path):
    environ = dict.fromkeys(NAMES, 'before')
    args = parse(
        '--ground-sh', '24', '--device-profile', 'cuda', '--compile-cache', str(tmp_path / 'nvrtc'),
        '--attach-cache', str(tmp_path / 'attach'),
    )
    flags_env.apply_run_options(args, environ)
    assert environ['SOSFUSE2_GROUND_SH_L'] == '24'
    assert environ['SOSFUSE2_DEVICE_PROFILE'] == 'cuda'
    assert environ['SOSFUSE2_COMPILE_CACHE'] == str((tmp_path / 'nvrtc').resolve())
    assert environ['STRATO_ATTACH_CACHE'] == str((tmp_path / 'attach').resolve())
    assert environ['SOSFUSE2_KERNEL_FLAGS'] == environ['SOSFUSE2_BLOCK_SIZE'] == 'before'


@pytest.mark.parametrize('text', ['off', 'OFF', ' Off '])
def test_attach_cache_off(text):
    environ = {}
    flags_env.apply_run_options(parse('--attach-cache', text), environ)
    assert environ == {'STRATO_ATTACH_CACHE': 'off'}


def test_kernel_settings_from_options_equal_those_from_variables(monkeypatch):
    from sosfuse2.jobs import run_state

    for name in NAMES:
        monkeypatch.delenv(name, raising=False)
    config = flags_env.load('C3')
    monkeypatch.setenv('SOSFUSE2_KERNEL_FLAGS', ','.join(config['kernel_flags']))
    monkeypatch.setenv('SOSFUSE2_BLOCK_SIZE', str(config['block_size']))
    monkeypatch.setenv('SOSFUSE2_GROUND_SH_L', '24')
    typed = run_state.kernel_settings()
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)
    environ = {}
    flags_env.apply_run_options(parse('--config', 'C3', '--ground-sh', '24'), environ)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    assert run_state.kernel_settings() == typed
    assert typed['ground_sh']['order'] == 24


def test_both_parsers_accept_the_options(tmp_path):
    from sosfuse2.jobs import pipeline, run_state

    options = ['--config', 'C3', '--ground-sh', '16', '--device-profile', 'cuda', '--compile-cache', str(tmp_path),
               '--attach-cache', 'off']
    a = run_state.build_parser().parse_args(['--id', 'clear_34', '--work-root', str(tmp_path), *options])
    b = pipeline.parser().parse_args(['--states', str(tmp_path / 'states.json'), *options])
    for args in (a, b):
        assert (args.config, args.ground_sh, args.device_profile, args.attach_cache) == ('C3', 16, 'cuda', 'off')
        assert args.compile_cache == str(tmp_path)


@pytest.mark.parametrize('text', ['', '  '])
def test_blank_cache_options_select_the_default_like_an_empty_variable(text):
    environ = {'SOSFUSE2_COMPILE_CACHE': '/before', 'STRATO_ATTACH_CACHE': '/before'}
    flags_env.apply_run_options(parse('--compile-cache', text, '--attach-cache', text), environ)
    assert environ == {'SOSFUSE2_COMPILE_CACHE': '', 'STRATO_ATTACH_CACHE': ''}


def test_run_options_environment_restores_the_variables():
    environ = {'SOSFUSE2_GROUND_SH_L': '16', 'OTHER': '1'}
    with flags_env.run_options_environment(parse('--ground-sh', '24', '--config', 'C3'), environ):
        assert environ['SOSFUSE2_GROUND_SH_L'] == '24' and 'SOSFUSE2_KERNEL_FLAGS' in environ
    assert environ == {'SOSFUSE2_GROUND_SH_L': '16', 'OTHER': '1'}
    unknown = parse('--ground-sh', '24', '--config', 'no_such_config')
    with pytest.raises(ValueError, match='E_CONFIG_UNKNOWN'), flags_env.run_options_environment(unknown, environ):
        raise AssertionError('the block must not run')
    assert environ == {'SOSFUSE2_GROUND_SH_L': '16', 'OTHER': '1'}


def test_pipeline_main_leaves_the_environment_as_it_was(tmp_path, monkeypatch, capsys):
    import os

    from sosfuse2.jobs import pipeline

    for name in flags_env.RUN_OPTION_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('SOSFUSE2_GROUND_SH_L', '16')
    before = {name: os.environ.get(name) for name in flags_env.RUN_OPTION_VARIABLES}
    # no --ids and no --plan: the preflight fails after the options were applied
    with pytest.raises(SystemExit):
        pipeline.main(['--states', str(tmp_path / 'states.json'), '--ground-sh', '24', '--config', 'C3'])
    assert 'E_PIPELINE_ARGUMENTS' in capsys.readouterr().err
    assert {name: os.environ.get(name) for name in flags_env.RUN_OPTION_VARIABLES} == before


def no_cuda():
    return False


def never():
    raise AssertionError('a typed backend must not look for a device')


@pytest.mark.parametrize(
    'typed, cuda, full_numpy, expected, note_count',
    [
        (('auto', False), lambda: True, False, ('cupy', False), 1),
        (('auto', True), lambda: True, False, ('cupy', True), 1),
        (('auto', False), no_cuda, False, ('numpy', True), 2),
        (('auto', True), no_cuda, False, ('numpy', True), 1),
        (('numpy', False), never, False, ('numpy', True), 1),
        (('numpy', True), never, False, ('numpy', True), 0),
        (('numpy', False), never, True, ('numpy', False), 0),
        (('cupy', False), never, False, ('cupy', False), 0),
        (('cupy', True), never, False, ('cupy', True), 0),
    ],
)
def test_resolve_backend_fills_only_what_was_not_typed(typed, cuda, full_numpy, expected, note_count):
    backend, reduced, notes = flags_env.resolve_backend(*typed, full_numpy=full_numpy, cuda=cuda)
    assert (backend, reduced) == expected
    assert len(notes) == note_count


def test_cuda_device_present(monkeypatch):
    from sosfuse2.backends import device_profile

    cupy_installed = [False]
    monkeypatch.setattr(flags_env.importlib.util, 'find_spec', lambda name: object() if cupy_installed[0] else None)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(flags_env.sys, 'platform', 'linux')
    monkeypatch.setattr(flags_env.shutil, 'which', lambda name: None)
    monkeypatch.setattr(device_profile, 'smi_device_name', never)
    assert flags_env.cuda_device_present() is False
    # CuPy without nvidia-smi: a CUDA machine is possible, so no silent choice of the reduced NumPy table
    cupy_installed[0] = True
    with pytest.raises(ValueError, match='E_BACKEND_AUTO: CuPy is installed'):
        flags_env.cuda_device_present()
    monkeypatch.setattr(flags_env.shutil, 'which', lambda name: '/usr/bin/' + name)
    monkeypatch.setattr(device_profile, 'smi_device_name', lambda: 'some device')
    assert flags_env.cuda_device_present() is True
    monkeypatch.setattr(device_profile, 'smi_device_name', lambda: None)
    with pytest.raises(ValueError, match='E_BACKEND_AUTO: nvidia-smi is installed'):
        flags_env.cuda_device_present()
    # every device hidden: nvidia-smi would still answer for device 0
    monkeypatch.setattr(device_profile, 'smi_device_name', never)
    for blank in ('', '  '):
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', blank)
        assert flags_env.cuda_device_present() is False
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    monkeypatch.setattr(flags_env.sys, 'platform', 'darwin')
    assert flags_env.cuda_device_present() is False


def test_pipeline_asks_for_ids_before_it_looks_for_a_device(monkeypatch, capsys):
    from sosfuse2.jobs import pipeline

    monkeypatch.setattr(flags_env, 'cuda_device_present', never)
    with pytest.raises(SystemExit):
        pipeline.main([])
    assert '--ids or --plan required' in capsys.readouterr().err


def test_plan_queue_commands_name_the_backend():
    import inspect

    from sosfuse2.jobs import plan_queue

    assert "'--backend',\n        'cupy'" in inspect.getsource(plan_queue.commands)


def resolved_run_state(monkeypatch, tmp_path, *argv):
    """The arguments run_state.main hands to run() for `argv`, on a machine without a CUDA device, started in tmp_path."""
    import sys

    from sosfuse2.jobs import run_state

    seen = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(flags_env, 'cuda_device_present', no_cuda)
    monkeypatch.setattr(run_state, 'run', lambda a: seen.update(vars(a)))
    monkeypatch.setattr(sys, 'argv', ['run_state', *argv])
    for name in ('SOSFUSE2_COLUMN_CACHE_JOB', 'SOSFUSE2_STATE_ID', 'STRATO_ATTACH_CACHE'):
        monkeypatch.setenv(name, '')  # main writes these; monkeypatch restores them
    run_state.main()
    return seen


FULL = ('--id', 'clear_34', '--work-root', 'work', '--backend', 'numpy', '--reduced')


def test_run_state_without_arguments_is_the_quick_start_command(monkeypatch, tmp_path, capsys):
    full = resolved_run_state(monkeypatch, tmp_path, *FULL)
    assert capsys.readouterr().err == ''
    bare = resolved_run_state(monkeypatch, tmp_path)
    assert bare == full
    assert (bare['id'], bare['backend'], bare['reduced']) == ('clear_34', 'numpy', True)
    assert bare['work_root'] == (tmp_path / 'work').resolve()
    err = capsys.readouterr().err
    assert '--backend auto: numpy' in err and '--reduced is on' in err


@pytest.mark.parametrize(
    'argv',
    [
        ('--id', 'clear_34'),
        ('--work-root', 'work'),
        ('--backend', 'numpy'),
        ('--reduced',),
        ('--backend', 'numpy', '--work-root', 'work'),
    ],
)
def test_run_state_with_some_arguments_equals_the_full_command(monkeypatch, tmp_path, argv):
    assert resolved_run_state(monkeypatch, tmp_path, *argv) == resolved_run_state(monkeypatch, tmp_path, *FULL)


def test_run_state_typed_values_are_kept(monkeypatch, tmp_path):
    a = resolved_run_state(monkeypatch, tmp_path, '--id', 'clear_40', '--work-root', str(tmp_path / 'elsewhere'))
    assert (a['id'], a['work_root']) == ('clear_40', (tmp_path / 'elsewhere').resolve())


def test_pipeline_defaults(monkeypatch, capsys):
    from sosfuse2.foundation import paths
    from sosfuse2.jobs import pipeline

    monkeypatch.setattr(flags_env, 'cuda_device_present', no_cuda)
    a = pipeline.parser().parse_args([])
    assert a.states == paths.ASSETS / 'states_v2.json' and a.states.is_file()
    assert (str(a.work_root), a.backend, a.reduced, a.ids) == ('work', 'auto', False, None)
    a = pipeline.fill_backend(a)
    assert (a.backend, a.reduced) == ('numpy', True)
    assert pipeline.fill_backend(a) is a and (a.backend, a.reduced) == ('numpy', True)
    # the queue itself is not defaulted: without --ids or --plan nothing starts
    with pytest.raises(SystemExit):
        pipeline.main([])
    assert '--ids or --plan required' in capsys.readouterr().err


def test_pipeline_validate_with_only_ids(monkeypatch, tmp_path):
    from sosfuse2.jobs import pipeline

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(flags_env, 'cuda_device_present', no_cuda)
    a = pipeline.validate(pipeline.parser().parse_args(['--ids', 'clear_34', '--workers', '1']))
    assert (a.ids, a.backend, a.reduced) == (['clear_34'], 'numpy', True)
    assert a.work_root == (tmp_path / 'work').resolve()
