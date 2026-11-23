"""Options of jobs/run_state.py and jobs/pipeline.py that set environment variables (backends/flags_env.py).

Each option writes the variable the package reads, so the settings derived from an option equal the settings derived
from the variable typed by hand.
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
