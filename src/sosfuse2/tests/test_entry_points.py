"""The front door `python -m sosfuse2 <command>`: its command table against the runnable modules, its dispatch, and
its refusal of unknown commands before a computing module is loaded."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from sosfuse2 import __main__ as front
from sosfuse2.foundation import paths

MAIN_TESTS = {"__name__ == '__main__'", "'__main__' == __name__"}


def runnable(path):
    """True when the file has a top-level `if __name__ == '__main__':` block that does more than set
    sys.dont_write_bytecode. solve/ray_geometry_cpu.py has only such a block: it is imported, never run, and is no
    command."""
    for node in ast.parse(path.read_bytes()).body:
        if isinstance(node, ast.If) and ast.unparse(node.test) in MAIN_TESTS:
            if any(ast.unparse(statement) != 'sys.dont_write_bytecode = True' for statement in node.body):
                return True
    return False


def module_file(module):
    return paths.PKG.joinpath(*module.split('.')[1:]).with_suffix('.py')


def python(*args):
    return subprocess.run([sys.executable, '-B', *args], capture_output=True, text=True, timeout=120, env=paths.child_env())


def test_command_names_are_module_names():
    rows = [module for _, group in front.GROUPS for module, _ in group]
    assert len(rows) == len(set(rows)) == len(front.COMMANDS)
    for name, module in front.COMMANDS.items():
        assert module.split('.') == ['sosfuse2', module.split('.')[1], name], (name, module)


@pytest.mark.parametrize('name', sorted(front.COMMANDS))
def test_command_runs_a_module(name):
    path = module_file(front.COMMANDS[name])
    assert path.is_file(), f'{name}: no module file {path.relative_to(paths.PKG)}'
    assert runnable(path), f'{name}: {path.relative_to(paths.PKG)} has no `__main__` block'


def test_every_runnable_module_has_a_command():
    found = set()
    for path in paths.PKG.rglob('*.py'):
        rel = path.relative_to(paths.PKG)
        if rel.parts[0] == 'tests' or rel == Path('__main__.py') or path.name.startswith('._'):
            continue
        if runnable(path):
            found.add('.'.join(('sosfuse2', *rel.with_suffix('').parts)))
    assert found - set(front.COMMANDS.values()) == set(), 'add these modules to the table in sosfuse2/__main__.py'


@pytest.mark.parametrize('args', [('C3_nostrato',), ('no such config',), ('--help',)])
def test_dispatch_passes_arguments_and_exit_status(args):
    """flags_env reads its first argument as a config name: a config that exists prints exports (status 0); a name
    with spaces or one that looks like an option is passed on unchanged, and the module stops with E_CONFIG_UNKNOWN
    on stderr and an empty stdout (status 1, no traceback) exactly as when it is run directly."""
    door = python('-m', 'sosfuse2', 'flags_env', *args)
    direct = python('-m', front.COMMANDS['flags_env'], *args)
    assert (door.returncode, door.stdout, door.stderr) == (direct.returncode, direct.stdout, direct.stderr)
    if args == ('C3_nostrato',):
        assert door.returncode == 0 and door.stdout.startswith('export SOSFUSE2_BLOCK_SIZE=')
    else:
        assert door.returncode == 1 and f'backends/configs/{args[0]}.json' in door.stderr
        assert door.stdout == '' and door.stderr.startswith('E_CONFIG_UNKNOWN: ') and 'Traceback' not in door.stderr


def loaded_modules(stderr):
    """Module names from the `-X importtime` lines of a run."""
    return {
        line.rpartition('|')[2].strip()
        for line in stderr.splitlines()
        if line.startswith('import time:') and not line.rstrip().endswith('imported package')
    }


@pytest.mark.parametrize(
    'args, status', [((), 0), (('--help',), 0), (('-h',), 0), (('no_such_command',), 2), (('jobs.run_state',), 2),
                     (('sosfuse2.jobs.run_state',), 2), (('Run_State',), 2), (('--run_state',), 2)]
)
def test_listing_and_refusal_load_no_computing_module(args, status):
    run = python('-X', 'importtime', '-m', 'sosfuse2', *args)
    assert run.returncode == status, run.stderr[-2000:]
    loaded = loaded_modules(run.stderr)
    assert 'sosfuse2' in loaded
    assert {m for m in loaded if m.startswith('sosfuse2.')} <= {'sosfuse2.__main__'}
    assert 'numpy' not in loaded
    if status == 2:
        assert run.stdout == '' and f'E_UNKNOWN_COMMAND: {args[0]!r}' in run.stderr
        return
    for name in front.COMMANDS:
        assert f'\n  {name} ' in run.stdout, name
    public, _, internal = run.stdout.partition('\nInternal')
    assert 'pipeline_execution' in internal and 'pipeline_execution' not in public
