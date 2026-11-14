"""Front door: `python -m sosfuse2 <command> [arguments ...]`.

A command is the name of a runnable module, and the front door replaces its own process with
`python -B -m sosfuse2.<layer>.<command> [arguments ...]` on the same interpreter. The arguments, the standard streams
and the exit status are therefore those of the module. This file imports only the standard library, so an unknown
command stops with exit status 2 before any computing module is loaded. Each module stays runnable directly with
`python -B -m sosfuse2.<layer>.<command>`.
"""

import os
import sys

# The command table. Each row is (module under sosfuse2, one-line summary); the command is the last part of the module
# name. tests/test_entry_points.py keeps this table equal to the modules that have a `__main__` block.
GROUPS = (
    ('Solve and tables', (
        ('jobs.run_state', 'solve one state and write its table and meta.json'),
        ('outputs.radiance_table', 'table from a stored source field'),
        ('jobs.tables_from_sources', 'tables for the stored source fields of a work root'),
    )),
    ('Queues and preparation', (
        ('jobs.pipeline', 'queue of states with overlapping preparation'),
        ('jobs.plan_queue', 'CPU work and cache hits that a queue would have'),
        ('jobs.cache_store', 'shared copy of the geometry, attach and column caches'),
        ('jobs.precompile', 'fill the NVRTC kernel cache before a queue starts'),
        ('asset_tools.prepare_minor_absorbers', 'prepare the O4 table in the work root, once'),
    )),
    ('Checks and diagnostics', (
        ('checks.gate', 'numerical gate against the fixed references'),
        ('checks.solver_diagnostics', 'solver self-test, benchmark and validation runs'),
        ('checks.kernel_diagnostics', 'kernel diagnostic records (not a gate)'),
        ('checks.stratosphere_checks', 'stratospheric layer measurements and checks'),
        ('foundation.available_cpus', 'cores this process may use'),
    )),
    ('Assets and kernel settings', (
        ('asset_tools.generate_density_profile', 'US 1976 density and ozone-shape profile asset'),
        ('asset_tools.generate_minor_absorbers', 'NO2 asset and O4 table from their source files'),
        ('asset_tools.generate_strato_optics', 'stratospheric sulfate Mie table'),
        ('asset_tools.generate_tropo_optics', 'tropospheric aerosol Mie table'),
        ('asset_tools.strato_states', 'nine sulfate diagnostic states'),
        ('backends.flags_env', 'shell exports for a kernel config (default C3)'),
    )),
    ('Reading the code', (
        ('jobs.walkthrough', 'follow one small state through the solver'),
    )),
    ('Internal (started by another command, not run by hand)', (
        ('jobs.pipeline_execution', 'GPU service of `pipeline --persistent`'),
    )),
)
COMMANDS = {module.rpartition('.')[2]: f'sosfuse2.{module}' for _, rows in GROUPS for module, _ in rows}

USAGE = 'usage: python -m sosfuse2 <command> [arguments ...]'
INTRO = (
    'Runs `python -B -m sosfuse2.<layer>.<command> [arguments ...]` and exits with its status. The arguments go to\n'
    'that module unchanged; its options are described in the module. The layer is the folder of the module file.'
)


def listing():
    width = max(map(len, COMMANDS)) + 2
    lines = [USAGE, '', INTRO]
    for title, rows in GROUPS:
        lines += ['', f'{title}:']
        lines += [f'  {m.rpartition(".")[2]:<{width}}{m.partition(".")[0]:<13}{summary}' for m, summary in rows]
    return '\n'.join(lines)


def main(argv):
    if not argv or argv[0] in ('-h', '--help'):
        print(listing())
        return 0
    module = COMMANDS.get(argv[0])
    if module is None:
        refusal = f'E_UNKNOWN_COMMAND: {argv[0]!r}; `python -m sosfuse2 --help` lists the commands'
        print(f'{USAGE}\n{refusal}', file=sys.stderr)
        return 2
    sys.stdout.flush()
    sys.stderr.flush()
    # exec, not a child process: the process keeps the module's own command line (tables_from_sources.live_workers
    # recognises a running solve by it) and its exit status.
    os.execv(sys.executable, [sys.executable, '-B', '-m', module, *argv[1:]])


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
