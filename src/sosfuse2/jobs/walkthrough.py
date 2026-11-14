"""Follow one small state through the real code: input -> create_configured -> Solver.solve -> make_table.

The example is the reduced NumPy path of jobs/run_state.py with the parameters of clear_34 (assets/states_v2.json),
O4 off and the stratospheric layer disabled, in an empty work root with the environment pinned. The driver repeats the
glue of run_state.main() and run() for that path (ScalarKernel, endpoint cache registration, the cell fill, s.ms, the
make_table arguments) and calls the same functions; it computes nothing itself and writes no archive (zstd is not
needed). tests/test_walkthrough.py compares its table with the one run_state makes for the same state.
--write renders docs/WALKTHROUGH.md (relative to the current directory); a failed run leaves an existing file as it was.
"""

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

if __name__ == '__main__':
    sys.dont_write_bytecode = True
from sosfuse2.foundation import paths

STATE_ID = 'walkthrough'
THREAD_VARIABLES = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'BLIS_NUM_THREADS',
)
# Set before NumPy is imported; inherited SOSFUSE2_* and STRATO_ATTACH_CACHE are removed first. Two threads = the
# run_state --threads default; one worker for each of the four preparation pools.
EXAMPLE_ENVIRONMENT = dict(
    {name: '2' for name in THREAD_VARIABLES},
    SOSFUSE2_OFFLINE='1',
    SOSFUSE2_COLUMN_WORKERS='1',
    SOSFUSE2_SUN_WORKERS='1',
    SOSFUSE2_GROUND_WORKERS='1',
    SOSFUSE2_GEOMETRY_WORKERS='1',
)

# Static map: read from the code, not traced. (step, what happens, 'file:function' pointers, boundary shown above or None)
STEPS = (
    ('input', 'Read the example state.', ('jobs/walkthrough.py:example_state',), 'input'),
    (
        'state',
        'Fill in the component amounts the state leaves out (NO2 column, O4 switch).',
        ('atmosphere/components.py:canonical_state',),
        'input',
    ),
    (
        'grid',
        'Choose the small calculation grid and the view directions.',
        ('jobs/run_state.py:settings_for', 'jobs/run_state.py:resolve_settings'),
        'input',
    ),
    ('assembly start', 'Create the per-state engine context and solver.', ('solve/engine.py:create',), None),
    (
        'density',
        'Build the height profiles of air, ozone and the minor absorbers.',
        ('atmosphere/density.py:build',),
        None,
    ),
    (
        'sun path',
        'Prepare the table of slant columns towards the sun.',
        ('rays/shell_segments.py:SunGeometry', 'precompute/host_cache.py:sun_geometry'),
        None,
    ),
    (
        'optical coefficients',
        'Turn wavelength and amounts into extinction and scattering coefficients.',
        ('atmosphere/extinction.py:coefficients',),
        None,
    ),
    (
        'assembly done',
        'Attach density, aerosol and layer data to the solver.',
        ('jobs/run_state.py:create_configured',),
        1,
    ),
    (
        'cells',
        'Register and build the endpoint cache as run() does, then build the rays of every cell for the solve.',
        (
            'jobs/run_state.py:register_geometry',
            'solve/ray_geometry_cpu.py:build_cells',
            'solve/ray_geometry.py:TransientGeometry.cell',
        ),
        1,
    ),
    ('segments', 'Cut each ray at the spherical shell boundaries.', ('rays/shell_segments.py:boundaries',), None),
    (
        'columns',
        'Integrate the component columns along each ray and pack the rays.',
        ('rays/ray_columns.py:integrate', 'solve/ray_geometry.py:pack'),
        None,
    ),
    (
        'solve start',
        'Start the fixed-point iteration from a zero source field.',
        ('solve/source_solver.py:Solver.solve',),
        2,
    ),
    (
        'first scattering',
        'Single scattering of direct sunlight (computed once, by the first apply).',
        ('solve/source_solver.py:Solver.initialize',),
        None,
    ),
    (
        'apply',
        'One update of the source: ground reflection, then atmospheric scattering.',
        ('solve/source_solver.py:Solver.apply',),
        None,
    ),
    ('transport', 'NumPy transport along the rays of one cell.', ('backends/numpy_reference.py:transport',), None),
    (
        'convergence',
        'Mix the next source (Anderson); stop when the observer and source changes are within the tolerance.',
        ('solve/source_solver.py:Solver.boxes', 'solve/source_solver.py:Solver.solve'),
        2,
    ),
    (
        'radiance table',
        'Turn the converged source into radiance for every view direction.',
        ('outputs/radiance_table.py:make_table',),
        3,
    ),
    ('summary', 'Write this page from the observed record.', ('jobs/walkthrough.py:render',), None),
)
# Production paths the example does not run: (what, pointers, why).
NOT_RUN = (
    (
        'full grids and the grid rules',
        ('jobs/run_state.py:refined_heights', 'jobs/run_state.py:refined_chi'),
        'the reduced grid is fixed; the same grid without --reduced is refused with E_SETTINGS_GRID',
    ),
    ('CUDA transport', ('backends/kernel.cu', 'solve/source_solver.py:Kernel.launch'), 'the NumPy backend is used'),
    (
        'endpoint residency on the device',
        ('solve/ray_geometry.py:TwoLayerGeometry.prepare',),
        'the endpoint cache is built but the reduced solve reads its rays from TransientGeometry; nothing is uploaded',
    ),
    (
        'parallel preparation with more than one worker',
        (
            'precompute/sun_parallel.py',
            'precompute/column_parallel.py',
            'precompute/ground_parallel.py',
            'solve/ray_geometry_parallel.py',
        ),
        'every preparation pool is pinned to one worker and takes its serial path',
    ),
    ('KSPLIT split launches', ('solve/ksplit.py', 'solve/source_solver.py:ksplit_setting'), 'CUDA only'),
    ('stratospheric aerosol layer', ('atmosphere/stratosphere.py:attach',), 'disabled in the example state'),
    (
        'O4 absorption',
        ('asset_tools/prepare_minor_absorbers.py',),
        'O4 is off, so the run needs no prepared O4 asset (the empty work root has none)',
    ),
    (
        'tropospheric Mie phase and two-layer aerosol',
        ('atmosphere/tropo_mie.py:attach', 'atmosphere/tropo_profile.py:engine_hooks'),
        'the example state has neither; aerosol scattering is Henyey-Greenstein with an exponential profile',
    ),
    ('spectral bands and wavelength chunks', ('atmosphere/spectral.py:block',), 'one wavelength, 550 nm'),
    (
        'signature, archive and verification',
        ('jobs/run_state.py:signed_contract', 'outputs/pack_state.py:pack_state', 'jobs/run_state.py:verify_pack'),
        'the table is saved as table.npz without a production signature, archive or verification',
    ),
    ('queue and persistent service', ('jobs/pipeline.py', 'jobs/pipeline_execution.py'), 'one state in one process'),
)


def example_state():
    """clear_34 of assets/states_v2.json with O4 off and the stratospheric layer disabled."""
    from sosfuse2.jobs.run_state import read_states

    state = next(s for s in read_states(paths.ASSETS / 'states_v2.json', True) if s['id'] == 'clear_34')
    state = copy.deepcopy(state)
    state.update(id=STATE_ID, description='clear_34 parameters, O4 off, stratosphere disabled')
    from sosfuse2.atmosphere.density import LEGACY

    state['density'] = dict(LEGACY)
    state['parameters']['o4'] = False
    state['stratosphere'] = dict(status='disabled')
    return state


def example_environment(root, base=None):
    """The process environment of the example: `base` (os.environ) without SOSFUSE2_* and STRATO_ATTACH_CACHE, plus
    EXAMPLE_ENVIRONMENT and SOSFUSE2_WORK_ROOT=root, so every cache the code finds through paths.work_root() is new."""
    env = {
        k: v
        for k, v in (os.environ if base is None else base).items()
        if not k.startswith('SOSFUSE2_') and k != 'STRATO_ATTACH_CACHE'
    }
    env.update(EXAMPLE_ENVIRONMENT, SOSFUSE2_WORK_ROOT=str(root))
    return env


def run_example(root):
    """Run the example in the empty folder `root`; return the observed record (JSON values) and save root/table.npz."""
    root = Path(root).resolve()
    env = example_environment(root)
    for k in set(os.environ) - set(env):
        del os.environ[k]
    os.environ.update(env)
    # run_state.main(): the geometry cache under the work root, column-cache job scope, attach cache default.
    geometry_cache = root / 'geometry_cache'
    os.environ.update(
        SOSFUSE2_COLUMN_CACHE_JOB=str(root),
        SOSFUSE2_STATE_ID=STATE_ID,
        STRATO_ATTACH_CACHE=str(geometry_cache / 'attach_cache'),
    )
    import fcntl
    from types import SimpleNamespace
    from sosfuse2.atmosphere.components import canonical_state, production_only
    from sosfuse2.jobs import run_state as rs

    production_only()
    import numpy as np
    from sosfuse2.solve import ray_geometry as gm
    from sosfuse2.solve.source_solver import Solver
    from sosfuse2.solve.ray_geometry_cpu import build_cells, cache_complete, ensure_column
    from sosfuse2.outputs.radiance_table import make_table

    class ScalarKernel(gm.Kernel):  # run(): the NumPy backend's host() is a plain array view
        def host(self, x):
            return np.asarray(x)

    gm.Kernel = ScalarKernel
    a = SimpleNamespace(reduced=True, settings_json=None, stratosphere=None)  # run_state --reduced, no other options
    state = rs.effective_state(example_state(), a)
    cfg = rs.resolve_settings(rs.settings_for(a), state)
    amounts = canonical_state(state)['parameters']
    wl, hs = np.asarray(cfg['wavelengths_nm']), np.asarray(cfg['solar_elevation_deg'])
    nr, nc = cfg['view_shape']
    el = 90 - (np.arange(nr) + 0.5) * 90 / nr
    az = (np.arange(nc) + 0.5) * 180 / nc
    seconds = {}
    t = time.monotonic()
    s, scope = rs.create_configured(
        root / 'tmp' / STATE_ID / 'prepare_0', cfg, wl, 'numpy', state, 128, geometry_cache=geometry_cache
    )
    seconds['create_configured'] = time.monotonic() - t
    off = dict(
        cuda=s.kernel.xp is not np,
        stratosphere=getattr(s, 'strato', None) is not None,
        tropo_mie=getattr(s, 'tropo', None) is not None,
        two_layer_aerosol=getattr(s, 'aerosol_profile', None) is not None,
    )
    if any(off.values()):
        raise RuntimeError('E_WALKTHROUGH_PATH: the example reached a path its page lists as not run: %r' % off)
    t = time.monotonic()
    leased = rs.register_geometry(s, scope, geometry_cache, 2)  # run_state --geometry-layers default
    success = False
    try:
        with (geometry_cache / (leased.stats['signature'] + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            cache_complete(leased)
            build_cells(leased)
        ensure_column(leased)
        geometry = gm.TransientGeometry(s, scope)  # reduced: the solve reads transient cells
        for i in range(s.geo.ns):
            geometry.cell(i)
        seconds['cells'] = time.monotonic() - t
        t = time.monotonic()
        field, stats = Solver(s, geometry, fixed_boxes=False).solve(64, rs.ANDERSON_TOLERANCE)  # --max-eval 64
        s.kernel.sync()
        host = s.kernel.host(field)
        seconds['Solver.solve'] = time.monotonic() - t
        s.ms = np.zeros((*host.shape[:2], 2 * s.W), 'f4')
        s.ms[..., ::2] = host
        t = time.monotonic()
        table, _ = make_table(s, host, hs, el, az, 32, 25)  # --group 32 --batch 25
        table = table[..., : len(wl)]
        seconds['make_table'] = time.monotonic() - t
        if not np.isfinite(table).all():
            raise RuntimeError('E_TABLE_FINITE')
        success = True
    finally:
        rs.release_geometry(geometry_cache, leased.cache, success, True)
    np.savez(
        root / 'table.npz',
        radiance=table,
        wavelength_nm=wl.astype('f4'),
        solar_elevation_deg=hs.astype('f4'),
        view_elevation_deg=el.astype('f4'),
        relative_azimuth_deg=az.astype('f4'),
    )
    last = stats['history'][-1]
    return dict(
        state=state,
        amounts=dict(no2_du=amounts['no2_du'], o4=amounts['o4']),
        settings=cfg,
        assembled=dict(
            wavelengths_nm=[float(w) for w in s.wl],
            heights_km=[float(x) for x in s.geo.h],
            solar_zenith_deg=[round(float(x), 6) for x in np.rad2deg(s.geo.chi)],
            cells=int(s.geo.ns),
            quadrature=dict(s.geo.cfg),
            L=int(s.kernel.L),
            K=int(s.kernel.K),
        ),
        geometry={k: int(geometry.stats[k]) for k in ('cells', 'rays', 'samples')},
        solved=dict(
            source_shape=list(host.shape),
            evaluations=int(stats['evaluations']),
            tolerance=stats['tolerance'],
            converged=stats['converged'],
            observer_change=max(last['observer_relative']),
            source_change=max(last['source_relative']),
        ),
        table=dict(
            shape=list(table.shape),
            finite=True,
            minimum=float(table.min()),
            maximum=float(table.max()),
            solar_elevation_deg=hs.tolist(),
            view_elevation_deg=el.tolist(),
            relative_azimuth_deg=az.tolist(),
            sunward=table[:, :, 0, 0].tolist(),
        ),
        seconds=seconds,
    )


def _pointer(ref):
    return '`%s`' % ref.replace(':', ': ')


def render(record):
    """docs/WALKTHROUGH.md from a run record: no paths, process ids, times, host or thread values."""
    g = '%.4g'
    a, sv, t = record['assembled'], record['solved'], record['table']
    shown = {'input': 'input', 1: '1', 2: '2', 3: '3', None: ''}
    out = [
        '# Walkthrough: one state from input to radiance table',
        '',
        'Generated by `python -m sosfuse2 walkthrough --write`; do not edit by hand. The sections marked *observed* '
        'come from running the calculation. The *static map* is read from the code and is not traced at run time.',
        '',
        '## Input (observed)',
        '',
        'State `%s`: the parameters of `clear_34` in `assets/states_v2.json`, with O4 off and the stratospheric layer '
        'disabled. Resolved amounts (`atmosphere/components.py: canonical_state`): NO2 %s DU, O4 %s.'
        % (STATE_ID, g % record['amounts']['no2_du'], 'on' if record['amounts']['o4'] else 'off'),
        'The example uses exp8/triangle/keep to run without downloads; the default is usstd76/usstd76/usstd.',
        '',
        '```json',
        json.dumps(record['state'], indent=1),
        '```',
        '',
        'Settings from `jobs/run_state.py: settings_for` with `--reduced`, then `resolve_settings`:',
        '',
        '```json',
        json.dumps(record['settings'], indent=1),
        '```',
        '',
        'The command runs in a new empty work root with `SOSFUSE2_OFFLINE=1`, one worker for each preparation pool '
        '(`%s`) and the BLAS/OpenMP thread variables pinned. Other `SOSFUSE2_*` variables are removed.'
        % '`, `'.join(k for k in EXAMPLE_ENVIRONMENT if k.endswith('_WORKERS')),
        '',
        '## Three boundaries (observed)',
        '',
        'The driver (`jobs/walkthrough.py: run_example`) repeats the glue of `jobs/run_state.py: run` for the reduced '
        'NumPy path and calls the same functions.',
        '',
        '| # | call | what came out |',
        '|---|---|---|',
        '| 1 | `jobs/run_state.py: create_configured`, then the cell fill | NumPy backend; %d wavelength (%s nm); '
        'heights %s km; %d solar zenith nodes (%s deg); %d cells; quadrature nmu %d, nphi %d, path_n %d; SH order L %d '
        '(%d scalar coefficients); %d rays in the cells |'
        % (
            len(a['wavelengths_nm']),
            ', '.join(g % w for w in a['wavelengths_nm']),
            ', '.join(g % h for h in a['heights_km']),
            len(a['solar_zenith_deg']),
            ', '.join(g % c for c in a['solar_zenith_deg']),
            a['cells'],
            a['quadrature']['nmu'],
            a['quadrature']['nphi'],
            a['quadrature']['path_n'],
            a['L'],
            a['K'],
            record['geometry']['rays'],
        ),
        '| 2 | `solve/source_solver.py: Solver.solve` | source field shape %s (cell, source slot, wavelength); '
        '%d evaluations; tolerance %s; converged %s; last observer change %s, last source change %s |'
        % (
            tuple(sv['source_shape']),
            sv['evaluations'],
            g % sv['tolerance'],
            'yes' if all(sv['converged']) else 'no',
            '%.2e' % sv['observer_change'],
            '%.2e' % sv['source_change'],
        ),
        '| 3 | `outputs/radiance_table.py: make_table` | table shape %s (solar elevation, view elevation, relative '
        'azimuth, wavelength); all values finite; minimum %s, maximum %s W m^-2 sr^-1 nm^-1 |'
        % (tuple(t['shape']), g % t['minimum'], g % t['maximum']),
        '',
        'Radiance in the azimuth column nearest the sun (relative azimuth %s deg), W m^-2 sr^-1 nm^-1. The evaluation '
        'count and these values are information only; the tests compare the table with `jobs/run_state.py` at rtol '
        '1e-3.' % (g % t['relative_azimuth_deg'][0]),
        '',
        '| solar elevation | ' + ' | '.join('view %s deg' % (g % e) for e in t['view_elevation_deg']) + ' |',
        '|---|' + '---|' * len(t['view_elevation_deg']),
        *(
            '| %s deg | ' % (g % h) + ' | '.join('%.4e' % v for v in row) + ' |'
            for h, row in zip(t['solar_elevation_deg'], t['sunward'])
        ),
        '',
        '## Static map (read from the code, not traced)',
        '',
        'Inside the boundaries, in the order the code reaches them. The last column names the boundary above where the '
        'outcome is shown.',
        '',
        '| # | step | what happens | where | shown |',
        '|---|---|---|---|---|',
        *(
            '| %d | %s | %s | %s | %s |' % (i, step, what, '<br>'.join(_pointer(r) for r in refs), shown[seen])
            for i, (step, what, refs, seen) in enumerate(STEPS, 1)
        ),
        '',
        '## Production paths this example does not run',
        '',
        'Each pointer names the function or module where the path starts. Several are reached here too and return at '
        'once because the path is off (stratosphere, troposphere, spectral block, KSPLIT setting, one-worker pools).',
        '',
        *('- %s (%s): %s.' % (what, ', '.join(_pointer(r) for r in refs), why) for what, refs, why in NOT_RUN),
        '',
        '## Regenerate',
        '',
        '```',
        'python -m sosfuse2 walkthrough --write',
        '```',
    ]
    return '\n'.join(out) + '\n'


def write_document(path, text):
    """Replace `path` with `text` through a temporary file in the same folder; on failure the old file stays."""
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument(
        '--write',
        nargs='?',
        const=Path('docs/WALKTHROUGH.md'),
        type=Path,
        metavar='PATH',
        help='also write the page (default docs/WALKTHROUGH.md under the current directory)',
    )
    p.add_argument(
        '--work-root',
        type=Path,
        help='empty folder to run in and keep, with table.npz (default: a temporary folder, removed after)',
    )
    a = p.parse_args(argv)
    if a.write is not None:
        target = a.write.resolve()
        if not target.parent.is_dir() or target.is_relative_to(paths.PKG):
            p.error('E_WALKTHROUGH_TARGET: %s needs an existing folder outside the installed package' % a.write)
    if a.work_root is not None and a.work_root.exists() and any(a.work_root.iterdir()):
        p.error('E_WALKTHROUGH_WORK_ROOT: --work-root must be empty or absent')
    root = Path(tempfile.mkdtemp(prefix='sosfuse2-walkthrough-')) if a.work_root is None else a.work_root
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        record = run_example(root)
    finally:
        if a.work_root is None:
            shutil.rmtree(root, ignore_errors=True)
    text = render(record)
    print(text, end='', flush=True)
    if a.write is not None:
        write_document(a.write, text)
        print('wrote', a.write)
    parts = ', '.join('%s %.1f' % kv for kv in record['seconds'].items())
    print('walkthrough seconds %.1f (%s)' % (time.monotonic() - started, parts), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
