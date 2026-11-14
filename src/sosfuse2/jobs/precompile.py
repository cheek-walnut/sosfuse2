#!/usr/bin/env python3
"""Fill the deployment NVRTC cache with every kernel a state uses, compiled in parallel processes.

A new deployment otherwise compiles its kernels one after another inside the first state (setup
154 s with an empty cache against 12 s with a warm one, measured during development). Run it once
per deployment, e.g. while geometry is copied:

    eval "$(python -m sosfuse2.backends.flags_env C3)"
    SOSFUSE2_COMPILE_CACHE=<deployment>/caches/nvrtc python -B -m sosfuse2.jobs.precompile --jobs 8

Each worker opens its own CUDA context (CuPy loads the module after compiling). The kernels and options are the ones
source_solver.Kernel builds (raw_kernel, fl_project_kernel, strato_lut_kernel), so the cache entries are the ones a run hits.
STRATO is added for stratospheric states; density tables are a runtime pointer/mask, their node count a compile constant
(--profile-step-km 0.25/0.125 adds -DPROFILE_N=321/641; 0.5 adds nothing). --W 1 warms the monochromatic (sigma C mode)
specializations without the FW15-only KSPLIT kernels. Use --no-strato
for C3_nostrato. Ray-range first light (LAUNCH_RAY_LIMIT + fl_project) is added when the device profile splits
launches (split profile) or SOSFUSE2_FL_PROJECT=1.
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

# (first, phase, boundary) launched by a state: first light, ground, warm with boundary, observer/table queries.
SPECS = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0), (1, 2, 0), (0, 2, 0))


def one(job):
    name, flags, block_size, cache, L, W, step = job
    from sosfuse2.backends.compile_cache import configure

    configure(cache, 'cupy', block_size)
    from sosfuse2.solve.source_solver import Kernel

    k = Kernel(1, backend='cupy', block_size=block_size, flags=flags, L=L, profile_step_km=step)
    started = time.perf_counter()
    if name == 'strato_lut_build':
        k.strato_lut_kernel(W)
    elif name == 'fl_project':
        k.raw_kernel(W, 1, 0, 0, ray_limit=True)
        k.fl_project_kernel(W, 0)
    elif name[0] == 'ksplit':
        # KSPLIT=3 (production default): the column record pass and VPROD (PROD module: sos_vprod_y, sos_vprod) per pass, the warm ray consumer and
        # projection on phase 0, per-cell SLIM on the ground phase, and the plan module (source_solver.Kernel.ksplit_* paths).
        _, phase, boundary = name
        k.raw_kernel(15, 0, phase, boundary, variant='record')
        k.vprod_kernels(15, phase, boundary)
        if phase == 0:
            k.raw_kernel(15, 0, 0, boundary, ray_limit=True, variant='slimray')
            k.slim_proj_kernel(15, boundary)
        else:
            k.raw_kernel(15, 0, phase, boundary, variant='slim')
        k.vprod_plan_module().get_function('vp_blocks')
    else:
        k.raw_kernel(W, *name)
    return str(name), time.perf_counter() - started


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', default=os.environ.get('SOSFUSE2_COMPILE_CACHE'), help='NVRTC cache root (deployment)')
    p.add_argument('--jobs', type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument('--no-strato', action='store_true')
    p.add_argument('--ray-limit', choices=('auto', 'on', 'off'), default='auto')
    p.add_argument(
        '--L', type=int, choices=(16, 24, 32), default=None, help='SH order (default: run_state.SH_L)'
    )
    p.add_argument(
        '--profile-step-km',
        type=float,
        choices=(0.5, 0.25, 0.125),
        default=0.5,
        help='PROFILE node spacing of the states (density.profile_step_km); 0.5 = default kernels',
    )
    p.add_argument(
        '--W', type=int, choices=(1, 15), default=15, help='kernel wavelength width (1: sigma monochromatic C mode)'
    )
    a = p.parse_args()
    if not a.cache:
        p.error('E_PRECOMPILE: --cache or SOSFUSE2_COMPILE_CACHE required')
    from sosfuse2.solve.source_solver import default_flags, ksplit_setting
    from sosfuse2.backends.flags import resolve_config, deterministic
    from sosfuse2.jobs.run_state import SH_L

    mode = deterministic()
    L = a.L or SH_L
    flags = resolve_config(set(default_flags()) | (set() if a.no_strato else {'STRATO'})).flags
    block_size = int(os.environ.get('SOSFUSE2_BLOCK_SIZE', '128'))
    ray = a.ray_limit == 'on'
    if a.ray_limit == 'auto':
        from sosfuse2.backends.device_profile import select, smi_device_name

        name = smi_device_name()
        ray = bool(select(name)[1]['launch_target_seconds']) or os.environ.get('SOSFUSE2_FL_PROJECT', '').strip() == '1'
    W, step = a.W, a.profile_step_km
    tail = (block_size, a.cache, L, W, step)
    jobs = [(spec, flags) + tail for spec in SPECS]
    if resolve_config(flags, W=W).strato:
        jobs.append(('strato_lut_build', flags) + tail)
    if ray and resolve_config(flags, W=W).split_first:
        jobs.append(('fl_project', flags) + tail)
    # KSPLIT kernels are FW15 only (source_solver.Kernel.setup: ksplit_supported); W=1 states run the unsplit kernel.
    if W == 15 and ksplit_setting('cupy', flags, L) == '3':
        jobs += [(('ksplit', phase, boundary), flags) + tail for phase, boundary in ((0, 0), (0, 1), (1, 0))]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, a.jobs), mp_context=mp.get_context('spawn')) as pool:
        results = list(pool.map(one, jobs))
    print(
        'PRECOMPILE '
        + json.dumps(
            dict(
                wall_seconds=time.perf_counter() - started,
                jobs=a.jobs,
                kernels=dict(results),
                deterministic=mode,
                flags=list(flags),
                block_size=block_size,
                ray_limit=ray,
                L=L,
                W=W,
                profile_step_km=step,
            )
        ),
        flush=True,
    )


if __name__ == '__main__':
    main()
