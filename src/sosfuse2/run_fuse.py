#!/usr/bin/env python3
"""D1 fused SOS driver. Writes only under its work folder (--work, default <work root>/run, which must lie inside the work root)."""
import argparse
import gc
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode=True
from sosfuse2 import paths
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ.setdefault(key,os.environ.get('OMP_NUM_THREADS',str(os.cpu_count() or 1)))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=('numpy','cupy'),default='numpy')
    action=p.add_mutually_exclusive_group(required=True)
    action.add_argument('--selftest',action='store_true')
    action.add_argument('--validate-mac',action='store_true')
    action.add_argument('--benchmark',action='store_true')
    action.add_argument('--iterate',action='store_true',help='one complete source apply, including Lambert boundary')
    action.add_argument('--solve',action='store_true')
    action.add_argument('--states',type=int,choices=(40,))
    p.add_argument('--grid',choices=('selftest','reduced','full'),default='full')
    p.add_argument('--wavelengths',choices=('550','15'),default='15')
    p.add_argument('--block-size',type=int,choices=(32,64,128,256),default=128)
    p.add_argument('--block-sizes',default='32,64,128,256',help='benchmark sweep only')
    p.add_argument('--geometry-mode',choices=('endpoints','endpoints-2layer'),help='explicit selection of the ~0.87 GB endpoint-resident option; endpoints-2layer = common layer without scale height + per-height aerosol column')
    p.add_argument('--ray-counts',default='10000,100000,full')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--max-eval',type=int,default=16)
    p.add_argument('--tolerance',type=float,default=.005)
    p.add_argument('--rss-gb',type=float,default=6.)
    p.add_argument('--fixed-boxes',action='store_true',help='A5: observer boxes via one fixed Bundle on the device source')
    from sosfuse2.density import add_arguments
    add_arguments(p)
    p.add_argument('--work',type=Path,default=None,help='run folder (default: <work root>/run, SOSFUSE2_WORK_ROOT)')
    return p


def peak():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)


def monitor(stop,limit):
    last=time.monotonic()
    while not stop.wait(.5):
        rss=peak()
        if rss>=limit*.95:
            print(f'E_RSS_LIMIT peak={rss} cap={limit}',flush=True);os._exit(75)
        if time.monotonic()-last>=60:
            print(f'SOSFUSE HEARTBEAT RSS_peak={rss/1e9:.3f}GB',flush=True);last=time.monotonic()


def save(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def run_state(args,output,wavelengths,state=None,cache=None):
    if args.benchmark:
        os.environ.setdefault('SOSFUSE2_KSPLIT_FALLBACK', '0')
    import numpy as np
    from sosfuse2.geometry import create,TransientGeometry,ResidentGeometry,TwoLayerGeometry
    from sosfuse2.fused import Solver
    from sosfuse2.timers import TIMERS
    TIMERS.reset()
    from sosfuse2.density import with_options
    from sosfuse2.geometry import normal_state
    state=with_options(normal_state() if state is None else state,args)
    started=time.perf_counter();s,scope,settings=create(output/'prepare',args.grid,wavelengths,args.backend,args.block_size,state)
    xp=s.kernel.xp
    if xp is not np:
        free,total=xp.cuda.runtime.memGetInfo()
        if free<4*1024**3:raise RuntimeError('E_VRAM_FREE: need at least 4 GiB free')
        xp.get_default_memory_pool().set_limit(size=min(20*1024**3,int(free)-1024**3))
    resident=TwoLayerGeometry if args.geometry_mode=='endpoints-2layer' else ResidentGeometry
    geometry=resident(s,scope,cache or output/'geometry') if args.grid=='full' else TransientGeometry(s,scope)
    solver=Solver(s,geometry,fixed_boxes=args.fixed_boxes);field=xp.zeros((s.geo.ns,1132 if getattr(s,'strato',None) is not None else 571,s.W),'f4')
    if args.solve or args.states:
        field,stats=solver.solve(args.max_eval,args.tolerance)
    elif args.iterate:
        field=solver.apply(field);stats=dict(evaluations=1)
    else:
        # The first complete apply is an untimed, nonzero seed. It includes
        # all direct/ground illumination and geometry upload; later samples
        # time source transport at fixed input, without changing convergence.
        field=solver.apply(field);records=[]
        if args.grid=='full':
            _,flux=geometry.evaluate(s,field,phase=1,ground_only=True)
            boundary=flux+solver.first_flux[:len(s.geo.chi)]
            blocks=[int(v) for v in args.block_sizes.split(',')]
            if any(v not in (32,64,128,256) for v in blocks):raise ValueError('E_BLOCK_SIZE: sweep uses 32,64,128,256')
            ray_counts=[geometry.bundle.rays if v=='full' else int(v) for v in args.ray_counts.split(',')]
            for block in blocks:
                s.kernel.block_size=block
                for ray_count in ray_counts:
                    bundle=geometry.prefix(ray_count);kernel_seconds=[];wall_seconds=[]
                    # Prime cache and compilation separately from measurements.
                    s.kernel.launch(bundle,s,field,phase=0,boundary=boundary);s.kernel.sync()
                    for _ in range(args.repeats):
                        s.kernel.sync();tt=time.perf_counter()
                        s.kernel.launch(bundle,s,field,phase=0,boundary=boundary,timed=True);s.kernel.sync()
                        wall_seconds.append(time.perf_counter()-tt);kernel_seconds.append(s.kernel.cuda_events[-1])
                    record=dict(rays=bundle.rays,cells=bundle.cells,block_size=block,CUDA_seconds=kernel_seconds,
                                wall_seconds=wall_seconds,attributes=dict(s.kernel.attributes[str(s.W)]),geometry_bytes=bundle.bytes,
                                scope='fixed-source fused transport+SH projection; exact ray prefix, ground boundary prepared outside timer')
                    records.append(record);save(output/'benchmark_progress.json',records)
                    print(f'SOSFUSE BENCH rays={bundle.rays} block={block} CUDA_s={min(kernel_seconds):.6f}',flush=True)
            s.kernel.block_size=args.block_size
        complete=[]
        for _ in range(args.repeats):
            s.kernel.sync();tt=time.perf_counter();solver.apply(field);s.kernel.sync();complete.append(time.perf_counter()-tt)
        stats=dict(prefix_benchmarks=records,complete_apply_seconds=complete,
                   scope='complete operator: ground transport+full transport+SH projection+source scaling; warm first light/geometry')
    np.savez_compressed(output/'source.npz',source=s.kernel.host(field),h=s.geo.h,chi=s.geo.chi,wavelength_nm=s.wl)
    from sosfuse2.stratosphere import computed_metadata
    report=dict(backend=args.backend,settings=settings,state=s.fuse_state['id'],solve=stats,timings=solver.timings,geometry=geometry.stats,
                kernel_attributes=s.kernel.attributes,kernel_work=s.kernel.metrics,peak_rss_bytes=peak(),seconds=time.perf_counter()-started,
                physics='polarized Rayleigh + fully depolarizing tropospheric HG + scalar Mie stratosphere + ozone + Lambert',
                **computed_metadata(s.fuse_state,getattr(s,'strato',None)))
    from sosfuse2.density import metadata
    report['density']=metadata(s)
    if xp is not np:
        report['VRAM']=dict(total_bytes=int(total),used_pool_bytes=int(xp.get_default_memory_pool().used_bytes()),
                            reserved_pool_bytes=int(xp.get_default_memory_pool().total_bytes()),sampled_peaks=s.kernel.pool_peaks)
        props=xp.cuda.runtime.getDeviceProperties(xp.cuda.runtime.getDevice())
        name=props['name'];name=name.decode() if isinstance(name,bytes) else str(name)
        report['device']=dict(name=name,compute_capability=[int(props['major']),int(props['minor'])],
                              CUDA_runtime=int(xp.cuda.runtime.runtimeGetVersion()),CuPy=xp.__version__)
    timing_path=output/f'timings-{time.time_ns()}.json'
    report['timing_file']=timing_path.name
    save(output/'benchmark.json',report)
    TIMERS.write(timing_path,dict(state=s.fuse_state['id'],scope='A1 stage timers; setup_seconds = first_light_total (includes geometry prepare)'))
    return report


def main():
    args=parser().parse_args();output=(args.work or paths.work_root()/'run').resolve()
    # sosfuse2 public split: run folders live under the work root (SOSFUSE2_WORK_ROOT or the XDG cache), never inside the package
    roots=[paths.work_root()]
    if not any(output==r or r in output.parents for r in roots):raise ValueError(f'E_OUTPUT_ROOT: work must be inside the work root {roots[0]}')
    if not 0<args.rss_gb<=6:raise ValueError('E_RSS_BUDGET: require 0<RSS<=6 GB')
    if args.repeats<1 or args.max_eval<2:raise ValueError('E_ITERATIONS: repeats>=1 and max-eval>=2')
    if args.tolerance!=.005:raise ValueError('E_CONVERGENCE_CONTRACT: fixed source/observer tolerance is 0.5%')
    output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('CUPY_CACHE_DIR',str(output/'nvrtc_cache'))
    os.environ['CUDA_CACHE_PATH']=str(output/'cuda_driver_cache')
    stop=threading.Event();threading.Thread(target=monitor,args=(stop,args.rss_gb*1e9),daemon=True).start();tt=time.perf_counter()
    try:
        if args.selftest or args.validate_mac:
            from sosfuse2.density import DEFAULTS
            if any(getattr(args,k,None) not in (None,v) for k,v in DEFAULTS.items()):raise ValueError('E_PROFILE_VALIDATION: validate-mac is the historical default gate; run enabled density profiles through run_gpu_state_v3 or gate.py')
            from sosfuse2.validate import validate
            if args.validate_mac and args.backend!='numpy':raise ValueError('E_MAC_BACKEND: Mac final checkpoint uses numpy')
            report=validate(output,args.backend,full=args.validate_mac,block_size=args.block_size)
            report.update(peak_rss_bytes=peak(),seconds=time.perf_counter()-tt)
            save(output/'validation.json',report);print(json.dumps(report,indent=2),flush=True)
            if not report['pass']:raise RuntimeError('E_VALIDATION_FAILED: inspect validation.json')
            return
        if sys.platform=='darwin' and args.grid=='full':raise ValueError('E_MAC_FULL: full benchmark/solve belongs on CUDA rig')
        if args.backend!='cupy' and args.grid=='full':raise ValueError('E_FULL_BACKEND: full work requires cupy')
        import numpy as np
        wavelengths=[550.] if args.wavelengths=='550' else (360+(np.arange(15)+.5)*470/15).tolist()
        if args.grid=='full' and args.geometry_mode is None:
            raise ValueError('E_GEOMETRY_SELECTION: choose --geometry-mode endpoints only after reviewing the ~0.87 GB geometry budget')
        if args.states:
            if args.grid!='full' or args.wavelengths!='15':raise ValueError('E_STATES_CONTRACT: 40 states require full grid and 15 wavelengths')
            states=json.loads((paths.ASSETS/'states_v1.json').read_text())['states'];reports=[]
            if len(states)!=40:raise RuntimeError('E_STATE_COUNT: expected exactly 40 accepted states')
            for state in states:
                target=output/state['id'];target.mkdir(exist_ok=True)
                report=run_state(args,target,wavelengths,state,output/'geometry_cache');reports.append(report)
                save(output/'states_progress.json',reports);gc.collect()
                import cupy
                cupy.get_default_memory_pool().free_all_blocks()
            report=dict(states=reports,seconds=time.perf_counter()-tt,peak_rss_bytes=peak())
            save(output/'states.json',report)
        else:report=run_state(args,output,wavelengths)
        print(json.dumps(report,indent=2),flush=True)
    finally:stop.set()


if __name__=='__main__':main()
