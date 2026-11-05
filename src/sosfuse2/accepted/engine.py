"""Read-only reuse of accepted solver definitions; all mutable state is local.
AST loading intentionally excludes legacy runners, imports, monitors and writes.

Public split (docs/SPEC_sosfuse2_public.md §2-2 A/B, §3-3): the accepted sources are byte-frozen copies in this
package directory (sosfuse2/accepted/), the native kernel source lives in sosfuse2/native/, build products go to the
work root, and every text patch states how many times it must match (E_ADAPTER_DRIFT otherwise).
"""
import ast, collections, ctypes, hashlib, json, math, os, resource, shutil, subprocess, sys, time, types
from pathlib import Path
import numpy as np
from .. import paths
ROOT=paths.PKG            # package dir: accepted sources are 'accepted/<name>', data 'assets/<name>'
HERE=Path(__file__).resolve().parent
R,Z,OBS=6371.,80.,.1
THREAD_KEYS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')
SOURCE_FILES=['accepted/sos_probe.py','accepted/sos_sky.py','accepted/sos_sky2.py','accepted/sos_full.py','accepted/sos_render.py',
              'accepted/b2_driver.py','accepted/a2_driver.py','assets/spectra.json']
SPECTRA=ROOT/'assets/spectra.json'

def patched(text, relative, replace):
    """Apply (old, new[, count]) text patches; count defaults to 1 and must match exactly (SPEC §3-3, oracle F03)."""
    for item in replace:
        old,new=item[0],item[1]; want=item[2] if len(item)>2 else 1
        n=text.count(old)
        if n!=want: raise RuntimeError(f'E_ADAPTER_DRIFT: {relative}: {old!r} matched {n} time(s), expected {want}')
        text=text.replace(old,new)
    return text

def definitions(relative, names, scope, replace=()):
    path=ROOT/relative; text=patched(path.read_text(),relative,replace)
    tree=ast.parse(text)
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes}!=set(names): raise RuntimeError(f'E_ADAPTER_DRIFT: {relative}: missing definition')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return types.SimpleNamespace(**scope)

def peak():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)

def fingerprint():
    """Package-relative keys ('sosfuse2/accepted/sos_sky.py') of everything the accepted CPU engine reads or compiles."""
    files=[ROOT/p for p in SOURCE_FILES]+[HERE/'engine.py',ROOT/'host/pack_state.py',ROOT/'native/sh_cpu.cpp',ROOT/'assets/rgb_weights.npy']
    return {paths.rel(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def compile_kernel(threads=2, force=False):
    source=paths.NATIVE/'sh_cpu.cpp'; digest=hashlib.sha256(source.read_bytes()+str(threads).encode()).hexdigest()
    # build products are per source digest and per host (-march=native), never inside the package
    target=paths.work_root()/'build'/f'kernel_t{threads}_{digest[:12]}.so';target.parent.mkdir(parents=True,exist_ok=True)
    stamp=target.with_suffix('.json')
    if not force and target.exists() and stamp.exists() and json.loads(stamp.read_text()).get('source_hash')==digest:return target
    compiler=shutil.which('g++') or shutil.which('clang++')
    if not compiler:raise RuntimeError('E_COMPILER_MISSING: integrated full_transport has no NumPy fallback; install g++ or clang++ on rig')
    tmp=target.with_suffix(f'.{os.getpid()}.tmp')
    args=[compiler,'-std=c++17','-O3','-march=native','-fPIC','-shared','-pthread',f'-DRIG_THREADS={threads}',str(source),'-o',str(tmp)]
    try:subprocess.run(args,check=True)
    except subprocess.CalledProcessError as e:raise RuntimeError('E_COMPILE_FAILED: no integrated NumPy fallback exists; compilation is required') from e
    os.replace(tmp,target);stamp.write_text(json.dumps(dict(source_hash=digest,command=args))+'\n');return target

class Kernel:
    def __init__(self, threads, height, g):
        self.lib=ctypes.CDLL(str(compile_kernel(threads)))
        self.lib.basis.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
        self.lib.full_transport.argtypes=[ctypes.c_int]*7+[ctypes.c_void_p]*20+[ctypes.c_double]*3+[ctypes.c_void_p]
        self.height,self.g=height,g
    def basis(self, dirs, L):
        dirs=np.ascontiguousarray(dirs,'f8');out=np.empty((len(dirs),(L+1)*(L+2)//2))
        self.lib.basis(len(dirs),L,dirs.ctypes.data,out.ctypes.data);return out
    def transport(self,z,s,field,first=False):
        out=np.empty((len(z['nu']),s.W,2,4));arrays=[s.active,z['off'],z['base'],z['lit'],z['point'],z['nu'],z['target'],z['dloc'],field,
            s.ext,s.asca,s.geo.chi,s.hr,s.ha,s.geo.poly,s.geo.left,s.goff,s.gdir,s.gval,s.gm]
        arrays=[np.ascontiguousarray(x) for x in arrays]
        self.lib.full_transport(len(out),s.W,len(s.geo.chi),s.geo.ns,s.mode,int(first),0,
            *[x.ctypes.data for x in arrays],s.delta,self.height,self.g,out.ctypes.data)
        return out

def create(state, cache, mode, threads, wavelength=550.):
    params=state['parameters'];H=params['scale_height_km'];gg=params['g']
    common=dict(np=np,math=math,time=time,json=json,Path=Path,R=R,Z=Z,ROOT=ROOT)
    probe=definitions('accepted/sos_probe.py',['gl','frame','boundaries'],dict(common))
    oldscope=dict(common,probe=probe,OBS=OBS,SPEC=json.loads(SPECTRA.read_text()))
    old=definitions('accepted/sos_sky.py',['coefficients','profiles','geometry_segments','total_columns','SunGeometry','grids'],oldscope,
                    [('np.maximum(h,0)/1.2','np.maximum(h,0)/AER_HEIGHT')])
    oldscope['AER_HEIGHT']=H
    # Functions share oldscope, so state-dependent profiles are used by every optical column.
    old.NORMAL=dict(aer=params['aod550']/(H*(-np.expm1(-Z/H))),ssa=params['ssa'],ang=params['angstrom'],g=gg,
                    albedo=params['albedo'],ozone_DU=params['ozone_du'])
    shscope=dict(common,old=old,ctypes=ctypes,NORMAL=old.NORMAL,OBS=OBS)
    sh=definitions('accepted/sos_sky2.py',['incoming','spatial_lookup'],shscope)
    legacy=definitions('accepted/sos_full.py',['ring_edges','incoming'],dict(common,sh=sh))
    b=types.SimpleNamespace(sh=sh,R=R,legacy=legacy,peak=peak,DIRS=[('sun',2,0),('sun',10,0),('sun',30,0),('anti',10,180),('anti',30,180),('side',10,90),('zenith',89,0)],HS=(2,-3,-6,-9,-12))
    polys=definitions('accepted/a2_driver.py',['chi_poly'],dict(common))
    scope=dict(common,b=b,C=3/(16*np.pi),AER_HEIGHT=H,AER_G=gg)
    definitions('accepted/b2_driver.py',['basis_at','pack','Solver'],scope,
                [('np.exp(-g.h/1.2)','np.exp(-g.h/AER_HEIGHT)',2),('self.af=.8**self.levels','self.af=AER_G**self.levels')])   # frozen b2_driver: 2 + 1 matches (rig/engine.py replaced all)
    AcceptedSolver=scope['Solver']
    tree=ast.parse((ROOT/'accepted/b2_driver.py').read_text())
    build=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Geometry')
    build=next(n for n in build.body if isinstance(n,ast.FunctionDef) and n.name=='build')
    # ast.unparse output is not a stable text format: both patches must match exactly once and the result must refer to self.cfg
    text=patched(ast.unparse(build),'accepted/b2_driver.py:Geometry.build',
                 [('12, 64, 4.0, True',"self.cfg['nmu'], self.cfg['nphi'], 4.0, True"),('d[start:start + 256], 12)',"d[start:start + 256], self.cfg['path_n'])")])
    if text.count('self.cfg[')!=3: raise RuntimeError('E_ADAPTER_DRIFT: accepted/b2_driver.py:Geometry.build: patched build does not read self.cfg three times')
    exec(compile(text,'accepted_geometry_build','exec'),scope)
    if mode=='full':h,ch=old.grids(1);nmu,nphi,path_n=12,64,12
    elif mode=='reduced':
        fullh,fullch=old.grids(1);h=fullh[[0,2,4,6,8,10,12,14,17,19]]
        degrees=np.rad2deg(fullch);ch=np.deg2rad(np.unique(np.r_[degrees[(degrees<85)|(degrees>110)],np.arange(85,111,1)]))
        nmu,nphi,path_n=2,4,6
    else:
        h=np.array([0.,.1,8.,80.]);ch=np.deg2rad([0.,60.,85.,90.,93.,96.,105.,110.,140.,180.]);nmu,nphi,path_n=1,2,4
    class Geometry:
        build=scope['build']
        def __init__(self):
            self.name=state['id'];self.h,self.chi=h,ch;self.ns=len(h)*len(ch);self.folder=cache;self.cache=cache/'blocks';self.cache.mkdir(parents=True,exist_ok=True)
            self.cfg=dict(nmu=nmu,nphi=nphi,path_n=path_n);self.compact=False
            self.poly,self.left=polys.chi_poly(ch);self.sun=old.SunGeometry(1,cache)
            self.query_n=24 if mode=='full' else 6
            self.ram=collections.OrderedDict();self.ram_bytes=0;self.ram_limit=64*1024**2
            self.meta=dict(cells=self.ns,nr=len(h),nchi=len(ch),incoming_rays=0,points=0,geometry_seconds=0.,first_seconds=0.,fold_seconds=0.,compression_seconds=0.,io_seconds=0.,largest_block_bytes=0,largest_basis_bytes=0,solid_angle_max_relative=0.,cache_bytes=0)
        def pack(self,p,d,n):return scope['pack'](p,d,self.h,self.chi,self.sun,n)
        def cell(self,i,s):
            if i in self.ram:z=self.ram.pop(i);self.ram[i]=z;return z
            path=self.cache/f'cell_{i:05d}.npz'
            if path.exists():
                with np.load(path) as f:z={k:f[k] for k in f.files}
            else:
                z=self.build(i,s)
                if shutil.disk_usage(cache).free<1_000_000_000:raise RuntimeError('E_DISK_LOW: <1 GB free; checkpoint retained')
                tmp=path.with_suffix('.tmp')
                with tmp.open('wb') as f:np.savez_compressed(f,**z)
                os.replace(tmp,path)
            size=sum(v.nbytes for v in z.values())
            while self.ram and self.ram_bytes+size>self.ram_limit:
                _,v=self.ram.popitem(last=False);self.ram_bytes-=sum(q.nbytes for q in v.values())
            if size<=self.ram_limit:self.ram[i]=z;self.ram_bytes+=size
            return z
    kernel=Kernel(threads,H,gg);geo=Geometry();solver=AcceptedSolver(geo,kernel,[wavelength],1,.03)
    solver.active[1::2]=0
    return solver,scope,dict(nr=len(h),chi_step_deg=.5 if mode=='full' else 1 if mode=='reduced' else None,nchi=len(ch),nmu=nmu,nphi=nphi,path_n=path_n,L=32,interpolation='log',observer_km=OBS)
