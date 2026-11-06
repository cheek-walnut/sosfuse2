"""CuPy RawKernel wrapper and NumPy reference, with no sample cache.

Only contiguous arrays are passed to NVRTC. Source layout is cell,571,wave
or cell,1132,wave for an explicitly enabled stratosphere;
the inactive scalar diagnostic lane is omitted and never substitutes for the
polarized calculation. Boundary and source outputs retain all 561 SH modes.
"""
import importlib.util
import json
import os
import math
import time
import warnings
from pathlib import Path
import numpy as np
from . import reference
from .components import NCOMP, SINV_F
from .arithmetic import F, norm_constants
from .sh_layout import count, padded, workspace, source_size, kernel_source, combined_aux
from .timers import TIMERS, stage
from .kernel_config import KERNEL_FLAGS, STAGE1_FLAGS, KSPLIT_VARIANTS, resolve_config, deterministic
from .verify_tol import compare_tol, compare_chunks, tolerances, log_summary
PROFILE_N_DEFAULT = 161   # density.PROFILE_STEPS[0.5]; kernel.cu #ifndef PROFILE_N

from . import paths  # noqa: E402  (public split: package-relative resources, work root for caches)
# R1 split (SOSFUSE2_KSPLIT=1): warm/ground launches become a column
# record pass once per bundle (SINV mode 1), then per chunk a producer (PROD: SH contraction -> own_aero) and a consumer
# (SLIM: transport from own_aero and the columns). Internal compile variants, never user kernel flags.
KSPLIT_DEFAULT = '3'                 # SOSFUSE2_KSPLIT when unset (Kernel.__init__)
SLIM_PROJ_THREADS = 128                # two wavelength groups per moment
SLIM_PROJ_MOMENTS = 64                 # includes the final flux row
KSPLIT_BLOCK = 128      # split kernels only (168 registers: 12 warps/SM at 128 vs 8 at 256, kdiag §7)
# SOSFUSE2_KSPLIT=2: PROD is replaced by kernel.cu sos_vprod (visitor-stream producer, same module and
# options as PROD). Plan per fixed chunk (aerosol cap, once per state): records (slot = (point-base)*4+corner) stable-sorted by
# source cell (one uint64 sort of cell<<32|slot), visitors = runs of one (cell, ray), blocks = <= VPROD_VISITORS visitors of one cell.
VPROD_VISITORS = 256
VPROD_ORDER = (64, 4)                # VPROD block order: (heights, chi columns) cell tiles, None = cell index order (vprod_build_plan)
VPROD_THREADS = 128


def aero_w(W):
    """Aerosol record row stride (floats): W padded to a multiple of 4 (kernel.cu AERO_W; 16 for FW 15). VPROD stores one float4 per
    thread, the consumer reads lanes < W; the pad lane is never read (measured 2.6x on a 5090)."""
    return ((W+3)//4)*4
VPROD_SENTINEL = 0xFFFFFFFF          # SOSFUSE2_KSPLIT_VERIFY: unwritten aerosol words before each producer
VPROD_VERIFY_STAGE = 16 << 20        # floats per device compare piece (64 MB)
VPROD_VERIFY_WINDOW = 512 << 20      # bytes of aerosol words per verify window (pinned host snapshot; cells are not split)
VPROD_STORE_BPP = 28                 # stored plan estimate, B/point (measured 25.4 eda, 26.1 4090; vprod_budget)
VPROD_COUNT_BLOCKS = 1024            # counting-sort blocks per plan (segment >= 4096 records; vprod_build_plan)
VPROD_COUNT_SHARED = 48 << 10        # its per-block cell counters (4 B/cell) must fit the default dynamic shared limit
VPROD_PLAN_SRC = r'''
extern "C" __global__ void vp_keys(long long nrec,const int*c0cp,int nc,unsigned long long*key){
 const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=nrec)return;
 const int c=c0cp[2*(i>>2)],k=(int)(i&3);
 const unsigned long long cell=c<0?0xFFFFFFFFull:(unsigned long long)(c+(k==0?0:k==1?1:k==2?nc:nc+1));
 key[i]=(cell<<32)|(unsigned long long)i;}
extern "C" __global__ void vp_split(long long n,const unsigned long long*key,int*slot,int*cell){
 const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 const unsigned long long x=key[i];slot[i]=(int)(x&0xFFFFFFFFull);cell[i]=(int)(x>>32);}
// Stable counting sort by source cell (same slot/cell as the key sort above, kernel clean-up study of 2026-10-29: 3.9x
// faster, 8.6 vs 24 B/record pool peak). Block b owns records [b*seg, (b+1)*seg): vp_count counts its records per cell,
// the host takes the exclusive cumsum of counts in [cell][block] order, and vp_scatter (one warp per block, 32-record
// tiles in order, __match_any_sync ranks) writes each record's slot after all lower cells, lower blocks and lower records.
// The scattered stores were the cost (real chunk: 2.57 -> 0.40 ms without them), so cell is not scattered: the sorted
// cell column is the run-length expansion of rstart[c] = offs[c*nb] (records in lower cells), which vp_rstart reads and
// vp_fill_cell writes coalesced (2.86 -> 1.17 ms with both). vp_scatter reads one point per lane for 4 tiles and keeps
// the next 4x32 points in flight; seg is a multiple of 128 records so the tiles match the 32-record order.
__device__ __forceinline__ int vp_rec_cell(long long i,const int*c0cp,int nc){
 const int c=c0cp[2*(i>>2)],k=(int)(i&3);return c<0?-1:c+(k==0?0:k==1?1:k==2?nc:nc+1);}
extern "C" __global__ void vp_count(long long n,const int*c0cp,int nc,int ncell,int seg,int nb,int*counts,int*err){
 extern __shared__ int cnt[];
 for(int c=threadIdx.x;c<ncell;c+=blockDim.x)cnt[c]=0;
 __syncthreads();
 const long long i0=(long long)blockIdx.x*seg,i1=min(n,i0+seg);
 for(long long i=i0+threadIdx.x;i<i1;i+=blockDim.x){const int c=vp_rec_cell(i,c0cp,nc);
  if(c>=ncell)atomicOr(err,1);else if(c>=0)atomicAdd(&cnt[c],1);}
 __syncthreads();
 for(int c=threadIdx.x;c<ncell;c+=blockDim.x)counts[(long long)c*nb+blockIdx.x]=cnt[c];}
#define VP_PF 4
extern "C" __global__ void vp_scatter(long long n,const int*c0cp,int nc,int ncell,int seg,int nb,const int*offs,int*slot){
 extern __shared__ int run[];
 const int lane=threadIdx.x;
 for(int c=lane;c<ncell;c+=32)run[c]=offs[(long long)c*nb+blockIdx.x];
 __syncwarp();
 const long long i0=(long long)blockIdx.x*seg,i1=min(n,i0+seg),p0=i0>>2,p1=(i1+3)>>2;
 int buf[VP_PF];
 #pragma unroll
 for(int j=0;j<VP_PF;j++)buf[j]=p0+32*j+lane<p1?c0cp[2*(p0+32*j+lane)]:-1;
 for(long long pb=p0;pb<p1;pb+=32*VP_PF){
  int cur[VP_PF];
  #pragma unroll
  for(int j=0;j<VP_PF;j++){cur[j]=buf[j];const long long q=pb+32*(VP_PF+j)+lane;buf[j]=q<p1?c0cp[2*q]:-1;}
  #pragma unroll
  for(int j=0;j<VP_PF;j++){
   if(pb+32*j>=p1)break;
   #pragma unroll
   for(int t=0;t<4;t++){
    const long long i=4*(pb+32*j)+32*t+lane;
    const int c0=__shfl_sync(0xffffffffu,cur[j],8*t+(lane>>2)),k=lane&3;
    const int c=i<i1&&c0>=0?c0+(k==0?0:k==1?1:k==2?nc:nc+1):-1;
    const unsigned peers=__match_any_sync(0xffffffffu,c);
    const int rank=__popc(peers&((1u<<lane)-1u)),pos=c>=0?run[c]+rank:-1;
    __syncwarp();
    if(c>=0 && rank==__popc(peers)-1)run[c]+=rank+1;
    __syncwarp();
    if(pos>=0)slot[pos]=(int)i;}}}}
extern "C" __global__ void vp_rstart(int ncell,int nb,long long nrec,const int*offs,long long*rstart){
 const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>ncell)return;rstart[c]=c<ncell?(long long)offs[(long long)c*nb]:nrec;}
extern "C" __global__ void vp_fill_cell(int ncell,const long long*rstart,int*cell){
 const int c=blockIdx.x;const long long a=rstart[c],b=rstart[c+1];
 for(long long i=a+threadIdx.x;i<b;i+=blockDim.x)cell[i]=c;}
extern "C" __global__ void vp_point_ray(int nray,long long r0,const unsigned int*eo,int nq,long long base,int*pray){
 const int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=nray)return;const long long r=r0+t;
 const long long a=((long long)eo[r]-r)*nq-base,b=((long long)eo[r+1]-(r+1))*nq-base;
 for(long long p=a;p<b;p++)pray[p]=t;}
extern "C" __global__ void vp_flags(long long n,const int*slot,const int*cell,const int*pray,unsigned char*flag){
 const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 flag[i]=(unsigned char)(i==0||cell[i]!=cell[i-1]||pray[slot[i]>>2]!=pray[slot[i-1]>>2]);}
extern "C" __global__ void vp_visitors(long long n,const unsigned char*flag,const int*scan,const int*slot,const int*pray,
 int*vis_rbeg,int*vis_ray){
 const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||!flag[i])return;
 const int j=scan[i]-1;vis_rbeg[j]=(int)i;vis_ray[j]=pray[slot[i]>>2];}
extern "C" __global__ void vp_cells(int ncell,long long nrec,int nvis,const long long*rstart,const int*scan,long long*cstart){
 const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>ncell)return;
 const long long r=rstart[c];cstart[c]=r<nrec?(long long)(scan[r]-1):(long long)nvis;}
extern "C" __global__ void vp_blocks(int ncell,const long long*cstart,const long long*bstart,int vb,const int*vis_rbeg,
 int*blk,long long*work){
 const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=ncell)return;
 const long long vs=cstart[c],ve=cstart[c+1];
 for(long long t=bstart[c];t<bstart[c+1];t++){const long long v0=vs+(t-bstart[c])*vb,v1=v0+vb<ve?v0+vb:ve;
  blk[3*t]=c;blk[3*t+1]=(int)v0;blk[3*t+2]=(int)v1;work[t]=(long long)(vis_rbeg[v1]-vis_rbeg[v0])+4*(v1-v0);}}
'''
# A6 kernel variants (kernel.cu header); each independent, all off = original kernel.
# r17 stage-1 compile options (strato-kernel-loop de3e13b): SOSFUSE2_<NAME>=1/0 overrides,
# unset = device profile stage1_defaults (configs/devices.json: rig FMAD+FASTMATH, eda/mac none).
# None active = the exact pre-r17 options, key and kernel tokens.
KPAD = 576
SECTION_CLOCK_NAMES = ('point', 'ground', 'sh_basis', 'transport', 'attenuation',
                       'projection', 'barrier', 'total')


def section_clock_summary(counters):
    """Lossless sums of disjoint lane-0 observer intervals, never wall seconds."""
    counters = np.asarray(counters)
    if (counters.dtype != np.dtype('uint64') or counters.ndim != 4 or
            counters.shape[-1] != len(SECTION_CLOCK_NAMES) or
            any(n <= 0 for n in counters.shape[:-1])):
        raise ValueError('E_SECTION_CLOCK_BUFFER: uint64 [cell,tile,warp,8] required')
    rows = counters.reshape(-1, len(SECTION_CLOCK_NAMES))
    if any(sum(map(int, row[:-1])) > int(row[-1]) for row in rows):
        raise ValueError('E_SECTION_CLOCK_OVERLAP: section intervals exceed observer total')
    sums = {name: sum(map(int, rows[:, i])) for i, name in enumerate(SECTION_CLOCK_NAMES)}
    total = sums['total']
    residual = total - sum(sums[name] for name in SECTION_CLOCK_NAMES[:-1])
    return dict(shape=list(counters.shape), observer_cycles=sums,
                residual_observer_cycles=residual,
                fractions={name: sums[name] / total if total else None
                           for name in SECTION_CLOCK_NAMES[:-1]},
                units='uint64 clock64 cycles per warp lane-0 observer',
                accounting='disjoint sections; ground includes internal warp sync; '
                           'barrier is observed synchronization interval, not pure waiting; '
                           'concurrent observer sums are not wall time')


def default_flags():
    import os
    names = tuple(f for f in os.environ.get('SOSFUSE2_KERNEL_FLAGS', '').replace(' ', '').split(',') if f)
    return names


def sh_aux(src, xp=np, L=32):
    """A6(c): SH source as a new contiguous [cell][wave][KPAD] array (not a view):
    sh[c, w, k] = src[c, 10+k, w] for k<561, zero padding to KPAD."""
    K=count(L); KPAD=padded(L)
    out = xp.zeros((src.shape[0], src.shape[2], KPAD), 'f4')
    out[:, :, :K] = xp.transpose(src[:, 10:10+K, :], (0, 2, 1))
    if not out.flags.c_contiguous: raise RuntimeError('E_SH_AUX_LAYOUT')
    return out


def sh_aux_pair(src, xp=np, L=32):
    """Exact FP32 [cell,ceil(W/2),KPAD,2], adjacent wavelengths per float2.

    Only the optional SH sidecar changes. src, outputs, and bundle stay intact.
    The padded final wavelength is zero and is never accumulated.
    """
    cells, _, waves = src.shape
    K=count(L); KPAD=padded(L)
    out = xp.zeros((cells, (waves+1)//2, KPAD, 2), 'f4')
    out[:, :, :K, 0] = xp.transpose(src[:, 10:10+K, 0::2], (0, 2, 1))
    if waves > 1:
        out[:, :waves//2, :K, 1] = xp.transpose(src[:, 10:10+K, 1::2], (0, 2, 1))
    if not out.flags.c_contiguous:
        raise RuntimeError('E_SH_AUX_PAIR_LAYOUT')
    return out


def load_engine():
    """A fresh, private copy of the accepted engine loader (callers patch its namespace; never the shared import).
    The module name keeps the package context so the loader's relative import of sosfuse2.paths resolves."""
    spec = importlib.util.spec_from_file_location('sosfuse2.accepted.engine_readonly', paths.ACCEPTED/'engine.py')
    engine = importlib.util.module_from_spec(spec); spec.loader.exec_module(engine)
    return engine


from .device_profile import first_light_split_error  # noqa: E402 (stdlib-only module; pipeline preflight uses it too)


def ksplit_setting(backend, flags, L):
    """Resolve KSPLIT for an FW15 physical configuration; setup defers FW1 to mode 0.
    The default needs CuPy, STRATO, L<=16 and uninstrumented transport.
    """
    config = resolve_config(flags)
    if config.instrumented:
        return '0'
    text = os.environ.get('SOSFUSE2_KSPLIT', '').strip()
    if text:
        return text
    fits = (backend == 'cupy' and config.ksplit_supported and L <= 16)
    return KSPLIT_DEFAULT if fits else '0'


class Kernel:
    def __init__(self, threads=4, height=1.2, g=.8, backend='numpy', block_size=128, flags=None, L=32, *, profile_step_km=.5):
        if backend not in ('numpy', 'cupy'):
            raise ValueError('E_BACKEND: select numpy or cupy explicitly')
        if block_size not in (32, 64, 128, 256):
            raise ValueError('E_BLOCK_SIZE: select 32,64,128,256')
        self.deterministic = deterministic()
        self.L=L; self.K=count(L)
        self.height, self.g, self.backend, self.block_size = height, g, backend, block_size
        # ②b: PROFILE node count is a compile constant (kernel.cu PROFILE_N); 0.5 km (161) adds no option and no key field.
        from .density import profile_n
        self.profile_n = profile_n(profile_step_km); self.profile_step_km = float(profile_step_km)
        self.flags = resolve_config(default_flags() if flags is None else flags).flags
        from .ground_sh import configured_order
        self.ground_sh_l = configured_order()
        # Production default since 2026-10-26:
        # unset means KSPLIT=3 (VPROD + warm ray consumer + projection) where it applies (CuPy, STRATO/FW15, L<=16, FP64
        # projection accumulator) and the unsplit kernel elsewhere; SOSFUSE2_KSPLIT=0 forces the unsplit kernel, and an
        # explicit 1-3 still fails loudly on an unsupported configuration.
        ks_text = ksplit_setting(backend, self.flags, L)
        if ks_text not in ('0', '1', '2', '3'):
            raise ValueError(f'E_KSPLIT_FLAGS: SOSFUSE2_KSPLIT must be 0, 1, 2 or 3 (got {ks_text!r})')
        self.ksplit = ks_text in ('1', '2', '3')
        self.ksplit_mode = int(ks_text) if self.ksplit else 0   # 2: VPROD instead of PROD; 3: + warm ray consumer and projection
        if self.ksplit_mode >= 2 and self.L > 16:
            raise ValueError('E_KSPLIT_FLAGS: SOSFUSE2_KSPLIT=2 (VPROD) is compiled for SH order L<=16 only (kernel.cu sos_vprod)')
        self.ksplit_verify = os.environ.get('SOSFUSE2_KSPLIT_VERIFY', '').strip() == '1'
        if self.ksplit_verify and self.ksplit_mode not in (2, 3) and os.environ.get('SOSFUSE2_KSPLIT', '').strip():
            raise ValueError('E_KSPLIT_VERIFY: SOSFUSE2_KSPLIT_VERIFY=1 compares PROD and VPROD and needs SOSFUSE2_KSPLIT=2 or 3')
        self.xp = np
        if backend=='cupy':
            import cupy
            if cupy.cuda.runtime.getDeviceCount()<1:
                raise RuntimeError('E_CUDA_DEVICE: no CUDA device')
            self.xp = cupy
        self.raw = {}; self.constants = {}; self.device_constants = {}
        self.ground_sh_tables = {}
        self.metrics = dict(samples=0, angular_pairs=0, launches=0, query_launches=0)
        self.attributes = {}; self.cuda_events = []
        self.pool_peaks=dict(used_bytes_sampled=0,reserved_bytes_sampled=0)
        TIMERS.bind(self.xp)

    def config(self, W=None):
        return resolve_config(self.flags, W=W if W is not None else getattr(self, '_W', 15))

    def memory(self):
        if self.xp is not np:
            pool=self.xp.get_default_memory_pool()
            self.pool_peaks['used_bytes_sampled']=max(self.pool_peaks['used_bytes_sampled'],int(pool.used_bytes()))
            self.pool_peaks['reserved_bytes_sampled']=max(self.pool_peaks['reserved_bytes_sampled'],int(pool.total_bytes()))

    def sync(self):
        if self.xp is not np:
            self.xp.cuda.get_current_stream().synchronize()

    def host(self, array):
        return np.asarray(array) if self.xp is np else self.xp.asnumpy(array)

    def basis(self, dirs, L):
        count(L)
        if L!=self.L:raise ValueError('E_SH_L_MISMATCH: basis and kernel degree differ')
        y = reference.basis(np.asarray(dirs)[:, 2], L)
        st = np.sqrt(np.maximum(0., 1-np.asarray(dirs)[:, 2]**2))
        cp = np.where(st>1e-12, dirs[:, 0]/np.maximum(st, 1e-30), 1.).astype('f4')
        cm = reference.cosine(cp)
        for l in range(L+1):
            for m in range(l+1):
                y[:, l*(l+1)//2+m] *= cm[:, m]
        return y.astype('f8')

    def setup(self, s, nq):
        key = (id(s), nq)
        if key not in self.constants:
          with stage('kernel_setup'):
            from .stratosphere import attach
            from .density import attach as attach_density
            density=attach_density(s,s.query.__func__.__globals__)
            if density is not None:self.check_profile(density,'setup')
            optics=attach(s,s.query.__func__.__globals__)
            if optics is not None:self.flags=tuple(sorted(set(self.flags)|{'STRATO'}))
            elif 'STRATO' in self.flags:raise ValueError('E_STRATO_FLAG_ZERO: STRATO is automatic for nonzero tau')
            from .tropo_mie import attach as attach_tropo
            tropo=attach_tropo(s)
            if self.ground_sh_l and (tropo is not None or optics is None):
                raise ValueError('E_GROUND_SH_PHASE: requires tropospheric HG and enabled stratosphere')
            if tropo is not None:self.flags=tuple(sorted(set(self.flags)|{'TROPO_MIE'}))
            elif 'TROPO_MIE' in self.flags:raise ValueError('E_TROPO_FLAG_DEFAULT: TROPO_MIE is automatic for an enabled troposphere_phase')
            from .tropo_profile import attach as attach_profile
            aer=attach_profile(s)
            if aer is not None:self.flags=tuple(sorted(set(self.flags)|{'AEROSOL_2L'}))
            elif 'AEROSOL_2L' in self.flags:raise ValueError('E_AEROSOL_2L_FLAG_DEFAULT: AEROSOL_2L is automatic for an enabled troposphere_profile')
            self._W = s.W
            cfg = self.config()
            if self.ground_sh_l and not cfg.ground_sh_supported:
                raise ValueError('E_GROUND_SH_FLAGS: requires fused strato ground, split first light and HG')
            ks = ksplit_setting(self.backend, self.flags, self.L) if s.W == 15 else os.environ.get('SOSFUSE2_KSPLIT', '').strip() or '0'
            if cfg.instrumented: ks = '0'
            if ks not in ('0', '1', '2', '3') or (ks != '0' and not cfg.ksplit_supported):
                raise ValueError('E_KSPLIT_FLAGS: needs STRATO, FW15 and an uninstrumented transport role')
            decision = getattr(s, 'ksplit_decision', None)
            self.ksplit_mode = decision['effective'] if decision is not None else int(ks)
            self.ksplit = bool(self.ksplit_mode)
            if self.ksplit_verify and self.ksplit_mode not in (2, 3):
                # Only a CuPy FW15 kernel can run KSPLIT: the persistent pipeline's CPU-prepare Kernel (NumPy backend) and FW1 kernels
                # have nothing to verify, so VERIFY=1 is inactive there with a warning (2026-11-03: before, SOSFUSE2_KSPLIT_VERIFY=1
                # could not reach the persistent service at all; the service's CuPy Kernel still enforces it).
                if self.xp is not np and s.W == 15:
                    raise ValueError('E_KSPLIT_VERIFY: requires KSPLIT=2 or 3')
                warnings.warn('W_KSPLIT_VERIFY_INACTIVE: SOSFUSE2_KSPLIT_VERIFY=1 does not apply to this kernel (NumPy backend or FW1)',
                              RuntimeWarning, stacklevel=2)
            u, wu = np.polynomial.legendre.leggauss(nq); u=(u+1)/2; wu=wu/2
            ng = int(max(np.diff(s.goff), default=0))
            gd = np.zeros((s.geo.ns, ng, 3), 'f4'); gv=np.zeros((s.geo.ns, ng, s.W), 'f4')
            for cell in range(s.geo.ns):
                a,b=s.goff[cell:cell+2];gd[cell, :b-a]=s.gdir[a:b];gv[cell, :b-a]=s.gval[a:b]
            c = dict(L=self.L, W=s.W, height=F(self.height), g=F(self.g), mix=F((1-s.delta)/(1+s.delta/2)),
                     albedo=float(s.fuse_state['parameters']['albedo']), u=u, wu=wu,
                     h=s.geo.h.astype('f4'), chi=s.geo.chi.astype('f4'), hr=s.hr.astype('f4'), ha=s.ha.astype('f4'),
                     ext=s.ext.astype('f4'), asca=s.asca.astype('f4'), sun_h=s.geo.sun.h.astype('f4'),
                     sun_col=s.geo.sun.col.astype('f4'), gm=s.gm.astype('f4'), gd=gd, gv=gv,
                     goff=s.goff, gdir=s.gdir, gval=s.gval)
            if self.config().c3_fw15:c.update(sh_combine=True,sh_combine_skip=True)
            if optics is not None:
                c.update(strato_ext=optics['ext'].astype('f4'),strato_sca=optics['sca'].astype('f4'),
                         strato_chi=optics['chi'].astype('f4'),strato_density=optics['source_density'].astype('f4'),
                         strato_sun_col=optics['sun_col'].astype('f4'),strato_profile=optics['profile'],strato_old=optics['old'])
            if density is not None:c['density_optics']=density
            if tropo is not None:c.update(tropo_lut=np.ascontiguousarray(tropo['lut'],'f4').ravel(),tropo_bins=int(tropo['bins']))
            if aer is not None:c['aer2']=np.ascontiguousarray(aer['kernel'],'f4')
            self.constants[key] = c
        c = self.constants[key]
        # Every setup() call, cache hit or miss and installed constants alike (NumPy transport, geometry,
        # table_fast and the persistent service all read prepared constants through here).
        if 'density_optics' in c: self.check_profile(c['density_optics'], 'setup')
        return c

    def tile_width(self, W, first=False, phase=0):
        return W

    def launch_cells(self):
        text=os.environ.get('SOSFUSE2_LAUNCH_CELLS','').strip()
        if not text:return 0
        if not text.isdigit():raise ValueError('E_LAUNCH_CELLS: SOSFUSE2_LAUNCH_CELLS must be a non-negative integer')
        return int(text)

    def device(self):
        """(name, profile, reason) from configs/devices.json, read once per Kernel (device_profile.py)."""
        if getattr(self, '_device', None) is None:
            from .device_profile import select, cuda_device_name
            self._device = select(cuda_device_name(self.xp) if self.xp is not np else None)
            self.launch_log = dict(profile=self._device[0], reason=self._device[2],
                                   budget_seconds=self._device[1]['launch_budget_seconds'],
                                   target_seconds=self._device[1]['launch_target_seconds'],
                                   launches=0, gpu_seconds=0.0, max_seconds=0.0, by_kind={})
        return self._device

    def launch_mode(self, ncell):
        """('single', ncell) | ('fixed', cells) | ('adaptive', None). SOSFUSE2_LAUNCH_CELLS keeps the old fixed split."""
        name, p, _ = self.device()
        fixed = p['launch_cells_fixed']
        if p.get('requires_split') and ncell > 1:
            # eda: never a whole-grid launch; SOSFUSE2_LAUNCH_CELLS is only an upper bound under the predictive cap.
            if fixed:
                return 'fixed', min(fixed, ncell)
            if p['launch_target_seconds']:
                return 'adaptive', None
            raise RuntimeError(f'E_LAUNCH_BUDGET_CONFIG: the {name} profile requires split launches; set '
                               'SOSFUSE2_LAUNCH_TARGET_SECONDS > 0 (or unset it) or SOSFUSE2_LAUNCH_CELLS. Not launched.')
        if fixed is not None:
            return ('fixed', fixed) if 0 < fixed < ncell else ('single', ncell)
        if p['launch_target_seconds']:
            return 'adaptive', None
        return 'single', ncell

    def cell_work(self, bundle):
        """Host cumulative work proxy over cells (edges of the cell's rays), cached on the Bundle."""
        cum = getattr(bundle, 'cell_work_cum', None)
        if cum is None:
            co = bundle.arrays['cell_off']; eo = bundle.arrays['edge_off']
            e = np.asarray(self.host(eo[co]), 'i8')
            cum = np.r_[0, np.cumsum(np.maximum(np.diff(e), 1))].astype('f8')
            bundle.cell_work_cum = cum
        return cum

    def ray_work(self, bundle):
        """Host cumulative work proxy over the bundle's rays (edges + 1 per ray), cached on the Bundle."""
        cum = getattr(bundle, 'ray_work_cum', None)
        if cum is None:
            co = self.host(bundle.arrays['cell_off']); r0, r1 = int(co[0]), int(co[-1])
            e = np.asarray(self.host(bundle.arrays['edge_off'][r0:r1+1]), 'i8')
            cum = (e - e[0]).astype('f8') + np.arange(r1-r0+1, dtype='f8')
            bundle.ray_work_cum = cum
        return cum

    PREDICT_MARGIN = 1.3     # predicted launch seconds are inflated by this before any comparison
    PREDICT_CAP = .8         # of launch_budget_seconds: a chunk predicted above it is halved, never launched
    PROBE_DIVISOR = 4        # first chunk of a kernel kind = initial size / 4 (nothing measured yet)

    def predict_seconds(self, kind, work):
        """Conservative launch time for [work] from this kind's recent (work, seconds): an affine fit
        t = L + r*w (L = per-launch latency floor, one cell ~0.8 s on the 3080; r = seconds per unit of work),
        times PREDICT_MARGIN. With chunks of one size only, L is taken as 0.8 of the fastest launch."""
        obs = self.__dict__.get('_launch_obs', {}).get(kind)
        if not obs:
            return None
        w = np.array([o[0] for o in obs], 'f8'); t = np.array([o[1] for o in obs], 'f8')
        fit = None
        if len(obs) >= 2 and w.max() > 1.5*w.min():
            (L, r), *_ = np.linalg.lstsq(np.vstack([np.ones_like(w), w]).T, t, rcond=None)
            if r > 0 and L >= 0:
                fit = (L, max(r, float(np.max((t-L)/w))))  # slope never below any single observation's
        if fit is None:
            L = .8*float(t.min()); fit = (L, float(np.max((t-L)/w)))
        return self.PREDICT_MARGIN*(fit[0] + fit[1]*float(work))

    def next_chunk_end(self, kind, c0, n, step, cum, initial=None):
        """Fixed: c0+step. Adaptive (cells, or rays for ray-range launches): the largest chunk whose predicted time
        (predict_seconds) fits the profile target, at most 1.5x the previous chunk; the first chunk of a kind is a
        probe (initial/PROBE_DIVISOR). Before returning, a chunk predicted above PREDICT_CAP of the budget is halved
        down to one unit; if even one unit is predicted above it, E_LAUNCH_BUDGET is raised before launching
        (the earlier size floor could exceed the 5 s budget). Blocks/rays are independent, so
        the split never changes the numbers."""
        if cum is None:   # no launch budget (rig profile): fixed step, or the whole rest when the caller has none (KSPLIT=3 projection)
            return n if step is None else min(n, c0+step)
        _, p, _ = self.device()
        initial = max(1, int(initial or p['launch_cells_initial'] or 1))
        obs = self.__dict__.setdefault('_launch_obs', {}).get(kind)
        if not obs:
            c1 = min(n, c0+max(1, initial//self.PROBE_DIVISOR), c0+step if step else n)
        else:
            last = obs[-1][2]
            hi = min(n, c0+max(int(1.5*last), last+1), c0+step if step else n)   # a fixed step is an upper bound here
            target = p['launch_target_seconds'] or self.PREDICT_CAP*(p['launch_budget_seconds'] or 0) or float('inf')
            lo, best = c0+1, c0+1
            while lo <= hi:   # largest c1 in (c0, hi] whose prediction fits the target (monotone in work)
                mid = (lo+hi)//2
                if self.predict_seconds(kind, cum[mid]-cum[c0]) <= target: best, lo = mid, mid+1
                else: hi = mid-1
            c1 = best
        budget = p['launch_budget_seconds']
        if budget and obs:
            cap = self.PREDICT_CAP*budget
            while c1 > c0+1 and self.predict_seconds(kind, cum[c1]-cum[c0]) > cap:
                c1 = c0+(c1-c0)//2
            predicted = self.predict_seconds(kind, cum[c1]-cum[c0])
            if predicted > cap:
                raise RuntimeError(f'E_LAUNCH_BUDGET: predicted {predicted:.3f}s > {cap:.3f}s for one unit of {kind} '
                                   f'(unit {c0}); not launched')
        return c1

    def timed_launch(self, kind, raw, grid, block, args, shared, units, cum=None, c0=0, c1=None):
        """One RawKernel launch timed by CUDA events; post-launch budget check (detects, never preempts)."""
        xp = self.xp; _, p, _ = self.device()
        if not p['launch_budget_seconds'] and cum is None:
            # No budget and nothing to adapt (mac profile, host test doubles): launch exactly as before.
            raw(grid, block, args, shared_mem=shared)
            self.launch_log['launches'] += 1
            return 0.0
        begin, end = xp.cuda.Event(), xp.cuda.Event()
        begin.record()
        raw(grid, block, args, shared_mem=shared)
        end.record(); end.synchronize()
        seconds = float(xp.cuda.get_elapsed_time(begin, end)/1000.)
        if not math.isfinite(seconds) or seconds < 0:
            raise RuntimeError('E_LAUNCH_EVENT: invalid CUDA event duration')
        log = self.launch_log
        k = log['by_kind'].setdefault(kind, dict(launches=0, gpu_seconds=0.0, max_seconds=0.0, max_units=0))
        k['launches'] += 1; k['gpu_seconds'] += seconds; k['max_seconds'] = max(k['max_seconds'], seconds)
        k['max_units'] = max(k['max_units'], int(units))
        log['launches'] += 1; log['gpu_seconds'] += seconds; log['max_seconds'] = max(log['max_seconds'], seconds)
        if cum is not None and c1 is not None and c1 > c0:
            work = max(float(cum[c1]-cum[c0]), 1.0)
            predicted = self.predict_seconds(kind, work)
            obs = self.__dict__.setdefault('_launch_obs', {}).setdefault(kind, [])
            obs.append((work, seconds, int(c1-c0))); del obs[:-12]
            print(f'LAUNCH_CHUNK {kind} {c0}:{c1} {seconds:.3f}s predicted='
                  f'{"none" if predicted is None else format(predicted, ".3f")}', flush=True)
        budget = p['launch_budget_seconds']
        if budget and seconds > budget:
            raise RuntimeError(f'E_LAUNCH_BUDGET: {seconds:.3f}s > {budget:.3f}s ({log["profile"]} profile); kind={kind} '
                               f'units={c0}:{c0+units}; no further launches issued; this launch already completed')
        if p['launch_pause_ms']:
            time.sleep(p['launch_pause_ms']/1000.)
        return seconds

    def bs(self, first=False, variant=None):
        if variant: return KSPLIT_BLOCK
        return self.block_size

    def shared_bytes(self, W, phase, first=False, variant=None):
        W = self.tile_width(W, first, phase)
        warps=self.bs(first, variant)//32
        # ACC_GLOBAL: the first-light phase-0 moment accumulator is the moments output (no shared rows).
        acc=(4+self.K) if phase==0 and not ((first or self.config(W).split_warm)
                                     or (self.config(W).c3_fw15 and not first)) else 0
        size=(acc+1)*W*8+warps*((workspace(self.L)+8*W+33)*4+300+40*W+4)
        size += warps*(8+16*W) if 'STRATO' in self.flags else 0
        # Match RAY_CONST_F: six two-word coordinates cached by PATH_DS kernels (phase-0 first light and warm).
        size += warps*(108 if (not first or phase == 0) else 60)
        size += warps*128
        # GROUND_FUSED (first-light phase 0): per-warp scratch 32 phases + 65 int2 LUT nodes (162 floats).
        # (with FL_RAYSPLIT the scratch lives in the warp's Y rows: no extra shared)
        size += warps*520 if (self.config(W).strato and not self.config(W).split_first and first and phase == 0) else 0
        # PT_BATCH (R4): 16 PointData slots per warp, non-first-light kernels only; RAY_SORT keys+perm.
        # KSPLIT=3 consumer (variant slimray): kernel.cu PointM2, 112 B (16-byte aligned) per batched sample instead of
        # PointData, plus 16 B slack for the aligned batch base
        pt_bytes = 112 if (variant or '').startswith('slimray') else 300 + (8 if 'STRATO' in self.flags else 0)
        size += warps * 16 * pt_bytes + (16 if (variant or '').startswith('slimray') else 0) if self.config(W).strato and (not first) else 0
        size += 512 if (self.config(W).strato and not first) else 0
        return size


    def stage1_flags(self):
        _, profile, _ = self.device()
        defaults = tuple(profile.get('stage1_defaults') or ())
        if set(defaults) - set(STAGE1_FLAGS):
            raise ValueError(f'E_DEVICE_PROFILE: unknown stage1_defaults {sorted(set(defaults)-set(STAGE1_FLAGS))}')
        out = []
        for name in STAGE1_FLAGS:
            text = os.environ.get('SOSFUSE2_'+name, '').strip()
            if text not in ('', '0', '1'):
                raise ValueError(f'E_STAGE1_FLAG: SOSFUSE2_{name} must be 0 or 1')
            if text == '1' or (not text and name in defaults):
                out.append(name)
        return tuple(out)

    def kernel_key(self, flags, W, width, first, phase, boundary, *, ground_sh_l=0, ray_limit=False, stage1=()):
        """The one RawKernel memo-key rule (raw_kernel, fl_project_kernel, vprod_kernels, slim_proj_kernel): shared raw dicts
        (persistent service, payload restore) must not mix SH orders or PROFILE node counts. N=161 adds no field."""
        key = (tuple(flags), W, width, int(first), int(phase), int(boundary), ('L', self.L))
        if self.profile_n != PROFILE_N_DEFAULT: key += (('PROFILE_N', self.profile_n),)
        if ground_sh_l: key += (('GROUND_SH_L', ground_sh_l),)
        if ray_limit: key += ('LAUNCH_RAY_LIMIT',)
        return (key, stage1) if stage1 else key

    def attribute_suffix(self):
        return '' if self.profile_n == PROFILE_N_DEFAULT else f'_profileN{self.profile_n}'

    def check_profile(self, optics, where, rows=None):
        """compiled N == declared optics N == table rows (== uploaded rows): density.check_consumed with this Kernel's
        PROFILE_N. Also on restored/persistent kernels whose setup() never ran (installed constants)."""
        from .density import check_consumed
        return check_consumed(optics, self.profile_n, rows, where)

    def check_upload(self, c, dc, where):
        """The uploaded PROFILE buffer still is the optics' table: N as check_profile, and the identity
        recorded at upload equals the current optics identity; a rebound table/mask/policy is E_PROFILE_STALE."""
        if 'density_optics' not in c: return
        from .density import profile_identity
        optics = c['density_optics']
        self.check_profile(optics, where, dc['profile_table'].shape[0])
        # Keyed by the uploaded buffer object: launch-local dict copies (KSPLIT record) share it.
        uploaded = self.__dict__.get('device_profile_identity', {}).get(id(dc['profile_table']))
        current = profile_identity(optics)['sha256']
        if uploaded != current:
            raise RuntimeError(f'E_PROFILE_STALE: {where}: uploaded PROFILE identity {str(uploaded)[:12]} != current optics '
                               f'{current[:12]}; a changed table needs a new optics/Solver (no rebinding after attach)')

    def raw_kernel(self, W, first=0, phase=0, boundary=0, ray_limit=False, variant=None, ground_sh_l=0):
        if W not in (1, 15):
            raise ValueError('E_WAVELENGTHS: kernel specializes to 1 or 15 wavelengths')
        width = self.tile_width(W, first, phase)
        flags = tuple(sorted(set(self.config(W).defines) | set(KSPLIT_VARIANTS[variant]))) if variant else self.config(W).defines
        stage1 = self.stage1_flags()
        # kernel_source(L) differs per SH order and PROFILE_N per node count; shared raw dicts must not mix them
        key = self.kernel_key(flags, W, width, first, phase, boundary, ground_sh_l=ground_sh_l, ray_limit=ray_limit, stage1=stage1)
        akey = ('+'.join(flags) or 'default')+':'+f'{W}_first{int(first)}_phase{int(phase)}_boundary{int(boundary)}'
        akey += f'_tile{width}' + ('_raylimit' if ray_limit else '')
        if ground_sh_l: akey += f'_groundsh{ground_sh_l}'
        akey += self.attribute_suffix()
        if key not in self.raw:
            code = kernel_source(self.L, combine=(self.config(W).c3_fw15 and not first))
            options = ('-std=c++17', '--fmad=true' if 'FMAD' in stage1 else '--fmad=false',
                       '--prec-div=false' if 'FASTMATH' in stage1 else '--prec-div=true',
                       '--prec-sqrt=false' if 'FASTMATH' in stage1 else '--prec-sqrt=true', f'-DFW={W}')  # no --ftz: CuPy 14 RawKernel always adds -ftz=true (NVRTC 12.8.93 'defined more than once'); production already ran ftz=true (T1 exp8 bit-identical to 10-01 v1 pack, AUTOPILOT 10-02 04:55)
            options += (f'-DSOSFUSE2_DETERMINISTIC={int(self.deterministic)}',)
            options += tuple('-DSOSFUSE2_'+f for f in stage1 if f != 'FMAD')
            options += tuple(f'-D{f}' for f in flags)
            options += (f'-DK_FIRST={int(first)}', f'-DK_PHASE={int(phase)}', f'-DK_BOUNDARY={int(boundary)}')
            if ray_limit: options += ('-DLAUNCH_RAY_LIMIT',)
            if ground_sh_l:
                from .ground_sh import GRID_INTERVALS
                options += (f'-DGROUND_SH_L={ground_sh_l}', f'-DGROUND_SH_GRID={GRID_INTERVALS}')
            if self.profile_n != PROFILE_N_DEFAULT: options += (f'-DPROFILE_N={self.profile_n}',)
            with stage('raw_compile'):
                raw = self.xp.RawKernel(code, 'sos_fused', options=options, backend='nvrtc')
                raw.compile()
            self.__dict__.setdefault('raw_options', {})[key] = (code, options)
            required=self.shared_bytes(width, 0, first=first, variant=variant)
            props=self.xp.cuda.runtime.getDeviceProperties(self.xp.cuda.runtime.getDevice())
            maximum=int(props.get('sharedMemPerBlockOptin', props['sharedMemPerBlock']))
            if required>maximum:
                raise RuntimeError(f'E_SHARED_MEMORY: need {required} bytes, device supports {maximum}; reduce --block-size')
            raw.max_dynamic_shared_size_bytes=required
            raw.preferred_shared_memory_carveout=100
            self.raw[key]=raw
            self.attributes[akey] = dict(raw.attributes, block_size=self.bs(first, variant) if variant else self.block_size, shared_bytes=required,
                                           compiler='NVRTC', implicit_fma='--fmad=true' in options, stage1=list(stage1), coordinate_precision='two-word FP32',
                                           flags=list(flags), **({} if self.profile_n == PROFILE_N_DEFAULT else dict(profile_n=self.profile_n)))
            if ground_sh_l:
                self.attributes[akey]['ground_sh'] = dict(order=ground_sh_l, grid_intervals=GRID_INTERVALS)
        required=self.shared_bytes(width,0,first=first,variant=variant)
        self.raw[key].max_dynamic_shared_size_bytes=required
        self.attributes[akey].update(block_size=self.bs(first, variant) if variant else self.block_size,shared_bytes=required)
        if not variant:self.attributes[str(W)]=self.attributes[akey]
        return self.raw[key]

    def fl_project_kernel(self, W, boundary=0, ground_sh_l=0):
        """kernel.cu fl_project from the same source and options as the ray-range first-light kernel."""
        stage1 = self.stage1_flags()
        key = self.kernel_key(self.config(W).defines, W, self.tile_width(W, True, 0), 1, 0, boundary, ground_sh_l=ground_sh_l,
                              ray_limit=True, stage1=stage1)   # same key as raw_kernel(..., ray_limit=True)
        pkey = key + ('fl_project',)
        if pkey not in self.raw:
            code, options = self.raw_options[key]
            with stage('raw_compile'):
                raw = self.xp.RawKernel(code, 'fl_project', options=options, backend='nvrtc')
                raw.compile()
            self.raw[pkey] = raw
        return self.raw[pkey]

    def constants_device(self, s, nq):
        key=(id(s), nq)
        if key not in self.device_constants:
            c=self.setup(s, nq)
            with stage('constants_device',device=True):
                names=('h','chi','hr','ha','ext','asca','sun_h','sun_col','gm','goff','gdir','gval','u','wu')
                dc={name:self.xp.asarray(np.ascontiguousarray(c[name])) for name in names}
                if 'strato_ext' in c:
                    for name in ('strato_ext','strato_sca','strato_chi','strato_density','strato_sun_col'):
                        dc[name]=self.xp.asarray(np.ascontiguousarray(c[name]))
                if 'strato_ext' in c and self.config().strato:
                    dc['strato_chi'] = self.strato_lut_device(dc['strato_chi'], c['W'])
                if self.ground_sh_l and c['gval'].size:
                    from .ground_sh import lut_eigenvalues, project, synthesize
                    # The accepted state is immutable, and ground illumination
                    # does not depend on the ray quadrature order nq. Observer,
                    # first-light and table transport share this one allocation.
                    ground_key = (id(s), self.ground_sh_l)
                    if ground_key not in self.ground_sh_tables:
                        lut = self.host(dc['strato_chi']).reshape(-1, c['W'])[33:]
                        eigen = lut_eigenvalues(lut, self.ground_sh_l)
                        coeff = project(c['goff'], c['gdir'], c['gval'], c['g'], eigen, self.ground_sh_l)
                        self.ground_sh_tables[ground_key] = synthesize(coeff, self.ground_sh_l, self.xp)
                    dc['ground_sh'] = self.ground_sh_tables[ground_key]
                    self.metrics['ground_sh_order'] = self.ground_sh_l
                    self.metrics['ground_sh_bytes'] = int(dc['ground_sh'].nbytes)
                if 'density_optics' in c:
                    from .density import upload_buffer, evidence
                    buf = upload_buffer(c['density_optics'])
                    self.check_profile(c['density_optics'], 'upload', buf.shape[0])
                    dc['profile_table'] = self.xp.asarray(buf)
                    self.profile_consumed = evidence(c['density_optics'], self.profile_n, buf)
                    # The identity this upload belongs to (checked on every reuse and at launch).
                    self.__dict__.setdefault('device_profile_identity', {})[id(dc['profile_table'])] = self.profile_consumed['profile_identity']
                if ('tropo_lut' in c)!=('TROPO_MIE' in self.flags):raise RuntimeError('E_TROPO_LUT_FLAGS: table and TROPO_MIE must come together')
                if 'tropo_lut' in c:
                    # kernel.cu TROPO_LUT = asca+FW: asca[0:W] unchanged, the per-wavelength phase table follows.
                    if c['tropo_lut'].size!=(c['tropo_bins']+1)*c['W'] or c['tropo_bins']!=16384:raise ValueError('E_TROPO_LUT_SHAPE')
                    dc['asca']=self.xp.asarray(np.ascontiguousarray(np.concatenate([np.asarray(c['asca'],'f4'),c['tropo_lut']]),'f4'))
                if ('aer2' in c)!=('AEROSOL_2L' in self.flags):raise RuntimeError('E_AEROSOL_2L_FLAGS: profile vector and AEROSOL_2L must come together')
                if 'aer2' in c:
                    # kernel.cu aerosol_two_layer reads sun_h+nsh: sun_h[0:nsh] unchanged (nsh = len(c['sun_h'])), the 7 parameters follow.
                    if c['aer2'].shape!=(7,):raise ValueError('E_AEROSOL_2L_SHAPE')
                    dc['sun_h']=self.xp.asarray(np.ascontiguousarray(np.concatenate([np.asarray(c['sun_h'],'f4'),c['aer2']]),'f4'))
                dc.update(zip(('aa','bb','dd'), [self.xp.asarray(v) for v in norm_constants(self.L)]))
                # sos_vprod_y (sh_basis_ds) reads the same constants as (hi, lo) float pairs
                for name, v in zip(('aa2','bb2','dd2'), norm_constants(self.L)):
                    hi = v.astype('f4'); dc[name] = self.xp.asarray(np.stack([hi, (v-hi.astype('f8')).astype('f4')], 1))
            self.device_constants[key]=dc
        self.check_upload(self.setup(s, nq), self.device_constants[key], 'device_constants')
        if self.config().strato and 'strato_chi' in self.device_constants[key]:
            bins=16384
            expected=(33+bins+1)*self.setup(s,nq)['W']
            if self.device_constants[key]['strato_chi'].size != expected:
                raise RuntimeError('E_STRATO_LUT_CACHE: phase flags changed after setup; create a fresh Kernel')
        return self.device_constants[key]

    def strato_lut_kernel(self, W):
        """strato_lut_build, compiled before its first launch (precompile.py compiles the same source/options)."""
        code=Path(__file__).with_name('kernel.cu').read_text()
        options=('-std=c++17','--fmad=false','--prec-div=true','--prec-sqrt=true',
                 f'-DFW={W}')
        options += (f'-DSOSFUSE2_DETERMINISTIC={int(self.deterministic)}',)
        options += tuple('-D'+f for f in resolve_config(self.flags, role='lut', W=W).flags)
        with stage('raw_compile'):
            builder=self.xp.RawKernel(code,'strato_lut_build',options=options,backend='nvrtc')
            builder.compile()
        return builder

    def strato_lut_device(self, coeff, W):
        """Same pointer argument, enlarged private allocation; nodes built on GPU."""
        bins=16384
        xp=self.xp
        if xp is np:
            raise RuntimeError('E_STRATO_LUT_BACKEND: this builder requires CuPy; use native phase tests on CPU')
        if coeff.size != 33*W:
            raise ValueError('E_STRATO_CHI_SHAPE: 33*W coefficients required')
        out=xp.empty((33+bins+1)*W, dtype='f4');out[:33*W]=coeff.ravel()
        builder=self.strato_lut_kernel(W)
        builder((((bins+1)*W+255)//256,), (256,), (out,))
        # Same stream orders the table producer before all consumers.
        self.metrics['strato_lut_bytes']=int(out.nbytes)
        self.metrics['strato_lut_intervals']=bins
        return out


    def strato_end_columns(self,bundle,c):
        """Bounded-memory sidecar on immutable Bundle; never part of cache schema."""
        from .stratosphere import columns
        profile=c['strato_profile'];key=tuple(profile[k] for k in ('center','sigma','norm'))
        if not hasattr(bundle,'strato_columns'):bundle.strato_columns={}
        if key not in bundle.strato_columns:
            from .column_cache import cached_columns
            def build():
                starts=self.host(bundle.arrays['starts']);co=self.host(bundle.arrays['cell_off'])
                dirs=self.host(bundle.arrays['dirs']);out=np.empty(bundle.rays,'f4')
                from .column_parallel import parallel_columns
                if parallel_columns(out,'strato',starts,co,dirs,c['strato_old'],profile):return out  # parallel preparation, rtol 1e-6
                # Do not generate full-grid [ray,GLnode,xyz] arrays on the host.
                for start in range(0,bundle.rays,2048):
                    end=min(start+2048,bundle.rays);index=np.searchsorted(co,np.arange(start,end),side='right')-1
                    out[start:end]=columns(starts[index],dirs[start:end],c['strato_old'],profile,24)
                return out
            out=cached_columns(self,bundle,c,'strato',build)
            bundle.strato_columns[key]=self.xp.asarray(out)
        return bundle.strato_columns[key]

    def density_end_columns(self,bundle,c):
        from .density import correct_columns
        from .density import profile_identity
        optics=c['density_optics'];key=profile_identity(optics)['sha256']   # effective table, grid, mask, policy
        if not hasattr(bundle,'density_columns'):bundle.density_columns={}
        if key not in bundle.density_columns:
            from .column_cache import cached_columns
            def build():
                starts=self.host(bundle.arrays['starts']);co=self.host(bundle.arrays['cell_off'])
                dirs=self.host(bundle.arrays['dirs']);out=self.host(bundle.arrays['end_col']).copy()
                from .column_parallel import parallel_columns
                if parallel_columns(out,'density',starts,co,dirs,optics['old'],optics):return out  # parallel preparation, rtol 1e-6
                for start in range(0,bundle.rays,2048):
                    end=min(start+2048,bundle.rays);index=np.searchsorted(co,np.arange(start,end),side='right')-1
                    out[start:end]=correct_columns(starts[index],dirs[start:end],out[start:end],optics['old'],optics,24)
                return out
            out=cached_columns(self,bundle,c,'density',build)
            bundle.density_columns[key]=self.xp.asarray(np.ascontiguousarray(out,'f4'))
        return bundle.density_columns[key]

    def ksplit_pointer_args(self, kb):
        """SINV_PARAM order. Resolve at launch time: aero can be resized by another bundle,
        while VERIFY and G/L allocate their buffers after the base argument list was built.
        The c0cp address is passed as a CUDA int2* (including mapped pinned host storage);
        its owning allocation remains in kb until the bundle is released. Zero means unused.
        """
        aero = getattr(self, 'ksplit_aero', None)
        return [aero if aero is not None else np.uint64(0),
                kb.get('c0cp_ptr', np.uint64(0)), kb.get('dump', np.uint64(0))]

    def ksplit_memory(self):
        """Usable free bytes, including cached pool blocks and respecting its limit."""
        pool = self.xp.get_default_memory_pool()
        free = int(self.xp.cuda.runtime.memGetInfo()[0]) + int(pool.free_bytes())
        if pool.get_limit():
            free = min(free, int(pool.get_limit())-int(pool.used_bytes()))
        return max(0, free)

    def ksplit_preflight(self, s, bundles):
        """One decision before any split allocation, over all phase-0/1 bundles.

        Spectral chunks have the same geometry and padded W; their caller carries
        this decision forward. Query (phase 2) never allocates split buffers.
        Other service states' measured allocations are added back only to decide
        whether this state could fit alone. Insufficient live room still waits.
        """
        # Geometry survives spectral chunks, but split records belong to the
        # chunk's solver. Release every old bundle before measuring free memory.
        for bundle, _ in bundles:
            if getattr(bundle, 'ksplit', None) is not None and bundle.ksplit['state'] != id(s):
                bundle.ksplit = None
        previous = getattr(s, 'ksplit_decision', None)
        if previous is not None:
            self.ksplit_mode = previous['effective']; self.ksplit = bool(self.ksplit_mode)
            if not self.ksplit or getattr(self, '_ksplit_preflight_state', None) == id(s):
                return previous
        requested = self.ksplit_mode
        if not requested:
            s.ksplit_decision = dict(requested=0, effective=0)
            return s.ksplit_decision
        flag = os.environ.get('SOSFUSE2_KSPLIT_FALLBACK', '1')
        if flag not in ('0', '1'):
            raise ValueError('E_KSPLIT_FALLBACK: expected 0 or 1')
        W = s.W; nh, nc = len(s.geo.h), len(s.geo.chi)
        field_bytes = nh*nc*source_size(s)*W*4
        text = os.environ.get('SOSFUSE2_KSPLIT_AERO_MB', '').strip()
        cap = max(1, int(float(text)*2**20)//(4*aero_w(W)*4)) if text else None
        rows = []
        for bundle, phase in bundles:
            co = np.asarray(self.host(bundle.arrays['cell_off']), 'i8')
            pcum = (np.asarray(self.host(bundle.arrays['edge_off'][bundle.arrays['cell_off']]), 'i8')-co)*bundle.nq
            points = int(pcum[-1]); column = (16+SINV_F*points)*4
            S = 2 if self.config().c3_fw15 else 1
            fixed = column + (S*bundle.cells*(4+self.K)*W*8 if phase == 0 else 0) + 5*nh*nc*self.K*W*4
            rows.append(dict(bundle=bundle, co=co, pcum=pcum, points=points, column=column, fixed=fixed))
        while True:
            live = self.ksplit_memory()
            solo = live + int(getattr(self, 'ksplit_other_bytes', lambda: 0)())
            if requested == 1:
                peak = sum(r['column'] for r in rows) + (cap or max(1, (1024<<20)//(4*aero_w(W)*4)))*4*aero_w(W)*4
                budget = int(.9*solo); live_budget = int(.9*live)
            else:
                # Enumerate the same stored/rebuilt and device/host options as
                # vprod_budget. Retained records from OTHER bundles are live too.
                import itertools
                choices = []
                for row in rows:
                    result = self.vprod_budget(row['co'], row['pcum'], row['points'], W, row['fixed'], solo,
                                               field_bytes, cap, minimum=True)
                    if result.get('error'):
                        raise MemoryError('E_KSPLIT_MEMORY: '+result['error'])
                    choices.append(result['options'])
                def state_peak(combo):
                    held = [row['column']+r['records_bytes']+r['store_bytes'] for row,r in zip(rows,combo)]
                    return sum(held)+max(r['peak']-h for r,h in zip(combo,held))
                combinations = list(itertools.product(*choices))
                peak = min(map(state_peak, combinations))
                budget = solo-16*field_bytes-max(256<<20, int(.03*solo))
                live_budget = live-16*field_bytes-max(256<<20, int(.03*live))
            decision = dict(requested=requested, effective=requested, peak=int(peak), budget=int(budget),
                            free=int(live), points=sum(r['points'] for r in rows), reason='fits')
            if peak > budget:
                if flag == '0' or previous is not None:
                    raise MemoryError(f'E_KSPLIT_MEMORY: state minimum peak {peak} exceeds solo budget {budget}')
                decision.update(effective=0, reason='state_minimum_exceeds_solo_budget')
                self.ksplit_mode = 0; self.ksplit = False
                hold = getattr(self, 'ksplit_hold', None)
                if hold is not None: hold(0)
                break
            if peak <= live_budget:
                if requested >= 2:
                    # Preserve vprod_budget's preference order, but size BOTH
                    # bundles together: the first cannot spend the second's room.
                    plans = list(next(c for c in combinations if state_peak(c) <= live_budget))
                    for i,row in enumerate(rows):
                        lo, hi = plans[i]['cap'], max(plans[i]['cap'], row['points'])
                        if cap is not None: hi = lo
                        while lo < hi:
                            mid = (lo+hi+1)//2
                            candidate = self.vprod_budget(row['co'], row['pcum'], row['points'], W, row['fixed'], live,
                                field_bytes, cap, option=(plans[i]['store'], plans[i]['resident'], mid))
                            trial = list(plans); trial[i] = candidate
                            if state_peak(trial) <= live_budget: lo = mid; plans = trial
                            else: hi = mid-1
                    for row,plan in zip(rows,plans):
                        # A plan may not grow its retained storage beyond the
                        # global estimate. Existing overflow handling rebuilds it.
                        plan['store_allowed'] = plan['store_bytes']
                        row['bundle']._ksplit_plan = (id(s), plan)
                    self.ksplit_planned_bytes = state_peak(plans)
                else:
                    self.ksplit_planned_bytes = peak
                hold = getattr(self, 'ksplit_hold', None)
                if hold is not None: hold(self.ksplit_planned_bytes)
                break
            wait = getattr(self, 'ksplit_wait', None)
            if wait is None or not wait():
                raise MemoryError('E_KSPLIT_MEMORY: other resident states leave insufficient room')
        self._ksplit_preflight_state = id(s)
        s.ksplit_decision = decision = previous or decision
        if decision['effective'] == 0:
            emit = getattr(self, 'ksplit_emit', None)
            if emit is not None:
                emit(decision)
            else:
                state_id = getattr(s, 'fuse_state', {}).get('id', 'UNKNOWN')
                from .column_parallel import event_fields
                print(json.dumps(dict(event='PHASE', phase='KSPLIT_FALLBACK', **(event_fields() | dict(id=state_id)), **decision)), flush=True)
        return decision

    def ksplit_columns(self, bundle, s, field, phase, boundary):
        """KSPLIT: the bundle's column records (kernel.cu SINV, 28 B/point, recorded once per state by a mode-1 pass) and
        the chunk aerosol buffer shared by all bundles ([point][corner][w] FP32, SOSFUSE2_KSPLIT_AERO_MB, default 1024).
        SOSFUSE2_KSPLIT=2 (VPROD) sizes and allocates its own way (vprod_columns); KSPLIT=1 is unchanged."""
        kb = getattr(bundle, 'ksplit', None)
        if kb is not None and kb['state'] == id(s):
            return kb
        if self.ksplit_mode >= 2:
            return self.vprod_columns(bundle, s, field, phase, boundary)
        bundle.ksplit = None; xp = self.xp; W = s.W; nq = bundle.nq
        co = np.asarray(self.host(bundle.arrays['cell_off']), 'i8')
        pcum = (np.asarray(self.host(bundle.arrays['edge_off'][bundle.arrays['cell_off']]), 'i8') - co)*nq
        points = int(pcum[-1])                              # SINV_REC indexes (seg-ray)*nq+step from ray 0
        text = os.environ.get('SOSFUSE2_KSPLIT_AERO_MB', '').strip() or '1024'
        AW = aero_w(W); cap_points = max(1, int(float(text)*2**20)//(4*AW*4))
        aero = self.__dict__.get('ksplit_aero')
        need = (16+SINV_F*points)*4 + (0 if aero is not None and aero.size >= cap_points*4*AW else cap_points*4*AW*4)
        pool = xp.get_default_memory_pool(); free = int(xp.cuda.runtime.memGetInfo()[0]) + int(pool.free_bytes())
        if need > .9*free:
            raise MemoryError(f'E_KSPLIT_MEMORY: KSPLIT needs {need/1e9:.2f} GB of device memory for {points} points '
                              f'({bundle.cells} cells, phase {phase}), {free/1e9:.2f} GB free; lower SOSFUSE2_KSPLIT_AERO_MB or unset SOSFUSE2_KSPLIT')
        if aero is None or aero.size < cap_points*4*AW:
            self.ksplit_aero = aero = None; aero = self.ksplit_aero = xp.empty(cap_points*4*AW, 'f4')
        buf = xp.empty(16+SINV_F*points, 'f4'); hdr32 = buf[:16].view('i4'); hdr64 = buf[:16].view('i8'); hdr64.fill(0)
        kb = bundle.ksplit = dict(state=id(s), buf=buf, hdr32=hdr32, hdr64=hdr64, pcum=pcum, cap_points=cap_points)
        print('KSPLIT '+json.dumps(dict(event='columns', phase=int(phase), cells=int(bundle.cells), rays=int(bundle.rays), nq=int(nq),
                                       points=points, column_bytes=int(buf.nbytes), aero_bytes=int(aero.nbytes),
                                       aero_points=cap_points, free_bytes=free, block=KSPLIT_BLOCK)), flush=True)
        hdr32[0:1].fill(1); started = time.perf_counter()
        self.launch(bundle, s, field, False, phase, boundary, _ksplit='record'); self.sync()
        print('KSPLIT '+json.dumps(dict(event='recorded', phase=int(phase), seconds=time.perf_counter()-started)), flush=True)
        return kb

    def vprod_columns(self, bundle, s, field, phase, boundary):
        """SOSFUSE2_KSPLIT=2 per bundle and state: columns, (c0, cp) records, the fixed chunk partition, the stored visitor plans
        and the shared aerosol buffer, all sized by vprod_budget. Order: columns + records (record pass) -> plans of every chunk
        (stored mode, before the aerosol buffer exists) -> aerosol buffer. The aerosol buffer is shared by the state's bundles
        and re-allocated at this bundle's largest chunk; each launch passes the current buffer pointer, and a
        bundle whose largest chunk no longer fits is re-partitioned there (vprod_fit)."""
        bundle.ksplit = None; xp = self.xp; W = s.W; nq = bundle.nq; nh, nc = len(s.geo.h), len(s.geo.chi); K = self.K
        co = np.asarray(self.host(bundle.arrays['cell_off']), 'i8')
        pcum = (np.asarray(self.host(bundle.arrays['edge_off'][bundle.arrays['cell_off']]), 'i8') - co)*nq
        points = int(pcum[-1])                              # SINV_REC indexes (seg-ray)*nq+step from ray 0
        S = 2 if self.config().c3_fw15 else 1
        # launch() arrays after this point: moments and CELL_SPLIT scratch (phase 0), the combined SH sidecar and its FP32
        # build temporaries (sh_layout.combined_aux, about 4 planes of [nh*nc][K][W])
        fixed = (16+SINV_F*points)*4 + (S*bundle.cells*(4+K)*W*8 if phase == 0 else 0) + 5*nh*nc*K*W*4
        text = os.environ.get('SOSFUSE2_KSPLIT_AERO_MB', '').strip()
        env_cap = max(1, int(float(text)*2**20)//(4*aero_w(W)*4)) if text else None
        hold = getattr(self, 'ksplit_hold', None); others = self.__dict__.setdefault('_ksplit_held', {}); others.pop(id(bundle), None)
        planned = getattr(bundle, '_ksplit_plan', None)
        while True:
            pool = xp.get_default_memory_pool(); limit = int(pool.get_limit())
            aero = self.__dict__.get('ksplit_aero'); held = int(aero.nbytes) if aero is not None else 0
            free = int(xp.cuda.runtime.memGetInfo()[0]) + int(pool.free_bytes())
            if limit:
                free = min(free, limit-int(pool.used_bytes()))   # pipeline_execution caps the pool at free-1 GB per service
            room = self.ksplit_room() if hold is not None else None
            if room is not None:  # persistent service, other states resident: stay out of their reservations (pipeline_execution)
                free = min(free, int(room)-sum(others.values())-held)
            free += held                                     # the shared aerosol buffer is released and re-sized below
            bud = (dict(planned[1]) if planned is not None and planned[0] == id(s) else
                   self.vprod_budget(co, pcum, points, W, fixed, free, nh*nc*source_size(s)*W*4, env_cap))
            # too little room beside the other resident states: wait until one releases memory (not while all of them wait)
            if not (bud.get('error') and room is not None and self.ksplit_wait()):
                break
        if bud.get('error'):
            raise MemoryError(f'E_KSPLIT_MEMORY: KSPLIT=2 for {points} points ({bundle.cells} cells, phase {phase}): {bud["error"]}; '
                              f'{free/1e9:.2f} GB available; lower SOSFUSE2_KSPLIT_AERO_MB or unset SOSFUSE2_KSPLIT')
        if aero is not None:
            self.ksplit_aero = aero = None                   # released before the plans are built
        buf = xp.empty(16+SINV_F*points, 'f4'); hdr32 = buf[:16].view('i4'); hdr64 = buf[:16].view('i8'); hdr64.fill(0)
        kb = bundle.ksplit = dict(state=id(s), buf=buf, hdr32=hdr32, hdr64=hdr64, pcum=pcum, co=co, cap_points=bud['cap'],
                                  chunks=bud['chunks'], pmax=bud['pmax'], resident=bud['resident'], store=bud['store'],
                                  plans={}, stored_bytes=0, iteration=0, c0cp_dev=None)
        # (c0, cp) records of every point, written by the record pass (kernel.cu SINV mode 1, vp_records argument); c0 = -1 marks
        # points of skipped segments (b-a <= 1e-8), which PROD never writes either.
        if bud['resident'] == 'device':
            kb['c0cp_dev'] = xp.full(2*points, -1, 'i4'); address = int(kb['c0cp_dev'].data.ptr)
        else:
            # Not enough device memory (pool limit) for the records (eda 10 GB): the record pass writes them into mapped pinned
            # host memory, and each chunk's points (8 B/point) are uploaded before its plan and VPROD launches.
            mem = kb['c0cp_mem'] = xp.cuda.PinnedMemory(8*max(points, 1), xp.cuda.runtime.hostAllocMapped)
            kb['c0cp_host'] = np.ndarray((2*max(points, 1),), np.int32, buffer=xp.cuda.PinnedMemoryPointer(mem, 0))
            # Unified addressing (64-bit Linux): a cudaHostAlloc pointer is used directly by kernels (CUDA programming guide).
            kb['c0cp_host'].fill(-1); address = int(mem.ptr)
            kb['c0cp_stage'] = xp.empty(2*max(bud['pmax'], 1), 'i4')
        kb['c0cp_ptr'] = np.uint64(address)  # CUDA int2* argument; kb owns device/pinned storage
        print('KSPLIT '+json.dumps(dict(event='columns', phase=int(phase), cells=int(bundle.cells), rays=int(bundle.rays), nq=int(nq),
                                       points=points, column_bytes=int(buf.nbytes), free_bytes=int(free), block=KSPLIT_BLOCK, mode=2,
                                       verify=bool(self.ksplit_verify), **{k: v for k, v in bud.items() if k != 'chunks'},
                                       chunk_count=len(bud['chunks']))), flush=True)
        hdr32[0:1].fill(1); started = time.perf_counter()
        self.launch(bundle, s, field, False, phase, boundary, _ksplit='record'); self.sync()
        print('KSPLIT '+json.dumps(dict(event='recorded', phase=int(phase), seconds=time.perf_counter()-started)), flush=True)
        if kb['store']:
            started = time.perf_counter(); allowed = bud['store_allowed']
            for ci, (c0, c1) in enumerate(kb['chunks']):
                base = int(pcum[c0]); npt = int(pcum[c1])-base
                plan = self.vprod_build_plan(kb, ci, bundle, self.vprod_c0cp(kb, base, npt), nh, nc)
                kb['plans'][ci] = plan; kb['stored_bytes'] += plan['bytes']
                if kb['stored_bytes'] > allowed:   # measured plans larger than the budget's estimate: rebuild per iteration
                    kb['plans'].clear(); kb['store'] = False; break
            self.sync()
            print('KSPLIT '+json.dumps(dict(event='plans', phase=int(phase), stored=bool(kb['store']), chunks=len(kb['chunks']),
                                           stored_bytes=int(kb['stored_bytes']), bytes_per_point=kb['stored_bytes']/max(points, 1),
                                           estimate_bytes=int(bud['store_bytes']), allowed_bytes=int(allowed),
                                           seconds=time.perf_counter()-started)), flush=True)
            if not kb['store']:
                kb['stored_bytes'] = 0
        # bytes this bundle keeps while the state's other bundles size theirs (the shared aerosol buffer is counted once below)
        others[id(bundle)] = int(buf.nbytes)+int(kb['stored_bytes'])+sum(int(kb[k].nbytes) for k in ('c0cp_dev', 'c0cp_stage')
                                                                           if kb.get(k) is not None)
        npts = bud['pmax']   # another bundle whose largest chunk no longer fits re-partitions at its next launch (vprod_fit)
        self.ksplit_aero = aero = xp.empty(max(npts, 1)*4*aero_w(W), 'f4')
        if hold is not None:  # the state's reservation grows by what KSPLIT holds, so later states are admitted only beside it
            hold(sum(others.values())+int(aero.nbytes))
        print('KSPLIT '+json.dumps(dict(event='aero', phase=int(phase), aero_points=npts, aero_bytes=int(aero.nbytes),
                                       auto=env_cap is None)), flush=True)
        return kb

    def vprod_budget(self, co, pcum, points, W, fixed, free, field_bytes, env_cap, *, minimum=False, option=None):
        """SOSFUSE2_KSPLIT=2 memory rule. Budget B = free (pool-limited, shared aerosol buffer counted as free) - reserve, reserve =
        16 x the source field (solver growth after this point: Anderson history, apply temporaries) + max(256 MB, 3 %). eda
        2026-10-25 (62.6 MB field): 8 x left cap 4.51e6 points and the cold solve ran out of memory in iteration 5
        (vprod2gf_run2); the earlier fixed 1 GB buffer (4.47e6 points) peaked at 8.44+ of 8.69 GB. Per option (stored plans first, then rebuild; records
        device-resident first, then pinned host), the largest aerosol chunk cap (points) whose peak fits B:
          run   = fixed + records + stored plans + aerosol 4W*4 B/pt of the largest chunk + y rows + verify piece
                  (+ build workspace when rebuilding every iteration)
          build = fixed + records + stored plans + build workspace (stored mode builds all plans before the aerosol buffer)
        records 8 B/point (device) or 8 B/pt of the largest chunk (host upload); stored plans estimated VPROD_STORE_BPP B/point
        (measured 25.1-26.1: slot 4/record, visitor 8, block 12); build workspace 21 B/record + 4 B/point (live bound; plan_ab.py
        pool 94 B/point incl. 8 B/point inputs); y K*4 B/ray. Stored mode needs a cap >= points/64 (at most 64 chunks);
        SOSFUSE2_KSPLIT_AERO_MB fixes the cap."""
        reserve = 16*field_bytes + max(256 << 20, int(.03*free)); B = free-reserve; K = self.K; ncell = len(pcum)-1
        lc = int(np.diff(pcum).max()) if ncell else 1
        stage = VPROD_VERIFY_STAGE*4 if self.ksplit_verify else 0
        mode = getattr(self, 'ksplit_mode', 2)

        def partition(cap):
            chunks = []; c0 = 0
            while c0 < ncell:
                hi = int(np.searchsorted(pcum, pcum[c0]+cap, side='right'))-1
                if hi <= c0:
                    return None
                chunks.append((c0, hi)); c0 = hi
            return chunks

        def peak(cap, store, resident):
            chunks = partition(cap)
            if chunks is None:
                return None
            pmax = max(int(pcum[b]-pcum[a]) for a, b in chunks); rmax = max(int(co[b]-co[a]) for a, b in chunks)
            ws = 21*4*pmax + 4*pmax + (1 << 20); y = rmax*K*4
            rec = 8*points if resident == 'device' else 8*pmax
            st = VPROD_STORE_BPP*points if store else 0
            run = fixed + rec + st + 4*aero_w(W)*4*pmax + y + stage + (0 if store else ws)
            if mode == 3:   # chunk ray radiances [ray][4W] FP32 (+ the VERIFY ray dump)
                run += 4*W*4*rmax*(2 if self.ksplit_verify else 1)
            build = fixed + rec + st + ws
            return dict(peak=max(run, build) if store else run, chunks=chunks, pmax=pmax, rmax=rmax, workspace=ws, y_bytes=y,
                        records_bytes=rec, store_bytes=st, run=run, cap=cap, store=store, resident=resident,
                        budget=int(B), reserve=int(reserve), store_allowed=st)

        if option is not None:
            store, resident, cap = option
            return peak(cap, store, resident)
        if minimum:
            options = []
            for store, resident in ((True, 'device'), (True, 'host'), (False, 'device'), (False, 'host')):
                cap = env_cap if env_cap is not None else max(lc, -(-points//64)) if store else lc
                r = peak(cap, store, resident)
                if r is not None: options.append(r)
            return dict(options=options) if options else dict(error='no whole cell fits the requested aerosol cap')
        tried = []
        for store, resident in ((True, 'device'), (True, 'host'), (False, 'device'), (False, 'host')):
            lo = max(lc, -(-points//64)) if store else lc
            if env_cap is not None:
                if env_cap < lc:
                    return dict(error=f'cell with {lc} points > aerosol chunk {env_cap} (SOSFUSE2_KSPLIT_AERO_MB)')
                cap = env_cap
                r = peak(cap, store, resident)
                if r is None or r['peak'] > B:
                    tried.append((store, resident, r and r['peak'])); continue
            else:
                hi = max(points, lc); r = peak(lo, store, resident)
                if r is None or r['peak'] > B:
                    tried.append((store, resident, r and r['peak'])); continue
                cap = lo
                while lo < hi:                       # largest cap that fits (peak grows with the cap)
                    mid = (lo+hi+1)//2; q = peak(mid, store, resident)
                    if q is not None and q['peak'] <= B: lo = mid
                    else: hi = mid-1
                cap = lo; r = peak(cap, store, resident)
            # stored plans may grow up to what the chosen cap leaves free (measured during the build)
            r.update(cap=cap, store=store, resident=resident, budget=int(B), reserve=int(reserve),
                     store_allowed=(int(B-(r['peak']-r['store_bytes'])) if store else 0))
            return r
        return dict(error=f'no option fits {B/1e9:.2f} GB (peaks {[(a, b, c and round(c/1e9, 2)) for a, b, c in tried]})')

    def vprod_fit(self, kb, bundle, cap, phase, nh, nc):
        """Re-partition a bundle whose largest chunk exceeds the shared aerosol buffer (re-sized by a later bundle): cells are
        not split; stored plans are rebuilt for the new chunks (the small ground bundle in practice)."""
        if kb['pmax'] <= cap:
            return
        pc = kb['pcum']; chunks = []; c0 = 0; ncell = len(pc)-1
        while c0 < ncell:
            hi = int(np.searchsorted(pc, pc[c0]+cap, side='right'))-1
            if hi <= c0:
                raise MemoryError(f'E_KSPLIT_MEMORY: cell {c0} has {int(pc[c0+1]-pc[c0])} points > shared aerosol chunk {cap}')
            chunks.append((c0, hi)); c0 = hi
        old = len(kb['chunks']); kb.update(chunks=chunks, pmax=max(int(pc[b]-pc[a]) for a, b in chunks), cap_points=cap)
        kb['plans'] = {}; kb['stored_bytes'] = 0
        if kb['store']:
            for ci, (c0, c1) in enumerate(chunks):
                base = int(pc[c0]); npt = int(pc[c1])-base
                plan = self.vprod_build_plan(kb, ci, bundle, self.vprod_c0cp(kb, base, npt), nh, nc)
                kb['plans'][ci] = plan; kb['stored_bytes'] += plan['bytes']
        print('KSPLIT '+json.dumps(dict(event='repartition', phase=int(phase), chunks_before=old, chunks=len(chunks), cap=int(cap),
                                       stored=bool(kb['store']), stored_bytes=int(kb['stored_bytes']))), flush=True)

    def vprod_kernels(self, W, phase, boundary):
        """sos_vprod_y and sos_vprod from the PROD module (same source and NVRTC options as the PROD kernel; cf. fl_project_kernel)."""
        self.raw_kernel(W, 0, phase, boundary, variant='prod')
        flags = tuple(sorted(set(self.config(W).defines) | set(KSPLIT_VARIANTS['prod'])))
        key = self.kernel_key(flags, W, self.tile_width(W, False, phase), 0, phase, boundary, stage1=self.stage1_flags())
        out = []
        for name in ('sos_vprod_y', 'sos_vprod'):
            nkey = key + (name,)
            if nkey not in self.raw:
                code, options = self.raw_options[key]
                with stage('raw_compile'):
                    raw = self.xp.RawKernel(code, name, options=options, backend='nvrtc')
                    raw.compile()
                self.raw[nkey] = raw
                self.attributes[f'{name}:{W}_phase{int(phase)}_boundary{int(boundary)}'+self.attribute_suffix()] = dict(raw.attributes, block_size=VPROD_THREADS)
            out.append(self.raw[nkey])
        return out

    def vprod_plan_module(self):
        mod = self.__dict__.get('_vprod_plan_mod')
        if mod is None:
            mod = self._vprod_plan_mod = self.xp.RawModule(code=VPROD_PLAN_SRC, options=('-std=c++17', f'-DSOSFUSE2_DETERMINISTIC={int(self.deterministic)}'))
        return mod

    def vprod_c0cp(self, kb, base, npt):
        """The chunk's (c0, cp) int2 records as an int32 [2*npt] device array (view, or upload of the pinned host copy)."""
        if kb['c0cp_dev'] is not None:
            return kb['c0cp_dev'][2*base:2*(base+npt)]
        stage_ = kb['c0cp_stage'][:2*npt]
        stage_.set(kb['c0cp_host'][2*base:2*(base+npt)])
        return stage_

    def vprod_plan(self, kb, ci, bundle, c0cp, nh, nc):
        """Visitor plan of fixed chunk ci: the stored one (built once per state by vprod_columns) or a new one."""
        if ci in kb['plans']:
            return kb['plans'][ci], True
        return self.vprod_build_plan(kb, ci, bundle, c0cp, nh, nc), False

    def vprod_build_plan(self, kb, ci, bundle, c0cp, nh, nc):
        """Records stable-sorted by source cell (slot order within a cell), visitors = runs of one (cell, ray), blocks of
        <= VPROD_VISITORS visitors of one cell. The y rows are not part of the plan (vprod_launch computes them per chunk)."""
        xp = self.xp; mod = self.vprod_plan_module(); T = 256
        g = lambda m: ((int(m)+T-1)//T,)
        c0, c1 = kb['chunks'][ci]; pc = kb['pcum']; co = kb['co']
        base = int(pc[c0]); npt = int(pc[c1])-base; r0, r1 = int(co[c0]), int(co[c1]); nray = r1-r0
        ncell = nh*nc; n = 4*npt
        nrec = 4*int(xp.count_nonzero(c0cp[0::2] >= 0)) if npt else 0
        slot = xp.empty(max(nrec, 1), 'i4'); cell = xp.empty(max(nrec, 1), 'i4')
        rstart = None
        if nrec and 4*ncell <= VPROD_COUNT_SHARED:
            seg = -(-max(4096, -(-n//VPROD_COUNT_BLOCKS))//128)*128; nb = -(-n//seg)   # 128 records = vp_scatter's 4 tiles
            err = xp.zeros(1, 'i4'); counts = xp.empty(ncell*nb, 'i4')
            args = (np.int64(n), c0cp, np.int32(nc), np.int32(ncell), np.int32(seg), np.int32(nb))
            mod.get_function('vp_count')((nb,), (T,), args+(counts, err), shared_mem=4*ncell)
            offs = xp.cumsum(counts, dtype=xp.int32); offs -= counts
            mod.get_function('vp_scatter')((nb,), (32,), args+(offs, slot), shared_mem=4*ncell)
            rstart = xp.empty(ncell+1, xp.int64)
            mod.get_function('vp_rstart')(g(ncell+1), (T,), (np.int32(ncell), np.int32(nb), np.int64(nrec), offs, rstart))
            mod.get_function('vp_fill_cell')((ncell,), (T,), (np.int32(ncell), rstart, cell))
            if int(err[0]):
                raise ValueError(f'E_VPROD_CELL_RANGE: a record cell is outside 0..{ncell-1} (chunk {ci})')
            del err, counts, offs
        elif nrec:   # more cells than one block's shared counters: the key sort (same slot/cell)
            key = xp.empty(n, xp.uint64)
            mod.get_function('vp_keys')(g(n), (T,), (np.int64(n), c0cp, np.int32(nc), key))
            key.sort()
            mod.get_function('vp_split')(g(nrec), (T,), (np.int64(nrec), key, slot, cell))
            del key
        pray = xp.empty(max(npt, 1), 'i4')
        if nray:
            mod.get_function('vp_point_ray')(g(nray), (T,), (np.int32(nray), np.int64(r0), bundle.arrays['edge_off'],
                                                            np.int32(bundle.nq), np.int64(base), pray))
        flag = xp.zeros(max(nrec, 1), xp.uint8)
        if nrec:
            mod.get_function('vp_flags')(g(nrec), (T,), (np.int64(nrec), slot, cell, pray, flag))
        scan = xp.cumsum(flag, dtype=xp.int32)
        nvis = int(scan[nrec-1]) if nrec else 0
        vis_rbeg = xp.empty(nvis+1, 'i4'); vis_ray = xp.empty(max(nvis, 1), 'i4')
        if nrec:
            mod.get_function('vp_visitors')(g(nrec), (T,), (np.int64(nrec), flag, scan, slot, pray, vis_rbeg, vis_ray))
        vis_rbeg[nvis:].fill(nrec)
        if rstart is None:
            rstart = xp.searchsorted(cell[:nrec], xp.arange(ncell+1, dtype=xp.int32)).astype(xp.int64)
        cstart = xp.empty(ncell+1, xp.int64)
        mod.get_function('vp_cells')(g(ncell+1), (T,), (np.int32(ncell), np.int64(nrec), np.int32(nvis), rstart, scan, cstart))
        del flag, scan, cell, rstart, pray
        nb = (xp.diff(cstart)+VPROD_VISITORS-1)//VPROD_VISITORS
        bstart = xp.concatenate([xp.zeros(1, xp.int64), xp.cumsum(nb)]); nblk = int(bstart[-1])
        blk = xp.empty(max(3*nblk, 3), 'i4'); work = xp.zeros(max(nblk, 1), xp.int64)
        mod.get_function('vp_blocks')(g(ncell), (T,), (np.int32(ncell), cstart, bstart, np.int32(VPROD_VISITORS), vis_rbeg, blk, work))
        if VPROD_ORDER and nblk:
            # Block issue order: source cells in tiles of th heights x tc chi columns (row-major
            # inside a tile, tiles row-major), a cell's blocks consecutive (stable argsort). Every record is computed and written
            # on its own, so the order changes no value; it only changes which y rows and aerosol lines share L2 in time.
            th, tc = VPROD_ORDER; b3 = blk[:3*nblk].reshape(nblk, 3); cb = b3[:, 0].astype(xp.int64); ih, ic = cb//nc, cb % nc
            order = xp.argsort(((ih//th)*((nc+tc-1)//tc)+ic//tc)*(th*tc)+(ih % th)*tc+ic % tc, kind='stable')
            blk = xp.ascontiguousarray(b3[order]).reshape(-1); work = work[order]; del b3, cb, ih, ic, order
        bcum = np.r_[0., np.cumsum(np.asarray(self.host(work[:nblk]), 'f8'))]
        del work, cstart, bstart, nb
        plan = dict(slot=slot, vis_rbeg=vis_rbeg, vis_ray=vis_ray, blk=blk, nrec=nrec, nvis=nvis, nblk=nblk, bcum=bcum,
                    npt=npt, nray=nray)
        plan['bytes'] = int(sum(a.nbytes for a in (slot, vis_rbeg, vis_ray, blk)))
        return plan

    def ksplit_chunk(self, kb, c0, c1):
        """Cap a chunk to the aerosol buffer and point the header at its first point (kernel.cu SLIM_AERO)."""
        pc = kb['pcum']; hi = int(np.searchsorted(pc, pc[c0]+kb['cap_points'], side='right'))-1
        if hi <= c0:
            raise MemoryError(f'E_KSPLIT_MEMORY: cell {c0} has {int(pc[c0+1]-pc[c0])} points > aerosol chunk {kb["cap_points"]}; '
                              'raise SOSFUSE2_KSPLIT_AERO_MB')
        kb['hdr64'][2:3].fill(int(pc[c0]))
        return min(c1, hi)

    def budget_end(self, kind, a, n, step, cum, initial=None):
        """End of the next launch inside a fixed KSPLIT=2 chunk [a, n): the whole rest when its predicted time (predict_seconds,
        the next_chunk_end model) fits the profile target, else next_chunk_end. Without this, next_chunk_end's 1.5x growth limit
        restarts from the previous chunk's tail and splits every chunk into small underfilled launches (eda 2026-10-25:
        SLIM 1192 launches 71.0 s vs 449 launches 32.4 s at KSPLIT=1, same work)."""
        if cum is not None and self.__dict__.get('_launch_obs', {}).get(kind) and (not step or n <= a+step):
            _, p, _ = self.device()
            target = p['launch_target_seconds'] or self.PREDICT_CAP*(p['launch_budget_seconds'] or 0) or float('inf')
            if self.predict_seconds(kind, cum[n]-cum[a]) <= min(target, self.PREDICT_CAP*(p['launch_budget_seconds'] or float('inf'))):
                return n
        return self.next_chunk_end(kind, a, n, step, cum, initial)

    def ksplit_cells(self, kind, raw, c0, c1, cell_part, geom, step, cum, report):
        """Cell-range launches of one split kernel inside fixed chunk [c0, c1) under the launch budget (budget_end)."""
        (nsplit, grid_y), block, shared = geom
        a = c0; launches = []
        while a < c1:
            b = self.budget_end(kind, a, c1, step, cum)
            seconds = self.timed_launch(kind, raw, ((b-a)*nsplit, grid_y), block, cell_part(a, b), shared, b-a, cum, a, b)
            report['gpu_seconds'] += seconds; report['max_seconds'] = max(report['max_seconds'], seconds)
            launches.append(((b-a)*nsplit, seconds)); a = b
        return launches

    def slim_proj_kernel(self, W, boundary):
        """sos_slim_proj from the phase-0 SLIM module (same source and NVRTC options as SLIM)."""
        self.raw_kernel(W, 0, 0, boundary, variant='slim')
        flags = tuple(sorted(set(self.config(W).defines) | set(KSPLIT_VARIANTS['slim'])))
        key = self.kernel_key(flags, W, self.tile_width(W, False, 0), 0, 0, boundary, stage1=self.stage1_flags())
        nkey = key + ('sos_slim_proj',)
        if nkey not in self.raw:
            code, options = self.raw_options[key]
            with stage('raw_compile'):
                raw = self.xp.RawKernel(code, 'sos_slim_proj', options=options, backend='nvrtc')
                raw.compile()
            self.raw[nkey] = raw
            self.attributes[f'sos_slim_proj:{W}_boundary{int(boundary)}'+self.attribute_suffix()] = dict(raw.attributes, block_size=SLIM_PROJ_THREADS)
        return self.raw[nkey]

    def slim_rays(self, kb, ctx, W, boundary, c0, c1, ray_rads, kind, report, variant='slimray', ranges=None):
        """KSPLIT=3 warm consumer: the chunk's rays (queue head/limit in ray_sync, chunk-local ray index) with SLIM's per-ray
        code (SLIM_RAYS: warm ray split, ray launches), each warp publishing its ray's radiance to ray_rads[ray-co[c0]][4W].
        Grid ~ rays/4, launches split by rays under the launch budget."""
        xp = self.xp; raw = self.raw_kernel(W, 0, 0, boundary, ray_limit=True, variant=variant)
        args = ctx['args']; li = ctx['lpt_index']; base = list(args[:li] if li is not None else args)
        base[0] = np.int32(c1-c0)
        si = ctx['sinv_index']; base[si:si+3] = self.ksplit_pointer_args(kb)
        for n, i in ctx['cell_index'].items():
            base[i] = args[i][c0:]
        base[ctx['mi']] = ctx['moments'][c0:c1]; base[ctx['fi']] = ctx['flux'][c0:c1]
        co = kb['co']; nray = int(co[c1]-co[c0]); ncel = c1-c0; sync = xp.zeros(ncel+2, 'i4')
        shared = self.shared_bytes(W, 0, variant=variant); cum = ctx['ray_cum']
        cumc = None if cum is None else cum[int(co[c0]-co[0]):int(co[c1]-co[0])+1]
        q0 = 0; launches = []; todo = list(ranges) if ranges is not None else None
        while q0 < nray:
            if todo is not None:   # G/L experiment: the G launch's ray ranges (post-launch budget check only)
                q0, q1 = todo.pop(0)
            else:
                q1 = self.budget_end(kind+'_slimray', q0, nray, None, cumc, initial=nray) if cumc is not None else nray
            sync[0:1].fill(q0); sync[ncel+1:ncel+2].fill(q1)
            nb = max(1, -(-(q1-q0)//(KSPLIT_BLOCK//32)))
            identify = variant == 'slimray' and os.environ.get('SOSFUSE2_P2MIN_NVTX', '0') == '1'
            if identify:
                print('P2MIN_SLIMRAY '+json.dumps(dict(kind=kind+'_'+variant, boundary=int(boundary),
                      cells=[c0, c1], rays=[q0, q1], grid=[nb, 1], block=[KSPLIT_BLOCK],
                      args=len(base)+2, shared=shared)), flush=True)
                xp.cuda.nvtx.RangePush('p2min_slimray')
            try:
                seconds = self.timed_launch(kind+'_'+variant, raw, (nb, 1), (KSPLIT_BLOCK,), tuple(base+[ray_rads, sync]), shared,
                                            q1-q0, cumc if todo is None else None, q0, q1)
            finally:
                if identify: xp.cuda.nvtx.RangePop()
            report['gpu_seconds'] += seconds; report['max_seconds'] = max(report['max_seconds'], seconds)
            launches.append((q0, q1, nb, seconds)); q0 = q1
        return launches

    def wf_dump(self, path, tag, kb, ctx, bundle, W, boundary, c0, c1, ray_rads, nray, r0, base, npt, slim_s, where):
        """exp/wavefront 2a: save one chunk's KSPLIT=3 slimray inputs and its ray radiances (ray_rads[(ray-r0)*4W + 4w+k]) so the
        streaming kernel can be run on real inputs and compared. Opt-in by SOSFUSE2_WF_DUMP=<dir>; production behaviour unchanged."""
        import pathlib
        xp = self.xp; H = self.host; args = ctx['args']; d = pathlib.Path(path); d.mkdir(parents=True, exist_ok=True)
        xp.cuda.Device().synchronize()   # the slimray launches above are asynchronous on their stream
        names = ('cells', 'nh', 'nc', 'nq', 'nsh', 'nsx', 'first', 'phase', 'boundary_flag', 'cell_off', 'starts', 'targets', 'dirs', 'weights',
                 'edge_off', 'edges', 'lower', 'end_j', 'end_f', 'end_col', 'hits', 'gu', 'gw', 'aa', 'bb', 'dd', 'h', 'chi', 'hr', 'ha', 'ext',
                 'asca', 'sun_h', 'sun_col', 'gm', 'goff', 'gdir', 'gval', 'height', 'g', 'mix', 'albedo')
        out = {}
        for n, a in zip(names, args[:len(names)]):
            if n in ('weights', 'aa', 'bb', 'dd', 'sun_h', 'sun_col', 'gm', 'goff', 'gdir', 'gval', 'asca', 'gw'): continue
            out[n] = np.asarray(H(a)) if hasattr(a, 'shape') else np.asarray(a)
        src = args[42]; out['src_rows4'] = np.asarray(H(src[:, :4, :]))     # [cell][k<4][W]: the only source rows SLIM reads
        out['boundary'] = np.asarray(H(args[43]))
        si = ctx['sinv_index']; stp = si+3+1                                 # SINV(3) + src_sh, then STRATO params
        for n, i in (('st_ext', stp), ('st_sca', stp+1), ('st_chi', stp+2), ('st_density', stp+3), ('st_end_col', stp+5)):
            out[n] = np.asarray(H(args[i]))
        AW = aero_w(W)
        out['aero'] = np.asarray(H(self.ksplit_aero[:npt*4*AW]))            # chunk-local [point][corner][AW], slot = point-base
        out['records'] = np.asarray(H(kb['buf'][16+SINV_F*base:16+SINV_F*(base+npt)]))   # this chunk's SINV_REC rows (7 floats/point); the whole bundle buffer is GBs
        out['hdr64'] = np.asarray(H(kb['hdr64'])); out['pcum'] = np.asarray(kb['pcum']); out['co'] = np.asarray(kb['co'])
        out['ray_rads'] = np.asarray(H(ray_rads[:nray*4*W])).reshape(nray, 4*W)
        out['meta'] = np.array(json.dumps(dict(where, c0=int(c0), c1=int(c1), r0=int(r0), nray=int(nray), base=int(base), npt=int(npt),
                                              W=int(W), AW=int(AW), boundary=bool(boundary), slim_gpu_seconds=float(slim_s),
                                              K=int(self.K), L=int(self.L), aero_rows_per_point=4)))
        np.savez(d/f'{tag}.npz', **out)
        raw = self.raw_kernel(W, 0, 0, boundary, ray_limit=True, variant='slimray')
        (d/'slimray_kernel.cu').write_text(raw.code); (d/'slimray_options.json').write_text(json.dumps(list(raw.options)))
        print('KSPLIT '+json.dumps(dict(event='wf_dump', path=str(d/f'{tag}.npz'), nray=int(nray), npt=int(npt), src_absmax=float(np.abs(out['src_rows4']).max()), rays_absmax=float(np.abs(out['ray_rads']).max()))), flush=True)

    def slim_proj(self, kb, ctx, bundle, W, boundary, c0, c1, y, ray_rads, kind, report, cum):
        """KSPLIT=3 projection of chunk cells [c0, c1): grid (cells, moment tiles of SLIM_PROJ_MOMENTS), cell launches split
        under the launch budget."""
        raw = self.slim_proj_kernel(W, boundary); args = ctx['args']; ci = ctx['cell_index']; co = kb['co']
        tiles = -(-(5+self.K)//SLIM_PROJ_MOMENTS); a = c0; launches = []
        while a < c1:
            b = self.budget_end(kind+'_proj', a, c1, None, cum)
            pargs = (np.int32(b-a), np.int32(KSPLIT_BLOCK//32), np.int64(co[c0]), args[ci['cell_off']][a:], args[ci['targets']][a:],
                     bundle.arrays['dirs'], bundle.arrays['weights'], bundle.arrays['edge_off'], y, ray_rads,
                     np.float64(ctx['albedo']), ctx['moments'][a:b], ctx['flux'][a:b])
            seconds = self.timed_launch(kind+'_proj', raw, (b-a, tiles), (SLIM_PROJ_THREADS,), pargs, 0, b-a, cum, a, b)
            report['gpu_seconds'] += seconds; report['max_seconds'] = max(report['max_seconds'], seconds)
            launches.append(((b-a)*tiles, seconds)); a = b
        return launches

    def vprod_launch(self, kb, bundle, s, dc, src_sh, W, phase, boundary, kind, raws, cell_part, geom, step, cum, report, ctx=None):
        """SOSFUSE2_KSPLIT=2 chunk loop. Per fixed chunk: plan (stored or rebuilt), y rows, VPROD over block ranges, SLIM over cell
        ranges, both under the launch budget. SOSFUSE2_KSPLIT_VERIFY=1: VPROD writes the sentinel-filled chunk, then per window of
        whole cells (<= VPROD_VERIFY_WINDOW bytes) its words go to a pinned host snapshot, the window is sentinel-filled again,
        PROD rewrites it for a tolerance comparison (E_KSPLIT_VERIFY); restore VPROD before SLIM reads it.
        Aero groups are complete target cells x wavelength, independent of chunk/window/piece sizes.
        SOSFUSE2_KSPLIT=3, phase 0: SLIM is replaced by the ray consumer (slim_rays -> chunk ray radiances) and the projection
        (slim_proj -> moments, flux); with VERIFY the per-cell SLIM runs first with a ray dump (SLIM_RAYDUMP), the consumer's ray
        radiances and projected moments/flux must pass tolerance against SLIM (E_KSPLIT_VERIFY)."""
        xp = self.xp; raw = dict(raws); ykern, vkern = self.vprod_kernels(W, phase, boundary)
        nh, nc = len(s.geo.h), len(s.geo.chi); aero = self.ksplit_aero; co = kb['co']; K = self.K
        AW = aero_w(W); self.vprod_fit(kb, bundle, aero.size//(4*AW), phase, nh, nc); pc = kb['pcum']
        kb['iteration'] += 1; it = kb['iteration']; rows = []; verify = self.ksplit_verify; used_max = pool_max = 0
        ray_buf = dump_buf = None
        projection_checks = []
        if verify:
            self._verify_tolerances = tolerances()
            wpts = max(1, VPROD_VERIFY_WINDOW//(4*AW*4))
            windows = {}
            for ci, (c0, c1) in enumerate(kb['chunks']):
                ws = []; wa = c0
                while wa < c1:
                    wb = min(c1, max(wa+1, int(np.searchsorted(pc, pc[wa]+wpts, side='right'))-1)); ws.append((wa, wb)); wa = wb
                windows[ci] = ws
            need = max(int(pc[wb]-pc[wa]) for ws in windows.values() for wa, wb in ws)*4*AW
            if getattr(self, '_vprod_snap', None) is None or self._vprod_snap.size < need:
                import cupyx
                self._vprod_snap = None; self._vprod_snap = cupyx.empty_pinned((need,), dtype=np.float32)
        for ci, (c0, c1) in enumerate(kb['chunks']):
            base = int(pc[c0]); npt = int(pc[c1])-base; nfl = npt*4*AW; r0, r1 = int(co[c0]), int(co[c1]); nray = r1-r0
            kb['hdr64'][2:3].fill(base)
            ev = {k: xp.cuda.Event() for k in ('p0', 'y0', 'p1', 'v1', 'w1', 'w2', 'c1', 'c2', 's1')}
            if verify:
                aero[:nfl].view(xp.uint32).fill(VPROD_SENTINEL)
            wall = time.perf_counter(); ev['p0'].record()
            c0cp = self.vprod_c0cp(kb, base, npt)
            dump = os.environ.get('SOSFUSE2_VPROD_DUMP', '').strip()
            if dump and phase == 0 and it == 1 and ci == 0:
                # Plan measurement input: this chunk's (c0, cp) records and ray offsets.
                np.savez(dump, c0cp=self.host(c0cp), eo=self.host(bundle.arrays['edge_off'][r0:r1+1]), base=base, r0=r0,
                         nq=int(bundle.nq), nh=nh, nc=nc)
            plan, stored = self.vprod_plan(kb, ci, bundle, c0cp, nh, nc)
            ev['y0'].record()
            y = xp.empty((max(nray, 1), K), 'f4')   # y rows of the chunk's rays (sh_basis_ds), every iteration
            if nray:
                ykern(((nray+VPROD_THREADS//32-1)//(VPROD_THREADS//32),), (VPROD_THREADS,),
                      (np.int32(nray), bundle.arrays['dirs'].reshape(-1)[3*r0:], dc['aa2'], dc['bb2'], dc['dd2'], y))
            ev['p1'].record(); wall = time.perf_counter()-wall
            free_now, total_dev = xp.cuda.runtime.memGetInfo(); used_max = max(used_max, int(total_dev)-int(free_now))
            pool_max = max(pool_max, int(xp.get_default_memory_pool().total_bytes()))
            def run_vprod(plan=plan, c0cp=c0cp, y=y):
                nblk = plan['nblk']; bc = plan['bcum'] if cum is not None else None; b0 = 0
                while b0 < nblk:
                    b1 = self.budget_end(kind+'_vprod', b0, nblk, None, bc, initial=nblk) if bc is not None else nblk
                    vargs = (plan['blk'][3*b0:], plan['vis_rbeg'], plan['vis_ray'], plan['slot'], c0cp, y, src_sh,
                             dc['hr'], dc['ha'], dc['strato_density'], np.int32(nh), np.int32(nc), aero)
                    seconds = self.timed_launch(kind+'_vprod', vkern, (b1-b0,), (VPROD_THREADS,), vargs, 0, b1-b0, bc, b0, b1)
                    report['gpu_seconds'] += seconds; report['max_seconds'] = max(report['max_seconds'], seconds)
                    b0 = b1
            run_vprod()
            ev['v1'].record(); prod = []
            if verify:
                valid_points = np.asarray(self.host(c0cp)).reshape(-1, 2)[:, 0] >= 0
                where = dict(phase=int(phase), iteration=it, chunk=ci, base=base, points=npt, cells=[c0, c1])
                for wa, wb in windows[ci]:
                    lo, hi = 4*AW*int(pc[wa]-base), 4*AW*int(pc[wb]-base)
                    aero[lo:hi].get(out=self._vprod_snap[:hi-lo])          # VPROD words (sentinel where nothing is written)
                    aero[lo:hi].view(xp.uint32).fill(VPROD_SENTINEL)
                    e0, e1 = xp.cuda.Event(), xp.cuda.Event(); e0.record()
                    self.ksplit_cells(kind+'_prod', raw['_prod'], wa, wb, cell_part, geom, step, cum, report)
                    e1.record(); prod.append((e0, e1))
                    self.vprod_verify(lo, hi, W, dict(where, window=[int(wa), int(wb)]), valid_points, pc)
                    aero[lo:hi].set(self._vprod_snap[:hi-lo])  # consumer must see the production VPROD values
                print('KSPLIT '+json.dumps(dict(event='verify', ok=True, words=nfl, windows=len(windows[ci]), **where)), flush=True)
            ev['w1'].record()
            mode3 = self.ksplit_mode == 3 and phase == 0
            if mode3:
                if ray_buf is None:
                    rmax = max(int(co[b]-co[a]) for a, b in kb['chunks'])
                    ray_buf = xp.empty(max(rmax, 1)*4*W, 'f4'); dump_buf = xp.empty(max(rmax, 1)*4*W, 'f4') if verify else None
                if verify:
                    ray_buf[:nray*4*W].view(xp.uint32).fill(VPROD_SENTINEL)
                    dump_buf[:nray*4*W].view(xp.uint32).fill(VPROD_SENTINEL)
                    ctx['moments'][c0:c1].fill(np.nan); ctx['flux'][c0:c1].fill(np.nan)
                    kb['dump'] = dump_buf; kb['hdr64'][5:6].fill(r0)
                    self.ksplit_cells(kind+'_slimdump', self.raw_kernel(W, 0, phase, boundary, variant='slimdump'), c0, c1, cell_part,
                                      geom, step, cum, report)
                    mref = ctx['moments'][c0:c1].copy(); fref = ctx['flux'][c0:c1].copy()
                ev['w2'].record()
                g0 = report['gpu_seconds']
                self.slim_rays(kb, ctx, W, boundary, c0, c1, ray_buf, kind, report)
                ev['c1'].record()
                if os.environ.get('SOSFUSE2_WF_DUMP') and phase == 0 and it in (1, 2) and ci == 0:
                    # exp/wavefront 2a: one chunk's slimray inputs and ray radiances
                    tag = f'chunk0_it{it}_{id(self) & 0xffff:04x}'
                    self.wf_dump(os.environ['SOSFUSE2_WF_DUMP'], tag, kb, ctx, bundle, W, boundary, c0, c1, ray_buf, nray, r0, base, npt,
                                 report['gpu_seconds']-g0, dict(phase=int(phase), iteration=it, chunk=ci, kind=kind))
                if verify:
                    self.slim2_verify_rays(ray_buf, dump_buf, nray*4*W, W, dict(phase=int(phase), iteration=it, chunk=ci, first_ray=r0))
                ev['c2'].record()
                if verify:
                    ctx['moments'][c0:c1].fill(np.nan); ctx['flux'][c0:c1].fill(np.nan)
                self.slim_proj(kb, ctx, bundle, W, boundary, c0, c1, y, ray_buf, kind, report, cum)
                ev['s1'].record()
                if verify:
                    projection_checks.append((c0, c1, *(self.host(x) for x in
                        (ctx['moments'][c0:c1], ctx['flux'][c0:c1], mref, fref))))
                    del mref, fref
            else:
                self.ksplit_cells(kind+'_slim', raw['_slim'], c0, c1, cell_part, geom, step, cum, report)
                ev['s1'].record()
            rows.append((ci, c0, c1, npt, {k: plan[k] for k in ('nray', 'nrec', 'nvis', 'nblk')}, stored, wall, ev, prod, mode3))
            report['chunks'] += 1; report['max_cells'] = max(report['max_cells'], c1-c0)
            del plan, c0cp, y, run_vprod   # stream-ordered pool reuse; a kept plan would hold ~4 B/record per chunk
        if projection_checks:
            self.slim2_verify_proj(projection_checks, dict(phase=int(phase), iteration=it))
        kb.pop('dump', None)
        if rows:
            rows[-1][7]['s1'].synchronize()
        tot = dict(plan_ms=0., y_ms=0., vprod_ms=0., slim_ms=0., cons_ms=0., proj_ms=0., prod_ms=0., plan_wall_s=0.)
        for ci, c0, c1, npt, plan, stored, wall, ev, prod, m3 in rows:
            ms = lambda a, b: float(xp.cuda.get_elapsed_time(ev[a], ev[b]))
            row = dict(event='vprod_chunk', phase=int(phase), iteration=it, chunk=ci, cells=[int(c0), int(c1)], points=npt,
                       rays=plan['nray'], records=plan['nrec'], visitors=plan['nvis'], blocks=plan['nblk'], stored=bool(stored),
                       plan_ms=ms('p0', 'y0'), y_ms=ms('y0', 'p1'), plan_wall_s=wall, vprod_ms=ms('p1', 'v1'), slim_ms=ms('w1', 's1'))
            if verify:
                row['prod_ms'] = sum(float(xp.cuda.get_elapsed_time(e0, e1)) for e0, e1 in prod)
            if m3:   # consumer + projection (the VERIFY ray dump SLIM and the comparisons are outside both)
                row.update(cons_ms=ms('w2', 'c1'), proj_ms=ms('c2', 's1')); row['slim_ms'] = row['cons_ms']+row['proj_ms']
            for k in tot:
                tot[k] += row.get(k, 0.)
            print('KSPLIT '+json.dumps(row), flush=True)
        print('KSPLIT '+json.dumps(dict(event='vprod_launch', phase=int(phase), iteration=it, chunks=len(rows), verify=bool(verify),
                                       resident=kb['resident'], store=bool(kb['store']), stored_plans=len(kb['plans']),
                                       stored_bytes=int(kb['stored_bytes']), aero_bytes=int(aero.nbytes),
                                       device_used_bytes_sampled=used_max, pool_total_bytes_sampled=pool_max, **tot)), flush=True)

    def verify_result(self, event, result, where):
        info = dict(where, **log_summary(result))
        print('KSPLIT '+json.dumps(dict(event=event, ok=result['pass_'], **info)), flush=True)
        if not result['pass_']:
            raise RuntimeError('E_KSPLIT_VERIFY: '+event+': '+json.dumps(info))

    def vprod_verify(self, lo, hi, W, where, valid_points, pc, aw=None):
        """Aero scale: all valid point/corner records of one target cell x wavelength. aw = record row stride (aero_w(W); the
        pad lanes >= W are dropped before comparing).
        Windows/chunks contain whole target cells; stage pieces cannot change S_g.
        c0cp metadata, not observed writes, defines the consumed slots.
        """
        r, alpha = self._verify_tolerances
        wa, wb = where['window']; base = where['base']
        p0, p1 = int(pc[wa])-base, int(pc[wb])-base
        aw = aw or aero_w(W); start, end = p0*4*aw, p1*4*aw
        actual = np.ascontiguousarray(self._vprod_snap[start-lo:end-lo].reshape(-1, 4, aw)[:, :, :W])
        ref = np.ascontiguousarray(self.host(self.ksplit_aero[start:end]).reshape(-1, 4, aw)[:, :, :W])
        # Metadata is generated only for each arithmetic piece, never for all
        # point/corner/wavelength slots in the window.
        def valid(i, j):
            return valid_points[p0+np.arange(i, j)//(4*W)]
        def groups(i, j):
            slots = np.arange(i, j)
            cells = np.searchsorted(pc, int(pc[wa])+slots//(4*W), side='right')-1
            return cells*W + slots % W
        result = compare_tol(actual, ref, valid, groups, r, alpha, details=False,
                             stage_size=VPROD_VERIFY_STAGE)
        info = dict(where, cells=wb-wa)
        if result.get('peak') is not None:
            point, corner, wave = result['peak']
            info.update(max_cell=int(np.searchsorted(pc, int(pc[wa])+point, side='right')-1), max_wavelength=wave)
        if result['first'] is not None:
            point, corner, wave = result['first']
            info.update(cell=int(np.searchsorted(pc, int(pc[wa])+point, side='right')-1),
                        point=int(pc[wa])+point, corner=corner, wavelength=wave)
        self.verify_result('verify_aero', result, info)

    def slim2_verify_rays(self, rays, dump, n, W, where):
        """Ray x wavelength groups include all four components, independent of launch chunks."""
        a = self.host(rays[:n]).reshape(-1, W, 4); b = self.host(dump[:n]).reshape(-1, W, 4)
        groups = (np.arange(a.shape[0])[:, None, None]+where['first_ray'])*W + np.arange(W)[None, :, None]
        result = compare_tol(a, b, True, groups, *self._verify_tolerances)
        if result['first'] is not None:
            ray, wave, comp = result['first']
            where = dict(where, ray=where['first_ray']+ray, wavelength=wave, component=comp)
        self.verify_result('verify_rays', result, where)

    def slim2_verify_proj(self, chunks, where):
        """Moment x wavelength (and separately flux x wavelength), across ALL chunk cells.
        Retain host projection snapshots until all chunks finish so S_g is global.
        """
        for name, ai, bi in (('moments', 2, 4), ('flux', 3, 5)):
            pieces = []
            for chunk in chunks:
                a, b = chunk[ai], chunk[bi]
                groups = np.arange(np.prod(b.shape[1:])).reshape((1, *b.shape[1:]))
                pieces.append((a, b, True, groups))
            result = compare_chunks(pieces, *self._verify_tolerances, details=False, stage_size=VPROD_VERIFY_STAGE)
            info = dict(where, quantity=name)
            if result['first'] is not None:
                index = result['first'][0]
                for chunk in chunks:
                    b = chunk[bi]
                    if index < b.size:
                        pos = np.unravel_index(index, b.shape)
                        info.update(chunk_cells=[chunk[0], chunk[1]], cell=chunk[0]+int(pos[0]),
                                    slot=[int(x) for x in pos[1:]])
                        break
                    index -= b.size
            self.verify_result('verify_proj', result, info)


    def launch(self, bundle, s, field, first=False, phase=0, boundary=None, timed=False, _ksplit=None):
        section_dir = None
        if self.config().instrumented:
            text = os.environ.get('SOSFUSE2_SECTION_CLOCKS_DIR', '').strip()
            if not text:
                raise ValueError('E_SECTION_CLOCK_OUTPUT: set SOSFUSE2_SECTION_CLOCKS_DIR')
            section_dir = Path(text).resolve()
            section_dir.mkdir(parents=True, exist_ok=True)
        xp=self.xp; c=self.setup(s, bundle.nq); dc=self.constants_device(s, bundle.nq)
        W=s.W; ncell=bundle.cells; nr=bundle.rays
        ks = self.ksplit and not first and phase in (0, 1)
        if ks:
            if W != 15 or 'strato_ext' not in c:
                raise ValueError('E_KSPLIT_STATE: SOSFUSE2_KSPLIT=1 needs 15 wavelengths and an enabled stratosphere '
                                 '(five separable columns plus the strato profile, kernel.cu SINV)')
            kb = bundle.ksplit if _ksplit else self.ksplit_columns(bundle, s, field, phase, boundary)
            if _ksplit == 'record':
                # Unit-basis extinction: five profile columns, then the separate stratosphere column.
                e = xp.zeros(dc['ext'].size, dc['ext'].dtype)
                for k in range(NCOMP): e[k*W+k] = 1
                se = xp.zeros(dc['strato_ext'].size, dc['strato_ext'].dtype); se[NCOMP] = 1
                dc = dict(dc, ext=e.reshape(dc['ext'].shape), strato_ext=se.reshape(dc['strato_ext'].shape))
            else:
                kb['hdr32'][0:1].fill(2)
        src=xp.ascontiguousarray(field, dtype='f4')
        moments=xp.empty((ncell,4+self.K,W), 'f8') if phase==0 else xp.empty(1, 'f8')
        flux=xp.empty((ncell,W), 'f8')
        query=xp.empty((nr,W,4), 'f4') if phase==2 else (kb['buf'] if ks else xp.empty(1, 'f4'))
        bd=xp.ascontiguousarray(boundary, dtype='f8') if boundary is not None else xp.zeros((len(s.geo.chi),W), 'f8')
        names=('cell_off','starts','targets','dirs','weights','edge_off','edges','lower','end_j','end_f','end_col','hits')
        args=[np.int32(v) for v in (ncell,len(s.geo.h),len(s.geo.chi),bundle.nq,len(c['sun_h']),c['sun_col'].shape[1],int(first),phase,int(boundary is not None))]
        args += [bundle.arrays[name] for name in names]
        if 'density_optics' in c:args[9+names.index('end_col')]=self.density_end_columns(bundle,c)
        args += [dc[name] for name in ('u','wu','aa','bb','dd','h','chi','hr','ha','ext','asca','sun_h','sun_col','gm','goff','gdir','gval')]
        ground_sh_l = self.ground_sh_l if first and phase in (0, 2) and 'ground_sh' in dc else 0
        if ground_sh_l:
            # Only the SH specialization interprets gval as its synthesized
            # convolution table. Empty A0 uses the
            # original binary, argument arrays, and shared-memory size.
            args[-1] = dc['ground_sh']
        args += [c['height'],c['g'],c['mix'],np.float64(c['albedo']),src,bd,moments,flux,query]
        sinv_index = len(args) if ks else None
        if ks: args += self.ksplit_pointer_args(kb)
        sh_index = len(args)   # the SH sidecar appended below (VPROD reads the same combined plane)
        with stage('sh_aux_build',device=True):
            if 'strato_ext' in c and self.config().strato:
                if src.shape != (len(s.geo.h)*len(s.geo.chi), 10+2*self.K, W):
                    raise ValueError('E_STRATO_SH_SOURCE: expected [nh*nc,1132,W]')
                if self.config().c3_fw15 and not first:
                    args.append(combined_aux(src,c,self.L,xp,skip_thin=True))
                else:
                    from .strato_parallel import strato_sh_aux
                    args.append(strato_sh_aux(src, xp, pair=True, L=self.L))
            else:
                args.append(sh_aux_pair(src,xp,self.L))
        if 'strato_ext' in c:
            # No optical sidecar is persisted in legacy/common endpoint files.
            endst=self.strato_end_columns(bundle,c)
            profile=c['strato_profile']
            args += [dc[name] for name in ('strato_ext','strato_sca','strato_chi','strato_density','strato_sun_col')]
            args += [endst,F(profile['center']),F(profile['sigma']),F(profile['norm'])]
        self.check_upload(c, dc, 'launch')   # N and upload identity (E_PROFILE_N_MISMATCH / E_PROFILE_STALE)
        args += [dc.get('profile_table', np.uint64(0)), np.int32(c.get('density_optics', {}).get('mask', 0)), F(c.get('density_optics', {}).get('air_surface', 1.))]
        lpt_index = None; split_count = None
        if not first and not (self.config().split_warm and phase == 0):
            # R4 LPT_ORDER: per-launch cell order by descending segment count (block work proxy).
            S = 2 if self.config().c3_fw15 else 1
            def lpt_order(c0, c1):
                co = bundle.arrays['cell_off'][c0:c1+1]; eo = bundle.arrays['edge_off']
                work = xp.diff(eo[co]) - xp.diff(co)
                if S > 1:  # units = cell*S+part, parts of a cell adjacent, cells by descending work
                    o = xp.argsort(-work).astype('i4')
                    return xp.ascontiguousarray((o[:, None]*S + xp.arange(S, dtype='i4')[None, :]).reshape(-1))
                return xp.ascontiguousarray(xp.argsort(-work).astype('i4'))
            lpt_index = len(args); args.append(lpt_order(0, ncell))
            if S > 1:
                args.append(xp.empty(max(1, (S-1)*ncell*(4+self.K)*W) if phase == 0 else 1, 'f8'))
                args.append(xp.empty(S*ncell*W, 'f8'))
                split_count = xp.zeros(ncell, 'i4'); args.append(split_count)
        section_buffer = None
        if section_dir is not None:
            section_buffer = xp.zeros((ncell, W // self.tile_width(W, first, phase),
                                       self.bs(first) // 32, len(SECTION_CLOCK_NAMES)),
                                      dtype='uint64')
            args.append(section_buffer)
        mode, step = self.launch_mode(ncell)
        if ks and mode == 'single': mode, step = 'fixed', ncell     # KSPLIT always goes through the chunk loop
        ray_sync = None; ray_mode = False
        if phase==0 and ((self.config().split_first and first) or (self.config().split_warm and not first)):
            # FL_RAYSPLIT private scratch: per-ray rad rows and [ray queue, per-cell done] counters (zeroed).
            # Cell-subset launches (eda): the kernel reads rays cell_off[0]..cell_off[cells] and indexes ray_rads
            # and the counters chunk-locally, so the counters are re-zeroed before every chunk.
            # WARM_RAYSPLIT is not ported to cell-subset launches.
            if mode != 'single' and (not first):
                raise ValueError('E_FL_RAYSPLIT_LAUNCH_CELLS: WARM_RAYSPLIT needs one launch per cell grid')
            # Split FL_RAYSPLIT first light goes by ray ranges over the whole cell grid (eda: 68 cells of first light
            # took 6.3 s on 2026-10-07, and fewer cells leave SMs idle because a block is a cell).
            # SOSFUSE2_FL_PROJECT=1 also runs one-launch first light as ray launch + fl_project (rig experiment: the
            # single-warp projections may dominate its 11.9 s first light on a 4090; not measured yet).
            ray_mode = mode != 'single' or os.environ.get('SOSFUSE2_FL_PROJECT', '').strip() == '1'
            ray_sync = xp.zeros(ncell+2,'i4')  # [queue head, cell counters, ray-range end (LAUNCH_RAY_LIMIT)]
            ray_rads = xp.empty((max(nr,1),4*W),'f4')
            args += [ray_rads, ray_sync]
        rs = bool(self.device()[1].get('requires_split'))
        if first and phase == 0 and (not ray_mode):
            err = first_light_split_error(self.flags, 'strato_ext' in c, self.device()[0], self.device()[1])
            if err: raise RuntimeError(err)
        variant = _ksplit or ('prod' if ks else None)
        raws = [('_prod', self.raw_kernel(W,first,phase,boundary is not None,variant='prod')),
                ('_slim', self.raw_kernel(W,first,phase,boundary is not None,variant='slim'))] if ks and not _ksplit else \
               [('', self.raw_kernel(W,first,phase,boundary is not None,ray_limit=ray_mode,variant=_ksplit,ground_sh_l=ground_sh_l))]
        raw = raws[0][1]
        if timed or section_buffer is not None:
            begin=xp.cuda.Event();end=xp.cuda.Event();begin.record()
        name='launch_first' if first else ('launch_query' if phase==2 else 'launch_ground' if phase==1 else 'launch_warm')
        with stage(name,device=True):
            # Cell-subset launches (eda thermal/watchdog cap; device profile, device_profile.py): blocks are
            # independent, so only the cell-indexed pointers move: cell_off/starts/targets (inputs) and
            # moments/flux (outputs); per-chunk counters (FL_RAYSPLIT ray_sync, CELL_SPLIT split_count) are
            # re-zeroed. Numerically equivalent to one launch within gate tolerance. 'adaptive' sizes each chunk from the last
            # measured seconds per unit of work of the same kernel kind; 'fixed' is SOSFUSE2_LAUNCH_CELLS.
            nsplit = (2 if self.config().c3_fw15 else 1) if lpt_index is not None else 1
            grid_y = W//self.tile_width(W, first, phase); block = (self.bs(first, variant),)
            shared = self.shared_bytes(W, phase, first=first, variant=variant)
            kind = f'first{int(first)}_phase{int(phase)}_W{W}' + ('_record' if _ksplit else '_split' if ks else '')
            report = None
            if mode == 'single' and not ray_mode:
                seconds = self.timed_launch(kind, raw, (ncell*nsplit, grid_y), block, tuple(args), shared, ncell)
                report = dict(chunks=1, gpu_seconds=seconds, max_seconds=seconds, max_cells=ncell)
            elif ray_mode:
                cum = self.ray_work(bundle); nrays = len(cum)-1; kind += '_rays'
                # Ray ranges start (and are floored) at half the rays of the initial cells: first-light rays cost about
                # 27 us each on the 3080 next to another GPU job (eda r4), so 24 cells' rays would be about 4 s.
                initial = max(1, (step if mode == 'fixed' else self.device()[1]['launch_cells_initial'] or 1)*nrays//max(ncell, 1)//(1 if mode == 'fixed' else 2))
                report = dict(chunks=0, gpu_seconds=0.0, max_seconds=0.0, max_cells=ncell, max_rays=0)
                q0 = 0
                while q0 < nrays:
                    q1 = nrays if mode == 'single' else min(nrays, q0+initial) if (mode == 'fixed' and not rs) else \
                        self.next_chunk_end(kind, q0, nrays, initial if mode == 'fixed' else None, cum, initial)
                    ray_sync[0:1].fill(q0); ray_sync[ncell+1:ncell+2].fill(q1)
                    seconds = self.timed_launch(kind, raw, (ncell, grid_y), block, tuple(args), shared, q1-q0, cum, q0, q1)
                    report['chunks'] += 1; report['gpu_seconds'] += seconds
                    report['max_seconds'] = max(report['max_seconds'], seconds); report['max_rays'] = max(report['max_rays'], q1-q0)
                    q0 = q1
                # With LAUNCH_RAY_LIMIT the ray launches only publish ray_rads; project the cells now (fl_project,
                # one block per cell, FP32 ray batches combined in FP64).
                proj = self.fl_project_kernel(W, boundary is not None, ground_sh_l=ground_sh_l); pkind = kind[:-5]+'_project'
                co_h = np.asarray(self.host(bundle.arrays['cell_off']), 'i8'); pcum = (co_h-co_h[0]).astype('f8')
                ci = 9+names.index('cell_off'); ti = 9+names.index('targets')
                fixed_args = (args[9+names.index('dirs')], args[9+names.index('weights')], dc['aa'], dc['bb'], dc['dd'], ray_rads,
                              np.float64(c['albedo']))
                c0 = 0
                while c0 < ncell:
                    c1 = ncell if mode == 'single' else min(ncell, c0+step) if (mode == 'fixed' and not rs) else \
                        self.next_chunk_end(pkind, c0, ncell, step if mode == 'fixed' else None, pcum)
                    seconds = self.timed_launch(pkind, proj, (c1-c0,), (256,),
                                                (np.int32(c1-c0), np.int64(co_h[0]), args[ci][c0:], args[ti][c0:])+fixed_args+
                                                (moments[c0:c1], flux[c0:c1]), 0, c1-c0, pcum, c0, c1)
                    report['chunks'] += 1; report['gpu_seconds'] += seconds; report['max_seconds'] = max(report['max_seconds'], seconds)
                    c0 = c1
            else:
                cum = self.cell_work(bundle) if (mode == 'adaptive' or rs) else None
                cell_index={n:9+names.index(n) for n in ('cell_off','starts','targets')}
                mi=next(i for i,a in enumerate(args) if a is moments);fi=next(i for i,a in enumerate(args) if a is flux)
                report = dict(chunks=0, gpu_seconds=0.0, max_seconds=0.0, max_cells=0)
                c0 = ncell if (ks and not _ksplit and self.ksplit_mode in (2, 3)) else 0
                if c0:
                    def cell_part(a, b):
                        part=list(args);part[0]=np.int32(b-a)
                        part[sinv_index:sinv_index+3] = self.ksplit_pointer_args(kb)
                        for n,i in cell_index.items():part[i]=args[i][a:]
                        if phase==0:part[mi]=moments[a:b]
                        part[fi]=flux[a:b]
                        if lpt_index is not None:part[lpt_index]=lpt_order(a, b)
                        if split_count is not None:split_count.fill(0)
                        return tuple(part)
                    ctx = dict(args=args, sinv_index=sinv_index, lpt_index=lpt_index, cell_index=cell_index, mi=mi, fi=fi, moments=moments, flux=flux,
                               albedo=c['albedo'], ray_cum=self.ray_work(bundle) if cum is not None else None)
                    self.vprod_launch(kb, bundle, s, dc, args[sh_index], W, phase, boundary is not None, kind, raws, cell_part,
                                      ((nsplit, grid_y), block, shared), step, cum, report, ctx)
                while c0 < ncell:
                    c1=self.next_chunk_end(kind+raws[0][0], c0, ncell, step, cum)
                    if ks and not _ksplit: c1=self.ksplit_chunk(kb, c0, c1)
                    part=list(args);part[0]=np.int32(c1-c0)
                    if ks: part[sinv_index:sinv_index+3] = self.ksplit_pointer_args(kb)
                    for n,i in cell_index.items():part[i]=args[i][c0:]
                    if phase==0:part[mi]=moments[c0:c1]
                    part[fi]=flux[c0:c1]
                    if section_buffer is not None:part[-1]=section_buffer[c0:c1]
                    if lpt_index is not None:part[lpt_index]=lpt_order(c0, c1)
                    if ray_sync is not None and c0 > 0:ray_sync.fill(0)
                    if split_count is not None and c0 > 0:split_count.fill(0)
                    for suffix, r in raws:   # KSPLIT: producer then consumer on the same chunk (one stream, in order)
                        seconds = self.timed_launch(kind+suffix, r, ((c1-c0)*nsplit, grid_y), block, tuple(part), shared,
                                                    c1-c0, cum, c0, c1)
                        report['gpu_seconds'] += seconds; report['max_seconds'] = max(report['max_seconds'], seconds)
                    report['chunks'] += 1; report['max_cells'] = max(report['max_cells'], c1-c0)
                    c0 = c1
            if report is not None:
                report.update(kind=kind, mode=mode, cells=ncell, profile=self.launch_log['profile'],
                              budget_seconds=self.launch_log['budget_seconds'])
                print('LAUNCH_REPORT '+json.dumps(report, allow_nan=False), flush=True)
        if timed or section_buffer is not None:
            end.record();end.synchronize()
            cuda_seconds = float(xp.cuda.get_elapsed_time(begin,end)/1000)
            if timed:self.cuda_events.append(cuda_seconds)
        if section_buffer is not None:
            counters = self.host(section_buffer)
            report = section_clock_summary(counters)
            path = section_dir / f'clock_{os.getpid()}_{time.time_ns()}_{self.metrics["launches"]}.npz'
            with path.open('xb') as stream:
                np.savez_compressed(stream, cycles=counters)
            report.update(schema='sosfuse2-section-clocks-v1', flags=list(self.flags),
                          gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), pid=os.getpid(),
                          launch=self.metrics['launches'], first=bool(first), phase=int(phase),
                          wavelengths=W, block_size=self.block_size,
                          grid=[ncell, W // self.tile_width(W, first, phase)],
                          launch_cells=self.launch_cells(), cuda_event_seconds=cuda_seconds,
                          raw_path=str(path))
            print('SECTION_CLOCKS ' + json.dumps(report, allow_nan=False), flush=True)
        self.metrics['launches']+=1
        if phase==2:self.metrics['query_launches']+=1
        self.memory()
        return moments,flux,query

    def transport(self, geometry, s, field, first=False):
        """Accepted query adapter, but with regenerated geometry, not raw points."""
        if 'edges' not in geometry:
            raise ValueError('E_SAMPLE_CACHE: fused path requires starts/directions/segments, not legacy point arrays')
        c=self.setup(s,geometry['nq'])
        packed = np.ascontiguousarray(self.host(field)[..., ::2], 'f4') if np.size(field) else np.zeros((s.geo.ns,source_size(s),s.W), 'f4')
        if self.xp is np:
            value=reference.transport(geometry,c,packed,first,statistics=self.metrics)
        else:
            from .geometry import Bundle
            bundle=Bundle([geometry],self.xp)
            _,_,value=self.launch(bundle,s,self.xp.asarray(packed),first,2)
            value=self.host(value)
        out=np.zeros((len(value),s.W,2,4), 'f8');out[:,:,0]=value
        return out


class Solver:
    def __init__(self, accepted, geometry, fixed_boxes=False, box_group=32, prepared=False, observer=None, box_evaluate=None):
        from .stratosphere import attach
        if not prepared:
            attach(accepted,accepted.query.__func__.__globals__)
        self.s=accepted;self.geometry=geometry;self.kernel=accepted.kernel;self.xp=self.kernel.xp
        self.first_moments=None;self.first_flux=None
        self.observer=observer;self.box_evaluate=box_evaluate
        # A5: opt-in observer path (fixed captured rays, device source read directly).
        self.fixed_boxes=fixed_boxes;self.box_group=box_group;self._box_rays=None
        self.timings=dict(apply_seconds=[], gpu_kernel_seconds=[], setup_seconds=0.)

    def initialize(self):
        if self.first_moments is not None:return
        tt=time.perf_counter();s=self.s;xp=self.xp
        with stage('first_light_total'):
            zero=xp.zeros((s.geo.ns,source_size(s),s.W), 'f4')
            first_moments,first_flux=self.geometry.evaluate(s,zero,first=True,phase=0)
            self.kernel.sync()
        del zero
        if xp is not np:
            if self.geometry.bundle is None:
                if hasattr(self.geometry, 'prepare'):
                    self.geometry.prepare()
                else:
                    from .geometry import Bundle
                    self.geometry.bundle=Bundle([self.geometry.cell(i) for i in range(s.geo.ns)],xp)
            if self.geometry.ground_bundle is None:
                from .geometry import Bundle
                self.geometry.ground_bundle=Bundle([self.geometry.cell(i) for i in range(len(s.geo.chi))],xp)
            bundles=[(self.geometry.ground_bundle,1),(self.geometry.bundle,0)]
            from .pipeline_execution import prepare_bundle
            for bundle,_ in bundles:
                prepare_bundle(self.kernel,s,bundle);self.kernel.constants_device(s,bundle.nq)
            self.timings['ksplit']=self.kernel.ksplit_preflight(s,bundles)
        self.first_moments,self.first_flux=first_moments,first_flux
        self.timings['setup_seconds']=time.perf_counter()-tt

    def apply(self, field):
        self.initialize();tt=time.perf_counter();s=self.s;xp=self.xp
        with stage('apply_ground'):
            _,flux=self.geometry.evaluate(s,field,phase=1,ground_only=True)
        boundary=flux+self.first_flux[:len(s.geo.chi)]
        with stage('apply_warm'):
            moment,_=self.geometry.evaluate(s,field,phase=0,boundary=boundary)
        moment += self.first_moments
        if os.environ.get('SOSFUSE2_EXPORT_MOMENT')=='1':self.last_moment=moment   # opt-in (2026-10-28, saddle control variate): incident-radiance SH moments of this evaluation
        out=xp.empty_like(field)
        out[:,:4]=(moment[:,:4]*xp.asarray(F(3/(8*math.pi))*s.ext[0].astype('f4'))[None,None,:]).astype('f4')
        out[:,4:10]=(moment[:,4:10]*xp.asarray(s.rf.astype('f4'))[None,:,None]*xp.asarray(s.ext[0].astype('f4'))[None,None,:]).astype('f4')
        if getattr(s,'tropo',None) is not None:
            from .tropo_mie import eigen_by_sh as tropo_eigen
            out[:,10:10+self.kernel.K]=(moment[:,4:]*xp.asarray(tropo_eigen(s.tropo,self.kernel.L).astype('f4'))[None,:,:]*xp.asarray(s.asca.astype('f4'))[None,None,:]).astype('f4')
        else:
            out[:,10:10+self.kernel.K]=(moment[:,4:]*xp.asarray(s.af.astype('f4'))[None,:,None]*xp.asarray(s.asca.astype('f4'))[None,None,:]).astype('f4')
        if getattr(s,'strato',None) is not None:
            from .stratosphere import eigen_by_sh
            out[:,10+self.kernel.K:]=(moment[:,4:]*xp.asarray(eigen_by_sh(s.strato,self.kernel.L).astype('f4'))[None,:,:]*xp.asarray(s.strato['sca'].astype('f4'))[None,None,:]).astype('f4')
        if xp is np:out[np.abs(out)<reference.TRACE_MIN]=0   # FP32 flush-to-zero of the device composition (no subnormal source on CUDA, reference.flush_to_zero)
        self.kernel.sync();self.kernel.memory();self.timings['apply_seconds'].append(time.perf_counter()-tt)
        return out

    def boxes(self, field):
        with stage('observer'):
            full=np.zeros((self.s.geo.ns,field.shape[1],2*self.s.W), 'f4');full[...,::2]=self.kernel.host(field)
            return self.s.boxes(full)[...,0].reshape(5,7,self.s.W)

    def boxes_fixed(self, field):
        """A5: same 35 boxes as boxes(); rays are captured once from the accepted
        s.boxes, packed into one fixed Bundle, and every call launches phase 2 on
        the device field (no host round trip of the source), then replays the
        accepted reshape/quadrature/solar factor with the returned ray values."""
        from .table_fast import _bundle, _evaluate
        s=self.s;kernel=self.kernel;dummy=np.empty(0,'f4')
        with stage('observer_fixed'):
            if self._box_rays is None:
                raws=[]
                def record(raw,solver,_field,first=False):
                    raws.append(raw);return np.zeros((len(raw['d']),solver.W,2,4),'f8')
                kernel.transport=record
                try:s.boxes(dummy,False)
                finally:del kernel.transport
                self._box_rays=(raws,_bundle(s,raws,self.box_group))
            raws,bundle=self._box_rays
            values=(self.box_evaluate or _evaluate)(s,raws,field,False,self.box_group,bundle)
            index=iter(range(len(raws)));pack=s.geo.pack
            def token(p,d,n):
                i=next(index)
                if len(d)!=len(raws[i]['d']) or not np.array_equal(d,raws[i]['d']):raise RuntimeError('E_REPLAY_DRIFT: observer ray order changed')
                return i
            def fetch(i,solver,_field,first=False):
                if first:raise RuntimeError('E_REPLAY_FLAG')
                out=np.zeros((len(values[i]),solver.W,2,4),'f8');out[:,:,0]=values[i];return out
            s.geo.pack=token;kernel.transport=fetch
            try:result=s.boxes(dummy,False)
            finally:
                s.geo.pack=pack;del kernel.transport
            if next(index,None) is not None:raise RuntimeError('E_REPLAY_COUNT')
            return result[...,0].reshape(5,7,s.W)

    def solve(self, max_eval=16, tolerance=.005):
        """Same Anderson4 normalization, regularization and candidate guard."""
        s=self.s;xp=self.xp;x=xp.zeros((s.geo.ns,source_size(s),s.W), 'f4')
        gs=[[] for _ in range(s.W)];fs=[[] for _ in range(s.W)];done=np.zeros(s.W,bool);history=[]
        started=time.perf_counter();boxes=self.observer or (self.boxes_fixed if self.fixed_boxes else self.boxes)
        for iteration in range(1,max_eval+1):
            evaluation_start=time.perf_counter();old=boxes(x);old_query_seconds=time.perf_counter()-evaluation_start
            apply_start=time.perf_counter();g=self.apply(x);apply_seconds=time.perf_counter()-apply_start
            # A NaN trace takes the kernel's linear path and propagates instead of vanishing: count and stop.
            nonfinite=int(xp.count_nonzero(~xp.isfinite(g)))
            if nonfinite:raise RuntimeError(f'E_NONFINITE_SOURCE: {nonfinite} values after evaluation {iteration}')
            g=xp.where(xp.asarray(done)[None,None,:],x,g);query_start=time.perf_counter();new=boxes(g)
            query_seconds=old_query_seconds+time.perf_counter()-query_start
            rr=np.max(abs(new-old)/np.maximum(abs(new),1e-30),axis=(0,1))
            ss=self.kernel.host(xp.max(abs(g-x),axis=(0,1))/xp.maximum(xp.max(abs(g),axis=(0,1)),1e-30))
            if iteration>=2:done |= np.maximum(rr,ss)<=tolerance
            history.append(dict(evaluation=iteration,observer_relative=rr.tolist(),source_relative=ss.tolist(),converged=done.tolist(),
                                apply_seconds=apply_seconds,observer_seconds=query_seconds,seconds=time.perf_counter()-started))
            print(f'SOSFUSE EVAL {iteration} obs={max(rr):.6g} source={max(ss):.6g} done={done.sum()}/{s.W}',flush=True)
            if done.all():
                x=g;history[-1]['evaluation_seconds']=time.perf_counter()-evaluation_start;history[-1]['Anderson_seconds']=0.;break
            anderson_start=time.perf_counter()
            nxt=g.copy()  # A1: Anderson time recorded below as stage 'anderson'
            for w in range(s.W):
                if done[w]:continue
                norm=xp.linalg.norm(g[:,:,w],axis=1);scale=xp.maximum(norm,norm.max()*1e-8)
                feature=((g[:,:,w]-x[:,:,w])/scale[:,None]).astype('f8').ravel()
                obs=xp.asarray(((new[:,:,w]-old[:,:,w])/np.maximum(abs(new[:,:,w]),1e-20)).ravel()*math.sqrt(len(feature)/35))
                gs[w].append(g[:,:,w].copy());fs[w].append(xp.concatenate([feature,obs]))
                if len(fs[w])>4:gs[w].pop(0);fs[w].pop(0)
                if len(fs[w])>1:
                    f=xp.stack(fs[w],1);gram=f.T@f;gram+=xp.eye(len(gram))*max(float(self.kernel.host(xp.trace(gram)))*1e-10,1e-30)
                    self.kernel.memory()
                    try:
                        alpha=xp.linalg.solve(gram,xp.ones(len(gram)));alpha/=alpha.sum();ah=self.kernel.host(alpha)
                        if np.isfinite(ah).all() and max(abs(ah))<8:nxt[:,:,w]=sum(c*q for c,q in zip(alpha,gs[w]))
                    except np.linalg.LinAlgError:
                        pass
            # A converged wavelength is frozen exactly as the accepted active mask.
            x=xp.where(xp.asarray(done)[None,None,:],g,nxt)
            self.kernel.sync();history[-1]['Anderson_seconds']=time.perf_counter()-anderson_start;TIMERS.add('anderson',history[-1]['Anderson_seconds'])
            history[-1]['evaluation_seconds']=time.perf_counter()-evaluation_start
        self.kernel.sync()
        if not done.all():raise RuntimeError(f'E_NOT_CONVERGED: {done.sum()}/{s.W} within {tolerance}')
        return x,dict(seconds=time.perf_counter()-started,evaluations=iteration,tolerance=tolerance,converged=done.tolist(),history=history)
