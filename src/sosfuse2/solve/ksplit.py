"""KSPLIT (split transport, SOSFUSE2_KSPLIT=1/2/3) host side of fused.Kernel: column records, the VPROD aerosol
producer and its plan, the slimray consumer, the chunk projection and the VERIFY comparisons. Kernel inherits these
methods from KsplitMixin; they use the shared Kernel members (raw_kernel, timed_launch, next_chunk_end, device state)
through self. Moved verbatim from fused.py; fused re-exports the module constants below."""

import json
import os
import time

import numpy as np

from ..atmosphere.components import SINV_F
from ..backends.flags import KSPLIT_VARIANTS
from ..backends.sh_layout import source_size
from ..foundation.timers import stage
from .ksplit_tolerance import compare_chunks, compare_tol, log_summary, tolerances

SLIM_PROJ_THREADS = 128  # two wavelength groups per moment
SLIM_PROJ_MOMENTS = 64  # includes the final flux row
KSPLIT_BLOCK = 128  # split kernels only (168 registers: 12 warps/SM at 128 vs 8 at 256)
# SOSFUSE2_KSPLIT=2: PROD is replaced by kernel.cu sos_vprod (visitor-stream producer, same module and
# options as PROD). Plan per fixed chunk (aerosol cap, once per state): records (slot = (point-base)*4+corner) stable-sorted by
# source cell (one uint64 sort of cell<<32|slot), visitors = runs of one (cell, ray), blocks = <= VPROD_VISITORS visitors of one cell.
VPROD_VISITORS = 256
VPROD_ORDER = (
    64,
    4,
)  # VPROD block order: (heights, chi columns) cell tiles, None = cell index order (vprod_build_plan)
VPROD_THREADS = 128


def aero_w(W):
    """Aerosol record row stride (floats): W padded to a multiple of 4 (kernel.cu AERO_W; 16 for FW 15). VPROD stores one float4 per
    thread, the consumer reads lanes < W; the pad lane is never read (measured 2.6x on a 5090)."""
    return ((W + 3) // 4) * 4


VPROD_SENTINEL = 0xFFFFFFFF  # SOSFUSE2_KSPLIT_VERIFY: unwritten aerosol words before each producer
VPROD_VERIFY_STAGE = 16 << 20  # floats per device compare piece (64 MB)
VPROD_VERIFY_WINDOW = 512 << 20  # bytes of aerosol words per verify window (pinned host snapshot; cells are not split)
VPROD_STORE_BPP = 28  # stored plan estimate, B/point (measured 25.4 eda, 26.1 4090; vprod_budget)
VPROD_COUNT_BLOCKS = 1024  # counting-sort blocks per plan (segment >= 4096 records; vprod_build_plan)
VPROD_COUNT_SHARED = 48 << 10  # its per-block cell counters (4 B/cell) must fit the default dynamic shared limit
VPROD_PLAN_SRC = r'''
extern "C" __global__ void vp_keys(long long nrec,const int*c0cp,int nc,unsigned long long*key){
 const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=nrec)return;
 const int c=c0cp[2*(i>>2)],k=(int)(i&3);
 const unsigned long long cell=c<0?0xFFFFFFFFull:(unsigned long long)(c+(k==0?0:k==1?1:k==2?nc:nc+1));
 key[i]=(cell<<32)|(unsigned long long)i;}
extern "C" __global__ void vp_split(long long n,const unsigned long long*key,int*slot,int*cell){
 const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 const unsigned long long x=key[i];slot[i]=(int)(x&0xFFFFFFFFull);cell[i]=(int)(x>>32);}
// Stable counting sort by source cell (same slot/cell as the key sort above, 3.9x
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


class KsplitMixin:
    """KSPLIT methods of fused.Kernel (see the module docstring)."""

    def ksplit_pointer_args(self, kb):
        """SINV_PARAM order. Resolve at launch time: aero can be resized by another bundle,
        while VERIFY and G/L allocate their buffers after the base argument list was built.
        The c0cp address is passed as a CUDA int2* (including mapped pinned host storage);
        its owning allocation remains in kb until the bundle is released. Zero means unused.
        """
        aero = getattr(self, 'ksplit_aero', None)
        return [
            aero if aero is not None else np.uint64(0),
            kb.get('c0cp_ptr', np.uint64(0)),
            kb.get('dump', np.uint64(0)),
        ]

    def ksplit_memory(self):
        """Usable free bytes, including cached pool blocks and respecting its limit."""
        pool = self.xp.get_default_memory_pool()
        free = int(self.xp.cuda.runtime.memGetInfo()[0]) + int(pool.free_bytes())
        if pool.get_limit():
            free = min(free, int(pool.get_limit()) - int(pool.used_bytes()))
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
            self.ksplit_mode = previous['effective']
            self.ksplit = bool(self.ksplit_mode)
            if not self.ksplit or getattr(self, '_ksplit_preflight_state', None) == id(s):
                return previous
        requested = self.ksplit_mode
        if not requested:
            s.ksplit_decision = dict(requested=0, effective=0)
            return s.ksplit_decision
        flag = os.environ.get('SOSFUSE2_KSPLIT_FALLBACK', '1')
        if flag not in ('0', '1'):
            raise ValueError('E_KSPLIT_FALLBACK: expected 0 or 1')
        W = s.W
        nh, nc = len(s.geo.h), len(s.geo.chi)
        field_bytes = nh * nc * source_size(s) * W * 4
        text = os.environ.get('SOSFUSE2_KSPLIT_AERO_MB', '').strip()
        cap = max(1, int(float(text) * 2**20) // (4 * aero_w(W) * 4)) if text else None
        rows = []
        for bundle, phase in bundles:
            co = np.asarray(self.host(bundle.arrays['cell_off']), 'i8')
            pcum = (np.asarray(self.host(bundle.arrays['edge_off'][bundle.arrays['cell_off']]), 'i8') - co) * bundle.nq
            points = int(pcum[-1])
            column = (16 + SINV_F * points) * 4
            S = 2 if self.config().c3_fw15 else 1
            fixed = (
                column + (S * bundle.cells * (4 + self.K) * W * 8 if phase == 0 else 0) + 5 * nh * nc * self.K * W * 4
            )
            rows.append(dict(bundle=bundle, co=co, pcum=pcum, points=points, column=column, fixed=fixed))
        while True:
            live = self.ksplit_memory()
            solo = live + int(getattr(self, 'ksplit_other_bytes', lambda: 0)())
            if requested == 1:
                peak = (
                    sum(r['column'] for r in rows)
                    + (cap or max(1, (1024 << 20) // (4 * aero_w(W) * 4))) * 4 * aero_w(W) * 4
                )
                budget = int(0.9 * solo)
                live_budget = int(0.9 * live)
            else:
                # Enumerate the same stored/rebuilt and device/host options as
                # vprod_budget. Retained records from OTHER bundles are live too.
                import itertools

                choices = []
                for row in rows:
                    result = self.vprod_budget(
                        row['co'], row['pcum'], row['points'], W, row['fixed'], solo, field_bytes, cap, minimum=True
                    )
                    if result.get('error'):
                        raise MemoryError('E_KSPLIT_MEMORY: ' + result['error'])
                    choices.append(result['options'])

                def state_peak(combo):
                    held = [row['column'] + r['records_bytes'] + r['store_bytes'] for row, r in zip(rows, combo)]
                    return sum(held) + max(r['peak'] - h for r, h in zip(combo, held))

                combinations = list(itertools.product(*choices))
                peak = min(map(state_peak, combinations))
                budget = solo - 16 * field_bytes - max(256 << 20, int(0.03 * solo))
                live_budget = live - 16 * field_bytes - max(256 << 20, int(0.03 * live))
            decision = dict(
                requested=requested,
                effective=requested,
                peak=int(peak),
                budget=int(budget),
                free=int(live),
                points=sum(r['points'] for r in rows),
                reason='fits',
            )
            if peak > budget:
                if flag == '0' or previous is not None:
                    raise MemoryError(f'E_KSPLIT_MEMORY: state minimum peak {peak} exceeds solo budget {budget}')
                decision.update(effective=0, reason='state_minimum_exceeds_solo_budget')
                self.ksplit_mode = 0
                self.ksplit = False
                hold = getattr(self, 'ksplit_hold', None)
                if hold is not None:
                    hold(0)
                break
            if peak <= live_budget:
                if requested >= 2:
                    # Preserve vprod_budget's preference order, but size BOTH
                    # bundles together: the first cannot spend the second's room.
                    plans = list(next(c for c in combinations if state_peak(c) <= live_budget))
                    for i, row in enumerate(rows):
                        lo, hi = plans[i]['cap'], max(plans[i]['cap'], row['points'])
                        if cap is not None:
                            hi = lo
                        while lo < hi:
                            mid = (lo + hi + 1) // 2
                            candidate = self.vprod_budget(
                                row['co'],
                                row['pcum'],
                                row['points'],
                                W,
                                row['fixed'],
                                live,
                                field_bytes,
                                cap,
                                option=(plans[i]['store'], plans[i]['resident'], mid),
                            )
                            trial = list(plans)
                            trial[i] = candidate
                            if state_peak(trial) <= live_budget:
                                lo = mid
                                plans = trial
                            else:
                                hi = mid - 1
                    for row, plan in zip(rows, plans):
                        # A plan may not grow its retained storage beyond the
                        # global estimate. Existing overflow handling rebuilds it.
                        plan['store_allowed'] = plan['store_bytes']
                        row['bundle']._ksplit_plan = (id(s), plan)
                    self.ksplit_planned_bytes = state_peak(plans)
                else:
                    self.ksplit_planned_bytes = peak
                hold = getattr(self, 'ksplit_hold', None)
                if hold is not None:
                    hold(self.ksplit_planned_bytes)
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
                from ..precompute.column_parallel import event_fields

                print(
                    json.dumps(
                        dict(event='PHASE', phase='KSPLIT_FALLBACK', **(event_fields() | dict(id=state_id)), **decision)
                    ),
                    flush=True,
                )
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
        bundle.ksplit = None
        xp = self.xp
        W = s.W
        nq = bundle.nq
        co = np.asarray(self.host(bundle.arrays['cell_off']), 'i8')
        pcum = (np.asarray(self.host(bundle.arrays['edge_off'][bundle.arrays['cell_off']]), 'i8') - co) * nq
        points = int(pcum[-1])  # SINV_REC indexes (seg-ray)*nq+step from ray 0
        text = os.environ.get('SOSFUSE2_KSPLIT_AERO_MB', '').strip() or '1024'
        AW = aero_w(W)
        cap_points = max(1, int(float(text) * 2**20) // (4 * AW * 4))
        aero = self.__dict__.get('ksplit_aero')
        need = (16 + SINV_F * points) * 4 + (
            0 if aero is not None and aero.size >= cap_points * 4 * AW else cap_points * 4 * AW * 4
        )
        pool = xp.get_default_memory_pool()
        free = int(xp.cuda.runtime.memGetInfo()[0]) + int(pool.free_bytes())
        if need > 0.9 * free:
            raise MemoryError(
                f'E_KSPLIT_MEMORY: KSPLIT needs {need / 1e9:.2f} GB of device memory for {points} points '
                f'({bundle.cells} cells, phase {phase}), {free / 1e9:.2f} GB free; lower SOSFUSE2_KSPLIT_AERO_MB or unset SOSFUSE2_KSPLIT'
            )
        if aero is None or aero.size < cap_points * 4 * AW:
            self.ksplit_aero = aero = None
            aero = self.ksplit_aero = xp.empty(cap_points * 4 * AW, 'f4')
        buf = xp.empty(16 + SINV_F * points, 'f4')
        hdr32 = buf[:16].view('i4')
        hdr64 = buf[:16].view('i8')
        hdr64.fill(0)
        kb = bundle.ksplit = dict(state=id(s), buf=buf, hdr32=hdr32, hdr64=hdr64, pcum=pcum, cap_points=cap_points)
        print(
            'KSPLIT '
            + json.dumps(
                dict(
                    event='columns',
                    phase=int(phase),
                    cells=int(bundle.cells),
                    rays=int(bundle.rays),
                    nq=int(nq),
                    points=points,
                    column_bytes=int(buf.nbytes),
                    aero_bytes=int(aero.nbytes),
                    aero_points=cap_points,
                    free_bytes=free,
                    block=KSPLIT_BLOCK,
                )
            ),
            flush=True,
        )
        hdr32[0:1].fill(1)
        started = time.perf_counter()
        self.launch(bundle, s, field, False, phase, boundary, _ksplit='record')
        self.sync()
        print(
            'KSPLIT ' + json.dumps(dict(event='recorded', phase=int(phase), seconds=time.perf_counter() - started)),
            flush=True,
        )
        return kb

    def vprod_columns(self, bundle, s, field, phase, boundary):
        """SOSFUSE2_KSPLIT=2 per bundle and state: columns, (c0, cp) records, the fixed chunk partition, the stored visitor plans
        and the shared aerosol buffer, all sized by vprod_budget. Order: columns + records (record pass) -> plans of every chunk
        (stored mode, before the aerosol buffer exists) -> aerosol buffer. The aerosol buffer is shared by the state's bundles
        and re-allocated at this bundle's largest chunk; each launch passes the current buffer pointer, and a
        bundle whose largest chunk no longer fits is re-partitioned there (vprod_fit)."""
        bundle.ksplit = None
        xp = self.xp
        W = s.W
        nq = bundle.nq
        nh, nc = len(s.geo.h), len(s.geo.chi)
        K = self.K
        co = np.asarray(self.host(bundle.arrays['cell_off']), 'i8')
        pcum = (np.asarray(self.host(bundle.arrays['edge_off'][bundle.arrays['cell_off']]), 'i8') - co) * nq
        points = int(pcum[-1])  # SINV_REC indexes (seg-ray)*nq+step from ray 0
        S = 2 if self.config().c3_fw15 else 1
        # launch() arrays after this point: moments and CELL_SPLIT scratch (phase 0), the combined SH sidecar and its FP32
        # build temporaries (sh_layout.combined_aux, about 4 planes of [nh*nc][K][W])
        fixed = (
            (16 + SINV_F * points) * 4
            + (S * bundle.cells * (4 + K) * W * 8 if phase == 0 else 0)
            + 5 * nh * nc * K * W * 4
        )
        text = os.environ.get('SOSFUSE2_KSPLIT_AERO_MB', '').strip()
        env_cap = max(1, int(float(text) * 2**20) // (4 * aero_w(W) * 4)) if text else None
        hold = getattr(self, 'ksplit_hold', None)
        others = self.__dict__.setdefault('_ksplit_held', {})
        others.pop(id(bundle), None)
        planned = getattr(bundle, '_ksplit_plan', None)
        while True:
            pool = xp.get_default_memory_pool()
            limit = int(pool.get_limit())
            aero = self.__dict__.get('ksplit_aero')
            held = int(aero.nbytes) if aero is not None else 0
            free = int(xp.cuda.runtime.memGetInfo()[0]) + int(pool.free_bytes())
            if limit:
                free = min(
                    free, limit - int(pool.used_bytes())
                )  # pipeline_execution caps the pool at free-1 GB per service
            room = self.ksplit_room() if hold is not None else None
            if (
                room is not None
            ):  # persistent service, other states resident: stay out of their reservations (pipeline_execution)
                free = min(free, int(room) - sum(others.values()) - held)
            free += held  # the shared aerosol buffer is released and re-sized below
            bud = (
                dict(planned[1])
                if planned is not None and planned[0] == id(s)
                else self.vprod_budget(co, pcum, points, W, fixed, free, nh * nc * source_size(s) * W * 4, env_cap)
            )
            # too little room beside the other resident states: wait until one releases memory (not while all of them wait)
            if not (bud.get('error') and room is not None and self.ksplit_wait()):
                break
        if bud.get('error'):
            raise MemoryError(
                f'E_KSPLIT_MEMORY: KSPLIT=2 for {points} points ({bundle.cells} cells, phase {phase}): {bud["error"]}; '
                f'{free / 1e9:.2f} GB available; lower SOSFUSE2_KSPLIT_AERO_MB or unset SOSFUSE2_KSPLIT'
            )
        if aero is not None:
            self.ksplit_aero = aero = None  # released before the plans are built
        buf = xp.empty(16 + SINV_F * points, 'f4')
        hdr32 = buf[:16].view('i4')
        hdr64 = buf[:16].view('i8')
        hdr64.fill(0)
        kb = bundle.ksplit = dict(
            state=id(s),
            buf=buf,
            hdr32=hdr32,
            hdr64=hdr64,
            pcum=pcum,
            co=co,
            cap_points=bud['cap'],
            chunks=bud['chunks'],
            pmax=bud['pmax'],
            resident=bud['resident'],
            store=bud['store'],
            plans={},
            stored_bytes=0,
            iteration=0,
            c0cp_dev=None,
        )
        # (c0, cp) records of every point, written by the record pass (kernel.cu SINV mode 1, vp_records argument); c0 = -1 marks
        # points of skipped segments (b-a <= 1e-8), which PROD never writes either.
        if bud['resident'] == 'device':
            kb['c0cp_dev'] = xp.full(2 * points, -1, 'i4')
            address = int(kb['c0cp_dev'].data.ptr)
        else:
            # Not enough device memory (pool limit) for the records (eda 10 GB): the record pass writes them into mapped pinned
            # host memory, and each chunk's points (8 B/point) are uploaded before its plan and VPROD launches.
            mem = kb['c0cp_mem'] = xp.cuda.PinnedMemory(8 * max(points, 1), xp.cuda.runtime.hostAllocMapped)
            kb['c0cp_host'] = np.ndarray((2 * max(points, 1),), np.int32, buffer=xp.cuda.PinnedMemoryPointer(mem, 0))
            # Unified addressing (64-bit Linux): a cudaHostAlloc pointer is used directly by kernels (CUDA programming guide).
            kb['c0cp_host'].fill(-1)
            address = int(mem.ptr)
            kb['c0cp_stage'] = xp.empty(2 * max(bud['pmax'], 1), 'i4')
        kb['c0cp_ptr'] = np.uint64(address)  # CUDA int2* argument; kb owns device/pinned storage
        print(
            'KSPLIT '
            + json.dumps(
                dict(
                    event='columns',
                    phase=int(phase),
                    cells=int(bundle.cells),
                    rays=int(bundle.rays),
                    nq=int(nq),
                    points=points,
                    column_bytes=int(buf.nbytes),
                    free_bytes=int(free),
                    block=KSPLIT_BLOCK,
                    mode=2,
                    verify=bool(self.ksplit_verify),
                    **{k: v for k, v in bud.items() if k != 'chunks'},
                    chunk_count=len(bud['chunks']),
                )
            ),
            flush=True,
        )
        hdr32[0:1].fill(1)
        started = time.perf_counter()
        self.launch(bundle, s, field, False, phase, boundary, _ksplit='record')
        self.sync()
        print(
            'KSPLIT ' + json.dumps(dict(event='recorded', phase=int(phase), seconds=time.perf_counter() - started)),
            flush=True,
        )
        if kb['store']:
            started = time.perf_counter()
            allowed = bud['store_allowed']
            for ci, (c0, c1) in enumerate(kb['chunks']):
                base = int(pcum[c0])
                npt = int(pcum[c1]) - base
                plan = self.vprod_build_plan(kb, ci, bundle, self.vprod_c0cp(kb, base, npt), nh, nc)
                kb['plans'][ci] = plan
                kb['stored_bytes'] += plan['bytes']
                if (
                    kb['stored_bytes'] > allowed
                ):  # measured plans larger than the budget's estimate: rebuild per iteration
                    kb['plans'].clear()
                    kb['store'] = False
                    break
            self.sync()
            print(
                'KSPLIT '
                + json.dumps(
                    dict(
                        event='plans',
                        phase=int(phase),
                        stored=bool(kb['store']),
                        chunks=len(kb['chunks']),
                        stored_bytes=int(kb['stored_bytes']),
                        bytes_per_point=kb['stored_bytes'] / max(points, 1),
                        estimate_bytes=int(bud['store_bytes']),
                        allowed_bytes=int(allowed),
                        seconds=time.perf_counter() - started,
                    )
                ),
                flush=True,
            )
            if not kb['store']:
                kb['stored_bytes'] = 0
        # bytes this bundle keeps while the state's other bundles size theirs (the shared aerosol buffer is counted once below)
        others[id(bundle)] = (
            int(buf.nbytes)
            + int(kb['stored_bytes'])
            + sum(int(kb[k].nbytes) for k in ('c0cp_dev', 'c0cp_stage') if kb.get(k) is not None)
        )
        npts = bud[
            'pmax'
        ]  # another bundle whose largest chunk no longer fits re-partitions at its next launch (vprod_fit)
        self.ksplit_aero = aero = xp.empty(max(npts, 1) * 4 * aero_w(W), 'f4')
        if (
            hold is not None
        ):  # the state's reservation grows by what KSPLIT holds, so later states are admitted only beside it
            hold(sum(others.values()) + int(aero.nbytes))
        print(
            'KSPLIT '
            + json.dumps(
                dict(
                    event='aero', phase=int(phase), aero_points=npts, aero_bytes=int(aero.nbytes), auto=env_cap is None
                )
            ),
            flush=True,
        )
        return kb

    def vprod_budget(self, co, pcum, points, W, fixed, free, field_bytes, env_cap, *, minimum=False, option=None):
        """SOSFUSE2_KSPLIT=2 memory rule. Budget B = free (pool-limited, shared aerosol buffer counted as free) - reserve, reserve =
        16 x the source field (solver growth after this point: Anderson history, apply temporaries) + max(256 MB, 3 %).
        Per option (stored plans first, then rebuild; records
        device-resident first, then pinned host), the largest aerosol chunk cap (points) whose peak fits B:
          run   = fixed + records + stored plans + aerosol 4W*4 B/pt of the largest chunk + y rows + verify piece
                  (+ build workspace when rebuilding every iteration)
          build = fixed + records + stored plans + build workspace (stored mode builds all plans before the aerosol buffer)
        records 8 B/point (device) or 8 B/pt of the largest chunk (host upload); stored plans estimated VPROD_STORE_BPP B/point
        (measured 25.1-26.1: slot 4/record, visitor 8, block 12); build workspace 21 B/record + 4 B/point (live bound;
        pool 94 B/point incl. 8 B/point inputs); y K*4 B/ray. Stored mode needs a cap >= points/64 (at most 64 chunks);
        SOSFUSE2_KSPLIT_AERO_MB fixes the cap."""
        reserve = 16 * field_bytes + max(256 << 20, int(0.03 * free))
        B = free - reserve
        K = self.K
        ncell = len(pcum) - 1
        lc = int(np.diff(pcum).max()) if ncell else 1
        stage = VPROD_VERIFY_STAGE * 4 if self.ksplit_verify else 0
        mode = getattr(self, 'ksplit_mode', 2)

        def partition(cap):
            chunks = []
            c0 = 0
            while c0 < ncell:
                hi = int(np.searchsorted(pcum, pcum[c0] + cap, side='right')) - 1
                if hi <= c0:
                    return None
                chunks.append((c0, hi))
                c0 = hi
            return chunks

        def peak(cap, store, resident):
            chunks = partition(cap)
            if chunks is None:
                return None
            pmax = max(int(pcum[b] - pcum[a]) for a, b in chunks)
            rmax = max(int(co[b] - co[a]) for a, b in chunks)
            ws = 21 * 4 * pmax + 4 * pmax + (1 << 20)
            y = rmax * K * 4
            rec = 8 * points if resident == 'device' else 8 * pmax
            st = VPROD_STORE_BPP * points if store else 0
            run = fixed + rec + st + 4 * aero_w(W) * 4 * pmax + y + stage + (0 if store else ws)
            if mode == 3:  # chunk ray radiances [ray][4W] FP32 (+ the VERIFY ray dump)
                run += 4 * W * 4 * rmax * (2 if self.ksplit_verify else 1)
            build = fixed + rec + st + ws
            return dict(
                peak=max(run, build) if store else run,
                chunks=chunks,
                pmax=pmax,
                rmax=rmax,
                workspace=ws,
                y_bytes=y,
                records_bytes=rec,
                store_bytes=st,
                run=run,
                cap=cap,
                store=store,
                resident=resident,
                budget=int(B),
                reserve=int(reserve),
                store_allowed=st,
            )

        if option is not None:
            store, resident, cap = option
            return peak(cap, store, resident)
        if minimum:
            options = []
            for store, resident in ((True, 'device'), (True, 'host'), (False, 'device'), (False, 'host')):
                cap = env_cap if env_cap is not None else max(lc, -(-points // 64)) if store else lc
                r = peak(cap, store, resident)
                if r is not None:
                    options.append(r)
            return dict(options=options) if options else dict(error='no whole cell fits the requested aerosol cap')
        tried = []
        for store, resident in ((True, 'device'), (True, 'host'), (False, 'device'), (False, 'host')):
            lo = max(lc, -(-points // 64)) if store else lc
            if env_cap is not None:
                if env_cap < lc:
                    return dict(error=f'cell with {lc} points > aerosol chunk {env_cap} (SOSFUSE2_KSPLIT_AERO_MB)')
                cap = env_cap
                r = peak(cap, store, resident)
                if r is None or r['peak'] > B:
                    tried.append((store, resident, r and r['peak']))
                    continue
            else:
                hi = max(points, lc)
                r = peak(lo, store, resident)
                if r is None or r['peak'] > B:
                    tried.append((store, resident, r and r['peak']))
                    continue
                cap = lo
                while lo < hi:  # largest cap that fits (peak grows with the cap)
                    mid = (lo + hi + 1) // 2
                    q = peak(mid, store, resident)
                    if q is not None and q['peak'] <= B:
                        lo = mid
                    else:
                        hi = mid - 1
                cap = lo
                r = peak(cap, store, resident)
            # stored plans may grow up to what the chosen cap leaves free (measured during the build)
            r.update(
                cap=cap,
                store=store,
                resident=resident,
                budget=int(B),
                reserve=int(reserve),
                store_allowed=(int(B - (r['peak'] - r['store_bytes'])) if store else 0),
            )
            return r
        return dict(
            error=f'no option fits {B / 1e9:.2f} GB (peaks {[(a, b, c and round(c / 1e9, 2)) for a, b, c in tried]})'
        )

    def vprod_fit(self, kb, bundle, cap, phase, nh, nc):
        """Re-partition a bundle whose largest chunk exceeds the shared aerosol buffer (re-sized by a later bundle): cells are
        not split; stored plans are rebuilt for the new chunks (the small ground bundle in practice)."""
        if kb['pmax'] <= cap:
            return
        pc = kb['pcum']
        chunks = []
        c0 = 0
        ncell = len(pc) - 1
        while c0 < ncell:
            hi = int(np.searchsorted(pc, pc[c0] + cap, side='right')) - 1
            if hi <= c0:
                raise MemoryError(
                    f'E_KSPLIT_MEMORY: cell {c0} has {int(pc[c0 + 1] - pc[c0])} points > shared aerosol chunk {cap}'
                )
            chunks.append((c0, hi))
            c0 = hi
        old = len(kb['chunks'])
        kb.update(chunks=chunks, pmax=max(int(pc[b] - pc[a]) for a, b in chunks), cap_points=cap)
        kb['plans'] = {}
        kb['stored_bytes'] = 0
        if kb['store']:
            for ci, (c0, c1) in enumerate(chunks):
                base = int(pc[c0])
                npt = int(pc[c1]) - base
                plan = self.vprod_build_plan(kb, ci, bundle, self.vprod_c0cp(kb, base, npt), nh, nc)
                kb['plans'][ci] = plan
                kb['stored_bytes'] += plan['bytes']
        print(
            'KSPLIT '
            + json.dumps(
                dict(
                    event='repartition',
                    phase=int(phase),
                    chunks_before=old,
                    chunks=len(chunks),
                    cap=int(cap),
                    stored=bool(kb['store']),
                    stored_bytes=int(kb['stored_bytes']),
                )
            ),
            flush=True,
        )

    def vprod_kernels(self, W, phase, boundary):
        """sos_vprod_y and sos_vprod from the PROD module (same source and NVRTC options as the PROD kernel; cf. fl_project_kernel)."""
        self.raw_kernel(W, 0, phase, boundary, variant='prod')
        flags = tuple(sorted(set(self.config(W).defines) | set(KSPLIT_VARIANTS['prod'])))
        key = self.kernel_key(
            flags, W, self.tile_width(W, False, phase), 0, phase, boundary, stage1=self.stage1_flags()
        )
        out = []
        for name in ('sos_vprod_y', 'sos_vprod'):
            nkey = key + (name,)
            if nkey not in self.raw:
                code, options = self.raw_options[key]
                with stage('raw_compile'):
                    raw = self.xp.RawKernel(code, name, options=options, backend='nvrtc')
                    raw.compile()
                self.raw[nkey] = raw
                self.attributes[f'{name}:{W}_phase{int(phase)}_boundary{int(boundary)}' + self.attribute_suffix()] = (
                    dict(raw.attributes, block_size=VPROD_THREADS)
                )
            out.append(self.raw[nkey])
        return out

    def vprod_plan_module(self):
        mod = self.__dict__.get('_vprod_plan_mod')
        if mod is None:
            mod = self._vprod_plan_mod = self.xp.RawModule(
                code=VPROD_PLAN_SRC, options=('-std=c++17', f'-DSOSFUSE2_DETERMINISTIC={int(self.deterministic)}')
            )
        return mod

    def vprod_c0cp(self, kb, base, npt):
        """The chunk's (c0, cp) int2 records as an int32 [2*npt] device array (view, or upload of the pinned host copy)."""
        if kb['c0cp_dev'] is not None:
            return kb['c0cp_dev'][2 * base : 2 * (base + npt)]
        stage_ = kb['c0cp_stage'][: 2 * npt]
        stage_.set(kb['c0cp_host'][2 * base : 2 * (base + npt)])
        return stage_

    def vprod_plan(self, kb, ci, bundle, c0cp, nh, nc):
        """Visitor plan of fixed chunk ci: the stored one (built once per state by vprod_columns) or a new one."""
        if ci in kb['plans']:
            return kb['plans'][ci], True
        return self.vprod_build_plan(kb, ci, bundle, c0cp, nh, nc), False

    def vprod_build_plan(self, kb, ci, bundle, c0cp, nh, nc):
        """Records stable-sorted by source cell (slot order within a cell), visitors = runs of one (cell, ray), blocks of
        <= VPROD_VISITORS visitors of one cell. The y rows are not part of the plan (vprod_launch computes them per chunk)."""
        xp = self.xp
        mod = self.vprod_plan_module()
        T = 256
        g = lambda m: ((int(m) + T - 1) // T,)
        c0, c1 = kb['chunks'][ci]
        pc = kb['pcum']
        co = kb['co']
        base = int(pc[c0])
        npt = int(pc[c1]) - base
        r0, r1 = int(co[c0]), int(co[c1])
        nray = r1 - r0
        ncell = nh * nc
        n = 4 * npt
        nrec = 4 * int(xp.count_nonzero(c0cp[0::2] >= 0)) if npt else 0
        slot = xp.empty(max(nrec, 1), 'i4')
        cell = xp.empty(max(nrec, 1), 'i4')
        rstart = None
        if nrec and 4 * ncell <= VPROD_COUNT_SHARED:
            seg = -(-max(4096, -(-n // VPROD_COUNT_BLOCKS)) // 128) * 128
            nb = -(-n // seg)  # 128 records = vp_scatter's 4 tiles
            err = xp.zeros(1, 'i4')
            counts = xp.empty(ncell * nb, 'i4')
            args = (np.int64(n), c0cp, np.int32(nc), np.int32(ncell), np.int32(seg), np.int32(nb))
            mod.get_function('vp_count')((nb,), (T,), args + (counts, err), shared_mem=4 * ncell)
            offs = xp.cumsum(counts, dtype=xp.int32)
            offs -= counts
            mod.get_function('vp_scatter')((nb,), (32,), args + (offs, slot), shared_mem=4 * ncell)
            rstart = xp.empty(ncell + 1, xp.int64)
            mod.get_function('vp_rstart')(
                g(ncell + 1), (T,), (np.int32(ncell), np.int32(nb), np.int64(nrec), offs, rstart)
            )
            mod.get_function('vp_fill_cell')((ncell,), (T,), (np.int32(ncell), rstart, cell))
            if int(err[0]):
                raise ValueError(f'E_VPROD_CELL_RANGE: a record cell is outside 0..{ncell - 1} (chunk {ci})')
            del err, counts, offs
        elif nrec:  # more cells than one block's shared counters: the key sort (same slot/cell)
            key = xp.empty(n, xp.uint64)
            mod.get_function('vp_keys')(g(n), (T,), (np.int64(n), c0cp, np.int32(nc), key))
            key.sort()
            mod.get_function('vp_split')(g(nrec), (T,), (np.int64(nrec), key, slot, cell))
            del key
        pray = xp.empty(max(npt, 1), 'i4')
        if nray:
            mod.get_function('vp_point_ray')(
                g(nray),
                (T,),
                (np.int32(nray), np.int64(r0), bundle.arrays['edge_off'], np.int32(bundle.nq), np.int64(base), pray),
            )
        flag = xp.zeros(max(nrec, 1), xp.uint8)
        if nrec:
            mod.get_function('vp_flags')(g(nrec), (T,), (np.int64(nrec), slot, cell, pray, flag))
        scan = xp.cumsum(flag, dtype=xp.int32)
        nvis = int(scan[nrec - 1]) if nrec else 0
        vis_rbeg = xp.empty(nvis + 1, 'i4')
        vis_ray = xp.empty(max(nvis, 1), 'i4')
        if nrec:
            mod.get_function('vp_visitors')(g(nrec), (T,), (np.int64(nrec), flag, scan, slot, pray, vis_rbeg, vis_ray))
        vis_rbeg[nvis:].fill(nrec)
        if rstart is None:
            rstart = xp.searchsorted(cell[:nrec], xp.arange(ncell + 1, dtype=xp.int32)).astype(xp.int64)
        cstart = xp.empty(ncell + 1, xp.int64)
        mod.get_function('vp_cells')(
            g(ncell + 1), (T,), (np.int32(ncell), np.int64(nrec), np.int32(nvis), rstart, scan, cstart)
        )
        del flag, scan, cell, rstart, pray
        nb = (xp.diff(cstart) + VPROD_VISITORS - 1) // VPROD_VISITORS
        bstart = xp.concatenate([xp.zeros(1, xp.int64), xp.cumsum(nb)])
        nblk = int(bstart[-1])
        blk = xp.empty(max(3 * nblk, 3), 'i4')
        work = xp.zeros(max(nblk, 1), xp.int64)
        mod.get_function('vp_blocks')(
            g(ncell), (T,), (np.int32(ncell), cstart, bstart, np.int32(VPROD_VISITORS), vis_rbeg, blk, work)
        )
        if VPROD_ORDER and nblk:
            # Block issue order: source cells in tiles of th heights x tc chi columns (row-major
            # inside a tile, tiles row-major), a cell's blocks consecutive (stable argsort). Every record is computed and written
            # on its own, so the order changes no value; it only changes which y rows and aerosol lines share L2 in time.
            th, tc = VPROD_ORDER
            b3 = blk[: 3 * nblk].reshape(nblk, 3)
            cb = b3[:, 0].astype(xp.int64)
            ih, ic = cb // nc, cb % nc
            order = xp.argsort(
                ((ih // th) * ((nc + tc - 1) // tc) + ic // tc) * (th * tc) + (ih % th) * tc + ic % tc, kind='stable'
            )
            blk = xp.ascontiguousarray(b3[order]).reshape(-1)
            work = work[order]
            del b3, cb, ih, ic, order
        bcum = np.r_[0.0, np.cumsum(np.asarray(self.host(work[:nblk]), 'f8'))]
        del work, cstart, bstart, nb
        plan = dict(
            slot=slot,
            vis_rbeg=vis_rbeg,
            vis_ray=vis_ray,
            blk=blk,
            nrec=nrec,
            nvis=nvis,
            nblk=nblk,
            bcum=bcum,
            npt=npt,
            nray=nray,
        )
        plan['bytes'] = int(sum(a.nbytes for a in (slot, vis_rbeg, vis_ray, blk)))
        return plan

    def ksplit_chunk(self, kb, c0, c1):
        """Cap a chunk to the aerosol buffer and point the header at its first point (kernel.cu SLIM_AERO)."""
        pc = kb['pcum']
        hi = int(np.searchsorted(pc, pc[c0] + kb['cap_points'], side='right')) - 1
        if hi <= c0:
            raise MemoryError(
                f'E_KSPLIT_MEMORY: cell {c0} has {int(pc[c0 + 1] - pc[c0])} points > aerosol chunk {kb["cap_points"]}; '
                'raise SOSFUSE2_KSPLIT_AERO_MB'
            )
        kb['hdr64'][2:3].fill(int(pc[c0]))
        return min(c1, hi)

    def budget_end(self, kind, a, n, step, cum, initial=None):
        """End of the next launch inside a fixed KSPLIT=2 chunk [a, n): the whole rest when its predicted time (predict_seconds,
        the next_chunk_end model) fits the profile target, else next_chunk_end. Without this, next_chunk_end's 1.5x growth limit
        restarts from the previous chunk's tail and splits every chunk into small underfilled launches."""
        if cum is not None and self.__dict__.get('_launch_obs', {}).get(kind) and (not step or n <= a + step):
            _, p, _ = self.device()
            target = p['launch_target_seconds'] or self.PREDICT_CAP * (p['launch_budget_seconds'] or 0) or float('inf')
            if self.predict_seconds(kind, cum[n] - cum[a]) <= min(
                target, self.PREDICT_CAP * (p['launch_budget_seconds'] or float('inf'))
            ):
                return n
        return self.next_chunk_end(kind, a, n, step, cum, initial)

    def ksplit_cells(self, kind, raw, c0, c1, cell_part, geom, step, cum, report):
        """Cell-range launches of one split kernel inside fixed chunk [c0, c1) under the launch budget (budget_end)."""
        (nsplit, grid_y), block, shared = geom
        a = c0
        launches = []
        while a < c1:
            b = self.budget_end(kind, a, c1, step, cum)
            seconds = self.timed_launch(
                kind, raw, ((b - a) * nsplit, grid_y), block, cell_part(a, b), shared, b - a, cum, a, b
            )
            report['gpu_seconds'] += seconds
            report['max_seconds'] = max(report['max_seconds'], seconds)
            launches.append(((b - a) * nsplit, seconds))
            a = b
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
            self.attributes[f'sos_slim_proj:{W}_boundary{int(boundary)}' + self.attribute_suffix()] = dict(
                raw.attributes, block_size=SLIM_PROJ_THREADS
            )
        return self.raw[nkey]

    def slim_rays(self, kb, ctx, W, boundary, c0, c1, ray_rads, kind, report, variant='slimray', ranges=None):
        """KSPLIT=3 warm consumer: the chunk's rays (queue head/limit in ray_sync, chunk-local ray index) with SLIM's per-ray
        code (SLIM_RAYS: warm ray split, ray launches), each warp publishing its ray's radiance to ray_rads[ray-co[c0]][4W].
        Grid ~ rays/4, launches split by rays under the launch budget."""
        xp = self.xp
        raw = self.raw_kernel(W, 0, 0, boundary, ray_limit=True, variant=variant)
        args = ctx['args']
        li = ctx['lpt_index']
        base = list(args[:li] if li is not None else args)
        base[0] = np.int32(c1 - c0)
        si = ctx['sinv_index']
        base[si : si + 3] = self.ksplit_pointer_args(kb)
        for n, i in ctx['cell_index'].items():
            base[i] = args[i][c0:]
        base[ctx['mi']] = ctx['moments'][c0:c1]
        base[ctx['fi']] = ctx['flux'][c0:c1]
        co = kb['co']
        nray = int(co[c1] - co[c0])
        ncel = c1 - c0
        sync = xp.zeros(ncel + 2, 'i4')
        shared = self.shared_bytes(W, 0, variant=variant)
        cum = ctx['ray_cum']
        cumc = None if cum is None else cum[int(co[c0] - co[0]) : int(co[c1] - co[0]) + 1]
        q0 = 0
        launches = []
        todo = list(ranges) if ranges is not None else None
        while q0 < nray:
            if todo is not None:  # G/L experiment: the G launch's ray ranges (post-launch budget check only)
                q0, q1 = todo.pop(0)
            else:
                q1 = (
                    self.budget_end(kind + '_slimray', q0, nray, None, cumc, initial=nray) if cumc is not None else nray
                )
            sync[0:1].fill(q0)
            sync[ncel + 1 : ncel + 2].fill(q1)
            nb = max(1, -(-(q1 - q0) // (KSPLIT_BLOCK // 32)))
            identify = variant == 'slimray' and os.environ.get('SOSFUSE2_P2MIN_NVTX', '0') == '1'
            if identify:
                print(
                    'P2MIN_SLIMRAY '
                    + json.dumps(
                        dict(
                            kind=kind + '_' + variant,
                            boundary=int(boundary),
                            cells=[c0, c1],
                            rays=[q0, q1],
                            grid=[nb, 1],
                            block=[KSPLIT_BLOCK],
                            args=len(base) + 2,
                            shared=shared,
                        )
                    ),
                    flush=True,
                )
                xp.cuda.nvtx.RangePush('p2min_slimray')
            try:
                seconds = self.timed_launch(
                    kind + '_' + variant,
                    raw,
                    (nb, 1),
                    (KSPLIT_BLOCK,),
                    tuple(base + [ray_rads, sync]),
                    shared,
                    q1 - q0,
                    cumc if todo is None else None,
                    q0,
                    q1,
                )
            finally:
                if identify:
                    xp.cuda.nvtx.RangePop()
            report['gpu_seconds'] += seconds
            report['max_seconds'] = max(report['max_seconds'], seconds)
            launches.append((q0, q1, nb, seconds))
            q0 = q1
        return launches

    def wf_dump(self, path, tag, kb, ctx, bundle, W, boundary, c0, c1, ray_rads, nray, r0, base, npt, slim_s, where):
        """Save one chunk's KSPLIT=3 slimray inputs and its ray radiances (ray_rads[(ray-r0)*4W + 4w+k]) so the
        streaming kernel can be run on real inputs and compared. Opt-in by SOSFUSE2_WF_DUMP=<dir>; production behaviour unchanged."""
        import pathlib

        xp = self.xp
        H = self.host
        args = ctx['args']
        d = pathlib.Path(path)
        d.mkdir(parents=True, exist_ok=True)
        xp.cuda.Device().synchronize()  # the slimray launches above are asynchronous on their stream
        names = (
            'cells',
            'nh',
            'nc',
            'nq',
            'nsh',
            'nsx',
            'first',
            'phase',
            'boundary_flag',
            'cell_off',
            'starts',
            'targets',
            'dirs',
            'weights',
            'edge_off',
            'edges',
            'lower',
            'end_j',
            'end_f',
            'end_col',
            'hits',
            'gu',
            'gw',
            'aa',
            'bb',
            'dd',
            'h',
            'chi',
            'hr',
            'ha',
            'ext',
            'asca',
            'sun_h',
            'sun_col',
            'gm',
            'goff',
            'gdir',
            'gval',
            'height',
            'g',
            'mix',
            'albedo',
        )
        out = {}
        for n, a in zip(names, args[: len(names)]):
            if n in ('weights', 'aa', 'bb', 'dd', 'sun_h', 'sun_col', 'gm', 'goff', 'gdir', 'gval', 'asca', 'gw'):
                continue
            out[n] = np.asarray(H(a)) if hasattr(a, 'shape') else np.asarray(a)
        src = args[42]
        out['src_rows4'] = np.asarray(H(src[:, :4, :]))  # [cell][k<4][W]: the only source rows SLIM reads
        out['boundary'] = np.asarray(H(args[43]))
        si = ctx['sinv_index']
        stp = si + 3 + 1  # SINV(3) + src_sh, then STRATO params
        for n, i in (
            ('st_ext', stp),
            ('st_sca', stp + 1),
            ('st_chi', stp + 2),
            ('st_density', stp + 3),
            ('st_end_col', stp + 5),
        ):
            out[n] = np.asarray(H(args[i]))
        AW = aero_w(W)
        out['aero'] = np.asarray(
            H(self.ksplit_aero[: npt * 4 * AW])
        )  # chunk-local [point][corner][AW], slot = point-base
        out['records'] = np.asarray(
            H(kb['buf'][16 + SINV_F * base : 16 + SINV_F * (base + npt)])
        )  # this chunk's SINV_REC rows (7 floats/point); the whole bundle buffer is GBs
        out['hdr64'] = np.asarray(H(kb['hdr64']))
        out['pcum'] = np.asarray(kb['pcum'])
        out['co'] = np.asarray(kb['co'])
        out['ray_rads'] = np.asarray(H(ray_rads[: nray * 4 * W])).reshape(nray, 4 * W)
        out['meta'] = np.array(
            json.dumps(
                dict(
                    where,
                    c0=int(c0),
                    c1=int(c1),
                    r0=int(r0),
                    nray=int(nray),
                    base=int(base),
                    npt=int(npt),
                    W=int(W),
                    AW=int(AW),
                    boundary=bool(boundary),
                    slim_gpu_seconds=float(slim_s),
                    K=int(self.K),
                    L=int(self.L),
                    aero_rows_per_point=4,
                )
            )
        )
        np.savez(d / f'{tag}.npz', **out)
        raw = self.raw_kernel(W, 0, 0, boundary, ray_limit=True, variant='slimray')
        (d / 'slimray_kernel.cu').write_text(raw.code)
        (d / 'slimray_options.json').write_text(json.dumps(list(raw.options)))
        print(
            'KSPLIT '
            + json.dumps(
                dict(
                    event='wf_dump',
                    path=str(d / f'{tag}.npz'),
                    nray=int(nray),
                    npt=int(npt),
                    src_absmax=float(np.abs(out['src_rows4']).max()),
                    rays_absmax=float(np.abs(out['ray_rads']).max()),
                )
            ),
            flush=True,
        )

    def slim_proj(self, kb, ctx, bundle, W, boundary, c0, c1, y, ray_rads, kind, report, cum):
        """KSPLIT=3 projection of chunk cells [c0, c1): grid (cells, moment tiles of SLIM_PROJ_MOMENTS), cell launches split
        under the launch budget."""
        raw = self.slim_proj_kernel(W, boundary)
        args = ctx['args']
        ci = ctx['cell_index']
        co = kb['co']
        tiles = -(-(5 + self.K) // SLIM_PROJ_MOMENTS)
        a = c0
        launches = []
        while a < c1:
            b = self.budget_end(kind + '_proj', a, c1, None, cum)
            pargs = (
                np.int32(b - a),
                np.int32(KSPLIT_BLOCK // 32),
                np.int64(co[c0]),
                args[ci['cell_off']][a:],
                args[ci['targets']][a:],
                bundle.arrays['dirs'],
                bundle.arrays['weights'],
                bundle.arrays['edge_off'],
                y,
                ray_rads,
                np.float64(ctx['albedo']),
                ctx['moments'][a:b],
                ctx['flux'][a:b],
            )
            seconds = self.timed_launch(
                kind + '_proj', raw, (b - a, tiles), (SLIM_PROJ_THREADS,), pargs, 0, b - a, cum, a, b
            )
            report['gpu_seconds'] += seconds
            report['max_seconds'] = max(report['max_seconds'], seconds)
            launches.append(((b - a) * tiles, seconds))
            a = b
        return launches

    def vprod_launch(
        self, kb, bundle, s, dc, src_sh, W, phase, boundary, kind, raws, cell_part, geom, step, cum, report, ctx=None
    ):
        """SOSFUSE2_KSPLIT=2 chunk loop. Per fixed chunk: plan (stored or rebuilt), y rows, VPROD over block ranges, SLIM over cell
        ranges, both under the launch budget. SOSFUSE2_KSPLIT_VERIFY=1: VPROD writes the sentinel-filled chunk, then per window of
        whole cells (<= VPROD_VERIFY_WINDOW bytes) its words go to a pinned host snapshot, the window is sentinel-filled again,
        PROD rewrites it for a tolerance comparison (E_KSPLIT_VERIFY); restore VPROD before SLIM reads it.
        Aero groups are complete target cells x wavelength, independent of chunk/window/piece sizes.
        SOSFUSE2_KSPLIT=3, phase 0: SLIM is replaced by the ray consumer (slim_rays -> chunk ray radiances) and the projection
        (slim_proj -> moments, flux); with VERIFY the per-cell SLIM runs first with a ray dump (SLIM_RAYDUMP), the consumer's ray
        radiances and projected moments/flux must pass tolerance against SLIM (E_KSPLIT_VERIFY)."""
        xp = self.xp
        raw = dict(raws)
        ykern, vkern = self.vprod_kernels(W, phase, boundary)
        nh, nc = len(s.geo.h), len(s.geo.chi)
        aero = self.ksplit_aero
        co = kb['co']
        K = self.K
        AW = aero_w(W)
        self.vprod_fit(kb, bundle, aero.size // (4 * AW), phase, nh, nc)
        pc = kb['pcum']
        kb['iteration'] += 1
        it = kb['iteration']
        rows = []
        verify = self.ksplit_verify
        used_max = pool_max = 0
        ray_buf = dump_buf = None
        projection_checks = []
        if verify:
            self._verify_tolerances = tolerances()
            wpts = max(1, VPROD_VERIFY_WINDOW // (4 * AW * 4))
            windows = {}
            for ci, (c0, c1) in enumerate(kb['chunks']):
                ws = []
                wa = c0
                while wa < c1:
                    wb = min(c1, max(wa + 1, int(np.searchsorted(pc, pc[wa] + wpts, side='right')) - 1))
                    ws.append((wa, wb))
                    wa = wb
                windows[ci] = ws
            need = max(int(pc[wb] - pc[wa]) for ws in windows.values() for wa, wb in ws) * 4 * AW
            if getattr(self, '_vprod_snap', None) is None or self._vprod_snap.size < need:
                import cupyx

                self._vprod_snap = None
                self._vprod_snap = cupyx.empty_pinned((need,), dtype=np.float32)
        for ci, (c0, c1) in enumerate(kb['chunks']):
            base = int(pc[c0])
            npt = int(pc[c1]) - base
            nfl = npt * 4 * AW
            r0, r1 = int(co[c0]), int(co[c1])
            nray = r1 - r0
            kb['hdr64'][2:3].fill(base)
            ev = {k: xp.cuda.Event() for k in ('p0', 'y0', 'p1', 'v1', 'w1', 'w2', 'c1', 'c2', 's1')}
            if verify:
                aero[:nfl].view(xp.uint32).fill(VPROD_SENTINEL)
            wall = time.perf_counter()
            ev['p0'].record()
            c0cp = self.vprod_c0cp(kb, base, npt)
            dump = os.environ.get('SOSFUSE2_VPROD_DUMP', '').strip()
            if dump and phase == 0 and it == 1 and ci == 0:
                # Plan measurement input: this chunk's (c0, cp) records and ray offsets.
                np.savez(
                    dump,
                    c0cp=self.host(c0cp),
                    eo=self.host(bundle.arrays['edge_off'][r0 : r1 + 1]),
                    base=base,
                    r0=r0,
                    nq=int(bundle.nq),
                    nh=nh,
                    nc=nc,
                )
            plan, stored = self.vprod_plan(kb, ci, bundle, c0cp, nh, nc)
            ev['y0'].record()
            y = xp.empty((max(nray, 1), K), 'f4')  # y rows of the chunk's rays (sh_basis_ds), every iteration
            if nray:
                ykern(
                    ((nray + VPROD_THREADS // 32 - 1) // (VPROD_THREADS // 32),),
                    (VPROD_THREADS,),
                    (np.int32(nray), bundle.arrays['dirs'].reshape(-1)[3 * r0 :], dc['aa2'], dc['bb2'], dc['dd2'], y),
                )
            ev['p1'].record()
            wall = time.perf_counter() - wall
            free_now, total_dev = xp.cuda.runtime.memGetInfo()
            used_max = max(used_max, int(total_dev) - int(free_now))
            pool_max = max(pool_max, int(xp.get_default_memory_pool().total_bytes()))

            def run_vprod(plan=plan, c0cp=c0cp, y=y):
                nblk = plan['nblk']
                bc = plan['bcum'] if cum is not None else None
                b0 = 0
                while b0 < nblk:
                    b1 = self.budget_end(kind + '_vprod', b0, nblk, None, bc, initial=nblk) if bc is not None else nblk
                    vargs = (
                        plan['blk'][3 * b0 :],
                        plan['vis_rbeg'],
                        plan['vis_ray'],
                        plan['slot'],
                        c0cp,
                        y,
                        src_sh,
                        dc['hr'],
                        dc['ha'],
                        dc['strato_density'],
                        np.int32(nh),
                        np.int32(nc),
                        aero,
                    )
                    seconds = self.timed_launch(
                        kind + '_vprod', vkern, (b1 - b0,), (VPROD_THREADS,), vargs, 0, b1 - b0, bc, b0, b1
                    )
                    report['gpu_seconds'] += seconds
                    report['max_seconds'] = max(report['max_seconds'], seconds)
                    b0 = b1

            run_vprod()
            ev['v1'].record()
            prod = []
            if verify:
                valid_points = np.asarray(self.host(c0cp)).reshape(-1, 2)[:, 0] >= 0
                where = dict(phase=int(phase), iteration=it, chunk=ci, base=base, points=npt, cells=[c0, c1])
                for wa, wb in windows[ci]:
                    lo, hi = 4 * AW * int(pc[wa] - base), 4 * AW * int(pc[wb] - base)
                    aero[lo:hi].get(out=self._vprod_snap[: hi - lo])  # VPROD words (sentinel where nothing is written)
                    aero[lo:hi].view(xp.uint32).fill(VPROD_SENTINEL)
                    e0, e1 = xp.cuda.Event(), xp.cuda.Event()
                    e0.record()
                    self.ksplit_cells(kind + '_prod', raw['_prod'], wa, wb, cell_part, geom, step, cum, report)
                    e1.record()
                    prod.append((e0, e1))
                    self.vprod_verify(lo, hi, W, dict(where, window=[int(wa), int(wb)]), valid_points, pc)
                    aero[lo:hi].set(self._vprod_snap[: hi - lo])  # consumer must see the production VPROD values
                print(
                    'KSPLIT ' + json.dumps(dict(event='verify', ok=True, words=nfl, windows=len(windows[ci]), **where)),
                    flush=True,
                )
            ev['w1'].record()
            mode3 = self.ksplit_mode == 3 and phase == 0
            if mode3:
                if ray_buf is None:
                    rmax = max(int(co[b] - co[a]) for a, b in kb['chunks'])
                    ray_buf = xp.empty(max(rmax, 1) * 4 * W, 'f4')
                    dump_buf = xp.empty(max(rmax, 1) * 4 * W, 'f4') if verify else None
                if verify:
                    ray_buf[: nray * 4 * W].view(xp.uint32).fill(VPROD_SENTINEL)
                    dump_buf[: nray * 4 * W].view(xp.uint32).fill(VPROD_SENTINEL)
                    ctx['moments'][c0:c1].fill(np.nan)
                    ctx['flux'][c0:c1].fill(np.nan)
                    kb['dump'] = dump_buf
                    kb['hdr64'][5:6].fill(r0)
                    self.ksplit_cells(
                        kind + '_slimdump',
                        self.raw_kernel(W, 0, phase, boundary, variant='slimdump'),
                        c0,
                        c1,
                        cell_part,
                        geom,
                        step,
                        cum,
                        report,
                    )
                    mref = ctx['moments'][c0:c1].copy()
                    fref = ctx['flux'][c0:c1].copy()
                ev['w2'].record()
                g0 = report['gpu_seconds']
                self.slim_rays(kb, ctx, W, boundary, c0, c1, ray_buf, kind, report)
                ev['c1'].record()
                if os.environ.get('SOSFUSE2_WF_DUMP') and phase == 0 and it in (1, 2) and ci == 0:
                    # Save one chunk's slimray inputs and ray radiances.
                    tag = f'chunk0_it{it}_{id(self) & 0xFFFF:04x}'
                    self.wf_dump(
                        os.environ['SOSFUSE2_WF_DUMP'],
                        tag,
                        kb,
                        ctx,
                        bundle,
                        W,
                        boundary,
                        c0,
                        c1,
                        ray_buf,
                        nray,
                        r0,
                        base,
                        npt,
                        report['gpu_seconds'] - g0,
                        dict(phase=int(phase), iteration=it, chunk=ci, kind=kind),
                    )
                if verify:
                    self.slim2_verify_rays(
                        ray_buf, dump_buf, nray * 4 * W, W, dict(phase=int(phase), iteration=it, chunk=ci, first_ray=r0)
                    )
                ev['c2'].record()
                if verify:
                    ctx['moments'][c0:c1].fill(np.nan)
                    ctx['flux'][c0:c1].fill(np.nan)
                self.slim_proj(kb, ctx, bundle, W, boundary, c0, c1, y, ray_buf, kind, report, cum)
                ev['s1'].record()
                if verify:
                    projection_checks.append(
                        (c0, c1, *(self.host(x) for x in (ctx['moments'][c0:c1], ctx['flux'][c0:c1], mref, fref)))
                    )
                    del mref, fref
            else:
                self.ksplit_cells(kind + '_slim', raw['_slim'], c0, c1, cell_part, geom, step, cum, report)
                ev['s1'].record()
            rows.append(
                (ci, c0, c1, npt, {k: plan[k] for k in ('nray', 'nrec', 'nvis', 'nblk')}, stored, wall, ev, prod, mode3)
            )
            report['chunks'] += 1
            report['max_cells'] = max(report['max_cells'], c1 - c0)
            del plan, c0cp, y, run_vprod  # stream-ordered pool reuse; a kept plan would hold ~4 B/record per chunk
        if projection_checks:
            self.slim2_verify_proj(projection_checks, dict(phase=int(phase), iteration=it))
        kb.pop('dump', None)
        if rows:
            rows[-1][7]['s1'].synchronize()
        tot = dict(
            plan_ms=0.0, y_ms=0.0, vprod_ms=0.0, slim_ms=0.0, cons_ms=0.0, proj_ms=0.0, prod_ms=0.0, plan_wall_s=0.0
        )
        for ci, c0, c1, npt, plan, stored, wall, ev, prod, m3 in rows:
            ms = lambda a, b: float(xp.cuda.get_elapsed_time(ev[a], ev[b]))
            row = dict(
                event='vprod_chunk',
                phase=int(phase),
                iteration=it,
                chunk=ci,
                cells=[int(c0), int(c1)],
                points=npt,
                rays=plan['nray'],
                records=plan['nrec'],
                visitors=plan['nvis'],
                blocks=plan['nblk'],
                stored=bool(stored),
                plan_ms=ms('p0', 'y0'),
                y_ms=ms('y0', 'p1'),
                plan_wall_s=wall,
                vprod_ms=ms('p1', 'v1'),
                slim_ms=ms('w1', 's1'),
            )
            if verify:
                row['prod_ms'] = sum(float(xp.cuda.get_elapsed_time(e0, e1)) for e0, e1 in prod)
            if m3:  # consumer + projection (the VERIFY ray dump SLIM and the comparisons are outside both)
                row.update(cons_ms=ms('w2', 'c1'), proj_ms=ms('c2', 's1'))
                row['slim_ms'] = row['cons_ms'] + row['proj_ms']
            for k in tot:
                tot[k] += row.get(k, 0.0)
            print('KSPLIT ' + json.dumps(row), flush=True)
        print(
            'KSPLIT '
            + json.dumps(
                dict(
                    event='vprod_launch',
                    phase=int(phase),
                    iteration=it,
                    chunks=len(rows),
                    verify=bool(verify),
                    resident=kb['resident'],
                    store=bool(kb['store']),
                    stored_plans=len(kb['plans']),
                    stored_bytes=int(kb['stored_bytes']),
                    aero_bytes=int(aero.nbytes),
                    device_used_bytes_sampled=used_max,
                    pool_total_bytes_sampled=pool_max,
                    **tot,
                )
            ),
            flush=True,
        )

    def verify_result(self, event, result, where):
        info = dict(where, **log_summary(result))
        print('KSPLIT ' + json.dumps(dict(event=event, ok=result['pass_'], **info)), flush=True)
        if not result['pass_']:
            raise RuntimeError('E_KSPLIT_VERIFY: ' + event + ': ' + json.dumps(info))

    def vprod_verify(self, lo, hi, W, where, valid_points, pc, aw=None):
        """Aero scale: all valid point/corner records of one target cell x wavelength. aw = record row stride (aero_w(W); the
        pad lanes >= W are dropped before comparing).
        Windows/chunks contain whole target cells; stage pieces cannot change S_g.
        c0cp metadata, not observed writes, defines the consumed slots.
        """
        r, alpha = self._verify_tolerances
        wa, wb = where['window']
        base = where['base']
        p0, p1 = int(pc[wa]) - base, int(pc[wb]) - base
        aw = aw or aero_w(W)
        start, end = p0 * 4 * aw, p1 * 4 * aw
        actual = np.ascontiguousarray(self._vprod_snap[start - lo : end - lo].reshape(-1, 4, aw)[:, :, :W])
        ref = np.ascontiguousarray(self.host(self.ksplit_aero[start:end]).reshape(-1, 4, aw)[:, :, :W])

        # Metadata is generated only for each arithmetic piece, never for all
        # point/corner/wavelength slots in the window.
        def valid(i, j):
            return valid_points[p0 + np.arange(i, j) // (4 * W)]

        def groups(i, j):
            slots = np.arange(i, j)
            cells = np.searchsorted(pc, int(pc[wa]) + slots // (4 * W), side='right') - 1
            return cells * W + slots % W

        result = compare_tol(actual, ref, valid, groups, r, alpha, details=False, stage_size=VPROD_VERIFY_STAGE)
        info = dict(where, cells=wb - wa)
        if result.get('peak') is not None:
            point, corner, wave = result['peak']
            info.update(max_cell=int(np.searchsorted(pc, int(pc[wa]) + point, side='right') - 1), max_wavelength=wave)
        if result['first'] is not None:
            point, corner, wave = result['first']
            info.update(
                cell=int(np.searchsorted(pc, int(pc[wa]) + point, side='right') - 1),
                point=int(pc[wa]) + point,
                corner=corner,
                wavelength=wave,
            )
        self.verify_result('verify_aero', result, info)

    def slim2_verify_rays(self, rays, dump, n, W, where):
        """Ray x wavelength groups include all four components, independent of launch chunks."""
        a = self.host(rays[:n]).reshape(-1, W, 4)
        b = self.host(dump[:n]).reshape(-1, W, 4)
        groups = (np.arange(a.shape[0])[:, None, None] + where['first_ray']) * W + np.arange(W)[None, :, None]
        result = compare_tol(a, b, True, groups, *self._verify_tolerances)
        if result['first'] is not None:
            ray, wave, comp = result['first']
            where = dict(where, ray=where['first_ray'] + ray, wavelength=wave, component=comp)
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
                        info.update(
                            chunk_cells=[chunk[0], chunk[1]],
                            cell=chunk[0] + int(pos[0]),
                            slot=[int(x) for x in pos[1:]],
                        )
                        break
                    index -= b.size
            self.verify_result('verify_proj', result, info)
