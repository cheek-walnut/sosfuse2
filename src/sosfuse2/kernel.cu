// All current accumulations are deterministic; add nondeterministic atomics only under #if !SOSFUSE2_DETERMINISTIC.
#ifdef SLIM
#define SLIM_ON 1
#else
#define SLIM_ON 0
#endif
// R1 split (fused.py KSPLIT, results/kdiag_20261025): PROD (producer) keeps geometry, tag cache, SH contraction and
// m-reduction, writes own_aero per corner, and compiles out the tensor/log/attenuation/Kahan/boundary/projection and
// moment/flux epilogue (its outputs are not meaningful); SLIM (consumer) reads own_aero instead (see SINV below).
#ifdef PROD
#define PROD_ON 1
#else
#define PROD_ON 0
#endif
// RECORD (light column record pass, SINV mode 1): no SH contraction (as SLIM), no tensor/boundary/projection/epilogue
// (as PROD); only geometry, the Simpson prefix and the column write. Same column values as the full kernel's mode 1.
#ifdef RECORD
#define REC_ON 1
#else
#define REC_ON 0
#endif
#define NOSH_ON (SLIM_ON || REC_ON)
#define NOOUT_ON (PROD_ON || REC_ON)
#ifdef SINV
// R1 column records (sinvC, 20 B/point, bundle-global [point][5]): [0] ds, [1..4] optical columns Q0..Q3 with
// tau_w = sum_k ext_k[w] Q_k. Mode 1 records them (host swaps in unit-basis extinction tables, so lanes 0..3 integrate
// the four profile columns); mode 2 rebuilds the attenuation from them instead of the Simpson prefix.
#define SINV_F 5
#define SINV_REC(step_) (sinv_rec+((int64_t)(seg-ray)*nq+(step_))*SINV_F)
#define SINV_SIMPSON if(sinv_mode!=2)
// SLIM consumer: compiles out the tag cache, SH contraction and m-reduction and reads the corner aerosol value the
// PROD launch of the same chunk wrote. Header (query buffer): int32 [0] mode (1 record columns, 2 use columns),
// int64 [1] aerosol offset in floats from query, int64 [2] first point of the chunk (aerosol is chunk-local,
// [point-base][corner][w]); column records (SINV_REC) are bundle-global.
#define SLIM_AERO(corner_) (slim_aero+((((int64_t)(seg-ray)*nq+step-slim_base)*4+(corner_))*AERO_W))
#define SINV_AND_SIMPSON && sinv_mode!=2
#else
#define SINV_SIMPSON
#define SINV_AND_SIMPSON
#endif
// NVRTC, CUDA 12. FW is specialized to 1 or 15; no fast-math or implicit FMA.
// One destination cell (and optional spectral tile)/block, one ray/warp. No global atomics.
// NVRTC: math builtins available without headers
typedef unsigned int uint32_t; typedef unsigned short uint16_t; typedef long long int64_t; typedef int int32_t; typedef unsigned char uint8_t;
#ifndef FW
#define FW 15
#endif
// Aerosol record row stride: FW padded to a multiple of 4 floats (16 for FW 15) so VPROD stores one float4 per thread
// (full 32 B sectors; the FW-stride scalar rows were partial-sector writes: results/wavefront_20261101/vprod_gemm, 2.6x on a 5090,
// bit-identical values). The pad lane is written as 0 by VPROD, never written by PROD and never read.
#define AERO_W (((FW+3)/4)*4)
// SLIM consumer on the RS2 path (the only one that reads SLIM_AERO): every corner ends in a ballot and __syncwarp, so
// the compiler cannot hoist the next corner's global reads and a step waits for 4+ serial L2 round trips (cq 1212-1214:
// L1TEX long_scoreboard ~41 % of warp stalls, no local spills). SLIM_PRELOAD issues a step's PROD aerosol, cell source
// rows and column record before the corner loop; the arithmetic and its order are unchanged.
#if defined(SINV) && defined(SLIM) && defined(STRATO) && FW==15 && !defined(LUT_ONLY)
#define SLIM_PRELOAD
#endif
// Phase-0 first light (FP64 pipe ~76 % after the p/d DS cache, results/kclean_20261029/fp64): point() and the Simpson
// nodes take the segment ends and quadrature nodes as DS computed once (segment) or held per lane (nodes, nq<=32)
// instead of converting the same doubles every step; the DS values equal ds() of them, so results are unchanged.
// Warm kernels do the same (slimray FP64 pipe 39 % at 926b06e, cq 1239): the ray's p/d DS cache, segment-end DS and
// lane-held node DS feed the batched point()/point_m2 and the Simpson prefix.
#if !defined(LUT_ONLY) && (!K_FIRST || K_PHASE==0)
#define PATH_DS
#endif
#define K 561
#define PI 3.14159265358979323846
// Smallest normal float (FLT_MIN). Traces below it take the linear path; NaN fails >= as well. The trace
// itself compares the same under FTZ, but subnormal terms inside it do not: CuPy always builds with
// -ftz=true, so reference.py runs transport under flush_to_zero() to match (taskB_simpson_20261022/REPORT_2L.md).
#define TRACE_MIN 1.17549435e-38f
// Transport defaults are isolated from the historical LUT translation unit.
#ifndef LUT_ONLY
// Specialized transport always uses paired SH sidecars, lane attenuation/ground,
// wavelength-owned transport and the existing accumulation order. The former C3
// domain is STRATO; FW1 omits its four FW15-only optimizations. No nostrato expansion.
#if !K_FIRST
#define WARM_LANE_ON
#endif
// First-light/ray-split moments, and FW15 strato warm moments, live in global rows.
#if (K_FIRST || defined(WARM_RAYSPLIT)) && K_PHASE==0
#define ACC_IN_GLOBAL
#endif
#if defined(STRATO) && FW==15 && !K_FIRST && K_PHASE==0
#define ACC_IN_GLOBAL
#endif
// Batched point geometry: lane=step, the original point arithmetic and warp copies.
#if defined(STRATO) && !K_FIRST
#define PT_ACTIVE
#define PT_NB 16
#endif
#if K_FIRST && K_PHASE==0
#define SOS_LAUNCH_BOUNDS __launch_bounds__(256,2)
#endif
// Thin-layer masking uses the FW15 reduce-scatter path only.
#if defined(STRATO) && FW==15 && !K_FIRST
#define SKIP_ACTIVE
#ifndef SKIP_TAU
#define SKIP_TAU 1e-7f
#endif
#endif
#if defined(PT_ACTIVE) && !(defined(WARM_RAYSPLIT) && K_PHASE==0)
#define RS_ACTIVE
#endif
// Longest cells first; FW15 strato also uses the original two-part cell reduction.
#if !K_FIRST && !(defined(WARM_RAYSPLIT) && K_PHASE==0)
#define LPT_ACTIVE
#endif
#if defined(LPT_ACTIVE) && defined(STRATO) && FW==15
#define SPLIT_ACTIVE
#define SPLIT_S 2
#endif
#if defined(LPT_ACTIVE) && defined(SPLIT_ACTIVE)
#define LPT_PARAM ,const int*cell_order,double*split_scratch,double*split_flux,int*split_count
#elif defined(LPT_ACTIVE)
#define LPT_PARAM ,const int*cell_order
#else
#define LPT_PARAM
#endif
// First-light phases 0/2 share the existing tropo+strato ground loop.
#if defined(STRATO) && K_FIRST && (K_PHASE==0 || K_PHASE==2)
#define GF_ON
#endif
// Instrumentation uses unsplit transport; warm split is an internal consumer role.
#if defined(WARM_RAYSPLIT) && defined(EXP_SECTION_CLOCKS)
#error E_FL_RAYSPLIT_REQUIRES_ACC_GLOBAL_NO_CLOCKS
#endif
#if !defined(EXP_SECTION_CLOCKS) && K_FIRST && K_PHASE==0
#define FL_SPLIT_ON
#define SPLIT_PARAM ,float*ray_rads,int*ray_sync
#elif defined(WARM_RAYSPLIT) && !K_FIRST && K_PHASE==0
#define FL_SPLIT_ON
#define SPLIT_PARAM ,float*ray_rads,int*ray_sync
#else
#define SPLIT_PARAM
#endif
#else // LUT_ONLY: retain the original unspecialized fallback translation unit.
#define LPT_PARAM
#define SPLIT_PARAM
#endif
#ifndef SOS_LAUNCH_BOUNDS
#define SOS_LAUNCH_BOUNDS
#endif
#ifndef GFG
#define GFG 8
#endif
#ifdef EXP_SECTION_CLOCKS
#define SECTION_PARAM ,unsigned long long*section_clocks
#define SECTION_BEGIN() if(lane==0){section_begin=clock64();}
#define SECTION_BEGIN_IF(cond) if(lane==0 && (cond)){section_begin=clock64();}
#define SECTION_END(k) if(lane==0){section_cycles[k]+=clock64()-section_begin;}
#define SECTION_END_IF(k,cond) if(lane==0 && (cond)){section_cycles[k]+=clock64()-section_begin;}
#define SECTION_SYNC(op) do { unsigned long long barrier_begin=0; \
 if(lane==0)barrier_begin=clock64(); op; \
 if(lane==0)section_cycles[6]+=clock64()-barrier_begin; } while(0)
#else
#define SECTION_PARAM
#define SECTION_BEGIN()
#define SECTION_BEGIN_IF(cond)
#define SECTION_END(k)
#define SECTION_END_IF(k,cond)
#define SECTION_SYNC(op) op
#endif
#ifndef LUT_ONLY
#define P_FIRST first_arg
#define P_PHASE phase_arg
#define P_BOUNDARY boundary_arg
#else
#define P_FIRST first
#define P_PHASE phase
#define P_BOUNDARY with_boundary
#endif
#ifndef LUT_ONLY
#define KPAD 576
#define SH_PARAM ,const float*src_sh
#else
#define SH_PARAM
#endif
#ifndef LUT_ONLY
struct LMTable { unsigned char l[K],m[K];
 constexpr LMTable():l(),m(){int k=0;for(int a=0;a<=32;a++)for(int b=0;b<=a;b++){l[k]=(unsigned char)a;m[k]=(unsigned char)b;k++;}} };
__constant__ LMTable LM=LMTable();
#endif
#ifndef LUT_ONLY
#define PROFILE_PARAM ,const float*profile_table,int profile_mask
#else
#define PROFILE_PARAM
#endif
#ifdef STRATO
#define STRATO_PARAM ,const float*st_ext,const float*st_sca,const float*st_chi,const float*st_density,const float*st_sun_col,const float*st_end_col,float st_center,float st_sigma,float st_norm
#define COPY_STRATO(DST,SRC) do { _Pragma("unroll") for(int cw=0;cw<TW;cw++)scache[DST][cw]=scache[SRC][cw];if(lane==0){ _Pragma("unroll") for(int cw=0;cw<TW;cw++)s.st_c32[DST][cw]=s.st_c32[SRC][cw]; } } while(0);
#else
#define STRATO_PARAM
#define COPY_STRATO(DST,SRC)
#endif
// Opt-in spectral tiles: local buffers use TW; global arrays keep FW.
#define TW FW
#define GW(w) w
// P3' (results/kclean_20261029/p3): slimray handles two steps per warp, step+h in half h = lane>>4, wavelength lane&15.
// Elsewhere the macros expand to the one-step forms.
#ifdef SLIM_RAYS
#define SR_INC 2
#define SR_ACT sr_act
#define SR_W sr_w
#define SR_J sj
#define SR_BIT(w_) (1u<<lane)
#define SR_POS0 (((1u<<TW)-1u)|(((1u<<TW)-1u)<<16))
#else
#define SR_INC 1
#define SR_ACT (lane<TW)
#define SR_W lane
#define SR_J step
#define SR_BIT(w_) (1u<<(w_))
#define SR_POS0 ((1u<<TW)-1u)
#endif
#define SLIM_AERO_J(j_,corner_) (slim_aero+((((int64_t)(seg-ray)*nq+(j_)-slim_base)*4+(corner_))*AERO_W))
// PATH_DS kernels (phase-0 first light, warm) reuse exact two-word ray coordinates.
#ifdef PATH_DS
#define RAY_CONST_F 27
#else
#define RAY_CONST_F 15
#endif
// Fixed-index copies preserve register promotion; no dynamic cache-bank access.
#define COPY_CORNER(DST,SRC) do { \
 _Pragma("unroll") for(int cw=0;cw<TW;cw++)cache[DST][cw]=cache[SRC][cw]; \
 if(lane==0){_Pragma("unroll") for(int cw=0;cw<TW;cw++)s.c32[DST][cw]=s.c32[SRC][cw];} \
 COPY_STRATO(DST,SRC) \
 tags[DST]=s.cell[DST]; \
} while(0)
struct DS { float h,l; };
__device__ DS ds(double x){float h=(float)x;return {h,(float)(x-(double)h)};}
__device__ DS ds(DS x){return x;}
__device__ DS dn(DS a){return {-a.h,-a.l};}
__device__ DS da(DS a,DS b){float s=a.h+b.h,v=s-a.h;
 float e=((a.h-(s-v))+(b.h-v))+(a.l+b.l),h=s+e;return {h,e-(h-s)};}
__device__ DS dx(DS a,DS b){return da(a,dn(b));}
__device__ DS dm(DS a,DS b){float p=a.h*b.h,e=fmaf(a.h,b.h,-p);
 e=(e+a.h*b.l)+a.l*b.h;float h=p+e;return {h,e-(h-p)};}
__device__ DS dr(DS a){float h=sqrtf(fmaxf(a.h,0.f));DS e=dx(a,dm({h,0},{h,0}));
 float l=(e.h+e.l)/fmaxf(2*h,1e-30f);return da({h,0},{l,0});}
__device__ DS ddiv(DS a,DS b){float h=a.h/b.h;DS e=dx(a,dm({h,0},b));float lo=(e.h+e.l)/b.h;return da({h,0},{lo,0});}
__device__ float dq(DS a,DS b){DS x=ddiv(a,b);return x.h+x.l;}
__device__ float dv(DS a){return a.h+a.l;}
// FTZ-B (gate v2 2026-10-21, results/gate_redesign_20261021/SPEC_V2.md 9): a solar-boundary DS value as an FTZ FP32 build
// computes it, on every build: components and sum below FLT_MIN count as 0; NaN passes (fabsf(NaN)<TRACE_MIN is false).
__device__ float ftz_dv(DS a){float h=fabsf(a.h)<TRACE_MIN?0.f:a.h,l=fabsf(a.l)<TRACE_MIN?0.f:a.l,v=h+l;return fabsf(v)<TRACE_MIN?0.f:v;}
__device__ float clip(float x,float lo,float hi){return fminf(hi,fmaxf(lo,x));}
__device__ int search(const float*a,int n,float x){int lo=0,hi=n;while(lo<hi){int m=(lo+hi)/2;if(a[m]<=x)lo=m+1;else hi=m;}return max(0,min(n-2,lo-1));}
#ifndef LUT_ONLY
// ②b (results/kclean_20261029/SPEC_codex_2b.md): PROFILE nodes 0..80 km at 0.5/0.25/0.125 km = 161/321/641 rows. The
// host passes -DPROFILE_N only for 321/641 (fused.Kernel profile_step_km); the default expands to the same literals.
#ifndef PROFILE_N
#define PROFILE_N 161
#endif
#if PROFILE_N==161
#define PROFILE_INV 2.f
#define PROFILE_LAST 159
#elif PROFILE_N==321
#define PROFILE_INV 4.f
#define PROFILE_LAST 319
#elif PROFILE_N==641
#define PROFILE_INV 8.f
#define PROFILE_LAST 639
#else
#error "PROFILE_N must be 161, 321 or 641 (0.5, 0.25, 0.125 km)"
#endif
__device__ float profile_value(float h,const float*table,int k){
 float x=clip(h,0.f,80.f)*PROFILE_INV;int i=min((int)x,PROFILE_LAST);float f=x-(float)i;
 return table[i*2+k]+f*(table[(i+1)*2+k]-table[i*2+k]);
}
#endif
#ifdef AEROSOL_2L
// Two-layer tropospheric aerosol (tropo_profile.py, opt-in by state; without -DAEROSOL_2L nothing here exists and
// s.prof[1] stays expf(-s.h/height)). P = [bl_top-w/2, 1/w, a_bl, k_bl, a_ft, k_ft, bl_top] follows sun_h[0..nsh) in
// the same device allocation (fused.constants_device), so the kernel signature and launch argument lists do not change.
// Smoothstep blend of the boundary layer into the free troposphere over the width w.
__device__ __forceinline__ float aerosol_two_layer(float h,const float*P){
 float x=clip((h-P[0])*P[1],0.f,1.f),s=x*x*(3.f-2.f*x);
 return (P[2]*expf(-h*P[3]))*(1.f-s)+(P[4]*expf(-(fmaxf(h,P[0])-P[6])*P[5]))*s;
}
#endif
__device__ int ix(int l,int m){return l*(l+1)/2+m;}
__device__ float hg(float u,float g){float den=fmaxf(1e-20f,(1.f+g*g)-(2.f*g)*u);return (1.f-g*g)/((float)(4*PI)*(den*sqrtf(den)));}
#ifdef STRATO
#define STRATO_LUT_BINS 16384
// GPU nodes use exactly the old recurrence, even in a LUT-enabled module.
__device__ float strato_phase_node(float u,int w,const float*chi){
 float p0=1.f,p1=u,v=p0*chi[w];v+=p1*chi[FW+w];
 for(int l=2;l<=32;l++){float pn=((float)(2*l-1)*u*p1-(float)(l-1)*p0)/(float)l;v+=pn*chi[l*FW+w];p0=p1;p1=pn;}
 return v/(float)(4*PI);
}
extern "C" __global__ void strato_lut_build(float*chi){
 int k=(int)blockIdx.x*(int)blockDim.x+(int)threadIdx.x;
 if(k<(STRATO_LUT_BINS+1)*FW){int i=k/FW,w=k%FW;
  float u=-1.f+(float)i*(2.f/(float)STRATO_LUT_BINS);
  chi[33*FW+k]=strato_phase_node(u,w,chi);
 }
}
__device__ float mie_phase(float u,int w,const float*chi){
 float x=(clip(u,-1.f,1.f)+1.f)*(float)(STRATO_LUT_BINS/2);
 int i=min((int)x,STRATO_LUT_BINS-1);float f=x-(float)i;
 float a=chi[(33+i)*FW+w],b=chi[(34+i)*FW+w];
 return a+f*(b-a);
}
#endif
__device__ void kah(float &t,float &c,float v){float y=v-c,z=t+y;c=(z-t)-y;t=z;}
#if defined(PT_ACTIVE)
struct PointData {
 float T[16],cm[33],sw[4],prof[3],h,ds,cs;
 float mu,exd,eyd,sc[3];int cell[4],ir,lit;
#ifdef STRATO
 float st_prof,st_sc;
#endif
};
struct Sample : PointData {
#ifdef STRATO
 float st_c32[4][TW];
#endif
 float val[TW][4],logs[TW];uint32_t positive;float c32[4][TW],att[TW];
};
static_assert(sizeof(PointData)==284
#ifdef STRATO
 +8
#endif
 ,"E_POINTDATA_LAYOUT");
static_assert(sizeof(Sample)==288+40*TW
#ifdef STRATO
 +8+16*TW
#endif
 ,"E_SAMPLE_LAYOUT");
// Copy only fields actually written by point(). Inactive first-light fields
// are deliberately not read; persistent c32/cache/val/log/att never move.
__device__ __forceinline__ void copy_point(Sample &s,const PointData &q,bool first){
 for(int k=0;k<16;k++)s.T[k]=q.T[k];
 for(int k=0;k<33;k++)s.cm[k]=q.cm[k];
 for(int k=0;k<4;k++){s.sw[k]=q.sw[k];s.cell[k]=q.cell[k];}
 for(int k=0;k<3;k++)s.prof[k]=q.prof[k];
 s.h=q.h;s.ds=q.ds;s.cs=q.cs;s.ir=q.ir;
 if(first){s.mu=q.mu;s.exd=q.exd;s.eyd=q.eyd;s.lit=q.lit;
  for(int k=0;k<3;k++)s.sc[k]=q.sc[k];}
#ifdef STRATO
 s.st_prof=q.st_prof;if(first)s.st_sc=q.st_sc;
#endif
}
#else
struct Sample {
 float T[16],cm[33],sw[4],prof[3],h,ds,cs;
 float mu,exd,eyd,sc[3];int cell[4],ir,lit;
 #ifdef STRATO
 float st_prof,st_sc,st_c32[4][TW];
#endif
 float val[TW][4],logs[TW];uint32_t positive;float c32[4][TW],att[TW];
};
#endif
#ifdef SOSFUSE2_FASTMATH
#define expf __expf
#define sinf __sinf
#define cosf __cosf
#endif
__device__ void sh_basis(double mu,int lane,float*y,const double*aa,const double*bb,const double*dd){
 mu=fmin(1.,fmax(-1.,mu));double st=sqrt(fmax(0.,1-mu*mu));
 for(int m=lane;m<=32;m+=32){
  double diag=1/sqrt(4*PI);for(int j=1;j<=m;j++)diag=(diag*dd[j])*st;
  double mult=m?sqrt(2.):1.,z0=diag;y[ix(m,m)]=(float)(mult*z0);
  if(m<32){double z1=(sqrt(2.*m+3)*mu)*z0;y[ix(m+1,m)]=(float)(mult*z1);
   for(int l=m+2;l<=32;l++){int k=ix(l,m);double z2=(aa[k]*mu)*z1-bb[k]*z0;
    y[k]=(float)(mult*z2);z0=z1;z1=z2;}
  }
 }
}
// sos_vprod_y: sh_basis's recurrence in double-single (DS, FMA products, ~1e-14 relative) instead of FP64, whose rate on
// consumer GPUs is 1/64 of FP32. aa2/bb2/dd2 are the host's (hi, lo) float pairs of aa/bb/dd, so the loop has no FP64.
// The float rows agree with sh_basis's to about 1 ulp (FP32 alone loses up to 3.5e-3 near |mu|=1 and 4.7e-6 rms with an
// FP64 st, above the 1e-6 prep-product contract; results/kclean_20261029/vprod_sort).
__device__ void sh_basis_ds(double mu,int lane,float*y,const float2*aa2,const float2*bb2,const float2*dd2){
 mu=fmin(1.,fmax(-1.,mu));const DS u=ds(mu),st=ds(sqrt(fmax(0.,1-mu*mu)));
 for(int m=lane;m<=32;m+=32){
  DS diag=ds(1/sqrt(4*PI));for(int j=1;j<=m;j++)diag=dm(dm(diag,DS{dd2[j].x,dd2[j].y}),st);
  const DS mult=m?ds(sqrt(2.)):DS{1.f,0.f};DS z0=diag;y[ix(m,m)]=dv(dm(mult,z0));
  if(m<32){DS z1=dm(dm(ds(sqrt(2.*m+3)),u),z0);y[ix(m+1,m)]=dv(dm(mult,z1));
   for(int l=m+2;l<=32;l++){int k=ix(l,m);
    DS z2=dx(dm(dm(DS{aa2[k].x,aa2[k].y},u),z1),dm(DS{bb2[k].x,bb2[k].y},z0));
    y[k]=dv(dm(mult,z2));z0=z1;z1=z2;}
  }
 }
}
// Task B (1) (2026-10-22, results/taskB_simpson_20261022/SPEC_IMPL.md): the profile is evaluated by one function, profile_at, for
// the quadrature nodes (point()) and the Simpson optical-depth prefix (segment ends, panel midpoints); reference.py profile_at,
// extinction, SimpsonPrefix use the same operations. Path coordinates, panel widths and midpoints are DS, like point().
#ifndef LUT_ONLY
#define PROFILE_ARG ,profile_table,profile_mask
#else
#define PROFILE_ARG
#endif
#ifdef STRATO
#define SP_ST_PARAM ,float st_center,float st_sigma,float st_norm
#define SP_ST_ARG ,st_center,st_sigma,st_norm
#define SP_EXT_PARAM ,const float*st_ext
#define SP_EXT_ARG ,st_ext
#else
#define SP_ST_PARAM
#define SP_ST_ARG
#define SP_EXT_PARAM
#define SP_EXT_ARG
#endif
// PROF_*: profile arguments (PROF_ARG at point()/kernel scope, PROF_FWD inside a function taking PROF_PARAM); BETA_* add ext.
#define PROF_PARAM float height,const float*aer2 PROFILE_PARAM SP_ST_PARAM
#define PROF_ARG height,sun_h+nsh PROFILE_ARG SP_ST_ARG
#define PROF_FWD height,aer2 PROFILE_ARG SP_ST_ARG
#define BETA_PARAM PROF_PARAM,const float*ext SP_EXT_PARAM
#define BETA_ARG PROF_ARG,ext SP_EXT_ARG
#define BETA_FWD PROF_FWD,ext SP_EXT_ARG
// Path coordinate of a GL node (u^2 map toward the lower end), DS.
__device__ __forceinline__ DS path_t(DS ad,DS bd,DS length,DS u,DS v,bool lower){
 return lower?da(ad,dm(length,dm(u,u))):dx(bd,dm(length,dm(v,v)));
}
template<class Scalar>
__device__ __forceinline__ DS node_t(Scalar a,Scalar b,bool lower,Scalar gu){
 DS ad=ds(a),bd=ds(b),u=ds(gu);return path_t(ad,bd,dx(bd,ad),u,lower?u:dx({1,0},u),lower);
}
// DS position q = p + t d, q*q and the radius |q|.
template<class Coord>
__device__ __forceinline__ DS path_q(const Coord*p,const Coord*d,DS t,DS*q,DS*q2){
 for(int k=0;k<3;k++){q[k]=da(ds(p[k]),dm(t,ds(d[k])));q2[k]=dm(q[k],q[k]);}
 return dr(da(da(q2[0],q2[1]),q2[2]));
}
__device__ __forceinline__ float path_h(DS rr){return clip(dv(dx(rr,{6371,0})),0,80);}
// Profile components at height h: [0] air, [1] tropospheric aerosol, [2] ozone, [3] stratospheric aerosol (0 without STRATO).
__device__ __forceinline__ void profile_at(float h,PROF_PARAM,float*prof){
 prof[0]=expf(-h/8.f);prof[1]=expf(-h/height);prof[2]=fmaxf(0.f,1.f-fabsf(h-25.f)/15.f);
#ifdef AEROSOL_2L
 prof[1]=aerosol_two_layer(h,aer2);
#endif
#ifndef LUT_ONLY
 if(profile_mask&1)prof[0]=profile_value(h,profile_table,0);
 if(profile_mask&2)prof[2]=profile_value(h,profile_table,1);
#endif
#ifdef STRATO
 float st_t=(h-st_center)/st_sigma;prof[3]=expf(-.5f*(st_t*st_t))/st_norm;
#else
 prof[3]=0.f;
#endif
}
// Extinction of wavelength column gw from profile components (sum_k prof_k ext_k, then the stratospheric layer).
__device__ __forceinline__ float extinction_at(float p0,float p1,float p2,float p3,const float*ext SP_EXT_PARAM,int gw){
 float b=0.f;b+=p0*ext[gw];b+=p1*ext[FW+gw];b+=p2*ext[2*FW+gw];
#ifdef STRATO
 b+=p3*st_ext[gw];
#endif
 return b;
}
template<class Coord>
__device__ __forceinline__ float beta_at(const Coord*p,const Coord*d,DS t,BETA_PARAM,int gw){
 DS q[3],q2[3];float pr[4];profile_at(path_h(path_q(p,d,t,q,q2)),PROF_FWD,pr);
 return extinction_at(pr[0],pr[1],pr[2],pr[3],ext SP_EXT_ARG,gw);
}
// Simpson on the panel [t0,t1] with end extinctions b0,b1: (t1-t0)/6 (b0 + 4 beta(mid) + b1), mid = (t0+t1)/2 (exact DS halving).
template<class Coord>
__device__ __forceinline__ float simpson_piece(const Coord*p,const Coord*d,DS t0,DS t1,float b0,float b1,BETA_PARAM,int gw){
 DS tm=da(t0,t1);tm.h*=.5f;tm.l*=.5f;
 const float bm=beta_at(p,d,tm,BETA_FWD,gw);
 return (dv(dx(t1,t0))/6.f)*((b0+b1)+4.f*bm);
}
#ifndef LUT_ONLY
#define POINT_INLINE __noinline__
#else
#define POINT_INLINE __forceinline__
#endif
// Coord/Scalar = double (every caller but phase-0 first light) or DS: first light passes the ray's DS p/d and the segment
// and node DS values, so the per-step ds() conversions (FP64 adds, the remaining first-light FP64 work) are not redone.
// pf(d[k]) is (float)d[k] = ds(d[k]).h either way; the values are identical.
__device__ __forceinline__ float pf(double x){return (float)x;}
__device__ __forceinline__ float pf(DS x){return x.h;}
template<class Coord,class Scalar>
#if defined(PT_ACTIVE)
__device__ POINT_INLINE void point(PointData &s,const Coord*p
#else
__device__ POINT_INLINE void point(Sample &s,const Coord*p
#endif
 ,const Coord*d,const float*B,const float*dl,float st,
 Scalar a,Scalar b,bool lower,Scalar gu,Scalar gw,const float*hgrid,int nh,const float*chi,int nc,float height,
 const float*sun_h,int nsh,const float*sun_col,int nsx,bool first PROFILE_PARAM
#ifdef STRATO
 ,const float*st_sun_col,float st_center,float st_sigma,float st_norm
#endif
 ){
 DS ad=ds(a),bd=ds(b),length=dx(bd,ad),u=ds(gu),wu=ds(gw),v=lower?u:dx({1,0},u);
 DS t=path_t(ad,bd,length,u,v,lower);
 s.ds=dv(dm(dm(length,v),wu))*2.f;
 DS q[3],q2[3];DS rr=path_q(p,d,t,q,q2),xy=dr(da(q2[0],q2[1]));
 s.h=path_h(rr);float rho=dv(xy);
 float cx=rho>1e-12f?dq(q[0],xy):1.f,sx=rho>1e-12f?dq(q[1],xy):0.f;
 s.cs=clip(dq(q[2],rr),-1,1);float angle=atan2f(rho,dv(q[2]));
 int ir=search(hgrid,nh,s.h),j=search(chi,nc,angle);s.ir=ir;
 float ah=clip((s.h-hgrid[ir])/(hgrid[ir+1]-hgrid[ir]),0,1),ac=clip((angle-chi[j])/(chi[j+1]-chi[j]),0,1);
 angle=chi[j]+ac*(chi[j+1]-chi[j]);float sn=sinf(angle),cc=cosf(angle);
 float Q[9]={-cc*cx,sx,sn*cx,-cc*sx,-cx,sn*sx,sn,0,cc},A[9]={},dot[3]={},kk[9];
 for(int k=0;k<3;k++)for(int col=0;col<3;col++){
  dot[col]+=pf(d[k])*Q[3*k+col];
  for(int row=0;row<3;row++)A[3*row+col]+=B[3*k+row]*Q[3*k+col];
 }
 for(int row=0;row<3;row++)for(int col=0;col<3;col++)kk[3*row+col]=A[3*row+col]-dl[row]*dot[col];
 const int ii[4]={0,1,2,0},jj[4]={0,1,2,2};
 for(int u2=0;u2<4;u2++){int i=ii[u2],z=jj[u2];
  s.T[4*u2]=kk[3*i]*kk[3*z];s.T[4*u2+1]=kk[3*i+1]*kk[3*z+1];s.T[4*u2+2]=kk[3*i+2]*kk[3*z+2];
  s.T[4*u2+3]=kk[3*i]*kk[3*z+2]+kk[3*i+2]*kk[3*z];
 }
 float cp=st>1e-12f?clip((pf(d[0])*cx+pf(d[1])*sx)/st,-1,1):1.f;
 s.cm[0]=1;s.cm[1]=cp;for(int m=2;m<=32;m++)s.cm[m]=(2.f*cp)*s.cm[m-1]-s.cm[m-2];
 s.sw[0]=(1.f-ah)*(1.f-ac);s.sw[1]=(1.f-ah)*ac;s.sw[2]=ah*(1.f-ac);s.sw[3]=ah*ac;
 s.cell[0]=ir*nc+j;s.cell[1]=s.cell[0]+1;s.cell[2]=s.cell[0]+nc;s.cell[3]=s.cell[2]+1;
 {float pr[4];profile_at(s.h,PROF_ARG,pr);s.prof[0]=pr[0];s.prof[1]=pr[1];s.prof[2]=pr[2];
#ifdef STRATO
  s.st_prof=pr[3];
#endif
 }
 if(first){
  s.mu=dot[2];s.exd=dot[0];s.eyd=dot[1];
  DS height=dx(rr,{6371,0});if(ftz_dv(height)<=0.f)height={0,0};DS rad=da({6371,0},height);
  DS mg=dn(dr(ddiv(dm(height,da({2.f*6371.f,0},height)),dm(rad,rad))));
  DS delta=dx(ddiv(q[2],rr),mg);if(ftz_dv(delta)==0.f)delta={0,0};s.lit=dv(delta)>=0;if(!s.lit)delta={0,0};
  float x=clip(dv(dr(ddiv(delta,dx({1,0},mg)))),0,1);
  int ih=search(sun_h,nsh,s.h);float ar=(s.h-sun_h[ih])/(sun_h[ih+1]-sun_h[ih]);
  float f=x*(nsx-1);int ic=max(0,min(nsx-2,(int)f));float br=f-(float)ic;
  for(int k=0;k<3;k++){
   float col=((1.f-ar)*(1.f-br))*sun_col[(ih*nsx+ic)*3+k];
   col+=(ar*(1.f-br))*sun_col[((ih+1)*nsx+ic)*3+k];
   col+=((1.f-ar)*br)*sun_col[(ih*nsx+ic+1)*3+k];
   col+=(ar*br)*sun_col[((ih+1)*nsx+ic+1)*3+k];s.sc[k]=col;
  }
#ifdef STRATO
  float col=((1.f-ar)*(1.f-br))*st_sun_col[ih*nsx+ic];
  col+=(ar*(1.f-br))*st_sun_col[(ih+1)*nsx+ic];
  col+=((1.f-ar)*br)*st_sun_col[ih*nsx+ic+1];
  col+=(ar*br)*st_sun_col[(ih+1)*nsx+ic+1];s.st_sc=col;
#endif
 }
}

#if defined(SLIM_RAYS) && defined(PT_ACTIVE)
// KSPLIT=3 consumer (results/slim2_20261025): mode-2 SLIM reads only T, sw, cell and ir of a sample, so the batch keeps this
// 112 B record (PointData 292 B) and point_m2 evaluates exactly point()'s expressions for those fields (nothing else).
// 16-byte aligned (112 B) so the step loop reads T, sw and cell as 128-bit shared loads (slimray is LSU-issue-bound, cq 1252).
struct __align__(16) PointM2 { float T[16],sw[4];int cell[4],ir; };
// Coord/Scalar as point(): PATH_DS passes the ray's DS p/d, the segment-end DS and the lane's node DS.
template<class Coord,class Scalar>
__device__ POINT_INLINE void point_m2(PointM2 &s,const Coord*p,const Coord*d,const float*B,const float*dl,
 Scalar a,Scalar b,bool lower,Scalar gu,const float*hgrid,int nh,const float*chi,int nc){
 DS ad=ds(a),bd=ds(b),length=dx(bd,ad),u=ds(gu),v=lower?u:dx({1,0},u);
 DS t=path_t(ad,bd,length,u,v,lower);
 DS q[3],q2[3];DS rr=path_q(p,d,t,q,q2),xy=dr(da(q2[0],q2[1]));
 const float sh=path_h(rr);float rho=dv(xy);
 float cx=rho>1e-12f?dq(q[0],xy):1.f,sx=rho>1e-12f?dq(q[1],xy):0.f;
 float angle=atan2f(rho,dv(q[2]));
 int ir=search(hgrid,nh,sh),j=search(chi,nc,angle);s.ir=ir;
 float ah=clip((sh-hgrid[ir])/(hgrid[ir+1]-hgrid[ir]),0,1),ac=clip((angle-chi[j])/(chi[j+1]-chi[j]),0,1);
 angle=chi[j]+ac*(chi[j+1]-chi[j]);float sn=sinf(angle),cc=cosf(angle);
 float Q[9]={-cc*cx,sx,sn*cx,-cc*sx,-cx,sn*sx,sn,0,cc},A[9]={},dot[3]={},kk[9];
 for(int k=0;k<3;k++)for(int col=0;col<3;col++){
  dot[col]+=pf(d[k])*Q[3*k+col];
  for(int row=0;row<3;row++)A[3*row+col]+=B[3*k+row]*Q[3*k+col];
 }
 for(int row=0;row<3;row++)for(int col=0;col<3;col++)kk[3*row+col]=A[3*row+col]-dl[row]*dot[col];
 const int ii[4]={0,1,2,0},jj[4]={0,1,2,2};
 for(int u2=0;u2<4;u2++){int i=ii[u2],z=jj[u2];
  s.T[4*u2]=kk[3*i]*kk[3*z];s.T[4*u2+1]=kk[3*i+1]*kk[3*z+1];s.T[4*u2+2]=kk[3*i+2]*kk[3*z+2];
  s.T[4*u2+3]=kk[3*i]*kk[3*z+2]+kk[3*i+2]*kk[3*z];
 }
 s.sw[0]=(1.f-ah)*(1.f-ac);s.sw[1]=(1.f-ah)*ac;s.sw[2]=ah*(1.f-ac);s.sw[3]=ah*ac;
 s.cell[0]=ir*nc+j;s.cell[1]=s.cell[0]+1;s.cell[2]=s.cell[0]+nc;s.cell[3]=s.cell[2]+1;
}
#endif


#ifndef LUT_ONLY
__device__ __forceinline__ float2 warm_sh_pair(const float *ptr,int64_t cell,int w,int k){
 const int64_t i=(cell*((FW+1)/2)+w/2)*KPAD+k;
 return ((const float2*)ptr)[i];
}

#endif // SH_AUX_PAIR

#ifdef TROPO_MIE
// Tropospheric aerosol Mie phase (tropo_mie.py, opt-in by state; without -DTROPO_MIE nothing below exists and every
// hg(u,aerosol_g) site is unchanged). Per-wavelength table per steradian behind the asca pointer: asca[0..FW) as
// before, table node i of wavelength w at asca[FW+i*FW+w], u=-1+i*2/TROPO_LUT_BINS (strato FINE node formula).
#define TROPO_LUT_BINS 16384
#define TROPO_LUT (asca+FW)
#if defined(GF_ON) && (!defined(STRATO) || STRATO_LUT_BINS!=TROPO_LUT_BINS)
#error E_TROPO_MIE_GF_REQUIRES_STRATO_MIE_LUT_FINE
#endif
#if defined(LUT_ONLY)
#error E_TROPO_MIE_UNSUPPORTED_GROUND_PATH
#endif
__device__ __forceinline__ float tropo_phase(float u,int w,const float*lut){
 float x=(clip(u,-1.f,1.f)+1.f)*(float)(TROPO_LUT_BINS/2);
 int i=min((int)x,TROPO_LUT_BINS-1);float f=x-(float)i;
 float a=lut[i*FW+w],b=lut[(i+1)*FW+w];
 return a+f*(b-a);
}
// Paired ground phase of direction j for wavelength w: the hg site's m0/m1 expressions, Mie table instead of hg.
__device__ __forceinline__ float tropo_pair(const Sample &s,const float*gdir,int64_t j,int w,const float*lut){
 float m0=s.mu*gdir[3*j+2]+s.exd*gdir[3*j],m1=fabsf(s.eyd)*gdir[3*j+1];
 return (tropo_phase(clip(m0+m1,-1,1),w,lut)+tropo_phase(clip(m0-m1,-1,1),w,lut))/2.f;
}
#define GF_TROPO_PARAM ,const float*tr_lut
#define GF_TROPO_ARG ,TROPO_LUT
#else
#define GF_TROPO_PARAM
#define GF_TROPO_ARG
#endif

#ifdef GROUND_SH_L
#if !defined(GF_ON) || defined(TROPO_MIE)
#error E_GROUND_SH_PATH
#endif
#define GSH_SIDE (GROUND_SH_GRID+3)
// The SH convolution is synthesized once per cell on a polar/azimuth grid.
// Ghost rows/columns are synthesized too, including the polar reflection.
// Catmull-Rom interpolation retains the signed SH sum (no intensity clipping).
__device__ __forceinline__ void ground_cubic(float t,float *w){
 const float t2=t*t,t3=t2*t;
 w[0]=-.5f*t+t2-.5f*t3;w[1]=1.f-2.5f*t2+1.5f*t3;
 w[2]=.5f*t+2.f*t2-1.5f*t3;w[3]=-.5f*t2+.5f*t3;
}
__device__ __forceinline__ void ground_sh_sums(const Sample &s,int lane,const int64_t*goff,
 const float*unused,const float*coeff,float &accw,float &st_acc){
 bool any=false;for(int c=0;c<4;c++)any|=goff[s.cell[c]]!=goff[s.cell[c]+1];
 if(!any)return;
 if(lane<TW){
  const float theta=atan2f(hypotf(s.exd,s.eyd),s.mu)*(GROUND_SH_GRID/(float)PI);
  const float phi=atan2f(fabsf(s.eyd),s.exd)*(GROUND_SH_GRID/(float)PI);
  const int it=min((int)theta,GROUND_SH_GRID-1),ip=min((int)phi,GROUND_SH_GRID-1);
  float wt[4],wp[4];ground_cubic(theta-it,wt);ground_cubic(phi-ip,wp);
  for(int corner=0;corner<4;corner++){
   const int cc=s.cell[corner];if(goff[cc]==goff[cc+1])continue;
   float a=0.f,b=0.f;
   #pragma unroll
   for(int i=0;i<4;i++){
    float ar=0.f,br=0.f;
    #pragma unroll
    for(int j=0;j<4;j++){
     const int64_t row=((int64_t)cc*GSH_SIDE+it+i)*GSH_SIDE+ip+j;
     const float2 v=((const float2*)coeff)[row*FW+GW(lane)];
     ar+=wp[j]*v.x;br+=wp[j]*v.y;
    }
    a+=wt[i]*ar;b+=wt[i]*br;
   }
   accw+=s.sw[corner]*a;st_acc+=s.sw[corner]*b;
  }
 }
}
#endif

#ifdef GF_ON
// Fused tropo+strato ground sums of one sample (GROUND_FUSED); scratch gph >= 32 floats + 66 int2.
__device__ __forceinline__ void gf_sums(const Sample &s,int lane,float *gph,const int64_t*goff,const float*gdir,
  const float*gval,const float*st_chi,float aerosol_g,float &accw,float &st_acc GF_TROPO_PARAM){
 const int w=lane&15;
 const float pmu=s.mu,pexd=s.exd,paey=fabsf(s.eyd);
 int2 *gl=(int2*)(((unsigned long long)(gph+32)+7ull)&~7ull); // [k]: + node, [33+k]: - node; {li, f bits}
 for(int corner=0;corner<4;corner++){int cc=s.cell[corner];const float swc=s.sw[corner];const int64_t je=goff[cc+1];
  for(int64_t j0=goff[cc];j0<je;j0+=32){
   int64_t j=j0+lane;
   if(j<je){float m0=pmu*gdir[3*j+2]+pexd*gdir[3*j],m1=paey*gdir[3*j+1];
    float up=clip(m0+m1,-1,1),um=clip(m0-m1,-1,1);
    gph[lane]=(hg(up,aerosol_g)+hg(um,aerosol_g))/2.f;
    float xp=(clip(up,-1.f,1.f)+1.f)*(float)(STRATO_LUT_BINS/2),xm=(clip(um,-1.f,1.f)+1.f)*(float)(STRATO_LUT_BINS/2);
    int lp=min((int)xp,STRATO_LUT_BINS-1),lm=min((int)xm,STRATO_LUT_BINS-1);
    gl[lane]=make_int2(lp,__float_as_int(xp-(float)lp));gl[33+lane]=make_int2(lm,__float_as_int(xm-(float)lm));}
   __syncwarp();
   const int count=(int)((je-j0)<32?(je-j0):32);
   const int2 *gs=gl+(lane<16?0:33);const float *gq=gval+j0*FW+GW(w);
   int k=0;
   for(;k+GFG<=count;k+=GFG){float a[GFG],b[GFG],f[GFG],g[GFG];
#ifdef TROPO_MIE
    float ta[GFG],tb[GFG];
#endif
    _Pragma("unroll") for(int q=0;q<GFG;q++){const int2 e=gs[k+q];f[q]=__int_as_float(e.y);a[q]=0.f;b[q]=0.f;g[q]=0.f;
     if(w<TW){a[q]=st_chi[(33+e.x)*FW+GW(w)];b[q]=st_chi[(34+e.x)*FW+GW(w)];}
#ifdef TROPO_MIE
     ta[q]=0.f;tb[q]=0.f;
     if(w<TW){ta[q]=tr_lut[e.x*FW+GW(w)];tb[q]=tr_lut[(e.x+1)*FW+GW(w)];}
#endif
     if(lane<TW)g[q]=gq[(k+q)*FW];}
    _Pragma("unroll") for(int q=0;q<GFG;q++){float paired=0.f;if(w<TW)paired=a[q]+f[q]*(b[q]-a[q]);
     float minus=__shfl_down_sync(0xffffffff,paired,16);
#ifdef TROPO_MIE
     // Same node index and fraction as the strato LUT (TROPO_LUT_BINS==STRATO_LUT_BINS): = tropo_phase(up/um).
     float tpaired=0.f;if(w<TW)tpaired=ta[q]+f[q]*(tb[q]-ta[q]);
     float tminus=__shfl_down_sync(0xffffffff,tpaired,16);
     if(lane<TW){accw+=(swc*((tpaired+tminus)/2.f))*g[q];float ph=(paired+minus)/2.f;st_acc+=(swc*ph)*g[q];}}
#else
     if(lane<TW){accw+=(swc*gph[k+q])*g[q];float ph=(paired+minus)/2.f;st_acc+=(swc*ph)*g[q];}}
#endif
   }
   for(;k<count;k++){const int2 e=gs[k];const float fk=__int_as_float(e.y);float paired=0.f,gk=0.f;
    if(w<TW){const float ak=st_chi[(33+e.x)*FW+GW(w)],bk=st_chi[(34+e.x)*FW+GW(w)];paired=ak+fk*(bk-ak);}
    if(lane<TW)gk=gq[k*FW];
    float minus=__shfl_down_sync(0xffffffff,paired,16);
#ifdef TROPO_MIE
    float tpaired=0.f;
    if(w<TW){const float tak=tr_lut[e.x*FW+GW(w)],tbk=tr_lut[(e.x+1)*FW+GW(w)];tpaired=tak+fk*(tbk-tak);}
    float tminus=__shfl_down_sync(0xffffffff,tpaired,16);
    if(lane<TW){accw+=(swc*((tpaired+tminus)/2.f))*gk;float ph=(paired+minus)/2.f;st_acc+=(swc*ph)*gk;}}
#else
    if(lane<TW){accw+=(swc*gph[k])*gk;float ph=(paired+minus)/2.f;st_acc+=(swc*ph)*gk;}}
#endif
   __syncwarp();
  }
 }
}
#endif
// Cross-allocation buffers must be explicit pointer arguments so CUDA replay tools can see them.
#ifdef SINV
// Three explicit buffers: aerosol values, point/corner metadata and optional VERIFY ray dump.
#define SINV_PARAM ,float*slim_aero,int2*vp_records,float*slim_dump
#else
#define SINV_PARAM
#endif
extern "C" __global__ void SOS_LAUNCH_BOUNDS sos_fused(
 int cells,int nh,int nc,int nq,int nsh,int nsx,int P_FIRST,int P_PHASE,int P_BOUNDARY,
 const int64_t*cell_off,const double*starts,const double*targets,const double*dirs,const float*weights,
 const uint32_t*edge_off,const double*edges,const uint16_t*lower,
 const int32_t*end_j,const float*end_f,const float*end_col,const uint8_t*hits,
 const double*gu,const double*gw,const double*aa,const double*bb,const double*dd,
 const float*h,const float*chi,const float*hr,const float*ha,const float*ext,const float*asca,
 const float*sun_h,const float*sun_col,const float*gm,const int64_t*goff,const float*gdir,const float*gval,
 float height,float aerosol_g,float mix,double albedo,
 const float*src,const double*boundary,double*moments,double*flux,float*query SINV_PARAM SH_PARAM STRATO_PARAM PROFILE_PARAM LPT_PARAM SECTION_PARAM SPLIT_PARAM){
#ifndef LUT_ONLY
 constexpr int first=K_FIRST,phase=K_PHASE,with_boundary=K_BOUNDARY;
#endif
#ifdef SINV
#ifdef SLIM_RAYS
 constexpr int sinv_mode=2;float *sinv_rec=query+16;   // the KSPLIT=3 consumer only ever reads column records
#else
 const int sinv_mode=(!first && phase!=2)?((const int*)query)[0]:0;float *sinv_rec=query+16;
#endif
 const int64_t slim_base=((const int64_t*)query)[2];
#endif
#if defined(LUT_ONLY)
 __trap(); // E_SIMPSON_LANE_ATTEN: the Simpson prefix (task B) exists on the LANE_ATTEN path only; fused.Kernel refuses to compile it
#endif
 int cell=blockIdx.x;
#ifdef SPLIT_ACTIVE
 if(cell>=cells*SPLIT_S)return;
 const int split_unit=cell_order[cell];cell=split_unit/SPLIT_S;const int split_part=split_unit%SPLIT_S;
#else
#ifndef SLIM_RAYS
 if(cell>=cells)return;
#endif
#ifdef LPT_ACTIVE
 cell=cell_order[cell];
#endif
#endif
 int tid=threadIdx.x,lane=tid&31,warp=tid>>5,nwarps=blockDim.x>>5;
#ifdef SKIP_ACTIVE
 // Live-layer masks for up to 64 heights (V14 has 44; before 2026-10-25 only nh<=32 skipped, sh_layout.combined_coefficients).
 unsigned long long live_a=~0ull,live_s=~0ull;
 if(nh<=64){bool la=false,ls=false,la2=false,ls2=false;
  if(lane<nh){const float r=hr[lane];la=ha[lane]>SKIP_TAU*r;ls=st_density[lane]>SKIP_TAU*r;}
  if(lane+32<nh){const float r=hr[lane+32];la2=ha[lane+32]>SKIP_TAU*r;ls2=st_density[lane+32]>SKIP_TAU*r;}
  live_a=(unsigned long long)__ballot_sync(0xffffffffu,la)|((unsigned long long)__ballot_sync(0xffffffffu,la2)<<32);
  live_s=(unsigned long long)__ballot_sync(0xffffffffu,ls)|((unsigned long long)__ballot_sync(0xffffffffu,ls2)<<32);}
#endif
#ifdef EXP_SECTION_CLOCKS
 unsigned long long section_cycles[8]={},section_begin=0;
 unsigned long long total_begin=0;if(lane==0)total_begin=clock64();
#endif
 extern __shared__ double shared[];
 int nm=phase==0?565*TW:0;
#ifdef ACC_IN_GLOBAL
#ifdef SPLIT_ACTIVE
 double *acc=(split_part==0?moments+(int64_t)cell*nm:split_scratch+((int64_t)(split_part-1)*cells+cell)*nm),*ground=shared;
#else
 double *acc=moments+(int64_t)cell*nm,*ground=shared;
#endif
#else
 double *acc=shared,*ground=acc+nm;
#endif
 float *Y=(float*)(ground+TW),*rads=Y+nwarps*K,*err=rads+nwarps*4*TW,*cosdest=err+nwarps*4*TW;
 Sample *samples=(Sample*)(cosdest+nwarps*33);
#ifndef LUT_ONLY
 float *ray_constants=(float*)(samples+nwarps)+warp*RAY_CONST_F;
#endif
#ifndef LUT_ONLY
 float *ground_phase=(float*)(samples+nwarps)
#ifndef LUT_ONLY
  +nwarps*RAY_CONST_F
#endif
#if defined(GF_ON) && !defined(FL_SPLIT_ON) && K_PHASE!=2
  +warp*162; // GROUND_FUSED scratch: 32 phases + 65 int2 nodes (+1 float alignment pad)
#else
  +warp*32;
#endif
#endif
#ifdef PT_ACTIVE
#ifdef SLIM_RAYS
 typedef PointM2 PtBatch;
#else
 typedef PointData PtBatch;
#endif
 PtBatch *pts=(PtBatch*)((float*)(samples+nwarps)
#ifndef LUT_ONLY
  +nwarps*RAY_CONST_F
#endif
#ifndef LUT_ONLY
  +nwarps*32
#endif
  )+warp*PT_NB;
#ifdef SLIM_RAYS
 pts=(PtBatch*)((((unsigned long long)(pts-warp*PT_NB))+15)&~15ull)+warp*PT_NB;   // 16-byte base; fused.py adds 16 B slack
#endif
#ifdef RS_ACTIVE
 unsigned char *rs_key=(unsigned char*)(pts-warp*PT_NB+nwarps*PT_NB),*rs_perm=rs_key+256;
#endif
#endif
#ifdef FL_SPLIT_ON
 // Only an empty cell's rows are written by its own block (the unchanged zero rows); every other row
 // is written by the warp that projects that cell.
#ifndef SLIM_RAYS
 if(cell_off[cell]==cell_off[cell+1]){double zero=0.;for(int k=tid;k<nm;k+=blockDim.x)acc[k]=zero;
  for(int w=tid;w<TW;w+=blockDim.x)flux[cell*FW+GW(w)]=zero*(albedo/PI);}
#endif
#elif defined(ACC_IN_GLOBAL)
 for(int k=tid;k<nm+TW;k+=blockDim.x){if(k<nm)acc[k]=0.;else ground[k-nm]=0.;}SECTION_SYNC(__syncthreads());
#else
 for(int k=tid;k<nm+TW;k+=blockDim.x)acc[k]=0.;SECTION_SYNC(__syncthreads());
#endif
#if   defined(FL_SPLIT_ON)
 const int64_t ray_lo=cell_off[0],ray_hi=cell_off[cells];
#ifdef LAUNCH_RAY_LIMIT
 // Ray-range launches (eda, 2026-10-07): this launch serves queue slots up to ray_sync[1+cells] (rays from
 // ray_lo); the host restarts the head ray_sync[0] at the previous end and keeps the cell counters and
 // ray_rads, so a cell is projected by whichever launch finishes its last ray, with the same code and order.
 const int q_lim=ray_sync[1+cells];
#endif
 for(;;){
  int64_t ray;{int q=0;if(lane==0)q=atomicAdd(ray_sync,1);q=__shfl_sync(0xffffffff,q,0);
#ifdef LAUNCH_RAY_LIMIT
   if(q>=q_lim||ray_lo+q>=ray_hi)break;ray=ray_lo+q;}
#else
   if(ray_lo+q>=ray_hi)break;ray=ray_lo+q;}
#endif
  int cell;{int lo=0,hi=cells;while(lo<hi){int mid=(lo+hi+1)>>1;if(cell_off[mid]<=ray)lo=mid;else hi=mid-1;}cell=lo;}
#endif
#ifdef FL_SPLIT_ON
  const double *p=starts+3*cell,*target=targets+9*cell;
  float B[9];for(int k=0;k<9;k++)B[k]=(float)target[k];
  float *y=Y+warp*K,*rad=rads+warp*4*TW,*rc=err+warp*4*TW,*cd=cosdest+warp*33;
  {
#else
 const double *p=starts+3*cell,*target=targets+9*cell;
 float B[9];for(int k=0;k<9;k++)B[k]=(float)target[k];
 int64_t begin=cell_off[cell],end=cell_off[cell+1];
#ifdef SPLIT_ACTIVE
 {const int64_t n_=end-begin;const int64_t b_=begin+n_*split_part/SPLIT_S,e_=begin+n_*(split_part+1)/SPLIT_S;begin=b_;end=e_;}
#endif
 for(int64_t round=begin;round<end;round+=nwarps){
#ifdef RS_ACTIVE
  const int rs_win=32*nwarps;
 const int64_t rs_wbeg=begin+((round-begin)/rs_win)*rs_win;
  const int rs_n=(int)min((int64_t)rs_win,end-rs_wbeg);
  if(round==rs_wbeg){
   for(int t=tid;t<rs_n;t+=blockDim.x){int64_t r=rs_wbeg+t;rs_key[t]=(unsigned char)min((int64_t)255,(int64_t)(edge_off[r+1]-edge_off[r]));}
   __syncthreads();
   for(int t=tid;t<rs_n;t+=blockDim.x){const int kt=rs_key[t];int rank=0;
    for(int j=0;j<rs_n;j++){const int kj=rs_key[j];rank+=(kj>kt)||(kj==kt&&j<t);}
    rs_perm[rank]=(unsigned char)t;}
   __syncthreads();
  }
#define RAY_AT(R) (rs_wbeg+(int64_t)rs_perm[(R)-rs_wbeg])
#else
#define RAY_AT(R) (R)
#endif
#ifdef RS_ACTIVE
  int64_t ray=round+warp<end?RAY_AT(round+warp):end;
#else
  int64_t ray=round+warp;
#endif
  float *y=Y+warp*K,*rad=rads+warp*4*TW,*rc=err+warp*4*TW,*cd=cosdest+warp*33;
  if(ray<end){
#endif
   const double *d=dirs+3*ray;
#ifndef LUT_ONLY
   // global_st, dl[3], pu[4], se[3], direct[4]; PATH_DS adds p/d as six DS values.
   float *dl=ray_constants+1,*pu=ray_constants+4,*se=ray_constants+8,*direct=ray_constants+11;
#ifdef PATH_DS
   DS *beta_p=(DS*)(ray_constants+15),*beta_d=beta_p+3;
#else
   const double *beta_p=p,*beta_d=d;
#endif

if(lane==0){ray_constants[0]=(float)sqrt(fmax(0.,1.-d[2]*d[2]));
#ifdef PATH_DS
   // Same conversion as path_q, once per ray; the existing syncwarp publishes both words.
   for(int k=0;k<3;k++){beta_p[k]=ds(p[k]);beta_d[k]=ds(d[k]);}
#endif


for(int j=0;j<3;j++){double x=0;for(int k=0;k<3;k++)x+=target[3*k+j]*d[k];dl[j]=(float)x;}

   pu[0]=1.f-dl[0]*dl[0];pu[1]=1.f-dl[1]*dl[1];pu[2]=1.f-dl[2]*dl[2];pu[3]=-dl[0]*dl[2];
   float sun_dot=(dl[0]*B[6]+dl[1]*B[7])+dl[2]*B[8];for(int k=0;k<3;k++)se[k]=B[6+k]-dl[k]*sun_dot;
   direct[0]=(pu[0]-se[0]*se[0])/2;direct[1]=(pu[1]-se[1]*se[1])/2;direct[2]=(pu[2]-se[2]*se[2])/2;direct[3]=(pu[3]-se[0]*se[2])/2;
   }
   SECTION_SYNC(__syncwarp());
   float global_st=ray_constants[0];
#ifdef WARM_LANE_ON
   // pu is fixed for the ray: register copies instead of 16 shared loads per step in the corner loop (slimray is
   // LSU-issue-bound, cq 1252: LSU pipe 72 %); same values.
   const float pu_r[4]={pu[0],pu[1],pu[2],pu[3]};
#endif
#else

float dl[3]={};float global_st=(float)sqrt(fmax(0.,1.-d[2]*d[2]));


for(int j=0;j<3;j++){double x=0;for(int k=0;k<3;k++)x+=target[3*k+j]*d[k];dl[j]=(float)x;}

   float pu[4]={1.f-dl[0]*dl[0],1.f-dl[1]*dl[1],1.f-dl[2]*dl[2],-dl[0]*dl[2]},direct[4];
   float sun_dot=(dl[0]*B[6]+dl[1]*B[7])+dl[2]*B[8],se[3];for(int k=0;k<3;k++)se[k]=B[6+k]-dl[k]*sun_dot;
   direct[0]=(pu[0]-se[0]*se[0])/2;direct[1]=(pu[1]-se[1]*se[1])/2;direct[2]=(pu[2]-se[2]*se[2])/2;direct[3]=(pu[3]-se[0]*se[2])/2;
#endif
   SECTION_BEGIN()
#if (!defined(FL_SPLIT_ON) || !K_FIRST) && !defined(SLIM_RAYS) && !(defined(GF_ON) && K_PHASE==2)
   sh_basis(d[2],lane,y,aa,bb,dd);   // SLIM_RAYS: y only feeds the projection (sos_slim_proj reads the chunk y)
#endif
   SECTION_END(2)
   if(lane==0){
#ifndef FL_SPLIT_ON

double st=sqrt(fmax(0.,1.-d[2]*d[2]));float cp=st>1e-12?(float)(d[0]/st):1.f;cp=clip(cp,-1,1);

    cd[0]=1;cd[1]=cp;for(int m=2;m<=32;m++)cd[m]=(2.f*cp)*cd[m-1]-cd[m-2];
#endif
    for(int k=0;k<4*TW;k++){rad[k]=0;rc[k]=0;}
   }
   SECTION_SYNC(__syncwarp());
#ifdef WARM_LANE_ON
   float lane_rad[4]={},lane_rc[4]={},lane_val[4],lane_log;
#endif
#ifdef SLIM_PRELOAD
   // Per-lane extinction rows of the column-record attenuation are constant over the ray: read them once.
   float ray_ext[3]={},ray_st_ext=0.f;
#ifdef SLIM_RAYS
   {const int rw=lane&15;if(rw<TW){for(int k=0;k<3;k++)ray_ext[k]=ext[k*FW+GW(rw)];ray_st_ext=st_ext[GW(rw)];}}
#else
   if(lane<TW){for(int k=0;k<3;k++)ray_ext[k]=ext[k*FW+GW(lane)];ray_st_ext=st_ext[GW(lane)];}
#endif
#endif
   float cache[4][TW];
#ifdef STRATO
   float scache[4][TW];
#endif
   int tags[4]={-1,-1,-1,-1};
#ifndef LUT_ONLY
   float tau_lane=0.f,tc_lane=0.f;
   // Simpson prefix state of lane w<TW (reference.py SimpsonPrefix): panel start (DS t, extinction) and the segment sum.
   DS sp_t={0.f,0.f};float sp_b=0.f,sp_acc=0.f;
#endif
   Sample &s=samples[warp];
   float sunphase=hg((float)d[2],aerosol_g);
#if (defined(STRATO) && !defined(LUT_ONLY))
   float strato_sunphase=0.f;
   if(first && lane<TW)strato_sunphase=mie_phase((float)d[2],GW(lane),st_chi);
#endif
#ifdef PATH_DS
   DS gu_l={0.f,0.f},gw_l={0.f,0.f};if(lane<nq && nq<=32){gu_l=ds(gu[lane]);gw_l=ds(gw[lane]);}
#endif
   for(uint32_t seg=edge_off[ray];seg+1<edge_off[ray+1];seg++){
    double a=edges[seg],b=edges[seg+1];if(b-a<=1e-8)continue;
    bool low=(lower[ray]>>(seg-edge_off[ray]))&1;
#ifdef PATH_DS
    const DS seg_a=ds(a),seg_b=ds(b);
#endif
#ifndef LUT_ONLY
    // Simpson prefix: beta(a) is evaluated anew for every segment (never carried over a skipped b-a<=1e-8 segment).
#ifdef PATH_DS
    sp_t=seg_a;
#else
    sp_t=ds(a);
#endif
    sp_acc=0.f;if(lane<TW && !PROD_ON SINV_AND_SIMPSON)sp_b=beta_at(beta_p,beta_d,sp_t,BETA_ARG,GW(lane));
#endif
#ifdef PT_ACTIVE
    for(int step0=0;step0<nq;step0+=PT_NB){
     const int nb=min(PT_NB,nq-step0);
#ifdef PATH_DS
     // Lane = step0+lane: its node DS from the lane that holds it (all lanes are converged here), else ds() as before.
     DS bu,bw;
     if(nq<=32){bu={__shfl_sync(0xffffffffu,gu_l.h,step0+lane),__shfl_sync(0xffffffffu,gu_l.l,step0+lane)};
      bw={__shfl_sync(0xffffffffu,gw_l.h,step0+lane),__shfl_sync(0xffffffffu,gw_l.l,step0+lane)};}
     else if(lane<nb){bu=ds(gu[step0+lane]);bw=ds(gw[step0+lane]);}
#if defined(SLIM_RAYS)
     if(lane<nb)point_m2(pts[lane],beta_p,beta_d,B,dl,seg_a,seg_b,low,bu,h,nh,chi,nc);
#else
     if(lane<nb)point(pts[lane],beta_p,beta_d,B,dl,global_st,seg_a,seg_b,low,bu,bw,h,nh,chi,nc,height,sun_h,nsh,sun_col,nsx,first
#ifndef LUT_ONLY
      ,profile_table,profile_mask
#endif
#ifdef STRATO
      ,st_sun_col,st_center,st_sigma,st_norm
#endif
      );
#endif
#elif defined(SLIM_RAYS)
     if(lane<nb)point_m2(pts[lane],p,d,B,dl,a,b,low,gu[step0+lane],h,nh,chi,nc);
#else
     if(lane<nb)point(pts[lane],p,d,B,dl,global_st,a,b,low,gu[step0+lane],gw[step0+lane],h,nh,chi,nc,height,sun_h,nsh,sun_col,nsx,first
#ifndef LUT_ONLY
      ,profile_table,profile_mask
#endif
#ifdef STRATO
      ,st_sun_col,st_center,st_sigma,st_norm
#endif
      );
#endif
     __syncwarp();
    for(int step=step0;step<step0+nb;step+=SR_INC){
     SECTION_BEGIN()
#ifdef SLIM_RAYS
     // P3': this lane's sample is step+h; an absent upper sample (odd tail) is inactive and reads the lower record.
     const int sr_h=lane>>4,sr_w=lane&15;
     const bool sr_odd=step+1<step0+nb;
     const int sj=(sr_h&&sr_odd)?step+1:step;
     const bool sr_act=sr_w<TW&&(sr_h==0||sr_odd);
     float sr_v[4]={0.f,0.f,0.f,0.f};
     // The batch record is read straight into registers (T 4, sw 1, cell 1 x 128-bit loads) instead of being copied into s
     // and re-read as scalars; the step uses only T, sw, cell and ir of it. Same values.
     const PointM2 &q_=pts[sj-step0];
     const float4 qT0=*(const float4*)(q_.T),qT1=*(const float4*)(q_.T+4),qT2=*(const float4*)(q_.T+8),qT3=*(const float4*)(q_.T+12);
     const float4 qsw=*(const float4*)q_.sw;const int4 qcell=*(const int4*)q_.cell;const int qir=q_.ir;
#define PT_IR qir
#define PT_CELL(c) ((c)==0?qcell.x:(c)==1?qcell.y:(c)==2?qcell.z:qcell.w)
#define PT_SW(c) ((c)==0?qsw.x:(c)==1?qsw.y:(c)==2?qsw.z:qsw.w)
#else
     {const int *pw=(const int*)&pts[step-step0];int *sw_=(int*)static_cast<PointData*>(&s);
      for(int k=lane;k<(int)(sizeof(PointData)/4);k+=32)sw_[k]=pw[k];}
#define PT_IR s.ir
#define PT_CELL(c) s.cell[c]
#define PT_SW(c) s.sw[c]
#endif
#else
    for(int step=0;step<nq;step++){
     SECTION_BEGIN()
#endif
#ifdef PATH_DS
     DS su,swu;   // all lanes are converged here
     if(nq<=32){su={__shfl_sync(0xffffffffu,gu_l.h,step),__shfl_sync(0xffffffffu,gu_l.l,step)};
      swu={__shfl_sync(0xffffffffu,gw_l.h,step),__shfl_sync(0xffffffffu,gw_l.l,step)};}
     else{su=ds(gu[step]);swu=ds(gw[step]);}
#endif
#if defined(PT_ACTIVE)
#elif defined(PATH_DS)
     if(lane==0)point(s,beta_p,beta_d,B,dl,global_st,seg_a,seg_b,low,su,swu,h,nh,chi,nc,height,sun_h,nsh,sun_col,nsx,first
#ifndef LUT_ONLY
      ,profile_table,profile_mask
#endif
#ifdef STRATO
      ,st_sun_col,st_center,st_sigma,st_norm
#endif
      );
#else
     if(lane==0)point(s,p,d,B,dl,global_st,a,b,low,gu[step],gw[step],h,nh,chi,nc,height,sun_h,nsh,sun_col,nsx,first
#ifndef LUT_ONLY
      ,profile_table,profile_mask
#endif
#ifdef STRATO
      ,st_sun_col,st_center,st_sigma,st_norm
#endif
      );
#endif
     SECTION_END(0)
     SECTION_SYNC(__syncwarp());
     if(lane==0){s.positive=(1u<<TW)-1u;}
#ifdef WARM_LANE_ON
     // Warp-uniform copy of s.positive: every lane holds the ballot results, so the per-corner lane-0 shared
     // read-modify-write and the shared re-read before the attenuation are not needed; same bits.
     uint32_t lane_pos=SR_POS0;
#endif
     SECTION_BEGIN_IF(first)
#ifdef GF_ON
     if(first){
      float accw=0.f,st_acc=0.f;
#if defined(GROUND_SH_L)
      ground_sh_sums(s,lane,goff,gdir,gval,accw,st_acc);
#else
#if defined(FL_SPLIT_ON) || K_PHASE==2
      // FL_RAYSPLIT projects after the ray; phase 2 has no projection. Both can borrow the warp's Y row.
      gf_sums(s,lane,y,goff,gdir,gval,st_chi,aerosol_g,accw,st_acc GF_TROPO_ARG);
#else
      gf_sums(s,lane,ground_phase,goff,gdir,gval,st_chi,aerosol_g,accw,st_acc GF_TROPO_ARG);
#endif
#endif
      if(lane<TW){s.c32[0][lane]=accw;s.st_c32[0][lane]=st_acc;}
      __syncwarp();
     }
#elif !defined(LUT_ONLY)
     if(first){
      float accw=0.f;
#if defined(LUT_ONLY)
      if(lane==0)for(int w=0;w<TW;w++)s.c32[0][w]=0.f;
#endif
      for(int corner=0;corner<4;corner++){int cc=s.cell[corner];
       for(int64_t j0=goff[cc];j0<goff[cc+1];j0+=32){
        int64_t j=j0+lane;
        if(j<goff[cc+1]){float m0=s.mu*gdir[3*j+2]+s.exd*gdir[3*j],m1=fabsf(s.eyd)*gdir[3*j+1];
         ground_phase[lane]=(hg(
clip(m0+m1,-1,1),aerosol_g)+hg(clip(m0-m1,-1,1),aerosol_g))/2.f;}
        __syncwarp();
        int count=(int)((goff[cc+1]-j0)<32?(goff[cc+1]-j0):32);
#ifndef LUT_ONLY
#ifdef TROPO_MIE
        // Mie: the per-direction phase depends on the wavelength, so lane w evaluates it itself (ground_phase unused).
        if(lane<TW){const float swc=s.sw[corner];const float *gq=gval+j0*FW+GW(lane);int k=0;
         for(;k+8<=count;k+=8){float g[8],ph[8];
          _Pragma("unroll") for(int q=0;q<8;q++){g[q]=gq[(k+q)*FW];ph[q]=tropo_pair(s,gdir,j0+k+q,GW(lane),TROPO_LUT);}
          _Pragma("unroll") for(int q=0;q<8;q++)accw+=(swc*ph[q])*g[q];}
         for(;k<count;k++)accw+=(swc*tropo_pair(s,gdir,j0+k,GW(lane),TROPO_LUT))*gq[k*FW];}
#else
        if(lane<TW){const float swc=s.sw[corner];const float *gq=gval+j0*FW+GW(lane);int k=0;
         for(;k+8<=count;k+=8){float g[8],ph[8];
          _Pragma("unroll") for(int q=0;q<8;q++){g[q]=gq[(k+q)*FW];ph[q]=ground_phase[k+q];}
          _Pragma("unroll") for(int q=0;q<8;q++)accw+=(swc*ph[q])*g[q];}
         for(;k<count;k++)accw+=(swc*ground_phase[k])*gq[k*FW];}
#endif
        if(false)
#endif
        for(int k=0;k<count;k++){
#ifdef TROPO_MIE
#define BATCH_PHASE tropo_pair(s,gdir,j0+k,GW(lane),TROPO_LUT)
#else
#define BATCH_PHASE ground_phase[k]
#endif
#ifndef LUT_ONLY
         if(lane<TW)accw+=(s.sw[corner]*BATCH_PHASE)*gval[(j0+k)*FW+GW(lane)];
#else
         if(lane==0)for(int w=0;w<TW;w++)s.c32[0][w]+=(s.sw[corner]*BATCH_PHASE)*gval[(j0+k)*FW+GW(w)];
#endif
#undef BATCH_PHASE
        }
        __syncwarp();
       }
      }
#ifndef LUT_ONLY
      if(lane<TW)s.c32[0][lane]=accw;
#endif
      __syncwarp();
     }
#elif !defined(LUT_ONLY)
     if(first){
      // Lane w owns wavelength w (TW<=31); lane 31 evaluates the paired phase
      // once per ground direction. Per-wavelength corner/direction order and
      // every product are unchanged; numerical equivalence is checked by the gates.
      float accw=0.f;
      for(int corner=0;corner<4;corner++){int cc=s.cell[corner];
       for(int64_t j=goff[cc];j<goff[cc+1];j++){
        float ph=0.f;
        if(lane==31){float m0=s.mu*gdir[3*j+2]+s.exd*gdir[3*j],m1=fabsf(s.eyd)*gdir[3*j+1];
         ph=(hg(clip(m0+m1,-1,1),aerosol_g)+hg(clip(m0-m1,-1,1),aerosol_g))/2.f;}
        ph=__shfl_sync(0xffffffff,ph,31);
        if(lane<TW)accw+=(s.sw[corner]*ph)*gval[j*FW+GW(lane)];
       }
      }
      if(lane<TW)s.c32[0][lane]=accw;
      __syncwarp();
     }
#else
     if(first && lane==0){
      // Ground phase geometry is independent of wavelength. The unused
      // first-light Fourier cache holds the wavelength accumulators.
      for(int w=0;w<TW;w++)s.c32[0][w]=0.f;
      for(int corner=0;corner<4;corner++){int cc=s.cell[corner];
       for(int64_t j=goff[cc];j<goff[cc+1];j++){
        float m0=s.mu*gdir[3*j+2]+s.exd*gdir[3*j],m1=fabsf(s.eyd)*gdir[3*j+1];
        float ph=(hg(clip(m0+m1,-1,1),aerosol_g)+hg(clip(m0-m1,-1,1),aerosol_g))/2.f;
        // Preserve corner/direction addition order for every wavelength.
        for(int w=0;w<TW;w++)s.c32[0][w]+=(s.sw[corner]*ph)*gval[j*FW+GW(w)];
       }
      }
     }
#endif
#ifndef GF_ON
#ifdef STRATO
#if (defined(STRATO) && !defined(LUT_ONLY))
     if(first){
      int w=lane&15;float st_acc=0.f;
#if (defined(STRATO) && !defined(LUT_ONLY))
      const float pmu=s.mu,pexd=s.exd,paey=fabsf(s.eyd);
#define GP_LUT(JJ,D0,D1,D2) if(w<TW){float m0=pmu*(D2)+pexd*(D0),m1=paey*(D1); \
        float u=lane<16?clip(m0+m1,-1,1):clip(m0-m1,-1,1); \
        float x=(clip(u,-1.f,1.f)+1.f)*(float)(STRATO_LUT_BINS/2);int li=min((int)x,STRATO_LUT_BINS-1); \
        ca=st_chi[(33+li)*FW+GW(w)];cb=st_chi[(34+li)*FW+GW(w)];cf=x-(float)li;} \
       if(lane<TW)cg=gval[(JJ)*FW+GW(lane)];
      for(int corner=0;corner<4;corner++){int cc=s.cell[corner];const float swc=s.sw[corner];
       int64_t j=goff[cc],je=goff[cc+1];
       float ca=0.f,cb=0.f,cf=0.f,cg=0.f,d0=0.f,d1=0.f,d2=0.f;
       if(j<je){GP_LUT(j,gdir[3*j],gdir[3*j+1],gdir[3*j+2])}
       if(j+1<je){d0=gdir[3*(j+1)];d1=gdir[3*(j+1)+1];d2=gdir[3*(j+1)+2];}
       for(;j<je;j++){
        const float a=ca,b=cb,f=cf,gvj=cg;
        if(j+1<je){GP_LUT(j+1,d0,d1,d2)}
        if(j+2<je){d0=gdir[3*(j+2)];d1=gdir[3*(j+2)+1];d2=gdir[3*(j+2)+2];}
        float paired=0.f;
        if(w<TW)paired=a+f*(b-a);
        float minus=__shfl_down_sync(0xffffffff,paired,16);
        if(lane<TW){float ph=(paired+minus)/2.f;
         st_acc+=(swc*ph)*gvj;}
       }
      }
#undef GP_LUT
#else
      for(int corner=0;corner<4;corner++){int cc=s.cell[corner];
       for(int64_t j=goff[cc];j<goff[cc+1];j++){
        float paired=0.f;
        if(w<TW){
         float m0=s.mu*gdir[3*j+2]+s.exd*gdir[3*j],m1=fabsf(s.eyd)*gdir[3*j+1];
         float u=lane<16?clip(m0+m1,-1,1):clip(m0-m1,-1,1);
         paired=mie_phase(u,GW(w),st_chi);
        }
        // All 32 lanes participate, including padding lanes 15 and 31.
        float minus=__shfl_down_sync(0xffffffff,paired,16);
        if(lane<TW){float ph=(paired+minus)/2.f;
         st_acc+=(s.sw[corner]*ph)*gval[j*FW+GW(lane)];}
       }
      }
#endif // GROUND_PREFETCH
      if(lane<TW)s.st_c32[0][lane]=st_acc;
      __syncwarp();
     }
#else
     if(first && lane==0){
      for(int w=0;w<TW;w++)s.st_c32[0][w]=0.f;
      for(int corner=0;corner<4;corner++){int cc=s.cell[corner];
       for(int64_t j=goff[cc];j<goff[cc+1];j++){
        float m0=s.mu*gdir[3*j+2]+s.exd*gdir[3*j],m1=fabsf(s.eyd)*gdir[3*j+1];
        for(int w=0;w<TW;w++){float ph=(mie_phase(clip(m0+m1,-1,1),GW(w),st_chi)+mie_phase(clip(m0-m1,-1,1),GW(w),st_chi))/2.f;
         s.st_c32[0][w]+=(s.sw[corner]*ph)*gval[j*FW+GW(w)];}
       }
      }
     }
#endif // STRATO_LANE_GROUND
#endif
#endif // GF_ON
     SECTION_END_IF(1,first)
#if defined(WARM_LANE_ON) || ((defined(STRATO) && !defined(LUT_ONLY)))
     // Includes lane-0 STRATO/scalar ground writes before other lanes read.
     SECTION_SYNC(__syncwarp());
#endif
     SECTION_BEGIN()
#if defined(WARM_LANE_ON) || ((defined(STRATO) && !defined(LUT_ONLY)))
     if(SR_ACT){int w=SR_W;
#else
     if(lane==0)for(int w=0;w<TW;w++){
#endif
#ifdef WARM_LANE_ON
      lane_log=0;for(int k=0;k<4;k++)lane_val[k]=0;
#else
      s.logs[w]=0;for(int k=0;k<4;k++)s.val[w][k]=0;
#endif
      if(first){
       float od=0;for(int k=0;k<3;k++)od+=s.sc[k]*ext[k*FW+GW(w)];
#ifdef STRATO
       od+=s.st_sc*st_ext[GW(w)];
#endif
       float sun=s.lit?expf(-od):0.f,ga=s.c32[0][w];
#ifdef TROPO_MIE
       float beam=(s.prof[1]*asca[GW(w)])*(sun*tropo_phase((float)d[2],GW(w),TROPO_LUT)+ga),ray=(sun*s.prof[0])*ext[GW(w)];
#else
       float beam=(s.prof[1]*asca[GW(w)])*(sun*sunphase+ga),ray=(sun*s.prof[0])*ext[GW(w)];
#endif
       for(int k=0;k<4;k++)s.val[w][k]=ray*(mix*(float)(3/(8*PI))*direct[k]+(1.f-mix)*(float)(1/(8*PI))*pu[k])+(beam*pu[k])/2.f;
#ifdef STRATO
#if (defined(STRATO) && !defined(LUT_ONLY))
       float sb=(s.st_prof*st_sca[GW(w)])*(sun*strato_sunphase+s.st_c32[0][w]);
#else
       float sb=(s.st_prof*st_sca[GW(w)])*(sun*mie_phase((float)d[2],GW(w),st_chi)+s.st_c32[0][w]);
#endif
       for(int k=0;k<4;k++)s.val[w][k]+=(sb*pu[k])/2.f;
#endif
      }
     }
     SECTION_END(3)
     SECTION_SYNC(__syncwarp());
     SECTION_BEGIN()
     if(!first && !NOSH_ON && tags[0]>=0){
      int di=s.cell[0]/nc-tags[0]/nc,dj=s.cell[0]%nc-tags[0]%nc;
      if(di==0 && dj==1){COPY_CORNER(0,1);COPY_CORNER(2,3);}
      else if(di==0 && dj==-1){COPY_CORNER(1,0);COPY_CORNER(3,2);}
      else if(di==1 && dj==0){COPY_CORNER(0,2);COPY_CORNER(1,3);}
      else if(di==-1 && dj==0){COPY_CORNER(2,0);COPY_CORNER(3,1);}
      else if(di==1 && dj==1){COPY_CORNER(0,3);}
      else if(di==1 && dj==-1){COPY_CORNER(1,2);}
      else if(di==-1 && dj==1){COPY_CORNER(2,1);}
      else if(di==-1 && dj==-1){COPY_CORNER(3,0);}
     }
#ifdef SLIM_RAYS
     // P2-min: sample invariants live only through this corner loop. No local array.
     const int sample_ir=qir;
     const float t00=qT0.x,t01=qT0.y,t02=qT0.z,t03=qT0.w;
     const float t10=qT1.x,t11=qT1.y,t12=qT1.z,t13=qT1.w;
     const float t20=qT2.x,t21=qT2.y,t22=qT2.z,t23=qT2.w;
     const float t30=qT3.x,t31=qT3.y,t32=qT3.z,t33=qT3.w;
#endif
#ifdef SLIM_PRELOAD
     float pre_aero[4]={},pre_rm[4][4]={},pre_rec[5]={},pre_hr[2]={},pre_sd[2]={};
     if(SR_ACT){
      if(!first){
       #pragma unroll
       for(int i=0;i<2;i++){pre_hr[i]=hr[PT_IR+i];pre_sd[i]=st_density[PT_IR+i];}
       #pragma unroll
       for(int corner=0;corner<4;corner++){
        pre_aero[corner]=SLIM_AERO_J(SR_J,corner)[SR_W];
        const int64_t pc=PT_CELL(corner);
        #pragma unroll
        for(int k=0;k<4;k++)pre_rm[corner][k]=src[(pc*1132+k)*FW+GW(SR_W)];
       }
      }
      if(sinv_mode==2){const float *r=SINV_REC(SR_J);
       #pragma unroll
       for(int k=0;k<5;k++)pre_rec[k]=r[k];}
     }
#endif
     #pragma unroll
     for(int corner=0;corner<4;corner++){
#ifdef SLIM_RAYS
      int cc=PT_CELL(corner),ir=sample_ir+corner/2;
#else
      int cc=s.cell[corner],ir=s.ir+corner/2;
#endif
#ifdef SKIP_ACTIVE
      const bool sk_a=ir>=64||((live_a>>ir)&1ull),sk_s=ir>=64||((live_s>>ir)&1ull);
#endif
      if(!first && !NOSH_ON && tags[corner]!=cc){
       #pragma unroll
       for(int w=0;w<TW;w++)cache[corner][w]=0.f;
#ifdef STRATO
       for(int w=0;w<TW;w++)scache[corner][w]=0.f;
#endif
       // Degree-major loads: consecutive active lanes read consecutive SHs.
       // Each lane still sums l=m..32 in precisely the reference order.
       for(int l=0;l<=32;l++){
        if(lane<=l){int k=ix(l,lane);float pref=y[k];
#ifndef SKIP_ACTIVE
        #pragma unroll
#endif
#ifndef LUT_ONLY
        #ifndef LUT_ONLY
#ifdef SKIP_ACTIVE
        if(sk_a){
        #pragma unroll
#endif
        for(int w=0;w<TW;w+=2){
         float2 pair=warm_sh_pair(src_sh,(cc),w,k);
         cache[corner][w]+=pref*pair.x;
         if(w+1<TW)cache[corner][w+1]+=pref*pair.y;
        }
#ifdef SKIP_ACTIVE
        }
#endif
#else
        for(int w=0;w<TW;w++)cache[corner][w]+=pref*src_sh[((int64_t)cc*FW+GW(w))*KPAD+k];
#endif
#else
        for(int w=0;w<TW;w++)cache[corner][w]+=pref*src[((int64_t)cc*
#ifdef STRATO
         1132
#else
         571
#endif
         +10+k)*FW+GW(w)];
#endif
#ifdef STRATO
#if (defined(STRATO) && !defined(LUT_ONLY))
#ifndef LUT_ONLY
#ifdef SKIP_ACTIVE
        if(sk_s){
        #pragma unroll
#endif
        for(int w=0;w<TW;w+=2){
         float2 pair=warm_sh_pair(src_sh,(nh*nc+(cc)),w,k);
         scache[corner][w]+=pref*pair.x;
         if(w+1<TW)scache[corner][w+1]+=pref*pair.y;
        }
#ifdef SKIP_ACTIVE
        }
#endif
#else
        for(int w=0;w<TW;w++)scache[corner][w]+=pref*src_sh[((int64_t)(nh*nc+cc)*FW+GW(w))*KPAD+k];
#endif
#else
        for(int w=0;w<TW;w++)scache[corner][w]+=pref*src[((int64_t)cc*1132+571+k)*FW+GW(w)];
#endif
#endif
       }}
       if(lane==0){
#ifndef SKIP_ACTIVE
        #pragma unroll
#endif
#ifndef LUT_ONLY
        #ifndef LUT_ONLY
#ifdef SKIP_ACTIVE
        if(sk_a){
        #pragma unroll
#endif
        for(int w=0;w<TW;w+=2){
         float2 pair=warm_sh_pair(src_sh,cc,w,560);
         float v=0.f;v+=y[560]*pair.x;s.c32[corner][w]=v;
         if(w+1<TW){float v1=0.f;v1+=y[560]*pair.y;s.c32[corner][w+1]=v1;}
        }
#ifdef SKIP_ACTIVE
        }
#endif
#else
        for(int w=0;w<TW;w++){float v=0.f;v+=y[560]*src_sh[((int64_t)cc*FW+GW(w))*KPAD+560];s.c32[corner][w]=v;}
#endif
#else
        for(int w=0;w<TW;w++){float v=0.f;v+=y[560]*src[((int64_t)cc*
#ifdef STRATO
         1132
#else
         571
#endif
         +570)*FW+GW(w)];s.c32[corner][w]=v;}
#endif
#ifdef STRATO
#if (defined(STRATO) && !defined(LUT_ONLY))
#ifndef LUT_ONLY
#ifdef SKIP_ACTIVE
        if(sk_s){
        #pragma unroll
#endif
        for(int w=0;w<TW;w+=2){
         float2 pair=warm_sh_pair(src_sh,(nh*nc+cc),w,560);
         float v=0.f;v+=y[560]*pair.x;s.st_c32[corner][w]=v;
         if(w+1<TW){float v1=0.f;v1+=y[560]*pair.y;s.st_c32[corner][w+1]=v1;}
        }
#ifdef SKIP_ACTIVE
        }
#endif
#else
        for(int w=0;w<TW;w++){float v=0.f;v+=y[560]*src_sh[((int64_t)(nh*nc+cc)*FW+GW(w))*KPAD+560];s.st_c32[corner][w]=v;}
#endif
#else
        for(int w=0;w<TW;w++){float v=0.f;v+=y[560]*src[((int64_t)cc*1132+1131)*FW+GW(w)];s.st_c32[corner][w]=v;}
#endif
#endif
       }
       tags[corner]=cc;
      }
#ifndef LUT_ONLY
      // Do not access cache[corner][lane]: retain compile-time wave indexes
      // so the Fourier cache can stay register promoted. Every lane executes
      // every shuffle; the selected owner keeps one reduced wavelength.
      float own_aero=0.f;
#ifdef STRATO
      float own_saero=0.f;
#endif
#if (defined(STRATO) && FW==15 && !defined(LUT_ONLY))
      if(!first){
       const float cml=s.cm[lane];
#define RS2_REDUCE(CACHE,C32,OUT) { float r[8]; const bool u16=(lane&16)!=0; \
        _Pragma("unroll") for(int V=0;V<8;V++){float xa=0.f,xb=0.f; \
         if(V<TW){xa=cml*CACHE[corner][V];if(lane==0)xa+=s.cm[32]*C32[corner][V];} \
         if(V+8<TW){xb=cml*CACHE[corner][V+8];if(lane==0)xb+=s.cm[32]*C32[corner][V+8];} \
         const float send=u16?xa:xb,keep=u16?xb:xa; r[V]=keep+__shfl_xor_sync(0xffffffff,send,16);} \
        _Pragma("unroll") for(int st=4;st;st>>=1){const bool up=(lane&(2*st))!=0; \
         _Pragma("unroll") for(int V=0;V<st;V++){const float send=up?r[V]:r[V+st],keep=up?r[V+st]:r[V]; \
          r[V]=keep+__shfl_xor_sync(0xffffffff,send,2*st);}} \
        const float fin=r[0]+__shfl_xor_sync(0xffffffff,r[0],1); \
        OUT=__shfl_sync(0xffffffff,fin,(2*lane)&31); }
#ifdef SKIP_ACTIVE
       if(sk_a)
#endif
       if(!NOSH_ON)RS2_REDUCE(cache,s.c32,own_aero)
#ifdef SKIP_ACTIVE
       if(sk_s)
#endif
       RS2_REDUCE(scache,s.st_c32,own_saero)
#undef RS2_REDUCE
      }
#ifdef SINV
      if(!first){
#ifdef SLIM_PRELOAD
       if(SLIM_ON)own_aero=SR_ACT?pre_aero[corner]:0.f;
#else
       if(SLIM_ON)own_aero=lane<TW?SLIM_AERO(corner)[lane]:0.f;
#endif
       else if(PROD_ON && lane<TW)SLIM_AERO(corner)[lane]=own_aero;
      }
#endif
#else
      if(!first){
       #pragma unroll
       for(int w=0;w<TW;w++){
        float aero=s.cm[lane]*cache[corner][w];
        if(lane==0)aero+=s.cm[32]*s.c32[corner][w];
        for(int stride=16;stride;stride>>=1)aero+=__shfl_down_sync(0xffffffff,aero,stride);
        aero=__shfl_sync(0xffffffff,aero,0);if(lane==w)own_aero=aero;
#ifdef STRATO
        float saero=s.cm[lane]*scache[corner][w];
        if(lane==0)saero+=s.cm[32]*s.st_c32[corner][w];
        for(int stride=16;stride;stride>>=1)saero+=__shfl_down_sync(0xffffffff,saero,stride);
        saero=__shfl_sync(0xffffffff,saero,0);if(lane==w)own_saero=saero;
#endif
       }
      }
#endif // TRANSPORT_RS2
      bool bad_trace=false;
      if(SR_ACT && !NOOUT_ON){
       int w=SR_W;float aero=own_aero;
#ifdef SLIM_RAYS
       const float corner_sw=PT_SW(corner);
#endif
#ifdef STRATO
       float saero=own_saero;
#endif
#ifdef SLIM_PRELOAD
       float rm[4];for(int k=0;k<4;k++)rm[k]=first?gm[((int64_t)cc*4+k)*FW+GW(w)]:pre_rm[corner][k];
#else
       float rm[4];for(int k=0;k<4;k++)rm[k]=first?gm[((int64_t)cc*4+k)*FW+GW(w)]:src[((int64_t)cc*
#ifdef STRATO
          1132
#else
          571
#endif
          +k)*FW+GW(w)];
#endif
#ifdef SLIM_PRELOAD
       float density=first?s.prof[0]:pre_hr[corner/2];if(!first)aero*=ha[ir];
       if(!first)saero*=pre_sd[corner/2];
#else
       float density=first?s.prof[0]:hr[ir];if(!first)aero*=ha[ir];
#ifdef STRATO
       if(!first)saero*=st_density[ir];
#endif
#endif
       float tr=(rm[0]+rm[1])+rm[2],tensors[3];
       #pragma unroll
       for(int component=0;component<4;component++){
#ifndef LUT_ONLY
        if(component==3 && phase==1)continue;
#endif
#ifdef SLIM_RAYS
        // component is unrolled; keep the original k=0,1,2,3 multiply/add order.
#define SLIM_T(k) (component==0?t0##k:component==1?t1##k:component==2?t2##k:t3##k)
        float rot=0.f;rot+=SLIM_T(0)*rm[0];rot+=SLIM_T(1)*rm[1];
        rot+=SLIM_T(2)*rm[2];rot+=SLIM_T(3)*rm[3];
#undef SLIM_T
#else
        float rot=0.f;for(int k=0;k<4;k++)rot+=s.T[4*component+k]*rm[k];
#endif
#ifdef WARM_LANE_ON
        const float puc=pu_r[component];
#else
        const float puc=pu[component];
#endif
        float tensor=density*(mix*rot+((1.f-mix)/3.f*tr)*puc);
        tensor+=(aero*puc)/2.f;
#ifdef STRATO
        if(!first)tensor+=(saero*puc)/2.f;
#endif
#ifdef WARM_LANE_ON
#ifdef SLIM_RAYS
        lane_val[component]+=corner_sw*tensor;
#else
        lane_val[component]+=s.sw[corner]*tensor;
#endif
#else
        s.val[w][component]+=s.sw[corner]*tensor;
#endif
        if(component<3)tensors[component]=tensor;
       }
       if(!first){float trace=(tensors[0]+tensors[1])+tensors[2];
        // Gate v2 (2026-10-21, user decision): only a weighted corner can force the linear fallback. A zero-weight
        // corner (sample exactly on a grid node) is ignored, so the mode no longer flips with the last bit of atan2f.
#ifdef SLIM_RAYS
        bad_trace=!(trace>=TRACE_MIN)&&corner_sw!=0.f;
#else
        bad_trace=!(trace>=TRACE_MIN)&&s.sw[corner]!=0.f;
#endif
#ifdef WARM_LANE_ON
#ifdef SLIM_RAYS
        lane_log+=corner_sw*logf(fmaxf(trace,TRACE_MIN));
#else
        lane_log+=s.sw[corner]*logf(fmaxf(trace,TRACE_MIN));
#endif
#else
        s.logs[w]+=s.sw[corner]*logf(fmaxf(trace,TRACE_MIN));
#endif
       }
      }
      // Replace lane-0 per-wave bit clears with an exact integer ballot.
      uint32_t bad=__ballot_sync(0xffffffff,bad_trace);
#ifdef WARM_LANE_ON
      if(!first)lane_pos&=~bad;
#else
      if(lane==0 && !first)s.positive&=~bad;
#endif
      // Order shared consumers before lane-0 c32 copies / next corner, and
      // complete all val/log writes before the next point/attenuation phase.
      __syncwarp();
#else
      #pragma unroll
      for(int w=0;w<TW;w++){
       float aero=0;
       if(!first){aero=s.cm[lane]*cache[corner][w];if(lane==0)aero+=s.cm[32]*s.c32[corner][w];
        for(int stride=16;stride;stride>>=1)aero+=__shfl_down_sync(0xffffffff,aero,stride);
       }
       aero=__shfl_sync(0xffffffff,aero,0);
#ifdef STRATO
       float saero=0.f;
       if(!first){saero=s.cm[lane]*scache[corner][w];if(lane==0)saero+=s.cm[32]*s.st_c32[corner][w];
        for(int stride=16;stride;stride>>=1)saero+=__shfl_down_sync(0xffffffff,saero,stride);}
       saero=__shfl_sync(0xffffffff,saero,0);
#endif
       float tensor=0.f;
       // Four component owners each retain the entire wavelength loop. This
       // avoids serializing all tensor arithmetic onto lane zero.
#ifndef LUT_ONLY
       if(lane<(phase==1?3:4)){
#else
       if(lane<4){
#endif
        float rm[4];for(int k=0;k<4;k++)rm[k]=first?gm[((int64_t)cc*4+k)*FW+GW(w)]:src[((int64_t)cc*
#ifdef STRATO
          1132
#else
          571
#endif
          +k)*FW+GW(w)];
        float density=first?s.prof[0]:hr[ir];if(!first)aero*=ha[ir];
        float tr=(rm[0]+rm[1])+rm[2],rot=0;
        for(int k=0;k<4;k++)rot+=s.T[4*lane+k]*rm[k];
        tensor=density*(mix*rot+((1.f-mix)/3.f*tr)*pu[lane]);
        tensor+=(aero*pu[lane])/2.f;
#ifdef STRATO
        if(!first){saero*=st_density[ir];tensor+=(saero*pu[lane])/2.f;}
#endif
        s.val[w][lane]+=s.sw[corner]*tensor;
       }
       float t1=__shfl_sync(0xffffffff,tensor,1),t2=__shfl_sync(0xffffffff,tensor,2);
       if(lane==0 && !first){float trace=(tensor+t1)+t2;if(!(trace>=TRACE_MIN)&&s.sw[corner]!=0.f)s.positive&=~(1u<<w);
         s.logs[w]+=s.sw[corner]*logf(fmaxf(trace,TRACE_MIN));}
      }
#endif
     }
     SECTION_END(3)
     SECTION_SYNC(__syncwarp());
     SECTION_BEGIN()
#ifndef LUT_ONLY
     if(SR_ACT && !PROD_ON){int w=SR_W;{
#else
     if(lane==0){
     #pragma unroll
     for(int w=0;w<TW;w++){
#endif
#ifdef WARM_LANE_ON
      float tr=(lane_val[0]+lane_val[1])+lane_val[2];
      float lane_scale=(lane_pos&SR_BIT(w))&&tr>=TRACE_MIN?expf(lane_log)/fmaxf(tr,TRACE_MIN):1.f;
#else
      if(!first){float tr=(s.val[w][0]+s.val[w][1])+s.val[w][2];
       float scale=(s.positive&(1u<<w))&&tr>=TRACE_MIN?expf(s.logs[w])/fmaxf(tr,TRACE_MIN):1.f;
       s.logs[w]=scale;
      }else s.logs[w]=1.f;
#endif
#ifndef LUT_ONLY
      // Simpson prefix to this node (task B (1)): panel [previous node or a, this node], node extinction from point()'s
      // profile, new midpoint evaluation; replaces the GL-weight prefix mid=(tau-tc)+dt/2, tau+=dt.
      SINV_SIMPSON {
#ifdef STRATO
       const float pst=s.st_prof;
#else
       const float pst=0.f;
#endif
#ifdef PATH_DS
       const DS tk=node_t(seg_a,seg_b,low,su);
#else
       const DS tk=node_t(a,b,low,gu[step]);
#endif
       const float bk=extinction_at(s.prof[0],s.prof[1],s.prof[2],pst,ext SP_EXT_ARG,GW(w));
       sp_acc+=simpson_piece(beta_p,beta_d,sp_t,tk,sp_b,bk,BETA_ARG,GW(w));sp_t=tk;sp_b=bk;
      }
      float mid=(tau_lane-tc_lane)+sp_acc;
#ifdef WARM_LANE_ON
      float lane_att=s.ds*expf(-mid);
#ifdef SINV
      // mode 1 runs with unit-basis extinction tables (host swap), so lanes 0..3 integrate the four profile columns.
      if(sinv_mode==1){float *r=SINV_REC(step);if(w<4)r[1+w]=mid;if(w==0)r[0]=s.ds;
       // KSPLIT=2 (VPROD): vp_records points to the per-point (c0, cp bits) int2 records (device memory, or mapped
       // pinned host memory when the device cannot hold them), bundle-global like SINV_REC; 0 (KSPLIT=1) writes nothing.
       // c0 = s.cell[0], cp = s.cm[1] (point()). RECORD variant only (compiled out of PROD/SLIM and the unsplit kernel).
       if(REC_ON && w==0){
        if(vp_records)vp_records[(int64_t)(seg-ray)*nq+step]=make_int2(s.cell[0],__float_as_int(s.cm[1]));}}
#ifdef SLIM_PRELOAD
      if(sinv_mode==2){const float *r=pre_rec;
       float tq=r[1]*ray_ext[0];tq+=r[2]*ray_ext[1];tq+=r[3]*ray_ext[2];
       tq+=r[4]*ray_st_ext;lane_att=r[0]*expf(-tq);}
#else
      if(sinv_mode==2){const float *r=SINV_REC(step);
       float tq=r[1]*ext[GW(w)];tq+=r[2]*ext[FW+GW(w)];tq+=r[3]*ext[2*FW+GW(w)];
       tq+=r[4]*st_ext[GW(w)];lane_att=r[0]*expf(-tq);}
#endif
#endif
      #pragma unroll
      for(int k=0;k<4;k++){
       if(REC_ON)break;
#ifndef LUT_ONLY
       if(k==3 && phase==1)continue;
#endif
#ifdef SLIM_RAYS
       sr_v[k]=(lane_val[k]*lane_scale)*lane_att;
#else
       kah(lane_rad[k],lane_rc[k],(lane_val[k]*lane_scale)*lane_att);
#endif
      }
#else
      s.att[w]=s.ds*expf(-mid);
#endif
#else
      s.att[w]=0.f; // unreachable: the kernel traps at entry without LANE_ATTEN (no Simpson prefix on the lane-0 path)
#endif
     }}
#ifdef SLIM_RAYS
     // P3' accumulation (K1): the lower half owns the Kahan state and adds the even sample, then the odd one, per component
     // (the one-step order). Every lane takes part in the shuffle; an absent odd sample adds nothing (no kah(0)).
     #pragma unroll
     for(int k=0;k<4;k++){
      if(k==3 && phase==1)continue;
      const float v_odd=__shfl_down_sync(0xffffffffu,sr_v[k],16);
      if(lane<TW){kah(lane_rad[k],lane_rc[k],sr_v[k]);if(sr_odd)kah(lane_rad[k],lane_rc[k],v_odd);}
     }
#endif
     SECTION_END(4)
#ifndef WARM_LANE_ON
     SECTION_SYNC(__syncwarp());
#ifndef LUT_ONLY
     if(lane<(phase==1?3:4)){
#else
     if(lane<4){
#endif
      #pragma unroll
      for(int w=0;w<TW;w++){float value=s.val[w][lane]*s.logs[w];kah(rad[4*w+lane],rc[4*w+lane],value*s.att[w]);}
     }
     SECTION_SYNC(__syncwarp());
#endif
    }
#ifdef PT_ACTIVE
    }
#endif
#ifndef LUT_ONLY
    // Simpson end piece to b (beta(b) anew), then the segment sum into tau (Kahan), as reference.py SimpsonPrefix.finish.
#ifdef PATH_DS
    if(lane<TW && !PROD_ON SINV_AND_SIMPSON){const DS te=seg_b;
#else
    if(lane<TW && !PROD_ON SINV_AND_SIMPSON){const DS te=ds(b);
#endif
    const float be=beta_at(beta_p,beta_d,te,BETA_ARG,GW(lane));
     sp_acc+=simpson_piece(beta_p,beta_d,sp_t,te,sp_b,be,BETA_ARG,GW(lane));kah(tau_lane,tc_lane,sp_acc);}
#endif
   }
#ifdef WARM_LANE_ON
   if(lane<TW){
    #pragma unroll
    for(int k=0;k<4;k++)rad[4*lane+k]=lane_rad[k];
   }
   __syncwarp();
#endif
   if(lane==0 && !NOOUT_ON && with_boundary){int j=end_j[ray];float f=end_f[ray];
    for(int w=0;w<TW;w++){float e=(1.f-f)*(float)boundary[j*FW+GW(w)];e+=f*(float)boundary[(j+1)*FW+GW(w)];
     float od=0;for(int k=0;k<3;k++)od+=end_col[3*ray+k]*ext[k*FW+GW(w)];
#ifdef STRATO
     od+=st_end_col[ray]*st_ext[GW(w)];
#endif
     e*=expf(-od)*(float)hits[ray];
#ifndef LUT_ONLY
     for(int k=0;k<(phase==1?3:4);k++)rad[4*w+k]+=(e*pu[k])/2.f;
#else
     for(int k=0;k<4;k++)rad[4*w+k]+=(e*pu[k])/2.f;
#endif
    }
   }
   if(phase==2){SECTION_SYNC(__syncwarp());for(int k=lane;k<4*TW;k+=32)
query[ray*4*FW+k]=rad[k];
}
#if defined(SINV) && defined(SLIM_RAYDUMP)
   // KSPLIT=3 VERIFY: this ray's final radiance (slim_dump argument; header int64 [5] = chunk first ray), as SLIM_RAYS publishes it.
   if(phase!=2){__syncwarp();float *dump_=slim_dump+(ray-((const int64_t*)query)[5])*4*TW;
    for(int k=lane;k<4*TW;k+=32)dump_[k]=rad[k];}
#endif
#ifdef FL_SPLIT_ON
   // Publish this ray's rad (every lane fences its own stores), then count it for its cell.
   __syncwarp();
   for(int k=lane;k<4*TW;k+=32)ray_rads[(ray-ray_lo)*4*TW+k]=rad[k];
#ifndef LAUNCH_RAY_LIMIT
   __threadfence();__syncwarp();
   const int64_t cb=cell_off[cell],ce=cell_off[cell+1];
   int done=0;if(lane==0)done=atomicAdd(ray_sync+1+cell,1);done=__shfl_sync(0xffffffff,done,0);
   if(done==(int)(ce-cb)-1){
    // Last ray of this cell: project rays cb..ce-1 in order (== original round-then-warp order).
    __threadfence();
    double *mrow=moments+(int64_t)cell*nm,gsum=0.;
    #pragma unroll 1
    for(int64_t r=cb;r<ce;r++){
     const double *drow=dirs+3*r;
     sh_basis(drow[2],lane,y,aa,bb,dd);
     if(lane==0){double st=sqrt(fmax(0.,1.-drow[2]*drow[2]));float cp=st>1e-12?(float)(drow[0]/st):1.f;cp=clip(cp,-1,1);
      cd[0]=1;cd[1]=cp;for(int m=2;m<=32;m++)cd[m]=(2.f*cp)*cd[m-1]-cd[m-2];}
     __syncwarp();
     const float *rr=ray_rads+(r-ray_lo)*4*TW;float wt=weights[r];
     #pragma unroll 1
     for(int k=lane;k<nm;k+=32){int moment=k/TW,w=k%TW;float v;
      if(moment<4)v=__ldcg(rr+4*w+moment)*wt;
      else {int sk=moment-4;
#ifndef LUT_ONLY
       int m=LM.m[sk];
#else
       int l=(int)((sqrtf(8.f*sk+1.f)-1.f)/2.f);while(ix(l+1,0)<=sk)l++;while(ix(l,0)>sk)l--;int m=sk-ix(l,0);
#endif
       float tr=(__ldcg(rr+4*w)+__ldcg(rr+4*w+1))+__ldcg(rr+4*w+2);v=(tr*wt)*(y[sk]*cd[m]);}
      double a=r==cb?0.:mrow[k];a+=(double)v;mrow[k]=a;
     }
     if(lane<TW){int w=lane;double dlz=0;for(int k=0;k<3;k++)dlz+=target[3*k+2]*drow[k];
      float tr=(__ldcg(rr+4*w)+__ldcg(rr+4*w+1))+__ldcg(rr+4*w+2),wg=weights[r]*fmaxf(0.f,(float)dlz);
      gsum+=(double)(tr*wg);}
     __syncwarp();
    }
    if(lane<TW)flux[cell*FW+GW(lane)]=gsum*(albedo/PI);
   }
#endif
  }
 }
#else
  }
#ifndef LUT_ONLY
  // Phase 2 has no cross-warp data: warp-private buffers only need the warp
  // to finish reading rad/y before its next ray overwrites them.
  if(phase==2)SECTION_SYNC(__syncwarp()); else
#endif
  SECTION_SYNC(__syncthreads());
  if(phase!=2 && !NOOUT_ON){
   SECTION_BEGIN()
   for(int k=tid;k<nm;k+=blockDim.x){int moment=k/TW,w=k%TW;
#if   defined(ACC_IN_GLOBAL)
    double acc_k=acc[k];
#define ACC_SLOT acc_k
#else
#define ACC_SLOT acc[k]
#endif
#if !defined(LUT_ONLY) && !K_FIRST
    // R4 ACC_F32SUM (precision track): the round's warp terms are summed in FP32, one FP64 add per round.
    float acc_part=0.f;
#define ACC_ADD(v) acc_part+=(v)
#else
#define ACC_ADD(v) ACC_SLOT+=(double)(v)
#endif
    for(int wp=0;wp<nwarps;wp++)if(round+wp<end){float *rr=rads+wp*4*TW;float wt=weights[RAY_AT(round+wp)],v;
     if(moment<4)v=rr[4*w+moment]*wt;
     else {int sk=moment-4;
#ifndef LUT_ONLY
      int m=LM.m[sk];
#else
      int l=(int)((sqrtf(8.f*sk+1.f)-1.f)/2.f);while(ix(l+1,0)<=sk)l++;while(ix(l,0)>sk)l--;int m=sk-ix(l,0);
#endif
      float tr=(rr[4*w]+rr[4*w+1])+rr[4*w+2];v=(tr*wt)*(Y[wp*K+sk]*cosdest[wp*33+m]);}
     ACC_ADD(v);
    }
#if !defined(LUT_ONLY) && !K_FIRST
    ACC_SLOT+=(double)acc_part;
#endif
#undef ACC_ADD
#if defined(ACC_IN_GLOBAL)
    acc[k]=acc_k;
#endif
#undef ACC_SLOT
   }
   for(int w=tid;w<TW;w+=blockDim.x)for(int wp=0;wp<nwarps;wp++)if(round+wp<end){
    int64_t rridx=RAY_AT(round+wp);const double*d=dirs+3*rridx;double dlz=0;for(int k=0;k<3;k++)dlz+=target[3*k+2]*d[k];
    float *rr=rads+wp*4*TW,tr=(rr[4*w]+rr[4*w+1])+rr[4*w+2],wt=weights[rridx]*fmaxf(0.f,(float)dlz);
    ground[w]+=(double)(tr*wt);
   }
   SECTION_END(5)
  }
#ifndef LUT_ONLY
  if(phase!=2)
#endif
  SECTION_SYNC(__syncthreads());
 }
#endif // FL_SPLIT_ON
#ifndef ACC_IN_GLOBAL
 if(phase==0 && !NOOUT_ON)for(int k=tid;k<nm;k+=blockDim.x)
#else
 if(false)for(int k=tid;k<nm;k+=blockDim.x)
#endif
moments[(int64_t)cell*nm+k]=acc[k];
#if   !defined(FL_SPLIT_ON)

#ifdef SPLIT_ACTIVE
 if(phase!=2 && !NOOUT_ON){
  for(int w=tid;w<TW;w+=blockDim.x)split_flux[((int64_t)split_part*cells+cell)*TW+w]=ground[w];
  __threadfence();__syncthreads();
  int split_last=0;if(tid==0)split_last=atomicAdd(split_count+cell,1)==SPLIT_S-1;
  split_last=__syncthreads_or(split_last);
  if(split_last){__threadfence();
   for(int w=tid;w<TW;w+=blockDim.x){double g=__ldcg(split_flux+(int64_t)cell*TW+w);
    for(int q=1;q<SPLIT_S;q++)g+=__ldcg(split_flux+((int64_t)q*cells+cell)*TW+w);
    flux[cell*FW+GW(w)]=g*(albedo/PI);}
   if(phase==0)for(int k=tid;k<nm;k+=blockDim.x){double v=__ldcg(moments+(int64_t)cell*nm+k);
    for(int q=1;q<SPLIT_S;q++)v+=__ldcg(split_scratch+((int64_t)(q-1)*cells+cell)*nm+k);
    moments[(int64_t)cell*nm+k]=v;}
  }
 }
#else
 if(phase!=2 && !NOOUT_ON)for(int w=tid;w<TW;w+=blockDim.x)flux[cell*FW+GW(w)]=ground[w]*(albedo/PI);
#endif
#endif
#ifdef EXP_SECTION_CLOCKS
 if(lane==0){
  section_cycles[7]=clock64()-total_begin;
  int64_t slot=(((int64_t)cell*(FW/TW)+(int)blockIdx.y)*nwarps+warp)*8;
  for(int k=0;k<8;k++)section_clocks[slot+k]=section_cycles[k];
 }
#endif
}

#if defined(FL_SPLIT_ON) && defined(LAUNCH_RAY_LIMIT)
// Batch eight rays: every warp builds one basis row, all threads load weighted radiance,
// and each output uses an FP32 batch partial followed by one FP64 addition. No atomics.
#define FLP_THREADS 256
#define FLP_RAYS 8
#define FLP_WAVES ((TW+1)/2)
#define FLP_MOMENTS ((565+127)/128)
extern "C" __global__ void __launch_bounds__(FLP_THREADS) fl_project(int cells,long long ray_lo,const int64_t*cell_off,
 const double*targets,const double*dirs,const float*weights,const double*aa,const double*bb,const double*dd,
 const float*ray_rads,double albedo,double*moments,double*flux){
 const int cell=blockIdx.x;if(cell>=cells)return;
 const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,nm=565*TW;
 const int mo=tid%128,wbase=(tid/128)*FLP_WAVES;
 __shared__ float y[FLP_RAYS][K],cd[FLP_RAYS][33];
 __shared__ float rad[FLP_RAYS][6*TW],wg[FLP_RAYS];
 const int64_t cb=cell_off[cell],ce=cell_off[cell+1];
 const double *target=targets+9*cell;
 double acc[FLP_MOMENTS][FLP_WAVES]={},gsum=0.;
 #pragma unroll 1
 for(int64_t rb=cb;rb<ce;rb+=FLP_RAYS){
  const int nr=(int)min((int64_t)FLP_RAYS,ce-rb);
  if(warp<nr){const double *d=dirs+3*(rb+warp);
   sh_basis(d[2],lane,y[warp],aa,bb,dd);
   if(lane==0){double st=sqrt(fmax(0.,1.-d[2]*d[2]));float cp=st>1e-12?(float)(d[0]/st):1.f;cp=clip(cp,-1,1);
    cd[warp][0]=1;cd[warp][1]=cp;for(int m=2;m<=32;m++)cd[warp][m]=(2.f*cp)*cd[warp][m-1]-cd[warp][m-2];
    double dlz=0;for(int k=0;k<3;k++)dlz+=target[3*k+2]*d[k];
    wg[warp]=weights[rb+warp]*fmaxf(0.f,(float)dlz);}
  }
  __syncthreads();
  for(int i=tid;i<nr*TW;i+=FLP_THREADS){const int r=i/TW,w=i%TW;
   const float *rr=ray_rads+(rb+r-ray_lo)*4*TW+4*w;const float wt=weights[rb+r];
   const float r0=rr[0],r1=rr[1],r2=rr[2],r3=rr[3],tr=(r0+r1)+r2;
   rad[r][w]=r0*wt;rad[r][TW+w]=r1*wt;rad[r][2*TW+w]=r2*wt;rad[r][3*TW+w]=r3*wt;
   rad[r][4*TW+w]=tr*wt;rad[r][5*TW+w]=tr*wg[r];}
  __syncthreads();
  #pragma unroll
  for(int j=0;j<FLP_MOMENTS;j++){
   const int moment=mo+j*128,sk=moment-4;
   if(moment<565){
    float part[FLP_WAVES]={};
    int m=0;
    if(moment>=4){
#ifndef LUT_ONLY
     m=LM.m[sk];
#else
     int l=(int)((sqrtf(8.f*sk+1.f)-1.f)/2.f);while(ix(l+1,0)<=sk)l++;while(ix(l,0)>sk)l--;m=sk-ix(l,0);
#endif
    }
    for(int r=0;r<nr;r++){
     const float yc=moment<4?1.f:y[r][sk]*cd[r][m];
     #pragma unroll
     for(int q=0;q<FLP_WAVES;q++)if(wbase+q<TW)
      part[q]+=rad[r][(moment<4?moment:4)*TW+wbase+q]*yc;
    }
    #pragma unroll
    for(int q=0;q<FLP_WAVES;q++)acc[j][q]+=(double)part[q];
   }
  }
  if(tid<TW){float part=0.f;for(int r=0;r<nr;r++)part+=rad[r][5*TW+tid];gsum+=(double)part;}
  __syncthreads();
 }
 #pragma unroll
 for(int j=0;j<FLP_MOMENTS;j++)if(mo+j*128<565){
  #pragma unroll
  for(int q=0;q<FLP_WAVES;q++)if(wbase+q<TW)moments[(int64_t)cell*nm+(mo+j*128)*TW+wbase+q]=acc[j][q];
 }
 if(tid<TW)flux[cell*FW+GW(tid)]=gsum*(albedo/PI);
}

#endif

#if defined(SINV) && PROD_ON && defined(SH_L) && !defined(LUT_ONLY) && (defined(STRATO) && FW==15 && !defined(LUT_ONLY)) && FW==15 && !K_FIRST
#if SH_L<=16
// VPROD (SOSFUSE2_KSPLIT=2, results/vprod_20261025): visitor-stream producer, compiled in the PROD module (same source,
// same NVRTC options) and writing the same chunk-local own_aero records [slot = (point-base)*4+corner][w] as PROD
// (row stride AERO_W = 16: one float4 store per thread, pad wave 15 = 0).
// Block = one source cell (combined SH plane in shared), visitors = (ray, cell) of a host plan (fused.vprod_plan).
// Arithmetic follows the PROD path (SH_COMBINE, TRANSPORT_RS2, L<=16), with absent SH terms set to compile-time zero:
//  y      : sos_vprod_y calls sh_basis(d[2],lane,...) per ray exactly as the warm kernel does (one warp per ray);
//  a_m[w] : +0 start, cache[w]+=y[ix(l,m)]*C[ix(l,m)][w] for l=m..SH_L ascending (warm_sh_pair addressing);
//  aero[w]: x_m=cm[m]*a_m[w] for m<=SH_L, otherwise +0; the absent m=32 contribution is also +0 (SH_L<=16),
//           then the RS2_REDUCE butterfly as a fixed tree: P1[j]=x_j+x_{j+16}, P2[j]=P1[j]+P1[j+8], P3[j]=P2[j]+P2[j+4],
//           P4[j]=P3[j]+P3[j+2], value=P4[0]+P4[1] (xor 16,8,4,2,1; float addition is commutative);
//  cm     : cm[0]=1, cm[1]=cp, cm[m]=(2cp)cm[m-1]-cm[m-2] from the recorded cp (point());
//  live   : SKIP_ACTIVE mask test of the warm kernel, sk_a||sk_s (SH_COMBINE); a dead layer writes +0 (RS2 skipped).
// Data movement (results/vprod_regs_20261026): the C plane and each
// tile's y rows go global->shared with cp.async (sm_80+, plain copies below), so a thread has all its copies in flight
// at once instead of one vis_ray->yb load chain per element (the element loop was not unrolled: ~38 serial global
// latencies per tile); one warp copies whole y rows (coalesced), the tile's rays come from one vis_ray load per lane;
// each visitor's next (slot, cp) record words are loaded before the current record's arithmetic; the next tile's y rows
// are copied while this tile's records are written (vp_y is free after the a_m pass).
#define VP_TILE 32
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__>=800
#define VP_CP_ASYNC
#endif
__device__ __forceinline__ void vp_copy4(float*dst,const float*src){
#ifdef VP_CP_ASYNC
 asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"((unsigned)__cvta_generic_to_shared(dst)),"l"(src):"memory");
#else
 *dst=*src;
#endif
}
__device__ __forceinline__ void vp_copy8(float2*dst,const float2*src){
#ifdef VP_CP_ASYNC
 asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"::"r"((unsigned)__cvta_generic_to_shared(dst)),"l"(src):"memory");
#else
 *dst=*src;
#endif
}
__device__ __forceinline__ void vp_copy_wait(){   // commit this thread's copies and wait for all of them
#ifdef VP_CP_ASYNC
 asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n":::"memory");
#endif
}
// y rows of visitors tb..tb+nv-1 (nv <= VP_TILE = warp width) into vy[j*K+k]: lane j reads visitor tb+j's ray once, each
// warp copies whole rows (warp-uniform j, coalesced k).
__device__ __forceinline__ void vp_fill(float*vy,const float*yb,const int*vis_ray,int tb,int nv,int lane,int wp,int nwp){
 const int ray=lane<nv?vis_ray[tb+lane]:0;
 for(int j=wp;j<nv;j+=nwp){
  const float *src=yb+(int64_t)__shfl_sync(0xffffffffu,ray,j)*K;float *dst=vy+j*K;
  for(int k=lane;k<K;k+=32)vp_copy4(dst+k,src+k);}
}
extern "C" __global__ void __launch_bounds__(128) sos_vprod_y(int nray,const double*dirs,const float2*aa2,const float2*bb2,
 const float2*dd2,float*yout){
 const int lane=threadIdx.x&31,t=(int)blockIdx.x*((int)blockDim.x>>5)+((int)threadIdx.x>>5);
 if(t>=nray)return;
 const double *d=dirs+3*(int64_t)t;
 sh_basis_ds(d[2],lane,yout+(int64_t)t*K,aa2,bb2,dd2);
}
// Stage 2 of one record for quad q (waves 4q..4q+3): cm recurrence from the record's cp, the fixed RS2-order butterfly over m,
// one float4 store into the AERO_W row (q=3, i=3 is the pad wave: its C column is zero, so it stores 0).
__device__ __forceinline__ void vp_record(int sl,float cp,bool live,const float (&a)[SH_L+1][4],int q,float*aero){
    float4 *out=(float4*)(aero+(int64_t)sl*AERO_W+4*q);   // 16 B aligned: AERO_W*4 B rows, quad q at +16q B
    if(!live){*out=make_float4(0.f,0.f,0.f,0.f);return;}
    float cm[33];cm[0]=1;cm[1]=cp;
    #pragma unroll
    for(int m=2;m<33;m++)cm[m]=(2.f*cp)*cm[m-1]-cm[m-2];
    const float c32t=0.f;   // SH_L<=16: no m=32 coefficient
    float res[4];   // quad 3, i=3 is the pad wave: its C column is zero, so it stores 0
    #pragma unroll
    for(int i=0;i<4;i++){
     float p[16];
     #pragma unroll
     for(int j=0;j<16;j++){
      // Ordinary products permit FMA contraction under --fmad=true; the reduction tree is unchanged.
      // VPROD/PROD agreement uses the numerical tolerance contract, not separate product rounding.
      float lo=(j<=SH_L)?cm[j]*a[min(j,SH_L)][i]:0.f;
      if(j==0)lo+=c32t;
      const float hi=(j+16<=SH_L)?cm[j+16]*a[min(j+16,SH_L)][i]:0.f;
      p[j]=lo+hi;}
     #pragma unroll
     for(int j=0;j<8;j++)p[j]=p[j]+p[j+8];
     #pragma unroll
     for(int j=0;j<4;j++)p[j]=p[j]+p[j+4];
     p[0]=p[0]+p[2];p[1]=p[1]+p[3];
     res[i]=p[0]+p[1];
    }
    *out=make_float4(res[0],res[1],res[2],res[3]);
}
extern "C" __global__ void __launch_bounds__(128) sos_vprod(const int*blk,const int*vis_rbeg,const int*vis_ray,
 const int*rec_slot,const int2*c0cp,const float*yb,const float*src_sh,const float*hr,const float*ha,const float*st_density,
 int nh,int nc,float*aero){
 __shared__ float4 vp_c[K*4];       // [k][wave quad]: combined SH plane of the block's cell (wave 15 = pad 0)
 __shared__ float vp_y[VP_TILE*K];  // y rows of the tile's visitors (row stride K: conflict-free for odd K)
 const int b=blockIdx.x,cell=blk[3*b],v0=blk[3*b+1],v1=blk[3*b+2];
 const int tid=threadIdx.x,vl=tid>>2,q=tid&3,lane=tid&31,wp=tid>>5,nwp=(int)blockDim.x>>5;
 bool live=true;
#ifdef SKIP_ACTIVE
 {const int ir=cell/nc;if(nh<=64 && ir<64){const float r=hr[ir];live=(ha[ir]>SKIP_TAU*r)||(st_density[ir]>SKIP_TAU*r);}}
#endif
 if(live){float *cf=(float*)vp_c;   // pair (w=2pr, 2pr+1) of k: warm_sh_pair's float2 at (cell*((FW+1)/2)+pr)*KPAD+k
  const float2 *cs=(const float2*)src_sh+(int64_t)cell*((FW+1)/2)*KPAD;
  for(int i=tid;i<8*K;i+=blockDim.x){const int pr=i/K,k=i-pr*K;vp_copy8((float2*)(cf+k*16+2*pr),cs+pr*KPAD+k);}
  if(v0<v1)vp_fill(vp_y,yb,vis_ray,v0,min(VP_TILE,v1-v0),lane,wp,nwp);}
 for(int tb=v0;tb<v1;tb+=VP_TILE){
  const int nv=min(VP_TILE,v1-tb);
  if(live)vp_copy_wait();
  __syncthreads();
  const int v=tb+vl;
  int r0=0,r1=0;
  float a[SH_L+1][4];
  if(vl<nv){
   r0=vis_rbeg[v];r1=vis_rbeg[v+1];
   #pragma unroll
   for(int m=0;m<=SH_L;m++){a[m][0]=0.f;a[m][1]=0.f;a[m][2]=0.f;a[m][3]=0.f;}
   if(live){const float *yv=vp_y+vl*K;
    #pragma unroll
    for(int l=0;l<=SH_L;l++){
     #pragma unroll
     for(int m=0;m<=l;m++){const int k=ix(l,m);const float y=yv[k];const float4 c=vp_c[k*4+q];
      a[m][0]+=y*c.x;a[m][1]+=y*c.y;a[m][2]+=y*c.z;a[m][3]+=y*c.w;}}
   }
  }
  __syncthreads();   // vp_y is free: the next tile's rows load while this tile's records are written
  if(live && tb+VP_TILE<v1)vp_fill(vp_y,yb,vis_ray,tb+VP_TILE,min(VP_TILE,v1-tb-VP_TILE),lane,wp,nwp);
  if(vl<nv){
   // Records in batches of 8: the 8 slots, then the 8 cp words, are independent loads in flight together instead of the
   // slot->cp->next-slot chain (results/wavefront_20261101/vprod_gemm V3b: +6 % on 5090/4090, bit-identical).
   for(int rb=r0;rb<r1;rb+=8){
    const int nb=min(8,r1-rb);int sl[8],cw[8];
    #pragma unroll
    for(int i=0;i<8;i++)sl[i]=i<nb?rec_slot[rb+i]:0;
    #pragma unroll
    for(int i=0;i<8;i++)cw[i]=(live&&i<nb)?c0cp[sl[i]>>2].y:0;
    #pragma unroll
    for(int i=0;i<8;i++)if(i<nb)vp_record(sl[i],__int_as_float(cw[i]),live,a,q,aero);
   }
  }
 }
 if(live && v0>=v1)vp_copy_wait();   // no tile: the C-plane copies still complete before exit
}
#endif
#endif

#if defined(SINV) && SLIM_ON && !defined(LUT_ONLY) && !K_FIRST && K_PHASE==0 && defined(ACC_IN_GLOBAL) &&  !defined(SLIM_RAYS) && !defined(SLIM_RAYDUMP)
// A 64-moment x wavelength register tile over 32 contiguous rays. Each thread owns half
// the wavelengths of one moment and reuses y*cd across them. The producer's ray sort is
// irrelevant to storage (ray_rads and yb are indexed by original ray); projection uses
// original ray order, FP32 batch partials, and deterministic FP64 batch/part combination.
#ifdef SPLIT_ACTIVE
#define SP_PARTS SPLIT_S
#else
#define SP_PARTS 1
#endif
#define SP_THREADS 128
#define SP_MOMENTS 64
#define SP_RAYS 32
#define SP_WAVES ((TW+1)/2)
extern "C" __global__ void __launch_bounds__(SP_THREADS) sos_slim_proj(int cells,int nw,long long ray_base,const int64_t*cell_off,
 const double*targets,const double*dirs,const float*weights,const uint32_t*edge_off,const float*yb,const float*ray_rads,
 double albedo,double*moments,double*flux){
 __shared__ float y[SP_RAYS][SP_MOMENTS+1],cd[SP_RAYS][33];
 __shared__ float rad[SP_RAYS][6*TW],wg[SP_RAYS];
 const int cell=blockIdx.x;if(cell>=cells)return;
 const int tid=threadIdx.x,mi=tid%SP_MOMENTS,mb=(int)blockIdx.y*SP_MOMENTS;
 const int moment=mb+mi,wbase=(tid/SP_MOMENTS)*SP_WAVES,nm=(4+K)*TW;
 const bool isM=moment<4+K,isF=moment==4+K;
 const int sk=moment-4,m=(isM&&moment>=4)?LM.m[sk]:0;
 const double *target=targets+9*cell;
 const int64_t cb=cell_off[cell],ce=cell_off[cell+1];
 double total[SP_WAVES]={};
 for(int part=0;part<SP_PARTS;part++){
  const int64_t begin=cb+(ce-cb)*part/SP_PARTS,end=cb+(ce-cb)*(part+1)/SP_PARTS;
  double acc[SP_WAVES]={};
  for(int64_t rb=begin;rb<end;rb+=SP_RAYS){
   const int nr=(int)min((int64_t)SP_RAYS,end-rb);
   if(tid<nr){const double *d=dirs+3*(rb+tid);
    double st=sqrt(fmax(0.,1.-d[2]*d[2]));float cp=st>1e-12?(float)(d[0]/st):1.f;cp=clip(cp,-1,1);
    cd[tid][0]=1;cd[tid][1]=cp;for(int mm=2;mm<33;mm++)cd[tid][mm]=(2.f*cp)*cd[tid][mm-1]-cd[tid][mm-2];
    double dlz=0;for(int k=0;k<3;k++)dlz+=target[3*k+2]*d[k];
    wg[tid]=weights[rb+tid]*fmaxf(0.f,(float)dlz);}
   // Consecutive lanes read consecutive coefficients in each original ray row.
   for(int i=tid;i<nr*SP_MOMENTS;i+=SP_THREADS){const int r=i/SP_MOMENTS,k=i%SP_MOMENTS,s=mb+k-4;
    y[r][k]=(s>=0&&s<K)?yb[(rb+r-ray_base)*K+s]:0.f;}
   __syncthreads();
   for(int i=tid;i<nr*TW;i+=SP_THREADS){const int r=i/TW,w=i%TW;
    const float *rr=ray_rads+(rb+r-ray_base)*4*TW+4*w;const float wt=weights[rb+r];
    const float r0=rr[0],r1=rr[1],r2=rr[2],r3=rr[3],tr=(r0+r1)+r2;
    rad[r][w]=r0*wt;rad[r][TW+w]=r1*wt;rad[r][2*TW+w]=r2*wt;rad[r][3*TW+w]=r3*wt;
    rad[r][4*TW+w]=tr*wt;rad[r][5*TW+w]=tr*wg[r];}
   __syncthreads();
   float sum[SP_WAVES]={};
   if(isM||isF)for(int r=0;r<nr;r++){
    const float yc=(moment<4||isF)?1.f:y[r][mi]*cd[r][m];
    #pragma unroll
    for(int q=0;q<SP_WAVES;q++)if(wbase+q<TW)
     sum[q]+=rad[r][(isF?5:moment<4?moment:4)*TW+wbase+q]*yc;
   }
   #pragma unroll
   for(int q=0;q<SP_WAVES;q++)acc[q]+=(double)sum[q];
   __syncthreads();
  }
  #pragma unroll
  for(int q=0;q<SP_WAVES;q++)total[q]+=acc[q];
 }
 #pragma unroll
 for(int q=0;q<SP_WAVES;q++)if(wbase+q<TW){
  if(isM)moments[(int64_t)cell*nm+moment*TW+wbase+q]=total[q];
  else if(isF)flux[cell*FW+GW(wbase+q)]=total[q]*(albedo/PI);
 }
}
#endif
