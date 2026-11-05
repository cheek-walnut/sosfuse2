// NVRTC, CUDA 12. FW is specialized to 1 or 15; no fast-math or implicit FMA.
// One destination cell (and optional spectral tile)/block, one ray/warp. No global atomics.
// NVRTC: math builtins available without headers
typedef unsigned int uint32_t; typedef unsigned short uint16_t; typedef long long int64_t; typedef int int32_t; typedef unsigned char uint8_t;
#ifndef FW
#define FW 15
#endif
#define K 561
#define PI 3.14159265358979323846
// sosfuse2 A6 variants, each an independent compile-time flag (all OFF = original):
//  SPECIALIZE (+K_FIRST,K_PHASE,K_BOUNDARY): first/phase/with_boundary become constexpr
//  LANE_GROUND: first-light ground pool, lane w accumulates wavelength w; phase on lane 31 + shfl
//  SH_AUX: SH source read from src_sh[cell][wave][KPAD] (KPAD 576), extra trailing argument
//  PHASE2_NOBARRIER: phase 2 uses __syncwarp instead of block barriers
//  LM_LUT: (l,m) of a projection moment from a constant table instead of sqrt + fixups
//  WAVE_TILE3 / WAVE_TILE5: 3 / 5 waves per block; mutually exclusive, FW1 unchanged.
//  EXP_SECTION_CLOCKS: separate per-cell/tile/warp lane-0 observer cycle buffer.
// Block size 256 is a launch option (fused.py), not a kernel flag.
// Follow-up candidates; every new behavior is opt-in. All new flags OFF
// preserves the supplied C+EXP_SECTION_CLOCKS translation-unit tokens.
// SH_AUX_PAIR: adjacent wavelength float2 sidecar loads (no spectral tiles).
// TRANSPORT_WAVE_LANES: unchanged SH sums/tree, wavelength-owned tensor/log.
// SH_AUX_FP16: PRECISION track; FP16 SH sidecar, FP32 synthesis, no new ABI arg.
// GROUND_BATCH_SHFL: C ground batch via shuffles, releases 128 bytes/warp.
// POINT_BATCH2: adjacent point geometry only; adds 284/292 bytes/warp.
// LANE_SOURCE_INIT: wavelength-owned initialization and first-light source.
#if (defined(SH_AUX_PAIR) || defined(SH_AUX_FP16)) && !defined(SH_AUX)
#error E_TRANSPORT_SH_AUX_REQUIRED
#endif
#if defined(SH_AUX_PAIR) && defined(SH_AUX_FP16)
#error E_TRANSPORT_STORAGE_CONFLICT
#endif
#if defined(SH_AUX_PAIR) && (defined(WAVE_TILE3) || defined(WAVE_TILE5))
#error E_SH_AUX_PAIR_TILE
#endif
#if defined(GROUND_BATCH_SHFL) && !defined(GROUND_PHASE_BATCH)
#error E_GROUND_BATCH_REQUIRED
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
#ifdef SPECIALIZE
#define P_FIRST first_arg
#define P_PHASE phase_arg
#define P_BOUNDARY boundary_arg
#else
#define P_FIRST first
#define P_PHASE phase
#define P_BOUNDARY with_boundary
#endif
#ifdef SH_AUX
#define KPAD 576
#define SH_PARAM ,const float*src_sh
#else
#define SH_PARAM
#endif
#ifdef LM_LUT
struct LMTable { unsigned char l[K],m[K];
 constexpr LMTable():l(),m(){int k=0;for(int a=0;a<=32;a++)for(int b=0;b<=a;b++){l[k]=(unsigned char)a;m[k]=(unsigned char)b;k++;}} };
__constant__ LMTable LM=LMTable();
#endif
#ifdef PROFILE
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
#if defined(WAVE_TILE3) && defined(WAVE_TILE5)
#error E_WAVE_TILE_CONFLICT
#endif
#if FW == 15 && (defined(WAVE_TILE3) || defined(WAVE_TILE5))
#define WAVE_TILED
#ifdef WAVE_TILE3
#define TW 3
#else
#define TW 5
#endif
#define GW(w) ((int)blockIdx.y*TW+(w))
#else
#define TW FW
#define GW(w) w
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
__device__ float clip(float x,float lo,float hi){return fminf(hi,fmaxf(lo,x));}
__device__ int search(const float*a,int n,float x){int lo=0,hi=n;while(lo<hi){int m=(lo+hi)/2;if(a[m]<=x)lo=m+1;else hi=m;}return max(0,min(n-2,lo-1));}
#ifdef PROFILE
__device__ float profile_value(float h,const float*table,int k){
 float x=clip(h,0.f,80.f)*2.f;int i=min((int)x,159);float f=x-(float)i;
 return table[i*2+k]+f*(table[(i+1)*2+k]-table[i*2+k]);
}
#endif
__device__ int ix(int l,int m){return l*(l+1)/2+m;}
__device__ float hg(float u,float g){float den=fmaxf(1e-20f,(1.f+g*g)-(2.f*g)*u);return (1.f-g*g)/((float)(4*PI)*(den*sqrtf(den)));}
#ifdef STRATO
__device__ float mie_phase(float u,int w,const float*chi){
 float p0=1.f,p1=u,v=p0*chi[w];v+=p1*chi[FW+w];
 for(int l=2;l<=32;l++){float pn=((float)(2*l-1)*u*p1-(float)(l-1)*p0)/(float)l;v+=pn*chi[l*FW+w];p0=p1;p1=pn;}
 return v/(float)(4*PI);
}
#endif
__device__ void kah(float &t,float &c,float v){float y=v-c,z=t+y;c=(z-t)-y;t=z;}
#ifdef POINT_BATCH2
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
#ifdef POINT_NOINLINE
#define POINT_INLINE __noinline__
#else
#define POINT_INLINE __forceinline__
#endif
#ifdef POINT_BATCH2
__device__ POINT_INLINE void point(PointData &s,const double*p
#else
__device__ POINT_INLINE void point(Sample &s,const double*p
#endif
 ,const double*d,const float*B,const float*dl,float st,
 double a,double b,bool lower,double gu,double gw,const float*hgrid,int nh,const float*chi,int nc,float height,
 const float*sun_h,int nsh,const float*sun_col,int nsx,bool first PROFILE_PARAM
#ifdef STRATO
 ,const float*st_sun_col,float st_center,float st_sigma,float st_norm
#endif
 ){
 DS ad=ds(a),bd=ds(b),length=dx(bd,ad),u=ds(gu),wu=ds(gw),v=lower?u:dx({1,0},u);
 DS t=lower?da(ad,dm(length,dm(u,u))):dx(bd,dm(length,dm(v,v)));
 s.ds=dv(dm(dm(length,v),wu))*2.f;
 DS q[3],q2[3];for(int k=0;k<3;k++){q[k]=da(ds(p[k]),dm(t,ds(d[k])));q2[k]=dm(q[k],q[k]);}
 DS rr=dr(da(da(q2[0],q2[1]),q2[2])),xy=dr(da(q2[0],q2[1]));
 s.h=clip(dv(dx(rr,{6371,0})),0,80);float rho=dv(xy);
 float cx=rho>1e-12f?dq(q[0],xy):1.f,sx=rho>1e-12f?dq(q[1],xy):0.f;
 s.cs=clip(dq(q[2],rr),-1,1);float angle=atan2f(rho,dv(q[2]));
 int ir=search(hgrid,nh,s.h),j=search(chi,nc,angle);s.ir=ir;
 float ah=clip((s.h-hgrid[ir])/(hgrid[ir+1]-hgrid[ir]),0,1),ac=clip((angle-chi[j])/(chi[j+1]-chi[j]),0,1);
 angle=chi[j]+ac*(chi[j+1]-chi[j]);float sn=sinf(angle),cc=cosf(angle);
 float Q[9]={-cc*cx,sx,sn*cx,-cc*sx,-cx,sn*sx,sn,0,cc},A[9]={},dot[3]={},kk[9];
 for(int k=0;k<3;k++)for(int col=0;col<3;col++){
  dot[col]+=(float)d[k]*Q[3*k+col];
  for(int row=0;row<3;row++)A[3*row+col]+=B[3*k+row]*Q[3*k+col];
 }
 for(int row=0;row<3;row++)for(int col=0;col<3;col++)kk[3*row+col]=A[3*row+col]-dl[row]*dot[col];
 const int ii[4]={0,1,2,0},jj[4]={0,1,2,2};
 for(int u2=0;u2<4;u2++){int i=ii[u2],z=jj[u2];
  s.T[4*u2]=kk[3*i]*kk[3*z];s.T[4*u2+1]=kk[3*i+1]*kk[3*z+1];s.T[4*u2+2]=kk[3*i+2]*kk[3*z+2];
  s.T[4*u2+3]=kk[3*i]*kk[3*z+2]+kk[3*i+2]*kk[3*z];
 }
 float cp=st>1e-12f?clip(((float)d[0]*cx+(float)d[1]*sx)/st,-1,1):1.f;
 s.cm[0]=1;s.cm[1]=cp;for(int m=2;m<=32;m++)s.cm[m]=(2.f*cp)*s.cm[m-1]-s.cm[m-2];
 s.sw[0]=(1.f-ah)*(1.f-ac);s.sw[1]=(1.f-ah)*ac;s.sw[2]=ah*(1.f-ac);s.sw[3]=ah*ac;
 s.cell[0]=ir*nc+j;s.cell[1]=s.cell[0]+1;s.cell[2]=s.cell[0]+nc;s.cell[3]=s.cell[2]+1;
 s.prof[0]=expf(-s.h/8.f);s.prof[1]=expf(-s.h/height);s.prof[2]=fmaxf(0.f,1.f-fabsf(s.h-25.f)/15.f);
#ifdef PROFILE
 if(profile_mask&1)s.prof[0]=profile_value(s.h,profile_table,0);
 if(profile_mask&2)s.prof[2]=profile_value(s.h,profile_table,1);
#endif
#ifdef STRATO
 float st_t=(s.h-st_center)/st_sigma;s.st_prof=expf(-.5f*(st_t*st_t))/st_norm;
#endif
 if(first){
  s.mu=dot[2];s.exd=dot[0];s.eyd=dot[1];
  DS height=dx(rr,{6371,0});if(dv(height)<0)height={0,0};DS rad=da({6371,0},height);
  DS mg=dn(dr(ddiv(dm(height,da({2.f*6371.f,0},height)),dm(rad,rad))));
  DS delta=dx(ddiv(q[2],rr),mg);s.lit=dv(delta)>=0;if(!s.lit)delta={0,0};
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

#ifdef SH_AUX_FP16
// Pointer-valued ABI is unchanged; under this flag only, its allocation is
// uint16 FP16 storage. Never dereference the pointer as float in this branch.
__device__ __forceinline__ float sh_half_load(const float *ptr,int64_t i){
 uint16_t bits=((const uint16_t*)ptr)[i];float v;
 asm("cvt.f32.f16 %0, %1;" : "=f"(v) : "h"(bits));return v;
}
#endif

extern "C" __global__ void sos_fused(
 int cells,int nh,int nc,int nq,int nsh,int nsx,int P_FIRST,int P_PHASE,int P_BOUNDARY,
 const int64_t*cell_off,const double*starts,const double*targets,const double*dirs,const float*weights,
 const uint32_t*edge_off,const double*edges,const uint16_t*lower,
 const int32_t*end_j,const float*end_f,const float*end_col,const uint8_t*hits,
 const double*gu,const double*gw,const double*aa,const double*bb,const double*dd,
 const float*h,const float*chi,const float*hr,const float*ha,const float*ext,const float*asca,
 const float*sun_h,const float*sun_col,const float*gm,const int64_t*goff,const float*gdir,const float*gval,
 float height,float aerosol_g,float mix,double albedo,
 const float*src,const double*boundary,double*moments,double*flux,float*query SH_PARAM STRATO_PARAM PROFILE_PARAM SECTION_PARAM){
#ifdef SPECIALIZE
 constexpr int first=K_FIRST,phase=K_PHASE,with_boundary=K_BOUNDARY;
#endif
 int cell=blockIdx.x;if(cell>=cells)return;int tid=threadIdx.x,lane=tid&31,warp=tid>>5,nwarps=blockDim.x>>5;
#ifdef EXP_SECTION_CLOCKS
 unsigned long long section_cycles[8]={},section_begin=0;
 unsigned long long total_begin=0;if(lane==0)total_begin=clock64();
#endif
 extern __shared__ double shared[];
 int nm=phase==0?565*TW:0;
 double *acc=shared,*ground=acc+nm;
 float *Y=(float*)(ground+TW),*rads=Y+nwarps*K,*err=rads+nwarps*4*TW,*cosdest=err+nwarps*4*TW;
 Sample *samples=(Sample*)(cosdest+nwarps*33);
#ifdef RAY_CONST_SHARED
 float *ray_constants=(float*)(samples+nwarps)+warp*15;
#endif
#if defined(GROUND_PHASE_BATCH) && !defined(GROUND_BATCH_SHFL)
 float *ground_phase=(float*)(samples+nwarps)
#ifdef RAY_CONST_SHARED
  +nwarps*15
#endif
  +warp*32;
#endif
#ifdef POINT_BATCH2
 PointData *pending=(PointData*)((float*)(samples+nwarps)
#ifdef RAY_CONST_SHARED
  +nwarps*15
#endif
#if defined(GROUND_PHASE_BATCH) && !defined(GROUND_BATCH_SHFL)
  +nwarps*32
#endif
  );
#endif
 for(int k=tid;k<nm+TW;k+=blockDim.x)acc[k]=0.;SECTION_SYNC(__syncthreads());
 const double *p=starts+3*cell,*target=targets+9*cell;
 float B[9];for(int k=0;k<9;k++)B[k]=(float)target[k];
 int64_t begin=cell_off[cell],end=cell_off[cell+1];
 for(int64_t round=begin;round<end;round+=nwarps){
  int64_t ray=round+warp;float *y=Y+warp*K,*rad=rads+warp*4*TW,*rc=err+warp*4*TW,*cd=cosdest+warp*33;
  if(ray<end){
   const double *d=dirs+3*ray;
#ifdef RAY_CONST_SHARED
   // global_st, dl[3], pu[4], se[3], direct[4]: 15 floats/warp.
   float *dl=ray_constants+1,*pu=ray_constants+4,*se=ray_constants+8,*direct=ray_constants+11;
   if(lane==0){ray_constants[0]=(float)sqrt(fmax(0.,1.-d[2]*d[2]));
   for(int j=0;j<3;j++){double x=0;for(int k=0;k<3;k++)x+=target[3*k+j]*d[k];dl[j]=(float)x;}
   pu[0]=1.f-dl[0]*dl[0];pu[1]=1.f-dl[1]*dl[1];pu[2]=1.f-dl[2]*dl[2];pu[3]=-dl[0]*dl[2];
   float sun_dot=(dl[0]*B[6]+dl[1]*B[7])+dl[2]*B[8];for(int k=0;k<3;k++)se[k]=B[6+k]-dl[k]*sun_dot;
   direct[0]=(pu[0]-se[0]*se[0])/2;direct[1]=(pu[1]-se[1]*se[1])/2;direct[2]=(pu[2]-se[2]*se[2])/2;direct[3]=(pu[3]-se[0]*se[2])/2;
   }
   SECTION_SYNC(__syncwarp());
   float global_st=ray_constants[0];
#else
   float dl[3]={};float global_st=(float)sqrt(fmax(0.,1.-d[2]*d[2]));
   for(int j=0;j<3;j++){double x=0;for(int k=0;k<3;k++)x+=target[3*k+j]*d[k];dl[j]=(float)x;}
   float pu[4]={1.f-dl[0]*dl[0],1.f-dl[1]*dl[1],1.f-dl[2]*dl[2],-dl[0]*dl[2]},direct[4];
   float sun_dot=(dl[0]*B[6]+dl[1]*B[7])+dl[2]*B[8],se[3];for(int k=0;k<3;k++)se[k]=B[6+k]-dl[k]*sun_dot;
   direct[0]=(pu[0]-se[0]*se[0])/2;direct[1]=(pu[1]-se[1]*se[1])/2;direct[2]=(pu[2]-se[2]*se[2])/2;direct[3]=(pu[3]-se[0]*se[2])/2;
#endif
   SECTION_BEGIN()
   sh_basis(d[2],lane,y,aa,bb,dd);
   SECTION_END(2)
   if(lane==0){double st=sqrt(fmax(0.,1.-d[2]*d[2]));float cp=st>1e-12?(float)(d[0]/st):1.f;cp=clip(cp,-1,1);
    cd[0]=1;cd[1]=cp;for(int m=2;m<=32;m++)cd[m]=(2.f*cp)*cd[m-1]-cd[m-2];
    for(int k=0;k<4*TW;k++){rad[k]=0;rc[k]=0;}
   }
   SECTION_SYNC(__syncwarp());
   float cache[4][TW];
#ifdef STRATO
   float scache[4][TW];
#endif
   int tags[4]={-1,-1,-1,-1};
#ifdef LANE_ATTEN
   float tau_lane=0.f,tc_lane=0.f;
#else
   float tau[TW]={},tc[TW]={};
#endif
   Sample &s=samples[warp];
   float sunphase=hg((float)d[2],aerosol_g);
   for(uint32_t seg=edge_off[ray];seg+1<edge_off[ray+1];seg++){
    double a=edges[seg],b=edges[seg+1];if(b-a<=1e-8)continue;
    bool low=(lower[ray]>>(seg-edge_off[ray]))&1;
    for(int step=0;step<nq;step++){
     SECTION_BEGIN()
#ifdef POINT_BATCH2
     if((step&1)==0){
      if(lane<2 && step+lane<nq){
       PointData &dst=lane==0?static_cast<PointData&>(s):pending[warp];
       point(dst,p,d,B,dl,global_st,a,b,low,gu[step+lane],gw[step+lane],h,nh,chi,nc,height,sun_h,nsh,sun_col,nsx,first
#ifdef PROFILE
      ,profile_table,profile_mask
#endif
#ifdef STRATO
      ,st_sun_col,st_center,st_sigma,st_norm
#endif
      );
      }
     }else if(lane==0)copy_point(s,pending[warp],first);
#else
     if(lane==0)point(s,p,d,B,dl,global_st,a,b,low,gu[step],gw[step],h,nh,chi,nc,height,sun_h,nsh,sun_col,nsx,first
#ifdef PROFILE
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
     SECTION_BEGIN_IF(first)
#ifdef GROUND_PHASE_BATCH
     if(first){
      float accw=0.f;
#ifndef LANE_GROUND
      if(lane==0)for(int w=0;w<TW;w++)s.c32[0][w]=0.f;
#endif
      for(int corner=0;corner<4;corner++){int cc=s.cell[corner];
       for(int64_t j0=goff[cc];j0<goff[cc+1];j0+=32){
        int64_t j=j0+lane;
#ifdef GROUND_BATCH_SHFL
        float phase_lane=0.f;
#endif
        if(j<goff[cc+1]){float m0=s.mu*gdir[3*j+2]+s.exd*gdir[3*j],m1=fabsf(s.eyd)*gdir[3*j+1];
#ifdef GROUND_BATCH_SHFL
         phase_lane=(hg(
#else
         ground_phase[lane]=(hg(
#endif
clip(m0+m1,-1,1),aerosol_g)+hg(clip(m0-m1,-1,1),aerosol_g))/2.f;}
        __syncwarp();
        int count=(int)((goff[cc+1]-j0)<32?(goff[cc+1]-j0):32);
        for(int k=0;k<count;k++){
#ifdef GROUND_BATCH_SHFL
         float phase_k=__shfl_sync(0xffffffff,phase_lane,k);
#define BATCH_PHASE phase_k
#else
#define BATCH_PHASE ground_phase[k]
#endif
#ifdef LANE_GROUND
         if(lane<TW)accw+=(s.sw[corner]*BATCH_PHASE)*gval[(j0+k)*FW+GW(lane)];
#else
         if(lane==0)for(int w=0;w<TW;w++)s.c32[0][w]+=(s.sw[corner]*BATCH_PHASE)*gval[(j0+k)*FW+GW(w)];
#endif
#undef BATCH_PHASE
        }
        __syncwarp();
       }
      }
#ifdef LANE_GROUND
      if(lane<TW)s.c32[0][lane]=accw;
#endif
      __syncwarp();
     }
#elif defined(LANE_GROUND)
     if(first){
      // Lane w owns wavelength w (TW<=31); lane 31 evaluates the paired phase
      // once per ground direction. Per-wavelength corner/direction order and
      // every product are unchanged, so the sums are bit-identical.
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
#ifdef STRATO
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
#endif
     SECTION_END_IF(1,first)
#ifdef LANE_SOURCE_INIT
     // Includes lane-0 STRATO/scalar ground writes before other lanes read.
     SECTION_SYNC(__syncwarp());
#endif
     SECTION_BEGIN()
#ifdef LANE_SOURCE_INIT
     if(lane<TW){int w=lane;
#else
     if(lane==0)for(int w=0;w<TW;w++){
#endif
      s.logs[w]=0;for(int k=0;k<4;k++)s.val[w][k]=0;
      if(first){
       float od=0;for(int k=0;k<3;k++)od+=s.sc[k]*ext[k*FW+GW(w)];
#ifdef STRATO
       od+=s.st_sc*st_ext[GW(w)];
#endif
       float sun=s.lit?expf(-od):0.f,ga=s.c32[0][w];
       float beam=(s.prof[1]*asca[GW(w)])*(sun*sunphase+ga),ray=(sun*s.prof[0])*ext[GW(w)];
       for(int k=0;k<4;k++)s.val[w][k]=ray*(mix*(float)(3/(8*PI))*direct[k]+(1.f-mix)*(float)(1/(8*PI))*pu[k])+(beam*pu[k])/2.f;
#ifdef STRATO
       float sb=(s.st_prof*st_sca[GW(w)])*(sun*mie_phase((float)d[2],GW(w),st_chi)+s.st_c32[0][w]);
       for(int k=0;k<4;k++)s.val[w][k]+=(sb*pu[k])/2.f;
#endif
      }
     }
     SECTION_END(3)
     SECTION_SYNC(__syncwarp());
     SECTION_BEGIN()
     if(!first && tags[0]>=0){
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
     #pragma unroll
     for(int corner=0;corner<4;corner++){
      int cc=s.cell[corner],ir=s.ir+corner/2;
      if(!first && tags[corner]!=cc){
       #pragma unroll
       for(int w=0;w<TW;w++)cache[corner][w]=0.f;
#ifdef STRATO
       for(int w=0;w<TW;w++)scache[corner][w]=0.f;
#endif
       // Degree-major loads: consecutive active lanes read consecutive SHs.
       // Each lane still sums l=m..32 in precisely the reference order.
       for(int l=0;l<=32;l++)if(lane<=l){int k=ix(l,lane);float pref=y[k];
        #pragma unroll
#ifdef SH_AUX
        #if defined(SH_AUX_PAIR)
        for(int w=0;w<TW;w+=2){
         float2 pair=((const float2*)src_sh)[((int64_t)cc*((FW+1)/2)+w/2)*KPAD+k];
         cache[corner][w]+=pref*pair.x;
         if(w+1<TW)cache[corner][w+1]+=pref*pair.y;
        }
#elif defined(SH_AUX_FP16)
        for(int w=0;w<TW;w++)cache[corner][w]+=pref*sh_half_load(src_sh,((int64_t)cc*FW+GW(w))*KPAD+k);
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
        for(int w=0;w<TW;w++)scache[corner][w]+=pref*src[((int64_t)cc*1132+571+k)*FW+GW(w)];
#endif
       }
       if(lane==0){
        #pragma unroll
#ifdef SH_AUX
        #if defined(SH_AUX_PAIR)
        for(int w=0;w<TW;w+=2){
         float2 pair=((const float2*)src_sh)[((int64_t)cc*((FW+1)/2)+w/2)*KPAD+560];
         float v=0.f;v+=y[560]*pair.x;s.c32[corner][w]=v;
         if(w+1<TW){float v1=0.f;v1+=y[560]*pair.y;s.c32[corner][w+1]=v1;}
        }
#elif defined(SH_AUX_FP16)
        for(int w=0;w<TW;w++){float v=0.f;v+=y[560]*sh_half_load(src_sh,((int64_t)cc*FW+GW(w))*KPAD+560);s.c32[corner][w]=v;}
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
        for(int w=0;w<TW;w++){float v=0.f;v+=y[560]*src[((int64_t)cc*1132+1131)*FW+GW(w)];s.st_c32[corner][w]=v;}
#endif
       }tags[corner]=cc;
      }
#ifdef TRANSPORT_WAVE_LANES
      // Do not access cache[corner][lane]: retain compile-time wave indexes
      // so the Fourier cache can stay register promoted. Every lane executes
      // every shuffle; the selected owner keeps one reduced wavelength.
      float own_aero=0.f;
#ifdef STRATO
      float own_saero=0.f;
#endif
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
      bool bad_trace=false;
      if(lane<TW){
       int w=lane;float aero=own_aero;
#ifdef STRATO
       float saero=own_saero;
#endif
       float rm[4];for(int k=0;k<4;k++)rm[k]=first?gm[((int64_t)cc*4+k)*FW+GW(w)]:src[((int64_t)cc*
#ifdef STRATO
          1132
#else
          571
#endif
          +k)*FW+GW(w)];
       float density=first?s.prof[0]:hr[ir];if(!first)aero*=ha[ir];
#ifdef STRATO
       if(!first)saero*=st_density[ir];
#endif
       float tr=(rm[0]+rm[1])+rm[2],tensors[3];
       #pragma unroll
       for(int component=0;component<4;component++){
#ifdef PHASE1_TRACE3
        if(component==3 && phase==1)continue;
#endif
        float rot=0.f;for(int k=0;k<4;k++)rot+=s.T[4*component+k]*rm[k];
        float tensor=density*(mix*rot+((1.f-mix)/3.f*tr)*pu[component]);
        tensor+=(aero*pu[component])/2.f;
#ifdef STRATO
        if(!first)tensor+=(saero*pu[component])/2.f;
#endif
        s.val[w][component]+=s.sw[corner]*tensor;
        if(component<3)tensors[component]=tensor;
       }
       if(!first){float trace=(tensors[0]+tensors[1])+tensors[2];
        bad_trace=trace<=0;
        s.logs[w]+=s.sw[corner]*logf(fmaxf(trace,1.40129846e-45f));
       }
      }
      // Replace lane-0 per-wave bit clears with an exact integer ballot.
      uint32_t bad=__ballot_sync(0xffffffff,bad_trace);
      if(lane==0 && !first)s.positive&=~bad;
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
#ifdef PHASE1_TRACE3
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
       if(lane==0 && !first){float trace=(tensor+t1)+t2;if(trace<=0)s.positive&=~(1u<<w);
         s.logs[w]+=s.sw[corner]*logf(fmaxf(trace,1.40129846e-45f));}
      }
#endif
     }
     SECTION_END(3)
     SECTION_SYNC(__syncwarp());
     SECTION_BEGIN()
#ifdef LANE_ATTEN
     if(lane<TW){int w=lane;{
#else
     if(lane==0){
     #pragma unroll
     for(int w=0;w<TW;w++){
#endif
      if(!first){float tr=(s.val[w][0]+s.val[w][1])+s.val[w][2];
       float scale=(s.positive&(1u<<w))&&tr>0?expf(s.logs[w])/fmaxf(tr,1.40129846e-45f):1.f;
       s.logs[w]=scale;
      }else s.logs[w]=1.f;
      float dt=0;for(int k=0;k<3;k++)dt+=(s.prof[k]*s.ds)*ext[k*FW+GW(w)];
#ifdef STRATO
      dt+=(s.st_prof*s.ds)*st_ext[GW(w)];
#endif
#ifdef LANE_ATTEN
      float mid=(tau_lane-tc_lane)+dt/2.f;s.att[w]=s.ds*expf(-mid);kah(tau_lane,tc_lane,dt);
#else
      float mid=(tau[w]-tc[w])+dt/2.f;s.att[w]=s.ds*expf(-mid);kah(tau[w],tc[w],dt);
#endif
     }}
     SECTION_END(4)
     SECTION_SYNC(__syncwarp());
#ifdef PHASE1_TRACE3
     if(lane<(phase==1?3:4)){
#else
     if(lane<4){
#endif
      #pragma unroll
      for(int w=0;w<TW;w++){float value=s.val[w][lane]*s.logs[w];kah(rad[4*w+lane],rc[4*w+lane],value*s.att[w]);}
     }
     SECTION_SYNC(__syncwarp());
    }
   }
   if(lane==0 && with_boundary){int j=end_j[ray];float f=end_f[ray];
    for(int w=0;w<TW;w++){float e=(1.f-f)*(float)boundary[j*FW+GW(w)];e+=f*(float)boundary[(j+1)*FW+GW(w)];
     float od=0;for(int k=0;k<3;k++)od+=end_col[3*ray+k]*ext[k*FW+GW(w)];
#ifdef STRATO
     od+=st_end_col[ray]*st_ext[GW(w)];
#endif
     e*=expf(-od)*(float)hits[ray];
#ifdef PHASE1_TRACE3
     for(int k=0;k<(phase==1?3:4);k++)rad[4*w+k]+=(e*pu[k])/2.f;
#else
     for(int k=0;k<4;k++)rad[4*w+k]+=(e*pu[k])/2.f;
#endif
    }
   }
   if(phase==2){SECTION_SYNC(__syncwarp());for(int k=lane;k<4*TW;k+=32)
#ifdef WAVE_TILED
query[ray*4*FW+GW(k/4)*4+k%4]=rad[k];
#else
query[ray*4*FW+k]=rad[k];
#endif
}
  }
#ifdef PHASE2_NOBARRIER
  // Phase 2 has no cross-warp data: warp-private buffers only need the warp
  // to finish reading rad/y before its next ray overwrites them.
  if(phase==2)SECTION_SYNC(__syncwarp()); else
#endif
  SECTION_SYNC(__syncthreads());
  if(phase!=2){
   SECTION_BEGIN()
   for(int k=tid;k<nm;k+=blockDim.x){int moment=k/TW,w=k%TW;
    for(int wp=0;wp<nwarps;wp++)if(round+wp<end){float *rr=rads+wp*4*TW;float wt=weights[round+wp],v;
     if(moment<4)v=rr[4*w+moment]*wt;
     else {int sk=moment-4;
#ifdef LM_LUT
      int m=LM.m[sk];
#else
      int l=(int)((sqrtf(8.f*sk+1.f)-1.f)/2.f);while(ix(l+1,0)<=sk)l++;while(ix(l,0)>sk)l--;int m=sk-ix(l,0);
#endif
      float tr=(rr[4*w]+rr[4*w+1])+rr[4*w+2];v=(tr*wt)*(Y[wp*K+sk]*cosdest[wp*33+m]);}
     acc[k]+=(double)v;
    }
   }
   for(int w=tid;w<TW;w+=blockDim.x)for(int wp=0;wp<nwarps;wp++)if(round+wp<end){
    int64_t rridx=round+wp;const double*d=dirs+3*rridx;double dlz=0;for(int k=0;k<3;k++)dlz+=target[3*k+2]*d[k];
    float *rr=rads+wp*4*TW,tr=(rr[4*w]+rr[4*w+1])+rr[4*w+2],wt=weights[rridx]*fmaxf(0.f,(float)dlz);
    ground[w]+=(double)(tr*wt);
   }
   SECTION_END(5)
  }
#ifdef PHASE2_NOBARRIER
  if(phase!=2)
#endif
  SECTION_SYNC(__syncthreads());
 }
 if(phase==0)for(int k=tid;k<nm;k+=blockDim.x)
#ifdef WAVE_TILED
moments[((int64_t)cell*565+k/TW)*FW+GW(k%TW)]=acc[k];
#else
moments[(int64_t)cell*nm+k]=acc[k];
#endif
 if(phase!=2)for(int w=tid;w<TW;w+=blockDim.x)flux[cell*FW+GW(w)]=ground[w]*(albedo/PI);
#ifdef EXP_SECTION_CLOCKS
 if(lane==0){
  section_cycles[7]=clock64()-total_begin;
  int64_t slot=(((int64_t)cell*(FW/TW)+(int)blockIdx.y)*nwarps+warp)*8;
  for(int k=0;k<8;k++)section_clocks[slot+k]=section_cycles[k];
 }
#endif
}
