#!/usr/bin/env python3
"""Gated SOS stage 4. Stage A uses the unchanged, full HG SH physics.

No old artifact is written. Incoming and spatial controls are independent;
all path caches are disposable inside results/sos_full only.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='6'
import argparse, gc, hashlib, json, math, resource, shutil, subprocess, sys, threading, time
from pathlib import Path
import numpy as np
import sos_sky2 as sh
from PIL import Image, ImageDraw
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='6'
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/sos_full'; DOC=ROOT/'docs/SOS_QUAD.md'
sh.OUT=OUT; sh.CPP=sh.CPP.replace('int mid=A.nr/2;\n std::thread t([&](){runpart<W>(A,0,mid,norm);});runpart<W>(A,mid,A.nr,norm);t.join();',
    'std::vector<std::thread> ts; for(int k=0;k<6;k++)ts.emplace_back([&,k](){runpart<W>(A,A.nr*k/6,A.nr*(k+1)/6,norm);});for(auto &t:ts)t.join();')
R,Z=sh.R,sh.Z

COMPACT_CPP=r'''
static void compactpart(int nr,int L,int nchi,int ns,int density,const int64_t*off,const int32_t*base,
 const float*p,const double*nu,const float*src,double*out,int lo,int hi){
 Norm norm(L);int K=norm.K,M=L+1,S=K+6*density,C=M+3*density,P=4+2*density;std::vector<double>Y(K);
 std::vector<int>stamp(ns,-1);std::vector<float>cache((int64_t)ns*C);
 for(int ray=lo;ray<hi;ray++){
  norm.eval(nu[ray],1.,Y.data());double sum=0;
  for(int64_t n=off[ray];n<off[ray+1];n++){
   const float*q=p+P*n;double ah=q[0],ac=q[1],cp=q[2],wt=q[3];
   int cells[4]={base[n],base[n]+1,base[n]+nchi,base[n]+nchi+1};
   double sw[4]={(1-ah)*(1-ac),(1-ah)*ac,ah*(1-ac),ah*ac};
   for(int c=0;c<4;c++)if(stamp[cells[c]]!=ray){
    float*v=cache.data()+(int64_t)cells[c]*C;std::fill(v,v+C,0.f);
    const float*s=src+(int64_t)cells[c]*S;
    if(density)for(int l=0;l<=2;l++)for(int m=0;m<=l;m++){int k=ix(l,m);v[M+m]+=(float)Y[k]*s[k];}
    for(int l=0;l<=L;l++)for(int m=0;m<=l;m++){int k=ix(l,m);v[m]+=(float)Y[k]*s[k+6*density];}
    stamp[cells[c]]=ray;
   }
   double cm=1,prev=cp,val=0;
   for(int m=0;m<M;m++){
    if(m==1){cm=cp;prev=1;}else if(m>1){double next=2*cp*cm-prev;prev=cm;cm=next;}
    for(int c=0;c<4;c++){
     val+=cm*sw[c]*cache[(int64_t)cells[c]*C+m]*(density?q[5]:1.);
     if(density && m<=2)val+=cm*sw[c]*cache[(int64_t)cells[c]*C+M+m]*q[4];
    }
   }
   sum+=wt*val;
  }
  out[ray]=sum;
 }
}
extern "C" void compact_transport(int nr,int L,int nchi,int ns,int density,const int64_t*off,const int32_t*base,
 const float*p,const double*nu,const float*src,double*out){
 std::vector<std::thread>ts;for(int k=0;k<6;k++)ts.emplace_back([&,k](){compactpart(nr,L,nchi,ns,density,off,base,p,nu,src,out,nr*k/6,nr*(k+1)/6);});for(auto&t:ts)t.join();
}
'''
sh.CPP+=COMPACT_CPP

class Kernel(sh.Kernel):
    def __init__(self):
        super().__init__();self.lib.compact_transport.argtypes=[sh.ctypes.c_int]*5+[sh.ctypes.c_void_p]*6
    def transport(self,z,solver,source,first=False):
        density=solver.geo.cfg.get('density',False)
        if z['point'].shape[1] not in (4,6):
            if not density or first:return super().transport(z,solver,source,first)
            z=compact_block(z,solver)
        assert solver.nw==1 and not first
        source=np.ascontiguousarray(source,'f4');out=np.empty((len(z['nu']),1))
        arrays=[np.ascontiguousarray(z[k]) for k in ('off','base','point','nu')]+[source,out]
        self.lib.compact_transport(len(out),solver.L,len(solver.geo.chi),solver.geo.ns,int(density),*[a.ctypes.data for a in arrays]);return out

def compact_block(raw,solver):
    p=raw['point'];ic=raw['base']%len(solver.geo.chi)
    ch=solver.geo.chi[ic]+p[:,1]*(solver.geo.chi[ic+1]-solver.geo.chi[ic])
    nu=np.repeat(raw['nu'],np.diff(raw['off']));den=np.sin(ch)*np.sqrt(np.maximum(0,1-nu*nu))
    cp=np.divide(p[:,2]-np.cos(ch)*nu,den,out=np.ones(len(p)),where=abs(den)>1e-12)
    weight=p[:,4]*np.exp(-p[:,6:9].astype(float)@solver.ext[:,0])
    fields=[p[:,0],p[:,1],np.clip(cp,-1,1),weight]
    if solver.geo.cfg.get('density',False):fields.extend([np.exp(-np.maximum(0,p[:,5])/8),np.exp(-np.maximum(0,p[:,5])/1.2)])
    return dict(off=raw['off'],base=raw['base'],nu=raw['nu'],point=np.column_stack(fields).astype('f4'))

def save(path,value): sh.dump(path,value)
def now(): return time.perf_counter()

def ring_edges(r,chi,phi,heights,prepared=None):
    """View-plane intersection with the exact shadow-cylinder/shell ring.

    The night-side solar shadow has x²+y²=R², z<0. At shell R+h
    its boundary is the ring z=-sqrt((R+h)²-R²). Project each of its
    two intersections with the ray's azimuth plane onto local mu.
    These are real endpoints of illumination in each density layer.
    """
    if prepared is None:
        p,up,ex=sh.old.probe.frame(np.array(r),np.array(chi));ey=np.cross(up,ex)
        levels=[-np.sqrt(max(0,(R+h)**2-R**2)) for h in heights]
    else:p,up,ex,ey,levels=prepared
    horizontal=np.cos(phi)*ex+np.sin(phi)*ey
    # Three scalar products in precisely np.cross's original order.
    normal=np.array([up[1]*horizontal[2]-up[2]*horizontal[1],
                     up[2]*horizontal[0]-up[0]*horizontal[2],
                     up[0]*horizontal[1]-up[1]*horizontal[0]])
    amp=R*np.hypot(normal[0],normal[1])
    if amp<1e-10:return []
    angle=np.arctan2(normal[1],normal[0]);answer=[]
    for z in levels:
        target=-normal[2]*z/amp
        if abs(target)>1:continue
        delta=np.arccos(target)
        for az in (angle-delta,angle+delta):
            q=np.array([R*np.cos(az),R*np.sin(az),z]);d=q-p;length=np.sqrt(d@d)
            if length<1e-8:continue
            d/=length
            if d@horizontal<=1e-10:continue
            # Reject rings hidden behind the opaque Earth.
            b=p@d;disc=b*b-r*r+R*R
            hit=-b-np.sqrt(max(0,disc))
            if disc>=0 and b<0 and hit>=-1e-7 and hit<length-1e-5:continue
            answer.append(float(d@up))
    return answer

def incoming(r,chi,nmu,nphi,a=4.,rings=True):
    mg=-np.sqrt(max(0,1-(R/r)**2));sa=np.sinh(a);u,wu=sh.old.probe.gl(nmu)
    mus=[];phis=[];weights=[]
    sinchi=np.sin(chi);coschi=np.cos(chi);rp=r*sinchi
    prepared=None
    if rings:
        p,up,ex=sh.old.probe.frame(np.array(r),np.array(chi));ey=np.cross(up,ex)
        levels=[-np.sqrt(max(0,(R+h)**2-R**2)) for h in (0.,1.2,8.,25.,40.,80.)]
        prepared=(p,up,ex,ey,levels)
    for j,phi in enumerate(np.linspace(0,np.pi,nphi+1)):
        edges=[-1.,1.,mg,0.,mg-.05234,mg+.05234,mg-.0061,mg+.0061]
        if rp>R+1e-8 and abs(sinchi)>1e-8:
            kk=R/np.sqrt(rp*rp-R*R)
            for sign in (-1,1):
                t=np.cos(phi)*coschi/sinchi+sign*np.sin(phi)/(kk*sinchi)
                mu=t/np.sqrt(1+t*t);st=np.sqrt(1-mu*mu)
                dx=mu*sinchi-st*np.cos(phi)*coschi;dy=-st*np.sin(phi)
                dz=mu*coschi+st*np.cos(phi)*sinchi
                tmin=-rp*dx/max(dx*dx+dy*dy,1e-25)
                if abs(dx)+abs(dy)<1e-10 or (tmin>0 and r*coschi+tmin*dz<0):edges.append(mu)
        if phi<np.pi/6:
            mu=coschi/np.sqrt(coschi**2+np.cos(phi)**2*sinchi**2)
            edges.extend([mu,mu-.012,mu+.012])
        if rings:edges+=ring_edges(r,chi,phi,(0.,1.2,8.,25.,40.,80.),prepared)
        edges=np.unique(np.clip(edges,-1,1));delta=edges-mg
        tt=np.sign(delta)*np.arcsinh(np.abs(delta)/np.where(delta<0,1+mg,1-mg)*sa)/a
        # Every interval is independent. Keep interval-major / GL-node-minor
        # ordering and the exact scalar operation order of the accepted loop.
        keep=tt[1:]-tt[:-1]>=1e-10
        left=tt[:-1][keep,None];right=tt[1:][keep,None]
        t=left+(right-left)*u;den=np.where(t<0,1+mg,1-mg)
        mus.append((mg+np.sign(t)*den*np.sinh(a*np.abs(t))/sa).ravel())
        weights.append(((right-left)*wu*den*a*np.cosh(a*np.abs(t))/sa*(2*np.pi/nphi)*(.5 if j in (0,nphi) else 1.)).ravel())
        phis.append(np.full(t.size,phi))
    mu=np.concatenate(mus);phi=np.concatenate(phis);st=np.sqrt(np.maximum(0,1-mu*mu))
    return np.column_stack([st*np.cos(phi),st*np.sin(phi),mu]),np.concatenate(weights)

def grids(rscale=1,cscale=1):
    h,ch=sh.old.grids(1)
    if rscale==2:h=np.sort(np.r_[h,(h[:-1]+h[1:])/2])
    if cscale>1:
        degrees=np.rad2deg(ch)
        # Refine only the requested twilight band, preserving outer nodes.
        ch=np.deg2rad(np.unique(np.r_[degrees[(degrees<85)|(degrees>110)],np.arange(85,110+.1/cscale,.5/cscale)]))
    return h,ch

def readonly_geometry(scale=1,angular=1):
    """Read the prior incoming cache without invoking any mutating method."""
    g=object.__new__(sh.Geometry);g.h,g.chi=sh.old.grids(scale);g.ns=len(g.h)*len(g.chi)
    g.folder=ROOT/f'results/sos_sky2/geo_s{scale}_q{angular}';g.s,g.q=scale,angular
    with np.load(g.folder/'incoming.npz') as z:
        for attr,key in (('offset','offset'),('dirs','dirs'),('w','weight'),('ground','ground'),('endchi','endchi'),('column','column'),('surfacecol','surfacecol'),('surfacelit','surfacelit')):setattr(g,attr,z[key])
    g.nr=len(g.dirs);g.meta=json.loads((g.folder/'meta.json').read_text());g.sun=sh.old.SunGeometry(scale,g.folder)
    g.blocks=json.loads((g.folder/'paths.json').read_text())['blocks'];g.ram_blocks={}
    return g

class Geometry(sh.Geometry):
    def __init__(self,cfg,name):
        self.cfg=cfg;self.s=self.q=1;self.h,self.chi=grids(cfg.get('rscale',1),cfg.get('cscale',1));self.ns=len(self.h)*len(self.chi)
        self.folder=OUT/f'geometry_{cfg.get("geometry",name)}';self.folder.mkdir(parents=True,exist_ok=True)
        self.sun=sh.old.SunGeometry(1,self.folder);self.blocks=[];self.ram_blocks={}
        f=self.folder/'incoming.npz';t=now()
        if not f.exists():self.build_incoming()
        with np.load(f) as z:
            for attr,key in (('offset','offset'),('dirs','dirs'),('w','weight'),('ground','ground'),('endchi','endchi'),('column','column'),('surfacecol','surfacecol'),('surfacelit','surfacelit')):setattr(self,attr,z[key])
        self.nr=len(self.dirs);self.meta=json.loads((self.folder/'meta.json').read_text());self.meta['load_seconds']=now()-t
    def rays(self,cell):
        p,d=super().rays(cell)
        # Cached local directions are float32; the shell intersection routines
        # require unit vectors, particularly when the origin is on the ground.
        d/=np.linalg.norm(d,axis=-1)[:,None]
        return p,d
    def build_incoming(self):
        t=now();fields={k:[] for k in ('dirs','weight','ground','endchi','column','surfacecol','surfacelit')};offset=[0];solid=0.
        for cell in range(self.ns):
            ih,ic=divmod(cell,len(self.chi));loc,w=incoming(R+self.h[ih],self.chi[ic],self.cfg['nmu'],self.cfg['nphi'],self.cfg['a'],self.cfg.get('rings',True))
            solid=max(solid,abs(w.sum()/(4*np.pi)-1))
            p,up,ex=sh.old.probe.frame(np.array(R+self.h[ih]),self.chi[ic]);ey=np.cross(up,ex)
            d=loc[:,0,None]*ex+loc[:,1,None]*ey+loc[:,2,None]*up;p=np.broadcast_to(p,d.shape)
            end=sh.old.probe.boundaries(p,d)[:,-1];qe=p+end[:,None]*d;re=np.linalg.norm(qe,axis=-1)
            ground=re<R+1e-5;ch=np.arccos(np.clip(qe[:,2]/re,-1,1));col=sh.old.total_columns(p,d,24);sc,sl=self.sun.lookup(qe,np.maximum(re,R))
            for key,val in zip(fields,(loc.astype('f4'),w,ground,ch,col.astype('f4'),sc.astype('f4'),sl)):fields[key].append(val)
            offset.append(offset[-1]+len(w))
            if cell%100==0:print(f'INCOMING {self.folder.name} {cell}/{self.ns} {now()-t:.1f}s',flush=True)
        np.savez_compressed(self.folder/'incoming.npz',offset=np.array(offset,np.int64),**{k:np.concatenate(v) for k,v in fields.items()})
        save(self.folder/'meta.json',dict(config=self.cfg,nr=len(self.h),nchi=len(self.chi),cells=self.ns,incoming_rays=offset[-1],incoming_seconds=now()-t,sun_seconds=self.sun.build_seconds,solid_angle_max_relative=solid,path_n=sh.PATH_N))
    def ensure_paths(self):
        if self.blocks:return
        t=now();size=points=0;last=t;budget=16*1024**3;raw_seconds=first_seconds=fold_seconds=0.
        self.first_values=np.empty((self.nr,1));maxcheck=0.
        for start in range(0,self.ns,4):
            count=min(4,self.ns-start);pp=[];dd=[]
            for cell in range(start,start+count):
                p,d=self.rays(cell);pp.append(p);dd.append(d)
            t1=now();raw=self.pack(np.concatenate(pp),np.concatenate(dd),sh.PATH_N);raw_seconds+=now()-t1
            t1=now();dummy=np.zeros((self.ns,self.solver.nc,1),'f4')
            value=self.solver.kernel.transport(raw,self.solver,dummy,True)
            self.first_values[self.offset[start]:self.offset[start+count]]=value;first_seconds+=now()-t1
            t1=now();z=compact_block(raw,self.solver);fold_seconds+=now()-t1
            if start==0:
                if not self.cfg.get('density',False):
                    source=np.random.default_rng(1949).normal(0,.0004,dummy.shape).astype('f4');source[:,0,0]=1.
                    a=self.solver.kernel.transport(raw,self.solver,source);b=self.solver.kernel.transport(z,self.solver,source)
                    maxcheck=float(np.max(sh.relative(a,b)));assert maxcheck<1e-6,maxcheck
                else:
                    source=dummy.copy();source[:,0,0]=np.sqrt(4*np.pi);source[:,6,0]=np.sqrt(4*np.pi)
                    p=raw['point'];row=np.repeat(np.arange(len(raw['nu'])),np.diff(raw['off']))
                    expected=np.bincount(row,weights=p[:,4]*np.exp(-p[:,6:9].astype(float)@self.solver.ext[:,0])*(np.exp(-np.maximum(0,p[:,5])/8)+np.exp(-np.maximum(0,p[:,5])/1.2)),minlength=len(raw['nu']))
                    actual=self.solver.kernel.transport(z,self.solver,source)[:,0]
                    maxcheck=float(np.max(sh.relative(actual,expected)));assert maxcheck<2e-6,maxcheck
            del raw,dummy
            name=f'path_{start:05d}.npz';nbytes=sum(v.nbytes for v in z.values())
            if size+nbytes<=budget:self.ram_blocks[name]=z;size+=nbytes
            else:np.savez_compressed(self.folder/name,**z)
            points+=len(z['base']);self.blocks.append(dict(file=name,start=int(self.offset[start]),nr=len(z['nu'])))
            if now()-last>30:print(f'PATHS {self.folder.name} {start}/{self.ns} {now()-t:.1f}s RAM={size/1e9:.2f}GB',flush=True);last=now()
            if sh.peak_bytes()>23*1024**3:raise MemoryError('23 GiB soft limit reached before the 24 GiB constraint')
        self.meta.update(path_seconds=now()-t,path_bytes=size+sum((self.folder/b['file']).stat().st_size for b in self.blocks if (self.folder/b['file']).exists()),ram_bytes=size,points=points,storage='compact_550nm_RAM16GiB_then_compressed_disk',raw_geometry_seconds=raw_seconds,first_seconds=first_seconds,fold_seconds=fold_seconds,compact_raw_check_max_relative=maxcheck)
        save(self.folder/'paths.json',dict(blocks=self.blocks,**self.meta))

class Solver(sh.Solver):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.angular_nc=self.nc
        if self.geo.cfg.get('density',False):self.nc+=6
    def scatter(self,radiance):
        if not self.geo.cfg.get('density',False):return super().scatter(radiance)
        g=self.geo;out=np.empty((g.ns,self.nc,1),'f4');self.last_local=np.empty_like(out)
        rf=((self.levels[:6]==0)+.1*(self.levels[:6]==2))*self.ext[0,0]
        af=self.g[0]**self.levels*self.asca[0]
        for cell in range(g.ns):
            lo,hi=g.offset[cell:cell+2];loc=g.dirs[lo:hi];weighted=radiance[lo:hi,0]*g.w[lo:hi]
            basis=self.kernel.basis(loc,self.L);mom=basis.T@weighted
            self.last_local[cell,:,0]=np.r_[mom[:6]*rf,mom*af]
            ch=g.chi[cell%len(g.chi)]
            glob=np.column_stack([loc[:,2]*np.sin(ch)-loc[:,0]*np.cos(ch),-loc[:,1],loc[:,2]*np.cos(ch)+loc[:,0]*np.sin(ch)])
            mom=self.kernel.basis(glob,self.L).T@weighted;out[cell,:,0]=np.r_[mom[:6]*rf,mom*af]
        return out
    def prepare_ground(self):
        t=now()
        super().prepare_ground()
        # Independent direct-ground angular integral. Holding it fixed isolates
        # volume incoming quadrature; pool doubling is measured separately.
        pool=readonly_geometry(1,2 if self.geo.cfg.get('ground_pool',1)==2 else 1)
        raw=sh.NORMAL['albedo']/np.pi*np.maximum(0,np.cos(pool.endchi))[:,None]*np.exp(-pool.surfacecol@self.ext)*pool.surfacelit[:,None]
        raw*=np.exp(-pool.column@self.ext)*pool.ground[:,None]*pool.w[:,None]
        # Ground-pool source corners must match the active spatial grid.
        # For refined space, rebuild the fixed 3/32 rule at each new cell.
        if not np.array_equal(pool.h,self.geo.h) or not np.array_equal(pool.chi,self.geo.chi):
            self.goff,self.gdir,self.gval=self.ground_pool_at_new_nodes()
        else:
            take=np.any(raw>0,axis=1);counts=[take[pool.offset[i]:pool.offset[i+1]].sum() for i in range(pool.ns)]
            self.goff=np.r_[0,np.cumsum(counts)].astype(np.int64);self.gdir=np.ascontiguousarray(pool.dirs[take],np.float32);self.gval=np.ascontiguousarray(raw[take],np.float32)
        del pool,raw;gc.collect();self.prepare_seconds=now()-t
    def ground_pool_at_new_nodes(self):
        ds=[];vs=[];off=[0];g=self.geo
        for cell in range(g.ns):
            ih,ic=divmod(cell,len(g.chi));loc,w=sh.incoming(R+g.h[ih],g.chi[ic],self.geo.cfg.get('ground_pool',1))
            p,up,ex=sh.old.probe.frame(np.array(R+g.h[ih]),g.chi[ic]);ey=np.cross(up,ex)
            d=loc[:,0,None]*ex+loc[:,1,None]*ey+loc[:,2,None]*up;p=np.broadcast_to(p,d.shape)
            end=sh.old.probe.boundaries(p,d)[:,-1];q=p+end[:,None]*d;r=np.linalg.norm(q,axis=-1);col=sh.old.total_columns(p,d,24);sc,lit=g.sun.lookup(q,np.maximum(R,r))
            ch=np.arccos(np.clip(q[:,2]/r,-1,1));raw=sh.NORMAL['albedo']/np.pi*np.maximum(0,np.cos(ch))[:,None]*np.exp(-(sc+col)@self.ext)*lit[:,None]*(r<R+1e-5)[:,None]*w[:,None]
            take=np.any(raw>0,axis=1);ds.append(loc[take]);vs.append(raw[take]);off.append(off[-1]+take.sum())
        return np.array(off,np.int64),np.ascontiguousarray(np.concatenate(ds),'f4'),np.ascontiguousarray(np.concatenate(vs),'f4')
    def solve(self,orders=10):
        self.geo.solver=self
        super().solve(orders)
    def transport(self,source,first=False):
        if first and hasattr(self.geo,'first_values'):return self.geo.first_values.copy()
        return super().transport(source,first)

def write_doc(result,state):
    result['state']=state;save(OUT/'result.json',result)
    lines=['# SOS 입사·공간 구적 A (D0 탐침)',f'상태: {state}.',
        '통과 조건: 550 nm normal/0.30, 태양고도 +2/−3/−6/−9/−12×7방향 multiple의 각 ×2 변화≤2%.',
        '앞 단계 미통과 시 B(통합 편광)·C(실측/민감도)는 실행하지 않는 명세의 단계 게이트를 따른다.',
        '물리: 원래 HG g=.8, SH L32, 원래 소광/1차 위상, delta-M/TMS 없음; 10차 고정.',
        '입사 구간: 지면 접선/μ=0/수평 띠/그림자 원통 접선/태양 피크 + 실제 그림자 원통과 0/1.2/8/25/40/80 km 껍질 교선.',
        '교선 중 방위면의 전방에 있고 지구에 가리지 않는 점만 채택; 조명층 하단과 상단 방향을 함께 분할.',
        'μ는 sinh(a) 변환 구간별 Gauss, φ는 태양면 대칭 사다리꼴. 방위와 μ는 독립적으로도 시험.',
        '1차 지면의 원래 HG 적분은 독립 고정 입사 풀(기본3/32); 풀×2 시험을 별도 기록. 지면 확산 조도는 시험 중인 구적.',
        '공간은 기본20r×75χ, χ85…110°=.5°; 요청한 r×2는 기존 중점 삽입대로39점(40 대신 실물 우선).',
        '시선 구간 Gauss12, 조회24. 해석적 그림자·지면·접점·오존 경계 분할. 4D 출사방향 보간 대신 SH 계수 r/χ 선형.',
        '|시험|μ/φ/a|r/χ|기하 s|투영 s|반복 s|RSS GB|','|---|---|---|---:|---:|---:|---:|']
    for name,row in result.get('cases',{}).items():
        c=row['config'];g=row['stats']['geometry'];s=row['stats'];lines.append(f'|{name}|{c["nmu"]}/{c["nphi"]}/{c["a"]}|{g["nr"]}/{g["nchi"]}|{g["incoming_seconds"]+g["sun_seconds"]+g.get("path_seconds",0):.1f}|{s["projection_seconds"]:.1f}|{s["seconds"]:.1f}|{row["peak_bytes"]/1e9:.2f}|')
    if result.get('comparisons'):
        lines+=['|기본→촘촘|+2°|−3°|−6°|−9°|−12°|최대|','|---|---:|---:|---:|---:|---:|---:|']
        for name,row in result['comparisons'].items():lines.append('|'+name+'|'+'|'.join(f'{100*row["by_height"][str(h)]:.3f}%' for h in sh.HS)+f'|{100*row["maximum"]:.3f}%|')
    lines+=result.get('notes',[])
    lines+=['수치 전문·방향별 차이는 results/sos_full/result.json; 입사 국소 오차 배열/PNG도 같은 폴더.',
        '스레드: NumPy 환경6, C++6(순차 호출); 사례별 별도 프로세스 RSS 측정, 메모리 상한24GiB.',
        '기하·광학열은 파장/날씨에 재사용; 감쇠/계수/원천 반복은 파장/날씨마다 재계산.',
        '미확인: A 게이트 뒤 편광 통합/PT 재대조/15파장 파노라마/Koomen/민감도는 아직 미실행.',
        '기존 파일 변경/commit/설치/새 PT 렌더 없음.']
    DOC.write_text('\n'.join(lines)+'\n')

def run_case(name,cfg):
    folder=OUT/name;folder.mkdir(parents=True,exist_ok=True);done=folder/'case.json'
    if done.exists():print('CASE CACHE '+name,flush=True);return
    start=now();kernel=Kernel();geo=Geometry(cfg,name);solver=Solver(geo,kernel,32,[550.],.8,name)
    solver.solve(cfg.get('orders',10));t=now();boxes={str(h):{k:v[:,0].tolist() for k,v in solver.boxes(h).items()} for h in sh.HS}
    data=dict(config=cfg,stats=solver.stats,boxes=boxes,query_seconds=now()-t,elapsed_seconds=now()-start,peak_bytes=sh.peak_bytes())
    assert data['peak_bytes']<=24*1024**3
    save(done,data);geo.release_paths();print(f'CASE COMPLETE {name} elapsed={data["elapsed_seconds"]:.1f}s peak={data["peak_bytes"]/1e9:.3f}GB',flush=True)

def compare(result,left,right):
    a=result['cases'][left];b=result['cases'][right];arrays={str(h):sh.relative(a['boxes'][str(h)]['multiple'],np.array(b['boxes'][str(h)]['multiple'])).tolist() for h in sh.HS}
    maxima={h:max(v) for h,v in arrays.items()};result.setdefault('comparisons',{})[left+'→'+right]=dict(by_direction=arrays,by_height=maxima,maximum=max(maxima.values()))
    # A posteriori cell source discrepancy, distinguished from the frozen-
    # radiance coarse/fine quadrature estimator generated by --estimator.
    za=np.load(OUT/left/f'state_global_o{a["config"].get("orders",10)}.npz')['local_source'][:,:,0]
    zb=np.load(OUT/right/f'state_global_o{b["config"].get("orders",10)}.npz')['local_source'][:,:,0]
    if za.shape==zb.shape:
        h,ch=grids(b['config'].get('rscale',1),b['config'].get('cscale',1));delta=np.linalg.norm(za-zb,axis=1)/np.maximum(np.linalg.norm(zb,axis=1),1e-30)
        map_array=100*delta.reshape(len(h),len(ch));np.savez(OUT/f'source_difference_{left}_{right}.npz',percent=map_array,height_km=h,chi_deg=np.rad2deg(ch))
        heatmap(map_array,h,ch,OUT/f'source_difference_{left}_{right}.png','source coefficients discrepancy % (feedback included)')
    return max(maxima.values())

def heatmap(value,h,chi,path,label):
    v=np.clip(value/20,0,1);color=np.stack([v,np.maximum(0,2*v-1),1-v],-1)
    image=Image.fromarray(np.uint8(color*255)).resize((900,400),Image.Resampling.NEAREST)
    canvas=Image.new('RGB',(1020,465),'white');canvas.paste(image,(70,40));draw=ImageDraw.Draw(canvas)
    draw.text((70,5),label,fill='black');draw.text((70,445),'chi deg: 0 -> 180 (nonuniform cells); red >=20%',fill='black')
    draw.text((5,40),f'{h[0]:g} km',fill='black');draw.text((5,420),f'{h[-1]:g} km',fill='black');canvas.save(path)

def run_estimator(name,reference,boundary_reference=None):
    """Project one common frozen total incident field on different rules."""
    t=now();cfg=json.loads((OUT/name/'case.json').read_text())['config'];kernel=Kernel();geo=Geometry(cfg,name)
    solver=Solver(geo,kernel,32,[550.],.8,'estimator_'+name);geo.solver=solver
    reference_source=np.load(OUT/reference/'state_global_o10.npz')['multiple']
    assert reference_source.shape==(geo.ns,solver.nc,1)
    geo.ensure_paths();radiance=solver.transport(reference_source)+geo.first_values
    if boundary_reference:
        b=np.load(OUT/f'estimator_{boundary_reference}.npz')['boundary']
    else:b=solver.ground_source(radiance)
    radiance+=solver.boundary(b);coeff=solver.scatter(radiance)
    np.savez(OUT/f'estimator_{name}.npz',global_coeff=coeff,local_coeff=solver.last_local,boundary=b,
             height_km=geo.h,chi_deg=np.rad2deg(geo.chi),seconds=now()-t,peak_bytes=sh.peak_bytes())
    geo.release_paths();print(f'ESTIMATOR {name} frozen={reference} seconds={now()-t:.1f}',flush=True)

def estimator_map(coarse,fine):
    with np.load(OUT/f'estimator_{coarse}.npz') as a,np.load(OUT/f'estimator_{fine}.npz') as b:
        ca=a['local_coeff'][:,:,0];cb=b['local_coeff'][:,:,0];h=b['height_km'];ch=b['chi_deg']
        numerator=np.linalg.norm(ca-cb,axis=1);denominator=np.linalg.norm(cb,axis=1)
        rel=100*numerator/np.maximum(denominator,1e-30);active=denominator>denominator.max()*1e-9
        shape=(len(h),len(ch));path=OUT/f'quadrature_error_{coarse}_{fine}'
        np.savez(path.with_suffix('.npz'),percent=rel.reshape(shape),absolute=numerator.reshape(shape),amplitude=denominator.reshape(shape),active=active.reshape(shape),height_km=h,chi_deg=ch)
        heatmap(np.where(active,rel,0).reshape(shape),h,np.deg2rad(ch),path.with_suffix('.png'),'frozen incident radiance: coarse/fine source projection %')
        loc=np.argmax(np.where(active,rel,-1));ih,ic=divmod(loc,len(ch))
        return dict(maximum_active_percent=float(rel[loc]),height_km=float(h[ih]),chi_deg=float(ch[ic]),inactive_threshold_relative=1e-9,
                    fine_seconds=float(b['seconds']),coarse_seconds=float(a['seconds']))

def query_timing(name):
    cfg=json.loads((OUT/name/'case.json').read_text())['config'];kernel=Kernel();geo=Geometry(cfg,name)
    solver=Solver(geo,kernel,32,[550.],.8,'lookup_'+name)
    solver.ms=np.load(OUT/name/f'state_global_o{cfg.get("orders",10)}.npz')['multiple']
    el=90-(np.arange(100)+.5)*.9;az=np.arange(360)+.5-180;ee,aa=np.meshgrid(el,az,indexing='ij');times={}
    for hs in sh.PHS:
        t=now();single=solver.query(hs,ee,aa,first=True);multiple=solver.query(hs,ee,aa)
        times[str(hs)]=now()-t
        assert np.isfinite(single).all() and np.isfinite(multiple).all()
        print(f'LOOKUP550 {name} hs={hs} seconds={times[str(hs)]:.3f}',flush=True)
    save(OUT/'query550.json',dict(case=name,pixels=36000,seconds=times,peak_bytes=sh.peak_bytes(),purpose='A scalar 550 nm full-sky lookup benchmark; B panorama not generated'))

def manifest():
    names=('sos_probe','sos_sky','sos_sky2','sos_pol','sos_pol_iprt','sos_render')
    return {str(p.relative_to(ROOT)):[p.stat().st_size,p.stat().st_mtime_ns] for name in names for base in (ROOT/f'scripts/{name}.py',ROOT/f'docs/{name.upper()}.md',ROOT/f'results/{name}') for p in ([base] if base.is_file() else base.rglob('*')) if p.is_file()}

def run_all():
    start=now();before=manifest();save(OUT/'frozen_initial.json',before)
    result=dict(stage='A',cases={},comparisons={},notes=[],deadline_seconds=10800,start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),dirs=[f'{n}{e}/{a}' for n,e,a in sh.DIRS])
    existing=OUT/'result.json'
    if existing.exists():result.update(json.loads(existing.read_text()))
    def case(name,**extra):
        cfg=dict(nmu=3,nphi=32,a=4.,rings=True,rscale=1,cscale=1,ground_pool=1);cfg.update(extra)
        write_doc(result,name+' 준비/계산')
        arg=json.dumps(cfg,separators=(',',':'))
        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--case',name,'--config',arg],check=True)
        result['cases'][name]=json.loads((OUT/name/'case.json').read_text());write_doc(result,name+' 완료')
    case('Q3P32')
    case('Q6P32',nmu=6);compare(result,'Q3P32','Q6P32');write_doc(result,'μ3→6 완료')
    case('Q6P64',nmu=6,nphi=64);compare(result,'Q6P32','Q6P64');write_doc(result,'φ32→64 완료')
    case('Q12P64',nmu=12,nphi=64);compare(result,'Q6P64','Q12P64');write_doc(result,'μ6→12 완료')
    case('Q12P128',nmu=12,nphi=128);compare(result,'Q12P64','Q12P128');write_doc(result,'φ64→128 완료')
    case('Q12P32',nmu=12,nphi=32);compare(result,'Q12P32','Q12P64');write_doc(result,'최소 φ32→64 완료')
    case('Q24P32',nmu=24,nphi=32)
    compare(result,'Q12P32','Q24P32');write_doc(result,'μ12→24 완료')
    case('Q48P32',nmu=48,nphi=32);compare(result,'Q24P32','Q48P32');write_doc(result,'μ24→48 완료')
    case('Q24P64',nmu=24,nphi=64);compare(result,'Q24P32','Q24P64');compare(result,'Q12P64','Q24P64');write_doc(result,'μ12→24 / 최종 φ32→64 완료')
    case('Q3P128',nmu=3,nphi=128)
    case('Q3P256',nmu=3,nphi=256);compare(result,'Q3P128','Q3P256');write_doc(result,'φ128→256 완료')
    case('Q6P32a6',nmu=6,nphi=32,a=6.)
    case('Q6P32a8',nmu=6,nphi=32,a=8.);compare(result,'Q6P32','Q6P32a6');compare(result,'Q6P32a6','Q6P32a8');write_doc(result,'sinh a4/6/8 완료')
    case('Q12P32C2',nmu=12,nphi=32,cscale=2);compare(result,'Q12P32','Q12P32C2');write_doc(result,'χ .5→.25° 완료')
    case('Q12P32C4',nmu=12,nphi=32,cscale=4);compare(result,'Q12P32C2','Q12P32C4');write_doc(result,'χ .25→.125° 완료')
    case('Q12P32R2',nmu=12,nphi=32,rscale=2);compare(result,'Q12P32','Q12P32R2');write_doc(result,'r20→39 완료')
    case('Q12P32Ground2',nmu=12,nphi=32,ground_pool=2);compare(result,'Q12P32','Q12P32Ground2');write_doc(result,'원래 HG 1차 지면 구적×2 완료')
    case('Q12P32O14',nmu=12,nphi=32,orders=14,geometry='Q12P32');compare(result,'Q12P32','Q12P32O14');write_doc(result,'차수10→14 확인 완료')
    if result['comparisons']['Q12P32→Q12P32R2']['maximum']>.02:
        result['notes'].append('추측: β(r)를 포함한 원천의 반지름 선형 보간이 공간 오차를 만든다. β_R/β_A를 분리해 시선점에서 해석적으로 곱하는 변형을 시험.')
        case('D12P32',nmu=12,nphi=32,density=True);compare(result,'Q12P32','D12P32')
        case('D12P32R2',nmu=12,nphi=32,rscale=2,density=True);compare(result,'D12P32','D12P32R2');write_doc(result,'해석적 밀도 분리 r×2 완료')
        case('D12P32C2',nmu=12,nphi=32,cscale=2,density=True);compare(result,'D12P32','D12P32C2');write_doc(result,'해석적 밀도 분리 χ×2 완료')
    result['A_pass']=False;result['B']='A 게이트 대기';result['C']='B 게이트 대기'
    result['elapsed_seconds']=now()-start;result['frozen_preserved']=before==manifest()
    write_doc(result,'A 구적/공간 시험 완료, 국소 오차·최종 게이트 평가 대기')
    print('STAGE A CONVERGENCE CASES COMPLETE',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--all',action='store_true');p.add_argument('--quad',action='store_true');p.add_argument('--case');p.add_argument('--config');p.add_argument('--estimator');p.add_argument('--reference');p.add_argument('--boundary-reference');p.add_argument('--timing');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);stop=threading.Event();t=now()
    def heartbeat():
        while not stop.wait(60):print(f'HEARTBEAT elapsed={now()-t:.0f}s peak_GB={sh.peak_bytes()/1e9:.3f}',flush=True)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        if a.case:run_case(a.case,json.loads(a.config))
        elif a.estimator:run_estimator(a.estimator,a.reference,a.boundary_reference)
        elif a.timing:query_timing(a.timing)
        elif a.all or a.quad:run_all()
        else:p.error('choose --all or --quad')
    finally:stop.set();thread.join()
if __name__=='__main__':main()
