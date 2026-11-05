#!/usr/bin/env python3
"""D0 full-HG spherical-harmonic successive orders, CPU only.
No delta-M, no TMS, no outgoing-direction source table, no PT rendering.
All generated code, geometry, states and products live under results/sos_sky2.
"""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='2'
import argparse, ctypes, hashlib, json, resource, shutil, subprocess, sys, threading, time
from pathlib import Path
import numpy as np
import sos_sky as old

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/sos_sky2'; DOC=ROOT/'docs/SOS_SKY2.md'
os.environ['DRJIT_CACHE_DIR']=str(OUT/'jit-cache')
R,Z,OBS=old.R,old.Z,old.OBS; WL=old.WL; DIRS=old.DIRS
HS=(2,-3,-6,-9,-12); PHS=(2,0,-3,-6,-9,-12)
NORMAL=old.NORMAL.copy(); PATH_N=12; THREADS=2

CPP=r'''
#include <cmath>
#include <algorithm>
#include <vector>
#include <thread>
#include <cstdint>
static constexpr double PI=3.14159265358979323846;
static int ix(int l,int m){return l*(l+1)/2+m;}
struct Norm {
 int L,K; std::vector<double>a,b,d;
 Norm(int ll):L(ll),K((ll+1)*(ll+2)/2),a(K),b(K),d(ll+1){
  for(int m=1;m<=L;m++)d[m]=-sqrt((2.*m+1)/(2*m));
  for(int l=1;l<=L;l++)for(int m=0;m<l;m++){
   a[ix(l,m)]=sqrt((4.*l*l-1)/(l*l-m*m));
   if(l>m+1)b[ix(l,m)]=sqrt((2.*l+1)*((l-1.)*(l-1)-m*m)/((2.*l-3)*(l*l-m*m)));
  }
 }
 void eval(double mu,double cp,double*y)const {
  mu=std::clamp(mu,-1.,1.);cp=std::clamp(cp,-1.,1.);
  double st=sqrt(std::max(0.,1-mu*mu)),diag=1/sqrt(4*PI),cm=1,prev=cp;
  for(int m=0;m<=L;m++){
   if(m==1){diag*=d[m]*st;cm=cp;prev=1;}
   else if(m>1){diag*=d[m]*st;double next=2*cp*cm-prev;prev=cm;cm=next;}
   double mult=m?std::sqrt(2.)*cm:1.;
   double z0=diag;y[ix(m,m)]=mult*z0;
   if(m<L){double z1=sqrt(2.*m+3)*mu*z0;y[ix(m+1,m)]=mult*z1;
    for(int l=m+2;l<=L;l++){int k=ix(l,m);double z2=a[k]*mu*z1-b[k]*z0;y[k]=mult*z2;z0=z1;z1=z2;}
   }
  }
 }
};
extern "C" void basis(int n,int L,const double*dirs,double*out){
 Norm norm(L);for(int i=0;i<n;i++){
  double mu=dirs[3*i+2],st=sqrt(std::max(0.,1-mu*mu));
  norm.eval(mu,st>1e-12?dirs[3*i]/st:1.,out+(int64_t)i*norm.K);
 }
}
static double hg(double u,double g){double d=std::max(1e-20,1+g*g-2*g*u);return (1-g*g)/(4*PI*d*sqrt(d));}
struct Args {
 int nr,nw,L,nchi,ns; const int64_t*off;const int32_t*base;const uint8_t*lit;
 const float*p;const double*nu;const float*src;const double*ext;const double*g;
 const int64_t*goff;const float*gdir;const float*gval;const double*scatter;
 double*out;int first;
};
template<int W> static void runpart(const Args&A,int lo,int hi,const Norm&norm){
 const int K=norm.K,M=A.L+1; std::vector<double>Y(K);std::vector<int>stamp(A.ns,-1);
 std::vector<float>cache((int64_t)A.ns*M*W);
 for(int ray=lo;ray<hi;ray++){
  double sum[W]={},pr=3*(1+A.nu[ray]*A.nu[ray])/(16*PI),pa[W];
  for(int w=0;w<W;w++)pa[w]=hg(A.nu[ray],A.g[w]);
  if(A.first!=1)norm.eval(A.nu[ray],1.,Y.data());
  for(int64_t n=A.off[ray];n<A.off[ray+1];n++){
   const float*q=A.p+13*n;double ah=q[0],ac=q[1],mu=q[2],cp=q[3],ds=q[4],h=q[5];
   int cells[4]={A.base[n],A.base[n]+1,A.base[n]+A.nchi,A.base[n]+A.nchi+1};
   double sw[4]={(1-ah)*(1-ac),(1-ah)*ac,ah*(1-ac),ah*ac};
   double val[W]={};
   if(A.first==1){
    double rr=exp(-std::max(0.,h)/8),aa=exp(-std::max(0.,h)/1.2);
    for(int w=0;w<W;w++){
     double ts=A.lit[n]?exp(-q[9]*A.ext[w]-q[10]*A.ext[W+w]-q[11]*A.ext[2*W+w]):0;
     val[w]=ts*(rr*A.ext[w]*pr+aa*A.scatter[w]*pa[w]);
    }
    double st=sqrt(std::max(0.,1-mu*mu)),sp=sqrt(std::max(0.,1-cp*cp));
    for(int c=0;c<4;c++)if(sw[c]>1e-12){
     for(int64_t j=A.goff[cells[c]];j<A.goff[cells[c]+1];j++){
      double x=A.gdir[3*j],y=A.gdir[3*j+1],z=A.gdir[3*j+2];
      double u0=mu*z+st*cp*x,u1=st*sp*y,up=std::clamp(u0+u1,-1.,1.),um=std::clamp(u0-u1,-1.,1.);
      double rayp=3*(2+up*up+um*um)/(32*PI);
      for(int w=0;w<W;w++)val[w]+=sw[c]*A.gval[j*W+w]*(rr*A.ext[w]*rayp+aa*A.scatter[w]*.5*(hg(up,A.g[w])+hg(um,A.g[w])));
     }
    }
   } else if(A.first==2) {
    norm.eval(A.nu[ray],q[12],Y.data());
    const float*c0=A.src+(int64_t)cells[0]*K*W,*c1=A.src+(int64_t)cells[1]*K*W;
    const float*c2=A.src+(int64_t)cells[2]*K*W,*c3=A.src+(int64_t)cells[3]*K*W;
    for(int k=0;k<K;k++)for(int w=0;w<W;w++){
     int t=k*W+w;val[w]+=Y[k]*(sw[0]*c0[t]+sw[1]*c1[t]+sw[2]*c2[t]+sw[3]*c3[t]);
    }
   } else {
    for(int c=0;c<4;c++)if(stamp[cells[c]]!=ray){
     float*v=cache.data()+(int64_t)cells[c]*M*W;std::fill(v,v+M*W,0.f);
     const float*s=A.src+(int64_t)cells[c]*K*W;
     for(int l=0;l<=A.L;l++)for(int m=0;m<=l;m++){
      int k=ix(l,m);float pref=(float)Y[k];for(int w=0;w<W;w++)v[m*W+w]+=pref*s[k*W+w];
     }
     stamp[cells[c]]=ray;
    }
    const float*c0=cache.data()+(int64_t)cells[0]*M*W,*c1=cache.data()+(int64_t)cells[1]*M*W;
    const float*c2=cache.data()+(int64_t)cells[2]*M*W,*c3=cache.data()+(int64_t)cells[3]*M*W;
    double cm=1,prev=q[12];
    for(int m=0;m<M;m++){
     if(m==1){cm=q[12];prev=1;}else if(m>1){double next=2*q[12]*cm-prev;prev=cm;cm=next;}
     for(int w=0;w<W;w++){int t=m*W+w;val[w]+=cm*(sw[0]*c0[t]+sw[1]*c1[t]+sw[2]*c2[t]+sw[3]*c3[t]);}
    }
   }
   for(int w=0;w<W;w++)sum[w]+=ds*exp(-q[6]*A.ext[w]-q[7]*A.ext[W+w]-q[8]*A.ext[2*W+w])*val[w];
  }
  for(int w=0;w<W;w++)A.out[(int64_t)ray*W+w]=sum[w];
 }
}
template<int W> static void dispatch(const Args&A){
 Norm norm(A.L);int mid=A.nr/2;
 std::thread t([&](){runpart<W>(A,0,mid,norm);});runpart<W>(A,mid,A.nr,norm);t.join();
}
extern "C" void transport(int nr,int nw,int L,int nchi,int ns,const int64_t*off,const int32_t*base,
 const uint8_t*lit,const float*p,const double*nu,const float*src,const double*ext,const double*g,
 const int64_t*goff,const float*gdir,const float*gval,const double*scatter,double*out,int first){
 Args A{nr,nw,L,nchi,ns,off,base,lit,p,nu,src,ext,g,goff,gdir,gval,scatter,out,first};
 switch(nw){case 1:dispatch<1>(A);break;case 4:dispatch<4>(A);break;case 15:dispatch<15>(A);break;}
}
'''

def dump(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=1,allow_nan=False)+'\n');os.replace(tmp,path)

def peak_bytes():return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)

class Kernel:
    def __init__(self):
        cpp=OUT/'sh_cpu.cpp';lib=OUT/'sh_cpu.dylib';digest=hashlib.sha256(CPP.encode()).hexdigest()
        if not lib.exists() or not cpp.exists() or hashlib.sha256(cpp.read_bytes()).hexdigest()!=digest:
            cpp.write_text(CPP)
            subprocess.run(['clang++','-std=c++17','-O3','-ffp-contract=fast','-dynamiclib',str(cpp),'-o',str(lib)],check=True)
        self.lib=ctypes.CDLL(str(lib));self.lib.basis.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
        self.lib.transport.argtypes=[ctypes.c_int]*5+[ctypes.c_void_p]*13+[ctypes.c_int]
    def basis(self,dirs,L):
        dirs=np.ascontiguousarray(dirs,np.float64);out=np.empty((len(dirs),(L+1)*(L+2)//2))
        self.lib.basis(len(dirs),L,dirs.ctypes.data,out.ctypes.data);return out
    def transport(self,z,solver,source,first=False):
        nr=len(z['nu']);out=np.empty((nr,solver.nw));source=np.ascontiguousarray(source,np.float32)
        names=('off','base','lit','point','nu');arrays=[np.ascontiguousarray(z[k]) for k in names]
        if arrays[3].shape[1]==12:
            point=arrays[3];ic=arrays[1]%len(solver.geo.chi)
            ch=solver.geo.chi[ic]+point[:,1]*(solver.geo.chi[ic+1]-solver.geo.chi[ic])
            nu=np.repeat(arrays[4],np.diff(arrays[0]));den=np.sin(ch)*np.sqrt(np.maximum(0,1-nu*nu))
            cp=np.divide(point[:,2]-np.cos(ch)*nu,den,out=np.ones(len(point)),where=abs(den)>1e-12)
            arrays[3]=np.ascontiguousarray(np.column_stack([point,np.clip(cp,-1,1)]),np.float32)
        args=arrays+[source,solver.ext,solver.g,solver.goff,solver.gdir,solver.gval,solver.asca,out]
        self.lib.transport(nr,solver.nw,solver.L,len(solver.geo.chi),solver.geo.ns,*[a.ctypes.data for a in args],int(first))
        return out

def incoming(r,chi,qscale):
    mg=-np.sqrt(max(0,1-(R/r)**2));a=4.;sa=np.sinh(a)
    nphi=32*qscale;u,wu=old.probe.gl(3*qscale);allmu=[];allphi=[];allw=[]
    for j,phi in enumerate(np.linspace(0,np.pi,nphi+1)):
        edges=[-1.,1.,mg,0.,mg-.05234,mg+.05234,mg-.0061,mg+.0061]
        rp=r*np.sin(chi)
        if rp>R+1e-8 and abs(np.sin(chi))>1e-8:
            kk=R/np.sqrt(rp*rp-R*R)
            for sign in (-1,1):
                t=np.cos(phi)*np.cos(chi)/np.sin(chi)+sign*np.sin(phi)/(kk*np.sin(chi))
                mu=t/np.sqrt(1+t*t);st=np.sqrt(1-mu*mu)
                dx=mu*np.sin(chi)-st*np.cos(phi)*np.cos(chi)
                dy=-st*np.sin(phi);dz=mu*np.cos(chi)+st*np.cos(phi)*np.sin(chi)
                tmin=-rp*dx/max(dx*dx+dy*dy,1e-25)
                if abs(dx)+abs(dy)<1e-10 or (tmin>0 and r*np.cos(chi)+tmin*dz<0):edges.append(mu)
        if phi<np.pi/6:
            mu=np.cos(chi)/np.sqrt(np.cos(chi)**2+np.cos(phi)**2*np.sin(chi)**2+1e-30)
            edges.extend([mu-.012,mu,mu+.012])
        edges=np.unique(np.clip(edges,-1,1));delta=edges-mg
        tt=np.sign(delta)*np.arcsinh(np.abs(delta)/np.where(delta<0,1+mg,1-mg)*sa)/a
        mus=[];weights=[]
        for left,right in zip(tt[:-1],tt[1:]):
            if right-left<1e-10:continue
            t=left+(right-left)*u;den=np.where(t<0,1+mg,1-mg)
            mus.extend(mg+np.sign(t)*den*np.sinh(a*np.abs(t))/sa)
            weights.extend((right-left)*wu*den*a*np.cosh(a*np.abs(t))/sa)
        wp=2*np.pi/nphi*(.5 if j in (0,nphi) else 1.)
        allmu.extend(mus);allphi.extend([phi]*len(mus));allw.extend(np.array(weights)*wp)
    mu=np.array(allmu);phi=np.array(allphi);st=np.sqrt(np.maximum(0,1-mu*mu))
    return np.column_stack([st*np.cos(phi),st*np.sin(phi),mu]),np.array(allw)

def spatial_lookup(q,r,d,h,chi):
    height=np.clip(r-R,0,Z);ch=np.arccos(np.clip(q[...,2]/r,-1,1))
    ih=np.clip(np.searchsorted(h,height,side='right')-1,0,len(h)-2)
    ic=np.clip(np.searchsorted(chi,ch,side='right')-1,0,len(chi)-2)
    ah=np.clip((height-h[ih])/(h[ih+1]-h[ih]),0,1);ac=np.clip((ch-chi[ic])/(chi[ic+1]-chi[ic]),0,1)
    mu=np.clip(np.sum(q*d,-1)/r,-1,1);cs=q[...,2]/r
    den=np.sqrt(np.maximum(0,(1-cs*cs)*(1-mu*mu)))
    cp=np.divide(d[...,2]-cs*mu,den,out=np.ones_like(mu),where=den>1e-12)
    return (ih*len(chi)+ic).astype(np.int32),ah,ac,mu,np.clip(cp,-1,1)

class Geometry:
    def __init__(self,spatial=1,angular=1):
        self.s,self.q=spatial,angular;self.h,self.chi=old.grids(spatial);self.ns=len(self.h)*len(self.chi)
        self.folder=OUT/f'geo_s{spatial}_q{angular}';self.folder.mkdir(parents=True,exist_ok=True)
        self.sun=old.SunGeometry(spatial,self.folder);self.blocks=[];self.ram_blocks={}
        f=self.folder/'incoming.npz';t=time.perf_counter()
        if not f.exists():self.build_incoming()
        z=np.load(f);self.offset=z['offset'];self.dirs=z['dirs'];self.w=z['weight'];self.ground=z['ground'];self.endchi=z['endchi']
        self.column=z['column'];self.surfacecol=z['surfacecol'];self.surfacelit=z['surfacelit'];z.close()
        self.nr=len(self.dirs);self.meta=json.loads((self.folder/'meta.json').read_text())
        pathmeta=self.folder/'paths.json'
        if pathmeta.exists():self.meta.update(json.loads(pathmeta.read_text()))
        self.meta['load_seconds']=time.perf_counter()-t
    def rays(self,cell):
        ih,ic=divmod(cell,len(self.chi));p,up,ex=old.probe.frame(np.array(R+self.h[ih]),self.chi[ic]);ey=np.cross(up,ex)
        lo,hi=self.offset[cell:cell+2];loc=self.dirs[lo:hi]
        d=loc[:,0,None]*ex+loc[:,1,None]*ey+loc[:,2,None]*up
        return np.broadcast_to(p,d.shape),d
    def build_incoming(self):
        t=time.perf_counter();dirs=[];weights=[];offset=[0];gr=[];ec=[];cols=[];sc=[];sl=[]
        for cell in range(self.ns):
            ih,ic=divmod(cell,len(self.chi));loc,w=incoming(R+self.h[ih],self.chi[ic],self.q)
            p,up,ex=old.probe.frame(np.array(R+self.h[ih]),self.chi[ic]);ey=np.cross(up,ex)
            d=loc[:,0,None]*ex+loc[:,1,None]*ey+loc[:,2,None]*up;p=np.broadcast_to(p,d.shape)
            end=old.probe.boundaries(p,d)[:,-1];qe=p+end[:,None]*d;re=np.sqrt(np.sum(qe*qe,-1))
            ground=re<R+1e-5;ch=np.arccos(np.clip(qe[:,2]/re,-1,1))
            col=old.total_columns(p,d,24);scol,lit=self.sun.lookup(qe,np.maximum(re,R))
            dirs.append(loc);weights.append(w);offset.append(offset[-1]+len(w));gr.append(ground);ec.append(ch)
            cols.append(col);sc.append(scol);sl.append(lit)
            if cell%300==0:print(f'INCOMING s{self.s}q{self.q} {cell}/{self.ns} {time.perf_counter()-t:.1f}s',flush=True)
        np.savez(self.folder/'incoming.npz',offset=np.array(offset,np.int64),dirs=np.concatenate(dirs),weight=np.concatenate(weights),
                 ground=np.concatenate(gr),endchi=np.concatenate(ec),column=np.concatenate(cols).astype('f4'),
                 surfacecol=np.concatenate(sc).astype('f4'),surfacelit=np.concatenate(sl))
        dump(self.folder/'meta.json',dict(spatial=self.s,angular=self.q,nr=len(self.h),nchi=len(self.chi),cells=self.ns,
             incoming_rays=offset[-1],incoming_seconds=time.perf_counter()-t,sun_seconds=self.sun.build_seconds,path_n=PATH_N))
    def pack(self,p,d,n):
        buffers={k:[] for k in ('point','base','lit','row')};nr=len(d)
        for active,q,r,ds,prefix in old.geometry_segments(p,d,n):
            base,ah,ac,mu,cp=spatial_lookup(q,r,d[active,None,:],self.h,self.chi)
            suncol,lit=self.sun.lookup(q,r);count=ds.shape[1]
            point=np.stack([ah,ac,mu,cp,ds,r-R,*[prefix[...,k] for k in range(3)],*[suncol[...,k] for k in range(3)]],-1)
            buffers['point'].append(point.reshape(-1,12).astype('f4'));buffers['base'].append(base.ravel());buffers['lit'].append(lit.ravel())
            buffers['row'].append(np.repeat(active,count))
        if not buffers['row']:
            return dict(off=np.zeros(nr+1,np.int64),base=np.empty(0,np.int32),lit=np.empty(0,np.uint8),point=np.empty((0,12),'f4'),nu=d[:,2].copy())
        row=np.concatenate(buffers.pop('row'));order=np.argsort(row,kind='stable')
        arrays={k:np.concatenate(v)[order] for k,v in buffers.items()}
        arrays['lit']=arrays['lit'].astype(np.uint8);arrays['off']=np.r_[0,np.cumsum(np.bincount(row,minlength=nr))].astype(np.int64)
        arrays['nu']=d[:,2].copy();return arrays
    def ensure_paths(self):
        f=self.folder/'paths.json'
        in_memory=self.s*self.q>1 and shutil.disk_usage(OUT).free<12*1024**3
        if f.exists() and not in_memory:
            data=json.loads(f.read_text())
            if all((self.folder/b['file']).exists() for b in data['blocks']):self.blocks=data['blocks'];self.meta.update(data);return
        t=time.perf_counter();blocks=[];points=0;last=t;ram_bytes=0
        budget=min(8*1024**3,max(0,12*1024**3-peak_bytes()))
        for start in range(0,self.ns,4):
            count=min(4,self.ns-start);pp=[];dd=[]
            for cell in range(start,start+count):
                p,d=self.rays(cell);pp.append(p);dd.append(d)
            z=self.pack(np.concatenate(pp),np.concatenate(dd),PATH_N);name=f'path_{start:05d}.npz'
            size=sum(v.nbytes for v in z.values())
            if in_memory:
                if ram_bytes+size<=budget:self.ram_blocks[name]=z;ram_bytes+=size
                (self.folder/name).unlink(missing_ok=True)
            else:np.savez(self.folder/name,**z)
            points+=len(z['base']);blocks.append(dict(file=name,start=int(self.offset[start]),nr=len(z['nu'])))
            if time.perf_counter()-last>30:print(f'PATHS s{self.s}q{self.q} {start}/{self.ns} {time.perf_counter()-t:.1f}s',flush=True);last=time.perf_counter()
        data=dict(blocks=blocks,path_seconds=time.perf_counter()-t,points=points,
                  path_bytes=sum((self.folder/b['file']).stat().st_size for b in blocks if (self.folder/b['file']).exists()),
                  storage='bounded_RAM_and_recomputed_blocks' if in_memory else 'disk',ram_bytes=ram_bytes,ram_budget_bytes=budget)
        dump(f,data);self.blocks=blocks;self.meta.update(data)
    def release_paths(self):
        # Only disposable files newly created by this probe are removed.
        for b in self.blocks:(self.folder/b['file']).unlink(missing_ok=True)
        self.ram_blocks.clear()
    def block_geometry(self,block):
        start=int(np.searchsorted(self.offset,block['start']));pp=[];dd=[]
        for cell in range(start,min(start+4,self.ns)):
            p,d=self.rays(cell);pp.append(p);dd.append(d)
        return self.pack(np.concatenate(pp),np.concatenate(dd),PATH_N)

class Solver:
    def __init__(self,geo,kernel,L,wavelengths,g,name):
        self.geo,self.kernel,self.L=geo,kernel,L;self.wl=np.asarray(wavelengths,float);self.g=np.broadcast_to(g,self.wl.shape).astype(float).copy()
        self.nw=len(self.wl);assert self.nw in (1,4,15);self.nc=(L+1)*(L+2)//2
        self.ext=np.ascontiguousarray(old.coefficients(self.wl,NORMAL));self.asca=np.ascontiguousarray(self.ext[1]*NORMAL['ssa'])
        self.solar=np.interp(self.wl,old.SPEC['wavelength_nm'],old.SPEC['solar'])
        self.folder=OUT/name;self.folder.mkdir(parents=True,exist_ok=True);self.stats={};self.snaps={}
        self.levels=np.array([l for l in range(L+1) for m in range(l+1)])
        self.prepare_ground()
    def prepare_ground(self):
        t=time.perf_counter();g=self.geo
        self.boundary_tr=np.exp(-g.column@self.ext)
        b0=NORMAL['albedo']/np.pi*np.maximum(0,np.cos(g.endchi))[:,None]*np.exp(-g.surfacecol@self.ext)*g.surfacelit[:,None]
        raw=b0*self.boundary_tr*g.ground[:,None]*g.w[:,None]
        take=np.any(raw>0,axis=1);counts=np.array([take[g.offset[i]:g.offset[i+1]].sum() for i in range(g.ns)])
        self.goff=np.r_[0,np.cumsum(counts)].astype(np.int64);self.gdir=np.ascontiguousarray(g.dirs[take],np.float32);self.gval=np.ascontiguousarray(raw[take],np.float32)
        self.factor=np.empty((len(g.h),self.nc,self.nw))
        chiR=(self.levels==0).astype(float)+.1*(self.levels==2)
        for ih,h in enumerate(g.h):
            self.factor[ih]=np.exp(-h/8)*chiR[:,None]*self.ext[0]+np.exp(-h/1.2)*self.g[None,:]**self.levels[:,None]*self.asca
        self.prepare_seconds=time.perf_counter()-t
    def boundary(self,b):
        g=self.geo;j=np.clip(np.searchsorted(g.chi,g.endchi,side='right')-1,0,len(g.chi)-2)
        a=np.clip((g.endchi-g.chi[j])/(g.chi[j+1]-g.chi[j]),0,1)
        return ((1-a[:,None])*b[j]+a[:,None]*b[j+1])*self.boundary_tr*g.ground[:,None]
    def ground_source(self,radiance):
        g=self.geo;out=np.zeros((len(g.chi),self.nw))
        for ic in range(len(g.chi)):
            lo,hi=g.offset[ic:ic+2];w=g.w[lo:hi]*np.maximum(0,g.dirs[lo:hi,2])
            out[ic]=NORMAL['albedo']/np.pi*w@radiance[lo:hi]
        return out
    def scatter(self,radiance):
        g=self.geo;out=np.empty((g.ns,self.nc,self.nw),np.float32);self.last_local=np.empty_like(out)
        for cell in range(g.ns):
            lo,hi=g.offset[cell:cell+2];basis=self.kernel.basis(g.dirs[lo:hi],self.L)
            weighted=radiance[lo:hi]*g.w[lo:hi,None];moment=basis.T@weighted
            self.last_local[cell]=moment*self.factor[cell//len(g.chi)]
            ch=g.chi[cell%len(g.chi)];loc=g.dirs[lo:hi]
            glob=np.column_stack([loc[:,2]*np.sin(ch)-loc[:,0]*np.cos(ch),-loc[:,1],loc[:,2]*np.cos(ch)+loc[:,0]*np.sin(ch)])
            gbasis=self.kernel.basis(glob,self.L);out[cell]=(gbasis.T@weighted)*self.factor[cell//len(g.chi)]
        return out
    def transport(self,source,first=False):
        g=self.geo;out=np.empty((g.nr,self.nw))
        for block in g.blocks:
            z=g.ram_blocks.get(block['file'])
            if z is None:
                path=g.folder/block['file'];z=np.load(path) if path.exists() else g.block_geometry(block)
            values=self.kernel.transport(z,self,source,first)
            out[block['start']:block['start']+block['nr']]=values
            if hasattr(z,'close'):z.close()
        return out
    def solve(self,orders=14):
        state=self.folder/f'state_global_o{orders}.npz';stats=self.folder/f'stats_global_o{orders}.json'
        if state.exists() and stats.exists():
            z=np.load(state);self.ms=z['multiple'];self.ms_local=z['local_source'];self.snaps={int(k[1:]):z[k] for k in z.files if k.startswith('o')};z.close()
            self.stats=json.loads(stats.read_text());self.stats['cache_hit']=True
            for checkpoint in self.folder.glob('checkpoint_global_o*.npz'):checkpoint.unlink()
            return
        self.geo.ensure_paths();t=time.perf_counter();dummy=np.zeros((self.geo.ns,self.nc,self.nw),np.float32)
        radiance=self.transport(dummy,True);b=self.ground_source(radiance);radiance+=self.boundary(b)
        self.ms=np.zeros_like(dummy);self.ms_local=np.zeros_like(dummy);history=[];proj=0.;line=0.
        for n in range(2,orders+1):
            t1=time.perf_counter();j=self.scatter(radiance);proj+=time.perf_counter()-t1
            t1=time.perf_counter();radiance=self.transport(j);line+=time.perf_counter()-t1
            b=self.ground_source(radiance);radiance+=self.boundary(b);self.ms+=j;self.ms_local+=self.last_local
            frac=float(np.max(np.abs(j))/max(float(np.max(np.abs(self.ms))),1e-30))
            history.append(dict(order=n,source_increment=frac,seconds=time.perf_counter()-t))
            if n in (6,10,14,20):self.snaps[n]=self.ms.copy()
            print(f'SH {self.folder.name} n={n} frac={frac:.5g} seconds={time.perf_counter()-t:.2f} project={proj:.2f} line={line:.2f}',flush=True)
            if n in (6,10,14):
                np.savez(self.folder/f'checkpoint_global_o{n}.npz',multiple=self.ms,local_source=self.ms_local,radiance=radiance,
                         **{f'o{k}':v for k,v in self.snaps.items()})
                for checkpoint in self.folder.glob('checkpoint_global_o*.npz'):
                    if checkpoint.name!=f'checkpoint_global_o{n}.npz':checkpoint.unlink()
        np.savez(state,multiple=self.ms,local_source=self.ms_local,**{f'o{k}':v for k,v in self.snaps.items()})
        self.stats=dict(orders=orders,seconds=time.perf_counter()-t,projection_seconds=proj,line_seconds=line,
                        prepare_seconds=self.prepare_seconds,history=history,cache_hit=False,peak_bytes=peak_bytes(),
                        L=self.L,wavelengths=self.wl.tolist(),g=self.g.tolist(),geometry=dict(self.geo.meta))
        dump(stats,self.stats)
        # Complete states already contain all requested order snapshots.
        for checkpoint in self.folder.glob('checkpoint_global_o*.npz'):
            checkpoint.unlink()
    def query(self,h,elevations,azimuths,source=None,first=False,reference=False):
        el,az=np.broadcast_arrays(elevations,azimuths);shape=el.shape;e,a=np.deg2rad(el.ravel()),np.deg2rad(az.ravel())
        p0,up,ex=old.probe.frame(np.array(R+OBS),np.deg2rad(90-h));ey=np.cross(up,ex)
        d=np.sin(e)[:,None]*up+np.cos(e)[:,None]*(np.cos(a)[:,None]*ex+np.sin(a)[:,None]*ey)
        out=np.empty((len(d),self.nw));field=self.ms if source is None else source
        for start in range(0,len(d),64):
            dd=d[start:start+64];p=np.broadcast_to(p0,dd.shape);z=self.geo.pack(p,dd,24)
            out[start:start+len(dd)]=self.kernel.transport(z,self,field,1 if first else 2 if reference else 0)
        return out.reshape(*shape,self.nw)*self.solar
    def boxes(self,h,source=None):
        u,w=np.polynomial.legendre.leggauss(4);ww=(w[:,None]*w[None,:]/4).ravel()
        el=np.array([e+de for _,e,a in DIRS for de in u for da in u]);az=np.array([a+da for _,e,a in DIRS for de in u for da in u])
        single=np.einsum('dnw,n->dw',self.query(h,el,az,first=True).reshape(7,16,self.nw),ww)
        multiple=np.einsum('dnw,n->dw',self.query(h,el,az,source=source).reshape(7,16,self.nw),ww)
        return dict(single=single,multiple=multiple,total=single+multiple)

def relative(a,b):return np.abs(np.asarray(a)-b)/np.maximum(np.abs(b),1e-30)

def frozen():
    out={}
    for name in ('sos_probe','sos_sky'):
        for base in (ROOT/f'scripts/{name}.py',ROOT/f'docs/{name.upper()}.md'):
            out[str(base.relative_to(ROOT))]=hashlib.sha256(base.read_bytes()).hexdigest()
        out[name]={str(p.relative_to(ROOT)):[p.stat().st_size,p.stat().st_mtime_ns] for p in (ROOT/f'results/{name}').rglob('*') if p.is_file()}
    return out

def progress(result,state):
    measured=[peak_bytes(),result.get('peak_bytes',0),result.get('historical_peak_bytes',0)]
    measured += [v.get('peak_bytes',0) for v in result.get('completed_cases',{}).values()]
    measured.append(result.get('spectral_solver',{}).get('peak_bytes',0))
    result['state']=state;result['peak_bytes']=max(measured);dump(OUT/'result.json',result)
    lines=['# SOS 구면조화 탐침 (D0)',f'상태: {state}.',
           'delta-M/TMS 없음. 국소 z=반지름/x=태양 접평면, cos 실수 SH. HG χ_l=g^l; Rayleigh χ_0=1, χ_2=.1.',
           '1차 태양광은 원래 소광·원래 HG/Rayleigh·해석적 그림자로 직접 계산. 1차 지면도 원래 HG 구적(행 정규화 없음).',
           '국소SH 계수 저장+회전된 태양축 작업계수; 시선 고정γ의 l합 캐시(float32), 최종 시선 누산double. 출사 방향 표 없음.',
           '입사: μ_h/0/μ_h±.0061/±.05234(약.35°/3°)/방위별 그림자 원통 접선/태양 근처 분할; sinh(a=4) Gauss3/6, φ32/64.',
           '공간 기본20r×75χ/두 배39r×149χ; χ85…110° 간격.5°/.25°. 시선 구간 Gauss12(조회24), 그림자·접점·오존 경계 분할.',
           '각/입사/공간 비교는 모두10차, 기준L32. 별도로L32 고정6/10/14/20차. g.7/.8/.85 동시550 nm(4번째 .8 중복 검산).',
           '2단계 ablation 재실행 없음. PT6장 읽기 전용 재사용; 새 PT 없음. source/engine/spectrum 정본은 2단계와 동일.']
    if 'rayleigh_check' in result:lines.append('Rayleigh SH/Q 최대 절대차='+str(result['rayleigh_check']['maximum_absolute_error'])+'.')
    if 'representation_check' in result:
        q=result['representation_check'];lines.append(f'캐시/직접double 최대상대차={max(q["fast_vs_double_reference_max_relative_by_height"].values()):.3g}; 좌표회전 최대절대차={q["rotation_max_absolute_error"]:.3g}; g.8 중복차={max(q["duplicate_g08_max_relative_by_height"].values()):.3g}.')
    if 'convergence' in result:
        has64='L48_to64' in next(iter(result['convergence'].values()))
        maximum=max(v for row in result['convergence'].values() for arr in row.values() for v in arr)
        lines.append('판정: '+('수렴 목표 각각2% 충족.' if maximum<=.02 else '미수렴; 각각2% 목표 미달. 기준을 완화하지 않음.'))
        lines+=['|고도|L16→24|L24→32|L32→48|'+('L48→64|' if has64 else '')+'입사×2|공간×2|',
                '|---|---:|---:|---:|'+('---:|' if has64 else '')+'---:|---:|']
        for h,row in result['convergence'].items():lines.append('|'+h+'|'+'|'.join(f'{100*max(v):.3f}%' for v in row.values())+'|')
        lines+=['목표 각각2%; 별도 손잡이이며 차수10으로 고정. 수렴표의 값은7방향 multiple 상대변화 최대.',
                '지배 손잡이(실측): '+result['dominant_control'],
                '|g (−6° 한 점, 7방향 최대)|L16→24|L24→32|L32→48|'+('L48→64|' if has64 else '')+'입사×2|공간×2|',
                '|---|---:|---:|---:|'+('---:|' if has64 else '')+'---:|---:|']
        for g,row in result['g_sensitivity'].items():lines.append('|'+g+'|'+'|'.join(f'{100*v:.3f}%' for v in row.values())+'|')
        lines+=['|고도|차수6→10|10→14|14→20|','|---|---:|---:|---:|']
        for h,row in result['order_convergence'].items():lines.append('|'+h+'|'+'|'.join(f'{100*max(v):.3f}%' for v in row.values())+'|')
    if 'pt' in result:
        lines+=['PT 표: 550 nm/기준L32·20차. 방향sun2/sun10/sun30/anti10/anti30/side10/zenith89.',
                '|고도|single/PT|multiple/PT|total/PT|MS 5%안 / 2SE안|','|---|---|---|---|---|']
        for h,row in result['pt'].items():
            vals=[' / '.join('N/A' if v is None else f'{v:.3f}' for v in row[k]['ratio']) for k in ('single','multiple','total')]
            v=row['multiple'];n=sum(v['eligible']);a=sum(x is True for x in v['within5pct']);b=sum(x is True for x in v['within2se'])
            lines.append('|'+h+'|'+'|'.join(vals)+f'|{a}/{n} / {b}/{n}|')
        lines.append('판정 제외: '+json.dumps(result['excluded'],ensure_ascii=False)+'. 5%와2SE는 개별 판정; SE>15% 및 두 seed0 제외.')
        lines.append('PT MS 상대SE(7방향): '+ '; '.join(h+'='+','.join(f'{100*v:.2f}%' for v in row['multiple']['se_relative']) for h,row in result['pt'].items()))
    if 'panorama' in result:
        p=result['panorama'];lines+=['15칸 spec/rgb float32, 360×100·90…0°·태양 중앙. +2/0/−3/−6/−9/−12, L32·14차.',
             'PNG는 동일 RGB행렬·중앙Y→.18·x/(1+x)·sRGB. 위SH/가운데lut0_fit/아래앱LUT, 행마다 자동노출.',
             '2단계 차이 PNG는 선형RGB 휘도 상대차 절대값%, 색 범위0…50% (수치는비절단 npy에도 저장).',
             '장당 조회초: '+', '.join(f'{h}={v["query_seconds"]:.2f}' for h,v in p['runs'].items()),
             '원천은 태양축 대칭(r,χ)1회 풀이. 관측자 태양고도를 바꿀 때 조회만 재사용.',
             f'추측: 기본15파장×25고도 표={p["estimate_table_seconds"]:.2f}s (기하+계수+14차1회+평균조회×25).']
    if 'artifact_check' in result:lines.append('산출물 검사: '+str(result['artifact_check']['valid'])+'; spec/rgb/diff float32 크기·유한값·PNG 크기·RGB행렬 일치 검사.')
    if 'timings' in result:
        t=result['timings'];lines+=['기하: '+json.dumps(t['geometry_seconds'])+'초. 큰 시선 캐시는 각 두 배 시험 뒤 새폴더 안에서만 제거.',
             f'550 nm4lane 공동 기준20차={t["numeric_base_seconds"]:.2f}s; 파장별g3종과중복1종을함께계산.',
             f'최대RSS={result["peak_bytes"]/1e9:.3f} GB; NumPy환경2/CPU커널2스레드(순차호출).']
        if 'numeric_concurrent_rss_upper_bound_bytes' in result:
            lines.append(f'L64 별도2스레드 병행(수치계산 합계4); 두 프로세스 RSS피크합 상한={result["numeric_concurrent_rss_upper_bound_bytes"]/1e9:.3f} GB (동시 최대 실측 아님).')
        if 'spectral_iterations_seconds' in t:lines.append(f'15파장 계수평균={t["spectral_prepare_seconds"]/15:.3f}s/파장, 반복평균={t["spectral_iterations_seconds"]/15:.3f}s/파장 (공동벡터화 총시간÷15).')
        if any(v.get('geometry',{}).get('storage')=='bounded_RAM_and_recomputed_blocks' for v in result.get('completed_cases',{}).values()):
            lines.append('디스크 부족: 두 배 기하는 RAM 최대8GiB+초과블록 동일구적 재계산. 격자 축소 없음; 완료 후 기하RAM 해제.')
    lines+=['기하·종별 광학열·구적은 파장/AOD/SSA/g/알베도 변경에 재사용; 감쇠/1차/원천 반복은 다시 계산.',
            '한계: 유한SH/입사/공간/시선 구적, 스칼라 복사장; 15파장 파노라마14차의20차 대조는 미측정.',
            '미확인: 550 nm 이외 PT/자기수렴, 더 높은L('+('64 초과' if 'L64' in result.get('completed_cases',{}) else '64')+'), 편광·태양원반·RAM경로 전체실행. 범위 밖 숫자도 그대로 보존.',
            '추측: 입사 복사장의 얇은 조명 띠·급한 각 변화를 기본 구적이 충분히 적분하지 못함. 성분별 원인 분리는 미실시.',
            '1·2단계 보존='+str(result.get('frozen_preserved','검사 대기'))+'. 기존파일 수정·commit·설치·PT 신규렌더 없음.']
    assert len(lines)<=60,len(lines);DOC.write_text('\n'.join(lines)+'\n')

def rayleigh_check(kernel,geo):
    dirs,w=incoming(R+1.,np.deg2rad(94),1);Y=kernel.basis(dirs,2)
    x,y,z=dirs.T;I=1+.3*x+.2*z+.4*x*z+.1*y*y;mom=Y.T@(I*w)
    rng=np.random.default_rng(731);mu=rng.uniform(-1,1,91);phi=rng.uniform(0,np.pi,91);st=np.sqrt(1-mu*mu)
    out=np.column_stack([st*np.cos(phi),st*np.sin(phi),mu]);YY=kernel.basis(out,2)
    levels=np.array([l for l in range(3) for m in range(l+1)]);a=YY@(mom*((levels==0)+.1*(levels==2)))
    qxx=np.sum(I*w*x*x);qyy=np.sum(I*w*y*y);qzz=np.sum(I*w*z*z);qxz=np.sum(I*w*x*z)
    b=3/(16*np.pi)*(qxx+qyy+qzz+qxx*out[:,0]**2+qyy*out[:,1]**2+qzz*out[:,2]**2+2*qxz*out[:,0]*out[:,2])
    error=float(np.max(np.abs(a-b)));assert error<1e-12,error
    return dict(maximum_absolute_error=error,solid_angle_error=float(abs(w.sum()-4*np.pi)))

def representation_check(solver):
    """Compare contracted float cache with direct SH/double evaluation."""
    el=np.array([.2,2,10,30,89,10,10]);az=np.array([0,0,0,0,0,90,180])
    fast={};duplicate={}
    for h in HS:
        a=solver.query(h,el,az);b=solver.query(h,el,az,reference=True)
        fast[str(h)]=float(np.max(relative(a,b)))
        duplicate[str(h)]=float(np.max(relative(a[:,1],a[:,3])))
    ch=np.deg2rad(94);d,w=incoming(R+1,ch,1)
    def rotate(v):return np.column_stack([v[:,2]*np.sin(ch)-v[:,0]*np.cos(ch),-v[:,1],v[:,2]*np.cos(ch)+v[:,0]*np.sin(ch)])
    I=1+.3*d[:,0]+.2*d[:,2]+.4*d[:,0]*d[:,2]+.1*d[:,1]**2
    factor=.8**solver.levels
    c=(solver.kernel.basis(d,solver.L).T@(I*w))*factor
    cg=(solver.kernel.basis(rotate(d),solver.L).T@(I*w))*factor
    rng=np.random.default_rng(621);mu=rng.uniform(-1,1,100);phi=rng.uniform(0,np.pi,100);st=np.sqrt(1-mu*mu)
    out=np.column_stack([st*np.cos(phi),st*np.sin(phi),mu])
    a=solver.kernel.basis(out,solver.L)@c;b=solver.kernel.basis(rotate(out),solver.L)@cg
    error=float(np.max(abs(a-b)))
    assert max(fast.values())<1e-4 and max(duplicate.values())<1e-12 and error<1e-12
    return dict(fast_vs_double_reference_max_relative_by_height=fast,duplicate_g08_max_relative_by_height=duplicate,
                rotation_max_absolute_error=error,rotation_max_relative_error=float(np.max(relative(a,b))))

def tables(result,solvers):
    boxes={name:{str(h):s.boxes(h,source=s.snaps[10])['multiple'] for h in HS} for name,s in solvers.items()}
    conv={};pairs=[('L16_to24','L16','L24'),('L24_to32','L24','L32'),('L32_to48','L32','L48'),('incoming_x2','L32','Q2'),('spatial_x2','L32','S2')]
    if 'L64' in solvers:pairs.insert(3,('L48_to64','L48','L64'))
    for h in HS:
        conv[str(h)]={n:relative(boxes[a][str(h)][:,1],boxes[b][str(h)][:,1]).tolist() for n,a,b in pairs}
    result['convergence']=conv
    maxima={n:max(v for row in conv.values() for v in row[n]) for n,a,b in pairs}
    result['dominant_control']=max(maxima,key=maxima.get)+f' ({100*max(maxima.values()):.3f}%)'
    result['g_sensitivity']={str(g):{n:float(np.max(relative(boxes[a]['-6'][:,lane],boxes[b]['-6'][:,lane]))) for n,a,b in pairs}
                            for lane,g in ((0,.7),(2,.85))}
    base=solvers['L32'];result['order_convergence']={}
    for h in HS:
        arr={n:base.boxes(h,base.snaps[n])['multiple'][:,1] for n in (6,10,14,20)}
        result['order_convergence'][str(h)]={f'{a}_to{b}':relative(arr[a],arr[b]).tolist() for a,b in ((6,10),(10,14),(14,20))}
    result['pt']={};result['pt_sources']={};excluded=[]
    for h in (2,-3,-6):
        data=base.boxes(h);pt,metas=old.pt_table(h);row={}
        result['pt_sources'][str(h)]=[dict(spectral_file=str((ROOT/f'results/sos_sky/pt_e{h}_s{seed}/spectral.npz').relative_to(ROOT)),
                                          metadata=meta,reused=True) for seed,meta in zip((1,2),metas)]
        for k in ('single','multiple','total'):
            mean=np.array(pt[k]['mean']);se=np.array(pt[k]['se'])/np.maximum(mean,1e-30);ratio=np.divide(data[k][:,1],mean,out=np.full(7,np.nan),where=mean>0)
            eligible=(mean>0)&(se<=.15);r=[float(v) if np.isfinite(v) else None for v in ratio]
            row[k]=dict(ratio=r,se_relative=se.tolist(),eligible=eligible.tolist(),pt_mean=mean.tolist(),sos=data[k][:,1].tolist(),
                        within2pct=[bool(abs(v-1)<=.02) if ok else None for v,ok in zip(ratio,eligible)],
                        within5pct=[bool(abs(v-1)<=.05) if ok else None for v,ok in zip(ratio,eligible)],
                        within2se=[bool(abs(v-1)<=2*err) if ok else None for v,err,ok in zip(ratio,se,eligible)])
            for i in np.flatnonzero(~eligible):excluded.append(dict(hs=h,dir=f'{DIRS[i][0]}{DIRS[i][1]}',channel=k,reason='both_seeds_zero' if mean[i]==0 else 'SE_over_15pct'))
        result['pt'][str(h)]=row
    result['excluded']=excluded

def panoramas(solver,result):
    from PIL import Image,ImageDraw
    helper=old.predecessor_helpers();helper['OUT']=OUT;w,res=old.rgb_weights();lut=helper['app_lut']()
    el=90-(np.arange(100)+.5)*.9;az=np.arange(360)+.5-180;ee,aa=np.meshgrid(el,az,indexing='ij')
    runs={};comps=[]
    for h in PHS:
        t=time.perf_counter();single=solver.query(h,ee,aa,first=True);multiple=solver.query(h,ee,aa)
        query=time.perf_counter()-t;spec=single+multiple;rgb=spec@w
        np.save(OUT/f'spec_e{h}.npy',spec.astype('f4'));np.save(OUT/f'rgb_e{h}.npy',rgb.astype('f4'));helper['to_png'](rgb,OUT/f'pano_e{h}.png')
        oldrgb=np.load(ROOT/f'results/sos_sky/rgb_e{h}.npy');y=rgb@np.array([.2126,.7152,.0722]);y0=oldrgb@np.array([.2126,.7152,.0722])
        diff=100*(y-y0)/np.maximum(abs(y0),1e-30);np.save(OUT/f'difference_e{h}.npy',diff.astype('f4'))
        v=np.clip(abs(diff)/50,0,1);color=np.stack([v,np.maximum(0,2*v-1),np.zeros_like(v)],-1)
        im=Image.fromarray(np.uint8(color*255));ImageDraw.Draw(im).text((3,3),'relative Y abs %: black 0 / yellow >=50',fill='white');im.save(OUT/f'difference_e{h}.png')
        sh=helper['pano_from_grid'](rgb[:,180:][::-1],el[::-1],np.arange(180)+.5)
        pt=helper['pano_from_fold'](helper['fold_table'](h));app=helper['app_pano'](h,lut)
        comp=np.concatenate([helper['label'](helper['to_png'](sh),f'Full HG SH L32 hs {h:+d}'),helper['label'](helper['to_png'](pt),f'PT lut0_fit hs {h:+d}'),helper['label'](helper['to_png'](app),f'app LUT hs {h:+d}')])
        Image.fromarray(comp).save(OUT/f'compare_e{h}.png');comps.append(comp)
        runs[str(h)]=dict(query_seconds=query,total_seconds=time.perf_counter()-t,nonfinite=int((~np.isfinite(spec)).sum()),negative=int((spec<0).sum()),
                         difference_abs_mean_percent=float(np.mean(abs(diff))),difference_abs_max_percent=float(np.max(abs(diff))))
        result['panorama']=dict(runs=runs,rgb_fit_residual=res,estimate_table_seconds=0.)
        progress(result,f'파노라마 {h:+d} 완료');print(f'PANO SH {h} {query:.2f}s',flush=True)
    Image.fromarray(np.concatenate(comps,axis=1)).save(OUT/'compare_all.png')
    np.save(OUT/'rgb_weights.npy',w)
    mean=np.mean([v['query_seconds'] for v in runs.values()]);geo=solver.geo.meta
    result['panorama']['estimate_table_seconds']=geo['incoming_seconds']+geo['sun_seconds']+geo.get('path_seconds',0)+solver.prepare_seconds+solver.stats['seconds']+25*mean

def artifact_check():
    from PIL import Image
    weights=np.load(OUT/'rgb_weights.npy');rows={}
    for h in PHS:
        spec=np.load(OUT/f'spec_e{h}.npy');rgb=np.load(OUT/f'rgb_e{h}.npy');diff=np.load(OUT/f'difference_e{h}.npy')
        error=float(np.max(abs(spec.astype(float)@weights-rgb))/max(float(np.max(abs(rgb))),1e-30))
        sizes={name:list(Image.open(OUT/f'{name}_e{h}.png').size) for name in ('pano','difference','compare')}
        ok=(spec.shape==(100,360,15) and rgb.shape==(100,360,3) and diff.shape==(100,360)
            and spec.dtype==np.float32 and rgb.dtype==np.float32 and diff.dtype==np.float32
            and np.isfinite(spec).all() and np.isfinite(rgb).all() and np.isfinite(diff).all()
            and sizes==dict(pano=[360,100],difference=[360,100],compare=[720,540]) and error<2e-6)
        rows[str(h)]=dict(valid=bool(ok),spec_shape=list(spec.shape),rgb_shape=list(rgb.shape),
                          rgb_matrix_relative_error=error,png_sizes=sizes)
    return dict(valid=all(v['valid'] for v in rows.values()),runs=rows)

def run_all():
    start=time.perf_counter();before=frozen();initial=OUT/'frozen_initial.json'
    if not initial.exists():
        original=json.loads((OUT/'frozen_before.json').read_text()) if (OUT/'frozen_before.json').exists() else before
        dump(initial,original)
    dump(OUT/'frozen_before.json',before)
    previous=json.loads((OUT/'result.json').read_text()) if (OUT/'result.json').exists() else {}
    result=dict(historical_peak_bytes=previous.get('peak_bytes',0));progress(result,'구현·기하 준비')
    kernel=Kernel();geo=Geometry();result['rayleigh_check']=rayleigh_check(kernel,geo);progress(result,'Rayleigh Q/SH 일치 확인')
    g=np.array([.7,.8,.85,.8]);w=np.full(4,550.);solvers={};timings={}
    for L in (32,16,24,48,64):
        progress(result,f'L{L} 계수·수치 반복 진행')
        s=Solver(geo,kernel,L,w,g,f'numeric_L{L}_s1_q1');s.solve(20 if L==32 else 10);solvers[f'L{L}']=s
        result['completed_cases']={n:v.stats for n,v in solvers.items()};progress(result,f'L{L} 수치 반복 완료')
    result['representation_check']=representation_check(solvers['L32'])
    result.update(dirs=[f'{n}{e}/{a}' for n,e,a in DIRS],wavelengths_nm=WL.tolist(),atmosphere=NORMAL,observer_km=OBS)
    for name,sp,ang in (('Q2',1,2),('S2',2,1)):
        progress(result,name+' 기하·계수·수치 반복 진행')
        gg=Geometry(sp,ang);s=Solver(gg,kernel,32,w,g,f'numeric_L32_s{sp}_q{ang}');s.solve(10);solvers[name]=s
        result['completed_cases']={n:v.stats for n,v in solvers.items()};progress(result,name+' 수치 반복 완료');gg.release_paths()
    tables(result,solvers);result['timings']=dict(geometry_seconds={n:v.geo.meta['incoming_seconds']+v.geo.meta['sun_seconds']+v.geo.meta.get('path_seconds',0) for n,v in solvers.items()},numeric_base_seconds=solvers['L32'].stats['seconds'])
    progress(result,'각·입사·공간·차수·g·PT 수치 표 완료')
    progress(result,'15파장 계수·원천 반복 진행')
    spec=Solver(geo,kernel,32,WL,.8,'spectral_L32_s1_q1');spec.solve(14)
    result['timings'].update(spectral_prepare_seconds=spec.prepare_seconds,spectral_iterations_seconds=spec.stats['seconds'])
    result['spectral_solver']=spec.stats;progress(result,'15파장 원천 반복 완료');panoramas(spec,result)
    result['artifact_check']=artifact_check()
    after=frozen();result['frozen_preserved']=before==after and json.loads(initial.read_text())==after
    result['elapsed_seconds']=time.perf_counter()-start
    result['memory_limit_bytes']=16*1024**3
    progress(result,'완료')
    parallel=OUT/'parallel_L64.json'
    if parallel.exists():
        result['parallel_L64']=json.loads(parallel.read_text())
        result['numeric_concurrent_rss_upper_bound_bytes']=result['peak_bytes']+result['parallel_L64']['peak_bytes']
    result['memory_within_limit']=max(result['peak_bytes'],result.get('numeric_concurrent_rss_upper_bound_bytes',0))<=result['memory_limit_bytes']
    progress(result,'완료')
    print(f'COMPLETE seconds={result["elapsed_seconds"]:.2f} peak_GB={result["peak_bytes"]/1e9:.3f} frozen_preserved={result["frozen_preserved"]}',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--all',action='store_true');a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.all:
        stop=threading.Event();started=time.perf_counter()
        def heartbeat():
            while not stop.wait(120):
                rows=DOC.read_text().splitlines() if DOC.exists() else []
                status=rows[1] if len(rows)>1 else '계산 중'
                print(f'HEARTBEAT elapsed={time.perf_counter()-started:.0f}s peak_GB={peak_bytes()/1e9:.3f} {status}',flush=True)
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        try:run_all()
        finally:stop.set();thread.join()
    else:p.error('choose --all')

if __name__=='__main__':main()
