#!/usr/bin/env python3
"""Prepared integrated tensor/HG solver; numerical execution is gated by A2.
Rayleigh coherency tensor plus scalar, unpolarized HG approximation; no delta-M.
Existing files are read-only. CPU4, one process, RSS soft stop7.5GB/hard8GB.
"""
import argparse,gc,hashlib,json,math,shutil,sys,time,types
from pathlib import Path
import a2_driver as a
np=a.np;b=a.b;ROOT=a.ROOT;OUT=ROOT/'results/sos_full/b2_run';OUT.mkdir(exist_ok=True)
DOC=ROOT/'docs/SOS_FULL.md';b.OUT=OUT;b.sh.OUT=OUT;b.DOC=DOC
a.os.environ['DRJIT_CACHE_DIR']=str(OUT/'jit-cache')
b.sh.CPP+=(Path(__file__).with_name('b2_tensor.cpp')).read_text()
from PIL import Image
sys.modules['b2_driver']=sys.modules[__name__]
S=571;K=561;C=3/(16*np.pi)
PAIRS=((0,0),(1,1),(2,2),(0,2))

def basis_at(q):
    r=np.linalg.norm(q,axis=-1);up=q/r[...,None];cos=up[...,2]
    ex=np.array([0.,0.,1.])-cos[...,None]*up;l=np.linalg.norm(ex,axis=-1)
    ex=np.divide(ex,l[...,None],out=np.zeros_like(ex),where=l[...,None]>1e-12)
    ex=np.where((l<1e-12)[...,None],np.stack([-np.where(cos>=0,1.,-1.),np.zeros_like(cos),np.zeros_like(cos)],-1),ex)
    return np.stack([ex,np.cross(up,ex),up],-1)

def pack(p,d,h,ch,sun,n):
    rows=[];pts=[];bases=[];lits=[]
    for active,q,r,ds,prefix in b.sh.old.geometry_segments(p,d,n):
        base,ah,ac,mu,cp=b.sh.spatial_lookup(q,r,d[active,None,:],h,ch)
        sc,lit=sun.lookup(q,r);rho=np.linalg.norm(q[...,:2],axis=-1)
        cx=np.divide(q[...,0],rho,out=np.ones_like(rho),where=rho>1e-12);sx=np.divide(q[...,1],rho,out=np.zeros_like(rho),where=rho>1e-12)
        z=np.stack([ah,ac,mu,cp,ds,r-b.R,*[prefix[...,k] for k in range(3)],*[sc[...,k] for k in range(3)],cx,sx],-1)
        rows.append(np.repeat(active,ds.shape[1]));pts.append(z.reshape(-1,14).astype('f4'));bases.append(base.ravel());lits.append(lit.ravel())
    B=basis_at(p)
    if not rows:
        return dict(off=np.zeros(len(d)+1,'i8'),base=np.empty(0,'i4'),point=np.empty((0,14),'f4'),lit=np.empty(0,'u1'),
            nu=d[:,2].copy(),target=np.ascontiguousarray(B),dloc=np.einsum('nij,ni->nj',B,d))
    row=np.concatenate(rows);order=np.argsort(row,kind='stable')
    return dict(off=np.r_[0,np.cumsum(np.bincount(row,minlength=len(d)))].astype('i8'),
        base=np.concatenate(bases)[order].astype('i4'),point=np.concatenate(pts)[order],lit=np.concatenate(lits)[order].astype('u1'),
        nu=d[:,2].copy(),target=np.ascontiguousarray(B),dloc=np.einsum('nij,ni->nj',B,d))

class Kernel(b.sh.Kernel):
    def __init__(self):
        self.code_hash=hashlib.sha256((b.sh.CPP+Path(__file__).read_text()).encode()).hexdigest()[:16]
        super().__init__();self.lib.full_transport.argtypes=[b.sh.ctypes.c_int]*7+[b.sh.ctypes.c_void_p]*20+[b.sh.ctypes.c_double,b.sh.ctypes.c_void_p]
    def transport(self,z,s,field,first=False):
        out=np.empty((len(z['nu']),s.W,2,4));arrays=[s.active,z['off'],z['base'],z['lit'],z['point'],z['nu'],z['target'],z['dloc'],field,
            s.ext,s.asca,s.geo.chi,s.hr,s.ha,s.geo.poly,s.geo.left,s.goff,s.gdir,s.gval,s.gm]
        arrays=[np.ascontiguousarray(x) for x in arrays]
        compact=z['point'].shape[1]==6
        assert not compact or (s.W==1 and not first),'COMPACT_550_MS_ONLY'
        self.lib.full_transport(len(out),s.W,len(s.geo.chi),s.geo.ns,s.mode,int(first),int(compact),*[x.ctypes.data for x in arrays],s.delta,out.ctypes.data)
        return out

class Geometry(a.Geometry):
    def __init__(self,name,nr,step,disk=True,compact=False):
        super().__init__(name,nr,step,disk);self.cfg['tensor']=True;self.compact=compact;self.nr_borrow=(nr==20 and step==.5 and not compact)
    def pack(self,p,d,n):return pack(p,d,self.h,self.chi,self.sun,n)
    def cell(self,i,s):
        if i in self.ram:z=self.ram.pop(i);self.ram[i]=z;return z
        path=self.cache/f'cell_{i:05d}.npz';borrow=ROOT/'results/sos_full/b_run/geometry_base550_d0.03_1_cf68b5993871da15/blocks'/path.name
        read_path=path if path.exists() else borrow if self.nr_borrow and borrow.exists() else path;t=time.perf_counter()
        if read_path.exists():
            with np.load(read_path) as f:z={k:f[k] for k in f.files}
            e=z.pop('point_bytes');z['point']=np.ascontiguousarray(e.transpose(2,0,1)).view('f4').reshape(e.shape[-1],e.shape[0]);self.meta['io_seconds']+=time.perf_counter()-t
        else:
            z=self.build(i,s)
            if self.disk and self.meta['cache_bytes']<14_000_000_000 and shutil.disk_usage(OUT).free>768*1024**2:
                t=time.perf_counter();v=dict(z);p=v.pop('point');v['point_bytes']=p.view('u1').reshape(len(p),p.shape[1],4).transpose(1,2,0).copy();b.zipped(path,v)
                self.meta['compression_seconds']+=time.perf_counter()-t;self.meta['cache_bytes']+=path.stat().st_size
            elif self.disk:self.meta['uncached_builds']=self.meta.get('uncached_builds',0)+1
        if not np.array_equal(z['wavelengths'],s.wl) or float(z['delta'])!=s.delta:
            assert z['point'].shape[1]==14,'COMPACT_CACHE_OPTICS_LOCKED'
            t=time.perf_counter();z['first']=s.kernel.transport(z,s,s.dummy,True).astype('f4');z['ground_tr']=np.exp(-z['end_columns']@s.ext)*z['hits_ground'][:,None];z['wavelengths']=s.wl.copy();z['delta']=np.array(s.delta)
            self.meta['first_seconds']+=time.perf_counter()-t
            if self.disk and shutil.disk_usage(OUT).free>768*1024**2:
                old_bytes=path.stat().st_size if path.exists() else 0;t=time.perf_counter();v=dict(z);p=v.pop('point');v['point_bytes']=p.view('u1').reshape(len(p),p.shape[1],4).transpose(1,2,0).copy();b.zipped(path,v)
                self.meta['compression_seconds']+=time.perf_counter()-t;self.meta['cache_bytes']+=path.stat().st_size-old_bytes
        size=sum(v.nbytes for v in z.values())
        while self.ram and self.ram_bytes+size>self.ram_limit:
            _,old=self.ram.popitem(last=False);self.ram_bytes-=sum(v.nbytes for v in old.values())
        if size<=self.ram_limit:self.ram[i]=z;self.ram_bytes+=size
        return z
    def build(self,i,s):
        t=time.perf_counter();ih,ic=divmod(i,len(self.chi));loc,w=b.legacy.incoming(b.R+self.h[ih],self.chi[ic],12,64,4.,True)
        self.meta['solid_angle_max_relative']=max(self.meta['solid_angle_max_relative'],abs(w.sum()/(4*np.pi)-1))
        p,up,ex=b.sh.old.probe.frame(np.array(b.R+self.h[ih]),self.chi[ic]);ey=np.cross(up,ex)
        d=loc[:,0,None]*ex+loc[:,1,None]*ey+loc[:,2,None]*up;p=np.broadcast_to(p,d.shape)
        end=b.sh.old.probe.boundaries(p,d)[:,-1];qe=p+end[:,None]*d;re=np.linalg.norm(qe,axis=-1);col=b.sh.old.total_columns(p,d,24)
        tr=np.exp(-col@s.ext)*(re<b.R+1e-5)[:,None];ec=np.arccos(np.clip(qe[:,2]/re,-1,1));j=np.clip(np.searchsorted(self.chi,ec,side='right')-1,0,len(self.chi)-2)
        frac=np.clip((ec-self.chi[j])/(self.chi[j+1]-self.chi[j]),0,1);self.meta['geometry_seconds']+=time.perf_counter()-t
        chunks=[];first=[];off=[0]
        for start in range(0,len(loc),256):
            t=time.perf_counter();z=self.pack(p[start:start+256],d[start:start+256],12);self.meta['geometry_seconds']+=time.perf_counter()-t
            t=time.perf_counter();first.append(s.kernel.transport(z,s,s.dummy,True).astype('f4'));self.meta['first_seconds']+=time.perf_counter()-t
            if self.compact:
                t=time.perf_counter();q=z['point'];ds=q[:,4]*np.exp(-q[:,6:9].astype(float)@s.ext[:,0]);z['point']=np.column_stack([q[:,0],q[:,1],ds,q[:,5],q[:,12],q[:,13]]).astype('f4');self.meta['fold_seconds']+=time.perf_counter()-t
            off.extend((z['off'][1:]+off[-1]).tolist());chunks.append(z)
        z=dict(off=np.array(off,'i8'),**{key:np.concatenate([v[key] for v in chunks]) for key in ('point','base','lit','nu','target','dloc')},
               loc=loc,weight=w,first=np.concatenate(first),ground_tr=tr,end_index=j.astype('i4'),end_fraction=frac,end_columns=col,hits_ground=re<b.R+1e-5,wavelengths=s.wl.copy(),delta=np.array(s.delta))
        self.meta['incoming_rays']+=len(w);self.meta['points']+=len(z['point']);self.meta['largest_block_bytes']=max(self.meta['largest_block_bytes'],sum(v.nbytes for v in z.values()))
        return z

class Solver:
    def __init__(self,g,kernel,wavelengths,mode=0,delta=.03):
        self.geo,self.kernel,self.mode,self.delta=g,kernel,mode,delta;self.folder=g.folder;self.wl=np.asarray(wavelengths,float);self.W=len(self.wl);self.V=2*self.W;self.active=np.ones(self.V,'i4')
        self.ext=np.ascontiguousarray(b.sh.old.coefficients(self.wl,b.sh.NORMAL));self.asca=np.ascontiguousarray(self.ext[1]*b.sh.NORMAL['ssa'])
        self.hr=np.exp(-g.h/8);self.ha=np.exp(-g.h/1.2);self.solar=np.interp(self.wl,b.sh.old.SPEC['wavelength_nm'],b.sh.old.SPEC['solar'])
        self.levels=np.array([l for l in range(33) for m in range(l+1)]);self.rf=(self.levels[:6]==0)+.1*(self.levels[:6]==2);self.af=.8**self.levels
        self.dummy=np.empty(0,'f4');t=time.perf_counter();self.prepare_ground();self.prepare_seconds=time.perf_counter()-t
        self.timings=dict(transport_seconds=0.,projection_seconds=0.,ground_seconds=0.)
    def prepare_ground(self):
        ds=[];vs=[];off=[0];g=self.geo;self.gm=np.zeros((g.ns,4,self.W),'f4')
        for cell in range(g.ns):
            ih,ic=divmod(cell,len(g.chi));loc,w=b.sh.incoming(b.R+g.h[ih],g.chi[ic],1)
            p,up,ex=b.sh.old.probe.frame(np.array(b.R+g.h[ih]),g.chi[ic]);ey=np.cross(up,ex);d=loc[:,0,None]*ex+loc[:,1,None]*ey+loc[:,2,None]*up;p=np.broadcast_to(p,d.shape)
            end=b.sh.old.probe.boundaries(p,d)[:,-1];q=p+end[:,None]*d;r=np.linalg.norm(q,axis=-1);col=b.sh.old.total_columns(p,d,24);sc,lit=g.sun.lookup(q,np.maximum(b.R,r))
            ch=np.arccos(np.clip(q[:,2]/r,-1,1));value=b.sh.NORMAL['albedo']/np.pi*np.maximum(0,np.cos(ch))[:,None]*np.exp(-(sc+col)@self.ext)*lit[:,None]*(r<b.R+1e-5)[:,None]*w[:,None]
            take=np.any(value>0,axis=1);ds.append(loc[take]);vs.append(value[take]);off.append(off[-1]+take.sum())
            factor=np.stack([(1-loc[:,0]**2)/2,(1-loc[:,1]**2)/2,(1-loc[:,2]**2)/2,-loc[:,0]*loc[:,2]/2],-1)
            self.gm[cell]=(factor.T@value)*(2*C*self.ext[0])[None,:]
        self.goff=np.array(off,'i8');self.gdir=np.ascontiguousarray(np.concatenate(ds),'f4');self.gval=np.ascontiguousarray(np.concatenate(vs),'f4')
    def radiance(self,i,field,boundary=None):
        z=self.geo.cell(i,self);t=time.perf_counter();L=self.kernel.transport(z,self,field)+z['first'];self.timings['transport_seconds']+=time.perf_counter()-t
        if boundary is not None:
            E=(1-z['end_fraction'][:,None,None])*boundary[z['end_index']]+z['end_fraction'][:,None,None]*boundary[z['end_index']+1]
            v=z['loc'];P=np.stack([1-v[:,0]**2,1-v[:,1]**2,1-v[:,2]**2,-v[:,0]*v[:,2]],-1)/2
            L+=E[...,None]*z['ground_tr'][:,:,None,None]*P[:,None,None,:]
        return z,L
    def boundary(self,field):
        t=time.perf_counter();out=np.empty((len(self.geo.chi),self.W,2))
        for i in range(len(out)):
            z,L=self.radiance(i,field);I=L[...,:3].sum(-1);out[i]=b.sh.NORMAL['albedo']/np.pi*np.einsum('n,nwl->wl',z['weight']*np.maximum(0,z['loc'][:,2]),I)
        self.timings['ground_seconds']+=time.perf_counter()-t;return out
    def apply(self,field):
        boundary=self.boundary(field);out=np.empty_like(field);t=time.perf_counter();last=t
        for i in range(self.geo.ns):
            z,L=self.radiance(i,field,boundary);tt=time.perf_counter();I=L[...,:3].sum(-1);ch=self.geo.chi[i%len(self.geo.chi)];v=z['loc']
            glob=np.column_stack([v[:,2]*np.sin(ch)-v[:,0]*np.cos(ch),-v[:,1],v[:,2]*np.cos(ch)+v[:,0]*np.sin(ch)]);Y=self.kernel.basis(glob,32)
            moment=Y.T@(I.reshape(len(I),self.V)*z['weight'][:,None]);matrix=np.einsum('n,nwlp->pwl',z['weight'],L).reshape(4,self.V)
            out[i,:4]=matrix*np.repeat(2*C*self.ext[0],2)[None,:];out[i,4:10]=moment[:6]*self.rf[:,None]*np.repeat(self.ext[0],2)[None,:]
            out[i,10:]=moment*self.af[:,None]*np.repeat(self.asca,2)[None,:]
            inactive=self.active==0;out[i,:,inactive]=field[i,:,inactive];self.timings['projection_seconds']+=time.perf_counter()-tt
            if time.perf_counter()-last>30:print(f'FULL STREAM {self.geo.name} cell={i}/{self.geo.ns} seconds={time.perf_counter()-t:.1f} RSS={b.peak()/1e9:.3f}GB',flush=True);last=time.perf_counter()
        return out
    def query(self,h,el,az,field=None,first=False):
        ee,aa=np.broadcast_arrays(el,az);shape=ee.shape;e,z=np.deg2rad(ee.ravel()),np.deg2rad(aa.ravel());p,up,ex=b.sh.old.probe.frame(np.array(b.R+b.sh.OBS),np.deg2rad(90-h));ey=np.cross(up,ex)
        d=np.sin(e)[:,None]*up+np.cos(e)[:,None]*(np.cos(z)[:,None]*ex+np.sin(z)[:,None]*ey);out=np.empty((len(d),self.W,2));field=self.ms if field is None else field
        active=self.active.copy();self.active[:]=1
        for j in range(0,len(d),64):
            dd=d[j:j+64];raw=self.geo.pack(np.broadcast_to(p,dd.shape),dd,getattr(self.geo,'query_n',24));out[j:j+len(dd)]=self.kernel.transport(raw,self,field,first)[...,:3].sum(-1)
        self.active=active;return out.reshape(*shape,self.W,2)*self.solar[:,None]
    def boxes(self,field,first=False):
        u,w=np.polynomial.legendre.leggauss(4);ww=(w[:,None]*w[None,:]/4).ravel();el=np.array([e+de for _,e,z in b.DIRS for de in u for dz in u]);az=np.array([z+dz for _,e,z in b.DIRS for de in u for dz in u])
        return np.stack([np.einsum('dnwl,n->dwl',self.query(h,el,az,field,first).reshape(7,16,self.W,2),ww) for h in b.HS])
    def solve(self,initial,max_eval=16,tolerance=.0005):
        x=initial.copy();history=[];gs=[[] for _ in range(self.V)];fs=[[] for _ in range(self.V)];done=np.zeros(self.V,bool);start=time.perf_counter()
        for it in range(1,max_eval+1):
            b.STATUS.update(stage=f'B residual {it}',case=self.geo.name);old=self.boxes(x).reshape(5,7,self.V);g=self.apply(x);new=self.boxes(g).reshape(5,7,self.V)
            rr=np.max(abs(new-old)/np.maximum(abs(new),1e-30),axis=(0,1));ss=np.max(abs(g-x),axis=(0,1))/np.maximum(np.max(abs(g),axis=(0,1)),1e-30)
            if it>=2:done|=np.maximum(rr,ss)<=tolerance
            history.append(dict(evaluation=it,observer_relative=rr.tolist(),source_relative=ss.tolist(),converged=done.tolist(),seconds=time.perf_counter()-start))
            print(f'FULL EVAL {self.geo.name} {it} maxObs={100*max(rr):.5f}% maxSrc={100*max(ss):.5f}% done={done.sum()}/{self.V}',flush=True)
            if done.all():x=g;break
            nextx=g.copy()
            for v in range(self.V):
                if done[v]:continue
                norm=np.linalg.norm(g[:,:,v],axis=1);scale=np.maximum(norm,norm.max()*1e-8);feature=((g[:,:,v]-x[:,:,v])/scale[:,None]).astype('f8').ravel();obs=((new[:,:,v]-old[:,:,v])/np.maximum(abs(new[:,:,v]),1e-20)).ravel()*np.sqrt(len(feature)/35)
                gs[v].append(g[:,:,v].copy());fs[v].append(np.r_[feature,obs])
                if len(fs[v])>4:fs[v].pop(0);gs[v].pop(0)
                if len(fs[v])>1:
                    f=np.stack(fs[v],1);G=f.T@f;G+=np.eye(len(G))*max(np.trace(G)*1e-10,1e-30);alpha=np.linalg.solve(G,np.ones(len(G)));alpha/=alpha.sum()
                    if np.isfinite(alpha).all() and max(abs(alpha))<8:nextx[:,:,v]=sum(c*f for c,f in zip(alpha,gs[v]))
            x=nextx;self.active=(~done).astype('i4');np.savez(self.folder/'checkpoint.npz',source=x)
            progress(RESULT,f'{self.geo.name} 잔차{it}: {done.sum()}/{self.V} 수렴')
        self.ms=x;self.active[:]=1
        return dict(seconds=time.perf_counter()-start,evaluations=it,converged=done.tolist(),history=history,prepare_seconds=self.prepare_seconds,geometry=json.loads(json.dumps(self.geo.meta)),timings=self.timings,peak_bytes=b.peak()),self.boxes(x)

def scalar_seed(s,grid,field):
    h,c=grid;f=a.interp(field,h,c,s.geo.h,s.geo.chi);incident=np.divide(f[:,:6],.013558*s.rf[None,:],out=np.zeros_like(f[:,:6]),where=s.rf[None,:]>0)
    # Mapping of the six low SH moments to local coherency M=integral I P/2.
    u,w=np.polynomial.legendre.leggauss(4);phi=np.arange(16)*2*np.pi/16;mu,ph=np.meshgrid(u,phi,indexing='ij');st=np.sqrt(1-mu*mu);glob=np.stack([st*np.cos(ph),st*np.sin(ph),mu],-1).reshape(-1,3);ww=np.broadcast_to(w[:,None]*2*np.pi/16,mu.shape).ravel();Y=s.kernel.basis(glob,2)
    out=np.empty((s.geo.ns,S,s.V),'f4');local=np.empty((s.geo.ns,4))
    for j,ch in enumerate(s.geo.chi):
        p,up,ex=b.sh.old.probe.frame(np.array(1.),ch);B=np.stack([ex,np.cross(up,ex),up],-1);d=glob@B
        Q=np.stack([(1-d[:,0]**2)/2,(1-d[:,1]**2)/2,(1-d[:,2]**2)/2,-d[:,0]*d[:,2]/2],-1);T=Y.T@(Q*ww[:,None]);idx=np.arange(j,s.geo.ns,len(s.geo.chi));local[idx]=incident[idx]@T
    out[:,:4]=local[:,:,None]*np.repeat(2*C*s.ext[0],2)[None,None,:]
    out[:,4:10]=f[:,:6,None]*np.repeat(s.ext[0]/.013558,2)[None,None,:]
    out[:,10:]=f[:,6:,None]*np.repeat(s.asca/(.125*.92),2)[None,None,:]
    return out

RESULT={}
def progress(r,state):
    r.update(state=state,peak_bytes=max(r.get('peak_bytes',0),b.peak()));b.save(OUT/'result.json',r)
    b.save(ROOT/'results/sos_full/full_result2.json',r)
    lines=['# SOS 편광·HG 통합 B',f'상태: {state}.','A2 통과 때만 실행. CPU4·계산1개·RSS8GB, 0.5초 감시7.5GB 자체 중단.',
        'Rayleigh: 국소 xx/yy/zz/xz coherency 텐서, 시선 표본마다 회전·투영. HG: 절단 없는 L32 SH.',
        '완전 탈편광 모형: F=p·diag(1,0,0,0), 입사 trace만 받고 I·Pω/2 방출. 편광 보존 모형: F=p·I₄, 산란면 기준 Stokes 보존. 둘 다 근사이며 어느 쪽도 상하한이 아님.',
        '분광은 엔진15칸; 직달태양·직달지면 산란은 원래 HG/소광. delta-M/TMS 없음. 기존 PT6장 읽기 전용.',
        '원천에서 밀도 분리, A2 선택 χ 보간 재사용. 텐서 log 후보는 물리 trace의 log값/선형 trace 비로 행렬을 배율(수치 보간).',
        '|사례|r/χ|파장|잔차평가|총s|','|---|---|---:|---:|---:|']
    for name,row in r.get('cases',{}).items():lines.append(f'|{name}|{row["nr"]}/{row["step"]}|{len(row["wl"])}|{row["stats"]["evaluations"]}|{row["stats"]["seconds"]:.1f}|')
    if r.get('pure_reference'):
        z=r['pure_reference'];lines.append(f'순수Rayleigh δ0·같은 기준 구적에서 원천을 새로 풂: 35방향 상대차 {z["maximum_relative"]:.3g}, 원천global {z["source_global_relative"]:.3g}, 1e−6 통과 {z["pass_"]}.')
    if 'scalar_off_max_relative' in r:lines.append(f'편광off/A2 최대차 {100*r["scalar_off_max_relative"]:.5f}%.')
    if 'scalar_off_operator' in r:
        z=r['scalar_off_operator'];lines.append(f'같은 A2 원천의 편광off 연산자 항등식: 최대상대차 {z["maximum"]:.3g}, 1e−6 통과 {z["pass_"]}. 정상 대기 원천 재수렴 검증은 별도 미확인.')
    if r.get('pt'):
        lines+=['|고도|single/PT 7칸|multiple/PT 7칸|total/PT 7칸|MS 5%/2SE|','|---|---|---|---|---|']
        for h,row in r['pt'].items():
            vals=['/'.join('N/A' if x is None else f'{x:.3f}' for x in row[k]['ratio']) for k in ('single','multiple','total')];m=row['multiple'];n=sum(m['eligible']);counts=[sum(v is True for v in m[k]) for k in ('within5pct','within2se')]
            lines.append('|'+h+'|'+'|'.join(vals)+f'|{counts[0]}/{n}, {counts[1]}/{n}|')
        lines+=['방향 sun2/sun10/sun30/anti10/anti30/side10/zenith89. 5%와2SE는 개별 판정. SE>15%와두seed0는 제외.',
            '판정 제외: '+json.dumps(r['excluded'],ensure_ascii=False),
            'PT MS 상대SE%: '+ '; '.join(h+'='+','.join('N/A' if x is None else f'{100*x:.2f}' for x in row['multiple']['se_relative']) for h,row in r['pt'].items())]
    if 'self_convergence' in r:lines.append(f'B multiple 공간 대조 최대 {100*r["self_convergence"]["maximum"]:.4f}%.')
    if 'panorama' in r:
        lines+=['15파장360×100 spec/rgb float32, +2/0/−3/−6/−9/−12. 같은분광→RGB행렬, 중앙Y→.18·x/(1+x)·sRGB.',
            '비교 PNG: 위통합/가운데PT lut0_fit/아래기존앱LUT.',
            '장당조회s: '+', '.join(h+'='+f'{v["query_seconds"]:.2f}' for h,v in r['panorama']['runs'].items())]
    if 'estimate_15w_25h_seconds' in r:lines.append(f'추측: 15파장×25고도 표 {r["estimate_15w_25h_seconds"]:.1f}s(원천1회+평균조회25회).')
    lines+=r.get('notes',[]);lines.append(f'최대RSS {r["peak_bytes"]/1e9:.3f}GB. B 산출완료 {r.get("complete",False)}·검증통과 {r.get("B_pass","미완료")}. C 미실행(범위 밖).');assert len(lines)<=80;DOC.write_text('\n'.join(lines)+'\n')

def pure_reference_check(kernel):
    """Check shared-field transport, then independently solve the pure case.
    Matching the reference quadratures separates identity from discretization.
    """
    import sos_pol as pol
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
        a.os.environ[key]='4'
    ref=pol.Solver(1,550,build=False);M=np.load(ROOT/'results/sos_pol/a0/w550/source.npz')['vector'];sun=ref.sun_table()
    g=types.SimpleNamespace(h=ref.h,chi=ref.chi,ns=ref.nv);g.poly,g.left=a.chi_poly(g.chi)
    def reference_pack(p,d,n):
        rows=[];points=[];bases=[];lits=[]
        for active,q,r,att in pol.p.segments(p,d,n,550):
            base,ah,ac,mu,cp=b.sh.spatial_lookup(q,r,d[active,None,:],g.h,g.chi);ts=sun.T(q,pol.SUN)
            rho=np.linalg.norm(q[...,:2],axis=-1);cx=np.divide(q[...,0],rho,out=np.ones_like(rho),where=rho>1e-12);sx=np.divide(q[...,1],rho,out=np.zeros_like(rho),where=rho>1e-12)
            ds=att/pol.p.density(r,550);sc=-np.log(np.maximum(ts,1e-300))/.013558;zero=np.zeros_like(sc)
            point=np.stack([ah,ac,mu,cp,ds,r-b.R,zero,zero,zero,sc,zero,zero,cx,sx],-1)
            rows.append(np.repeat(active,r.shape[1]));points.append(point.reshape(-1,14));bases.append(base.ravel());lits.append((ts>0).ravel())
        row=np.concatenate(rows);order=np.argsort(row,kind='stable');B=basis_at(p)
        return dict(off=np.r_[0,np.cumsum(np.bincount(row,minlength=len(d)))].astype('i8'),base=np.concatenate(bases)[order].astype('i4'),
            point=np.concatenate(points)[order].astype('f4'),lit=np.concatenate(lits)[order].astype('u1'),nu=d[:,2].copy(),target=B,dloc=np.einsum('nij,ni->nj',B,d))
    g.pack=reference_pack;g.query_n=48;s=object.__new__(Solver);s.geo=g;s.kernel=kernel;s.mode=0;s.delta=0;s.W=1;s.V=2;s.active=np.ones(2,'i4');s.ext=np.array([[.013558],[0.],[0.]])
    s.asca=np.array([0.]);s.hr=np.exp(-g.h/8);s.ha=np.exp(-g.h/1.2);s.solar=np.array([1.]);s.goff=np.zeros(g.ns+1,'i8');s.gdir=np.empty((0,3),'f4');s.gval=np.empty((0,1),'f4');s.gm=np.zeros((g.ns,4,1),'f4')
    s.ms=np.zeros((g.ns,S,2),'f4');s.ms[:,:4,0]=M[:g.ns]*2*C*.013558
    el=np.array([x[1] for x in b.DIRS]);az=np.array([x[2] for x in b.DIRS]);rows={}
    for h in b.HS:
        old=ref.query(h,el,az,mode='vector')+ref.query(h,el,az,M,'vector');new=s.query(h,el,az,first=True)[:,0,0]+s.query(h,el,az)[:,0,0]
        rel=b.sh.relative(new,old);rows[str(h)]=dict(relative=rel.tolist(),maximum=float(rel.max()),reference=old.tolist(),integrated=new.tolist())
    maximum=max(row['maximum'] for row in rows.values());transport=dict(maximum_relative=maximum,pass_=maximum<=1e-6,by_height=rows)
    # Fresh integrated source solve versus a freshly tightened cached reference.
    # The same discretization is essential: this is an implementation identity
    # test, while A2/B spatial comparisons test discretization independently.
    start=time.perf_counter();_,ref_sums,ref_stats=ref.solve(0.,tolerance=1e-9,maximum=45)
    old_grid=a.grids;a.grids=lambda *args:(ref.h.copy(),ref.chi.copy())
    try:gg=Geometry('pure_fresh_'+kernel.code_hash,20,.5)
    finally:a.grids=old_grid
    gg.pack=reference_pack;gg.query_n=48
    def reference_build(self,i,solver):
        ih,ic=divmod(i,len(self.chi));pp,dd,moment=pol.p.angular(b.R+self.h[ih],self.chi[ic],8,16);weight=moment[:,:3].sum(-1);B=basis_at(pp);loc=np.einsum('nij,ni->nj',B,dd)
        tt=time.perf_counter();z=reference_pack(pp,dd,24);self.meta['geometry_seconds']+=time.perf_counter()-tt
        tt=time.perf_counter();z['first']=solver.kernel.transport(z,solver,solver.dummy,True).astype('f4');self.meta['first_seconds']+=time.perf_counter()-tt
        z.update(loc=loc,weight=weight,ground_tr=np.zeros((len(dd),1)),end_index=np.zeros(len(dd),'i4'),end_fraction=np.zeros(len(dd)),
            wavelengths=solver.wl.copy(),delta=np.array(solver.delta),end_columns=np.zeros((len(dd),3)),hits_ground=np.zeros(len(dd),bool))
        self.meta['incoming_rays']+=len(dd);self.meta['points']+=len(z['point']);return z
    gg.build=types.MethodType(reference_build,gg)
    class Pure(Solver):
        def prepare_ground(self):
            self.goff=np.zeros(self.geo.ns+1,'i8');self.gdir=np.empty((0,3),'f4');self.gval=np.empty((0,1),'f4');self.gm=np.zeros((self.geo.ns,4,1),'f4')
        def boundary(self,field):return np.zeros((len(self.geo.chi),1,2))
    fresh=Pure(gg,kernel,[550.],0,0.);fresh.ext[1:]=0;fresh.asca[:]=0;fresh.solar[:]=1
    path=gg.folder/'case.json'
    if path.exists():stats=json.loads(path.read_text());fresh.ms=np.load(gg.folder/'source.npz')['source']
    else:
        stats,_=fresh.solve(np.zeros((gg.ns,S,2),'f4'),max_eval=28,tolerance=1e-7);b.save(path,stats);np.savez(gg.folder/'source.npz',source=fresh.ms)
    comparison={}
    for h in b.HS:
        old=ref.query(h,el,az,mode='vector')+ref.query(h,el,az,ref_sums['vector'],'vector');new=fresh.query(h,el,az,first=True)[:,0,0]+fresh.query(h,el,az)[:,0,0];rel=b.sh.relative(new,old)
        comparison[str(h)]=dict(relative=rel.tolist(),reference=old.tolist(),integrated=new.tolist())
    actual_M=fresh.ms[:,:4,0]/(2*C*.013558);refM=ref_sums['vector'][:gg.ns];field_error=float(np.max(abs(actual_M-refM))/np.max(abs(refM)))
    maximum=max(max(row['relative']) for row in comparison.values());gg.clean_cache()
    return dict(maximum_relative=maximum,source_global_relative=field_error,pass_=maximum<=1e-6 and field_error<=1e-6 and all(stats['converged']),
        by_height=comparison,transport_only=transport,integrated_stats=stats,reference_stats=ref_stats,seconds=time.perf_counter()-start,
        scope='Fresh source solves with matching sos_pol spatial/incoming/path/Sun discretization; delta0/aerosol0/ozone0/albedo0; scalar off and tensor normal comparison at normal A2 grid are separate.')

def validation(s,ar,r):
    data=s.boxes(s.ms);one=s.boxes(s.ms,True);r['pt']={};r['excluded']=[];r['scalar_off_comparison']={}
    selected=ar['minimal_setting'];name=f"r{selected['nr']}_c{int(round(selected['chi_step']*100)):03d}";row=ar['cases'][name];j=row['modes'].index(s.mode);reference=np.array(row['boxes'])[:,:,j]
    for ih,h in enumerate(b.HS):
        diff=b.sh.relative(data[ih,:,0,1],reference[ih]);r['scalar_off_comparison'][str(h)]=diff.tolist()
        if h not in (2,-3,-6):continue
        pt,metas=b.sh.old.pt_table(h);r['pt'][str(h)]={}
        for kind,values in (('single',one[ih,:,0,0]),('multiple',data[ih,:,0,0]),('total',one[ih,:,0,0]+data[ih,:,0,0])):
            mean=np.array(pt[kind]['mean']);absolute_se=np.array(pt[kind]['se']);se=np.divide(absolute_se,mean,out=np.full(7,np.inf),where=mean>0);eligible=(mean>0)&(se<=.15)
            ratio=np.divide(values,mean,out=np.zeros(7),where=mean>0);row=dict(sos=values.tolist(),pt_mean=mean.tolist(),ratio=[float(x) if m>0 else None for x,m in zip(ratio,mean)],
                se_relative=[float(x) if np.isfinite(x) else None for x in se],eligible=eligible.tolist(),within5pct=[bool(abs(x-1)<=.05) if ok else None for x,ok in zip(ratio,eligible)],within2se=[bool(abs(x-1)<=2*se[i]) if ok else None for i,(x,ok) in enumerate(zip(ratio,eligible))])
            r['pt'][str(h)][kind]=row
            for i in np.flatnonzero(~eligible):r['excluded'].append(dict(h=h,component=kind,direction=f'{b.DIRS[i][0]}{b.DIRS[i][1]}',reason='both_seeds_zero' if mean[i]==0 else 'relative_SE_above_15pct'))
        r.setdefault('pt_sources',{})[str(h)]=[str((ROOT/f'results/sos_sky/pt_e{h}_s{k}/spectral.npz').relative_to(ROOT)) for k in (1,2)]
    r['scalar_off_max_relative']=max(max(x) for x in r['scalar_off_comparison'].values())
    z=np.load(a.OUT/name/'source.npz');input_field=scalar_seed(s,(z['h'],z['chi']),z['source'][:,:,j]);z.close()
    shared=s.boxes(input_field)[:,:,0,1];delta=b.sh.relative(shared,reference)
    r['scalar_off_operator']=dict(directions=delta.tolist(),maximum=float(delta.max()),same_source=True,pass_=bool(delta.max()<=1e-6))
    progress(r,'550nm PT 재사용·편광off 비교 완료')

def scalar_operator_check(kernel,ar):
    """Independent scalar-off identity using the converged A2 source.
    A fresh normal-atmosphere fixed point is still a separate validation.
    """
    setting=ar['minimal_setting'];mode=a.NAMES.index(setting['candidate']);name=f"r{setting['nr']}_c{int(round(setting['chi_step']*100)):03d}"
    row=ar['cases'][name];j=row['modes'].index(mode);g=Geometry('operator550_'+kernel.code_hash,setting['nr'],setting['chi_step'],disk=False)
    s=Solver(g,kernel,[550.],mode,0.);z=np.load(a.OUT/name/'source.npz');s.ms=scalar_seed(s,(z['h'],z['chi']),z['source'][:,:,j]);z.close()
    start=time.perf_counter();actual=s.boxes(s.ms)[:,:,0,1];expected=np.array(row['boxes'])[:,:,j];rel=b.sh.relative(actual,expected)
    return dict(maximum=float(rel.max()),directions=rel.tolist(),pass_=bool(rel.max()<=1e-6),prepare_seconds=s.prepare_seconds,query_seconds=time.perf_counter()-start,scope='same-source scalar transport identity; not a fresh fixed point or a PT comparison')

def panoramas(s,r):
    helper=b.sh.old.predecessor_helpers();weights,residual=b.sh.old.rgb_weights();lut=helper['app_lut']();ee,aa=np.meshgrid(90-(np.arange(100)+.5)*.9,np.arange(360)+.5-180,indexing='ij');runs={};allcomp=[]
    for h in b.sh.PHS:
        b.STATUS.update(stage='B panorama',case=str(h));t=time.perf_counter();one=s.query(h,ee,aa,first=True);ms=s.query(h,ee,aa);query=time.perf_counter()-t
        spec=(one+ms)[...,:15,0];rgb=spec@weights;np.save(OUT/f'spec_e{h}.npy',spec.astype('f4'));np.save(OUT/f'rgb_e{h}.npy',rgb.astype('f4'));helper['to_png'](rgb,OUT/f'pano_e{h}.png')
        full=helper['pano_from_grid'](rgb[:,180:][::-1],ee[:,0][::-1],np.arange(180)+.5);pt=helper['pano_from_fold'](helper['fold_table'](h));app=helper['app_pano'](h,lut)
        comp=np.concatenate([helper['label'](helper['to_png'](full),f'Integrated polarized + completely depolarizing aerosol hs {h:+d}'),helper['label'](helper['to_png'](pt),f'PT lut0_fit hs {h:+d}'),helper['label'](helper['to_png'](app),f'app LUT hs {h:+d}')]);Image.fromarray(comp).save(OUT/f'compare_e{h}.png');allcomp.append(comp)
        runs[str(h)]=dict(query_seconds=query,output_seconds=time.perf_counter()-t,negative=int((spec<0).sum()),nonfinite=int((~np.isfinite(spec)).sum()),shape=list(spec.shape));r['panorama']=dict(runs=runs,rgb_fit_residual=residual)
        progress(r,f'15파장 파노라마 {h:+d} 완료');print(f'FULL PANO {h} query={query:.2f}s',flush=True)
    Image.fromarray(np.concatenate(allcomp,axis=1)).save(OUT/'compare_all.png');np.save(OUT/'rgb_weights.npy',weights)
    total=r['cases']['spectral']['stats']['seconds']+r['cases']['spectral']['stats']['prepare_seconds'];r['estimate_15w_25h_seconds']=total+25*np.mean([v['query_seconds'] for v in runs.values()])

def gallery(r):
    import b2_gallery
    b2_gallery.run(r)


def run(delta):
    global RESULT
    ar=json.loads((ROOT/'results/sos_full/a2_result.json').read_text());assert ar['A2_pass'],'A2_GATE_FAILED'
    setting=ar['minimal_setting'];mode=a.NAMES.index(setting['candidate']);RESULT=dict(A2_setting=setting,delta=delta,cases={},notes=[]);progress(RESULT,'B 검증 시작')
    kernel=Kernel();RESULT['pure_reference']=json.loads((ROOT/'results/sos_full/b_run/pure_validation.json').read_text())['pure_reference'];progress(RESULT,'순수Rayleigh 기준 조회 검사 완료')
    name=f"r{setting['nr']}_c{int(round(setting['chi_step']*100)):03d}";src=np.load(a.OUT/name/'source.npz');j=ar['cases'][name]['modes'].index(mode);warm=(src['h'],src['chi'],src['source'][:,:,j]);src.close()
    def case(name,nr,step,wl,geometry=None,cleanup=True):
        b.STATUS.update(stage='B prepare',case=name);progress(RESULT,name+' 준비')
        g=geometry if geometry is not None else Geometry(f'geometry_{name}_d{delta:g}_{mode}_{kernel.code_hash}',nr,step,compact=name=='fine550')
        if geometry is not None:g.ram.clear();g.ram_bytes=0
        s=Solver(g,kernel,wl,mode,delta);s.folder=OUT/f'{name}_d{delta:g}_{mode}_{kernel.code_hash}';s.folder.mkdir(exist_ok=True);path=s.folder/'case.json';before_geometry=json.loads(json.dumps(g.meta))
        if path.exists():
            row=json.loads(path.read_text());assert row['delta']==delta and row['mode']==mode and row['wl']==list(wl) and row['kernel_hash']==kernel.code_hash
            s.ms=np.load(s.folder/'source.npz')['source']
        else:
            initial=scalar_seed(s,warm[:2],warm[2]);stats,boxes=s.solve(initial);stats['geometry_increment']={k:stats['geometry'][k]-before_geometry[k] for k in ('geometry_seconds','first_seconds','compression_seconds','io_seconds')}
            row=dict(nr=nr,step=step,wl=list(wl),delta=delta,mode=mode,kernel_hash=kernel.code_hash,geometry_reused=geometry is not None,stats=stats,boxes=boxes.tolist());b.save(path,row);np.savez(s.folder/'source.npz',source=s.ms,h=g.h,chi=g.chi)
        RESULT['cases'][name]=row;progress(RESULT,name+' 완료')
        if cleanup:g.clean_cache()
        return s
    base=case('base550',setting['nr'],setting['chi_step'],[550.],cleanup=False);validation(base,ar,RESULT)
    fine=case('fine550',39,.25,[550.]);coarse=np.array(RESULT['cases']['base550']['boxes'])[:,:,:,0];dense=np.array(RESULT['cases']['fine550']['boxes'])[:,:,:,0];rel=b.sh.relative(coarse,dense)
    RESULT['self_convergence']=dict(directions=rel[:,:,0].tolist(),maximum=float(rel.max()));progress(RESULT,'B 공간 대조 완료')
    del fine;gc.collect()
    import b2_preserve
    RESULT['models']=b2_preserve.compare(base,kernel,RESULT);progress(RESULT,'완전 탈편광 대 편광 보존 비교 완료')
    spec=case('spectral',setting['nr'],setting['chi_step'],list(b.sh.WL),geometry=base.geo);panoramas(spec,RESULT)
    del spec,base;gc.collect()
    RESULT['numerical_outputs_complete']=True;RESULT['B_pass']=bool(RESULT['pure_reference']['pass_'] and RESULT['scalar_off_operator']['pass_'] and RESULT['scalar_off_max_relative']<=.001 and RESULT['self_convergence']['maximum']<=.02 and all(all(x['stats']['converged']) for x in RESULT['cases'].values()))
    RESULT['notes']+=['순수Rayleigh는 같은 기준 공간/입사/경로/태양열 구적·δ0에서 각각 원천을 새로 풀어1e−6 대조. 정상 대기의 δ는 별도 명시.',
        '한(r,χ) 해에서 여러 관측 태양고도는 조회만 재사용. 기하·종별 광학열은 파장 변경 때 공유; 감쇠·직달·계수·반복은 재계산.']
    progress(RESULT,'B 수치·분광 산출 완료');gallery(RESULT);RESULT['complete']=True;progress(RESULT,'B 전체 산출 완료');b.save(ROOT/'results/sos_full/full_result2.json',RESULT);print('FULL COMPLETE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--delta',type=float);ap.add_argument('--pure-only',action='store_true');ap.add_argument('--deadline-utc');args=ap.parse_args()
    if not args.pure_only and args.delta is None:ap.error('--delta is required for the normal atmosphere')
    def frozen_snapshot():
        snapshot={k:v for k,v in b.manifest().items() if k!='docs/SOS_FULL.md'}
        for name in ('scripts/sos_render.py','src/engine.py'):
            p=ROOT/name;snapshot[name]=[p.stat().st_size,p.stat().st_mtime_ns]
        return snapshot
    frozen_before=frozen_snapshot()
    if args.deadline_utc:
        from datetime import datetime,timezone
        import signal
        deadline=datetime.fromisoformat(args.deadline_utc).replace(tzinfo=timezone.utc).timestamp()
        def stop_at_deadline():
            while not b.STOP.wait(.5):
                if time.time()>=deadline:
                    RESULT['deadline_stop_utc']=args.deadline_utc
                    print('B DEADLINE STOP; preserving last checkpoint',flush=True)
                    a.os.kill(a.os.getpid(),signal.SIGINT);return
        a.threading.Thread(target=stop_at_deadline,daemon=True).start()
    try:
        if args.pure_only:
            ar=json.loads((ROOT/'results/sos_full/a2_result.json').read_text());assert ar['A2_pass'],'A2_GATE_FAILED'
            RESULT.update(A2_setting=ar['minimal_setting'],cases={},notes=['순수 기준 검사만 실행; 정상 대기·PT·15파장·갤러리는 아직 미실행.'])
            kernel=Kernel();progress(RESULT,'B 편광off 동일 원천 연산자 검사');RESULT['scalar_off_operator']=scalar_operator_check(kernel,ar);progress(RESULT,'B 순수Rayleigh 검사 시작');RESULT['pure_reference']=pure_reference_check(kernel);progress(RESULT,'B 순수Rayleigh 검사 완료')
            b.save(ROOT/'results/sos_full/full_result2.json',RESULT)
        else:run(args.delta)
    except BaseException as exc:
        RESULT['interruption']=dict(type=type(exc).__name__,message=str(exc),last_state=RESULT.get('state'))
        progress(RESULT,'B 중단; 체크포인트 보존')
        raise
    finally:
        RESULT['frozen_preserved']=frozen_before==frozen_snapshot()
        progress(RESULT,RESULT.get('state','B 종료'))
        b.STOP.set()
