#!/usr/bin/env python3
"""A2: cell-streamed density-normalized, log and chi-cubic SOS probes.
No old script or result is modified. The three candidates share geometry.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
import argparse,collections,gc,inspect,json,math,sys,threading,time,types,zipfile
from pathlib import Path
import stream_driver as b
b.STOP.set()
import numpy as np
ROOT=b.ROOT; OUT=ROOT/'results/sos_full/a2_run'; OUT.mkdir(exist_ok=True)
DOC=ROOT/'docs/SOS_QUAD.md'; OLD=ROOT/'results/sos_full/stream_run'
OLD_GRID=b.legacy.grids; BASE_H,BASE_C=OLD_GRID(); b.OUT=OUT;b.sh.OUT=OUT
b.STOP=threading.Event();b.STATUS={'stage':'A2 import','case':None}
def bounded_monitor():
    last=time.monotonic()
    while not b.STOP.wait(.5):
        value=b.peak()
        if value>=b.SOFT:
            target=b.OUT/'memory_abort.json';b.save(target,dict(reason='RSS_7_5_GB_soft_stop',peak_bytes=value,cap_bytes=b.CAP,status=b.STATUS))
            b.DOC.write_text(f'# SOS RSS 안전 중단\n최대RSS {value/1e9:.3f}GB. 상세 {target.relative_to(ROOT)}\n')
            print('MEMORY STOP',value,b.STATUS,flush=True);os._exit(75)
        if time.monotonic()-last>=60:print(f'HEARTBEAT RSS_peak={value/1e9:.3f}GB {b.STATUS}',flush=True);last=time.monotonic()
b.monitor=bounded_monitor
threading.Thread(target=b.monitor,daemon=True).start()
CPP=r'''
extern "C" void source_interp(int nr,int L,int nc,int ns,int V,const int*mode,
 const int64_t*off,const int32_t*base,const float*p,const double*nu,const float*src,
 const double*hr,const double*ha,const double*poly,const int*left,double*out){
 auto worker=[&](int lo,int hi){
  Norm norm(L);int K=norm.K,M=L+1,S=K+6,C=M+3;
  std::vector<double>Y(K);std::vector<int>stamp((int64_t)ns*V,-1);
  std::vector<float>cache((int64_t)ns*V*C);
  for(int ray=lo;ray<hi;ray++){
   norm.eval(nu[ray],1.,Y.data());std::vector<double>sum(V,0.);
   for(int64_t n=off[ray];n<off[ray+1];n++){
    const float*q=p+6*n;int ir=base[n]/nc,j=base[n]%nc;
    double ah=q[0],t=q[1],cp=q[2],cm[65];cm[0]=1;
    if(M>1)cm[1]=cp;for(int m=2;m<M;m++)cm[m]=2*cp*cm[m-1]-cm[m-2];
    for(int v=0;v<V;v++)if(mode[v]>=0){
     int count=mode[v]==2?4:2;int start=mode[v]==2?left[j]:j;
     double wr[4];if(count==2){wr[0]=1-t;wr[1]=t;}
     else for(int c=0;c<4;c++){const double*a=poly+16*j+4*c;wr[c]=((a[3]*t+a[2])*t+a[1])*t+a[0];}
     double val=0,logs=0;bool positive=true;
     for(int h=0;h<2;h++)for(int c=0;c<count;c++){
      int cell=(ir+h)*nc+start+c,tag=cell*V+v;
      float*z=cache.data()+(int64_t)tag*C;
      if(stamp[tag]!=ray){
       std::fill(z,z+C,0.f);const float*s=src+(int64_t)cell*S*V+v;
       for(int l=0;l<=2;l++)for(int m=0;m<=l;m++)z[M+m]+=(float)Y[ix(l,m)]*s[ix(l,m)*V];
       for(int l=0;l<=L;l++)for(int m=0;m<=l;m++)z[m]+=(float)Y[ix(l,m)]*s[(6+ix(l,m))*V];
       stamp[tag]=ray;
      }
      double r=0,a=0;for(int m=0;m<M;m++){a+=cm[m]*z[m];if(m<=2)r+=cm[m]*z[M+m];}
      double w=(h?ah:1-ah)*wr[c];
      double value=mode[v]==1?hr[ir+h]*r+ha[ir+h]*a:q[4]*r+q[5]*a;
      val+=w*value;
      if(mode[v]==1){if(value<=0)positive=false;else logs+=w*log(value);}
     }
     if(mode[v]==1&&positive)val=exp(logs);
     sum[v]+=q[3]*val;
    }
   }
   for(int v=0;v<V;v++)out[(int64_t)ray*V+v]=sum[v];
  }
 };
 std::vector<std::thread>ts;for(int k=0;k<4;k++)ts.emplace_back(worker,nr*k/4,nr*(k+1)/4);for(auto&t:ts)t.join();
}
'''
b.sh.CPP+=CPP
NAMES=('density','log','cubic'); K=b.K; S=K+6

def grids(nr=20,step=.5):
    h=BASE_H.copy()
    if nr==30:h=np.sort(np.r_[h,(h[:-1]+h[1:])[9:19]/2])
    elif nr==39:h=np.sort(np.r_[h,(h[:-1]+h[1:])/2])
    elif nr!=20:raise ValueError(nr)
    ch=BASE_C.copy()
    if step<.5:
        degrees=np.round(np.rad2deg(ch),12);ch=np.deg2rad(np.unique(np.r_[degrees[(degrees<85)|(degrees>110)],np.arange(85,110+step/2,step)]))
    assert np.all(np.diff(ch)>1e-10)
    return h,ch

def chi_poly(ch):
    poly=np.empty((len(ch)-1,4,4));left=np.empty(len(ch)-1,'i4')
    for j in range(len(ch)-1):
        lo=min(max(j-1,0),len(ch)-4);left[j]=lo;x=(ch[lo:lo+4]-ch[j])/(ch[j+1]-ch[j])
        for c in range(4):
            p=np.polynomial.Polynomial([1.])
            for k in range(4):
                if k!=c:p*=np.polynomial.Polynomial([-x[k],1])/(x[c]-x[k])
            poly[j,c]=p.coef
    return poly,left

class Geometry(b.CellGeometry):
    def __init__(self,name,nr=20,step=.5,disk=True):
        h,ch=grids(nr,step);b.legacy.grids=lambda *args:(h.copy(),ch.copy())
        super().__init__(name,nmu=12,nphi=64,a=4.,disk=disk,shuffle=True)
        self.cfg.update(density=True,nr=nr,chi_step=step)
        self.meta['cache_bytes']=sum(p.stat().st_size for p in self.cache.glob('*.npz'))
        self.poly,self.left=chi_poly(ch)
    def cell(self,i,solver):
        if i in self.ram:
            z=self.ram.pop(i);self.ram[i]=z;return z
        path=self.cache/f'cell_{i:05d}.npz'
        if path.exists():
            t=b.clock()
            with np.load(path) as f:z={k:f[k] for k in f.files}
            e=z.pop('point_bytes');z['point']=np.ascontiguousarray(e.transpose(2,0,1)).view('f4').reshape(e.shape[-1],e.shape[0]);del e
            self.meta['io_seconds']+=b.clock()-t
        else:
            z=self.build(i,solver)
            import shutil
            can_cache=self.disk and self.meta['cache_bytes']<14_000_000_000 and shutil.disk_usage(OUT).free>768*1024**2
            if can_cache:
                t=b.clock();stored=dict(z);p=z['point'];e=np.ascontiguousarray(p).view('u1').reshape(len(p),p.shape[1],4).transpose(1,2,0).copy()
                back=np.ascontiguousarray(e.transpose(2,0,1)).view('f4').reshape(p.shape);assert np.array_equal(p.view('u4'),back.view('u4'))
                stored['point_bytes']=e;del stored['point'];del back;b.zipped(path,stored)
                self.meta['compression_seconds']+=b.clock()-t;self.meta['cache_bytes']+=path.stat().st_size
            elif self.disk:
                self.meta['uncached_cell_builds']=self.meta.get('uncached_cell_builds',0)+1
        size=sum(v.nbytes for v in z.values())
        while self.ram and self.ram_bytes+size>self.ram_limit:
            _,old=self.ram.popitem(last=False);self.ram_bytes-=sum(v.nbytes for v in old.values())
        if size<=self.ram_limit:self.ram[i]=z;self.ram_bytes+=size
        return z

class Kernel(b.legacy.Kernel):
    def __init__(self):
        super().__init__();self.lib.source_interp.argtypes=[b.sh.ctypes.c_int]*5+[b.sh.ctypes.c_void_p]*11
    def transport(self,z,solver,source,first=False):
        if first:return super().transport(z,solver,source,True)
        if z['point'].shape[1]!=6:z=b.legacy.compact_block(z,solver)
        source=np.ascontiguousarray(source,'f4');V=source.shape[-1];out=np.empty((len(z['nu']),V))
        args=[solver.active_modes,z['off'],z['base'],z['point'],z['nu'],source,solver.hr,solver.ha,solver.geo.poly,solver.geo.left,out]
        args=[np.ascontiguousarray(a) for a in args]
        self.lib.source_interp(len(out),32,len(solver.geo.chi),solver.geo.ns,V,*[a.ctypes.data for a in args])
        return out

class Solver(b.StreamSolver):
    def __init__(self,g,kernel,modes=(0,1,2)):
        super().__init__(g,kernel);self.V=len(modes);self.modes=np.array(modes,'i4');self.active_modes=self.modes.copy();self.nc=S
        self.hr=np.exp(-g.h/8);self.ha=np.exp(-g.h/1.2)
        self.rf=((self.levels[:6]==0)+.1*(self.levels[:6]==2))*self.ext[0,0]
        self.af=.8**self.levels*self.asca[0]
    def radiance(self,i,source,boundary=None):
        z=self.geo.cell(i,self);t=b.clock();value=self.kernel.transport(z,self,source)+z['first'][:,None]
        self.timings['transport_seconds']+=b.clock()-t
        if boundary is not None:value+=((1-z['end_fraction'][:,None])*boundary[z['end_index']]+z['end_fraction'][:,None]*boundary[z['end_index']+1])*z['ground_tr'][:,None]
        return z,value
    def boundary(self,source):
        t=b.clock();out=np.empty((len(self.geo.chi),self.V))
        for ic in range(len(out)):
            z,L=self.radiance(ic,source);out[ic]=b.sh.NORMAL['albedo']/np.pi*((z['weight']*np.maximum(0,z['loc'][:,2]))@L)
        self.timings['ground_seconds']+=b.clock()-t;return out
    def apply(self,source):
        boundary=self.boundary(source);out=np.empty_like(source);t=b.clock();last=t
        for i in range(self.geo.ns):
            z,L=self.radiance(i,source,boundary);tt=b.clock();ch=self.geo.chi[i%len(self.geo.chi)];v=z['loc']
            glob=np.column_stack([v[:,2]*np.sin(ch)-v[:,0]*np.cos(ch),-v[:,1],v[:,2]*np.cos(ch)+v[:,0]*np.sin(ch)])
            Y=self.kernel.basis(glob,32);mom=Y.T@(L*z['weight'][:,None]);out[i]=np.concatenate([mom[:6]*self.rf[:,None],mom*self.af[:,None]],axis=0)
            for j in range(self.V):
                if self.active_modes[j]<0:out[i,:,j]=source[i,:,j]
            self.timings['projection_seconds']+=b.clock()-tt
            if b.clock()-last>30:print(f'A2 STREAM {self.geo.name} cells={i}/{self.geo.ns} elapsed={b.clock()-t:.1f}s RSS={b.peak()/1e9:.3f}GB',flush=True);last=b.clock()
        return out
    def query(self,h,el,az,source=None,first=False):
        ee,aa=np.broadcast_arrays(el,az);shape=ee.shape;e,a=np.deg2rad(ee.ravel()),np.deg2rad(aa.ravel())
        p,up,ex=b.sh.old.probe.frame(np.array(b.R+b.sh.OBS),np.deg2rad(90-h));ey=np.cross(up,ex)
        d=np.sin(e)[:,None]*up+np.cos(e)[:,None]*(np.cos(a)[:,None]*ex+np.sin(a)[:,None]*ey)
        out=np.empty((len(d),1 if first else self.V));field=self.ms if source is None else source
        active=self.active_modes.copy();self.active_modes=self.modes.copy()
        for start in range(0,len(d),64):
            dd=d[start:start+64];z=self.geo.pack(np.broadcast_to(p,dd.shape),dd,24);out[start:start+len(dd)]=self.kernel.transport(z,self,field,first)
        self.active_modes=active;return out.reshape(*shape,out.shape[-1])*self.solar
    def boxes(self,source):
        u,w=np.polynomial.legendre.leggauss(4);ww=(w[:,None]*w[None,:]/4).ravel()
        el=np.array([e+de for _,e,a in b.DIRS for de in u for da in u]);az=np.array([a+da for _,e,a in b.DIRS for de in u for da in u])
        return np.stack([np.einsum('dnv,n->dv',self.query(h,el,az,source).reshape(7,16,self.V),ww) for h in b.HS])
    def solve(self,initial,max_eval=14,resume=None):
        t=b.clock();x=initial.copy();history=[];xs=[[] for _ in range(self.V)];gs=[[] for _ in range(self.V)];fs=[[] for _ in range(self.V)];done=np.zeros(self.V,bool)
        prior=(resume or {}).get('evaluations',0);prior_seconds=(resume or {}).get('seconds',0.)
        for it in range(1,max_eval+1):
            b.STATUS.update(case=self.geo.name,stage=f'Anderson residual evaluation {it+prior}');old=self.boxes(x);g=self.apply(x);new=self.boxes(g)
            rr=np.max(abs(new-old)/np.maximum(abs(new),1e-30),axis=(0,1));ss=np.max(abs(g-x),axis=(0,1))/np.maximum(np.max(abs(g),axis=(0,1)),1e-30)
            if it+prior>=2:done|=(np.maximum(rr,ss)<=.0005)
            history.append(dict(evaluation=it+prior,observer_relative=rr.tolist(),source_relative=ss.tolist(),seconds=prior_seconds+b.clock()-t,converged=done.tolist()))
            print(f'A2 EVAL {self.geo.name} {it+prior} obs%={np.round(100*rr,5)} src%={np.round(100*ss,5)} done={done} elapsed={prior_seconds+b.clock()-t:.1f}s',flush=True)
            if done.all():x=g;break
            nextx=g.copy()
            for v in range(self.V):
                if done[v]:continue
                scale=np.maximum(np.linalg.norm(g[:,:,v],axis=1),np.max(np.linalg.norm(g[:,:,v],axis=1))*1e-8)
                feature=((g[:,:,v]-x[:,:,v])/scale[:,None]).astype('f8').ravel()
                obs=((new[:,:,v]-old[:,:,v])/np.maximum(abs(new[:,:,v]),1e-20)).ravel()*np.sqrt(len(feature)/35)
                f=np.r_[feature,obs];xs[v].append(x[:,:,v].copy());gs[v].append(g[:,:,v].copy());fs[v].append(f)
                if len(fs[v])>4:xs[v].pop(0);gs[v].pop(0);fs[v].pop(0)
                if len(fs[v])>=2:
                    ff=np.stack(fs[v],1);G=ff.T@ff;G+=np.eye(len(G))*max(np.trace(G)*1e-10,1e-30)
                    alpha=np.linalg.solve(G,np.ones(len(G)));alpha/=alpha.sum()
                    if np.isfinite(alpha).all() and max(abs(alpha))<8:nextx[:,:,v]=sum(a*z for a,z in zip(alpha,gs[v]))
            x=nextx;self.active_modes=np.where(done,-1,self.modes).astype('i4')
            np.savez(self.geo.folder/'checkpoint.npz',source=x)
            save_progress(CURRENT,f'{self.geo.name}: 잔차 평가{it+prior}, 관측최대 {100*max(rr):.4f}%')
        self.active_modes=self.modes.copy();self.ms=x;boxes=self.boxes(x)
        return dict(seconds=prior_seconds+b.clock()-t,prepare_seconds=self.prepare_seconds,history=history,converged=done.tolist(),evaluations=it+prior,geometry=self.geo.meta,timings=self.timings,
            peak_bytes=max(b.peak(),(resume or {}).get('peak_bytes',0)),resume=resume,geometry_timing_scope='after_resume_only' if resume else 'whole_case'),boxes

def seed(g,solver,old_name=None,previous=None):
    if previous is not None:
        hh,cc,field=previous;out=np.empty((g.ns,S,solver.V),'f4')
        for v in range(solver.V):
            tmp=b.interpolate(field[:,:,v,None],hh,cc,g.h,g.chi) if field.shape[1]==K else interp(field[:,:,v],hh,cc,g.h,g.chi)
            out[:,:,v]=tmp.reshape(g.ns,S)
        return out
    name=old_name or ('space39' if len(g.h)==39 and len(g.chi)>75 else 'mu12');h0,c0=OLD_GRID(2,2) if name=='space39' else OLD_GRID()
    J=np.load(OLD/name/'source.npz')['multiple'][:,:,0]
    levels=np.array([l for l in range(33) for m in range(l+1)]);f=np.array([np.exp(-h/8)*((levels==0)+.1*(levels==2))*solver.ext[0,0]+np.exp(-h/1.2)*solver.af for h in h0]);f=np.repeat(f,len(c0),0)
    mom=np.divide(J,f,out=np.zeros_like(J),where=f>1e-35);normalized=np.concatenate([mom[:,:6]*solver.rf,mom*solver.af],1).astype('f4')
    if not(np.array_equal(g.h,h0) and np.array_equal(g.chi,c0)):normalized=interp(normalized,h0,c0,g.h,g.chi)
    return np.repeat(normalized[:,:,None],solver.V,axis=2)

def interp(field,h0,c0,h,c):
    n=field.shape[-1];field=field.reshape(len(h0),len(c0),n);tmp=np.empty((len(h0),len(c),n))
    for i in range(len(h0)):
        j=np.clip(np.searchsorted(c0,c,side='right')-1,0,len(c0)-2);a=(c-c0[j])/(c0[j+1]-c0[j]);tmp[i]=(1-a[:,None])*field[i,j]+a[:,None]*field[i,j+1]
    j=np.clip(np.searchsorted(h0,h,side='right')-1,0,len(h0)-2);a=(h-h0[j])/(h0[j+1]-h0[j]);return ((1-a[:,None,None])*tmp[j]+a[:,None,None]*tmp[j+1]).reshape(-1,n).astype('f4')

CURRENT={}
def save_progress(r,state):
    peaks=[r.get('peak_bytes',0),b.peak()]+[x['stats']['peak_bytes'] for x in r.get('cases',{}).values()]
    r.update(state=state,peak_bytes=max(peaks));b.save(OUT/'result.json',r)
    lines=['# SOS A2 공간 수렴','',f'상태: {state}.','μ12/φ64/a4, HG L32, normal/.30, 550 nm. RSS8GB/0.5초 감시7.5GB 중단·CPU4·계산1개.',
      '이전 입력 읽기 전용. 새 실행기 results/sos_full/a2_driver.py; 셀별 스트리밍·128MiB 캐시·무손실 바이트 압축. 디스크 캐시14GB 초과/잔여768MiB 이하는 동일 블록 재계산(격자 유지).',
      '(a) Rayleigh/Aerosol 원천에서 각 밀도를 분리, 실제 고도에서 지수 밀도 곱함. (b) 방향별 물리J>0인 네 모서리만 log; 그 외 선형. SH 계수 자체에는 log하지 않음.',
      '(c) r는(a), χ는 비균일 네 점 Lagrange3차. signed SH/3차의 음수는 임의로 clamp하지 않음.',
      'Anderson4는 반복 가속만; F(x)-x의 원천·35조회 잔차 모두.05% 이하로 검증. 통과선은 공간변화2% 그대로.',
      '|격자|후보|평가수|총 s|기하 s|투영 s|','|---|---|---:|---:|---:|---:|']
    for name,row in r.get('cases',{}).items():
        for v,mode in enumerate(row['modes']):
            s=row['stats'];lines.append(f'|{name}|{NAMES[mode]}|{s["evaluations"]}|{s["seconds"]:.1f}|{s["geometry"]["geometry_seconds"]:.1f}|{s["timings"]["projection_seconds"]:.1f}|')
        if row['stats'].get('resume'):
            z=row['stats']['resume'];lines.append(f'{name} 총시간은 이전 두 평가 {z["seconds"]:.1f}s와 재시작 계산의 합(중단된 일부 평가·반복 준비 제외). 초기 기하+F1 {z["historical_initial_geometry_and_first_evaluation_seconds"]:.1f}s. 이 사례의 기하/투영 상세계측은 재시작 이후만; 최소설정 사례는 전 과정 계측.')
    if r.get('comparisons'):
        lines+=['','|비교|후보|+2|−3|−6|−9|−12|최대|','|---|---|---:|---:|---:|---:|---:|---:|']
        for name,row in r['comparisons'].items():
            for label,vals in row.items():lines.append('|'+name+'|'+label+'|'+'|'.join(f'{100*x:.3f}%' for x in vals['by_height'])+f'|{100*vals["maximum"]:.3f}%|')
    if r.get('minimal_setting'):
        z=r['minimal_setting'];lines.append(f'최소 시험 통과 설정: r{z["nr"]}, χ {z["chi_step"]}°(85…110°), {z["candidate"]}. r20은20×75=1500점, r39/.25는39×125=4875점. .125° 추가 정련은 기준 통과로 불필요.')
        st=r['cases']['r20_c050']['stats'];lines.append(f'기본 비용: 준비 {st["prepare_seconds"]:.2f}s, 기하 {st["geometry"]["geometry_seconds"]:.2f}s, 반복 포함 {st["seconds"]:.2f}s(세 후보 동시 평가4회; log 단독 시간은 미측정, 기하 시간은 총시간에 포함).')
    for h,z in r.get('lookups',{}).items():lines.append(f'{h}° 360×100 조회: single {z["single_seconds"]:.3f}s / multiple {z["multiple_seconds"]:.3f}s; 전하늘 공간변화 최대 {z["full_sky_max_percent"]:.3f}%.')
    if r.get('source_hotspot'):
        z=r['source_hotspot'];lines.append(f'원천계수 norm 차이 hotspot: 고도 {z["height_km"]:g}km·χ {z["chi_deg"]:g}°, {z["percent"]:.3f}%(휘도 오차와 다른 지표).')
    lines+=r.get('notes',[]);lines.append(f'전체 최대RSS {r["peak_bytes"]/1e9:.3f}GB; 판정 {r.get("A2_pass","미완료")}; B 도달 {r.get("B_started",False)}.')
    DOC.write_text('\n'.join(lines)+'\n')

def compare(r,a,c):
    x=np.array(r['cases'][a]['boxes']);y=np.array(r['cases'][c]['boxes']);out={}
    for v,mode in enumerate(r['cases'][a]['modes']):
        vv=r['cases'][c]['modes'].index(mode);rel=b.sh.relative(x[:,:,v],y[:,:,vv]);out[NAMES[mode]]=dict(by_height=rel.max(1).tolist(),maximum=float(rel.max()),directions=rel.tolist())
    r['comparisons'][a+'→'+c]=out;return out

def run():
    global CURRENT
    before=b.manifest();kernel=Kernel();r=dict(cases={},comparisons={},notes=[],dirs=b.DIRS,hs=list(b.HS));CURRENT=r;save_progress(r,'세 후보·공유 기하 시작')
    def case(name,nr,step,modes=(0,1,2),previous=None):
        b.STATUS.update(case=name,stage='prepare');save_progress(r,name+' 준비')
        g=Geometry(name,nr,step);done=g.folder/'case.json'
        if done.exists():
            data=json.loads(done.read_text())
            with np.load(g.folder/'source.npz') as z:
                assert np.array_equal(g.h,z['h']) and np.array_equal(g.chi,z['chi']),'CACHE_GRID_MISMATCH'
                s=types.SimpleNamespace(ms=z['source'])
        else:
            s=Solver(g,kernel,modes)
            resume_path=g.folder/'resume.json';resume=json.loads(resume_path.read_text()) if resume_path.exists() else None
            initial=np.load(g.folder/'checkpoint.npz')['source'] if resume else seed(g,s,previous=previous)
            stats,boxes=s.solve(initial,resume=resume)
            data=dict(nr=nr,chi_step=step,modes=list(modes),stats=stats,boxes=boxes.tolist());np.savez(g.folder/'source.npz',source=s.ms,h=g.h,chi=g.chi);b.save(done,data)
        r['cases'][name]=data;save_progress(r,name+' 완료');g.clean_cache();return g,s
    g20,s20=case('r20_c050',20,.5);g39,s39=case('r39_c025',39,.25)
    results=compare(r,'r20_c050','r39_c025');save_progress(r,'세 후보 공간×2 완료')
    viable=[v for v,m in enumerate(NAMES) if results[m]['maximum']<=.02 and r['cases']['r20_c050']['stats']['converged'][v] and r['cases']['r39_c025']['stats']['converged'][v]]
    chosen=(0 if 0 in viable else min(viable,key=lambda v:results[NAMES[v]]['maximum'])) if viable else int(np.argmin([results[m]['maximum'] for m in NAMES]))
    r['chosen_candidate']=NAMES[chosen]
    prev=(g39.h,g39.chi,s39.ms[:,:,chosen:chosen+1]);g30,s30=case('r30_c050',30,.5,(chosen,),prev)
    # Compare the radial30 ladder against the common finer reference.
    compare(r,'r30_c050','r39_c025');save_progress(r,'r20/30/39 비균일 격자 완료')
    r['A2_pass']=bool(viable)
    if viable:r['minimal_setting']=dict(nr=20,chi_step=.5,candidate=NAMES[chosen])
    elif r['comparisons']['r30_c050→r39_c025'][NAMES[chosen]]['maximum']<=.02 and all(r['cases']['r30_c050']['stats']['converged']) and r['cases']['r39_c025']['stats']['converged'][chosen]:
        r['A2_pass']=True;r['minimal_setting']=dict(nr=30,chi_step=.5,candidate=NAMES[chosen])
    else:
        # Isolate chi before the optional 0.125-degree control.
        prev=(g39.h,g39.chi,s39.ms[:,:,chosen:chosen+1]);gq,sq=case('r30_c025',30,.25,(chosen,),prev);compare(r,'r30_c050','r30_c025');compare(r,'r30_c025','r39_c025')
        radial=r['comparisons']['r30_c025→r39_c025'][NAMES[chosen]]['maximum']
        g8,s8=case('r30_c0125',30,.125,(chosen,),(gq.h,gq.chi,sq.ms));compare(r,'r30_c025','r30_c0125')
        chi=r['comparisons']['r30_c025→r30_c0125'][NAMES[chosen]]['maximum'];r['A2_pass']=radial<=.02 and chi<=.02 and all(r['cases']['r30_c0125']['stats']['converged']) and all(r['cases']['r30_c025']['stats']['converged']) and r['cases']['r39_c025']['stats']['converged'][chosen]
        if r['A2_pass']:r['minimal_setting']=dict(nr=30,chi_step=.25,candidate=NAMES[chosen])
    r['frozen_preserved']=before==b.manifest();r['notes'].append('수렴 판정은 모든35칸 multiple. r30은 기존20점과 2–80km의 큰10구간 중점(2.5/3.75/5.5/7.75/11/15.5/22/33/50/70);39는 모든19개 중점. 0–3km 촘촘한 원래 비균일 격자를 유지.')
    save_progress(r,'A2 완료');b.save(ROOT/'results/sos_full/a2_result.json',r);print('VERDICT A2',r['A2_pass'],'RSS',r['peak_bytes']/1e9,flush=True)

if __name__=='__main__':
    try:run()
    finally:b.STOP.set()
