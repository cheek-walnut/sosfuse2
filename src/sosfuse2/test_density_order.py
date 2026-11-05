#!/usr/bin/env python3
"""Actual PROFILE C++ helper/sample path vs operation-matched NumPy; lane parity."""
import argparse,copy,ctypes,itertools,json,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sosfuse2 import reference, testing
from sosfuse2.density import interpolate32
from sosfuse2.geometry import create,TransientGeometry,normal_state
from sosfuse2.fused import sh_aux
from sosfuse2.strato_states import build
from sosfuse2.test_kernel_static import STUB
try:
    from .numerical_checks import assert_equivalent, within_tol
except ImportError:  # standalone research harness
    from numerical_checks import assert_equivalent, within_tol


def run(out):
    out.mkdir(parents=True,exist_ok=True);kernel=Path(__file__).with_name('kernel.cu').resolve()
    testing.require_tool('clang++')
    stub=out/'stub.h';stub.write_text(STUB)
    static=[]
    for W,strato,flags in itertools.product((1,15),(False,True),((),('SH_AUX','LANE_GROUND','SPECIALIZE','LM_LUT','PHASE2_NOBARRIER'))):
        opts=[f'-DFW={W}','-DPROFILE']+(['-DSTRATO'] if strato else [])+[f'-D{f}' for f in flags]
        if 'SPECIALIZE' in flags:opts+=['-DK_FIRST=1','-DK_PHASE=2','-DK_BOUNDARY=1']
        p=subprocess.run(['clang++','-std=c++17','-ffp-contract=off','-fsyntax-only','-x','c++','-include',str(stub),*opts,str(kernel)],capture_output=True,text=True)
        static.append(dict(W=W,strato=strato,flags=flags,pass_=p.returncode==0,stderr=p.stderr))
    cpp=out/'profile_host.cpp';lib=out/'profile_host.dylib'
    cpp.write_text(STUB+'\nDim blockIdx{},threadIdx{},blockDim{};extern "C" { double shared[50000]; }\n'+f'#include "{kernel}"\n'+'''
extern "C" float profile_c(float h,const float*t,int k){return profile_value(h,t,k);}
extern "C" void sample_c(const double*p,const double*d,const float*B,const float*dl,
 double a,double b,int lower,double gu,double gw,const float*h,int nh,const float*chi,int nc,float height,
 const float*sun_h,int nsh,const float*sun_col,int nsx,const float*table,int mask,float*out){
 Sample s{};point(s,p,d,B,dl,0.f,a,b,lower,gu,gw,h,nh,chi,nc,height,sun_h,nsh,sun_col,nsx,false,table,mask);
 out[0]=s.h;for(int k=0;k<3;k++)out[k+1]=s.prof[k];}
''')
    subprocess.run(['clang++','-std=c++17','-O2','-ffp-contract=off','-DFW=15','-DPROFILE','-dynamiclib',str(cpp),'-o',str(lib)],check=True)
    native=ctypes.CDLL(str(lib));P=ctypes.c_void_p;I=ctypes.c_int;D=ctypes.c_double;F=ctypes.c_float
    native.profile_c.argtypes=[F,P,I];native.profile_c.restype=F
    native.sample_c.argtypes=[P,P,P,P,D,D,I,D,D,P,I,P,I,F,P,I,P,I,P,I,P]
    rng=np.random.default_rng(20261002);samples=[];cases=[]
    for name,mask,strato in (('ray',1,False),('ozone',2,False),('both',3,False),('both_strato',3,True)):
        state=copy.deepcopy(build()['states'][5] if strato else normal_state());state['density']=dict(density_profile='usstd76' if mask&1 else 'exp8',ozone_profile='usstd76' if mask&2 else 'triangle')
        s,scope,_=create(out/name/'geometry','selftest',360+(np.arange(15)+.5)*470/15,state=state)
        geometry=TransientGeometry(s,scope);g=geometry.cell(12);c=s.kernel.setup(s,g['nq']);table=np.ascontiguousarray(c['density_optics']['table'],'f4')
        h=np.r_[np.arange(161)*.5,rng.uniform(0,80,1000),[-.5,80.5]].astype('f4');actual=interpolate32(h,table)
        expected=np.array([[native.profile_c(float(v),table.ctypes.data,k) for k in (0,1)] for v in h],'f4')
        helper=(within_tol(actual, expected));matched=0;maxerr=0.;maxh=0.
        for seg in range(g['edges'].shape[1]-1):
            for step in range(len(c['u'])):
                q=reference.sample(g,seg,step,c)
                for row in range(len(g['d'])):
                    if not q['valid'][row]:continue
                    arrays=[np.ascontiguousarray(g[key][row],dtype) for key,dtype in (('p','f8'),('d','f8'),('target','f4'),('dloc','f4'))]
                    outp=np.empty(4,'f4')
                    native.sample_c(*[v.ctypes.data for v in arrays],g['edges'][row,seg],g['edges'][row,seg+1],int(g['lower'][row,seg]),c['u'][step],c['wu'][step],c['h'].ctypes.data,len(c['h']),c['chi'].ctypes.data,len(c['chi']),float(c['height']),c['sun_h'].ctypes.data,len(c['sun_h']),c['sun_col'].ctypes.data,c['sun_col'].shape[1],table.ctypes.data,mask,outp.ctypes.data)
                    maxh=max(maxh,float(abs(outp[0]-q['h'][row])))
                    for col,bit in ((0,1),(2,2)):
                        if mask&bit:maxerr=max(maxerr,float(abs(outp[col+1]-q['prof'][row,col])));matched+=1
        samples.append(dict(name=name,helper_within_tol=helper,profile_sample_max_absolute=maxerr,height_sample_max_absolute=maxh,selected_samples=matched,pass_=helper and maxerr<=1e-12))
        field=rng.normal(0,1e-5,(s.geo.ns,1132 if strato else 571,15)).astype('f4');field[:,:3]=abs(field[:,:3]);boundary=rng.random((len(s.geo.chi),15))*1e-4
        for first,bd,aux in ((True,None,False),(False,None,False),(False,boundary,False),(False,boundary,True)):
            full,Y=reference.transport(g,c,field,first,boundary=bd,return_basis=True,sh_source=sh_aux(field) if aux else None);m,f=reference.project(g,full,c,Y)
            rays=[];moments=[];flux=[]
            for w in range(15):
                one=dict(c,W=1)
                for key in ('ext','asca','gm','gv','gval','strato_ext','strato_sca','strato_chi'):
                    if key in c:one[key]=np.ascontiguousarray(c[key][...,w:w+1])
                src=np.ascontiguousarray(field[...,w:w+1]);b=None if bd is None else bd[:,w:w+1]
                value,y=reference.transport(g,one,src,first,boundary=b,return_basis=True,sh_source=sh_aux(src) if aux else None);mm,ff=reference.project(g,value,one,y)
                rays.append(value);moments.append(mm);flux.append(ff)
            rp=np.concatenate(rays,axis=1);mp=np.concatenate(moments,axis=1);fp=np.concatenate(flux)
            row=dict(name=name,first=first,boundary=bd is not None,SH_AUX=aux,ray_within_tol=(within_tol(full, rp)),moment_max_absolute=float(np.max(abs(m-mp))),ground_within_tol=(within_tol(f, fp)))
            row['pass_']=row['ray_within_tol'] and row['moment_max_absolute']<=1e-12 and row['ground_within_tol'];cases.append(row)
    report=dict(static=static,samples=samples,cases=cases,pass_=all(r['pass_'] for r in [*static,*samples,*cases]),scope='clang CUDA-stub host helper+point profile; per-wave NumPy transport, not CUDA execution')
    (out/'density_order.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();raise SystemExit(0 if run(a.out)['pass_'] else 1)
