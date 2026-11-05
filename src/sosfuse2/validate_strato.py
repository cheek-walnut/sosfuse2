#!/usr/bin/env python3
"""Local D2 measurements, with strict thresholds and explicit unverified gates."""
import argparse,copy,hashlib,json,os,resource,sys,time
from pathlib import Path
import numpy as np
from sosfuse2 import reference
from sosfuse2.numerical_checks import within_tol
from sosfuse2.geometry import create,normal_state,TransientGeometry
from sosfuse2.fused import Solver
from sosfuse2.stratosphere import build_optics,density
# --baseline is required: the npz pair of the fixed measurement lives outside the package (2026-11-03 release review).


def zero_bytes(current,baseline):
    report={}
    for name in ('selftest_values.npz','full_35_values.npz'):
        a=current/name;b=baseline/name
        if not a.exists() or not b.exists():report[name]=dict(pass_=False,reason='E_BASELINE_MISSING');continue
        with np.load(a,allow_pickle=False) as az,np.load(b,allow_pickle=False) as bz:
            # A key on one side only (e.g. 'record', added 2026-10-22) is a mismatch, not a KeyError.
            arrays={k:k in az.files and k in bz.files and within_tol(az[k], bz[k], rtol=1e-5)
                    for k in sorted(set(az.files)|set(bz.files))}
        report[name]=dict(pass_=all(arrays.values()),arrays=arrays,
                          archive_sha256=hashlib.sha256(b.read_bytes()).hexdigest(),sha256=hashlib.sha256(a.read_bytes()).hexdigest(),baseline=str(b))
    return dict(pass_=all(v['pass_'] for v in report.values()),files=report)


def query_geometry(s,hs,el,az):
    p,up,ex=s.query.__func__.__globals__['b'].sh.old.probe.frame(np.array(6371.1),np.deg2rad(90-hs));ey=np.cross(up,ex)
    e,z=np.deg2rad([el,az]);d=np.sin(e)*up+np.cos(e)*(np.cos(z)*ex+np.sin(z)*ey)
    return s.geo.pack(p[None,:],d[None,:],s.geo.query_n)


def independent_first(g,optics,n=16001):
    """Independent uniform view trapezoid + GL128 sun integral; no sample()/columns()."""
    p=g['p'][0];d=g['d'][0];length=g['edges'][0,-1]
    t=np.linspace(0,length,n);q=p[None,:]+t[:,None]*d;h=np.linalg.norm(q,axis=1)-6371.
    prof=optics['profile'];den=np.exp(-.5*((h-prof['center'])/prof['sigma'])**2)/prof['norm']
    dt=t[1]-t[0];prefix=np.r_[0.,np.cumsum((den[:-1]+den[1:])*dt/2)]
    end=np.sqrt(np.maximum(0,(6451.)**2-q[:,0]**2-q[:,1]**2))-q[:,2]
    mg=-np.sqrt(np.maximum(0,1-(6371./np.linalg.norm(q,axis=1))**2));lit=q[:,2]/np.linalg.norm(q,axis=1)>=mg
    u,w=np.polynomial.legendre.leggauss(128);u=(u+1)/2;w=w/2;suncol=np.zeros(n)
    for start in range(0,n,1024):
        qq=np.broadcast_to(q[start:start+1024,None,:],(min(1024,n-start),len(u),3)).copy();qq[:,:,2]+=end[start:start+1024,None]*u
        sh=np.linalg.norm(qq,axis=-1)-6371.;sd=np.exp(-.5*((sh-prof['center'])/prof['sigma'])**2)/prof['norm']
        suncol[start:start+len(qq)]=end[start:start+len(qq)]*(sd@w)
    phase=np.polynomial.legendre.legval(d[2],optics['chi'])/(4*np.pi)
    value=np.trapezoid(den[:,None]*optics['sca']*phase*np.exp(-(prefix[:,None]+suncol[:,None])*optics['ext'])*lit[:,None],t,axis=0)
    return value


def thin_case(out):
    state=copy.deepcopy(normal_state());state.setdefault('stratosphere',{})['status']='enabled';state['parameters'].update(aod550=0.,stratosphere_tau550=1e-6,albedo=0.,ozone_du=0.)
    s,scope,_=create(out/'v2_geometry','selftest',[550.],state=state);s.ext[:]=0;s.asca[:]=0;s.solar[:]=1.
    solver=Solver(s,TransientGeometry(s,scope));g=query_geometry(s,2.,20.,0.);c=s.kernel.setup(s,g['nq'])
    field=np.zeros((s.geo.ns,1132,1),'f4');actual=reference.transport(g,c,field,True)[...,:3].sum(-1)[0].astype('f8')
    independent=independent_first(g,s.strato);tight=independent_first(g,s.strato,32001)
    err=float(np.max(abs(actual-independent)/np.maximum(abs(independent),1e-30)))
    return dict(actual=actual.tolist(),independent=independent.tolist(),max_relative=err,tolerance=.01,pass_=err<=.01,
                independent_grid_convergence=float(np.max(abs(tight-independent)/abs(tight))),
                tau550=1e-6,solar_elevation_deg=2.,view_elevation_deg=20.,azimuth_deg=0.,rayleigh=False,trop=False,ozone=False,ground=False),s,scope,g


def additivity(s,scope,g=None):
    # 96 distinct physical rays; same extinction, only component scattering
    # switched. Every output element has its own FP32 ULP/relative bound.
    old=scope['b'].sh.old
    p,up,ex=old.probe.frame(np.array(6371.1),np.deg2rad(88.));ey=np.cross(up,ex)
    el,az=np.meshgrid(np.linspace(2,89,8),np.linspace(0,180,12),indexing='ij')
    e,z=np.deg2rad(el.ravel()),np.deg2rad(az.ravel())
    dirs=np.sin(e)[:,None]*up+np.cos(e)[:,None]*(np.cos(z)[:,None]*ex+np.sin(z)[:,None]*ey)
    g=s.geo.pack(np.broadcast_to(p,dirs.shape),dirs,s.geo.query_n)
    c=copy.copy(s.kernel.setup(s,g['nq']));c['asca']=np.array([.00001],'f4');c['ext']=np.zeros((3,1),'f4')
    c['gm']=np.zeros_like(c['gm']);c['gval']=np.zeros_like(c['gval']);c['gv']=np.zeros_like(c['gv'])
    field=np.zeros((s.geo.ns,1132,1),'f4')
    both=reference.transport(g,c,field,True)
    ct=copy.copy(c);ct['strato_sca']=np.zeros(1,'f4');trop=reference.transport(g,ct,field,True)
    cs=copy.copy(c);cs['asca']=np.zeros(1,'f4');strat=reference.transport(g,cs,field,True)
    expected=(trop+strat).astype('f4');delta=abs(both.astype('f8')-expected.astype('f8'))
    spacing=np.maximum(abs(np.spacing(expected)).astype('f8'),np.nextafter(np.float32(0),np.float32(1)))
    relative=delta/np.maximum(abs(expected.astype('f8')),1e-30);ulp=delta/spacing
    per_element=(ulp<=1)|(relative<=1e-6)
    c2=copy.copy(cs);c2['strato_sca']=cs['strato_sca']*np.float32(2)
    doubled=reference.transport(g,c2,field,True)
    scale_exact=within_tol(doubled, strat*np.float32(2))
    return dict(rays=len(dirs),elements=int(delta.size),max_ulp=float(ulp.max()),max_relative=float(relative.max()),
                element_ulp_tolerance=1,element_relative_tolerance=1e-6,element_failures=int(np.count_nonzero(~per_element)),
                scattering_2x_within_tol=scale_exact,scattering_2x_nonzero_elements=int(np.count_nonzero(strat)),
                pass_=bool(per_element.all() and scale_exact and np.count_nonzero(strat)>0),
                attenuation='common fixed extinction; only component scattering sources switched',
                scope='production FP32 96-ray additivity and power-of-two scaling within tol; no FP64 identity gate')


def energy():
    st=copy.deepcopy(normal_state());st['stratosphere']['status']='enabled';st['parameters']['stratosphere_tau550']=.03
    opt=build_optics(st,np.linspace(0,80,8001),[550.]);z=np.linspace(0,80,8001)
    absorb=float(np.trapezoid((opt['beta_e']-opt['beta_s'])[:,0],z));mu,w=np.polynomial.legendre.leggauss(128)
    norm=float(w@np.polynomial.legendre.legval(mu,opt['chi'][:,0])/2)
    return dict(tau550=.03,column_absorption=absorb,scattering_source_angular_mean=norm,
                source_integral_pass_=absorb==0 and abs(norm-1)<1e-12,global_flux_balance='unverified',
                scope='zero material absorption and normalized scalar source; not whole-solver flux conservation')


def profile_grid(scope):
    h=scope['b'].sh.old.grids(1)[0];state=copy.deepcopy(normal_state())
    state['stratosphere']['status']='enabled';state['parameters']['stratosphere_tau550']=.003
    fine=np.linspace(0,80,160001);rows=[]
    for nodes in (h,np.sort(np.r_[h,(h[:-1]+h[1:])/2])):
        opt=build_optics(state,nodes,[550.]);q=density(fine,opt['profile']);avg=opt['source_density'];recon=np.interp(fine,nodes,avg)
        column=float(avg@np.diff(opt['cell_edges']))
        rows.append(dict(nr=len(nodes),column=column,linear_reconstruction_column=float(np.trapezoid(avg,nodes)),
                         node_before_column=float(np.trapezoid(opt['density'],nodes)),
                         shape_L1_relative=float(np.trapezoid(abs(recon-q),fine)/np.trapezoid(q,fine)),
                         at21_reconstructed=float(np.interp(21,nodes,avg)),at21_exact=float(density(21,opt['profile'])),
                         column_relative_error=abs(column-1),pass_=abs(column-1)<=1e-6))
    return dict(cases=rows,pass_=all(r['pass_'] for r in rows),shape_metric='integral abs(linear source density - analytic q) dz / integral q dz',
                decision='column conserved; grid refinement remains a later user decision')


def timing(out,repeats=3):
    rows=[]
    for tau in (0.,.003):
        st=copy.deepcopy(normal_state());st['stratosphere']['status']='enabled';st['parameters']['stratosphere_tau550']=tau
        s,scope,_=create(out/f'cost_{tau}','selftest',[550.],state=st);geo=TransientGeometry(s,scope);sv=Solver(s,geo)
        field=np.zeros((s.geo.ns,1132 if tau else 571,1),'f4');field=sv.apply(field)
        # Keep same nonzero input through measured complete applications.
        seconds=[]
        for i in range(repeats):
            t=time.perf_counter();value=sv.apply(field);seconds.append(time.perf_counter()-t)
        rows.append(dict(tau550=tau,warm_complete_apply_seconds=seconds,median_seconds=float(np.median(seconds)),source_shape=list(value.shape),finite=bool(np.isfinite(value).all())))
        print('V6',tau,seconds,flush=True)
    return dict(cases=rows,time_increase_fraction=rows[1]['median_seconds']/rows[0]['median_seconds']-1,
                scope='40 cells, 1 wave, original selftest quadrature, warm fixed nonzero source, first light/setup excluded')


def main():
    # CLI-only process settings (importing this module must not change the caller's bytecode or thread environment)
    sys.dont_write_bytecode=True
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[key]='4'
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True,help='directory holding selftest_values.npz and full_35_values.npz of the fixed measurement');p.add_argument('--skip-timing',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    report=dict(risk_class='D2',V1=zero_bytes(a.out/'val_zero',a.baseline))
    report['V2'],s,scope,g=thin_case(a.out);print('V2',json.dumps(report['V2']),flush=True)
    report['V3']=energy();report['V4']=additivity(s,scope,g);print('V4',json.dumps(report['V4']),flush=True)
    report['profile_grid']=profile_grid(scope)
    if not a.skip_timing:report['V6']=timing(a.out)
    with open(Path(__file__).with_name('assets')/'strato_sulfate_rg0.08_sg1.6_n1.43.json') as f:report['Mie']=json.load(f)['checks']
    report['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['local_numeric_pass_']=report['V1']['pass_'] and report['V2']['pass_'] and report['V4']['pass_'] and report['profile_grid']['pass_'] and all(v['pass_'] for v in report['Mie'].values())
    (a.out/'strato_validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('STRATO numeric gates',report['local_numeric_pass_'],flush=True)
    return report
if __name__=='__main__':raise SystemExit(0 if main()['local_numeric_pass_'] else 1)
