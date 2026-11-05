#!/usr/bin/env python3
"""Fast folded radiance table from a stored fused source (source.npz + state -> table.npz).

Same rays, same endpoint packs, same CUDA kernel (phase 2), same host
post-processing as `s.query` (results/sos_full/b2_driver.py Solver.query):

* capture: `s.query` is run with a recording transport, so directions and the
  64-ray endpoint packs are produced by the unchanged accepted code;
* launch: all captured rays of many solar elevations go into ONE multi-block
  Bundle per `first` flag (the original launched one CUDA block per 64 rays,
  i.e. one SM of the GPU, and re-packed/uploaded the whole source per launch);
* replay: `s.query` is run again with a transport that returns the computed
  per-ray values, so reshape, [...,:3].sum(-1) and the solar factor are the
  accepted operations.

Per-ray kernel arithmetic does not depend on the block partition (phase 2 has
no cross-warp reduction; kernel.cu:117-239), so no approximation is introduced.
"""
import argparse, json, os, sys, time
from pathlib import Path
sys.dont_write_bytecode=True
from sosfuse2 import paths
import numpy as np

UNITS='W m^-2 sr^-1 nm^-1'


def axes(reduced=False):
    wl=np.array([550.]) if reduced else 360+(np.arange(15)+.5)*470/15
    hs=np.array([-3.,-6.]) if reduced else np.arange(6.,-19.,-1.)
    nr,nc=(4,6) if reduced else (100,180)
    el=90-(np.arange(nr)+.5)*90/nr
    az=(np.arange(nc)+.5)*180/nc
    return wl,hs,el,az


def _capture(s,h,ee,aa,dummy):
    """Run the accepted query once with a recording transport; returns raw packs."""
    raws=[];kernel=s.kernel
    def record(raw,solver,field,first=False):
        raws.append(raw);return np.zeros((len(raw['d']),solver.W,2,4),'f8')
    kernel.transport=record
    try:s.query(float(h),ee,aa,field=dummy,first=True)
    finally:del kernel.transport
    return raws


def _replay(s,h,ee,aa,dummy,raws,values,first):
    """Run the accepted query again, returning precomputed per-chunk values."""
    kernel=s.kernel;geo=s.geo;index=iter(range(len(raws)));pack=geo.pack
    def token(p,d,n):
        i=next(index)
        if len(d)!=len(raws[i]['d']) or not np.array_equal(d,raws[i]['d']):raise RuntimeError('E_REPLAY_DRIFT: ray order changed')
        return i
    def fetch(i,solver,field,first_flag=False):
        if bool(first_flag)!=bool(first):raise RuntimeError('E_REPLAY_FLAG')
        value=values[i]
        out=np.zeros((len(value),solver.W,2,4),'f8');out[:,:,0]=value
        return out
    geo.pack=token;kernel.transport=fetch
    try:result=s.query(float(h),ee,aa,field=dummy,first=first)
    finally:
        geo.pack=pack;del kernel.transport
    if next(index,None) is not None:raise RuntimeError('E_REPLAY_COUNT')
    return result


def _groups(raws,group):
    """Block partition: consecutive `group`-ray slices of every captured pack."""
    from sosfuse2.geometry import subset
    groups=[];sizes=[]
    for g in raws:
        n=len(g['d']);sizes.append(n)
        for k in range(0,n,group):groups.append(subset(g,slice(k,min(k+group,n))))
    return groups,sizes


def _bundle(s,raws,group):
    """A5: one Bundle per captured batch, shared by the first and multiple launches
    (the ray set/order is identical; only the source field and FIRST differ)."""
    from sosfuse2.geometry import Bundle
    from sosfuse2.timers import stage
    if s.kernel.xp is np:return None
    with stage('table_bundle'):
        groups,sizes=_groups(raws,group);bundle=Bundle(groups,s.kernel.xp);bundle.sizes=sizes
    return bundle


def _evaluate(s,raws,field,first,group,bundle=None):
    """Per-chunk phase-2 values (n,W,4) float32 for every captured raw pack."""
    from sosfuse2 import reference
    from sosfuse2.timers import stage
    kernel=s.kernel;xp=kernel.xp
    if xp is np:
        # NumPy has no block partition: `group` cannot change these values.
        packed=np.ascontiguousarray(field,'f4')
        with stage('table_launch'):
            return [reference.transport(g,kernel.setup(s,g['nq']),packed,first,statistics=kernel.metrics) for g in raws]
    if bundle is None:bundle=_bundle(s,raws,group)
    sizes=bundle.sizes
    with stage('table_launch',device=True):
        _,_,query=kernel.launch(bundle,s,field,first,2)
    with stage('table_d2h',device=True):
        value=kernel.host(query)
    if value.shape!=(sum(sizes),s.W,4):raise RuntimeError('E_QUERY_SHAPE')
    return np.split(value,np.cumsum(sizes)[:-1])


def make_table(s,source,hs,el,az,group=32,batch=25,log=print):
    """Return (table[len(hs),nr,nc,W] float32, per-batch seconds)."""
    from sosfuse2.timers import stage
    kernel=s.kernel;xp=kernel.xp
    source=np.ascontiguousarray(source,'f4')
    if source.shape!=(s.geo.ns,10+s.kernel.K*(2 if getattr(s,"strato",None) is not None else 1),s.W):raise ValueError(f'E_SOURCE_SHAPE: {source.shape}')
    field=source if xp is np else xp.asarray(source)
    dummy=np.empty(0,'f4')
    ee,aa=np.meshgrid(el,az,indexing='ij')
    table=np.empty((len(hs),len(el),len(az),s.W),'f4');seconds=[]
    for b0 in range(0,len(hs),batch):
        t=time.monotonic();part=list(range(b0,min(b0+batch,len(hs))))
        with stage('table_capture'):
            captured=[_capture(s,hs[j],ee,aa,dummy) for j in part]
        flat=[g for raws in captured for g in raws]
        results={};bundle=_bundle(s,flat,group)
        for first in (True,False):
            results[first]=_evaluate(s,flat,field,first,group,bundle)
        kernel.sync();del bundle
        offset=0
        for j,raws in zip(part,captured):
          with stage('table_replay'):
            one=_replay(s,hs[j],ee,aa,dummy,raws,results[True][offset:offset+len(raws)],True)
            multiple=_replay(s,hs[j],ee,aa,dummy,raws,results[False][offset:offset+len(raws)],False)
            offset+=len(raws)
            spec=(one+multiple)[...,0].astype('f4')
            if not np.isfinite(spec).all() or (spec<0).any():raise RuntimeError('E_RADIANCE_INVALID: nonfinite/negative')
            table[j]=spec
        seconds.append(time.monotonic()-t)
        log(f'TABLE_FAST sun={hs[part[0]]:+g}..{hs[part[-1]]:+g} rays={sum(len(g["d"]) for g in flat)} seconds={seconds[-1]:.2f}')
    return table,seconds


def save_table(path,table,wl,hs,el,az):
    from sosfuse2.timers import stage
    with stage('table_save'):
        _save_table(path,table,wl,hs,el,az)


def _save_table(path,table,wl,hs,el,az):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,radiance=table,wavelength_nm=wl.astype('f4'),solar_elevation_deg=hs.astype('f4'),
            view_elevation_deg=el.astype('f4'),relative_azimuth_deg=az.astype('f4'),units=np.array(UNITS))
    os.replace(tmp,path)


def load_source(path,s,wl):
    with np.load(path) as f:data={k:f[k] for k in f.files}
    if not np.allclose(data['wavelength_nm'],wl,rtol=0,atol=1e-9):raise ValueError('E_SOURCE_WAVELENGTH')
    if not (np.array_equal(data['h'],s.geo.h) and np.array_equal(data['chi'],s.geo.chi)):raise ValueError('E_SOURCE_GRID')
    return data['source']


def build_solver(state,prepare,reduced,backend):
    """Same constructor call as run_gpu_state_v3 (optics, ground pool, sun columns)."""
    from sosfuse2 import geometry as geometry_module
    if backend=='numpy':
        class NumpyScalarKernel(geometry_module.Kernel):
            def host(self,array):return np.asarray(array)
        geometry_module.Kernel=NumpyScalarKernel
    wl,_,_,_=axes(reduced)
    return geometry_module.create(prepare,'selftest' if reduced else 'full',wl,backend,128,state)


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--id',required=True)
    p.add_argument('--states',type=Path,default=paths.ASSETS/'states_v1.json')
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--prepare',type=Path,required=True,help='solver scratch folder (sun columns cache)')
    p.add_argument('--out',type=Path,required=True,help='table.npz path')
    p.add_argument('--backend',choices=['cupy','numpy'],default='cupy')
    p.add_argument('--reduced',action='store_true')
    p.add_argument('--group',type=int,default=32,help='rays per CUDA block (A5: 4 -> 32; 64 was also bit-identical on NumPy, CUDA timing is a rig TODO)')
    p.add_argument('--batch',type=int,default=25)
    a=p.parse_args()
    state=next((s for s in json.loads(a.states.read_text())['states'] if s['id']==a.id),None)
    if state is None:p.error(f'E_STATE_ID: no state {a.id!r} in {a.states}')
    t=time.monotonic();s,scope,settings=build_solver(state,a.prepare,a.reduced,a.backend);create_seconds=time.monotonic()-t
    wl,hs,el,az=axes(a.reduced)
    source=load_source(a.source,s,wl)
    t=time.monotonic();table,seconds=make_table(s,source,hs,el,az,a.group,a.batch,lambda m:print(m,flush=True))
    total=time.monotonic()-t
    save_table(a.out,table,wl,hs,el,az)
    print(f'TABLE_FAST_DONE id={a.id} create_seconds={create_seconds:.2f} table_seconds={total:.2f} shape={list(table.shape)}',flush=True)

if __name__=='__main__':
    main()
