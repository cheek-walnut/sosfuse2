#!/usr/bin/env python3
"""D0 spherical SOS with direct HG angular quadrature, ozone and Lambert ground.
All writes, including JIT and Monte Carlo references, are confined to sos_sky.
Geometry is wavelength independent; atmospheric scattering orders count volume
events only, matching engine.trace. Ground reflections stay in the same order.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
             'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import sos_probe as probe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/sos_sky'
R, Z, OBS = 6371., 80., .1
WL = 360 + (np.arange(15)+.5)*470/15
SPEC = json.loads((ROOT/'assets/spectra.json').read_text())
HS = (2, 0, -3, -6, -9, -12)
DIRS = probe.DIRS
NORMAL = dict(aer=.125, ssa=.92, ang=1., g=.8, albedo=.30, ozone_DU=300.)
os.environ['DRJIT_CACHE_DIR'] = str(OUT/'jit-cache')


def dump(path, value):
    path.write_text(json.dumps(value, indent=1, allow_nan=False)+'\n')


def coefficients(wavelengths, p):
    w = np.asarray(wavelengths)
    return np.stack([.013558*(550/w)**4, p['aer']*(550/w)**p['ang'],
                     np.interp(w,SPEC['wavelength_nm'],SPEC['ozone'])*p['ozone_DU']*2.687e20/15])


def profiles(h):
    return np.stack([np.exp(-np.maximum(h,0)/8), np.exp(-np.maximum(h,0)/1.2),
                     np.maximum(0,1-np.abs(h-25)/15)], -1)


def hg(nu, g=.8):
    return (1-g*g)/(4*np.pi*np.maximum(1+g*g-2*g*nu,1e-20)**1.5)


def rayleigh(nu):
    return 3*(1+nu*nu)/(16*np.pi)


def grids(scale):
    h = np.array([0,.05,.1,.2,.35,.55,.8,1.1,1.5,2,3,4.5,6.5,9,13,18,26,40,60,80.])
    chi = np.unique(np.r_[np.arange(0,81,10),np.arange(80,85,1),
                           np.arange(85,110,.5),np.arange(110,120,2),np.arange(120,181,10)])
    if scale == 2:
        h = np.sort(np.r_[h,(h[:-1]+h[1:])/2])
        chi = np.sort(np.r_[chi,(chi[:-1]+chi[1:])/2])
    return h, np.deg2rad(chi)


def angles(r, nmu, nphi):
    u,w=probe.gl(nmu)
    mg=-np.sqrt(max(0,1-(R/r)**2))
    mu=np.r_[-1.,mg-(1+mg)*u*u,mg*(1-u),u*u,1.]
    wm=np.r_[0.,2*(1+mg)*u*w,-mg*w,2*u*w,0.]
    order=np.argsort(mu,kind='stable');mu,wm=mu[order],wm[order]
    phi=np.linspace(0,np.pi,nphi+1)
    wp=np.full(nphi+1,2*np.pi/nphi);wp[[0,-1]]*=.5
    MU,PHI=np.meshgrid(mu,phi,indexing='ij')
    st=np.sqrt(np.maximum(1-MU*MU,0))
    dirs=np.stack([st*np.cos(PHI),st*np.sin(PHI),MU],-1).reshape(-1,3)
    weights=(wm[:,None]*wp[None,:]).ravel()
    return mu,dirs,weights


def geometry_segments(p,d,n):
    edges=probe.boundaries(p,d)
    end=edges[:,-1]
    b=np.sum(p*d,-1);pp=np.sum(p*p,-1)
    more=[]
    for height in (10.,25.,40.):
        disc=b*b-pp+(R+height)**2
        for sign in (-1,1):
            root=-b+sign*np.sqrt(np.maximum(disc,0))
            more.append(np.where((disc>0)&(root>0)&(root<end),root,end))
    edges=np.sort(np.column_stack([edges,*more]),axis=-1)
    u,wu=probe.gl(n)
    tau=np.zeros((len(d),3))
    for k in range(edges.shape[1]-1):
        active=np.flatnonzero(edges[:,k+1]-edges[:,k]>1e-8)
        if not len(active):continue
        a,b=edges[active,k,None],edges[active,k+1,None]
        pa,da=p[active],d[active]
        ra=np.sum((pa+a*da)**2,-1);rb=np.sum((pa+b*da)**2,-1)
        lower=(ra<=rb)[:,None]
        t=np.where(lower,a+(b-a)*u*u,b-(b-a)*(1-u)**2)
        ds=np.where(lower,2*(b-a)*u*wu,2*(b-a)*(1-u)*wu)
        q=pa[:,None,:]+t[...,None]*da[:,None,:]
        r=np.sqrt(np.sum(q*q,-1));h=r-R
        dt=profiles(h)*ds[...,None]
        prefix=tau[active,None,:]+np.cumsum(dt,axis=1)-dt/2
        tau[active]+=dt.sum(axis=1)
        yield active,q,r,ds,prefix


def total_columns(p,d,n):
    out=np.zeros((len(d),3))
    for active,q,r,ds,prefix in geometry_segments(p,d,n):
        out[active]+=np.sum(profiles(r-R)*ds[...,None],axis=1)
    return out


class SunGeometry:
    def __init__(self,scale,folder):
        path=folder/'sun_columns.npz';t=time.perf_counter()
        if path.exists():
            z=np.load(path);self.h,self.x,self.col=z['h'],z['x'],z['col'];z.close()
            self.build_seconds=0.;return
        self.h=np.linspace(0,np.sqrt(Z),80*scale+1)**2
        self.x=np.linspace(0,1,400*scale+1)
        col=np.empty((len(self.h),len(self.x),3))
        for i,h in enumerate(self.h):
            r=R+h;mg=-np.sqrt(max(0,1-(R/r)**2))
            mu=mg+(1-mg)*self.x**2;mu[0]=mg+1e-9
            p=np.column_stack([np.zeros(len(mu)),np.zeros(len(mu)),np.full(len(mu),r)])
            d=np.column_stack([np.sqrt(np.maximum(1-mu*mu,0)),np.zeros(len(mu)),mu])
            col[i]=total_columns(p,d,32*scale)
        self.col=col;np.savez(path,h=self.h,x=self.x,col=col)
        self.build_seconds=time.perf_counter()-t

    def lookup(self,q,r):
        h=np.clip(r-R,0,Z);mu=q[...,2]/r
        mg=-np.sqrt(np.maximum(0,1-(R/np.maximum(r,R))**2))
        x=np.sqrt(np.clip((mu-mg)/(1-mg),0,1))
        i=np.clip(np.searchsorted(self.h,h,side='right')-1,0,len(self.h)-2)
        a=(h-self.h[i])/(self.h[i+1]-self.h[i])
        f=x*(len(self.x)-1);j=np.clip(f.astype(int),0,len(self.x)-2);b=f-j
        col=((1-a)[...,None]*(1-b)[...,None]*self.col[i,j]+a[...,None]*(1-b)[...,None]*self.col[i+1,j]+
             (1-a)[...,None]*b[...,None]*self.col[i,j+1]+a[...,None]*b[...,None]*self.col[i+1,j+1])
        return col,mu>=mg


class Geometry:
    def __init__(self,scale):
        self.scale=scale;self.h,self.chi=grids(scale)
        self.nmu,self.nphi,self.npath=4*scale,8*scale,12*scale
        self.nm=3*self.nmu+2;self.np=self.nphi+1;self.nd=self.nm*self.np
        self.ns=len(self.h)*len(self.chi)
        # Explicit version covers only geometry; spectral/weather changes reuse it.
        self.folder=OUT/f'geometry_v1_x{scale}';self.folder.mkdir(parents=True,exist_ok=True)
        self.sun=SunGeometry(scale,self.folder)
        self.mu=np.array([angles(R+h,self.nmu,self.nphi)[0] for h in self.h])
        self.local=np.array([angles(R+h,self.nmu,self.nphi)[1] for h in self.h])
        self.weights=np.array([angles(R+h,self.nmu,self.nphi)[2] for h in self.h])
        self.meta_path=self.folder/'meta.json'
        self.cache_hit=self.meta_path.exists()
        if not self.cache_hit:self.build()
        self.meta=json.loads(self.meta_path.read_text())
        self.compact()
        z=np.load(self.folder/'boundary.npz')
        self.ground,self.endchi,self.column=z['ground'],z['chi'],z['column'];z.close()

    def compact(self):
        """Store the five multilinear coordinates instead of 16 repeated pairs.
        Characterize every converted block against its original indices/weights.
        A block replacement is atomic; interrupted mixed caches remain readable.
        """
        if self.meta.get('compact_schema')==1:return
        t=time.perf_counter();maximum=0.;last=t
        for block in self.meta['blocks']:
            path=self.folder/block['file'];z=np.load(path)
            if 'idx' not in z.files:z.close();continue
            idx,iw=z['idx'],z['iw']
            ah=iw[:,8:].sum(-1);ac=iw[:,[4,5,6,7,12,13,14,15]].sum(-1);ap=iw[:,1::2].sum(-1)
            am0=np.divide(iw[:,[2,3,6,7]].sum(-1),1-ah,out=np.zeros(len(iw),np.float32),where=ah<1)
            am1=np.divide(iw[:,[10,11,14,15]].sum(-1),ah,out=np.zeros(len(iw),np.float32),where=ah>0)
            data={key:z[key] for key in z.files if key not in ('idx','iw')}
            data['base']=idx[:,0]
            difference=idx[:,8]-idx[:,0]-len(self.chi)*self.nd
            assert np.all(difference%self.np==0)
            data['dm']=(difference//self.np).astype(np.int8)
            data['coords']=np.clip(np.column_stack([ah,ac,am0,am1,ap]),0,1).astype(np.float32)
            checkidx,checkw=mapping(data,self)
            assert np.array_equal(idx,checkidx)
            error=float(np.max(np.abs(iw-checkw)));maximum=max(maximum,error)
            assert error<1e-6
            z.close();tmp=path.with_name(path.name+'.tmp.npz');np.savez(tmp,**data);os.replace(tmp,path)
            if time.perf_counter()-last>30:
                print(f'COMPACT x{self.scale} {block["start"]}/{self.ns} {time.perf_counter()-t:.2f}s',flush=True);last=time.perf_counter()
        self.meta.update(compact_schema=1,compact_seconds=time.perf_counter()-t,
                         compact_maximum_weight_difference=maximum,
                         bytes=sum(p.stat().st_size for p in self.folder.glob('*.npz')))
        self.meta['setup_seconds']=self.meta['build_seconds']+self.meta['sun_build_seconds']+self.meta['compact_seconds']
        dump(self.meta_path,self.meta)

    def spatial_rays(self,row):
        ih,ic=divmod(row,len(self.chi))
        p,up,ex=probe.frame(np.array(R+self.h[ih]),np.array(self.chi[ic]))
        local=self.local[ih];ey=np.cross(up,ex)
        d=local[:,0,None]*ex+local[:,1,None]*ey+local[:,2,None]*up
        return np.broadcast_to(p,d.shape),d

    def lookup(self,q,r,d):
        h=np.clip(r-R,0,Z);angle=np.arccos(np.clip(q[...,2]/r,-1,1))
        ih=np.clip(np.searchsorted(self.h,h,side='right')-1,0,len(self.h)-2)
        ic=np.clip(np.searchsorted(self.chi,angle,side='right')-1,0,len(self.chi)-2)
        ah=(h-self.h[ih])/(self.h[ih+1]-self.h[ih])
        ac=np.clip((angle-self.chi[ic])/(self.chi[ic+1]-self.chi[ic]),0,1)
        dd=np.broadcast_to(d,q.shape)
        mu=np.clip(np.sum(q*dd,-1)/r,-1,1);cs=np.clip(q[...,2]/r,-1,1)
        denom=np.sqrt(np.maximum((1-cs*cs)*(1-mu*mu),0))
        cp=np.divide(dd[...,2]-cs*mu,denom,out=np.ones_like(mu),where=denom>1e-12)
        phi=np.arccos(np.clip(cp,-1,1))*self.nphi/np.pi
        ip=np.clip(phi.astype(int),0,self.nphi-1);ap=np.clip(phi-ip,0,1)
        ids=[];ws=[]
        for dh in (0,1):
            hh=ih+dh;mh=self.mu[hh]
            im=np.sum(mu[...,None]>=mh,axis=-1)-1;im=np.clip(im,0,self.nm-2)
            m0=np.take_along_axis(mh,im[...,None],axis=-1)[...,0]
            m1=np.take_along_axis(mh,(im+1)[...,None],axis=-1)[...,0]
            am=np.clip(np.divide(mu-m0,m1-m0,out=np.zeros_like(mu),where=m1>m0),0,1)
            for dc in (0,1):
                base=((hh*len(self.chi)+ic+dc)*self.nm+im)*self.np+ip
                sw=(ah if dh else 1-ah)*(ac if dc else 1-ac)
                for dm in (0,1):
                    for dp in (0,1):
                        ids.append(base+dm*self.np+dp)
                        ws.append(sw*(am if dm else 1-am)*(ap if dp else 1-ap))
        return np.stack(ids,-1).astype(np.int32),np.stack(ws,-1).astype(np.float32)

    def build(self):
        t0=time.perf_counter();last=t0;blocks=[];npoints=0
        ground=np.zeros((self.ns,self.nd),bool);endchi=np.zeros((self.ns,self.nd))
        column=np.zeros((self.ns,self.nd,3))
        buffer={key:[] for key in ('row','idx','iw','ds','prefix','suncol','lit','height','nu')}
        start=0
        for row in range(self.ns):
            p,d=self.spatial_rays(row);end=probe.boundaries(p,d)[:,-1]
            qe=p+end[:,None]*d;re=np.sqrt(np.sum(qe*qe,-1))
            ground[row]=re<R+1e-5;endchi[row]=np.arccos(np.clip(qe[:,2]/re,-1,1))
            for active,q,r,ds,prefix in geometry_segments(p,d,self.npath):
                idx,iw=self.lookup(q,r,d[active,None,:])
                scol,lit=self.sun.lookup(q,r)
                count=ds.shape[1]
                buffer['row'].append(np.repeat((row-start)*self.nd+active,count).astype(np.int32))
                buffer['idx'].append(idx.reshape(-1,16));buffer['iw'].append(iw.reshape(-1,16))
                for key,val in [('ds',ds),('prefix',prefix),('suncol',scol),('lit',lit),
                                ('height',r-R),('nu',np.broadcast_to(d[active,2,None],ds.shape))]:
                    buffer[key].append(val.reshape((-1,3) if key in ('prefix','suncol') else -1).astype(bool if key=='lit' else np.float32))
                column[row,active]+=np.sum(profiles(r-R)*ds[...,None],axis=1)
            if (row+1)%8==0 or row+1==self.ns:
                name=f'block_{len(blocks):04d}.npz'
                packed={key:np.concatenate(values) for key,values in buffer.items()}
                np.savez(self.folder/name,**packed)
                npoints+=len(packed['row']);blocks.append(dict(file=name,start=start,count=row+1-start))
                buffer={key:[] for key in buffer};start=row+1
            if time.perf_counter()-last>30 or row+1==self.ns:
                print(f'GEOMETRY x{self.scale} {row+1}/{self.ns} {time.perf_counter()-t0:.2f}s',flush=True);last=time.perf_counter()
        np.savez(self.folder/'boundary.npz',ground=ground,chi=endchi,column=column.astype(np.float32))
        dump(self.meta_path,dict(scale=self.scale,nr=len(self.h),nchi=len(self.chi),nmu=self.nm,nphi=self.np,
                                directions=self.nd,npath_per_segment=self.npath,points=npoints,
                                build_seconds=time.perf_counter()-t0,sun_build_seconds=self.sun.build_seconds,
                                bytes=sum(p.stat().st_size for p in self.folder.glob('*.npz')),blocks=blocks,
                                h_km=self.h.tolist(),chi_deg=np.rad2deg(self.chi).tolist()))


def interpolate(field,idx,iw):
    result=np.zeros((len(idx),field.shape[-1]))
    # Avoid allocating (all paths,16,15); each corner is one bounded gather.
    for corner in range(16):result+=field[idx[:,corner]]*iw[:,corner,None]
    return result


def mapping(z,g):
    if 'idx' in z:return z['idx'],z['iw']
    base=z['base'];delta=z['dm'].astype(np.int32)*g.np
    ah,ac,am0,am1,ap=z['coords'].T
    ids=[];weights=[]
    for dh in (0,1):
        am=am1 if dh else am0
        for dc in (0,1):
            corner=base+dh*(len(g.chi)*g.nd)+dh*delta+dc*g.nd
            spatial=(ah if dh else 1-ah)*(ac if dc else 1-ac)
            for dm in (0,1):
                for dp in (0,1):
                    ids.append(corner+dm*g.np+dp)
                    weights.append(spatial*(am if dm else 1-am)*(ap if dp else 1-ap))
    return np.stack(ids,-1),np.stack(weights,-1)


class SkySolver:
    def __init__(self,geometry,wavelengths,p=NORMAL,name='normal'):
        self.geo=geometry;self.wl=np.asarray(wavelengths,float);self.p=p.copy();self.name=name
        self.ext=coefficients(self.wl,p);self.nw=len(self.wl)
        self.solar=np.interp(self.wl,SPEC['wavelength_nm'],SPEC['solar'])
        self.folder=OUT/f'coeff_x{geometry.scale}_{name}';self.folder.mkdir(parents=True,exist_ok=True)
        self.state_path=self.folder/'state_v2.npz';self.stats_path=self.folder/'stats_v2.json'
        self.setup_phase();self.prepare()

    def setup_phase(self):
        t=time.perf_counter();g=self.geo;self.pr=[];self.pa=[];self.raw_pr=[];self.raw_pa=[];norms=[]
        for ih in range(len(g.h)):
            d=g.local[ih];mirror=d.copy();mirror[:,1]*=-1
            nu=np.clip(d@d.T,-1,1);num=np.clip(d@mirror.T,-1,1)
            wr=g.weights[ih][None,:]
            pr=.5*(rayleigh(nu)+rayleigh(num))*wr
            pa=.5*(hg(nu,self.p['g'])+hg(num,self.p['g']))*wr
            norms.extend(pa.sum(-1).tolist())
            self.raw_pr.append(pr.copy());self.raw_pa.append(pa.copy())
            # Preserve constant radiance exactly despite finite forward quadrature.
            pr/=pr.sum(-1)[:,None];pa/=pa.sum(-1)[:,None]
            self.pr.append(pr);self.pa.append(pa)
        self.phase_stats=dict(setup_seconds=time.perf_counter()-t,hg_raw_row_norm_range=[min(norms),max(norms)])

    def prepare(self):
        t=time.perf_counter();path=self.folder/'prepare.json';g=self.geo
        signature=dict(wavelengths=self.wl.tolist(),p=self.p,geometry_scale=g.scale)
        if path.exists() and json.loads(path.read_text())['signature']==signature:
            self.prepare_stats=json.loads(path.read_text());self.prepare_cache=True
        else:
            self.prepare_cache=False
            # A changed wavelength/weather signature must invalidate the source
            # field even when a caller deliberately reuses the same cache name.
            self.state_path.unlink(missing_ok=True)
            self.stats_path.unlink(missing_ok=True)
            for block in g.meta['blocks']:
                z=np.load(g.folder/block['file'])
                att=np.exp(-z['prefix']@self.ext)*z['ds'][:,None]
                sun=np.exp(-z['suncol']@self.ext)*z['lit'][:,None]
                height=z['height'];nu=z['nu'];prof=profiles(height)
                beam=sun*(prof[:,0,None]*self.ext[0]*rayleigh(nu)[:,None]+
                          prof[:,1,None]*self.ext[1]*self.p['ssa']*hg(nu,self.p['g'])[:,None])
                np.savez(self.folder/block['file'],att=att.astype(np.float32),beam=beam.astype(np.float32))
                z.close()
            self.prepare_stats=dict(signature=signature,seconds=time.perf_counter()-t)
            dump(path,self.prepare_stats)
        self.boundary_tr=np.exp(-g.column@self.ext)
        p,up,ex=probe.frame(np.full(len(g.chi),R),g.chi)
        col,lit=g.sun.lookup(p,np.full(len(g.chi),R))
        self.b0=self.p['albedo']/np.pi*np.maximum(0,np.cos(g.chi))[:,None]*np.exp(-col@self.ext)*lit[:,None]
        ground0=self.boundary(self.b0)
        # The first atmospheric scattering of directly reflected sunlight also
        # uses the original, unnormalised HG evaluation. Only later orders use
        # the conservative finite-quadrature row correction.
        self.g1=self.scatter(ground0,raw=True)
        self.direct_grid=self.beam_grid()+self.g1

    def beam_grid(self):
        g=self.geo;out=np.zeros((g.ns,g.nd,self.nw))
        for row in range(g.ns):
            ih,ic=divmod(row,len(g.chi));p,d=g.spatial_rays(row)
            r=np.full(g.nd,R+g.h[ih]);col,lit=g.sun.lookup(p,r)
            prof=profiles(np.array(g.h[ih]));nu=d[:,2]
            out[row]=np.exp(-col@self.ext)*lit[:,None]*(prof[0]*self.ext[0]*rayleigh(nu)[:,None]+
                     prof[1]*self.ext[1]*self.p['ssa']*hg(nu,self.p['g'])[:,None])
        return out

    def boundary(self,b):
        g=self.geo;x=g.endchi.ravel();j=np.clip(np.searchsorted(g.chi,x,side='right')-1,0,len(g.chi)-2)
        a=np.clip((x-g.chi[j])/(g.chi[j+1]-g.chi[j]),0,1)
        val=((1-a[:,None])*b[j]+a[:,None]*b[j+1]).reshape(g.ns,g.nd,self.nw)
        return val*self.boundary_tr*g.ground[...,None]

    def scatter(self,radiance,raw=False):
        g=self.geo;out=np.empty_like(radiance)
        data=radiance.reshape(len(g.h),len(g.chi),g.nd,self.nw)
        result=out.reshape(data.shape)
        for ih,h in enumerate(g.h):
            rr=data[ih].transpose(0,2,1).reshape(-1,g.nd)
            ray=(rr@(self.raw_pr[ih] if raw else self.pr[ih]).T).reshape(len(g.chi),self.nw,g.nd).transpose(0,2,1)
            aer=(rr@(self.raw_pa[ih] if raw else self.pa[ih]).T).reshape(len(g.chi),self.nw,g.nd).transpose(0,2,1)
            result[ih]=np.exp(-h/8)*self.ext[0]*ray+np.exp(-h/1.2)*self.ext[1]*self.p['ssa']*aer
        return out

    def transport(self,source,first=False):
        g=self.geo;field=source.reshape(-1,self.nw);out=np.zeros((g.ns*g.nd,self.nw))
        for block in g.meta['blocks']:
            z=np.load(g.folder/block['file']);opt=np.load(self.folder/block['file'])
            idx,iw=mapping(z,g);values=interpolate(field,idx,iw)
            if first:values+=opt['beam']
            values*=opt['att'];rows=z['row'];n=block['count']*g.nd
            sums=np.column_stack([np.bincount(rows,weights=values[:,k],minlength=n) for k in range(self.nw)])
            start=block['start']*g.nd;out[start:start+n]=sums
            z.close();opt.close()
        return out.reshape(g.ns,g.nd,self.nw)

    def ground_source(self,volume):
        g=self.geo;upweights=g.weights[0]*np.maximum(g.local[0,:,2],0)
        return self.p['albedo']/np.pi*np.einsum('cdw,d->cw',volume[:len(g.chi)],upweights)

    def solve(self):
        if self.state_path.exists() and self.stats_path.exists():
            z=np.load(self.state_path);self.ms=z['multiple'];self.single_ground=z['single_ground'];self.ms_ground=z['multiple_ground'];z.close()
            self.stats=json.loads(self.stats_path.read_text());self.stats['cache_hit']=True;return self.stats
        t=time.perf_counter();g=self.geo
        volume=self.transport(self.g1,first=True);b1=self.ground_source(volume)
        radiance=volume+self.boundary(b1)
        single_peak=np.max(self.direct_grid,axis=(0,1));total_source=self.direct_grid.copy()
        self.single_ground=b1;self.ms=np.zeros_like(self.direct_grid);self.ms_ground=np.zeros_like(b1)
        history=[];el=np.array([e for _,e,a in DIRS]);az=np.array([a for _,e,a in DIRS])
        totals={str(h):self.query(h,el,az,first=True) for h in HS}
        for order in range(2,21):
            source=self.scatter(radiance);volume=self.transport(source);bn=self.ground_source(volume)
            radiance=volume+self.boundary(bn)
            self.ms+=source;self.ms_ground+=bn;total_source+=source
            source_fraction=float(np.max(np.max(source,axis=(0,1))/np.maximum(np.max(total_source,axis=(0,1)),1e-300)))
            observer_fraction=0.
            for h in HS:
                increment=self.query(h,el,az,source=source)
                totals[str(h)]+=increment
                observer_fraction=max(observer_fraction,float(np.max(increment/np.maximum(totals[str(h)],1e-300))))
            history.append(dict(order=order,source_increment=source_fraction,observer_increment=observer_fraction))
            print(f'SOS x{g.scale} {self.name} n={order} source={source_fraction:.5g} observer={observer_fraction:.5g} {time.perf_counter()-t:.2f}s',flush=True)
            if order>=3 and max(source_fraction,observer_fraction)<.005:break
        np.savez(self.state_path,multiple=self.ms,single_ground=self.single_ground,multiple_ground=self.ms_ground)
        self.stats=dict(seconds=time.perf_counter()-t,orders_used=order,history=history,
                        capped=max(source_fraction,observer_fraction)>=.005,cache_hit=False,
                        nonfinite=int((~np.isfinite(self.ms)).sum()),negative=int((self.ms<0).sum()),
                        phase=self.phase_stats,prepare=self.prepare_stats)
        dump(self.stats_path,self.stats);return self.stats

    def query(self,hs,elevations,azimuths,source=None,first=False):
        g=self.geo;el,az=np.broadcast_arrays(elevations,azimuths);shape=el.shape
        e,a=np.deg2rad(el.ravel()),np.deg2rad(az.ravel())
        p0,up,ex=probe.frame(np.array(R+OBS),np.deg2rad(np.array(90-hs)));ey=np.cross(up,ex)
        directions=np.sin(e)[:,None]*up+np.cos(e)[:,None]*(np.cos(a)[:,None]*ex+np.sin(a)[:,None]*ey)
        field=(self.g1 if first else source).reshape(-1,self.nw)
        result=np.zeros((len(directions),self.nw))
        for start in range(0,len(directions),128):
            d=directions[start:start+128];p=np.broadcast_to(p0,d.shape);local=np.zeros((len(d),self.nw))
            for active,q,r,ds,prefix in geometry_segments(p,d,g.npath*2):
                idx,iw=g.lookup(q,r,d[active,None,:]);values=interpolate(field,idx.reshape(-1,16),iw.reshape(-1,16)).reshape(*r.shape,self.nw)
                if first:
                    col,lit=g.sun.lookup(q,r);prof=profiles(r-R);nu=d[active,2]
                    beam=np.exp(-col@self.ext)*lit[...,None]*(prof[...,0,None]*self.ext[0]*rayleigh(nu)[:,None,None]+
                         prof[...,1,None]*self.ext[1]*self.p['ssa']*hg(nu,self.p['g'])[:,None,None])
                    values+=beam
                local[active]+=np.sum(values*np.exp(-prefix@self.ext)*ds[...,None],axis=1)
            result[start:start+len(d)]=local
        return result.reshape(*shape,self.nw)

    def boxes(self,hs):
        # The PT means cover two adjacent jittered pixels in each axis. Match
        # their entire 2-degree box, rather than just four pixel centres.
        nodes,w=np.polynomial.legendre.leggauss(4);weights=(w[:,None]*w[None,:]/4).ravel()
        el=np.array([e+de for _,e,a in DIRS for de in nodes for da in nodes])
        az=np.array([a+da for _,e,a in DIRS for de in nodes for da in nodes])
        single=np.einsum('dpw,p->dw',self.query(hs,el,az,first=True).reshape(7,16,self.nw),weights)*self.solar
        multiple=np.einsum('dpw,p->dw',self.query(hs,el,az,source=self.ms).reshape(7,16,self.nw),weights)*self.solar
        return dict(single=single,multiple=multiple,total=single+multiple)


def pt_worker():
    sys.path.insert(0,str(ROOT/'src'))
    from render import render
    for hs in (2,-3,-6):
        for seed in (1,2):
            target=OUT/f'pt_e{hs}_s{seed}'
            if (target/'metadata.json').exists():continue
            c=json.loads((ROOT/'scenes/default.json').read_text())
            c.update(geometry='sphere',intersection='analytic',backend='metal',camera_preset=None,
                     camera_model='equirect',camera_altitude=OBS,elev_min=-10.,elev_max=90.,
                     width=360,height=100,spp=65536,seed=seed,wavelength_bins=1,wavelength_override_nm=550.,
                     stage=5,max_depth=0,batch_spp=512,tile_rows=5,
                     aerosol_extinction=.125,aerosol_scattering=.115,aerosol_angstrom=1.,g=.8,
                     ozone_DU=300.,ground_albedo=.3,sun_elevation=hs,sun_azimuth=0.)
            dump(OUT/f'pt_config_e{hs}_s{seed}.json',c)
            render(c,target)


def pt_table(hs):
    samples={key:[] for key in ('single','multiple','total')};metas=[]
    for seed in (1,2):
        folder=OUT/f'pt_e{hs}_s{seed}'
        z=np.load(folder/'spectral.npz');meta=json.loads((folder/'metadata.json').read_text());metas.append(meta)
        assert meta['completed_spp']==65536 and z['wavelength_nm'][0]==550
        expected=dict(radius=R,atmosphere_height=Z,camera_altitude=OBS,rayleigh_550=.013558,rayleigh_height=8.,
                      aerosol_extinction=.125,aerosol_scattering=.115,aerosol_height=1.2,aerosol_angstrom=1.,g=.8,
                      ozone_DU=300.,ground_albedo=.3,sun_elevation=hs,sun_azimuth=0.,width=360,height=100,
                      wavelength_override_nm=550.,seed=seed,stage=5,max_depth=0,elev_min=-10.,elev_max=90.)
        for key,value in expected.items():
            if meta['parameters'].get(key)!=value:raise ValueError(f'PT_REFERENCE_CONFIG_MISMATCH:{folder.name}:{key}')
        for key in samples:
            image=(z['single']+z['multiple'] if key=='total' else z[key])[...,0]
            vals=[]
            for _,el,az in DIRS:
                i=int(round(90-el))-1;cols=[(int(az)-1)%360,int(az)%360]
                vals.append(float(image[i:i+2][:,cols].mean()))
            samples[key].append(vals)
        z.close()
    return {key:dict(mean=np.mean(value,axis=0).tolist(),se=(np.abs(np.array(value[0])-value[1])/2).tolist()) for key,value in samples.items()},metas


def predecessor_helpers():
    names={'to_png','pano_coords','pano_from_grid','pano_from_fold','app_lut','app_pano','label','tag','fold_table'}
    tree=ast.parse((ROOT/'scripts/sasktran_probe.py').read_text())
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    env=dict(np=np,ROOT=ROOT,OUT=OUT,importlib=__import__('importlib'))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'sasktran_probe_selected_helpers','exec'),env)
    return env


def rgb_weights():
    # Exact same least-squares inputs as sasktran_probe.rgb_weights. Bitmap reads
    # the EXR because OpenEXR is absent in the supplied venv; no rendering here.
    import mitsuba as mi
    folder=ROOT/'results/purkinje_sweep/pkd_sphere_e+020_s1'
    sp=np.load(folder/'spectral.npz')['single'].reshape(-1,15).astype(float)
    rgb=np.array(mi.Bitmap(str(folder/'single_scattering.exr')),copy=False).astype(float).reshape(-1,3)
    w,*_=np.linalg.lstsq(sp,rgb,rcond=None)
    return w,float(np.linalg.norm(sp@w-rgb)/np.linalg.norm(rgb))


def panoramas(solver):
    from PIL import Image
    helper=predecessor_helpers();w,residual=rgb_weights();lut=helper['app_lut']()
    el=90-(np.arange(100)+.5)*.9;az=np.arange(360)+.5-180
    ee,aa=np.meshgrid(el,az,indexing='ij');timings={};comparisons=[]
    for hs in HS:
        t=time.perf_counter()
        single=solver.query(hs,ee,aa,first=True);multiple=solver.query(hs,ee,aa,source=solver.ms)
        lookup_seconds=time.perf_counter()-t
        spec=(single+multiple)[...,:15]*solver.solar[:15]
        rgb=spec@w;np.save(OUT/f'spec_e{hs}.npy',spec.astype(np.float32));np.save(OUT/f'rgb_e{hs}.npy',rgb.astype(np.float32))
        helper['to_png'](rgb,OUT/f'pano_e{hs}.png')
        # Reuse predecessor's 720x180 sampling and top/middle/bottom layout.
        sos=helper['pano_from_grid'](rgb[:,180:][::-1],el[::-1],np.arange(180)+.5)
        pt=helper['pano_from_fold'](helper['fold_table'](hs))
        app=helper['app_pano'](hs,lut)
        comp=np.concatenate([helper['label'](helper['to_png'](sos),f'Own SOS normal a0.30 hs {hs:+d}'),
             helper['label'](helper['to_png'](pt),f'PT lut0_fit normal a0.30 hs {hs:+d}'),
             helper['label'](helper['to_png'](app),f'app twilight_lut_v1 hs {hs:+d}')],axis=0)
        Image.fromarray(comp).save(OUT/f'compare_e{hs}.png');comparisons.append(comp)
        timings[str(hs)]=dict(lookup_seconds=lookup_seconds,lookup_and_output_seconds=time.perf_counter()-t,per_wavelength_seconds=lookup_seconds/15,
                              shape=list(spec.shape),nonfinite=int((~np.isfinite(spec)).sum()),negative=int((spec<0).sum()))
        print(f'PANO {hs} {timings[str(hs)]["lookup_and_output_seconds"]:.2f}s',flush=True)
        dump(OUT/'pano_timing.json',timings)
    Image.fromarray(np.concatenate(comparisons,axis=1)).save(OUT/'compare_all.png')
    np.save(OUT/'rgb_weights.npy',w)
    return dict(runs=timings,rgb_weights_fit_residual=residual)


def preserve_phase1():
    state={}
    for path in (ROOT/'scripts/sos_probe.py',ROOT/'docs/SOS_PROBE.md'):
        state[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    state['results']={str(p.relative_to(ROOT)):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
                      for p in (ROOT/'results/sos_probe').rglob('*') if p.is_file()}
    return state


def verify_products():
    from PIL import Image
    w=np.load(OUT/'rgb_weights.npy');rows={}
    for h in HS:
        sp=np.load(OUT/f'spec_e{h}.npy');rgb=np.load(OUT/f'rgb_e{h}.npy')
        assert sp.shape==(100,360,15) and rgb.shape==(100,360,3)
        assert sp.dtype==np.float32 and rgb.dtype==np.float32
        assert np.isfinite(sp).all() and np.isfinite(rgb).all() and np.all(sp>=0)
        assert np.array_equal(sp,sp[:,::-1])
        difference=float(np.max(np.abs(rgb-sp@w)))
        assert difference<2e-6*np.max(np.abs(rgb))+1e-12
        with Image.open(OUT/f'pano_e{h}.png') as im:assert im.size==(360,100)
        with Image.open(OUT/f'compare_e{h}.png') as im:assert im.size==(720,540)
        rows[str(h)]=dict(spectral_shape=list(sp.shape),dtype=str(sp.dtype),nonfinite=0,
                          negative_spectrum=0,mirror_max=0.,rgb_matrix_max=difference)
    with Image.open(OUT/'compare_all.png') as im:assert im.size==(4320,540)
    return rows


def report(result):
    rows=result['validation'];lines=['# 자체 SOS 실제 대기 탐침 (D0)',
        'normal: AOD550=.15, aerosol ext=.125/km, SSA=.92, H1.2 km, HG g.8; Rayleigh .013558/km H8 km.',
        '오존300 DU, 10–25–40 km 텐트; Lambert a.30, R6371 km/상단80 km/관측자0.1 km.',
        '정본 src/engine.py coefficients/phase/trace; LUT 카메라 고도는 r13 sphere metadata parameters.',
        '일반 HG는 직접 방향 격자 구적, φ 반사대칭; HG 행을 정규화해 등방 복사장 보존.',
        '1차 태양·직달지면 산란은 절단 없는 원래 Rayleigh+HG; 2차부터 유한 각 구적의 HG 행 정규화.',
        '태양 그림자·접점·오존 경계를 분할하고 1차 태양광은 보간 대신 해석적 원천으로 조회.',
        '지면 직달 B0 및 각 차수 확산 조도 Bn=a/π∫L_n μdΩ; 지면 반사는 대기 차수를 증가시키지 않음.',
        '파노라마는 기본 격자 15파장, 검증은550 nm 기본/두 배; 원천 증분0.5%·상한20차.',
        '|격자|r|χ|μ|φ|방향|시선/구간|기하초|기하GB|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,geo in result['geometry'].items():
        lines.append(f'|{name}|{geo["nr"]}|{geo["nchi"]}|{geo["nmu"]}|{geo["nphi"]}|{geo["directions"]}|{geo["npath_per_segment"]}|{geo["setup_seconds"]:.2f}|{geo["bytes"]/1e9:.3f}|')
    lines+=['방향: sun2/sun10/sun30/anti10/anti30/side10/zenith89. PT2×2픽셀 평균/SOS4×4Gauss 상자 평균.',
            '|태양고도|single/PT (7칸)|multiple/PT (7칸)|total/PT (7칸)|', '|---|---|---|---|']
    for hs,row in rows.items():
        cols=[' / '.join('N/A' if v is None else f'{v:.3f}' for v in row[k+'_ratio']) for k in ('single','multiple','total')]
        lines.append('|'+hs+'|'+'|'.join(cols)+'|')
    lines+=['|태양고도|PT multiple 상대SE (7칸)|single 자기수렴 최대|multiple 자기수렴 최대|', '|---|---|---:|---:|']
    for hs,row in rows.items():
        se=' / '.join(f'{100*v:.2f}%' for v in row['multiple_se_relative'])
        lines.append(f'|{hs}|{se}|{100*max(row["self_conv_single"]):.3f}%|{100*max(row["self_conv_multiple"]):.3f}%|')
    lines.append('그외550 nm multiple 자기수렴 최대: '+', '.join(f'{h}={100*max(result["self_convergence_all_hs"][str(h)]["multiple"]):.3f}%' for h in (0,-9,-12)))
    lines+=['보고 참고선 single±2%, multiple max(5%,2SE). 합격/불합격으로 산출을 중단하지 않음.',
            'PT 평균0인 비는 N/A(판정 불가); 유한 비의 범위와 참고선 초과 집계에서 제외.',
            '참고선 초과 칸: '+json.dumps(result['reference_exceedances'],ensure_ascii=False),
            '각 성분0 변형(에어로졸/오존/지면) 비는 result.json ablations. 외부 PT 추가 렌더 없음.',
            '성분0의 multiple 자기수렴 최대(3고도×7방향): '+', '.join(f'{name}={100*max(v for row in data.values() for v in row["multiple"]):.3f}%' for name,data in result['ablation_self_conv'].items()),
            *[f'{name}0/full multiple 비(두 배,+2/−3/−6): '+', '.join(f'{h}=[{min(row["multiple"]):.3f},{max(row["multiple"]):.3f}]' for h,row in data.items()) for name,data in result['ablations_double'].items()],
            'PT 정확히6장: +2/−3/−6 × seed1/2, 360×100, 550 nm, 65536 spp, 전체 차수.',
            'PT 검증 카메라90…−10°(선례1°/행); SOS 파노라마90…0°(.9°/행), 태양 중앙.',
            '15파장=360+(i+.5)*470/15; 분광 물리단위 W m^-2 sr^-1 nm^-1, spec (100,360,15) float32.',
            'RGB 행렬은 predecessor와 같은 EXR/spectral 적합; OpenEXR 부재로 Mitsuba Bitmap 읽기만 사용.',
            f'RGB 행렬 상대잔차={result["panorama"]["rgb_weights_fit_residual"]:.6g}. PNG는 중앙Y→.18, x/(1+x), sRGB.',
            '비교 그림 위SOS/가운데lut0_fit/아래앱LUT; 각각 동일한 자동노출 규칙, 노출값은 행별로 다름.',
            '기하(시선·보간·종별 광학열·태양열)는 파장/AOD/SSA/g/알베도 변경에 재사용; R/H/격자 변경은 캐시 버전 갱신·재생성 필요.',
            '보간 캐시는16쌍 대신5개 좌표로 저장; 원래 인덱스 완전 일치와 가중치차<1e−6을 모든 블록에서 확인.',
            '파장/날씨 변경 시 감쇠·정확한1차·HG 행렬(g 변경)·원천 반복은 다시 계산.',
            f'15칸+참조550 nm 공동 계수 준비={result["base_solver"]["prepare"]["seconds"]:.3f}s, 반복={result["base_solver"]["seconds"]:.3f}s, 차수={result["base_solver"]["orders_used"]}.',
            f'550 두 배 계수 준비={result["fine_solver"]["prepare"]["seconds"]:.3f}s, 반복={result["fine_solver"]["seconds"]:.3f}s, 차수={result["fine_solver"]["orders_used"]}.',
            f'마지막 관측 증분: 기본={result["base_solver"]["history"][-1]["observer_increment"]:.6g}, 두 배={result["fine_solver"]["history"][-1]["observer_increment"]:.6g}; 상한 도달={result["base_solver"]["capped"]}/{result["fine_solver"]["capped"]}.',
            '6장 순수조회/조회+출력 초: '+', '.join(f'{h}={v["lookup_seconds"]:.2f}/{v["lookup_and_output_seconds"]:.2f}' for h,v in result['panorama']['runs'].items()),
            f'추측: 기본 15파장×25고도 표={result["estimate_table_seconds"]:.2f}s (16장 공동 실측을 보수적으로 사용; 기하·계수·원천1회+조회평균×25).',
            '파장당 계수/반복/조회는 공동 벡터화 실측총시간/파장수 평균; 독립 파장 실행시간은 미측정.',
            f'HG 원래 행 적분 범위: 기본{result["base_solver"]["phase"]["hg_raw_row_norm_range"]}, 두 배{result["fine_solver"]["phase"]["hg_raw_row_norm_range"]}.',
            '한계: 정규화된 유한 HG 각 구적 및4D 선형 보간, 유한 격자·20차 상한·스칼라 복사장.',
            '미확인: 550 nm 이외 정량 PT 대조·자기수렴, 일반 위상함수의 더 높은 각 해상도, 편광·유한 태양 원반.',
            '추측: 성분0 비교는 민감도 측정이며 성분별 오차의 인과 증명은 아님.',
            f'최종 --all(기존 캐시 재사용)={result["elapsed_seconds"]:.3f}s; CPU numpy각프로세스2스레드, PT Metal작업1개 직렬.',
            f'1단계 보존 검사={result["phase1_preserved"]}. 기존 파일/commit/설치 변경 없음.']
    (ROOT/'docs/SOS_SKY.md').write_text('\n'.join(lines)+'\n')


def run_all():
    t0=time.perf_counter();before=preserve_phase1();dump(OUT/'phase1_before.json',before)
    log=(OUT/'pt.log').open('a')
    worker=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--pt-worker'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    base_geo=Geometry(1)
    base=SkySolver(base_geo,np.r_[WL,550.],name='normal16');base.solve()
    pano=panoramas(base)
    fine_geo=Geometry(2);fine=SkySolver(fine_geo,[550.],name='normal550');fine.solve()
    variants={};fine_variants={};variant_stats={}
    for name,key in [('no_aerosol','aer'),('no_ozone','ozone_DU'),('no_ground','albedo')]:
        p=NORMAL.copy();p[key]=0.
        solver=SkySolver(base_geo,[550.],p,name);solver.solve()
        variants[name]={str(h):{k:v[:,0].tolist() for k,v in solver.boxes(h).items()} for h in (2,-3,-6)}
        fine_variant=SkySolver(fine_geo,[550.],p,name);fine_variant.solve()
        fine_variants[name]={str(h):{k:v[:,0].tolist() for k,v in fine_variant.boxes(h).items()} for h in (2,-3,-6)}
        variant_stats[name]=dict(base=solver.stats,double=fine_variant.stats)
    worker.wait();log.close()
    if worker.returncode:raise RuntimeError('PT_WORKER_FAILED: see results/sos_sky/pt.log; panoramas already written')
    validation={};exceedances=[];pt_seconds=[];self_convergence={}
    base_boxes={h:base.boxes(h) for h in HS};fine_boxes={h:fine.boxes(h) for h in HS}
    for h in HS:
        self_convergence[str(h)]={k:(np.abs(base_boxes[h][k][:,-1]-vals[:,0])/np.maximum(vals[:,0],1e-300)).tolist()
                                 for k,vals in fine_boxes[h].items()}
    for hs in (2,-3,-6):
        b=base_boxes[hs];f=fine_boxes[hs];pt,metas=pt_table(hs)
        row=dict(sos={k:v[:,0].tolist() for k,v in f.items()},base_sos={k:v[:,-1].tolist() for k,v in b.items()},pt=pt)
        for kind in ('single','multiple','total'):
            mean=np.array(pt[kind]['mean']);se=np.array(pt[kind]['se'])/np.maximum(mean,1e-300)
            valid=mean>0
            ratio=np.divide(f[kind][:,0],mean,out=np.full(7,np.nan),where=valid)
            row[kind+'_ratio']=[float(v) if ok else None for v,ok in zip(ratio,valid)]
            row[kind+'_valid']=valid.tolist();row[kind+'_se_relative']=se.tolist()
            if kind!='total':
                tol=np.full(7,.02) if kind=='single' else np.maximum(.05,2*se)
                row[kind+'_tolerance']=tol.tolist()
                for i in np.flatnonzero(valid & (np.abs(ratio-1)>tol)):
                    exceedances.append(dict(hs=hs,dir=f'{DIRS[i][0]}{DIRS[i][1]}',kind=kind,ratio=float(ratio[i]),tolerance=float(tol[i])))
            row['self_conv_'+kind]=(np.abs(b[kind][:,-1]-f[kind][:,0])/np.maximum(f[kind][:,0],1e-300)).tolist()
        validation[str(hs)]=row;pt_seconds.extend(m['render_seconds'] for m in metas)
    ablations={};ablations_double={};ablation_self_conv={}
    for name,data in variants.items():
        ablations[name]={h:{k:(np.array(vals)/np.maximum(validation[h]['base_sos'][k],1e-300)).tolist()
                           for k,vals in kinds.items()} for h,kinds in data.items()}
        ablations_double[name]={h:{k:(np.array(vals)/np.maximum(validation[h]['sos'][k],1e-300)).tolist()
                                  for k,vals in kinds.items()} for h,kinds in fine_variants[name].items()}
        ablation_self_conv[name]={h:{k:(np.abs(np.array(data[h][k])-vals)/np.maximum(vals,1e-300)).tolist()
                                  for k,vals in kinds.items()} for h,kinds in fine_variants[name].items()}
    lookup_mean=np.mean([v['lookup_seconds'] for v in pano['runs'].values()])
    result=dict(wavelengths_nm=WL.tolist(),atmosphere=NORMAL,observer_km=OBS,dirs=[f'{n}{e}/{a}' for n,e,a in DIRS],
                geometry=dict(base=base_geo.meta,double=fine_geo.meta),base_solver=base.stats,fine_solver=fine.stats,
                validation=validation,reference_exceedances=exceedances,ablations=ablations,
                ablations_double=ablations_double,ablation_self_conv=ablation_self_conv,variant_stats=variant_stats,
                self_convergence_all_hs=self_convergence,
                panorama=pano,pt_render_seconds=pt_seconds,
                estimate_table_seconds=base_geo.meta['setup_seconds']+base.stats['prepare']['seconds']+base.stats['seconds']+lookup_mean*25,
                timing_per_wavelength=dict(shared_geometry_seconds=base_geo.meta['setup_seconds'],
                    coefficients_seconds=base.stats['prepare']['seconds']/16,iterations_seconds=base.stats['seconds']/16,
                    panorama_lookup_seconds=lookup_mean/15),
                phase1_preserved=before==preserve_phase1(),verification=verify_products(),elapsed_seconds=time.perf_counter()-t0)
    dump(OUT/'result.json',result);report(result)
    assert len((ROOT/'docs/SOS_SKY.md').read_text().splitlines())<=60
    print(f'COMPLETE seconds={result["elapsed_seconds"]:.3f} reference_exceedances={len(exceedances)} phase1_preserved={result["phase1_preserved"]}',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--all',action='store_true');ap.add_argument('--pt-worker',action='store_true')
    a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.pt_worker:pt_worker()
    elif a.all:run_all()
    else:ap.error('choose --all')


if __name__=='__main__':main()
