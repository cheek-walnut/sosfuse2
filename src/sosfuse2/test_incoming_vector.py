"""Byte oracle: the accepted interval loop, including ring and horizon edges."""
# ruff: noqa: F821  -- accepted_ring_edges/accepted are verbatim copies of sos_full code, rebound at run time to the
# sos_full module globals (sh, R, ...) with types.FunctionType (see the test body), so those names are not module names here.
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
import numpy as np
from .numerical_checks import assert_equivalent
from . import geometry


def accepted_ring_edges(r,chi,phi,heights):
    """View-plane intersection with the exact shadow-cylinder/shell ring.

    The night-side solar shadow has x²+y²=R², z<0. At shell R+h
    its boundary is the ring z=-sqrt((R+h)²-R²). Project each of its
    two intersections with the ray's azimuth plane onto local mu.
    These are real endpoints of illumination in each density layer.
    """
    p,up,ex=sh.old.probe.frame(np.array(r),np.array(chi));ey=np.cross(up,ex)
    horizontal=np.cos(phi)*ex+np.sin(phi)*ey
    normal=np.cross(up,horizontal);amp=R*np.hypot(normal[0],normal[1])
    if amp<1e-10:return []
    angle=np.arctan2(normal[1],normal[0]);answer=[]
    for h in heights:
        z=-np.sqrt(max(0,(R+h)**2-R**2));target=-normal[2]*z/amp
        if abs(target)>1:continue
        delta=np.arccos(np.clip(target,-1,1))
        for az in (angle-delta,angle+delta):
            q=np.array([R*np.cos(az),R*np.sin(az),z]);d=q-p;length=np.linalg.norm(d)
            if length<1e-8:continue
            d/=length
            if d@horizontal<=1e-10:continue
            # Reject rings hidden behind the opaque Earth.
            b=p@d;disc=b*b-r*r+R*R
            hit=-b-np.sqrt(max(0,disc))
            if disc>=0 and b<0 and hit>=-1e-7 and hit<length-1e-5:continue
            answer.append(float(d@up))
    return answer

def accepted(r,chi,nmu,nphi,a=4.,rings=True):
    mg=-np.sqrt(max(0,1-(R/r)**2));sa=np.sinh(a);u,wu=sh.old.probe.gl(nmu)
    mus=[];phis=[];weights=[]
    for j,phi in enumerate(np.linspace(0,np.pi,nphi+1)):
        edges=[-1.,1.,mg,0.,mg-.05234,mg+.05234,mg-.0061,mg+.0061]
        rp=r*np.sin(chi)
        if rp>R+1e-8 and abs(np.sin(chi))>1e-8:
            kk=R/np.sqrt(rp*rp-R*R)
            for sign in (-1,1):
                t=np.cos(phi)*np.cos(chi)/np.sin(chi)+sign*np.sin(phi)/(kk*np.sin(chi))
                mu=t/np.sqrt(1+t*t);st=np.sqrt(1-mu*mu)
                dx=mu*np.sin(chi)-st*np.cos(phi)*np.cos(chi);dy=-st*np.sin(phi)
                dz=mu*np.cos(chi)+st*np.cos(phi)*np.sin(chi)
                tmin=-rp*dx/max(dx*dx+dy*dy,1e-25)
                if abs(dx)+abs(dy)<1e-10 or (tmin>0 and r*np.cos(chi)+tmin*dz<0):edges.append(mu)
        if phi<np.pi/6:
            mu=np.cos(chi)/np.sqrt(np.cos(chi)**2+np.cos(phi)**2*np.sin(chi)**2)
            edges.extend([mu,mu-.012,mu+.012])
        if rings:edges+=ring_edges(r,chi,phi,(0.,1.2,8.,25.,40.,80.))
        edges=np.unique(np.clip(edges,-1,1));delta=edges-mg
        tt=np.sign(delta)*np.arcsinh(np.abs(delta)/np.where(delta<0,1+mg,1-mg)*sa)/a
        for left,right in zip(tt[:-1],tt[1:]):
            if right-left<1e-10:continue
            t=left+(right-left)*u;den=np.where(t<0,1+mg,1-mg)
            mus.extend(mg+np.sign(t)*den*np.sinh(a*np.abs(t))/sa)
            weights.extend((right-left)*wu*den*a*np.cosh(a*np.abs(t))/sa*(2*np.pi/nphi)*(.5 if j in (0,nphi) else 1.))
            phis.extend([phi]*len(t))
    mu=np.array(mus);phi=np.array(phis);st=np.sqrt(np.maximum(0,1-mu*mu))
    return np.column_stack([st*np.cos(phi),st*np.sin(phi),mu]),np.array(weights)


class IncomingTests(unittest.TestCase):
    def test_original_loop_bytes(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ,
                {'SOSFUSE2_SUN_WORKERS':'1','SOSFUSE2_GROUND_WORKERS':'1'}):
            _,scope,_=geometry.create(Path(tmp),'selftest',[550.],'numpy')
        legacy=scope['b'].legacy
        env=dict(np=np,R=6371.,sh=scope['b'].sh)
        env['ring_edges']=types.FunctionType(accepted_ring_edges.__code__,env)
        oracle=types.FunctionType(accepted.__code__,env,argdefs=accepted.__defaults__)
        for height in (0., .001, 1.2, 10., 25., 40., 79.9):
            for degrees in (0.,80.,90.,93.,99.,110.,180.):
                for rings in (False,True):
                    for nmu,nphi in ((3,16),(4,32)):
                        args=(6371.+height,np.deg2rad(degrees),nmu,nphi,4.,rings)
                        expected=oracle(*args);actual=legacy.incoming(*args)
                        for a,b in zip(expected,actual):
                            assert_equivalent(a, b)


if __name__ == '__main__':
    unittest.main()
