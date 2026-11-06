"""Accepted cell formulas, actual spawn failures, and cgroup quota ancestry."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock
import numpy as np
from . import column_parallel as cp, density, geometry, ground_parallel as gp, host_cache, host_limits as hl, stratosphere as st
try:
    from .numerical_checks import assert_equivalent, within_tol
except ImportError:  # standalone research harness
    from numerical_checks import assert_equivalent, within_tol


def fail_cell(task):
    if isinstance(task, int) and task == 1 or isinstance(task, tuple) and len(task[0]) < 8192:
        raise RuntimeError('E_TEST_GROUND_WORKER')
    return gp._run(task)


def accepted(s, scope, kind, optics=None, profile=None, suncol=None, method=None):
    # method: host_cache.prepare_ground passes density.sun_method() (③ 0(b)); this GL24 reference is the default only.
    assert method in (None, 'gl'), method
    old=scope['b'].sh.old;rows=[]
    for cell in range(s.geo.ns):
        ih,ic=divmod(cell,len(s.geo.chi));r=6371.+s.geo.h[ih];ch=s.geo.chi[ic]
        loc,w=scope['b'].sh.incoming(r,ch,1)
        p,up,ex=old.probe.frame(np.array(r),ch);ey=np.cross(up,ex)
        d=loc[:,0,None]*ex+loc[:,1,None]*ey+loc[:,2,None]*up;p=np.broadcast_to(p,d.shape)
        end=old.probe.boundaries(p,d)[:,-1];q=p+end[:,None]*d;re=np.linalg.norm(q,axis=-1)
        col=old.total_columns(p,d,24)
        if optics is not None:col=density.correct_columns(p,d,col,old,optics,24)
        sc,lit=s.geo.sun.lookup(q,np.maximum(6371.,re));ec=np.arccos(np.clip(q[:,2]/re,-1,1))
        if kind == 'host':
            rows.append((loc,w,q,re,col[:,[0,2,3,4]],(np.cos(ec)>0)&lit&(re<6371.+1e-5)&(w>0)))
        else:
            column=st.columns(p,d,old,profile,24);ssc,_=st.sun_lookup(q,np.maximum(6371.,re),s.geo.sun,suncol)
            rows.append((loc,w,sc,lit,col,column,ssc,ec,re))
    return rows


class GroundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        with mock.patch.dict(os.environ,{'SOSFUSE2_SUN_WORKERS':'1',gp.ENV:'1'}):
            s,cls.scope,_=geometry.create(Path(cls.tmp.name)/'prepare','selftest',[550.],'numpy')
        # Real incoming quadratures at illuminated, terminator, shadow and altitude cells.
        s.geo.h=np.array([.1,25.]);s.geo.chi=np.deg2rad([80.,93.]);s.geo.ns=4
        cls.s=s;cls.old=cls.scope['b'].sh.old
        cls.optics=density.build(dict(density=dict(density_profile='usstd76',ozone_profile='usstd76')))
        cls.profile=st.profile_config({})
        cls.suncol=np.zeros(s.geo.sun.col.shape[:2])

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def check_close(self,a,b):
        self.assertEqual(len(a),len(b))
        for ra,rb in zip(a,b):
            for x,y in zip(ra,rb):
                assert_equivalent(x, y)

    def mapped(self,call,workers,fail=False):
        log=io.StringIO()
        with mock.patch.dict(os.environ,{gp.ENV:str(workers)}),contextlib.redirect_stdout(log):
            if fail:
                with mock.patch.object(gp,'_run',fail_cell):value=call()
            else:value=call()
        events=[json.loads(l) for l in log.getvalue().splitlines() if '"ground_parallel"' in l]
        if workers>1:
            self.assertEqual(events[-1]['result'],'fallback' if fail else 'parallel')
            if fail:self.assertEqual(events[-1]['reason'],'E_TEST_GROUND_WORKER')
        return value

    def test_cells(self):
        for kind,optics in (('host',self.optics),('strato',None),('strato',self.optics)):
            args=(self.s,self.scope,kind,optics,self.profile,self.suncol)
            expected=accepted(*args)
            for count,fail in ((1,False),(2,False),(2,True)):
                self.check_close(expected,self.mapped(lambda:gp.cells(*args),count,fail))

    def test_aerosol_chunks(self):
        p=np.broadcast_to([0.,0.,6372.],(8193,3));d=np.broadcast_to([0.,0.,-1.],p.shape)
        expected=np.concatenate([self.old.total_columns(p[i:i+8192],d[i:i+8192],24)[:,1]
                                 for i in range(0,len(d),8192)])
        for count,fail in ((1,False),(2,False),(2,True)):
            value=self.mapped(lambda:gp.aerosol(self.scope,self.s.geo.sun,p,d),count,fail)
            assert_equivalent(expected, value)

    def test_host_outputs(self):
        def run():
            with mock.patch.object(host_cache,'context',return_value=(Path(self.tmp.name),{})), \
                    mock.patch.object(host_cache,'cached',side_effect=lambda folder,key,build,**kw:build()):
                host_cache.prepare_ground(self.s,self.scope,self.optics)
            return [getattr(self.s,k).copy() for k in ('gm','goff','gdir','gval')]
        with mock.patch.object(gp,'cells',accepted):
            expected=self.mapped(run,1)
        actual=self.mapped(run,2)
        self.check_close([expected],[actual])

    def test_strato_entry_and_noprefix(self):
        with mock.patch.dict(os.environ,{'STRATO_ATTACH_CACHE':'','SOSFUSE2_STRATO_HOST_NOPREFIX':'1'}), \
                mock.patch.object(st,'solar_grid',return_value=self.suncol):
            expected=accepted(self.s,self.scope,'strato',getattr(self.s,'density_optics',None),self.profile,self.suncol)
            _,actual=self.mapped(lambda:st._ground_geometry(self.s,self.scope,self.old,self.profile),2)
            self.check_close(expected,actual)

    def test_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'a/b').mkdir(parents=True)
            (root/'cpu.max').write_text('800000 100000')
            (root/'a/cpu.max').write_text('250000 100000')
            (root/'a/b/cpu.max').write_text('max 100000')
            mount=f'1 0 0:1 / {root} rw - cgroup2 cgroup rw'
            self.assertEqual(min(cp._quota_limits('0::/a/b',mount)),2)
            self.assertEqual(cp._quota_limits('0::/',mount.replace(' / ',' /a ')),[8])
            # default_workers lives in host_limits (re-exported by column_parallel); patch its own module globals.
            with mock.patch.object(hl.os,'sched_getaffinity',return_value=set(range(4)),create=True), \
                    mock.patch.object(hl.os,'cpu_count',return_value=28), \
                    mock.patch.object(hl.Path,'read_text',side_effect=['0::/a/b',mount]), \
                    mock.patch.object(hl,'_quota_limits',return_value=[2]):
                self.assertEqual(cp.default_workers(),2)
            with mock.patch.object(hl.os,'sched_getaffinity',return_value=set(range(4)),create=True), \
                    mock.patch.object(hl.os,'cpu_count',return_value=28), \
                    mock.patch.object(hl.Path,'read_text',side_effect=['0::/a/b',mount]), \
                    mock.patch.object(hl,'_quota_limits',return_value=[8]):
                self.assertEqual(cp.default_workers(),4)
        with mock.patch.dict(os.environ,{gp.ENV:''}),mock.patch.object(gp,'default_workers',return_value=3):
            self.assertEqual(gp.workers(),3)
        for value in ('0','-1','invalid'):
            with mock.patch.dict(os.environ,{gp.ENV:value}),self.assertRaisesRegex(ValueError,'E_GROUND_WORKERS'):
                gp.workers()
