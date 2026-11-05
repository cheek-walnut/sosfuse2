#!/usr/bin/env python3
"""D1 host-cache regressions; run directly with the project's Python."""
import contextlib
import copy
import fcntl
import time
import warnings
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sosfuse2 import host_cache as hc


def concurrent_writer(folder):
    def build():
        with (Path(folder)/'builds').open('a') as f:f.write('build\n')
        return dict(columns=np.arange(6000,dtype='f4').reshape(-1,2))
    got=hc.cached(folder,dict(test='concurrent'),build)
    assert np.array_equal(got['columns'],np.arange(6000,dtype='f4').reshape(-1,2))


class CacheTests(unittest.TestCase):
    def test_atomic_four_workers_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            ctx=mp.get_context('spawn');workers=[ctx.Process(target=concurrent_writer,args=(folder,)) for _ in range(4)]
            for w in workers:w.start()
            for w in workers:w.join(60);self.assertEqual(w.exitcode,0)
            self.assertEqual((Path(folder)/'builds').read_text(),'build\n')
            file=next(Path(folder).glob('*.hcache'));original=file.read_bytes()
            for corrupt in (original[:-1],original[:-1]+bytes([original[-1]^1])):
                file.write_bytes(corrupt)
                with self.assertWarnsRegex(RuntimeWarning,'E_HOST_CACHE_CORRUPT'):concurrent_writer(folder)
                self.assertEqual(file.read_bytes(),original)
            self.assertFalse(list(Path(folder).glob('*.tmp')))

    def test_keys_and_failure_do_not_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            key=dict(common='one',asset_sha256='a',density_profile='usstd76',ozone_profile='usstd76',rayleigh_column='keep',mask=3,lengths=[1,2])
            hc.cached(folder,key,lambda:dict(a=np.array([1.])))
            for name in key:
                changed=dict(key);changed[name]='changed'
                got=hc.cached(folder,changed,lambda:dict(a=np.array([2.])))
                self.assertEqual(got['a'][0],2.)
            def fail():raise RuntimeError('E_TEST_BUILD')
            with self.assertRaisesRegex(RuntimeError,'E_TEST_BUILD'):hc.cached(folder,dict(fail=True),fail)
            self.assertFalse(list(Path(folder).glob('*.tmp')))

    def test_reduced_accepted_bytes_and_height_reuse(self):
        from types import SimpleNamespace
        from sosfuse2 import run_gpu_state_v3 as runner
        from sosfuse2.geometry import normal_state
        cfg=runner.settings_for(SimpleNamespace(reduced=True,settings_json=None))
        with tempfile.TemporaryDirectory() as folder,patch.object(hc,'ROOT',Path(folder)/'cache'):
            state=normal_state();state['density']=dict(density_profile='usstd76',ozone_profile='usstd76',rayleigh_column='usstd')
            # A -> B -> A: the third case shares the first case's scale height with other optics, so its host
            # aerosol column must come from the cache (ground_parallel.aerosol is not called) and stay within tolerance.
            cases=[(1.2,.15,15),(3.,.02,15),(1.2,.4,15),(1.,0.,1)]
            from sosfuse2 import ground_parallel
            def no_aerosol(*_):raise AssertionError('E_TEST_AEROSOL_RECOMPUTED')
            for i,(height,aod,W) in enumerate(cases):
                current=copy.deepcopy(state);current['parameters'].update(scale_height_km=height,aod550=aod,ssa=.92+.03*(i==2))
                if aod==0:current['parameters']['albedo']=0.
                wl=[550.] if W==1 else 360+(np.arange(15)+.5)*470/15
                with patch.dict(os.environ,SOSFUSE2_HOST_CACHE='0'):
                    a,_=runner.create_configured(Path(folder)/f'off{i}',cfg,wl,'numpy',current)
                with patch.dict(os.environ,SOSFUSE2_HOST_CACHE='1'),patch.object(ground_parallel,'aerosol',no_aerosol) if i==2 else contextlib.nullcontext():
                    b,_=runner.create_configured(Path(folder)/'reused',cfg,wl,'numpy',current)
                for name,value in vars(a).items():
                    if isinstance(value,np.ndarray):
                        self.assertEqual(value.dtype,getattr(b,name).dtype,name)
                        self.assertEqual(value.shape,getattr(b,name).shape,name)
                        self.assertEqual(value.tobytes(),getattr(b,name).tobytes(),(i,name))
                self.assertEqual(a.geo.sun.col.tobytes(),b.geo.sun.col.tobytes())
            # Default profiles still require one real prepare_ground after the skipped constructors.
            state.pop('density')
            with patch.dict(os.environ,SOSFUSE2_HOST_CACHE='0'):
                a,_=runner.create_configured(Path(folder)/'default_off',cfg,[550.],'numpy',state)
            with patch.dict(os.environ,SOSFUSE2_HOST_CACHE='1'):
                b,_=runner.create_configured(Path(folder)/'default_on',cfg,[550.],'numpy',state)
            for name in ('gm','goff','gdir','gval'):self.assertEqual(getattr(a,name).tobytes(),getattr(b,name).tobytes(),name)

    def test_query_capture_cache_bytes(self):
        """Cached query packs equal a fresh _capture: field order, dtype, shape, bytes, nq; second call is a hit."""
        from types import SimpleNamespace
        from sosfuse2 import run_gpu_state_v3 as runner, table_fast
        from sosfuse2 import pipeline_execution as pe
        from sosfuse2.geometry import normal_state
        cfg=runner.settings_for(SimpleNamespace(reduced=True,settings_json=None))
        wl,hs,el,az=table_fast.axes(True)
        ee,aa=np.meshgrid(el,az,indexing='ij');dummy=np.empty(0,'f4')
        with tempfile.TemporaryDirectory() as folder:
            states=[normal_state(),normal_state()];states[1]['parameters'].update(scale_height_km=3.,aod550=.02)
            fresh=None
            for i,state in enumerate(states):
                s,_=runner.create_configured(Path(folder)/f'p{i}',cfg,wl,'numpy',state,geometry_cache=Path(folder)/'geo')
                if fresh is None:fresh=[table_fast._capture(s,hs[j],ee,aa,dummy) for j in range(len(hs))]
                with patch.object(table_fast,'_capture',side_effect=AssertionError('E_TEST_RECAPTURE')) if i else contextlib.nullcontext():
                    got=pe.captured_queries(s,hs,list(range(len(hs))),ee,aa,dummy)
                self.assertEqual([len(r) for r in got],[len(r) for r in fresh])
                for raws_a,raws_b in zip(fresh,got):
                    for x,y in zip(raws_a,raws_b):
                        self.assertEqual(list(x),list(y))
                        self.assertEqual((type(x['nq']),x['nq']),(type(y['nq']),y['nq']))
                        for k in x:
                            if k=='nq':continue
                            self.assertEqual((x[k].dtype,x[k].shape,x[k].tobytes()),(y[k].dtype,y[k].shape,y[k].tobytes()),k)
            self.assertEqual(len(list((Path(folder)/'geo').rglob('query_cache/*.hcache'))),1)

    def test_mode_bytes_and_hit(self):
        with tempfile.TemporaryDirectory() as d:
            key={'test':'mode'}
            arrays={'a':np.array([1.,2.])}
            hc.cached(d,key,lambda:arrays)
            path=next(Path(d).glob('*.hcache'));before=path.read_bytes()
            self.assertEqual(path.stat().st_mode & 0o777,0o644)
            hc.cached(d,key,lambda:self.fail('unexpected build'))
            self.assertEqual(before,path.read_bytes())
    def test_unwritable(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'file';path.touch()
            with self.assertWarnsRegex(RuntimeWarning,'E_HOST_CACHE_UNWRITABLE'), self.assertRaises(hc.CacheUnavailable):
                hc.cached(path/'child',{},lambda:self.fail('unexpected build'))
    def test_read_permission(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/(hc.digest({})+'.hcache');path.touch()
            original=Path.open
            def denied(p,*a,**k):
                if p==path:raise PermissionError('injected read denial')
                return original(p,*a,**k)
            with patch.object(Path,'open',denied), warnings.catch_warnings(record=True) as ws:
                with self.assertRaises(hc.CacheUnavailable):hc.cached(d,{},lambda:self.fail('unexpected build'))
            self.assertEqual(len(ws),1)
            self.assertIn('E_HOST_CACHE_PERMISSION',str(ws[0].message))
            self.assertEqual(path.read_bytes(),b'')
    def test_cleanup_and_stale(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);old=folder/'old.tmp';young=folder/'young.tmp';busy=folder/'busy.tmp'
            for p in (old,young,busy):p.write_bytes(b'tmp')
            for p in (old,busy):os.utime(p,(time.time()-3601,)*2)
            stale=folder/('a'*64+'.hcache');stale.write_bytes(b'old-cache')
            with busy.open('rb') as f:
                fcntl.flock(f,fcntl.LOCK_EX)
                with self.assertWarnsRegex(RuntimeWarning,'HOST_CACHE_STALE n=1 bytes=9'):
                    hc.cached(folder,{},lambda:{'a':np.array([1.])})
            self.assertFalse(old.exists());self.assertTrue(young.exists());self.assertTrue(busy.exists())
            self.assertEqual(stale.read_bytes(),b'old-cache')

    def test_root_and_unwritable_fallback_bytes(self):
        from types import SimpleNamespace
        from sosfuse2 import run_gpu_state_v3 as runner
        from sosfuse2.geometry import normal_state
        host_cache=hc
        cfg=runner.settings_for(SimpleNamespace(reduced=True,settings_json=None))
        state=copy.deepcopy(normal_state());state['density']=dict(density_profile='usstd76',ozone_profile='usstd76',rayleigh_column='usstd')
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);cache=folder/'custom-cache';blocked=folder/'blocked';blocked.touch()
            with patch.object(host_cache,'ROOT',folder/'unused'),patch.dict(os.environ,SOSFUSE2_HOST_CACHE='1'):
                normal,_=runner.create_configured(folder/'normal',cfg,[550.],'numpy',state,geometry_cache=cache)
                # sun, ground, and the ground aerosol column of this scale height
                self.assertEqual(len(list(cache.glob('endpoints_*/host_cache/*.hcache'))),3)
                self.assertFalse((folder/'unused').exists())
                with self.assertWarnsRegex(RuntimeWarning,'E_HOST_CACHE_UNWRITABLE'):
                    fallback,_=runner.create_configured(folder/'fallback',cfg,[550.],'numpy',state,geometry_cache=blocked)
            with patch.dict(os.environ,SOSFUSE2_HOST_CACHE='0'):
                old,_=runner.create_configured(folder/'off',cfg,[550.],'numpy',state)
            for obj in (normal,fallback):
                for name,value in vars(old).items():
                    if isinstance(value,np.ndarray):self.assertEqual(value.tobytes(),getattr(obj,name).tobytes(),name)
                self.assertEqual(old.geo.sun.col.tobytes(),obj.geo.sun.col.tobytes())


if __name__=='__main__':unittest.main(verbosity=2)
