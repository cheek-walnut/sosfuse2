"""r11 host port: three planner/control tests plus exact NOPREFIX geometry parity."""
import ast
import itertools
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sosfuse2 import stratosphere, device_profile


class WallHost(unittest.TestCase):



    def test_noprefix_real_geometry_bytes(self):
        from sosfuse2 import paths
        root=paths.ACCEPTED   # frozen accepted sources (sky-sos scripts/sos_*.py byte copies)
        def functions(path,names,namespace):
            tree=ast.parse(path.read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
            exec(compile(tree,str(path),'exec'),namespace);return SimpleNamespace(**namespace)
        probe=functions(root/'sos_probe.py',{'boundaries','gl'},dict(np=np,R=6371.,Z=80.))
        old=functions(root/'sos_sky.py',{'geometry_segments','profiles'},dict(np=np,probe=probe,R=6371.))
        profile=stratosphere.profile_config({});rng=np.random.default_rng(17)
        for height,n in itertools.product((0.,.1,10.,21.,25.,40.,79.),(24,32)):
            d=rng.normal(size=(65,3));d/=np.linalg.norm(d,axis=1)[:,None]
            d[:5]=[[0,0,1],[0,0,-1],[1,0,0],[1,0,1e-10],[1,0,-1e-10]]
            p=np.broadcast_to([0.,0.,6371.+height],d.shape)
            with patch.dict(os.environ,{'SOSFUSE2_STRATO_HOST_NOPREFIX':'0'}):expected=stratosphere.columns(p,d,old,profile,n)
            with patch.dict(os.environ,{'SOSFUSE2_STRATO_HOST_NOPREFIX':'1'}):
                with patch.object(old,'geometry_segments',side_effect=AssertionError('unused optical prefix evaluated')):
                    actual=stratosphere.columns(p,d,old,profile,n)
            self.assertEqual(actual.tobytes(),expected.tobytes())


if __name__=='__main__':unittest.main(verbosity=2)
