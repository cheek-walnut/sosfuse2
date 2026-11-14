"""attach_lock_name (run_gpu_state_v3): synthetic state dicts, no solver, no CuPy."""

from sosfuse2.foundation import paths
import copy
import json
from pathlib import Path
import sys
import types
import unittest
import numpy as np
from sosfuse2.jobs.run_state import attach_lock_name
from sosfuse2.atmosphere.stratosphere import _geometry_key, profile_config

STATES = json.loads((paths.FIXTURES / 'states_lutv2_strato_usstd.json').read_text())['states']


def variant(base, center=None, sigma=None, status='enabled', **params):
    state = copy.deepcopy(base)
    state['stratosphere']['status'] = status
    state['parameters'].update(params)
    if center is not None:
        state['stratosphere']['center_km'] = center
    if sigma is not None:
        state['stratosphere']['sigma_km'] = sigma
    return state


class AttachLockName(unittest.TestCase):
    base = next(s for s in STATES if s['id'] == 'dual_st0p01_aod0p15')

    def test_profile_separates(self):
        names = {
            attach_lock_name(variant(self.base, c, s)) for c, s in ((21.0, 3.0), (22.0, 5.0), (21.0, 5.0), (22.0, 3.0))
        }
        self.assertEqual(len(names), 4)

    def test_optics_only_shares(self):
        a = attach_lock_name(self.base)
        self.assertEqual(a, attach_lock_name(variant(self.base, aod550=0.4, stratosphere_tau550=0.05, albedo=0.1)))
        self.assertEqual(a, attach_lock_name(variant(self.base, center=21, sigma=3)))  # int vs float, as profile_config
        self.assertNotEqual(a, attach_lock_name(variant(self.base, scale_height_km=2.0)))
        self.assertEqual(a, attach_lock_name(variant(self.base, ssa=0.001, g=0.0, angstrom=0.5, ozone_du=0.0)))

    def test_geometry_key_ignores_optics_only(self):
        """ssa/g/angstrom/ozone_du share one attach file (measured byte-equal in cpu_prep_multistate_20261024); H does not."""
        geo = types.SimpleNamespace(
            h=np.arange(4.0),
            chi=np.arange(3.0),
            cfg=dict(nmu=1, nphi=2, path_n=4),
            sun=types.SimpleNamespace(h=np.arange(2.0), x=np.arange(3.0), col=np.ones((2, 3))),
        )
        key = lambda state: _geometry_key(types.SimpleNamespace(geo=geo, fuse_state=state), None, profile_config(state))
        a = key(self.base)
        for change in (
            dict(ssa=0.001),
            dict(g=0.0),
            dict(angstrom=0.5),
            dict(ozone_du=0.0),
            dict(aod550=0.6, albedo=0.3),
        ):
            self.assertEqual(a, key(variant(self.base, **change)), change)
        self.assertNotEqual(a, key(variant(self.base, scale_height_km=2.4)))

    def test_layer_off_has_no_lock(self):
        self.assertIsNone(attach_lock_name(variant(self.base, status='disabled')))
        self.assertIsNone(attach_lock_name(variant(self.base, stratosphere_tau550=0.0)))

    def test_never_finer_than_geometry_key(self):
        """Equal _geometry_key (fixed synthetic grid) must imply equal lock name."""
        geo = types.SimpleNamespace(
            h=np.arange(4.0),
            chi=np.arange(3.0),
            cfg=dict(nmu=1, nphi=2, path_n=4),
            sun=types.SimpleNamespace(h=np.arange(2.0), x=np.arange(3.0), col=np.ones((2, 3))),
        )
        states = [s for s in STATES if attach_lock_name(s) is not None]
        states += [
            variant(self.base, 21, 3, aod550=0.9, albedo=0.0),
            variant(self.base, 22.0, 5.0, stratosphere_tau550=0.2),
        ]
        states += [
            variant(self.base, **{k: v}) for k, v in (('ssa', 0.001), ('g', 0.0), ('angstrom', 0.5), ('ozone_du', 0.0))
        ]
        names = {}
        for state in states:
            key = _geometry_key(types.SimpleNamespace(geo=geo, fuse_state=state), None, profile_config(state))
            names.setdefault(key, set()).add(attach_lock_name(state))
        self.assertTrue(all(len(v) == 1 for v in names.values()), names)
        self.assertEqual(len(names), 2)  # 21 km s3 and 22 km s5 in the rig set


if __name__ == '__main__':
    unittest.main(verbosity=2)
