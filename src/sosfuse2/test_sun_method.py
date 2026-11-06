"""③ stage 0(b): SOSFUSE2_SUN_METHOD (density.sun_method) = exact PROFILE columns of the sun table and first-ground rays.

CPU only; every cache root is a temporary folder. The reference is GL64/GL128 between the PROFILE knot crossings and the
tangent point (no antiderivative), as measured in a larger sweep (observed maxima).
"""
import contextlib
import copy
import io
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

import numpy as np
from .numerical_checks import assert_equivalent

from . import density, geometry, ground_parallel as gp, host_cache, stratosphere as st, sun_parallel as sp
from .strato_states import build as strato_states

R = 6371.
SERIAL = {sp.ENV: '1', gp.ENV: '1', 'SOSFUSE2_COLUMN_WORKERS': '1', 'SOSFUSE2_GEOMETRY_WORKERS': '1'}
USSTD = dict(density_profile='usstd76', ozone_profile='usstd76', rayleigh_column='usstd')


def reference(p, d, end, table, n):
    z = density.grid(len(table)); sa = float(p@d); rho = float(np.linalg.norm(np.cross(p, d)))
    cuts = [0., end]+([-sa] if 0 < -sa < end else [])
    c2 = (R+z)**2-rho*rho; c = np.sqrt(c2[c2 > 0])
    cuts = np.unique(cuts+[t for t in np.r_[-sa-c, -sa+c] if 0 < t < end])
    if len(cuts) < 2: return np.zeros(4)
    x, w = np.polynomial.legendre.leggauss(n); x = (x+1)/2; w = w/2
    lo, hi = cuts[:-1], cuts[1:]; h = np.hypot(rho, sa+lo[:, None]+(hi-lo)[:, None]*x)-R
    vals = [np.interp(h, z, table[:, k]) for k in (0, 1, 2)]
    vals.append((vals[0]/table[0, 0])**2)
    return np.array([np.sum(np.sum(v*w, 1)*(hi-lo)) for v in vals])


def boundary_rays():
    """x=0 (mu=mg+1e-9), top zero length, vertical (rho=0), knot tangents both sides, ground grazing/ends, short paths."""
    rows = []
    def add(h0, cz):rows.append(([0., 0., R+h0], [math.sqrt(max(1-cz*cz, 0.)), 0., cz]))
    for h0 in (0., .1, 40., 80.):
        r = R+h0; mg = -math.sqrt(max(0., 1-(R/r)**2)); add(h0, mg+1e-9); add(h0, mg+1e-6)
        if h0 > 0:add(h0, mg-1e-9); add(h0, mg-1e-5)
    for cz in (1., -1.):add(30., cz)
    add(80., .3); add(80., 1.); add(80.-1e-7, 1.); add(79.999, .5); add(5e-4, -1.); add(5e-4, -.3)
    for ht in (25., 10.5, .5):
        for off in (0., 1e-7, -1e-4):
            c = (R+ht+off)/(R+80.); add(80., -math.sqrt(1-c*c))
    for cz in (1e-8, 3.9e-7, 1e-6, 1e-4):add(0., cz)   # leaving the ground almost horizontally
    p, d = (np.array(v, 'f8') for v in zip(*rows))
    return p, d


def external641():
    z = density.grid(641); base = density.build(dict(density=dict(USSTD, profile_step_km=.125)))
    oz = base['table'][:, 1]*2.3+.4*np.exp(-.5*((z-31.)/1.5)**2); oz[::37] *= 1.6
    return density.build(density.with_profile(dict(id='t', parameters={}), air_surface_number_density_m3=2.5469e25, z_km=z,
                                              table=np.column_stack([base['table'][:, 0]*(1+.05*np.sin(z/3.)), oz]),
                                              mask=3, normalize_ozone=False, scale=1.0))


class SunMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.root = Path(cls.tmp.name)
        with mock.patch.dict(os.environ, SERIAL), mock.patch.object(host_cache, 'ROOT', cls.root/'hc'):
            os.environ.pop(density.SUN_METHOD_ENV, None)
            s, cls.scope, _ = geometry.create(cls.root/'prepare', 'selftest', [550.], 'numpy')
        cls.old = cls.scope['b'].sh.old
        s.geo.h = np.array([0., .1, 25.]); s.geo.chi = np.deg2rad([0., 80., 93.]); s.geo.ns = 9; cls.s = s
        cls.optics = {m: dict(density.build(dict(density=dict(USSTD, density_profile='usstd76' if m & 1 else 'exp8',
                                                              ozone_profile='usstd76' if m & 2 else 'triangle'))), old=cls.old)
                      for m in (1, 2, 3)}
        cls.fine = dict(density.build(dict(density=dict(USSTD, profile_step_km=.125))), old=cls.old)
        cls.external = dict(external641(), old=cls.old)
        cls.h = np.array([0., .1, 13., 40., 80.]); cls.x = np.linspace(0, 1, 21)
        with mock.patch.dict(os.environ, SERIAL):
            cls.base = sp.rows(cls.old, cls.h, cls.x, R, 32)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def rows(self, optics, method, workers=1):
        log = io.StringIO()
        with mock.patch.dict(os.environ, {sp.ENV: str(workers)}), contextlib.redirect_stdout(log):
            out = sp.rows(self.old, self.h, self.x, R, 32, self.base, optics, method)
        return out, [json.loads(l) for l in log.getvalue().splitlines()]

    def test_1_switch_values(self):
        for value, want in ((None, 'gl'), ('', 'gl'), (' gl ', 'gl'), ('exact', 'exact')):
            with mock.patch.dict(os.environ, {} if value is None else {density.SUN_METHOD_ENV: value}):
                if value is None:os.environ.pop(density.SUN_METHOD_ENV, None)
                self.assertEqual(density.sun_method(), want)
        for value in ('GL', 'gl32', 'exact '+'x'):
            with mock.patch.dict(os.environ, {density.SUN_METHOD_ENV: value}), self.assertRaisesRegex(ValueError, 'E_SUN_METHOD'):
                density.sun_method()
        with self.assertRaisesRegex(ValueError, 'E_SUN_METHOD'):self.rows(self.optics[3], 'exact32')
        with self.assertRaisesRegex(ValueError, 'E_SUN_METHOD'):gp.cells(self.s, self.scope, 'host', self.optics[3], method='GL')

    def test_2_exact_against_independent_reference_at_boundaries(self):
        p, d = boundary_rays(); end = self.old.probe.boundaries(p, d)[:, -1]
        self.assertTrue((end == 0).any())   # the top upward rays
        for optics, bound in ((self.optics[3], 1e-9), (self.external, 1e-6)):
            ex = density.exact_profile_columns(p, d, end, optics); t = np.asarray(optics['table'])
            for i in range(len(p)):
                r64, r128 = reference(p[i], d[i], end[i], t, 64), reference(p[i], d[i], end[i], t, 128)
                self.assertLess(np.abs(r64-r128).max(), 1e-9)
                np.testing.assert_allclose(ex[i], r128, rtol=0, atol=bound, err_msg=str((p[i], d[i], end[i])))
            self.assertTrue(np.all(ex[end == 0] == 0))

    def test_3_mask_columns_scale_and_refined_grid(self):
        for mask, optics in self.optics.items():
            out, _ = self.rows(optics, 'exact')
            gl, _ = self.rows(optics, 'gl')
            for col, bit in ((0, 1), (1, 0), (2, 2)):
                if mask & bit:self.assertFalse(np.array_equal(out[..., col], gl[..., col]))
                else:assert_equivalent(out[..., col], self.base[..., col])
        # 161 and 641 nodes of the same piecewise-linear function (asset linear refinement): equal exact columns.
        a, _ = self.rows(self.optics[3], 'exact'); b, _ = self.rows(self.fine, 'exact')
        np.testing.assert_allclose(a[..., [0, 2]], b[..., [0, 2]], rtol=0, atol=1e-8)
        # External unnormalized table: the values are used as given (no rescaling).
        p, d = boundary_rays(); end = self.old.probe.boundaries(p, d)[:, -1]
        doubled = dict(self.external, table=np.asarray(self.external['table'])*2)
        np.testing.assert_allclose(density.exact_profile_columns(p, d, end, doubled),
                                   density.exact_profile_columns(p, d, end, self.external)*np.array([2,2,2,1]), rtol=1e-13, atol=0)

    def test_4_sun_rows_parallel_serial_fallback(self):
        expected, events = self.rows(self.external, 'exact'); self.assertEqual(events, [])
        actual, events = self.rows(self.external, 'exact', 2)
        self.assertEqual(events[0]['result'], 'parallel'); assert_equivalent(actual, expected)
        def fail(task):
            if task[0] == 2:raise RuntimeError('E_TEST_SUN_WORKER')
            return sp._run(task)
        with mock.patch.object(sp, '_run', fail):actual, events = self.rows(self.external, 'exact', 2)
        self.assertEqual(events[0]['result'], 'fallback'); assert_equivalent(actual, expected)
        with mock.patch.dict(os.environ, {density.SUN_METHOD_ENV: 'exact'}):   # omitted method = the switch
            assert_equivalent(self.rows(self.external, None)[0], expected)

    def test_5_ground_cells_host_and_strato(self):
        profile = st.profile_config({}); suncol = np.zeros(self.s.geo.sun.col.shape[:2])
        def run(kind, method, workers):
            with mock.patch.dict(os.environ, {gp.ENV: str(workers)}), contextlib.redirect_stdout(io.StringIO()):
                return gp.cells(self.s, self.scope, kind, self.external, profile, suncol, method=method)
        for kind in ('host', 'strato'):
            exact = run(kind, 'exact', 1); gl = run(kind, 'gl', 1)
            for a, b in zip(exact, run(kind, 'exact', 2)):
                for x, y in zip(a, b):assert_equivalent(x, y)
            for cell, (e, g) in enumerate(zip(exact, gl)):
                cols = e[4] if kind == 'host' else e[4][:, [0, 2, 3, 4]]   # host: columns (n,2); strato: col (n,3)
                ih, ic = divmod(cell, len(self.s.geo.chi)); r = R+self.s.geo.h[ih]
                loc = e[0]; p, up, ex = self.old.probe.frame(np.array(r), self.s.geo.chi[ic]); ey = np.cross(up, ex)
                d = loc[:, 0, None]*ex+loc[:, 1, None]*ey+loc[:, 2, None]*up; p = np.broadcast_to(p, d.shape)
                end = self.old.probe.boundaries(p, d)[:, -1]
                assert_equivalent(cols, density.exact_profile_columns(p, d, end, self.external))
                for i in range(0, len(d), 97):
                    np.testing.assert_allclose(cols[i], reference(p[i], d[i], end[i], np.asarray(self.external['table']), 128),
                                               rtol=0, atol=1e-6)
                if kind == 'strato':
                    assert_equivalent(e[4][:, 1], g[4][:, 1])   # aerosol column
                    for k in (2, 3, 5, 6, 7, 8):assert_equivalent(e[k], g[k])   # sc, lit, st, ssc, ec, re

    def test_6_create_paths_and_cache_keys(self):
        tokens = []; original = host_cache.cached
        def record(folder, key, build, compressed=False):
            tokens.append((key['kind'], host_cache.digest(key), key.get('sun_method')))
            return original(folder, key, build, compressed)
        def create(state, method, cache, name, forbid=False):
            env = dict(SERIAL, SOSFUSE2_HOST_CACHE=cache, **{density.SUN_METHOD_ENV: method})
            patches = [mock.patch.dict(os.environ, env), mock.patch.object(host_cache, 'ROOT', self.root/'keys'),
                       mock.patch.object(host_cache, 'cached', record), contextlib.redirect_stdout(io.StringIO())]
            if forbid:patches += [mock.patch.object(sp, 'rows', side_effect=AssertionError('E_TEST_SUN_MISS')),
                                  mock.patch.object(gp, 'cells', side_effect=AssertionError('E_TEST_GROUND_MISS'))]
            with contextlib.ExitStack() as stack:
                for patch in patches:stack.enter_context(patch)
                s, _, _ = geometry.create(self.root/name, 'selftest', [550.], 'numpy', state=state)
            return {k: getattr(s, k) for k in ('gm', 'goff', 'gdir', 'gval', 'ext', 'hr')} | dict(sun=s.geo.sun.col,
                                                                                             method=s.density_sun_method)
        def same(a, b, keys=('sun', 'gm', 'goff', 'gdir', 'gval', 'ext', 'hr')):
            for k in keys:assert_equivalent(a[k], b[k])
        normal = dict(geometry.normal_state(), density=USSTD)
        gl = create(normal, 'gl', '1', 'gl_cold'); n = len(tokens)
        exact = create(normal, 'exact', '1', 'exact_cold')
        self.assertEqual((gl['method'], exact['method']), ('gl', 'exact'))
        kinds = {k: {t for kk, t, _ in tokens if kk == k} for k in ('sun02', 'ground', 'ground_aerosol01')}
        self.assertEqual((len(kinds['sun02']), len(kinds['ground']), len(kinds['ground_aerosol01'])), (2, 2, 1))
        self.assertFalse(np.array_equal(gl['sun'], exact['sun'])); same(gl, exact, ('ext', 'hr'))
        assert_equivalent(gl['sun'][..., 1], exact['sun'][..., 1])
        same(gl, create(normal, 'gl', '1', 'gl_warm', forbid=True))
        same(exact, create(normal, 'exact', '1', 'exact_warm', forbid=True))
        same(exact, create(normal, 'exact', '0', 'exact_uncached'))   # density.attach's own sun loop and total_columns patch
        self.assertEqual(len(tokens), 4*n)   # two cold + two warm cached runs; the uncached run asks no cache
        strato = dict(strato_states()['states'][5], density=USSTD)
        same(create(strato, 'exact', '0', 'strato_uncached'), create(strato, 'exact', '1', 'strato_cached'))

    def test_7_strato_attach_key_separates_methods(self):
        geo = NS(h=np.arange(4.), chi=np.arange(3.), cfg=dict(nmu=1, nphi=2, path_n=4),
                 sun=NS(h=np.arange(2.), x=np.arange(3.), col=np.ones((2, 3))))
        state = strato_states()['states'][5]; profile = st.profile_config(state)
        keys = {}
        for density_optics in (None, self.optics[3]):
            for method in ('gl', 'exact'):
                s = NS(geo=geo, fuse_state=state) if density_optics is None else NS(geo=geo, fuse_state=state, density_optics=density_optics)
                with mock.patch.dict(os.environ, {density.SUN_METHOD_ENV: method}):
                    keys[density_optics is None, method] = st._geometry_key(s, None, profile)
        self.assertEqual(keys[True, 'gl'], keys[True, 'exact'])     # no PROFILE: the method reads nothing
        self.assertNotEqual(keys[False, 'gl'], keys[False, 'exact'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
