"""Accepted cell formulas, actual spawn failures, and cgroup quota ancestry."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
from sosfuse2.precompute import column_parallel as cp, ground_parallel as gp, host_cache
from sosfuse2.atmosphere import density, stratosphere as st
from sosfuse2.solve import ray_geometry
from sosfuse2.foundation import host_limits as hl

from sosfuse2.checks.numerical_checks import assert_equivalent


def fail_cell(task):
    if isinstance(task, int) and task == 1 or isinstance(task, tuple) and len(task[0]) < 8192:
        raise RuntimeError('E_TEST_GROUND_WORKER')
    return gp._run(task)


def accepted(s, scope, kind, optics=None, profile=None, suncol=None, method=None):
    # method: host_cache.prepare_ground passes density.sun_method(); this GL24 reference is the default only.
    assert method in (None, 'gl'), method
    old = scope.b.sh.old
    rows = []
    for cell in range(s.geo.ns):
        ih, ic = divmod(cell, len(s.geo.chi))
        r = 6371.0 + s.geo.h[ih]
        ch = s.geo.chi[ic]
        loc, w = scope.b.sh.incoming(r, ch, 1)
        p, up, ex = old.probe.frame(np.array(r), ch)
        ey = np.cross(up, ex)
        d = loc[:, 0, None] * ex + loc[:, 1, None] * ey + loc[:, 2, None] * up
        p = np.broadcast_to(p, d.shape)
        end = old.probe.boundaries(p, d)[:, -1]
        q = p + end[:, None] * d
        re = np.linalg.norm(q, axis=-1)
        col = old.total_columns(p, d, 24)
        if optics is not None:
            col = density.correct_columns(p, d, col, old, optics, 24)
        sc, lit = s.geo.sun.lookup(q, np.maximum(6371.0, re))
        ec = np.arccos(np.clip(q[:, 2] / re, -1, 1))
        if kind == 'host':
            rows.append((loc, w, q, re, col[:, [0, 2, 3, 4]], (np.cos(ec) > 0) & lit & (re < 6371.0 + 1e-5) & (w > 0)))
        else:
            column = st.columns(p, d, old, profile, 24)
            ssc, _ = st.sun_lookup(q, np.maximum(6371.0, re), s.geo.sun, suncol)
            rows.append((loc, w, sc, lit, col, column, ssc, ec, re))
    return rows


class GroundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        with mock.patch.dict(os.environ, {'SOSFUSE2_SUN_WORKERS': '1', gp.ENV: '1'}):
            s, cls.scope, _ = ray_geometry.create(Path(cls.tmp.name) / 'prepare', 'selftest', [550.0], 'numpy')
        # Real incoming quadratures at illuminated, terminator, shadow and altitude cells.
        s.geo.h = np.array([0.1, 25.0])
        s.geo.chi = np.deg2rad([80.0, 93.0])
        s.geo.ns = 4
        cls.s = s
        cls.old = cls.scope.b.sh.old
        cls.optics = density.build(dict(density=dict(density_profile='usstd76', ozone_profile='usstd76', rayleigh_column='keep')))
        cls.profile = st.profile_config({})
        cls.suncol = np.zeros(s.geo.sun.col.shape[:2])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check_close(self, a, b):
        self.assertEqual(len(a), len(b))
        for ra, rb in zip(a, b):
            for x, y in zip(ra, rb):
                assert_equivalent(x, y)

    def mapped(self, call, workers, fail=False):
        log = io.StringIO()
        with mock.patch.dict(os.environ, {gp.ENV: str(workers)}), contextlib.redirect_stdout(log):
            if fail:
                with mock.patch.object(gp, '_run', fail_cell):
                    value = call()
            else:
                value = call()
        events = [json.loads(l) for l in log.getvalue().splitlines() if '"ground_parallel"' in l]
        if workers > 1:
            self.assertEqual(events[-1]['result'], 'fallback' if fail else 'parallel')
            if fail:
                self.assertEqual(events[-1]['reason'], 'E_TEST_GROUND_WORKER')
        return value

    def test_cells(self):
        for kind, optics in (('host', self.optics), ('strato', None), ('strato', self.optics)):
            args = (self.s, self.scope, kind, optics, self.profile, self.suncol)
            expected = accepted(*args)
            for count, fail in ((1, False), (2, False), (2, True)):
                self.check_close(expected, self.mapped(lambda: gp.cells(*args), count, fail))

    def test_aerosol_chunks(self):
        p = np.broadcast_to([0.0, 0.0, 6372.0], (8193, 3))
        d = np.broadcast_to([0.0, 0.0, -1.0], p.shape)
        expected = np.concatenate(
            [self.old.total_columns(p[i : i + 8192], d[i : i + 8192], 24)[:, 1] for i in range(0, len(d), 8192)]
        )
        for count, fail in ((1, False), (2, False), (2, True)):
            value = self.mapped(lambda: gp.aerosol(self.scope, self.s.geo.sun, p, d), count, fail)
            assert_equivalent(expected, value)

    def test_host_outputs(self):
        def run():
            with (
                mock.patch.object(host_cache, 'context', return_value=(Path(self.tmp.name), {})),
                mock.patch.object(host_cache, 'cached', side_effect=lambda folder, key, build, **kw: build()),
            ):
                host_cache.prepare_ground(self.s, self.scope, self.optics)
            return [getattr(self.s, k).copy() for k in ('gm', 'goff', 'gdir', 'gval')]

        with mock.patch.object(gp, 'cells', accepted):
            expected = self.mapped(run, 1)
        actual = self.mapped(run, 2)
        self.check_close([expected], [actual])

    def test_strato_entry_and_noprefix(self):
        with (
            mock.patch.dict(os.environ, {'STRATO_ATTACH_CACHE': '', 'SOSFUSE2_STRATO_HOST_NOPREFIX': '1'}),
            mock.patch.object(st, 'solar_grid', return_value=self.suncol),
        ):
            expected = accepted(
                self.s, self.scope, 'strato', getattr(self.s, 'density_optics', None), self.profile, self.suncol
            )
            _, actual = self.mapped(lambda: st._ground_geometry(self.s, self.scope, self.old, self.profile), 2)
            self.check_close(expected, actual)

    def test_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'a/b').mkdir(parents=True)
            (root / 'cpu.max').write_text('800000 100000')
            (root / 'a/cpu.max').write_text('250000 100000')
            (root / 'a/b/cpu.max').write_text('max 100000')
            mount = f'1 0 0:1 / {root} rw - cgroup2 cgroup rw'
            self.assertEqual(min(cp._quota_limits('0::/a/b', mount)), 2)
            self.assertEqual(cp._quota_limits('0::/', mount.replace(' / ', ' /a ')), [8])
            # default_workers lives in host_limits (re-exported by column_parallel); patch its own module globals.
            with (
                mock.patch.object(hl.os, 'sched_getaffinity', return_value=set(range(4)), create=True),
                mock.patch.object(hl.os, 'cpu_count', return_value=28),
                mock.patch.object(hl.Path, 'read_text', side_effect=['0::/a/b', mount]),
                mock.patch.object(hl, '_quota_limits', return_value=[2]),
            ):
                self.assertEqual(cp.default_workers(), 2)
            with (
                mock.patch.object(hl.os, 'sched_getaffinity', return_value=set(range(4)), create=True),
                mock.patch.object(hl.os, 'cpu_count', return_value=28),
                mock.patch.object(hl.Path, 'read_text', side_effect=['0::/a/b', mount]),
                mock.patch.object(hl, '_quota_limits', return_value=[8]),
            ):
                self.assertEqual(cp.default_workers(), 4)
        with mock.patch.dict(os.environ, {gp.ENV: ''}), mock.patch.object(gp, 'default_workers', return_value=3):
            self.assertEqual(gp.workers(), 3)
        for value in ('0', '-1', 'invalid'):
            with mock.patch.dict(os.environ, {gp.ENV: value}), self.assertRaisesRegex(ValueError, 'E_GROUND_WORKERS'):
                gp.workers()


def test_ground_values_keep_dark_cells_and_spectral_underflow():
    from types import SimpleNamespace

    n = 67
    # One empty block, a partly visible cell, and a lane below the FP32 range.
    chi = np.array([np.pi, np.pi, 0.])
    loc = np.tile(np.eye(3), (n, 1))
    q = 6371. * np.column_stack((np.sin(chi), np.zeros(3), np.cos(chi)))
    q = np.tile(q, (n, 1))
    q[:64*3, 2] = -6371.
    r = np.full(3*n, 6371.)
    w = np.tile([0.2, 0.3, 0.5], n)
    columns = np.full((n*3, 4), 2.)
    z = dict(q=q, r=r, loc=loc, w=w, off=np.arange(0, 3*n+1, 3),
             columns=columns, candidate=q[:, 2] > 0)
    def lookup(points, radius):
        return np.ones((len(points), 5)), np.ones(len(points), bool)
    s = SimpleNamespace(geo=SimpleNamespace(ns=n, sun=SimpleNamespace(lookup=lookup)),
                        W=2, ext=np.tile([0.1, 1000.], (5, 1)))
    scope = SimpleNamespace(C=0.25, b=SimpleNamespace(R=6371., sh=SimpleNamespace(NORMAL={'albedo': 0.2})))
    host_cache._ground_values(s, scope, z, np.full(3, 2.))
    np.testing.assert_array_equal(s.goff, np.r_[np.zeros(65, int), 1, 2, 3])
    np.testing.assert_array_equal(s.gdir, loc[z['candidate']].astype('f4'))
    value = (0.2 / np.pi * np.exp(-15. * np.array([0.1, 1000.])) * 0.5)
    np.testing.assert_allclose(s.gval, np.tile(value, (3, 1)), rtol=1e-6)
    np.testing.assert_array_equal(s.gm[:64], 0.)
    np.testing.assert_allclose(s.gm[64:, 0], np.tile(0.5 * value * (2 * scope.C * s.ext[0]), (3, 1)), rtol=1e-6)
