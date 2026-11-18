"""Worker spawn, prepared payload and observer roles for extended domains."""

import multiprocessing
import collections
import json
import re
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace as NS

import numpy as np
import pytest

from sosfuse2.tests.test_extended_top import small_solver, serial  # noqa: F401
from sosfuse2.foundation.top import Domain
from sosfuse2.foundation import paths
from sosfuse2.precompute import column_parallel as cp, sun_parallel as sp, ground_parallel as gp
from sosfuse2.solve import ray_geometry as rg, ray_geometry_parallel as rp, source_solver as ss


def test_boundary_literal_inventory():
    """Exact file/line patterns and multiplicities; additions require classification."""
    pattern = re.compile(r'\b(80|79|6451|161|321|641|159|319|639)\b|TOP_KM|LEGACY_COLUMN|PROFILE_INV')
    allowed = json.loads((paths.PKG / 'tests/boundary_literals.json').read_text())
    unexpected = {}
    for path in (*paths.production_files(), paths.KERNEL_CU):
        hits = collections.Counter(line.strip() for line in path.read_text().splitlines() if pattern.search(line))
        extra = hits - collections.Counter(allowed.get(str(path.relative_to(paths.PKG)), {}))
        if extra:
            unexpected[str(path.relative_to(paths.PKG))] = dict(extra)
    assert not unexpected


def worker_values(column, sun, ground, geometry, geo, optics, sun_data):
    """Use the same initializer/run pairs in the parent and a fresh spawn process."""
    p = np.array([[0., 0., 6371. + 79.]])
    d = np.array([[0., 0., 1.]])
    cp._init(column, 'density', p, np.array([0, 1]), optics, cp.SOURCE_HASHES)
    col = cp._run((0, 1, d, np.zeros((1, 5))))[1][0]
    sp._init(sun, np.array([80., 120.]), np.array([.5, 1.]), 6371., 24, None, 'gl', cp.SOURCE_HASHES)
    sun_value = sp._run((1, None))[1]
    gp._init(ground, sun_data, 'host', cp.SOURCE_HASHES)
    ground_value = gp._run(0)
    rp._init(geometry, geo, False, optics, None, cp.SOURCE_HASHES)
    worker = rp._GEOMETRY.scope.b.sh.old
    return dict(column=col, sun=sun_value, ground=ground_value,
                ray=worker.total_columns(p, d, 24),
                domains=[cp._WORKER['old'].domain, sp._STATE['old'].domain,
                         gp._STATE[0].domain, worker.domain])


def test_spawn_and_payload(tmp_path, monkeypatch):
    from sosfuse2.jobs import pipeline_execution as pe

    s, ctx = small_solver(tmp_path)
    old, sun = ctx.b.sh.old, s.geo.sun
    d = s.kernel.domain
    optics = {k: v for k, v in s.density_optics.items() if k != 'old'}
    geometry = rg.TransientGeometry(s, ctx)
    frozen = (cp.freeze(old, ['geometry_segments', 'probe']), cp.freeze(old, ['profiles', 'probe']),
              cp.freeze(gp._functions(ctx, sun), list(vars(gp._functions(ctx, sun)))),
              rp.freeze(NS(scope=ctx)))
    geo = NS(h=s.geo.h, chi=s.geo.chi, cfg=s.geo.cfg)
    sun_data = (geo.h, geo.chi, ctx.b.R, NS(h=sun.h, x=sun.x, col=sun.col, domain=d),
                optics, None, None, 'exact')
    args = (*frozen, geo, optics, sun_data)
    monkeypatch.setattr(cp, '_WORKER', {})
    monkeypatch.setattr(sp, '_STATE', {})
    monkeypatch.setattr(gp, '_STATE', None)
    monkeypatch.setattr(gp, '_SOURCE_DRIFT', None)
    monkeypatch.setattr(rp, '_GEOMETRY', None)
    monkeypatch.setattr(rp, '_SOURCE_DRIFT', None)
    expected = worker_values(*args)
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context('spawn')) as pool:
        actual = pool.submit(worker_values, *args).result(timeout=60)
    assert actual['domains'] == expected['domains'] == [d] * 4
    for name in ('column', 'sun', 'ray'):
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-3, atol=0)
    for a, b in zip(actual['ground'], expected['ground']):
        np.testing.assert_allclose(a, b, rtol=1e-3, atol=0)
    original = ss.Kernel

    def cpu_kernel(threads, height, g, backend, block_size, **kw):
        return original(threads, height, g, 'numpy', block_size, **kw)

    monkeypatch.setattr(ss, 'Kernel', cpu_kernel)
    monkeypatch.setattr(pe, 'promote_bundle', lambda bundle, xp: bundle)
    pe.promote_solver(s, geometry, 'numpy', 128, 1)
    payload = pe.device_payload(s, geometry)
    restored, _ = pe.restore_device_payload(payload, ({}, {}))
    assert payload['domain'] == restored.kernel.domain == d
    assert restored.kernel.domain.defines() == d.defines()
    with pytest.raises(RuntimeError, match='E_PROFILE_N_MISMATCH'):
        pe.install_constants(restored.kernel, restored, {1: dict(domain=Domain())})


def test_observer_stop_acceleration_and_record(monkeypatch):
    """An isolated observer perturbation must enter the appropriate solve decisions."""
    vectors = []
    original = np.concatenate

    def capture(parts, *args, **kw):
        result = original(parts, *args, **kw)
        if len(parts) == 2 and np.ndim(parts[0]) == np.ndim(parts[1]) == 1:
            vectors.append(np.array(result))
        return result

    monkeypatch.setattr(np, 'concatenate', capture)
    from sosfuse2.atmosphere.stratosphere import source_size

    for T, index, should_converge in ((80, 4, False), (120, 5, False), (120, 6, True)):
        k = ss.Kernel(backend='numpy', L=16, domain=Domain(T, .5, (120.,) if T > 80 else (), 'vacuum' if T > 80 else None))
        s = NS(kernel=k, geo=NS(ns=1), W=1)
        wrap = ss.Solver(s, None, prepared=True)
        calls = iter(range(8))

        def boxes(x):
            out = np.ones((5 if T == 80 else 7, 7, 1))
            out[index] = 1 if next(calls) % 2 == 0 else 2
            return out

        wrap.observer = boxes
        wrap.apply = lambda x: np.ones_like(x)
        vectors.clear()
        if should_converge:
            _, meta = wrap.solve(max_eval=2, tolerance=.1)
            assert meta['history'][-1]['record_observer_relative'] == [[.5]]
            assert meta['history'][-1]['observer_relative'] == [0.]
        else:
            with pytest.raises(RuntimeError, match='E_NOT_CONVERGED'):
                wrap.solve(max_eval=2, tolerance=.1)
        n = source_size(s)
        count = 35 if T == 80 else 42
        assert vectors[0].size == n + count
        tail = vectors[0][n:]
        if should_converge:
            assert np.all(tail == 0)
        else:
            np.testing.assert_allclose(tail[-7:], .5 * np.sqrt(n / count), rtol=1e-12)
            assert np.all(tail[:-7] == 0)
