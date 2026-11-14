"""S2 host physics and interface rules; no network, GPU, or redistributed CIA data."""

import copy
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sosfuse2.atmosphere import components as C
from sosfuse2.atmosphere import density as D
from sosfuse2.asset_tools import generate_minor_absorbers as G
from sosfuse2.atmosphere import minor_absorbers as M
from sosfuse2.atmosphere import rayleigh as R
from sosfuse2.atmosphere import spectral as SP
from sosfuse2.checks import testkit


@pytest.fixture
def asset(tmp_path, monkeypatch):
    path = testkit.synthetic_o4(tmp_path)
    monkeypatch.setattr(M, 'o4_path', lambda: path)
    return path


def coeff(state, wl=(450.0, 550.0, 577.0), sigma=None):
    from functools import partial

    from sosfuse2.atmosphere.extinction import coefficients

    wl = np.array(wl, dtype='f8')
    opt = D.build(state)
    block = SP.block(
        dict(spectral_mode='point', bin_edges_nm=np.r_[360.0, (wl[:-1] + wl[1:]) / 2, 840.0].tolist()), wl, False
    )
    spec = SP.spec_table(block, wl, state)
    if sigma is not None:
        spec.update(sigma)
    params = dict(
        aer=0.1, ang=1.3, ozone_DU=280.0, **C.optics_state(state, opt), no2_H_f_km=opt['meta']['no2']['H_f_km']
    )
    fn = partial(coefficients, spec)
    return fn(wl, params), opt


def test_loader_normal_and_sampling(asset):
    from sosfuse2.foundation.content_identity import asset_digest

    published_raw_sha = '23bdb5fb1f8a2e99f968ad2d3ce8395c9e3190c0aa8fb2e30289beea37f0fb9e'
    no2 = M.load('no2')
    assert no2['raw_sha256'] == published_raw_sha  # Byte-integrity pin of the published NO2 asset.
    assert no2['sha256'] == asset_digest(M.NO2.name, published_raw_sha)  # Computation identity.
    assert M.load('o4')['values'].dtype == np.float64
    w = np.array(SP.PRODUCT['wavelengths_nm'])
    b = SP.block(dict(spectral_mode='band', bin_edges_nm=SP.PRODUCT_EDGES), w, True)
    spec = SP.spec_table(b, w)
    for key in ('no2', 'o4'):
        a = M.load(key)
        expected = []
        for lo, hi in zip(SP.PRODUCT_EDGES[:-1], SP.PRODUCT_EDGES[1:]):
            x = np.r_[lo, a['wavelength_nm'][(a['wavelength_nm'] > lo) & (a['wavelength_nm'] < hi)], hi]
            expected.append(np.trapezoid(np.interp(x, a['wavelength_nm'], a['values']), x) / (hi - lo))
        np.testing.assert_allclose(spec[key], expected, rtol=1e-13)
    point = SP.block(dict(spectral_mode='point', bin_edges_nm=SP.PRODUCT_EDGES), w, False)
    assert point['point_definition']['o3'] == 'point'
    assert point['point_definition']['no2'] == '0.1 nm cell-mean interpolant'
    np.testing.assert_array_equal(
        SP.spec_table(point, w)['o4'], np.interp(w, M.load('o4')['wavelength_nm'], M.load('o4')['values'])
    )


@pytest.mark.parametrize('bad', ['grid', 'negative', 'units', 'format'])
def test_loader_rejects(asset, bad):
    doc = json.loads(asset.read_text())
    if bad == 'grid':
        doc['wavelength_nm'][1] += 0.001
    elif bad == 'negative':
        doc['o4'][123] = -1e-56
    elif bad == 'units':
        doc['units']['o4'] = 'cm^5'
    else:
        doc['format'] = 'unknown'
    asset.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='E_O4_ASSET_INVALID'):
        M.load('o4')
    doc = json.loads(testkit.synthetic_o4(asset.parent.parent).read_text())
    doc['o4'][123] = float('nan')
    asset.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='E_O4_ASSET_INVALID'):
        M.load('o4')


def test_loader_missing(asset):
    asset.unlink()
    assert M.load('o4', required=False) is None
    with pytest.raises(ValueError, match='E_O4_ASSET_MISSING:.*prepare_minor_absorbers'):
        M.load('o4')
    # O4 off needs no cache, while the default solver spectrum does.
    coeff(dict(parameters=dict(o4=False)))
    with pytest.raises(ValueError, match='E_O4_ASSET_MISSING'):
        coeff({})


def test_loader_unreadable_parent_is_missing(asset, monkeypatch):
    """Path.is_file() raises PermissionError under an unreadable parent before Python 3.14; the loader treats it as a
    missing asset and keeps the error code."""
    import pathlib

    def denied(self):
        raise PermissionError(13, 'Permission denied', str(self))

    monkeypatch.setattr(pathlib.Path, 'is_file', denied)
    assert M.load('o4', required=False) is None
    with pytest.raises(ValueError, match='E_O4_ASSET_MISSING'):
        M.load('o4')


@pytest.mark.parametrize('field', ['rules_version', 'sha256'])
def test_loader_stale(asset, field):
    doc = json.loads(asset.read_text())
    doc['generator' if field == 'rules_version' else 'source'][field] = 'old'
    asset.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='E_O4_ASSET_STALE'):
        M.load('o4')


@pytest.mark.parametrize('missing,code', [(0.0, 'E_SOURCE_GAP'), (-1.0, 'E_SOURCE_SENTINEL'), (-999.0, 'E_SOURCE_GAP')])
def test_generator_missing_rejections(tmp_path, missing, code):
    # Replacement for the original escalation probe: now rejection is required.
    data = np.full((5, 7), 1e-19)
    data[:, 0] = 1e7 / np.arange(700.0, 705.0)
    data[2, 5] = missing
    path = tmp_path / 'no2cp_97.dat'
    np.savetxt(path, data, fmt='%.17g')
    with pytest.raises(SystemExit, match=code):
        G.bira(path, 6)
    if missing == -999.0:
        data[2, 2] = missing
        path = tmp_path / 'no2c_97.dat'
        np.savetxt(path, data, fmt='%.17g')
        with pytest.raises(SystemExit, match=code):
            G.bira(path, 3)


def test_generator_end_splice_and_45_integrals():
    x, y = G.contiguous(np.arange(5.0), [0, 2e-19, 3e-19, 0, 0], G.MISSING['no2cp_97.dat'], 'synthetic')
    np.testing.assert_array_equal(x, [1, 2])
    a = G.Source([359.0, 660.0, 666.0], [1.0, 2.0, 2.0])
    b = G.Source([650.0, 666.0, 841.0], [2.0, 2.0, 3.0])
    pieces = G.no2_pieces_of(a, b)
    grid = SP.asset()['wl']
    values = G.piecewise_cells(grid, pieces)
    for lo, hi in zip(SP.PRODUCT_EDGES[:-1], SP.PRODUCT_EDGES[1:]):
        # Independent integral: split at EVERY source knot and product splice.
        total = 0.0
        for s, lower, upper in pieces:
            left, right = max(lo, lower), min(hi, upper)
            if right <= left:
                continue
            nodes = np.r_[left, s.x[(s.x > left) & (s.x < right)], right]
            total += np.trapezoid(np.interp(nodes, s.x, s.y), nodes)
        assert SP.band_mean(values, lo, hi) == pytest.approx(total / (hi - lo), rel=1e-5)
    end = G.Source([500.0, 600.0], [2.0, 2.0])
    np.testing.assert_allclose(
        G.piecewise_cells(np.array([499.9, 500.0, 550.0, 600.0, 600.1]), [(end, -np.inf, np.inf)]),
        [0, 1, 2, 1, 0],
        rtol=1e-10,
        atol=1e-10,
    )


def test_original_reproduction_optional(tmp_path):
    bira, cia = os.environ.get('SOSFUSE2_BIRA_DIR'), os.environ.get('SOSFUSE2_HITRAN_CIA')
    if not bira and not cia:
        pytest.skip('optional original data: SOSFUSE2_BIRA_DIR / SOSFUSE2_HITRAN_CIA')
    if bira:
        from sosfuse2.foundation.content_identity import asset_identity

        raw = (json.dumps(G.build_no2(bira), indent=1, allow_nan=False) + '\n').encode()
        # Computation identity, not file bytes: the regenerated file takes the current wavelength_reference and
        # generator hashes, which the identity leaves out.
        assert asset_identity(raw, M.NO2.name) == M.load('no2')['sha256']
    if cia:
        doc = G.build_o4(cia)
        path = tmp_path / 'reproduced.json'
        path.write_text(json.dumps(doc))
        assert M.load('o4', path=path)['meta']['source']['sha256'] == G.CIA_SHA


def synthetic_cia(tmp_path, monkeypatch):
    lines = []
    for lo, hi, temp, ref in ((20000, 33670, 293, 54), (16645, 29800, 293, 32), (15290, 16664, 287, 32)):
        lines.append(f'O2-O2 {lo} {hi} 4 {temp} 0 {ref}')
        lines.extend(f'{v:.12g} 1e-46' for v in np.linspace(lo, hi, 4))
    path = tmp_path / 'synthetic.cia'
    path.write_text('\n'.join(lines) + '\n')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(G, 'CIA_SHA', sha)
    monkeypatch.setattr(M, 'CIA_SHA', sha)
    return path


def test_prepare_cia(asset, tmp_path, monkeypatch):
    from sosfuse2.asset_tools.prepare_minor_absorbers import prepare

    cia = synthetic_cia(tmp_path, monkeypatch)
    # The preparation module imported o4_path by value.
    from sosfuse2.asset_tools import prepare_minor_absorbers as P

    monkeypatch.setattr(P, 'o4_path', lambda: asset)
    result = prepare(cia)
    assert result['sha256'] == hashlib.sha256(asset.read_bytes()).hexdigest()
    assert M.load('o4')['values'].max() > 0
    original = asset.read_bytes()
    cia.write_text('corrupt')
    with pytest.raises(SystemExit, match='E_SOURCE_SHA'):
        prepare(cia)
    assert asset.read_bytes() == original


def test_prepare_offline(asset, monkeypatch):
    from sosfuse2.asset_tools import prepare_minor_absorbers as P

    monkeypatch.setenv('SOSFUSE2_OFFLINE', '1')
    monkeypatch.setattr(P.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('network called offline'))
    with pytest.raises(ValueError, match='E_O4_OFFLINE:.*hitran.org'):
        P.prepare(download=True)


def test_prepare_mock_download(asset, tmp_path, monkeypatch, capsys):
    from sosfuse2.asset_tools import prepare_minor_absorbers as P

    cia = synthetic_cia(tmp_path, monkeypatch)
    calls = []

    def urlopen(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(cia.read_bytes())

    monkeypatch.setattr(P, 'o4_path', lambda: asset)
    monkeypatch.setattr(P.urllib.request, 'urlopen', urlopen)
    monkeypatch.setenv('SOSFUSE2_OFFLINE', '0')
    P.main(['--download'])
    # The preparation report authenticates generated file bytes, not computation identity.
    reported_raw_sha = json.loads(capsys.readouterr().out)['sha256']
    assert reported_raw_sha == hashlib.sha256(asset.read_bytes()).hexdigest()
    assert reported_raw_sha == M.load('o4')['raw_sha256']
    assert calls == [(G.CIA_URL, 60)]


def test_rayleigh_vacuum_anchor_and_columns(asset):
    ratios = R.coefficient([368.0, 550.0, 822.0]) / R.coefficient(450.0)
    np.testing.assert_allclose(ratios, [2.30826100, 0.439024374, 0.08605419], rtol=1e-6)
    assert R.vacuum_nm(450.0) == pytest.approx(450.126238, abs=5e-7)
    direct = R.sigma([368.0, 550.0, 822.0]) / R.sigma(450.0)
    np.testing.assert_allclose((ratios / direct - 1) * 1e6, [-37.4, 23.3, 48.0], atol=0.06)
    keep = D.build(dict(density=dict(D.LEGACY)))
    exact450 = 0.013558 * (550 / 450) ** 4 * 8 * (1 - np.exp(-10.0))
    # Full precision of the stipulated anchor and truncated exp8 column; the
    # six-digit .242029/.1062566 pair is not precise enough for a 1 ppm test.
    assert exact450 == pytest.approx(0.2420285517358466, rel=1e-6)
    assert keep['meta']['rayleigh_tau550'] == pytest.approx(0.1062564333130518, rel=1e-6)
    assert keep['meta']['rayleigh_tau550'] == pytest.approx(exact450 * ratios[1], rel=1e-12)
    state = dict(density=dict(density_profile='usstd76', ozone_profile='usstd76', rayleigh_column='usstd'))
    usstd = D.build(state)
    z = D.grid(len(usstd['table']))
    tau450 = np.trapezoid(usstd['table'][:, 0], z) * usstd['scale'] * R.coefficient(450.0)
    assert tau450 == pytest.approx(0.221502, rel=1e-6)
    assert usstd['meta']['rayleigh_tau550'] == pytest.approx(0.0972448, rel=1e-6)


def test_no2_independent_cm2_du(asset):
    raw_cm2 = np.array([1.23e-19, 4.56e-19, 7.89e-20])
    ext, opt = coeff(dict(parameters=dict(no2_du=0.207)), sigma=dict(no2=raw_cm2 * 1e-4))
    expected = raw_cm2 * 2.687e16 * 0.207
    np.testing.assert_allclose(ext[3] * opt['meta']['no2']['H_f_km'], expected, rtol=1e-9)


def test_o4_numeric_column_n0_square_and_float32(asset):
    state = dict(density=dict(density_profile='usstd76', ozone_profile='usstd76', rayleigh_column='usstd'))
    cross = 1.234e-55
    ext, opt = coeff(state, sigma=dict(o4=[cross] * 3))
    z = D.grid(len(opt['table']))
    q = opt['table'][:, 0] / opt['air_surface']
    n0 = opt['meta']['o4']['n0']
    u, weights = np.polynomial.legendre.leggauss(3)
    t = (z[1:, None] + z[:-1, None]) / 2 + np.diff(z)[:, None] * u / 2
    physical_density = np.interp(t, z, q) * n0 * 0.20946
    expected = cross * 1000 * np.sum(np.diff(z)[:, None] / 2 * weights * physical_density**2)
    column = np.sum(np.diff(z) * (q[:-1] ** 2 + q[:-1] * q[1:] + q[1:] ** 2) / 3)
    assert ext[4, 0] * column == pytest.approx(expected, rel=1e-6)
    external = D.with_profile(state, z_km=z, table=opt['table'][:, :2], mask=3, air_surface_number_density_m3=2 * n0)
    ext2, _ = coeff(external, sigma=dict(o4=[cross] * 3))
    np.testing.assert_allclose(ext2[4], 4 * ext[4], rtol=1e-14)
    assert np.float32(cross) == 0 and np.all(ext[4].astype('f4') > 0)
    np.testing.assert_allclose(ext[4].astype('f4'), ext[4], rtol=6e-8)
    from sosfuse2.rays.column_table import exact_columns, molecular

    keep = D.build(dict(density=dict(D.LEGACY)))
    val = exact_columns(
        molecular(keep, 3), np.array([[0.0, 0.0, 6371.0]]), np.array([[0.0, 0.0, 1.0]]), np.array([80.0])
    )[0]
    assert val == pytest.approx(4 * (1 - np.exp(-20.0)), rel=1e-12)


def test_external_air_validation_and_tau(asset):
    z = D.grid(161)
    t = np.column_stack([np.exp(-z / 8), np.maximum(0, 1 - abs(z - 25) / 15)])
    st = D.with_profile({}, z_km=z, table=t, mask=3)
    with pytest.raises(ValueError, match='E_O4_N0'):
        D.build(st)
    st['parameters'] = dict(o4=False)
    first = D.build(st)['meta']['rayleigh_tau550']
    t[:, 0] *= 2
    second = D.build(D.with_profile(st, z_km=z, table=t, mask=3))['meta']['rayleigh_tau550']
    assert second == pytest.approx(2 * first, rel=1e-14)
    t[0, 0] = 0
    with pytest.raises(ValueError, match='E_PROFILE_AIR'):
        D.with_profile(st, z_km=z, table=t, mask=3)


@pytest.mark.parametrize('no2,o4', [(0.0, False), (0.207, False), (0.0, True), (0.207, True)])
def test_switches(asset, no2, o4):
    ext, _ = coeff(dict(parameters=dict(no2_du=no2, o4=o4)))
    np.testing.assert_array_equal(ext[0], R.coefficient([450.0, 550.0, 577.0]))
    assert bool(ext[3].any()) == bool(no2)
    assert bool(ext[4].any()) == o4


def test_optics_signature_resolved_defaults(asset, monkeypatch):
    from sosfuse2.jobs import run_state as v3

    monkeypatch.setattr(v3, 'package_hashes', lambda: {})
    monkeypatch.setattr(v3, 'grid_metadata', lambda cfg: {})
    sign = lambda s: v3.signed_contract(s, {}, 'numpy', True, 'depolarizing')[0]
    assert sign({}) == sign(dict(parameters=dict(no2_du=0.207, o4=True)))
    assert sign({}) != sign(dict(parameters=dict(o4=False)))
    for bad in (0, 1, 'true', None):
        with pytest.raises(ValueError, match='E_O4_SWITCH'):
            sign(dict(parameters=dict(o4=bad)))
    with pytest.raises(ValueError, match='E_OPTICS_STATE_MIXED'):
        C.check_optics(C.optics_state({}), C.optics_state(dict(parameters=dict(no2_du=0))))


def test_identity_q16_sources_and_loss(asset, tmp_path):
    from sosfuse2.outputs import radiance_table, quantized_table

    physical = C.physics()
    amounts = C.optics_state({})
    meta = dict(contract='v4', physics=physical, optics_state=amounts)
    assert C.identity(physical) == physical and C.identity(None) == C.LEGACY
    assert C.check_same(C.LEGACY, C.LEGACY) == C.LEGACY
    with pytest.raises(ValueError, match='E_PHYSICS_CONTRACT'):
        C.identity(dict(rayleigh='lambda4_v0', components=list(C.COMPONENTS), minor_absorbers='off'))
    changed = copy.deepcopy(physical)
    changed['minor_absorbers']['o4']['asset_sha256'] = 'f' * 64
    assert C.identity(changed) == changed
    for other in (changed, C.LEGACY, C.S1_REFERENCE):
        with pytest.raises(ValueError, match='E_PHYSICS_MIXED'):
            C.check_same(physical, other)
    with pytest.raises(ValueError, match='E_OPTICS_STATE_MIXED'):
        SP.check_meta_contract(dict(meta, optics_state=None))
    with pytest.raises(ValueError, match='E_PHYSICS_LOST'):
        SP.check_meta_contract(dict(meta, physics=None))
    b = SP.block(dict(spectral_mode='point'), [550.0], False)
    del b['minor_absorber_sha256']
    with pytest.raises(ValueError, match='E_PHYSICS_CONTRACT'):
        SP.block_key(b)
    (tmp_path / 'meta.json').write_text(json.dumps(meta))
    radiance_table.save_table(
        tmp_path / 'table.npz',
        np.arange(1, 9, dtype='f4').reshape(2, 2, 2, 1),
        np.array([550.0]),
        np.array([-3.0, -6.0]),
        np.array([10.0, 50.0]),
        np.array([0.0, 90.0]),
    )
    quantized_table.save(tmp_path / 'table.npz', tmp_path / 'q.npz')
    assert quantized_table.load(tmp_path / 'q.npz')['header']['optics_state'] == amounts
    quantized_table.round_trip(tmp_path / 'table.npz', tmp_path / 'q.npz')
    source = tmp_path / 'source.npz'
    state = dict(parameters=dict(o4=False))
    s = SimpleNamespace(fuse_state=state, geo=SimpleNamespace(h=np.array([0.0]), chi=np.array([0.0])))
    np.savez(
        source,
        source=np.zeros(1),
        h=s.geo.h,
        chi=s.geo.chi,
        wavelength_nm=[550.0],
        physics=np.array(json.dumps(physical)),
        optics_state=np.array(json.dumps(amounts)),
        source_contract=np.array(json.dumps(dict(state=C.canonical_state(state)))),
    )
    with pytest.raises(ValueError, match='E_OPTICS_STATE_MIXED'):
        radiance_table.load_source(source, s, [550.0])
    with C._s1_reference():
        assert C.physics() == C.S1_REFERENCE
        with pytest.raises(ValueError, match='E_PHYSICS_REFERENCE'):
            C.production_only()


def test_ksplit_fallback_strict_and_wait(monkeypatch, capsys):
    from sosfuse2.solve.source_solver import Kernel

    for mode in (1, 2, 3):

        def objects(free, other=0):
            k = Kernel.__new__(Kernel)
            k.ksplit_mode = mode
            k.ksplit = True
            k.ksplit_verify = False
            k.K = 153
            k.host = np.asarray
            k.ksplit_memory = lambda: free
            k.ksplit_other_bytes = lambda: other
            k.config = lambda: SimpleNamespace(c3_fw15=False)
            k.ksplit_emit = lambda d: events.append(d)
            s = SimpleNamespace(W=15, geo=SimpleNamespace(h=np.arange(4), chi=np.arange(10)), strato=None)
            bundle = SimpleNamespace(
                nq=4,
                cells=2,
                arrays=dict(cell_off=np.array([0, 2, 4]), edge_off=np.array([0, 4, 8, 12, 16])),
                ksplit=dict(state=-1),
            )
            bundles = [(bundle, 0), (copy.copy(bundle), 1)]

            def free_memory():
                assert all(b.ksplit is None for b, _ in bundles), 'old spectral-chunk records counted as live'
                return free

            k.ksplit_memory = free_memory
            return k, s, bundles

        events = []
        monkeypatch.setenv('SOSFUSE2_KSPLIT_FALLBACK', '1')
        k, s, bundles = objects(1024)
        d = k.ksplit_preflight(s, bundles)
        assert d['effective'] == 0 and len(events) == 1 and d['peak'] > d['budget']
        k.ksplit_preflight(s, bundles)
        assert len(events) == 1
        # Standalone/promoted kernels use the default trace emitter instead of
        # the service callback; its clock helper already carries a default id.
        k, s, bundles = objects(1024)
        del k.ksplit_emit
        s.fuse_state = dict(id='S2')
        k.ksplit_preflight(s, bundles)
        event = json.loads(capsys.readouterr().out)
        assert event['id'] == 'S2' and event['phase'] == 'KSPLIT_FALLBACK'
        assert all(event[key] > 0 for key in ('pid', 'timestamp_ns', 'monotonic_ns'))
        monkeypatch.setenv('SOSFUSE2_KSPLIT_FALLBACK', '0')
        k, s, bundles = objects(1024)
        with pytest.raises(MemoryError, match='E_KSPLIT_MEMORY'):
            k.ksplit_preflight(s, bundles)
        monkeypatch.setenv('SOSFUSE2_KSPLIT_FALLBACK', '1')
        k, s, bundles = objects(1024, 8 << 30)
        waits = []

        def wait():
            waits.append(True)
            k.ksplit_memory = lambda: 8 << 30
            k.ksplit_other_bytes = lambda: 0
            return True

        k.ksplit_wait = wait
        d = k.ksplit_preflight(s, bundles)
        assert d['effective'] == mode and len(waits) == 1
        if mode >= 2:
            plans = [bundle._ksplit_plan[1] for bundle, _ in bundles]
            held = [(16 + C.SINV_F * 12 * 4) * 4 + p['records_bytes'] + p['store_bytes'] for p in plans]
            # Both retained records plus the largest live scratch must fit; the
            # same plans are consumed by vprod_columns, not re-sized greedily.
            peak = sum(held) + max(p['peak'] - h for p, h in zip(plans, held))
            assert peak <= k.ksplit_planned_bytes <= d['budget']
            assert all(p['store_allowed'] == p['store_bytes'] for p in plans)
        # A later FW15 chunk inherits the state-wide fallback; it cannot switch.
        k2, s2, _ = objects(8 << 30)
        s2.ksplit_decision = dict(requested=mode, effective=0)
        assert k2.ksplit_preflight(s2, bundles)['effective'] == 0
        assert not k2.ksplit


def test_lut_launch_only():
    import ast

    from sosfuse2.solve import source_solver, ksplit
    from sosfuse2.backends import flags

    # Kernel's methods live in solve/source_solver.py and, for the split transport, solve/ksplit.py (KsplitMixin).
    trees = [ast.parse(Path(m.__file__).read_text()) for m in (source_solver, ksplit)]
    users = []
    for node in (n for tree in trees for n in ast.walk(tree)):
        if isinstance(node, ast.FunctionDef) and any(
            isinstance(k, ast.keyword)
            and k.arg == 'role'
            and isinstance(k.value, ast.Constant)
            and k.value.value == 'lut'
            for k in ast.walk(node)
        ):
            users.append(node.name)
    assert users == ['strato_lut_kernel']
    assert "'strato_lut_build'" in inspect.getsource(source_solver.Kernel.strato_lut_kernel)
    assert 'strato_lut_kernel' in inspect.getsource(source_solver.Kernel.strato_lut_device)
    assert flags.resolve_config(('STRATO',), role='lut').role == 'lut'
    calls = []

    class Raw:
        def __init__(self, code, name, **kw):
            calls.append(('kernel', name, kw))
            self.name = name

        def compile(self):
            calls.append(('compile', self.name))

        def __call__(self, grid, block, args):
            calls.append(('launch', self.name, grid, block, args))

    k = source_solver.Kernel.__new__(source_solver.Kernel)
    k.xp = SimpleNamespace(empty=np.empty, RawKernel=Raw)
    k.flags = ('STRATO',)
    k.deterministic = True
    k.metrics = {}
    coeff = np.ones((33, 15), 'f4')
    out = k.strato_lut_device(coeff, 15)
    assert [row[:2] for row in calls] == [
        ('kernel', 'strato_lut_build'),
        ('compile', 'strato_lut_build'),
        ('launch', 'strato_lut_build'),
    ]
    assert '-DLUT_ONLY' in calls[0][2]['options']
    assert calls[-1][2:4] == ((((16384 + 1) * 15 + 255) // 256,), (256,))
    assert calls[-1][4][0] is out
    np.testing.assert_array_equal(out[: coeff.size], coeff.ravel())


def test_a_b_a_geometry_columns_and_optics(asset, tmp_path, monkeypatch):
    import subprocess
    import sys
    import time

    from sosfuse2.rays import column_table
    from sosfuse2.solve import ray_geometry
    from sosfuse2.precompute import host_cache
    from sosfuse2.foundation import paths
    from sosfuse2.jobs import pipeline_execution as pe
    from sosfuse2.jobs import run_state as v3

    monkeypatch.setenv('SOSFUSE2_HOST_CACHE', '1')
    monkeypatch.setenv('SOSFUSE2_SUN_WORKERS', '1')
    monkeypatch.setenv('SOSFUSE2_GROUND_WORKERS', '1')
    # Same convergent reduced fixture as test_pipeline_persistent; its normal
    # high-AOD counterpart is unsuitable for this coarse convergence grid.
    from sosfuse2.asset_tools.strato_states import build

    state = build()['states'][1]
    cfg = v3.resolve_settings(v3.settings_for(SimpleNamespace(reduced=True, settings_json=None)), state)
    solvers = []
    scopes = []
    for i, amount in enumerate((0.207, 0.0, 0.207)):
        st = copy.deepcopy(state)
        st['parameters'].update(no2_du=amount, o4=bool(amount))
        s, scope = v3.create_configured(
            tmp_path / f'prepare{i}', cfg, [550.0], 'numpy', st, geometry_cache=tmp_path / 'shared'
        )
        solvers.append(s)
        scopes.append(scope)
    from sosfuse2.precompute.endpoint_contract import identity

    assert identity(solvers[0]) == identity(solvers[1]) == identity(solvers[2])
    for col in (0, 2, 3):
        keys = [column_table.table_key(column_table.molecular(s.density_optics, col)) for s in solvers]
        assert keys[0] == keys[1] == keys[2]
    assert solvers[0].ext[3].any() and not solvers[1].ext[3].any()
    assert solvers[0].ext[4].any() and not solvers[1].ext[4].any()
    np.testing.assert_array_equal(solvers[0].ext, solvers[2].ext)
    cache = [host_cache.context(s, s.density_optics) for s in solvers]
    assert cache[0] == cache[1] == cache[2]
    from sosfuse2.atmosphere.stratosphere import _geometry_key

    attach_keys = [_geometry_key(s, scope, s.strato['profile']) for s, scope in zip(solvers, scopes)]
    assert attach_keys[0] == attach_keys[1] == attach_keys[2]
    # Exercise the actual IPC/service lifecycle in A -> B -> A order. The
    # service reuses its context but must install each state's fresh constants.
    with testkit.socket_root('s2-service-') as directory, (tmp_path / 'service.log').open('w') as log:
        root = Path(directory)
        monkeypatch.setenv('SOS_PERSISTENT_ROOT', str(root))
        monkeypatch.setenv('SOS_PERSISTENT_KEY', os.urandom(32).hex())
        proc = subprocess.Popen(
            [
                sys.executable,
                '-B',
                '-m',
                'sosfuse2.jobs.pipeline_execution',
                str(root),
                'numpy',
                str(tmp_path / 'gpu.lock'),
                '1',
                '6',
            ],
            env=paths.child_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 15
            while not (root / 'ready').exists():
                assert proc.poll() is None and time.monotonic() < deadline, (tmp_path / 'service.log').read_text()
                time.sleep(0.05)
            results = []
            args = SimpleNamespace(backend='numpy', fixed_boxes=True, block_size=128, threads=1, max_eval=16)
            for i, (s, scope) in enumerate(zip(solvers, scopes)):
                geom = ray_geometry.TransientGeometry(s, scope)
                pe.prepare_geometry(s, geom, True)
                result = pe.persistent_chunk(
                    s, geom, dict(batches=[], group=32), args, f'ABA{i}', lambda *a, **kw: None
                )
                results.append(result[0])
            np.testing.assert_array_equal(results[0], results[2])
            assert not np.array_equal(results[0], results[1])
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    events = [json.loads(line) for line in (tmp_path / 'service.log').read_text().splitlines() if line.startswith('{')]
    acquired = [e for e in events if e.get('phase') == 'gpu_acquired' and e['id'] != 'GPU_SERVICE']
    assert [e['id'] for e in acquired] == ['ABA0', 'ABA1', 'ABA2']
    assert len({e['pid'] for e in acquired}) == 1


def test_o4_asset_not_an_object_is_invalid(tmp_path):
    path = tmp_path / 'o4.json'
    for text in ('[]', '"text"', '1'):
        path.write_text(text)
        with pytest.raises(ValueError, match='^E_O4_ASSET_INVALID'):
            M.load('o4', path=path)


def test_prepare_download_http_error_is_coded(tmp_path, monkeypatch):
    """A truncated HTTP response (http.client.HTTPException, not OSError) is E_O4_DOWNLOAD with its reason."""
    import http.client

    from sosfuse2.asset_tools import prepare_minor_absorbers as P

    monkeypatch.setattr(P, 'o4_path', lambda: tmp_path / 'assets' / M.O4_NAME)
    monkeypatch.setenv('SOSFUSE2_OFFLINE', '')

    def truncated(*a, **kw):
        raise http.client.IncompleteRead(b'x', 10)

    monkeypatch.setattr(P.urllib.request, 'urlopen', truncated)
    with pytest.raises(RuntimeError, match='^E_O4_DOWNLOAD: .*IncompleteRead'):
        P.prepare(download=True)
