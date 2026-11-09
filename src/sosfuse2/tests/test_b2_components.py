"""S1 independent columns, slot separation, zero-coefficient regression and interface rules."""

import copy
import json

import numpy as np
import pytest

from sosfuse2 import column_table as ct
from sosfuse2 import density, paths, ray_columns, reference, spectral
from sosfuse2.components import COMPONENTS, NCOMP, SINV_F, physics


def atmosphere(table=False, step=0.5):
    z = density.grid(density.PROFILE_STEPS[step])
    if table:
        # Non-unit surface and substantial curvature discriminate q from p,
        # and square(interpolate(p)) from interpolate(square(p)).
        air = 2.3 * np.exp(-z / 9) * (1 + 0.07 * np.sin(z / 2))
        return density.build(
            density.with_profile(
                {},
                air_surface_number_density_m3=2.5469e25,
                z_km=z,
                table=np.column_stack([air, np.maximum(0, 1 - abs(z - 25) / 15)]),
                mask=3,
            )
        )
    return density.build(dict(density=dict(profile_step_km=step)))


def old_solver():
    return sun_scope()


def independent(f, p, d, end, knots):
    """Test-local GL48, split at every shell crossing and tangent, in XYZ coordinates."""
    pd = float(p @ d)
    rho2 = float(p @ p) - pd * pd
    cuts = [0.0, end]
    if 0 < -pd < end:
        cuts.append(-pd)
    for h in knots:
        disc = (6371.0 + h) ** 2 - rho2
        if disc < 0:
            continue
        for t in (-pd - np.sqrt(disc), -pd + np.sqrt(disc)):
            if 0 < t < end:
                cuts.append(t)
    cuts = np.unique(cuts)
    u, w = np.polynomial.legendre.leggauss(48)
    t = (cuts[1:, None] + cuts[:-1, None]) / 2 + np.diff(cuts)[:, None] * u / 2
    h = np.linalg.norm(p + t[..., None] * d, axis=-1) - 6371.0
    return float(np.sum(np.diff(cuts) * np.sum(f(h) * w, axis=1) / 2))


def functions(o):
    z = density.grid(len(o['table']))
    no2 = lambda h: np.interp(h, z, o['table'][:, 2])
    o4 = (
        (lambda h: (np.interp(h, z, o['table'][:, 0]) / o['table'][0, 0]) ** 2)
        if o['mask'] & 1
        else lambda h: np.exp(-h / 4)
    )
    return no2, o4


def test_fixed_component_and_kernel_layout():
    from sosfuse2.fused import Kernel
    from sosfuse2.geometry_contract import HEIGHT_DEPENDENT_END_COLUMNS, PACKED_LAYOUT_VERSION, TRAIL

    cu = (paths.PKG / 'kernel.cu').read_text()
    assert COMPONENTS == ('air', 'tropo_aerosol', 'o3', 'no2', 'o4')
    assert f'#define NCOMP {NCOMP}' in cu and SINV_F == 7
    assert 'prof[NCOMP]' in cu and 'sc[NCOMP]' in cu and 'st_prof,st_sc' in cu
    assert 'end_col[NCOMP*ray+k]' in cu and 'if(w<NCOMP+1)' in cu
    assert TRAIL['end_col'] == (5,) and PACKED_LAYOUT_VERSION == 3 and HEIGHT_DEPENDENT_END_COLUMNS == (1,)
    assert Kernel(backend='numpy').shared_bytes(15, 0) > 0


@pytest.mark.parametrize('step', [0.5, 0.25, 0.125])
def test_no2_grid_normalization(step):
    o = atmosphere(step=step)
    z = density.grid(len(o['table']))
    asset_z, asset_f, sha = density.no2_asset()
    np.testing.assert_array_equal(o['table'][:, 2], np.interp(z, asset_z, asset_f))
    assert o['meta']['no2']['asset_sha256'] == sha
    assert o['meta']['no2']['H_f_km'] == pytest.approx(np.trapezoid(asset_f, asset_z), rel=1e-14)
    assert np.max(o['table'][:, 2]) == 1 and np.all(o['table'][:, 2].astype('f4') > 0)


def test_external_two_column_api_and_surface():
    o = atmosphere(True)
    assert o['table'].shape == (161, 3) and o['air_surface'] == 2.3
    assert 'external' in o['meta']['no2']['source']
    with pytest.raises(ValueError, match='E_PROFILE_TABLE'):
        density.with_profile(
            {}, air_surface_number_density_m3=2.5469e25, z_km=density.grid(161), table=o['table'], mask=3
        )


def test_squared_linear_cell_average():
    z = density.grid(161)
    air = np.ones(161) * 2
    air[1:] = 0.6
    o = density.build(
        density.with_profile(
            {}, air_surface_number_density_m3=2.5469e25, z_km=z, table=np.column_stack([air, np.ones(161)]), mask=1
        )
    )
    got = ct.exact_columns(
        ct.molecular(o, 3), np.array([[0.0, 0.0, 6371.0]]), np.array([[0.0, 0.0, 1.0]]), np.array([0.5])
    )[0]
    assert got == pytest.approx(0.5 * (1 + 0.3 + 0.3**2) / 3, rel=1e-10)
    assert abs(got - 0.5 * (1 + 0.3**2) / 2) > 0.03


@pytest.mark.parametrize('ray', ['vertical', 'tangent', 'short'])
def test_minor_columns_independent_and_refinement(ray, monkeypatch):
    monkeypatch.setenv('SOSFUSE2_COLUMN_METHOD', 'exact')
    p = np.array([0.0, 0.0, 6371.0 + (0 if ray == 'vertical' else 11.37)])
    d = np.array([0.0, 0.0, 1.0]) if ray == 'vertical' else np.array([1.0, 0.0, 0.0])
    end = 80.0 if ray == 'vertical' else 0.0003 if ray == 'short' else 300.0
    for table in (False, True):
        first = None
        for step in (0.5, 0.25, 0.125):
            o = atmosphere(table, step)
            for k, f in zip((2, 3), functions(o)):
                got = ct.column(ct.molecular(o, k), p[None], d[None], np.array([end]))[0]
                want = independent(f, p, d, end, density.grid(len(o['table'])))
                assert got == pytest.approx(want, rel=2e-7, abs=2e-8)
            if not table:
                if first is not None:
                    assert got == pytest.approx(first, rel=1e-12)
                first = got


@pytest.mark.parametrize('path', ['sun', 'endpoint', 'ground', 'interior'])
def test_all_column_consumers(path, monkeypatch):
    from sosfuse2.sun_parallel import _row

    monkeypatch.setenv('SOSFUSE2_COLUMN_METHOD', 'exact')
    old = old_solver()
    for table in (False, True):
        o = atmosphere(table)
        p = np.array([0.0, 0.0, 6391.0])
        d = np.array([0.0, 0.0, 1.0])
        if path == 'sun':
            base = old.total_columns(p[None], d[None], 32)
            got = _row(old, 20.0, np.array([0.0, 1.0]), 6371.0, 32, np.repeat(base, 2, axis=0), o, 'gl')[-1]
            end = 60.0
        else:
            if path == 'ground':
                d = -d
            end = 20.0 if path == 'ground' else 60.0
            if path == 'interior':
                end = 0.5
                prof = reference.profile_at(
                    np.array([20.0, 20.25, 20.5], 'f4'), dict(height=np.float32(1.2), density_optics=o)
                )
                got = end * (prof[0] + 4 * prof[1] + prof[2]) / 6
            else:
                base, corrected, _ = ray_columns.integrate(
                    p[None], d[None], old, legacy=True, optics=o, main=path == 'endpoint'
                )
                got = ray_columns.replace(base, corrected, o)[0]
        for k, f in zip((3, 4), functions(o)):
            want = independent(f, p, d, end, density.grid(len(o['table'])))
            assert got[k] == pytest.approx(want, rel=2e-6, abs=2e-8)


@pytest.mark.parametrize('component', range(5))
def test_one_component_optical_depth(component):
    o = atmosphere(True, 0.125)
    e = np.zeros((5, 1), 'f4')
    e[component] = 0.03
    g = dict(p=np.array([[0.0, 0.0, 6371.0]]), d=np.array([[0.0, 0.0, 1.0]]), edges=np.arange(641)[None] * 0.125)
    c = dict(W=1, height=np.float32(1.2), ext=e, density_optics=o)
    prefix = reference.SimpsonPrefix(g, c)
    for seg in range(640):
        prefix.begin(seg)
        prefix.finish()
    z = density.grid(641)
    funcs = [
        lambda h: np.interp(h, z, o['table'][:, 0]),
        lambda h: np.exp(-h / 1.2),
        lambda h: np.interp(h, z, o['table'][:, 1]),
        *functions(o),
    ]
    want = independent(funcs[component], g['p'][0], g['d'][0], 80.0, z) * float(e[component, 0])
    assert prefix.tau[0, 0] == pytest.approx(want, rel=2e-6, abs=1e-8)


def test_stratosphere_separate_from_minor_slots():
    h = np.array([0.0, 10.0, 20.0, 30.0], 'f4')
    c = dict(W=1, height=np.float32(1.2), ext=np.ones((5, 1), 'f4'), density_optics=atmosphere())
    layer = dict(c, strato_ext=np.array([0.2], 'f4'), strato_profile=dict(center=20.0, sigma=3.0, norm=7.5))
    np.testing.assert_array_equal(reference.profile_at(h, c), reference.profile_at(h, layer))
    st = reference.strato_profile(h, layer)
    prof = reference.profile_at(h, c)
    np.testing.assert_allclose(
        reference.extinction(prof, layer, st) - reference.extinction(prof, c), 0.2 * st[:, None], rtol=2e-6, atol=1e-7
    )


def test_prepared_endpoint_without_solver_module():
    import pickle

    from sosfuse2.pipeline_execution import PreparedSegments

    old = old_solver()
    o = atmosphere(True)
    p = np.array([[0.0, 0.0, 6391.0]])
    d = np.array([[0.0, 0.0, 1.0]])
    prepared = pickle.loads(pickle.dumps(PreparedSegments(old, {0: dict(p=p, d=d)})))
    assert not hasattr(prepared, 'probe')
    col = density.correct_columns(p, d, old.total_columns(p, d, 24), prepared, o, end=np.array([60.0]))[0]
    for k, f in zip((3, 4), functions(o)):
        assert col[k] == pytest.approx(independent(f, p[0], d[0], 60.0, density.grid(161)), rel=2e-6, abs=2e-8)


def test_zero_minor_coefficients_keep_legacy_values():
    from functools import partial

    from sosfuse2.solver_optics import coefficients

    p = paths.ASSETS / 'spectra.json'
    f = partial(coefficients, json.loads(p.read_text()))
    params = dict(aer=0.1, ang=1.3, ozone_DU=280.0)
    w = np.array([400.0, 550.0, 700.0])
    from sosfuse2.components import _s1_reference

    with _s1_reference():
        e = f(w, params)
    spec = json.loads(p.read_text())
    legacy = np.stack(
        [
            0.013558 * (550 / w) ** 4,
            params['aer'] * (550 / w) ** params['ang'],
            np.interp(w, spec['wavelength_nm'], spec['ozone']) * params['ozone_DU'] * 2.687e20 / 15,
        ]
    )
    np.testing.assert_array_equal(e[:3], legacy)
    assert not e[3:].any()
    h = np.linspace(0, 80, 129).astype('f4')
    c = dict(W=3, height=np.float32(1.2), ext=e.astype('f4'), density_optics=atmosphere())
    prof = reference.profile_at(h, c)
    old = np.zeros((len(h), 3), 'f4')
    for k in range(3):
        old += prof[:, k, None] * c['ext'][k]
    np.testing.assert_array_equal(reference.extinction(prof, c), old)


def test_column_keys_depend_on_shapes_not_amounts():
    z = density.grid(161)
    a = np.exp(-z / 8)
    b = np.exp(-z / 4)
    b *= np.trapezoid(a, z) / np.trapezoid(b, z)
    states = [
        density.with_profile(
            {}, air_surface_number_density_m3=2.5469e25, z_km=z, table=np.column_stack([v, np.ones(161)]), mask=1
        )
        for v in (a, b)
    ]
    oa, ob = map(density.build, states)
    assert ct.table_key(ct.molecular(oa, 3)) != ct.table_key(ct.molecular(ob, 3))
    assert ct.table_key(ct.molecular(oa, 2)) == ct.table_key(ct.molecular(ob, 2))
    changed = copy.deepcopy(states[0])
    changed['parameters'] = dict(no2_du=1.2)
    assert density.column_identity(oa) == density.column_identity(density.build(changed))
    assert density.column_identity(oa) == density.column_identity(dict(oa, scale=2.0))


def test_physics_legacy_read_alone_mixed_rejected(tmp_path):
    """A product without physics reads as the legacy identity on its own; combining it
    with current physics is E_PHYSICS_MIXED; an identity this code does not know is E_PHYSICS_CONTRACT."""
    from sosfuse2.components import LEGACY, canonical_state, check_same, identity, optics_state

    b = spectral.block(dict(spectral_mode='band'), np.array([550.0]), False)
    assert b['physics'] == physics()
    legacy = copy.deepcopy(b)
    for key in ('physics', 'minor_absorber_sha256', 'point_definition'):
        legacy.pop(key)
    assert identity(None) == LEGACY == identity(dict(LEGACY)) and identity(physics()) == physics()
    unknown = [
        dict(physics(), components=list(COMPONENTS[:3])),
        dict(physics(), minor_absorbers='on'),
        dict(LEGACY, rayleigh='r2'),
        'lambda4_v0',
    ]
    for bad in unknown:
        with pytest.raises(ValueError, match='E_PHYSICS_CONTRACT'):
            identity(bad)
    # table + weights
    spectral.check_contract(b, copy.deepcopy(b))
    spectral.check_contract(legacy, copy.deepcopy(legacy))
    assert spectral.block_key(legacy) != spectral.block_key(b)
    for x, y in ((b, legacy), (legacy, b)):
        with pytest.raises(ValueError, match='E_PHYSICS_MIXED'):
            spectral.check_contract(x, y)
    for bad in unknown[:2]:
        with pytest.raises(ValueError, match='E_PHYSICS_CONTRACT'):
            spectral.check_contract(b, dict(b, physics=bad))
    old_spec = spectral.spec_table(legacy, np.array([550.0]))
    new_spec = spectral.spec_table(b, np.array([550.0]))
    assert all(old_spec[k] == new_spec[k] for k in ('ozone', 'solar', 'wavelength_nm', 'source'))
    assert old_spec['no2'] == old_spec['o4'] == [0.0]  # a legacy table's original absorbers are preserved
    # table meta: alone, signed or derived; its own block must agree with it
    for signed in (False, True):
        sig = dict(signature='test') if signed else {}
        new = dict(contract='v4', physics=physics(), optics_state=optics_state({}), settings=dict(spectral=b), **sig)
        old = dict(contract='v4', settings=dict(spectral=legacy), **sig)
        assert spectral.table_block(new) == b and spectral.table_block(old) == legacy
        assert spectral.check_meta_contract(dict(contract='v4', **sig)) == 'v4'
        for mixed in (dict(new, settings=dict(spectral=legacy)), dict(old, physics=physics())):
            with pytest.raises(ValueError, match='E_PHYSICS_MIXED'):
                spectral.table_block(mixed)
        with pytest.raises(ValueError, match='E_PHYSICS_CONTRACT'):
            spectral.check_meta_contract(dict(contract='v4', physics=unknown[1], **sig))
    # q16: the header records the identity (legacy for a table without physics or without meta.json)
    from types import SimpleNamespace

    from sosfuse2 import table_fast, table_q

    def q16(name, meta):
        folder = tmp_path / name
        folder.mkdir()
        if meta is not None:
            (folder / 'meta.json').write_text(meta if isinstance(meta, str) else json.dumps(meta))
        table_fast.save_table(
            folder / 'table.npz',
            np.arange(1, 9, dtype='f4').reshape(2, 2, 2, 1),
            np.array([550.0]),
            np.array([-3.0, -6.0]),
            np.array([10.0, 50.0]),
            np.array([0.0, 90.0]),
        )
        table_q.save(folder / 'table.npz', folder / 'q.npz')
        table_q.round_trip(folder / 'table.npz', folder / 'q.npz')
        return folder / 'q.npz'

    current = q16('new', dict(contract='v4', physics=physics(), optics_state=optics_state({})))
    assert table_q.load(current)['header']['physics'] == physics()
    for name, meta in (('old', dict(contract='v4')), ('bare', None)):
        assert table_q.load(q16(name, meta))['header']['physics'] == LEGACY
    with pytest.raises(table_q.TableQError, match='E_PHYSICS_CONTRACT'):
        q16('unreadable', '{not json')
    with np.load(current) as z:
        q = dict(z)
    header = json.loads(str(q['header']))
    for name, value in (('old_header.npz', None), ('unknown_header.npz', unknown[1])):
        h = dict(header)
        h.pop('physics')
        h.pop('optics_state', None)
        if value is not None:
            h['physics'] = value
        np.savez(tmp_path / name, **dict(q, header=np.array(json.dumps(h))))
    old_header = table_q.load(tmp_path / 'old_header.npz')['header']
    assert check_same(old_header.get('physics'), LEGACY) == LEGACY
    with pytest.raises(ValueError, match='E_PHYSICS_MIXED'):
        check_same(old_header.get('physics'), physics(), what='q16 pair')
    with pytest.raises(ValueError, match='E_PHYSICS_CONTRACT'):
        table_q.load(tmp_path / 'unknown_header.npz')
    # round_trip is a comparison: a legacy q16 against a table whose meta now says current physics is refused
    legacy_q = tmp_path / 'old' / 'q.npz'
    (tmp_path / 'old' / 'meta.json').write_text(
        json.dumps(dict(contract='v4', physics=physics(), optics_state=optics_state({})))
    )
    with pytest.raises(ValueError, match='E_PHYSICS_MIXED'):
        table_q.round_trip(tmp_path / 'old' / 'table.npz', legacy_q)
    # a source re-queried by this solver must carry this code's physics
    s = SimpleNamespace(fuse_state={}, geo=SimpleNamespace(h=np.array([0.0]), chi=np.array([0.0])))
    base = dict(source=np.zeros(1), h=np.array([0.0]), chi=np.array([0.0]), wavelength_nm=np.array([550.0]))
    np.savez(tmp_path / 'source_old.npz', **base)
    np.savez(
        tmp_path / 'source_new.npz',
        **base,
        physics=np.array(json.dumps(physics(), sort_keys=True)),
        optics_state=np.array(json.dumps(optics_state({}), sort_keys=True)),
        source_contract=np.array(
            json.dumps(dict(state=canonical_state({}), settings={}, backend='numpy', polarization_model='depolarizing'))
        ),
    )
    with pytest.raises(ValueError, match='E_PHYSICS_MIXED'):
        table_fast.load_source(tmp_path / 'source_old.npz', s, [550.0])
    assert table_fast.load_source(
        tmp_path / 'source_new.npz', s, [550.0], settings={}, backend='numpy', full_settings=True
    ).shape == (1,)


def sun_scope(height=1.2):
    """Module callables bound only to the selected aerosol profile."""
    from functools import partial
    from types import SimpleNamespace

    from sosfuse2 import solver_columns as columns
    from sosfuse2.accepted.engine import SunGeometryBinder
    from sosfuse2.tropo_profile import ExponentialProfile

    aerosol = ExponentialProfile(height, clamp=True)
    return SimpleNamespace(
        R=columns.R,
        Z=columns.Z,
        probe=SimpleNamespace(gl=columns.gl, frame=columns.frame, boundaries=columns.boundaries),
        profiles=partial(columns.profiles, aerosol),
        geometry_segments=partial(columns.geometry_segments, aerosol),
        total_columns=partial(columns.total_columns, aerosol),
        SunGeometry=SunGeometryBinder(aerosol, height),
        AER_HEIGHT=height,
        AER_PROFILE=aerosol,
        aerosol_descriptor=aerosol.descriptor(),
    )


def test_sun_table_identity_rebuilds_stale_and_reuses_current(tmp_path, monkeypatch):
    """2026-11-04 judge run: an unkeyed 3-column sun_columns.npz from before B2 reached the 5-column solver."""
    import warnings

    from sosfuse2 import sun_parallel

    monkeypatch.setenv('SOSFUSE2_SUN_WORKERS', '1')
    old = sun_scope()
    folder = tmp_path / 'state'
    folder.mkdir()
    path = folder / 'sun_columns.npz'
    h = np.linspace(0, np.sqrt(80.0), 81) ** 2
    x = np.linspace(0, 1, 401)
    np.savez(path, h=h, x=x, col=np.zeros((81, 401, 3)))  # the pre-B2 layout, no identity
    with pytest.warns(RuntimeWarning, match='SUN_TABLE_STALE: .*no identity'):
        sun = old.SunGeometry(1, folder)
    assert sun.col.shape == (81, 401, NCOMP) and sun.build_seconds > 0
    assert old.SunGeometry.stored_identity(path) == old.SunGeometry.identity(1)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        again = old.SunGeometry(1, folder)  # current identity: reused, not rebuilt
        assert sun_parallel.prebuild(old, 1, folder) is None
    assert again.build_seconds == 0.0 and np.array_equal(again.col, sun.col)
    other = sun_scope(height=2.0)  # same folder, another aerosol height
    assert other.SunGeometry.identity(1) != old.SunGeometry.identity(1)
    with pytest.warns(RuntimeWarning, match='SUN_TABLE_STALE: .*identity differs'):
        rebuilt = other.SunGeometry(1, folder)
    assert not np.array_equal(rebuilt.col[..., 1], sun.col[..., 1]) and np.array_equal(
        rebuilt.col[..., 0], sun.col[..., 0]
    )
    # prebuild (the parallel writer the host cache runs first) refuses to skip a stale file as well
    np.savez(path, h=h, x=x, col=np.zeros((81, 401, 3)))
    with pytest.warns(RuntimeWarning, match='SUN_TABLE_STALE: .*no identity'):
        assert sun_parallel.prebuild(old, 1, folder) is not None
    assert old.SunGeometry.stored_identity(path) == old.SunGeometry.identity(1)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        fresh = old.SunGeometry(1, folder)
    assert fresh.build_seconds == 0.0 and np.allclose(fresh.col, sun.col, rtol=1e-6, atol=0)
    assert not list(folder.glob('*.tmp'))


def test_legacy_spectral_block_never_signs_a_new_run(monkeypatch):
    """A block copied from an old meta (no physics) must not sign current physics."""
    from sosfuse2 import run_gpu_state_v3 as v3

    monkeypatch.setattr(v3, 'package_hashes', lambda: {})
    monkeypatch.setattr(v3, 'grid_metadata', lambda cfg: {})
    b = spectral.block(dict(spectral_mode='band'), np.array([550.0]), False)
    legacy = copy.deepcopy(b)
    for key in ('physics', 'minor_absorber_sha256', 'point_definition'):
        legacy.pop(key)
    with pytest.raises(ValueError, match='E_PHYSICS_MIXED'):
        v3.signed_contract({}, dict(spectral=legacy), 'numpy', True, 'x')
    with pytest.raises(ValueError, match='E_PHYSICS_CONTRACT'):
        v3.signed_contract(
            {}, dict(spectral=dict(b, physics=dict(physics(), minor_absorbers='on'))), 'numpy', True, 'x'
        )


def test_accepted_cells_folder_keyed_by_code_layout_grid(tmp_path, monkeypatch):
    """Pre-B2 blocks/cell_*.npz rows are 14 floats, the native transport reads 18."""
    from sosfuse2.accepted import engine

    (tmp_path / 'blocks').mkdir()
    np.savez(tmp_path / 'blocks' / 'cell_00000.npz', point=np.zeros((1, 14), 'f4'))  # pre-B2 layout
    h = np.array([0.0, 0.1, 8.0, 80.0])
    chi = np.deg2rad([0.0, 60.0, 90.0])
    cfg = dict(nmu=1, nphi=2, path_n=4)
    a = engine.cell_folder(tmp_path, h, chi, cfg)
    assert a.parent == tmp_path / 'blocks' and a.is_dir() and not (a / 'cell_00000.npz').exists()
    assert engine.cell_folder(tmp_path, h, chi, cfg) == a
    assert (
        engine.cell_folder(tmp_path, h[:3], chi, cfg) != a
        and engine.cell_folder(tmp_path, h, chi, dict(cfg, path_n=6)) != a
    )
    real = engine.fingerprint
    monkeypatch.setattr(engine, 'fingerprint', lambda: dict(real(), **{'sosfuse2/solver_core.py': 'edited'}))
    assert engine.cell_folder(tmp_path, h, chi, cfg) != a


def test_no2_amount_signature_and_validation(monkeypatch):
    from sosfuse2 import run_gpu_state_v3 as v3

    monkeypatch.setattr(v3, 'package_hashes', lambda: {})
    monkeypatch.setattr(v3, 'grid_metadata', lambda cfg: {})
    a = dict(parameters=dict(no2_du=0.0))
    b = dict(parameters=dict(no2_du=0.207))
    assert v3.signed_contract(a, {}, 'numpy', True, 'x')[0] != v3.signed_contract(b, {}, 'numpy', True, 'x')[0]
    assert density.column_identity(density.build(a)) == density.column_identity(density.build(b))
    for bad in (-1.0, float('nan'), float('inf'), True):
        with pytest.raises(ValueError, match='E_NO2_DU'):
            density.build(dict(parameters=dict(no2_du=bad)))
