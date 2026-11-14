"""Height and solar zenith grid rules in the settings (h_refine, chi_step_deg, chi_refine_deg), CPU only.

The grids are built from solver_columns.grids(1); the solver is stopped at its boundary, so no optics are prepared."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from sosfuse2.solve import source_solver, ray_geometry
from sosfuse2.precompute import host_cache
from sosfuse2.jobs import run_state as v3
from sosfuse2.rays.shell_segments import grids as base_grids

H, CHI = base_grids(1)
DEGREES = np.round(np.rad2deg(CHI), 12)


def settings(tmp_path, raw=None, reduced=False):
    path = None
    if raw is not None:
        path = tmp_path / 'settings.json'
        path.write_text(json.dumps(raw))
    return v3.settings_for(SimpleNamespace(reduced=reduced, settings_json=path))


class Captured(Exception):
    def __init__(self, geo):
        self.geo = geo


def solver_grid(tmp_path, cfg):
    """The h and chi that create_configured hands to the solver."""
    from sosfuse2.solve import engine_core

    class StopSolver:
        def __init__(self, ctx, geo, *args):
            raise Captured(geo)

    engine = source_solver.load_engine()
    state = ray_geometry.normal_state()
    cfg = v3.resolve_settings(cfg, state)  # the product band block takes its ozone identity from the state
    with (
        patch.object(source_solver, 'load_engine', return_value=engine),
        patch.object(engine_core, 'Solver', StopSolver),
        patch.object(ray_geometry, 'Kernel', return_value=None),
        patch.object(host_cache, 'sun_geometry', return_value=None),
        patch.dict('os.environ', SOSFUSE2_HOST_CACHE='0'),
        pytest.raises(Captured) as caught,
    ):
        v3.create_configured(tmp_path / 'prepare', cfg, cfg['wavelengths_nm'][:1], 'numpy', state)
    return caught.value.geo


def heights_1_3_4(name):
    """The three height grids of 1.3.4, written out as they were."""
    if name == 'none':
        return H
    if name == 'mid':
        return np.sort(np.r_[H, (H[:-1] + H[1:]) / 2])
    pairs = list(zip(H[:-1], H[1:]))
    return np.sort(
        np.r_[
            H,
            [(u + v) / 2 for u, v in pairs if v <= 13 + 1e-9],
            [u + (v - u) * f for u, v in pairs if u >= 13 - 1e-9 for f in (1 / 3, 2 / 3)],
        ]
    )


def test_aliases_keep_the_1_3_4_heights():
    for name, rule, nr in (('none', 'b1a1', 20), ('mid', 'b2a2', 39), ('below13mid_above13thirds', 'b2a3', 44)):
        old = heights_1_3_4(name)
        assert len(old) == nr
        assert np.array_equal(v3.refined_heights(H, name), old)
        assert np.array_equal(v3.refined_heights(H, rule), old)


def test_rules_and_node_counts(tmp_path):
    # nr = 20 + 14 (m - 1) + sum(k_i - 1); b2a6 gives 59 nodes, b4a12 gives 117
    for rule, nr in (('b2a6', 59), ('b4a12', 117), ('b2a3.3.6.6.6', 53), ('b1a1.1.1.1.12', 31), ('b12a12', 229)):
        assert len(v3.refined_heights(H, rule)) == nr
        cfg = settings(tmp_path, {'h_refine': rule})
        assert (cfg['nr'], cfg['h_refine']) == (nr, rule)
        assert settings(tmp_path, {'h_refine': rule, 'nr': nr})['nr'] == nr
    # b2a6: midpoints below 13 km, sixths of every interval above
    h = v3.refined_heights(H, 'b2a6')
    assert np.allclose(h[(h > 13) & (h < 18)], 13 + 5 * np.arange(1, 6) / 6, rtol=0, atol=1e-12)
    assert np.all(np.diff(h) > 0)


def test_nr_alone_still_names_only_the_1_3_4_grids(tmp_path):
    for nr, name in ((20, 'none'), (39, 'mid'), (44, 'below13mid_above13thirds')):
        assert settings(tmp_path, {'nr': nr})['h_refine'] == name
    with pytest.raises(ValueError, match=r'E_SETTINGS_GRID: received=\(59, None\)'):
        settings(tmp_path, {'nr': 59})


def test_bad_height_rules_are_refused(tmp_path):
    for rule in (
        'b0a3',
        'b2a0',
        'b2a13',
        'b13a2',
        'b02a3',
        'b2a3.3',
        'b2a3.3.3.3.3.3',
        'b2a3.',
        'B2A3',
        'x',
        '',
        None,
        [],
        3,
    ):
        with pytest.raises(ValueError, match='E_SETTINGS_GRID: received='):
            settings(tmp_path, {'h_refine': rule})
    with pytest.raises(ValueError, match=r'E_SETTINGS_GRID: received=\(44, .b2a6.\); h_refine gives nr 59'):
        settings(tmp_path, {'h_refine': 'b2a6', 'nr': 44})


def test_chi_steps_and_the_default(tmp_path):
    cfg = settings(tmp_path)
    assert (cfg['chi_step_deg'], cfg['nchi']) == (0.5, 75)
    assert 'chi_refine_deg' not in cfg
    assert v3.refined_chi(CHI, 0.5) is CHI
    outside = DEGREES[(DEGREES < 85) | (DEGREES > 110)]
    for step, nchi in ((0.25, 125), (0.125, 225)):
        chi = np.rad2deg(v3.refined_chi(CHI, step))
        assert len(chi) == nchi == settings(tmp_path, {'chi_step_deg': step})['nchi']
        assert np.allclose(chi[(chi >= 85) & (chi <= 110)], np.arange(85, 110 + step / 2, step), rtol=0, atol=1e-9)
        assert np.allclose(chi[(chi < 85) | (chi > 110)], outside, rtol=0, atol=1e-9)
    for step in ('fine', 0.3, 1, True, None):
        with pytest.raises(ValueError, match='E_SETTINGS_GRID: chi_step_deg'):
            settings(tmp_path, {'chi_step_deg': step})


def test_chi_refine_window(tmp_path):
    # 0.5 deg plus the 24 midpoints 88.25 ... 99.75 deg, nchi 99
    cfg = settings(tmp_path, {'chi_refine_deg': [88, 100]})
    assert (cfg['nchi'], cfg['chi_refine_deg']) == (99, [88.0, 100.0])
    chi = np.rad2deg(v3.refined_chi(CHI, 0.5, cfg['chi_refine_deg']))
    added = np.setdiff1d(np.round(chi, 9), np.round(DEGREES, 9))
    assert np.allclose(added, np.arange(88.25, 100, 0.5), rtol=0, atol=1e-9)
    # a window on the 0.25 deg grid halves that grid's step inside it
    assert settings(tmp_path, {'chi_step_deg': 0.25, 'chi_refine_deg': [90, 95]})['nchi'] == 125 + 20
    # null is the same settings as no key; the whole 85-110 window equals the uniform 0.25 deg grid
    assert settings(tmp_path, {'chi_refine_deg': None}) == settings(tmp_path)
    assert np.array_equal(v3.refined_chi(CHI, 0.5, [85.0, 110.0]), v3.refined_chi(CHI, 0.25))
    assert settings(tmp_path, {'chi_refine_deg': [85, 110]})['nchi'] == 125
    # an end within 1e-9 deg of a node is replaced by that node
    assert settings(tmp_path, {'chi_refine_deg': [88 + 1e-10, 100]})['chi_refine_deg'] == [88.0, 100.0]
    for window in (
        [88.1, 100],
        [100, 88],
        [88, 88],
        [88, 88.0000000005],
        [10**400, 100],
        [float('inf'), 100],
        [80, 100],
        [88, 112],
        [88],
        [88, 95, 100],
        [True, 100],
        ['88', 100],
        [float('nan'), 100],
        '88-100',
        {'lo': 88},
    ):
        with pytest.raises(ValueError, match='E_SETTINGS_GRID: chi_refine_deg'):
            settings(tmp_path, {'chi_refine_deg': window})
    with pytest.raises(ValueError, match='E_SETTINGS_GRID: received nchi 98; chi_step_deg and chi_refine_deg give 99'):
        settings(tmp_path, {'chi_refine_deg': [88, 100], 'nchi': 98})


def test_reduced_grid_is_fixed(tmp_path):
    cfg = settings(tmp_path, reduced=True)
    assert (cfg['nr'], cfg['h_refine'], cfg['chi_step_deg'], cfg['nchi']) == (4, 'none', None, 10)
    for raw in (
        {'h_refine': 'mid'},
        {'h_refine': 'b2a6'},
        {'nr': 44},
        {'chi_step_deg': 0.25},
        {'chi_refine_deg': [88, 100]},
        {'nchi': 11},
    ):
        with pytest.raises(ValueError, match='E_SETTINGS_GRID: received='):
            settings(tmp_path, raw, reduced=True)


def test_saved_source_settings_round_trip(tmp_path):
    # table_fast.restored_settings rebuilds the settings of a saved source through settings_for and compares the dict
    from sosfuse2.foundation import paths
    from sosfuse2.outputs import radiance_table
    from sosfuse2.atmosphere.components import canonical_state

    state = next(s for s in json.loads((paths.ASSETS / 'states_v1.json').read_text())['states'] if s.get('runnable'))
    for raw in ({}, {'chi_refine_deg': None}, {'h_refine': 'b2a6', 'L': 24, 'chi_refine_deg': [88, 100]}):
        cfg = v3.resolve_settings(
            settings(tmp_path, dict(raw, view_shape=[1, 2], solar_elevation_deg=[-3.0, -6.0])), state
        )
        contract = dict(state=canonical_state(state), settings=cfg, backend='cupy', polarization_model='depolarizing')
        path = tmp_path / 'source.npz'
        np.savez(path, wavelength_nm=cfg['wavelengths_nm'], source_contract=np.array(json.dumps(contract)))
        assert radiance_table.restored_settings(path, state, False, 'cupy') == cfg, raw


def test_solver_grid_matches_settings_and_signature(tmp_path):
    for raw in (
        {},
        {'nr': 39},
        {'h_refine': 'b2a6', 'chi_refine_deg': [88, 100]},
        {'h_refine': 'b4a12', 'chi_step_deg': 0.125},
    ):
        cfg = settings(tmp_path, raw)
        geo = solver_grid(tmp_path, cfg)
        assert (len(geo.h), len(geo.chi)) == (cfg['nr'], cfg['nchi'])
        # grid_metadata (the signed height bytes) and the solver build the heights through the same rule
        assert v3.grid_metadata(cfg) == v3.grid_metadata(cfg, geo.h)
        assert np.array_equal(geo.chi, v3.refined_chi(CHI, cfg['chi_step_deg'], cfg.get('chi_refine_deg')))


def test_same_size_grids_stay_apart(tmp_path):
    # Two windows with the same node count give different nodes, so the geometry identity (coordinates included) differs.
    from sosfuse2.precompute import endpoint_contract

    a, b = (settings(tmp_path, {'chi_refine_deg': w}) for w in ([88, 100], [86, 98]))
    assert a['nchi'] == b['nchi'] == 99
    geo_a, geo_b = solver_grid(tmp_path, a), solver_grid(tmp_path, b)
    assert not np.array_equal(geo_a.chi, geo_b.chi)
    assert (
        endpoint_contract.identity(SimpleNamespace(geo=geo_a))[1]
        != endpoint_contract.identity(SimpleNamespace(geo=geo_b))[1]
    )
