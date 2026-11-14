"""V14 grid rules, CPU only: stop at the solver boundary before any optical preparation."""

import hashlib
import importlib.util
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from sosfuse2.solve import source_solver, ray_geometry
from sosfuse2.precompute import host_cache
from sosfuse2.jobs import run_state as v3
from sosfuse2.checks import testkit

if testkit.sky_sos_root() is None:  # the V14 grid harness is a sky-sos research record, loaded at import
    pytest.skip(f'V14 harness results/quad2_deep_twilight_20261020 needs {testkit.ENV}', allow_module_level=True)
HARNESS = testkit.sky_sos('results/quad2_deep_twilight_20261020')
sys.path.insert(0, str(HARNESS))
spec = importlib.util.spec_from_file_location('v14_grid_harness', HARNESS / 'run_grid_eval.py')
harness = importlib.util.module_from_spec(spec)
with testkit.legacy_module_aliases('host_cache', 'fused', 'run_gpu_state_v3'):
    spec.loader.exec_module(harness)


def settings(tmp_path, raw=None, reduced=False):
    path = None
    if raw is not None:
        path = tmp_path / 'settings.json'
        path.write_text(json.dumps(raw))
    return v3.settings_for(SimpleNamespace(reduced=reduced, settings_json=path))


class Captured(Exception):
    def __init__(self, geo):
        self.geo = geo


def grids(tmp_path, cfg, creator=v3.create_configured):
    engine = source_solver.load_engine()
    from sosfuse2.solve import engine_core

    class StopSolver:
        def __init__(self, ctx, geo, *args):
            raise Captured(geo)

    state = ray_geometry.normal_state()
    with (
        patch.object(source_solver, 'load_engine', return_value=engine),
        patch.object(engine_core, 'Solver', StopSolver),
        patch.object(ray_geometry, 'Kernel', return_value=None),
        patch.object(host_cache, 'sun_geometry', return_value=None),
        patch.dict('os.environ', SOSFUSE2_HOST_CACHE='0'),
    ):
        with pytest.raises(Captured) as caught:
            creator(tmp_path / 'prepare', cfg, [550.0], 'numpy', state)
    return caught.value.geo


def test_defaults_and_compatible_overrides(tmp_path):
    cfg = settings(tmp_path)
    assert (cfg['nr'], cfg['h_refine']) == (44, 'below13mid_above13thirds')
    for nr, refine in ((20, 'none'), (39, 'mid'), (44, 'below13mid_above13thirds')):
        for raw in ({'nr': nr}, {'nr': nr, 'h_refine': refine}):
            assert settings(tmp_path, raw)['h_refine'] == refine
    reduced = settings(tmp_path, reduced=True)
    assert (reduced['nr'], reduced['h_refine'], reduced['nchi']) == (4, 'none', 10)
    assert grids(tmp_path, reduced).h.tobytes() == np.array([0.0, 0.1, 8.0, 80.0]).tobytes()


def test_reject_incompatible_pairs(tmp_path):
    for nr in (20, 39, 44):
        for refine in ('none', 'mid', 'below13mid_above13thirds', 'below2_above13x3', None, []):
            if (nr, refine) in ((20, 'none'), (39, 'mid'), (44, 'below13mid_above13thirds')):
                continue
            with pytest.raises(ValueError, match='E_SETTINGS_GRID:') as caught:
                settings(tmp_path, {'nr': nr, 'h_refine': refine})
            assert 'received=' + repr((nr, refine)) in str(caught.value)
            # an alias of another node count names the count it gives; anything else names the accepted forms
            assert ('h_refine gives nr' in str(caught.value)) == (refine in ('none', 'mid', 'below13mid_above13thirds'))
            assert 'h_refine gives nr' in str(caught.value) or 'h_refine must be none, mid' in str(caught.value)
    for raw, reduced in (({'nr': 30}, False), ({'nr': 44}, True), ({'h_refine': 'mid'}, True)):
        with pytest.raises(ValueError, match='E_SETTINGS_GRID:'):
            settings(tmp_path, raw, reduced)
    with pytest.raises(ValueError, match='E_SETTINGS_VALUE: nr'):
        settings(tmp_path, {'nr': []})


def test_actual_grids_match_original_harness_and_nr20(tmp_path):
    from sosfuse2.rays.shell_segments import grids as column_grids

    h, chi = column_grids(1)
    # Frozen pre-port harness V14 rule (SPEC.md height definition), preserving operation order and float64 bytes.
    expected = np.sort(
        np.r_[
            h,
            [(u + v) / 2 for u, v in zip(h[:-1], h[1:]) if v <= 13 + 1e-9],
            [u + (v - u) * f for u, v in zip(h[:-1], h[1:]) if u >= 13 - 1e-9 for f in (1 / 3, 2 / 3)],
        ]
    )
    cfg = settings(tmp_path)
    actual = grids(tmp_path, cfg)
    peer = grids(tmp_path, dict(cfg, **harness.GRIDS['V14']), harness.creator('V14', v3.SUN_SCALE))
    assert len(actual.h) == 44
    for other in (expected, peer.h):
        assert actual.h.tobytes() == other.tobytes()
    assert actual.chi.tobytes() == peer.chi.tobytes() == chi.tobytes()
    for nr, expected_h in ((20, h), (39, np.sort(np.r_[h, (h[:-1] + h[1:]) / 2]))):
        legacy = grids(tmp_path, settings(tmp_path, {'nr': nr}))
        assert legacy.h.tobytes() == expected_h.tobytes()
        assert legacy.chi.tobytes() == chi.tobytes()


def test_harness_variants_still_construct_their_grids(tmp_path):
    assert harness.GRIDS['B'] == {'nr': 20, 'h_refine': 'none'}
    assert harness.GRIDS['B_nphi16']['nr'] == 20
    for name, overrides in harness.GRIDS.items():
        cfg = dict(settings(tmp_path), **overrides)
        for scale in (v3.SUN_SCALE, 1):
            actual = grids(tmp_path, cfg, harness.creator(name, scale))
            assert (len(actual.h), len(actual.chi)) == (cfg['nr'], cfg['nchi']), name


def test_height_hash_in_metadata_and_signature(tmp_path):
    for nr in (4, 20, 39, 44):
        cfg = settings(tmp_path, {'nr': nr}, reduced=nr == 4)
        actual = grids(tmp_path, cfg)
        meta = v3.grid_metadata(cfg, actual.h)
        assert meta == v3.grid_metadata(cfg)
        assert meta['h_sha256'] == hashlib.sha256(actual.h.tobytes()).hexdigest()
        state = ray_geometry.normal_state()
        dumps = json.dumps
        contracts = []

        def record(value, **kwargs):
            if isinstance(value, dict) and value.get('contract') == v3.CONTRACT_VERSION:
                contracts.append(value)
            return dumps(value, **kwargs)

        with patch.object(v3.json, 'dumps', side_effect=record):
            first, _ = v3.signed_contract(state, cfg, 'numpy', nr == 4, 'depolarizing')
        assert contracts[0]['grid'] == meta
        with patch.object(v3, 'grid_metadata', return_value=dict(meta, h_sha256='changed')):
            second, _ = v3.signed_contract(state, cfg, 'numpy', nr == 4, 'depolarizing')
        assert first != second


def test_grid_count_mismatch_still_fails(tmp_path):
    cfg = settings(tmp_path)
    with pytest.raises(ValueError, match='E_SETTINGS_GRID_COUNT:'):
        grids(tmp_path, dict(cfg, nr=20))


def test_rayleigh_settings_and_legacy_adapters_load(tmp_path, monkeypatch):
    import shutil

    spec = importlib.util.spec_from_file_location('rayleigh_replay', HARNESS / 'run_rayleigh_bias.py')
    bias = importlib.util.module_from_spec(spec)
    with testkit.legacy_module_aliases('geometry', 'host_cache', 'fused', 'run_gpu_state_v3'):
        spec.loader.exec_module(bias)
    shutil.copytree(HARNESS / 'rayleigh_bias/settings', tmp_path / 'rayleigh_bias/settings')
    (tmp_path / 'states.json').write_text(
        json.dumps(
            dict(
                states=[
                    dict(
                        id='q2d_pure_rayleigh_control',
                        parameters=dict(ray_geometry.normal_state()['parameters'], stratosphere_tau550=0),
                    )
                ]
            )
        )
    )
    monkeypatch.setattr(bias, 'HERE', tmp_path)
    monkeypatch.setattr(host_cache, 'ROOT', host_cache.ROOT)  # run() assigns this shared module global.
    # Real run() settings/adapter paths; the existing CPU helper stops before optical preparation.
    with patch.dict('os.environ', {}, clear=False):
        for name, nr, nchi in (
            ('B', 20, 75),
            ('V1', 20, 75),
            ('V2', 20, 75),
            ('V3', 20, 75),
            ('V4', 39, 125),
            ('V5', 39, 75),
            ('V6', 20, 125),
            ('V7', 25, 75),
            ('V8', 34, 75),
            ('V9', 30, 75),
        ):
            a = SimpleNamespace(
                id='q2d_pure_rayleigh_control',
                reduced=False,
                first_only=False,
                variant=name,
                day=False,
                backend='numpy',
            )
            with testkit.legacy_module_aliases('run_gpu_state_v3'):
                actual = grids(tmp_path, {}, creator=lambda *args, **kwargs: bias.run(a))
            assert (len(actual.h), len(actual.chi)) == (nr, nchi), name
            assert actual.cfg == dict(
                nmu=24 if name == 'V2' else 12, nphi=64 if name == 'V1' else 16, path_n=24 if name == 'V3' else 12
            ), name
