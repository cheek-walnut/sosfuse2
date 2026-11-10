"""accepted/engine.py create() keyword hooks (1.2.0, formerly text patches): no GPU, no network, no sky-sos root, no compiler.

An enabled `troposphere_profile` block reaches the engine as `create(aerosol=tropo_profile.engine_hooks(state))`; the
production settings reach it as observer_km / kernel_factory / grid / spec / sun_geometry / prepare_ground
(run_gpu_state_v3.create_configured). These tests stop at the solver boundary (a StopSolver captures the geometry,
kernel and scope) so they cost a density build and the engine setup, not a solve.
Run: python -m pytest -q src/sosfuse2/tests/test_tropo_patches.py
"""

import copy
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from sosfuse2 import geometry as gm
from sosfuse2 import tropo_profile as tp
from sosfuse2.fused import load_engine

TWO_LAYER = dict(status='enabled', model='two_layer')


class Stop(Exception):
    pass


def stopping_engine(captured, monkeypatch):
    """Capture context and geometry at the module Solver boundary."""
    from sosfuse2 import solver_core

    class StopSolver:
        def __init__(self, ctx, geo, kernel, *args):
            captured.update(geo=geo, kernel=kernel, scope=ctx)
            raise Stop

        def prepare_ground(self):
            return 'original'

    monkeypatch.setattr(solver_core, 'Solver', StopSolver)
    return load_engine()


def test_engine_hooks_absent_disabled_enabled():
    assert tp.engine_hooks(dict(parameters=dict(aod550=0.15))) is None
    assert (
        tp.engine_hooks(
            dict(parameters=dict(aod550=0.15), troposphere_profile=dict(status='disabled', model='two_layer'))
        )
        is None
    )
    hooks = tp.engine_hooks(dict(parameters=dict(aod550=0.15), troposphere_profile=dict(TWO_LAYER)))
    assert hooks.column == 1.0 and len(hooks.vector) == 7 and hooks.tag
    h = np.array([0.0, 1.0, 5.0, 30.0])
    np.testing.assert_allclose(hooks.profile(h), tp.profile(h, hooks.vector), rtol=0, atol=0)


def test_create_defaults_match_the_module_constants(tmp_path, monkeypatch):
    captured = {}
    engine = stopping_engine(captured, monkeypatch)
    state = copy.deepcopy(gm.normal_state())
    H = state['parameters']['scale_height_km']
    with pytest.raises(Stop):
        engine.create(state, tmp_path, 'selftest', 2, 550.0, kernel_factory=lambda t, H, g: ('kernel', t, H, g))
    geo, scope = captured['geo'], captured['scope']
    assert scope.b.sh.OBS == engine.OBS
    assert (
        geo.cfg == dict(nmu=1, nphi=2, path_n=4) and len(geo.h) == 4 and len(geo.chi) == 10
    )  # selftest grid untouched
    assert geo.folder == tmp_path and type(geo.sun).__name__ == 'SunGeometry'  # no aerosol folder, accepted sun
    old = scope.b.sh.old
    np.testing.assert_allclose(
        old.NORMAL['aer'], state['parameters']['aod550'] / (H * (-np.expm1(-80.0 / H))), rtol=1e-15
    )
    # Two defaults, kept distinct on purpose: the column profile clamps h at 0, the solver's ha profile does not.
    assert old.AER_PROFILE(np.array([-1.0])) == 1.0 and scope.AER_PROFILE(np.array([-1.0])) > 1.0
    assert scope.prepare_ground and captured['kernel'][0] == 'kernel'


def test_create_hooks_reach_geometry_kernel_and_scope(tmp_path, monkeypatch):
    captured = {}
    engine = stopping_engine(captured, monkeypatch)
    state = copy.deepcopy(gm.normal_state())
    state['troposphere_profile'] = dict(TWO_LAYER)
    hooks = tp.engine_hooks(state)
    seen = {}

    def grid(mode, h, ch, nmu, nphi, path_n):
        seen['mode'] = mode
        return h[:3], ch[:4], 7, 8, 9

    def sun(old, cache, H):
        seen['sun_cache'] = cache
        return ('sun', H)

    with pytest.raises(Stop):
        engine.create(
            state,
            tmp_path,
            'selftest',
            2,
            550.0,
            observer_km=0.8,
            kernel_factory=lambda t, H, g: ('kernel', t),
            grid=grid,
            sun_geometry=sun,
            prepare_ground=False,
            aerosol=hooks,
        )
    geo, scope = captured['geo'], captured['scope']
    assert (
        seen['mode'] == 'selftest'
        and len(geo.h) == 3
        and len(geo.chi) == 4
        and geo.cfg == dict(nmu=7, nphi=8, path_n=9)
    )
    assert (
        geo.folder == tmp_path
        and seen['sun_cache'] == tmp_path / 'aerosol_profile' / hooks.tag
        and geo.sun == ('sun', state['parameters']['scale_height_km'])
    )
    assert scope.b.sh.OBS == 0.8 and scope.AER_PROFILE is hooks.profile
    old = scope.b.sh.old
    assert old.AER_PROFILE is hooks.profile and old.NORMAL['aer'] == state['parameters']['aod550'] / hooks.column
    assert scope.prepare_ground is False
    assert captured['kernel'] == ('kernel', 2)  # a non-native kernel takes no vector


def profile_scope(state, cache, monkeypatch):
    captured = {}
    with pytest.raises(Stop):
        stopping_engine(captured, monkeypatch).create(
            state,
            cache,
            'selftest',
            1,
            kernel_factory=lambda *args: None,
            sun_geometry=lambda *args: None,
            aerosol=tp.engine_hooks(state),
        )
    return captured['scope']


def test_two_layer_column_callables_freeze_and_pickle(tmp_path, monkeypatch):
    from sosfuse2 import column_parallel, column_table

    state = copy.deepcopy(gm.normal_state())
    state['troposphere_profile'] = dict(TWO_LAYER)
    old = profile_scope(state, tmp_path, monkeypatch).b.sh.old
    p = np.array([[0.0, 0.0, old.R + 0.1]])
    d = np.array([[0.0, 0.0, 1.0]])
    h = np.array([-1.0, 0.0, 1.5, 3.0, 80.0])
    for names in (['geometry_segments', 'probe'], ['profiles', 'probe']):
        frozen = column_parallel.freeze(old, [*names, "aerosol_descriptor"])
        worker = column_parallel.thaw(pickle.loads(pickle.dumps(frozen)))
        if names[0] == 'profiles':
            np.testing.assert_allclose(worker.profiles(h), old.profiles(h), rtol=1e-6, atol=0)
            # The selected descriptor travels with the worker bindings.
            assert column_table.aerosol(worker) == column_table.aerosol(old)
        else:
            actual = list(worker.geometry_segments(p, d, 24))
            expected = list(old.geometry_segments(p, d, 24))
            assert len(actual) == len(expected)
            for segment, reference in zip(actual, expected):
                for value, baseline in zip(segment, reference):
                    np.testing.assert_allclose(value, baseline, rtol=1e-6, atol=0)
    hooks = tp.engine_hooks(state)
    restored = pickle.loads(pickle.dumps(hooks))
    assert restored.profile == hooks.profile
    assert restored.folder(tmp_path) == hooks.folder(tmp_path)
    column = tp.column_function(SimpleNamespace(fuse_state=state))
    np.testing.assert_allclose(pickle.loads(pickle.dumps(column))(h), hooks.profile(h), rtol=1e-6, atol=0)
    assert column.descriptor() == column_table.aerosol(old)


def test_profile_identities_keep_state_dependencies(tmp_path, monkeypatch):
    from sosfuse2 import column_table

    states = [copy.deepcopy(gm.normal_state()) for _ in range(3)]
    states[1]['parameters']['scale_height_km'] = 2.4
    states[2]['troposphere_profile'] = dict(TWO_LAYER)
    old = [profile_scope(state, tmp_path, monkeypatch).b.sh.old for state in states]
    identities = [json.loads(o.SunGeometry.identity(1)) for o in old]
    assert [i['aerosol'] for i in identities] == [column_table.aerosol(o) for o in old]
    assert len({json.dumps(i, sort_keys=True) for i in identities}) == 3
    # the default and the two-layer state share the scale height: only the descriptor separates their tables
    assert identities[0]['aerosol_height'] == identities[2]['aerosol_height']
    assert column_table.aerosol(old[0]) != column_table.aerosol(old[2])
    for o, state in zip(old, states):
        assert o.AER_HEIGHT == state['parameters']['scale_height_km']
        expected = column_table.aerosol(o)
        # The explicit descriptor is sufficient, even without any function internals.
        assert column_table.aerosol(SimpleNamespace(aerosol_descriptor=expected)) == expected
        del o.aerosol_descriptor
        with pytest.raises(ValueError, match='E_COLUMN_AER_DESCRIPTOR:'):
            column_table.aerosol(o)
    with pytest.raises(ValueError, match='E_COLUMN_AER_DESCRIPTOR:'):
        column_table.aerosol(SimpleNamespace(profiles=lambda h: h))


def test_native_kernel_transport_passes_the_vector_only_when_set():
    engine = load_engine()
    calls = []

    class Lib:
        def full_transport(self, *args):
            calls.append(args)

    kernel = object.__new__(engine.Kernel)
    kernel.lib = Lib()
    kernel.height, kernel.g = 1.2, 0.7
    W = 2
    s = type('S', (), {})()
    s.W = W
    s.active = np.zeros(1, 'i4')
    s.ext = np.ones(W)
    s.asca = np.array([0.3, 0.4])
    s.hr = s.ha = np.ones(3)
    s.geo = type('G', (), {})()
    s.geo.chi = np.zeros(3)
    s.geo.poly = s.geo.left = np.zeros(3)
    s.geo.ns = 3
    s.goff = np.zeros(1, 'i8')
    s.gdir = s.gval = s.gm = np.zeros(1, 'f4')
    s.delta = 0.03
    s.mode = 1
    z = {k: np.zeros(1) for k in ('nu', 'off', 'base', 'lit', 'point', 'target', 'dloc')}
    assert engine.Kernel.aerosol_vector is None
    kernel.transport(z, s, np.zeros(1))
    kernel.aerosol_vector = np.ascontiguousarray([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 2.0], 'f8')
    kernel.transport(z, s, np.zeros(1))
    (plain, two_layer) = calls
    assert plain[-3] == 1.2 and two_layer[-3] == -1.0  # aerosol_height: the native kernel's two-layer switch
    assert plain[:7] == two_layer[:7] and len(plain) == len(two_layer)
