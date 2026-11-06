"""accepted/engine.py create() keyword hooks (1.2.0, formerly text patches): no GPU, no network, no sky-sos root, no compiler.

An enabled `troposphere_profile` block reaches the engine as `create(aerosol=tropo_profile.engine_hooks(state))`; the
production settings reach it as observer_km / kernel_factory / grid / spec / sun_geometry / prepare_ground
(run_gpu_state_v3.create_configured). These tests stop at the solver boundary (a StopSolver captures the geometry,
kernel and scope) so they cost a density build and the definition loads, not a solve.
Run: python -m pytest -q src/sosfuse2/test_tropo_patches.py
"""
import copy

import numpy as np
import pytest

from sosfuse2 import geometry as gm
from sosfuse2 import tropo_profile as tp
from sosfuse2.fused import load_engine

TWO_LAYER = dict(status='enabled', model='two_layer')


class Stop(Exception):
    pass


def stopping_engine(captured):
    """A fresh engine whose Solver records (geo, kernel, scope) and stops before any optical preparation."""
    engine = load_engine(); original = engine.definitions

    def definitions(relative, names, scope):
        result = original(relative, names, scope)
        if relative == 'solver_core.py' and 'Solver' in names:
            class StopSolver:
                def __init__(self, geo, kernel, *args):
                    captured.update(geo=geo, kernel=kernel, scope=scope); raise Stop

                def prepare_ground(self):
                    return 'original'
            scope['Solver'] = StopSolver
        return result
    engine.definitions = definitions
    return engine


def test_engine_hooks_absent_disabled_enabled():
    assert tp.engine_hooks(dict(parameters=dict(aod550=.15))) is None
    assert tp.engine_hooks(dict(parameters=dict(aod550=.15), troposphere_profile=dict(status='disabled', model='two_layer'))) is None
    hooks = tp.engine_hooks(dict(parameters=dict(aod550=.15), troposphere_profile=dict(TWO_LAYER)))
    assert hooks.column == 1. and len(hooks.vector) == 7 and hooks.tag
    h = np.array([0., 1., 5., 30.])
    np.testing.assert_allclose(hooks.profile(h), tp.profile(h, hooks.vector), rtol=0, atol=0)


def test_create_defaults_match_the_module_constants(tmp_path):
    captured = {}; engine = stopping_engine(captured)
    state = copy.deepcopy(gm.normal_state()); H = state['parameters']['scale_height_km']
    with pytest.raises(Stop):
        engine.create(state, tmp_path, 'selftest', 2, 550., kernel_factory=lambda t, H, g: ('kernel', t, H, g))
    geo, scope = captured['geo'], captured['scope']
    assert scope['b'].sh.OBS == engine.OBS and scope['OBS'] == engine.OBS if 'OBS' in scope else True
    assert geo.cfg == dict(nmu=1, nphi=2, path_n=4) and len(geo.h) == 4 and len(geo.chi) == 10        # selftest grid untouched
    assert geo.folder == tmp_path and type(geo.sun).__name__ == 'SunGeometry'                           # no aerosol folder, accepted sun
    old = scope['b'].sh.old
    np.testing.assert_allclose(old.NORMAL['aer'], state['parameters']['aod550']/(H*(-np.expm1(-80./H))), rtol=1e-15)
    # Two defaults, kept distinct on purpose: the column profile clamps h at 0, the solver's ha profile does not.
    assert old.profiles.__globals__['AER_PROFILE'](np.array([-1.])) == 1. and scope['AER_PROFILE'](np.array([-1.])) > 1.
    assert 'HOST_ORIGINAL_GROUND' not in scope and captured['kernel'][0] == 'kernel'


def test_create_hooks_reach_geometry_kernel_and_scope(tmp_path):
    captured = {}; engine = stopping_engine(captured)
    state = copy.deepcopy(gm.normal_state()); state['troposphere_profile'] = dict(TWO_LAYER)
    hooks = tp.engine_hooks(state); seen = {}

    def grid(mode, h, ch, nmu, nphi, path_n):
        seen['mode'] = mode; return h[:3], ch[:4], 7, 8, 9

    def sun(old, cache, H):
        seen['sun_cache'] = cache; return ('sun', H)
    with pytest.raises(Stop):
        engine.create(state, tmp_path, 'selftest', 2, 550., observer_km=.8, kernel_factory=lambda t, H, g: ('kernel', t),
                      grid=grid, sun_geometry=sun, prepare_ground=False, aerosol=hooks)
    geo, scope = captured['geo'], captured['scope']
    assert seen['mode'] == 'selftest' and len(geo.h) == 3 and len(geo.chi) == 4 and geo.cfg == dict(nmu=7, nphi=8, path_n=9)
    assert geo.folder == tmp_path and seen['sun_cache'] == tmp_path/'aerosol_profile'/hooks.tag and geo.sun == ('sun', state['parameters']['scale_height_km'])
    assert scope['b'].sh.OBS == .8 and scope['AER_PROFILE'] is hooks.profile
    old = scope['b'].sh.old
    assert old.profiles.__globals__['AER_PROFILE'] is hooks.profile and old.NORMAL['aer'] == state['parameters']['aod550']/hooks.column
    assert scope['Solver'].prepare_ground(None) is None and scope['HOST_ORIGINAL_GROUND'](None) == 'original'   # deferred: no-op until the caller restores
    assert captured['kernel'] == ('kernel', 2)                                                     # a non-native kernel takes no vector


def test_native_kernel_transport_passes_the_vector_only_when_set():
    engine = load_engine(); calls = []

    class Lib:
        def full_transport(self, *args):
            calls.append(args)
    kernel = object.__new__(engine.Kernel); kernel.lib = Lib(); kernel.height, kernel.g = 1.2, .7
    W = 2
    s = type('S', (), {})(); s.W = W; s.active = np.zeros(1, 'i4'); s.ext = np.ones(W); s.asca = np.array([.3, .4]); s.hr = s.ha = np.ones(3)
    s.geo = type('G', (), {})(); s.geo.chi = np.zeros(3); s.geo.poly = s.geo.left = np.zeros(3); s.geo.ns = 3
    s.goff = np.zeros(1, 'i8'); s.gdir = s.gval = s.gm = np.zeros(1, 'f4'); s.delta = .03; s.mode = 1
    z = {k: np.zeros(1) for k in ('nu', 'off', 'base', 'lit', 'point', 'target', 'dloc')}
    assert engine.Kernel.aerosol_vector is None
    kernel.transport(z, s, np.zeros(1))
    kernel.aerosol_vector = np.ascontiguousarray([0., 1., 1., 0., 0., 1., 2.], 'f8')
    kernel.transport(z, s, np.zeros(1))
    (plain, two_layer) = calls
    assert plain[-3] == 1.2 and two_layer[-3] == -1.                      # aerosol_height: the native kernel's two-layer switch
    assert plain[:7] == two_layer[:7] and len(plain) == len(two_layer)
