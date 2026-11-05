"""tropo_profile source patches against the accepted engine loader (accepted/engine.py): no GPU, no network, no sky-sos root.

An enabled `troposphere_profile` block rewrites the text of engine.create() and Kernel.transport(). Each old string must
match exactly once in the current loader, or the block fails with E_ADAPTER_DRIFT for every state that enables it
(the 3-tuple (old, new, 2) of accepted/engine.py:94 once made PATCHES[4] match zero times).
Run: python -m pytest -q src/sosfuse2/test_tropo_patches.py
"""
import inspect
import re
import textwrap

import pytest

from sosfuse2 import tropo_profile as tp
from sosfuse2.fused import load_engine

STATE = dict(parameters=dict(aod550=.15), troposphere_profile=dict(status='enabled', model='two_layer'))


@pytest.fixture(scope='module')
def engine():
    return load_engine()


def test_patch_old_strings_match_the_loader_exactly_once(engine):
    create = inspect.getsource(engine.create)
    for old, _ in tp.PATCHES:
        assert create.count(old) == 1, f'PATCHES old string matches {create.count(old)} time(s) in engine.create: {old!r}'
    assert sum(len(re.findall(old, create)) == 1 for old, _ in tp.SUN) == 1, 'exactly one SUN constructor pattern must match once'
    transport = textwrap.dedent(inspect.getsource(engine.Kernel.transport))
    for old, _ in tp.TRANSPORT:
        assert transport.count(old) == 1, f'TRANSPORT old string matches {transport.count(old)} time(s) in Kernel.transport: {old!r}'


def test_patch_source_has_no_drift_and_compiles(engine):
    create = inspect.getsource(engine.create)
    scope = {}
    patched = tp.patch_source(create, STATE, scope)
    assert patched != create
    compile(patched, 'tropo_profile_engine_create', 'exec')
    assert {'AER_PROFILE', 'AER_COLUMN', 'AER_FOLDER', 'AER_KERNEL'} <= set(scope)
    unchanged = {'parameters': STATE['parameters']}
    assert tp.patch_source(create, unchanged, {}) is create          # no block: source returned as the same object


def test_patch_source_reports_drift_when_a_count_changes(engine):
    create = inspect.getsource(engine.create)
    drifted = create.replace("AER_HEIGHT)',2)", "AER_HEIGHT)',3)")
    assert drifted != create
    with pytest.raises(RuntimeError, match='E_ADAPTER_DRIFT'):
        tp.patch_source(drifted, STATE, {})


def test_transport_wrapper_applies_without_a_compiled_kernel(engine):
    kernel = object.__new__(engine.Kernel); kernel.lib = object()    # a native Kernel without the CDLL (no compiler needed)
    out = tp.wrap_kernel(kernel, dict(engine.__dict__), [0., 1., 1., 0., 0., 1., 2.])
    assert out is kernel and hasattr(kernel, 'aerosol_vector')
    assert kernel.transport.__func__.__name__ == 'transport'
