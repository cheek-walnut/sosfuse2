"""GPU promotion and the persistent payload keep the solver's SH order (they rebuilt Kernel at L32).

No CUDA: source_solver.Kernel is replaced by a recorder, so this runs without a GPU."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

# ray_geometry binds source_solver.Kernel at import; import it before Kernel is patched so the Recorder never leaks to later tests.
from sosfuse2.solve import source_solver, ray_geometry
from sosfuse2.jobs import pipeline_execution as pe  # noqa: E402,F401


class Recorder:
    made = []

    def __init__(self, threads, height, g, backend, block_size, flags=None, L=32):
        self.height, self.g, self.block_size, self.flags, self.L = height, g, block_size, flags, L
        self.xp = np
        self.raw = {}
        self.attributes = {}
        self.constants = {}
        Recorder.made.append(self)

    def sync(self):
        pass


@pytest.mark.parametrize('L', (16, 24, 32))
def test_promote_and_payload_keep_L(L):
    old = SimpleNamespace(height=1.2, g=0.8, flags=('SPECIALIZE',), block_size=256, L=L, constants={})
    s = SimpleNamespace(kernel=old, W=15, ext=0, rf=0, af=0, asca=0, geo=SimpleNamespace(ns=1, h=0, chi=0))
    geometry_instance = SimpleNamespace(s=None, bundle=None, ground_bundle=None)
    with mock.patch.object(source_solver, 'Kernel', Recorder), mock.patch.object(pe, 'promote_bundle', lambda b, xp: b):
        pe.promote_solver(s, geometry_instance, 'cupy', 256, 2)
        assert s.kernel.L == L
        payload = pe.device_payload(s, geometry_instance)
        assert payload['L'] == L
        restored, _ = pe.restore_device_payload(payload, ({}, {}))
        assert restored.kernel.L == L
