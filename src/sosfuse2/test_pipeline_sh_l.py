"""GPU promotion and the persistent payload keep the solver's SH order (753cfb4 review: they rebuilt Kernel at L32).

No CUDA: fused.Kernel is replaced by a recorder, so this runs on the Mac."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# geometry binds fused.Kernel at import; import it before Kernel is patched so the Recorder never leaks to later tests.
from sosfuse2 import fused, geometry, pipeline_execution as pe  # noqa: E402,F401


class Recorder:
    made = []

    def __init__(self, threads, height, g, backend, block_size, flags=None, L=32):
        self.height, self.g, self.block_size, self.flags, self.L = height, g, block_size, flags, L
        self.xp = np; self.raw = {}; self.attributes = {}; self.constants = {}
        Recorder.made.append(self)

    def sync(self):
        pass


@pytest.mark.parametrize('L', (16, 24, 32))
def test_promote_and_payload_keep_L(L):
    old = SimpleNamespace(height=1.2, g=.8, flags=('SPECIALIZE',), block_size=256, L=L, constants={})
    s = SimpleNamespace(kernel=old, W=15, ext=0, rf=0, af=0, asca=0, geo=SimpleNamespace(ns=1, h=0, chi=0))
    geometry = SimpleNamespace(s=None, bundle=None, ground_bundle=None)
    with mock.patch.object(fused, 'Kernel', Recorder), mock.patch.object(pe, 'promote_bundle', lambda b, xp: b):
        pe.promote_solver(s, geometry, 'cupy', 256, 2)
        assert s.kernel.L == L
        payload = pe.device_payload(s, geometry)
        assert payload['L'] == L
        restored, _ = pe.restore_device_payload(payload, ({}, {}))
        assert restored.kernel.L == L
