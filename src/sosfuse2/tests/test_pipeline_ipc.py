"""Persistent-service IPC error path (pipeline_execution.wire_recv): a service failure string in place of the array
descriptor keeps its reason (R3 train0 4090, 2026-10-26: ENOSPC on the service side surfaced as a ValueError)."""

import multiprocessing as mp
import numpy as np
import pytest
from sosfuse2.pipeline_execution import wire_recv, wire_send


def _quiet(*_, **__):
    pass


def test_service_error_string_raises_with_its_reason():
    a, b = mp.Pipe()
    a.send('E_PERSISTENT_STATE: OSError: [Errno 28] No space left on device')
    with pytest.raises(RuntimeError, match=r'^E_PERSISTENT_STATE: OSError: \[Errno 28\]'):
        wire_recv(b, _quiet, 'S')
    a.send('unexpected text')
    with pytest.raises(RuntimeError, match='^E_PERSISTENT_STATE: unexpected text$'):
        wire_recv(b, _quiet, 'S')


def test_descriptor_round_trip_still_works(tmp_path):
    a, b = mp.Pipe()
    value = dict(x=np.arange(5.0), n=3)
    with wire_send(a, value, tmp_path, _quiet, 'S'):
        got = wire_recv(b, _quiet, 'S')
    assert got['n'] == 3 and np.array_equal(got['x'], value['x'])
