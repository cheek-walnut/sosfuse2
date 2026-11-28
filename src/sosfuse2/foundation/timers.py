"""Stage timers.  Measurement only: no array is read, written or reordered.

Every stage records host wall time (perf_counter).  A stage marked device=True
additionally records a CUDA Event pair on the current stream when the backend
is CuPy; events are resolved (synchronized) only in report(), so the timed
code path is not serialized by the timer. On NumPy only wall time exists.
The report names CuPy when its collection interval contains CUDA events.
"""

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path


class Timers:
    def __init__(self):
        self.xp = None
        self.reset()

    def reset(self):
        self.wall = defaultdict(float)
        self.count = defaultdict(int)
        self.events = []

    def bind(self, xp):
        """Set the event provider for device stages. NumPy selects wall timing only."""
        import numpy as np

        self.xp = None if xp is np else xp

    @contextmanager
    def stage(self, name, device=False):
        cuda = self.xp.cuda if device and self.xp is not None else None
        pair = None
        if cuda is not None:
            pair = (cuda.Event(), cuda.Event())
            pair[0].record()
        started = time.perf_counter()
        try:
            yield
        finally:
            if pair is not None:
                pair[1].record()
                self.events.append((name, cuda, pair[0], pair[1]))
            self.wall[name] += time.perf_counter() - started
            self.count[name] += 1

    def add(self, name, seconds):
        self.wall[name] += seconds
        self.count[name] += 1

    def report(self):
        cuda = defaultdict(float)
        for name, provider, begin, end in self.events:
            end.synchronize()
            cuda[name] += float(provider.get_elapsed_time(begin, end)) / 1000
        return dict(
            wall_seconds=dict(self.wall),
            counts=dict(self.count),
            cuda_event_seconds=dict(cuda),
            backend='cupy' if self.events else 'numpy',
            note='wall = host perf_counter incl. Python; cuda_event_seconds = stream time between Event pairs (CuPy only)',
        )

    def snapshot(self, reset=True):
        r = self.report()
        if reset:
            self.reset()
        return r

    def write(self, path, extra=None):
        data = dict(self.report(), **(extra or {}))
        Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
        return data


TIMERS = Timers()
stage = TIMERS.stage
