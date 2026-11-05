"""A1 stage timers.  Measurement only: no array is read, written or reordered.

Every stage records host wall time (perf_counter).  A stage marked device=True
additionally records a CUDA Event pair on the current stream when the backend
is CuPy; events are resolved (synchronized) only in report(), so the timed
code path is not serialized by the timer.  On NumPy only wall time exists.
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
        self.wall = defaultdict(float); self.count = defaultdict(int); self.events = []

    def bind(self, xp):
        """Kernel.__init__ binds the array module (numpy -> wall only)."""
        import numpy as np
        self.xp = None if xp is np else xp

    @contextmanager
    def stage(self, name, device=False):
        pair = None
        if device and self.xp is not None:
            pair = (self.xp.cuda.Event(), self.xp.cuda.Event()); pair[0].record()
        started = time.perf_counter()
        try:
            yield
        finally:
            if pair is not None:
                pair[1].record(); self.events.append((name, pair[0], pair[1]))
            self.wall[name] += time.perf_counter()-started; self.count[name] += 1

    def add(self, name, seconds):
        self.wall[name] += seconds; self.count[name] += 1

    def report(self):
        cuda = defaultdict(float)
        for name, begin, end in self.events:
            end.synchronize(); cuda[name] += float(self.xp.cuda.get_elapsed_time(begin, end))/1000
        return dict(wall_seconds=dict(self.wall), counts=dict(self.count), cuda_event_seconds=dict(cuda),
                    backend='cupy' if self.xp is not None else 'numpy',
                    note='wall = host perf_counter incl. Python; cuda_event_seconds = stream time between Event pairs (CuPy only)')

    def snapshot(self, reset=True):
        r = self.report()
        if reset: self.reset()
        return r

    def write(self, path, extra=None):
        data = dict(self.report(), **(extra or {}))
        Path(path).write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')
        return data


TIMERS = Timers()
stage = TIMERS.stage
