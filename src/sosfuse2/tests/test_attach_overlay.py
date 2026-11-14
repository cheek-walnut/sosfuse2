"""attach_cache_event overlay (run_gpu_state_v3): two processes attaching one key never share a '<key>.tmp.npz'.

CPU only, no solver. Each worker stands in for stratosphere._ground_geometry: it writes a fixed-name tmp file into
STRATO_ATTACH_CACHE, holds it, checks the bytes are still its own and renames it to a published attach file. The
geometry-only prefetch uses record=False, which keeps this overlay and writes no attach_cache PHASE record.
"""

import json
import multiprocessing
import os
import time
from pathlib import Path
from types import SimpleNamespace

from sosfuse2.foundation import paths

STATE = json.loads((paths.FIXTURES / 'states_lutv2_strato_usstd.json').read_text())['states'][0]
PUBLISHED = 'attach_geometry_' + 'a' * 24 + '.npz'


def attach_once(root, record, start, out):
    from sosfuse2.jobs import run_state as V3

    os.environ['STRATO_ATTACH_CACHE'] = root
    records = []
    V3.phase = lambda *args, **kw: records.append(args[1])
    start.wait()
    with V3.attach_cache_event('state', 0, STATE, record=record) as made:
        folder = Path(os.environ['STRATO_ATTACH_CACHE'])
        tmp = folder / 'key.tmp.npz'
        mine = str(os.getpid()).encode()
        tmp.write_bytes(mine)
        time.sleep(0.3)
        intact = tmp.read_bytes() == mine
        os.replace(tmp, folder / PUBLISHED)
        made.append((SimpleNamespace(strato=None), {}))
    out.put(dict(private=folder != Path(root), intact=intact, records=records))


def run_pair(tmp_path, record):
    ctx = multiprocessing.get_context('spawn')
    start, out = ctx.Event(), ctx.Queue()
    procs = [ctx.Process(target=attach_once, args=(str(tmp_path), record, start, out)) for _ in range(2)]
    for p in procs:
        p.start()
    start.set()
    results = [out.get(timeout=60) for _ in procs]
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    return results


def test_two_writers_of_one_key_stay_private(tmp_path):
    results = run_pair(tmp_path, record=False)
    assert all(r['private'] and r['intact'] for r in results)
    assert all(r['records'] == [] for r in results)  # record=False: no attach_cache PHASE
    assert (tmp_path / PUBLISHED).is_file()
    assert not list(tmp_path.glob('*.tmp.npz')) and not list(tmp_path.glob('overlay-*'))


def test_record_true_keeps_the_phase_record(tmp_path):
    results = run_pair(tmp_path, record=True)
    assert all(r['records'] == ['attach_cache'] for r in results)
