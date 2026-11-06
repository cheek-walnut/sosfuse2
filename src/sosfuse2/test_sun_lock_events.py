"""2026-10-21 development GPU host regressions: PHASE lines without timestamp_ns broke pipeline_trace.py,
and concurrent states built the same solar table at once. No CUDA, no spawn."""
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sosfuse2 import host_cache, sun_parallel  # noqa: E402


def test_every_phase_line_carries_trace_fields():
    # pipeline_trace.py:75 reads pid/timestamp_ns from every PHASE event of a state log.
    missing = []
    for path in sorted(HERE.glob('*.py')):
        if path.name.startswith('test_'):continue
        for m in re.finditer(r"event='PHASE'(?:[^\n]*\n){0,3}", path.read_text()):   # the dict call and its next lines
            line = m.group(0)
            if 'event_fields()' not in line and 'timestamp_ns' not in line and 'id=identifier' not in line:
                missing.append(f'{path.name}: {line[:80]}')
    assert not missing, missing


def test_concurrent_sun_builds_are_serialized(tmp_path):
    active = []; peak = []; built = []
    def prebuild(old, scale, folder):
        active.append(1); peak.append(len(active)); time.sleep(.2)
        path = folder/'sun_columns.npz'
        if not path.exists():path.write_bytes(b'x'); built.append(1)
        active.pop(); return None
    class Sun:
        def __init__(self, scale, folder):
            self.h = [0]*(80*scale+1); self.x = [0]*(400*scale+1); self.build_seconds = 0.
    old = SimpleNamespace(SunGeometry=Sun)
    with mock.patch.object(sun_parallel, 'prebuild', prebuild):
        threads = [threading.Thread(target=host_cache.sun_geometry, args=(old, 4, tmp_path, 1.2)) for _ in range(3)]
        for t in threads:t.start()
        for t in threads:t.join()
    assert max(peak) == 1 and len(built) == 1


def _fake(tmp_path):
    built = []
    def prebuild(old, scale, folder):
        time.sleep(.1)
        path = folder/'sun_columns.npz'
        if not path.exists():path.write_bytes(b'x'); built.append(folder)
    class Sun:
        def __init__(self, scale, folder):
            self.h = [0]*(80*scale+1); self.x = [0]*(400*scale+1); self.build_seconds = 0.
    def total_columns(p, d, n):
        return None
    return SimpleNamespace(SunGeometry=Sun, total_columns=total_columns, R=6371., Z=80.), prebuild, built


def test_states_share_one_table_through_the_geometry_cache(tmp_path):
    old, prebuild, built = _fake(tmp_path); shared = tmp_path/'geometry'
    states = [tmp_path/f'state{i}' for i in range(3)] + [tmp_path/'state3'/'aerosol_profile'/'tagA']
    with mock.patch.object(sun_parallel, 'prebuild', prebuild):
        threads = [threading.Thread(target=host_cache.sun_geometry, args=(old, 4, s, 1.2), kwargs=dict(shared=shared))
                   for s in states]
        for t in threads:t.start()
        for t in threads:t.join()
    # One table for the three plain states, a separate one for the aerosol profile; nothing in the state folders.
    assert len(built) == 2 and all(shared in b.parents for b in built)
    assert any('aerosol_profile' in b.parts for b in built)
    assert not any((s/'host_sun').exists() for s in states)


def test_unwritable_shared_root_falls_back_to_the_state_folder(tmp_path):
    old, prebuild, built = _fake(tmp_path); shared = tmp_path/'readonly'; shared.write_text('not a folder')
    with mock.patch.object(sun_parallel, 'prebuild', prebuild), pytest.warns(RuntimeWarning, match='HOST_SUN_SHARED_UNAVAILABLE'):
        host_cache.sun_geometry(old, 4, tmp_path/'state', 1.2, shared=shared)
    assert built == [tmp_path/'state'/'host_sun'/'scale4'/(1.2).hex()]


def test_profile_code_change_misses_the_shared_table(tmp_path):
    # the aerosol_profile tag hashes the profile config, so tropo_profile.py's code must key it.
    old, prebuild, built = _fake(tmp_path); shared = tmp_path/'geometry'; state = tmp_path/'aerosol_profile'/'tagA'
    real = Path.read_bytes
    for edit in (b'', b'# changed profile code'):
        def read_bytes(self, edit=edit):
            return real(self)+edit if self.name == 'tropo_profile.py' else real(self)
        with mock.patch.object(sun_parallel, 'prebuild', prebuild), mock.patch.object(Path, 'read_bytes', read_bytes):
            host_cache.sun_geometry(old, 4, state, 1.2, shared=shared)
    assert len(built) == 2 and built[0] != built[1]


def test_shared_write_failure_after_lock_falls_back_with_errno(tmp_path):
    # mkdir and the lock succeed, then prebuild's write is refused (e.g. folder made read-only).
    old, prebuild, built = _fake(tmp_path); shared = tmp_path/'geometry'
    def refusing(old_, scale, folder):
        if shared in folder.parents:raise PermissionError(13, 'Permission denied')
        prebuild(old_, scale, folder)
    with mock.patch.object(sun_parallel, 'prebuild', refusing), \
            pytest.warns(RuntimeWarning, match='HOST_SUN_SHARED_UNAVAILABLE: .* errno=13'):
        host_cache.sun_geometry(old, 4, tmp_path/'state', 1.2, shared=shared)
    assert built == [tmp_path/'state'/'host_sun'/'scale4'/(1.2).hex()]
