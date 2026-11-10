"""sosfuse2.cache_store, the sky-sos rig/bake/bake_run.py cache_new and the cold paths, without network: a fake rclone
(local folder as the store) records every call. Run: python -m pytest -q src/sosfuse2/tests/test_cache_store.py"""

import errno
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from sosfuse2 import cache_store as cs  # noqa: E402
from sosfuse2 import gate, paths, testing  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope='module')
def br():
    """rig/bake/bake_run.py: the sky-sos rig bake driver (consumer of the package, not part of it)."""
    return load('bake_run', testing.sky_sos('rig/bake/bake_run.py'))


@pytest.fixture(scope='module')
def eda():
    return load('eda_bake', testing.sky_sos('rig/bake/eda_bake.py'))


SIG, SIG2 = '54b6bf909981453cea6a', '0123456789abcdef0123'
HC = 'ab' * 32
COL, COL_BAD, COL_NOSIDE = 'c1' * 32, 'c2' * 32, 'c3' * 32
ATT, ATT_TMP, GROUP = 'aa' * 12, 'bb' * 12, 'fbd59228429a6a62'
SUN = 'geometry/host_sun/5ad941fb88adf28ff430/scale4/0x1.3333333333333p+0/sun_columns.npz'
FAKE = '''import json, os, shutil, sys, time
from pathlib import Path
a = sys.argv[1:]
if a[:1] == ['--config']: a = a[2:]
log = os.environ.get('FAKE_RCLONE_LOG')
if os.environ.get('FAKE_RCLONE_FAIL'): sys.stderr.write(os.environ['FAKE_RCLONE_FAIL']); sys.exit(7)
if a[0] == 'lsjson':
    root = Path(a[-1])
    if not root.is_dir(): print('['); sys.stderr.write('NOTICE: Failed to lsjson: error in ListJSON: directory not found'); sys.exit(3)
    rows = [dict(Path=p.relative_to(root).as_posix(), Size=p.stat().st_size) for p in sorted(root.rglob('*')) if p.is_file()]
    rows += [r for r in rows if r['Path'] == os.environ.get('FAKE_RCLONE_DUP')]
    print(json.dumps(rows)); sys.exit(0)
if a[0] == 'cat':
    data = Path(a[-1]).read_bytes()
    if '--head' in a: data = data[:int(a[a.index('--head')+1])]
    sys.stdout.buffer.write(data); sys.exit(0)
if a[0] == 'copy':
    files = Path(a[a.index('--files-from-raw')+1]).read_text().splitlines(); src, dst = Path(a[-2]), Path(a[-1])
    if log: open(log, 'a').write(json.dumps(dict(argv=a, files=files))+'\\n')
    time.sleep(float(os.environ.get('FAKE_RCLONE_DELAY', '0')))
    for f in files:
        d = dst/f
        if d.exists() and '--ignore-existing' in a: continue
        d.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src/f, d.with_name(d.name+'.partial')); os.replace(d.with_name(d.name+'.partial'), d)
    sys.exit(0)
sys.exit(9)
'''


@pytest.fixture
def store(tmp_path, monkeypatch):
    fake = tmp_path / 'bin/rclone'
    fake.parent.mkdir()
    fake.write_text(f'#!{sys.executable}\n' + FAKE)
    fake.chmod(0o755)
    conf = tmp_path / 'rclone.conf'
    conf.write_text('')
    remote = tmp_path / 'store'
    monkeypatch.setenv('SKY_CACHE_RCLONE', str(fake))
    monkeypatch.setattr(cs, 'CONFIG', str(conf))
    monkeypatch.setattr(cs, 'REMOTE', str(remote))
    monkeypatch.setenv('FAKE_RCLONE_LOG', str(tmp_path / 'rclone.log'))
    monkeypatch.setenv('TMPDIR', str(tmp_path / 'tmp'))
    monkeypatch.setenv('SOSFUSE2_WORK_ROOT', str(tmp_path / 'work'))
    env = dict(os.environ, SKY_CACHE_REMOTE=str(remote), SKY_CACHE_RCLONE_CONFIG=str(conf))  # for CLI subprocesses
    return NS(
        remote=remote,
        env=env,
        calls=lambda: [json.loads(x) for x in (tmp_path / 'rclone.log').read_text().splitlines()],
    )


def put(root, rel, data=b'x'):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    return p


def npz(root, rel, **arrays):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'wb') as f:
        np.savez(f, **arrays)
    return p


BASE = (
    'identity.json',
    'source_height.json',
    'packed/journal.jsonl',
    'packed/cell_off.bin',
    'packed/dirs.bin',
    'packed/layout.json',
    'packed/complete.json',
)


def endpoints(root, sig, complete=True, height=1.2):
    """A complete (or unfinished) endpoints folder; height changes source_height.json and dirs.bin like a build from
    another scale height (end_col[:,1] differs, the signature does not: geometry_contract.py:71-78)."""
    e = f'geometry/endpoints_{sig}'
    put(root, f'{e}/identity.json', '{"version": 3}\n')
    put(root, f'{e}/source_height.json', json.dumps(dict(height=height)) + '\n')
    put(root, f'{e}/packed/journal.jsonl', '{"cell": 0}\n')
    put(root, f'{e}/packed/cell_off.bin', b'\0' * 24)
    put(root, f'{e}/packed/dirs.bin', bytes([int(height * 10)]) * 96)
    if complete:
        put(
            root,
            f'{e}/packed/layout.json',
            json.dumps(dict(shapes=dict(cell_off=[3], dirs=[4, 3]), dtypes=dict(cell_off='i8', dirs='f8'))),
        )
        put(root, f'{e}/packed/complete.json', '{"cells": 2}\n')
    return e


def tree(root, height=1.2):
    """A cache folder as eda left it (2026-10-22 rebake widget layout) plus every kind of non-unit file."""
    e = endpoints(root, SIG, height=height)
    endpoints(root, SIG2, complete=False)
    put(root, 'geometry/.v3-leases.lock', '')
    put(root, f'geometry/{SIG}.lock', '')
    put(root, f'{e}/users/4242', 'v3\n')
    put(root, f'{e}/identity.json.4242.tmp', '{')
    put(root, f'{e}/aerosol/H_0x1.3333333333333p+0.f4', b'\2' * 16)
    put(root, f'{e}/aerosol/H_0x1.3333333333333p+0.f4.lock', '')
    put(root, f'geometry/endpoints_{SIG2}/aerosol/H_0x1p+0.f4', b'\3' * 16)
    body = b'PAYLOAD'
    put(root, f'{e}/host_cache/{HC}.hcache', json.dumps(dict(size=len(body))).encode() + b'\n' + body)
    put(root, f'{e}/host_cache/{HC}.lock', '')
    npz(root, SUN, h=np.zeros(2), x=np.zeros(2), col=np.zeros((2, 2)))
    put(root, str(Path(SUN).parent / '.build.lock'), '')
    put(root, str(Path(SUN).parent / 'sun_columns.k2j3.tmp'), 'partial')
    npz(root, f'attach/attach_geometry_{ATT}.npz', offsets=np.arange(3), suncol=np.ones(2))
    put(root, f'attach/attach_{GROUP}.done', '')
    put(root, f'attach/attach_{GROUP}.lock', '')
    put(root, f'attach/attach_geometry_{ATT_TMP}.tmp.npz', 'partial')
    put(root, 'attach/overlay-x1y2/attach_geometry_' + ATT + '.npz', 'link')
    for key, good in ((COL, True), (COL_BAD, False)):
        data = put(root, f'columns/{key}.npy', key[:4]).read_bytes()
        put(root, f'columns/{key}.sha256', cs.hashlib.sha256(data).hexdigest() if good else '0' * 64)
        put(root, f'columns/{key}.lock', '')
    put(root, f'columns/{COL_NOSIDE}.npy', 'no sidecar')
    return e


def unit_files(e):
    """Local files of every complete unit of tree() (what a mirror holds after a full pull)."""
    return {f'{e}/{f}' for f in BASE} | {
        f'{e}/aerosol/H_0x1.3333333333333p+0.f4',
        f'{e}/host_cache/{HC}.hcache',
        SUN,
        f'attach/attach_geometry_{ATT}.npz',
        f'attach/attach_{GROUP}.done',
        f'columns/{COL}.npy',
        f'columns/{COL}.sha256',
    }


def store_files(e):
    return unit_files(e) - {f'{e}/{f}' for f in BASE} | {e + '.tar'}


def files(root):
    return {p.relative_to(root).as_posix() for p in Path(root).rglob('*') if p.is_file()}


def content(root):
    return {f: (Path(root) / f).read_bytes() for f in files(root) if not f.endswith('.lock')}


def test_push_complete_units_one_object_per_geometry_key(tmp_path, store):
    """Only complete units travel (no lock, tmp, lease marker, overlay, unchecked column, unbuilt endpoints); the
    geometry key is one tar with its manifest first; markers go after all content; the source is not written to."""
    src = tmp_path / 'cache'
    e = tree(src)
    before = {p: p.stat().st_mtime_ns for p in src.rglob('*')}
    row = cs.push(src)
    assert files(store.remote) == store_files(e) and row['codes'] == ['E_CACHE_ORPHAN']
    assert row['orphans'] == [f'geometry/endpoints_{SIG2}/aerosol/H_0x1p+0.f4'] and row['files_sent'] == len(
        store_files(e)
    )
    m = cs.read_manifest((store.remote / (e + '.tar')).read_bytes()[: cs.HEAD])
    assert [r[0] for r in m['files']][-1] == 'packed/complete.json' and {r[0] for r in m['files']} == set(BASE)
    with tarfile.open(store.remote / (e + '.tar')) as t:
        assert t.getnames() == ['MANIFEST.json'] + [r[0] for r in m['files']]
    tar_call, single, markers = store.calls()
    assert tar_call['files'] == [e + '.tar'] and set(markers['files']) == {
        f'columns/{COL}.sha256',
        f'attach/attach_{GROUP}.done',
    }
    assert all('--ignore-existing' in c['argv'] for c in (tar_call, single, markers))
    assert {p: p.stat().st_mtime_ns for p in src.rglob('*')} == before and not (src / cs.STAGE).exists()
    assert cs.push(src)['files_sent'] == 0  # nothing new: nothing sent


def test_mixed_copies_are_refused_never_combined(tmp_path, store):
    """One key, two valid local copies built from different scale heights. The store must never end up with
    parts of both under one marker: a stored tar with another manifest is E_CACHE_CONFLICT, an unfinished loose (legacy)
    upload gets no marker, a stored npy without marker only gets ours if its bytes hash equal."""
    a, b = tmp_path / 'copyA', tmp_path / 'copyB'
    e = tree(a, height=1.2)
    tree(b, height=2.5)
    # The old per-file push interrupted after copy A's content: loose files, no complete.json.
    for f in ('identity.json', 'source_height.json', 'packed/dirs.bin'):
        put(store.remote, f'{e}/{f}', (a / e / f).read_bytes())
    put(store.remote, f'columns/{COL}.npy', 'A-bytes')  # A's npy, marker never sent
    row = cs.push(b)
    assert f'{e}/packed/complete.json' not in files(store.remote)  # no marker on A's loose content
    assert {c['key']: c['code'] for c in row['conflicts']} == {f'columns/{COL}': 'E_CACHE_CONFLICT'}
    assert (
        f'columns/{COL}.sha256' not in files(store.remote)
        and (store.remote / f'columns/{COL}.npy').read_text() == 'A-bytes'
    )
    stored = cs.read_manifest((store.remote / (e + '.tar')).read_bytes()[: cs.HEAD])
    assert stored['content_sha256'] == cs.manifest(b, cs.scan(b)[0][e])['content_sha256']
    again = cs.push(a)  # A now: its geometry differs from the stored B
    assert {c['key']: c['code'] for c in again['conflicts']} == {
        e: 'E_CACHE_CONFLICT',
        f'columns/{COL}': 'E_CACHE_CONFLICT',
    }
    assert cs.read_manifest((store.remote / (e + '.tar')).read_bytes()[: cs.HEAD]) == stored and again['codes'] == [
        'E_CACHE_CONFLICT',
        'E_CACHE_ORPHAN',
    ]
    mirror = tmp_path / 'm'
    got = cs.pull(mirror, keys=[e])  # the mirror gets B entirely, nothing of A
    assert got['codes'] == [] and all((mirror / e / f).read_bytes() == (b / e / f).read_bytes() for f in BASE)
    # A stored npy that IS ours gets its marker; a store that lists one object twice is refused.
    shutil.rmtree(store.remote / 'columns')
    put(store.remote, f'columns/{COL}.npy', (b / f'columns/{COL}.npy').read_bytes())
    assert (
        'E_CACHE_CONFLICT' not in cs.push(b)['codes']
        and (store.remote / f'columns/{COL}.sha256').read_text() == (b / f'columns/{COL}.sha256').read_text()
    )
    os.environ['FAKE_RCLONE_DUP'] = e + '.tar'
    try:
        assert {c['code'] for c in cs.push(b)['conflicts']} == {'E_CACHE_DUPLICATE'} and e not in cs.group_remote(
            *cs.remote_listing()
        )
    finally:
        del os.environ['FAKE_RCLONE_DUP']


def test_pull_publishes_under_the_pipeline_locks_and_two_pulls_race(tmp_path, store):
    src = tmp_path / 'cache'
    e = tree(src)
    cs.push(src)
    mirror = tmp_path / 'sky-cache'
    (mirror / 'geometry').mkdir(parents=True)
    # A pipeline holding this signature's builder lock (run_gpu_state_v3.py:470) delays the endpoints publication.
    holder = subprocess.Popen(
        [
            sys.executable,
            '-c',
            'import fcntl,sys,time; f=open(sys.argv[1],"a"); fcntl.flock(f,fcntl.LOCK_EX); '
            'open(sys.argv[2],"w").write("1"); time.sleep(1.5)',
            str(mirror / f'geometry/{SIG}.lock'),
            str(tmp_path / 'held'),
        ]
    )
    while not (tmp_path / 'held').exists():
        time.sleep(0.02)
    t0 = time.monotonic()
    row = cs.pull(mirror)
    waited = time.monotonic() - t0
    holder.wait()
    assert waited >= 1.2 and row['codes'] == [] and row['published'] == 7
    assert content(mirror) == {f: (src / f).read_bytes() for f in unit_files(e)} and not (mirror / cs.STAGE).exists()
    for rel in (
        f'{e}/packed/complete.json',
        f'{e}/aerosol/H_0x1.3333333333333p+0.f4',
        SUN,
        f'attach/attach_geometry_{ATT}.npz',
        f'columns/{COL}.npy',
    ):
        assert (mirror / rel).is_file(), rel  # the paths pipeline.py & co. open
    # Two pulls into one new mirror at once (slow transfers overlap): every unit published once, the rest 'present'.
    race = tmp_path / 'race'
    env = dict(store.env, FAKE_RCLONE_DELAY='0.4')
    procs = [
        subprocess.Popen(
            paths.module_command('cache_store', 'pull', '--all', race, '--report', tmp_path / f'r{i}.json'),
            env=paths.child_env(env),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        for i in (1, 2)
    ]
    assert [p.wait(timeout=120) for p in procs] == [0, 0], [p.stderr.read()[-400:] for p in procs]
    rows = [json.loads((tmp_path / f'r{i}.json').read_text()) for i in (1, 2)]
    assert (
        sum(r['published'] for r in rows) == 7
        and all(r['codes'] == [] for r in rows)
        and content(race) == content(mirror)
    )
    assert not (race / cs.STAGE).exists()
    # One geometry key by name (SPEC.md CLI order); a file the mirror already has is kept.
    other = tmp_path / 'other'
    put(other, f'columns/{COL}.npy', 'LOCAL')
    assert cs.main(['pull', '--keys', e, f'columns/{COL}', str(other)]) == 0
    assert set(content(other)) == {f for f in unit_files(e) if f.startswith(e + '/')} | {f'columns/{COL}.npy'}
    assert (other / f'columns/{COL}.npy').read_text() == 'LOCAL' and cs.pull(other, keys=[e])['files'] == 0


def test_gf_and_eda_cold_do_not_use_the_mirror(tmp_path, monkeypatch, br, eda):
    mirror = tmp_path / 'sky-cache'
    monkeypatch.setattr(cs, 'MIRROR', str(mirror))
    monkeypatch.setenv('SKY_CACHE_MIRROR', str(mirror))
    seen = []

    def child(cmd, log, env, cwd=None, timeout=None):
        seen.append((cmd, env))
        return 1, [], 'E_STOP'

    monkeypatch.setattr(gate, 'child', child)
    reference = tmp_path / 'reference.npz'
    np.savez(reference, radiance=np.ones(1))
    monkeypatch.setattr(gate, 'manifest', lambda: dict(gf=dict(radiance_sha256=gate.sha(np.ones(1)))))
    args = NS(out=tmp_path / 'gate', reference=reference, decision=None)
    with pytest.raises(gate.GateError, match='E_GATE_WORKER'):
        gate.gf(args, gate.load_fixtures(), dict(gates={}))
    ((cmd, env),) = seen
    work = tmp_path / 'gate/gf/work'
    assert cmd[cmd.index('--geometry-cache') + 1] == str(work / 'cache/geometry') and cmd[
        cmd.index('--column-cache') + 1
    ] == str(work / 'cache/columns')
    assert env['STRATO_ATTACH_CACHE'] == str(work / 'cache/attach') and str(mirror) not in json.dumps(
        [cmd, {k: v for k, v in env.items() if k != 'SKY_CACHE_MIRROR'}]
    )
    work.mkdir(parents=True, exist_ok=True)  # the E_GATE_CONFIG cold check still stands
    with pytest.raises(gate.GateError, match='E_GATE_CONFIG'):
        gate.gf(args, gate.load_fixtures(), dict(gates={}))
    states = tmp_path / 'states.json'
    states.write_text(json.dumps(dict(states=[dict(gate.load_fixtures()['state'], runnable=True)])))
    sid = gate.load_fixtures()['state']['id']
    mods = (gate, cs, br)
    base = dict(PATH='/usr/bin')
    cold = eda.plan(
        eda.parse(['--states', str(states), '--ids', sid, '--out', str(tmp_path / 'run'), '--cold']), mods, base
    )
    c = cold.cmds['C3']
    assert (
        cold.cache == (tmp_path / 'run/work/cache').resolve()
        and not cold.push
        and str(mirror) not in json.dumps([c, cold.envs])
    )
    assert c[c.index('--geometry-cache') + 1] == str(cold.cache / 'geometry') and cold.envs['C3'][
        'STRATO_ATTACH_CACHE'
    ] == str(cold.cache / 'attach')
    warm = eda.plan(
        eda.parse(['--states', str(states), '--ids', sid, '--out', str(tmp_path / 'run2'), '--workers', '1']),
        mods,
        base,
    )
    w = warm.cmds['C3']
    assert (
        warm.cache == mirror.resolve() and warm.push and '--keep-geometry' in w and w[w.index('--workers') + 1] == '1'
    )
    assert w[w.index('--column-cache') + 1] == str(mirror.resolve() / 'columns') and warm.envs['C3'][
        'STRATO_ATTACH_CACHE'
    ] == str(mirror.resolve() / 'attach')
    assert (
        warm.envs['C3']['SOSFUSE2_KERNEL_FLAGS']
        == gate.fixture_env(gate.load_fixtures(), 'FX-C3', base)['SOSFUSE2_KERNEL_FLAGS']
    )
    # a push that reports any code makes the eda run exit 12, not 0.
    assert [
        eda.final_exit(True, dict(codes=[])),
        eda.final_exit(True, dict(codes=['E_CACHE_ORPHAN'])),
        eda.final_exit(True, dict(code='E_CACHE_RCLONE')),
        eda.final_exit(False, {}),
    ] == [0, 12, 12, 10]


def test_bake_run_cache_new_cap_disk_and_enospc(tmp_path, store, monkeypatch, br):
    P = br.layout(tmp_path / 'job')
    cache = P.work / 'cache'
    e = tree(cache)
    P.out.mkdir(parents=True)
    before = br.cache_snapshot(P)
    new_e = endpoints(cache, 'f' * 20)
    put(cache, f'{new_e}/aerosol/H_0x1p+1.f4', b'\4' * 8)
    data = put(cache, f'columns/{"d" * 64}.npy', 'new column').read_bytes()
    put(cache, f'columns/{"d" * 64}.sha256', cs.hashlib.sha256(data).hexdigest())
    put(cache, f'{e}/aerosol/H_0x1p+2.f4', b'\5' * 8)  # new column of an endpoints folder from the input
    put(cache, f'columns/{"e" * 64}.npy', 'unpublished')  # no sidecar: never taken
    row = br.collect_cache_new(P, before)
    base = {f'{new_e}/{f}' for f in BASE}
    want = {
        f'{new_e}/aerosol/H_0x1p+1.f4',
        f'{e}/aerosol/H_0x1p+2.f4',
        f'columns/{"d" * 64}.npy',
        f'columns/{"d" * 64}.sha256',
    } | base
    assert files(P.out / 'cache_new') == want and row['codes'] == [] and row['cap_bytes'] == br.CACHE_NEW_MAX
    pushed = cs.push(P.out / 'cache_new')  # the Mac's push after retrieval
    assert files(store.remote) == {
        f'{new_e}/aerosol/H_0x1p+1.f4',
        f'columns/{"d" * 64}.npy',
        f'columns/{"d" * 64}.sha256',
        new_e + '.tar',
    }
    assert pushed['orphans'] == [f'{e}/aerosol/H_0x1p+2.f4']  # its endpoints folder is not in this store
    shutil.rmtree(P.out / 'cache_new')  # cap below the endpoints unit
    small = br.collect_cache_new(P, before, cap=200)
    assert small['codes'] == ['E_CACHE_NEW_CAP', 'E_CACHE_ORPHAN'] and set(small['taken']) == {
        f'{e}/aerosol/H_0x1p+2.f4',
        f'columns/{"d" * 64}',
    }
    shutil.rmtree(P.out / 'cache_new')  # results already fill out/: no room
    with open(P.out / 'states.bin', 'wb') as f:
        f.truncate(int(br.RESULT_MAX - 5e7))
    full = br.collect_cache_new(P, before)
    assert full['cap_bytes'] == 0 and full['taken'] == [] and full['codes'] == ['E_CACHE_NEW_CAP', 'E_CACHE_ORPHAN']
    (P.out / 'states.bin').unlink()
    # the free disk bounds the cap, and an ENOSPC mid-copy removes out/cache_new and still writes the record.
    real = shutil.disk_usage
    monkeypatch.setattr(br.shutil, 'disk_usage', lambda p: real(p)._replace(free=int(br.CACHE_NEW_DISK_RESERVE + 300)))
    tight = br.collect_cache_new(P, before)
    assert (
        tight['cap_bytes'] == 300
        and 'E_CACHE_NEW_CAP' in tight['codes']
        and tight['disk_free_bytes'] == br.CACHE_NEW_DISK_RESERVE + 300
    )
    monkeypatch.setattr(br.shutil, 'disk_usage', real)
    shutil.rmtree(P.out / 'cache_new', ignore_errors=True)
    copy, calls = shutil.copy2, []

    def full_disk(s, d):
        calls.append(d)
        if len(calls) == 3:
            Path(d).write_bytes(b'half')
            raise OSError(errno.ENOSPC, 'No space left on device')
        return copy(s, d)

    monkeypatch.setattr(br.shutil, 'copy2', full_disk)
    hit = br.collect_cache_new(P, before)
    assert not (P.out / 'cache_new').exists() and hit['taken'] == [] and 'E_CACHE_NEW_ENOSPC' in hit['codes']
    assert json.loads((P.out / 'cache_new.json').read_text())['codes'] == hit['codes']


def test_reason_codes_are_kept_and_secrets_removed(tmp_path, store, monkeypatch, capsys):
    src = tmp_path / 'cache'
    e = tree(src)
    monkeypatch.setenv(
        'FAKE_RCLONE_FAIL',
        'ERROR : upload https://www.googleapis.com/upload/drive?access_token=SECRET123 failed: '
        'Authorization: Bearer ya29.SECRET456 token=SECRET789',
    )
    assert cs.main(['push', str(src)]) == 2
    out = capsys.readouterr()
    text = out.out + out.err
    assert '"code": "E_CACHE_RCLONE"' in text and 'rc=7' in text and 'SECRET' not in text and 'googleapis' not in text
    monkeypatch.delenv('FAKE_RCLONE_FAIL')
    with pytest.raises(cs.CacheStoreError) as exc:
        cs.pull(tmp_path / 'm', keys=['columns/' + '9' * 64])
    assert exc.value.code == 'E_CACHE_KEY_UNKNOWN'
    cs.push(src)
    put(store.remote, f'columns/{COL}.npy', 'flipped bytes')  # store copy damaged
    tar = store.remote / (e + '.tar')
    raw = bytearray(tar.read_bytes())
    with tarfile.open(tar) as t:
        at = t.getmember('packed/dirs.bin').offset_data
    raw[at] ^= 1
    tar.write_bytes(bytes(raw))  # one bit in a member
    row = cs.pull(tmp_path / 'm')
    assert row['failed'] == {
        f'columns/{COL}': 'E_CACHE_STAGED_INCOMPLETE',
        e: 'E_CACHE_STAGED_INCOMPLETE',
        f'{e}/aerosol/H_0x1.3333333333333p+0.f4': 'E_CACHE_ORPHAN',
    }
    assert (
        not (tmp_path / f'm/columns/{COL}.npy').exists()
        and not (tmp_path / 'm' / e / 'packed').exists()
        and cs.main(['pull', '--all', str(tmp_path / 'm2')]) == 1
    )
    monkeypatch.setattr(cs, 'CONFIG', str(tmp_path / 'none.conf'))
    with pytest.raises(cs.CacheStoreError, match='E_CACHE_CONFIG_MISSING'):
        cs.list_store()
    monkeypatch.setenv('SKY_CACHE_RCLONE', str(tmp_path / 'no-rclone'))
    with pytest.raises(cs.CacheStoreError, match='E_CACHE_RCLONE_MISSING'):
        cs.rclone_exe()
    assert (
        cs.clean('see https://x.r2.dev/a?X-Amz-Signature=abc and X-Amz-Credential: AKIA123')
        == 'see <url> and X-Amz-Credential=<redacted>'
    )


def test_list_without_a_remote_is_a_coded_error(monkeypatch):
    """No built-in store: with neither remote= nor SKY_CACHE_REMOTE a store operation stops with E_CACHE_REMOTE_MISSING."""
    monkeypatch.setattr(cs, 'REMOTE', None)
    monkeypatch.delenv('SKY_CACHE_REMOTE', raising=False)
    with pytest.raises(cs.CacheStoreError) as exc:
        cs.list_store()
    assert exc.value.code == 'E_CACHE_REMOTE_MISSING'


def test_pack_for_rig_tar_and_retry_after_failure(tmp_path, store, monkeypatch):
    src = tmp_path / 'cache'
    e = tree(src)
    cs.push(src)
    out = tmp_path / 'rig/cache.tar'
    out.parent.mkdir()
    monkeypatch.setenv('FAKE_RCLONE_FAIL', 'ERROR : drive quota')  # pull fails inside pack
    with pytest.raises(cs.CacheStoreError, match='E_CACHE_RCLONE'):
        cs.pack_for_rig(None, out)
    assert sorted(p.name for p in out.parent.iterdir()) == []  # no .mirror, no .part, no tar
    monkeypatch.delenv('FAKE_RCLONE_FAIL')
    row = cs.pack_for_rig(None, out)  # the same command again
    with tarfile.open(out) as t:
        names = set(t.getnames())
        t.extractall(tmp_path / 'work', filter='data')
    assert names == {'cache/' + f for f in unit_files(e)} and row['sha256'] == cs._sha_file(out)
    assert (
        sorted(p.name for p in out.parent.iterdir()) == ['cache.tar'] and (tmp_path / 'work/cache/geometry').is_dir()
    )  # bake_run cache_input
    assert set(cs.scan(tmp_path / 'work/cache')[0]) == set(cs.scan(src)[0]) - {
        f'geometry/endpoints_{SIG2}/aerosol/H_0x1p+0.f4'
    }
    one = cs.pack_for_rig([f'columns/{COL}'], tmp_path / 'rig/one.tar', local=src)
    with tarfile.open(tmp_path / 'rig/one.tar') as t:
        assert set(t.getnames()) == {f'cache/columns/{COL}.npy', f'cache/columns/{COL}.sha256'}
    assert one['units'] == 1


def test_legacy_loose_store_is_read_only_and_migrates(tmp_path, store):
    """The first fill stored the geometry key as loose files. list marks it, pull skips it unless --legacy,
    push never adds to it but stores the tar beside it; afterwards the tar is what pull uses. Nothing is deleted."""
    src = tmp_path / 'cache'
    e = tree(src)
    for f in unit_files(e):
        put(store.remote, f, (src / f).read_bytes())  # 10-23 layout
    listing = cs.list_store()
    assert {r['key']: r['kind'] for r in listing['units']}[e] == 'endpoints(legacy)'
    plain = cs.pull(tmp_path / 'm1')
    assert plain['legacy_skipped'] == [e] and not (tmp_path / 'm1' / e / 'packed').exists()
    assert plain['failed'] == {f'{e}/aerosol/H_0x1.3333333333333p+0.f4': 'E_CACHE_ORPHAN'}
    old = cs.pull(tmp_path / 'm2', legacy=True)
    assert old['codes'] == [] and content(tmp_path / 'm2') == {f: (src / f).read_bytes() for f in unit_files(e)}
    loose = {
        f
        for f in files(store.remote)
        if f.startswith(e + '/packed/') or f in (e + '/identity.json', e + '/source_height.json')
    }
    row = cs.push(src)  # migration: the tar appears beside
    assert row['codes'] == ['E_CACHE_ORPHAN'] and row['files_sent'] == 1 and (store.remote / (e + '.tar')).is_file()
    assert loose <= files(store.remote) and cs.list_store()['legacy_superseded'] == sorted(loose)
    new = cs.pull(tmp_path / 'm3')
    assert new['codes'] == [] and new['legacy_skipped'] == []
    assert content(tmp_path / 'm3') == content(tmp_path / 'm2')
