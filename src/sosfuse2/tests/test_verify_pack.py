"""jobs.run_state.verify_pack: a valid pack passes, a zstd failure keeps its exit code and stderr, an earlier reason wins.

CPU only. A fake `zstd` on PATH (a small Python script) replays a plain tar so the test needs no zstd and can force the
failure path deterministically; the real-zstd case runs only when zstd is installed.
"""

import io
import json
import os
import sys
import tarfile
from unittest.mock import Mock

import pytest

from sosfuse2.jobs import run_state as V3
from sosfuse2.checks import testkit

MEMBERS = ('table.npz', 'meta.json', 'preview_m03.png', 'preview_m06.png')


def plain_tar(path, signature, members=MEMBERS, **metadata):
    with tarfile.open(path, 'w') as t:
        for name in members:
            data = json.dumps(dict(complete=True, signature=signature, **metadata)).encode() if name == 'meta.json' else b'x' * 4096
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return path


def fake_zstd(tmp_path, monkeypatch, exit_code=0, stderr=''):
    """`zstd -q -dc <file>` that writes <file> unchanged (a plain tar), then exits with exit_code."""
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    script = bindir / 'zstd'
    script.write_text(
        f'#!{sys.executable}\nimport sys\n'
        'data = sys.stdin.buffer.read() if "-c" in sys.argv else open(sys.argv[-1], "rb").read()\n'
        'sys.stdout.buffer.write(data)\n'
        f'sys.stdout.flush()\nsys.stderr.write({stderr!r})\nsys.exit({exit_code})\n'
    )
    script.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bindir}{os.pathsep}{os.environ["PATH"]}')


@pytest.mark.parametrize('metadata', [{}, {'settings': None}])
def test_valid_pack_passes(tmp_path, monkeypatch, metadata):
    fake_zstd(tmp_path, monkeypatch)
    V3.verify_pack(plain_tar(tmp_path / 'p.tar', 'sig', **metadata), 'sig')


def test_zstd_failure_reports_exit_code_and_stderr(tmp_path, monkeypatch):
    fake_zstd(tmp_path, monkeypatch, exit_code=3, stderr='corrupted block detected')
    with pytest.raises(RuntimeError, match=r'E_PACK_VERIFY: zstd decode failed \(exit 3: corrupted block detected\)'):
        V3.verify_pack(plain_tar(tmp_path / 'p.tar', 'sig'), 'sig')


def test_identity_error_is_not_masked_by_the_exit_code(tmp_path, monkeypatch):
    fake_zstd(tmp_path, monkeypatch, exit_code=3)
    with pytest.raises(RuntimeError, match='E_PACK_VERIFY: members or identity'):
        V3.verify_pack(plain_tar(tmp_path / 'p.tar', 'sig'), 'other')


def test_real_zstd_round_trip(tmp_path):
    testkit.require_tool('zstd')
    import subprocess

    tar = plain_tar(tmp_path / 'p.tar', 'sig')
    packed = tmp_path / 'p.tar.zst'
    subprocess.run(['zstd', '-q', '-f', str(tar), '-o', str(packed)], check=True)
    V3.verify_pack(packed, 'sig')


def test_exported_batch_checks_loose_and_archived_first_order(tmp_path, monkeypatch, capsys):
    import numpy as np
    from sosfuse2.atmosphere.components import physics, optics_state
    from sosfuse2.jobs import tables_from_sources as batch
    from sosfuse2.outputs import pack_state, radiance_table as rt

    fake_zstd(tmp_path, monkeypatch)
    out = tmp_path / 'out/state'
    out.mkdir(parents=True)
    state = dict(id='state', runnable=True, parameters={})
    states = tmp_path / 'states.json'
    states.write_text(json.dumps(dict(states=[state])))
    source = tmp_path / 'tmp/state/source.npz'
    source.parent.mkdir(parents=True)
    source.touch()
    monkeypatch.setattr(rt, 'restored_settings', lambda *args: None)
    monkeypatch.setattr(batch, 'signature_of', lambda *args: ('exported', {}))
    monkeypatch.setattr(pack_state, 'previews', lambda path: None)
    monkeypatch.setattr(sys, 'argv', ['batch', '--states', str(states), '--work-roots', str(tmp_path),
                                     '--backend', 'numpy', '--reduced'])
    for key in batch.THREAD_KEYS:
        monkeypatch.delenv(key, raising=False)
    axes = [np.array([1.])] * 4
    data = np.ones((1, 1, 1, 1), 'f4')
    first = rt.save_first_order(out, [data], *axes)
    table = out / 'first_order.npz'
    rt.save_table(out / 'table.npz', data, *axes)
    meta = dict(complete=True, contract=V3.CONTRACT_VERSION, signature='exported', physics=physics(),
                optics_state=optics_state(state), settings=dict(export_first_order=True), **first)
    meta_file = out / 'meta.json'
    meta_file.write_text(json.dumps(meta))
    for name in ('preview_m03.png', 'preview_m06.png'):
        (out / name).write_bytes(b'preview')
    pack = tmp_path / 'pack/state.tar.zst'
    assert batch.main() == 0  # a missing pack is recoverable
    assert batch.output_complete(out, pack)
    rt.save_first_order(out, [data * 2], *axes)
    assert not batch.output_complete(out, pack)
    with monkeypatch.context() as patch:
        patch.setattr(V3, 'verify_pack', Mock(side_effect=PermissionError('archive read denied')))
        repack = Mock(wraps=pack_state.pack_state)
        patch.setattr(pack_state, 'pack_state', repack)
        assert batch.main() == 1
        status = capsys.readouterr().out
        assert f'FAIL {tmp_path.name}/state: E_PACK_VERIFY: PermissionError: archive read denied' in status
        repack.assert_not_called()
    assert batch.main() == 1
    assert f'FAIL {tmp_path.name}/state: E_PACK_VERIFY: first-order table differs' in capsys.readouterr().out
    with tarfile.open(pack) as archive, np.load(io.BytesIO(archive.extractfile(table.name).read())) as saved:
        np.testing.assert_allclose(saved['radiance'], data, rtol=1e-6, atol=0)
    table.write_bytes(b'')
    assert not batch.output_complete(out, pack)
    meta_file.write_text('[]')
    assert not batch.output_complete(out, pack)


@pytest.mark.parametrize('settings', ['bad', [], False])
def test_pack_and_verify_reject_malformed_settings(tmp_path, monkeypatch, settings):
    from sosfuse2.outputs.pack_state import pack_state

    fake_zstd(tmp_path, monkeypatch)
    (tmp_path / 'meta.json').write_text(json.dumps(dict(complete=True, settings=settings)))
    with pytest.raises(ValueError, match='E_FIRST_ORDER_SETTINGS'):
        pack_state(tmp_path, tmp_path / 'pack')
    with pytest.raises(ValueError, match='E_FIRST_ORDER_SETTINGS'):
        V3.verify_pack(plain_tar(tmp_path / 'bad.tar', 'sig', settings=settings), 'sig')


@pytest.mark.parametrize('axis', range(4))
def test_first_order_requires_matching_coordinates(tmp_path, axis):
    import numpy as np
    from sosfuse2.outputs import radiance_table as rt

    axes = [np.array([x]) for x in (450., -3., 10., 45.)]
    data = np.ones((1, 1, 1, 1), 'f4')
    rt.save_table(tmp_path / 'table.npz', data, *axes)
    axes[axis] += 1
    meta = rt.save_first_order(tmp_path, [data], *axes)
    assert not rt.first_order_complete(tmp_path, dict(settings=dict(export_first_order=True), **meta))
