"""run_gpu_state_v3.verify_pack: a valid pack passes, a zstd failure keeps its exit code and stderr, an earlier reason wins.

CPU only. A fake `zstd` on PATH (a small Python script) replays a plain tar so the test needs no zstd and can force the
failure path deterministically; the real-zstd case runs only when zstd is installed.
"""

import io
import json
import os
import sys
import tarfile

import pytest

from sosfuse2.jobs import run_state as V3
from sosfuse2.checks import testkit

MEMBERS = ('table.npz', 'meta.json', 'preview_m03.png', 'preview_m06.png')


def plain_tar(path, signature, members=MEMBERS):
    with tarfile.open(path, 'w') as t:
        for name in members:
            data = json.dumps(dict(complete=True, signature=signature)).encode() if name == 'meta.json' else b'x' * 4096
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
        f'#!{sys.executable}\nimport sys\nsys.stdout.buffer.write(open(sys.argv[-1], "rb").read())\n'
        f'sys.stdout.flush()\nsys.stderr.write({stderr!r})\nsys.exit({exit_code})\n'
    )
    script.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bindir}{os.pathsep}{os.environ["PATH"]}')


def test_valid_pack_passes(tmp_path, monkeypatch):
    fake_zstd(tmp_path, monkeypatch)
    V3.verify_pack(plain_tar(tmp_path / 'p.tar', 'sig'), 'sig')


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
