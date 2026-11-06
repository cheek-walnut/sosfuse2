"""Explicit one-time preparation of the unbundled HITRAN O4 table."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

from . import generate_minor_absorbers as generator
from .minor_absorbers import load, o4_path


def prepare(cia=None, *, download=False):
    target = o4_path()
    if download and os.environ.get('SOSFUSE2_OFFLINE', '') not in ('', '0'):
        raise ValueError(f'E_O4_OFFLINE: download {generator.CIA_URL} separately and use --cia FILE')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='minor-', dir=target.parent) as work:
        if download:
            cia = Path(work)/'input.cia'
            try:
                with urllib.request.urlopen(generator.CIA_URL, timeout=60) as response:
                    cia.write_bytes(response.read())
            except OSError as exc:
                raise RuntimeError(f'E_O4_DOWNLOAD: {generator.CIA_URL}: {type(exc).__name__}: {exc}') from exc
        doc = generator.build_o4(cia)  # validates the pinned original SHA before parsing
        tmp = Path(work)/'asset.json'
        tmp.write_text(json.dumps(doc, indent=1, allow_nan=False)+'\n')
        load('o4', path=tmp)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        os.replace(tmp, target)
    return dict(path=str(target), source_sha256=generator.CIA_SHA, sha256=digest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--cia', type=Path)
    group.add_argument('--download', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.cia, download=args.download), sort_keys=True))


if __name__ == '__main__':
    main()
