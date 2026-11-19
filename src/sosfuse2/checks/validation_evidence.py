"""Narrow read-only evidence for the exact reference inputs used by validation."""

import hashlib
from ..foundation import paths
from ..solve.source_solver import load_engine
from .validate import LEGACY, LEGACY_META, legacy_available


def key(path):
    """Use a package-relative key or prefix an upstream checkout path with 'upstream/'."""
    root = paths.sky_sos_root()
    if root is not None:
        try:
            return 'upstream/' + path.resolve().relative_to(root).as_posix()
        except ValueError:
            pass
    return paths.rel(path)


def capture(full_source=None):
    """Record reference inputs under package-relative or logical keys."""
    engine = load_engine()
    out = dict(engine.fingerprint())
    files = [
        paths.FIXTURES / n
        for n in ('selftest_ref_meta.json', 'selftest_ref_simpson.npz', 'selftest_ref_simpson_meta.json')
    ] + [paths.ASSETS / 'states_v1.json']
    if legacy_available():
        files.extend((LEGACY, LEGACY_META))
    out.update({key(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    if full_source is not None and full_source.is_file():
        out['full_source'] = hashlib.sha256(full_source.read_bytes()).hexdigest()
    return out


def same(before, after):
    changed = sorted(p for p in before.keys() | after.keys() if p not in before or p not in after or before[p] != after[p])
    return dict(
        pass_=not changed,
        checked_files=len(before),
        changed=changed,
        scope='reference inputs and original solver and runner source files',
    )
