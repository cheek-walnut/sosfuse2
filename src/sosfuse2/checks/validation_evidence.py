"""Narrow read-only evidence for the exact reference inputs used by validation."""

import hashlib
from ..foundation import paths
from ..solve.source_solver import load_engine
from .validate import FULL_SOURCE, LEGACY


def key(path):
    """Provenance key: 'sosfuse2/<relative>' inside the package, 'sky-sos/<relative to the checkout>' for a sky-sos extra,
    never a host path."""
    root = paths.sky_sos_root()
    if root is not None:
        try:
            return 'sky-sos/' + path.resolve().relative_to(root).as_posix()
        except ValueError:
            pass
    return paths.rel(path)


def capture():
    """Package-relative keys; sky-sos-only extras (legacy fixture, converged full source) are included only when present
    and keyed 'sky-sos/<path in the checkout>'."""
    engine = load_engine()
    out = dict(engine.fingerprint())
    files = [
        paths.FIXTURES / n
        for n in ('selftest_ref_meta.json', 'selftest_ref_simpson.npz', 'selftest_ref_simpson_meta.json')
    ] + [paths.ASSETS / 'states_v1.json']
    files += [p for p in (LEGACY, FULL_SOURCE) if p is not None and p.is_file()]
    out.update({key(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    return out


def same(before, after):
    changed = [p for p in before if before[p] != after.get(p)]
    return dict(
        pass_=not changed,
        checked_files=len(before),
        changed=changed,
        scope='reference inputs and original solver/rig source files; not a whole-filesystem assertion',
    )
