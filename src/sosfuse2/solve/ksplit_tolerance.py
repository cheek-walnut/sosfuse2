"""CPU-only KSPLIT comparison. Groups identify physical units, never staging pieces.

Call compare_tol with complete semantic groups, or compare_chunks with all pieces
of those groups. Missing/unwritten valid slots fail even when both sides agree.
"""

import os
import numpy as np


def tolerances(environ=None):
    env = os.environ if environ is None else environ
    values = []
    for name, default in (('RTOL', '1e-5'), ('ATOL', '1e-7')):
        key = 'SOSFUSE2_KSPLIT_VERIFY_' + name
        try:
            value = float(env.get(key, default))
        except (TypeError, ValueError):
            raise ValueError(f'E_KSPLIT_VERIFY: {key} must be finite and nonnegative') from None
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'E_KSPLIT_VERIFY: {key} must be finite and nonnegative')
        values.append(value)
    return tuple(values)


# Arithmetic scratch is independent of the device transfer/window size. In
# particular, 16M device words must never become 16M float64 comparison records.
COMPARE_STAGE = 1 << 16


def _slice(value, shape, start, stop):
    """Slice a broadcast view without flattening/copying the complete domain.

    Lazy metadata providers receive flat [start, stop) offsets. They must be
    repeatable, since scales and comparisons traverse the same domain twice.
    """
    if callable(value):
        return np.asarray(value(start, stop))
    return np.broadcast_to(value, shape).flat[start:stop]


def _pieces(chunks, stage_size, *, reference_only=False):
    offset = 0
    for a, b, valid, groups in chunks:
        for start in range(0, b.size, stage_size):
            stop = min(start + stage_size, b.size)
            mask = _slice(valid, b.shape, start, stop).astype(bool, copy=False)
            indices = np.flatnonzero(mask)
            keys = _slice(groups, b.shape, start, stop)[mask]
            # Index first, then promote only this bounded piece.
            yield (
                None if reference_only else a.flat[start:stop][mask].astype('f8', copy=False),
                b.flat[start:stop][mask].astype('f8', copy=False),
                keys,
                indices + offset + start,
            )
        offset += b.size


def _compare(chunks, r, alpha, details, stage_size):
    if not all(np.isfinite(x) and x >= 0 for x in (r, alpha)):
        raise ValueError('E_KSPLIT_VERIFY: tolerances must be finite and nonnegative')
    if stage_size <= 0:
        raise ValueError('E_KSPLIT_VERIFY: stage_size must be positive')
    stage_size = min(stage_size, COMPARE_STAGE)
    out = dict(
        pass_=False,
        E_inf=float('inf'),
        violations=1,
        max_abs=float('inf'),
        max_rel=float('inf'),
        groups={},
        first=None,
        rtol=float(r),
        atol=float(alpha),
    )
    chunks = [(np.asarray(a), np.asarray(b), mask, keys) for a, b, mask, keys in chunks]
    for a, b, _, _ in chunks:
        if a.shape != b.shape:
            return dict(out, reason=f'shape {a.shape} vs {b.shape}')

    # One global scale pass; no piece is a semantic group. Only O(groups) state
    # survives a piece, including groups whose references are all nonfinite.
    labels = None
    scale = np.empty(0)
    for _, b, keys, _ in _pieces(chunks, stage_size, reference_only=True):
        local, inv = np.unique(keys, return_inverse=True)
        if not len(local):
            continue
        local_scale = np.zeros(len(local))
        np.maximum.at(local_scale, inv, np.where(np.isfinite(b), np.abs(b), 0.0))
        merged = local if labels is None else np.union1d(labels, local)
        merged_scale = np.zeros(len(merged))
        if labels is not None:
            merged_scale[np.searchsorted(merged, labels)] = scale
        loc = np.searchsorted(merged, local)
        merged_scale[loc] = np.maximum(merged_scale[loc], local_scale)
        labels, scale = merged, merged_scale
    if labels is None:
        labels = np.array([], dtype=int)
    dl = np.zeros(len(labels))
    bl = np.zeros(len(labels))
    peak = first = failure = None
    emax = max_abs = max_rel = 0.0
    violations = 0
    for a, b, keys, indices in _pieces(chunks, stage_size):
        if not len(b):
            continue
        inv = np.searchsorted(labels, keys)
        sg = scale[inv]
        finite = np.isfinite(a) & np.isfinite(b)
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            delta = np.where(finite, np.abs(a - b), np.inf)
            abs_b = np.abs(b)
            bound = r * abs_b + alpha * sg
            error = np.divide(delta, bound, out=np.where(delta == 0, 0.0, np.inf), where=bound > 0)
            error[~finite] = np.inf
            rel = np.divide(delta, abs_b, out=np.where(delta == 0, 0.0, np.inf), where=abs_b > 0)
            rel[~finite] = np.inf
            bad = (~finite) | (delta > bound)
            norm_delta = np.divide(delta, sg, out=np.where(delta == 0, 0.0, np.inf), where=sg > 0)
            norm_ref = np.divide(b, sg, out=np.zeros_like(b), where=sg > 0)
            # add.at preserves input accumulation order across staging boundaries.
            np.add.at(dl, inv, norm_delta**2)
            np.add.at(bl, inv, norm_ref**2)
        local_peak = int(error.argmax())
        if peak is None or error[local_peak] > emax or (np.isnan(error[local_peak]) and not np.isnan(emax)):
            peak = int(indices[local_peak])
            emax = float(error[local_peak])
        max_abs = max(max_abs, float(delta.max()))
        max_rel = max(max_rel, float(rel.max()))
        violations += int(bad.sum())
        if first is None and bad.any():
            i = int(np.flatnonzero(bad)[0])
            first = int(indices[i])
            failure = (int(inv[i]), float(a[i]), float(b[i]))
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        l2 = np.sqrt(np.divide(dl, bl, out=np.where(dl == 0, 0.0, np.inf), where=bl > 0))
    rows = {str(k): dict(scale=float(scale[i]), rel_l2=float(l2[i])) for i, k in enumerate(labels)} if details else {}
    failure_group = None
    if failure is not None:
        g, a, b = failure
        failure_group = dict(key=str(labels[g]), scale=float(scale[g]), rel_l2=float(l2[g]), actual=a, reference=b)
    return dict(
        out,
        pass_=not violations,
        E_inf=emax,
        violations=violations,
        max_abs=max_abs,
        max_rel=max_rel,
        groups=rows,
        first=None if first is None else [first],
        peak=None if peak is None else [peak],
        failure_group=failure_group,
        scale_min=float(scale.min()) if len(scale) else 0.0,
        scale_max=float(scale.max()) if len(scale) else 0.0,
    )


def compare_tol(actual, reference, valid, groups, r=1e-5, alpha=1e-7, *, details=True, stage_size=COMPARE_STAGE):
    """Elementwise |a-b| <= r*|b| + alpha*max_group|b|; L2 is diagnostic.

    valid/groups broadcast to the exact array shape, or lazily provide flat
    slices. Groups are complete semantic domains, independent of stage_size.
    Nonfinite valid slots fail; zero error/zero bound gives E=0, otherwise inf.
    """
    out = _compare([(actual, reference, valid, groups)], r, alpha, details, stage_size)
    for key in ('first', 'peak'):
        if out.get(key) is not None:
            out[key] = [int(i) for i in np.unravel_index(out[key][0], np.shape(reference))]
    return out


def compare_chunks(chunks, r=1e-5, alpha=1e-7, *, details=True, stage_size=COMPARE_STAGE):
    """Compare retained pieces using global scales, without concatenating them.

    First/peak are flat indices in concatenated input order. Input arrays remain
    views; only fixed-size arithmetic scratch and per-group statistics are added.
    """
    return _compare(chunks, r, alpha, details, stage_size)


def log_summary(result):
    """Bounded diagnostics: never serialize the per-group table on a hot path."""
    out = {k: v for k, v in result.items() if k not in ('groups', 'failure_group', 'first')}
    if not result['pass_']:
        out.update(first=result['first'], failure_group=result.get('failure_group'))
    return out
