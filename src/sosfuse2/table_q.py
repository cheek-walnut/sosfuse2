"""Lossy radiance table store for the R3 bake (sky-sos docs/SPEC_R3_1000state_bake.md section 2 and section 7 B2).

table_q.npz holds ln(radiance) quantized to uint16 over [log_lo, log_hi] of each slice along the first axis (solar
elevation: the whole table spans W 25.7-26.0 on the 2026-10-26 L4 R3-corner states, one elevation at most 6.4, so a
per-state range would sit at the 26.2 limit and quantize 4x coarser), differenced (mod 2^16) along
the delta axis (the second to last: relative azimuth for (solar, view, azimuth, wavelength)) and stored as two byte
planes (high bytes, then low bytes), plus the coordinate arrays of table.npz verbatim. Both steps are lossless; the
pack's zstd -10 gives about 3.8 MB per V14 state (36 local tables: max rel 6.0e-5, slice W at most 7.9; 2026-10-26).
The rounding error in ln is at most W/(2*65535) for W = log_hi - log_lo of the slice, so the relative error is at
most exp(W/131070) - 1; W <= 26.2 keeps it under 2e-4 (MAX_REL). Reader: I = exp(lo + q/65535*(hi - lo)) with lo, hi
broadcast along the first axis. Every failure is explicit (E_TABLE_Q_*): zero, negative or non-finite radiance, a
constant slice, W > MAX_WIDTH, a header that disagrees with the arrays, and a
round trip through the consumer reader that misses shape, coordinates or MAX_REL.
"""
import io, json, math
from pathlib import Path
import numpy as np

FORMAT = 'table_q/2'                  # /2: log range per first-axis slice
LEVELS = 65535
MAX_REL = 2e-4
MAX_WIDTH = 26.2                      # exp(26.2/131070) - 1 = 1.9991e-4
COORDS = ('wavelength_nm', 'solar_elevation_deg', 'view_elevation_deg', 'relative_azimuth_deg', 'units')


class TableQError(ValueError):
    pass


def delta_axis(ndim):
    return max(ndim-2, 0)


def slices(lo, ndim):
    return np.asarray(lo, 'f8').reshape((-1,)+(1,)*(ndim-1))


def encode(radiance):
    """(log_lo[n0], log_hi[n0], planes uint8[2, n]) for a strictly positive finite radiance array of ndim >= 2."""
    r = np.asarray(radiance)
    if r.size == 0 or r.ndim < 2: raise TableQError('E_TABLE_Q_EMPTY')
    if not np.isfinite(r).all(): raise TableQError('E_TABLE_Q_NONFINITE')
    if (r <= 0).any(): raise TableQError(f'E_TABLE_Q_NONPOSITIVE: {int((r <= 0).sum())} elements')
    ln = np.log(r.astype('f8'))
    axes = tuple(range(1, r.ndim)); lo, hi = ln.min(axis=axes), ln.max(axis=axes)
    if (hi == lo).any(): raise TableQError(f'E_TABLE_Q_CONSTANT: slice {int(np.argmax(hi == lo))}')
    if (hi-lo).max() > MAX_WIDTH: raise TableQError(f'E_TABLE_Q_RANGE: W {(hi-lo).max():.3f} > {MAX_WIDTH} at slice {int(np.argmax(hi-lo))}')
    L, H = slices(lo, r.ndim), slices(hi, r.ndim)
    q = np.rint((ln-L)/(H-L)*LEVELS).astype('<u2')
    d = np.diff(q, axis=delta_axis(q.ndim), prepend=np.uint16(0)).astype('<u2').ravel()   # wraps mod 2^16
    return lo, hi, np.ascontiguousarray(d.view('u1').reshape(-1, 2)[:, ::-1].T)


def decode(lo, hi, planes, shape):
    planes = np.asarray(planes, 'u1')
    d = ((planes[0].astype('<u2') << 8) | planes[1]).reshape(shape)
    q = np.cumsum(d, axis=delta_axis(len(shape)), dtype='<u2')
    L, H = slices(lo, len(shape)), slices(hi, len(shape))
    return np.exp(L+q.astype('f8')/LEVELS*(H-L))


def save(table_npz, out_path):
    """Write table_q.npz from a table.npz; returns its header dict (log range, width)."""
    with np.load(table_npz, allow_pickle=False) as z:
        r = z['radiance']; coords = {k: z[k] for k in COORDS}
        extra = sorted(set(z.files)-set(COORDS)-{'radiance'})
    if extra: raise TableQError(f'E_TABLE_Q_MEMBERS: unexpected {extra}')
    lo, hi, planes = encode(r)
    header = dict(format=FORMAT, log_lo=lo.tolist(), log_hi=hi.tolist(), width=float((hi-lo).max()), levels=LEVELS,
                  shape=list(r.shape), dtype=r.dtype.str, delta_axis=delta_axis(r.ndim))
    buf = io.BytesIO()
    np.savez(buf, header=np.array(json.dumps(header)), log_lo=lo.astype('f8'), log_hi=hi.astype('f8'),
             shape=np.array(r.shape, 'i8'), q_planes=planes, **coords)
    out_path = Path(out_path); tmp = out_path.with_suffix('.npz.tmp')
    tmp.write_bytes(buf.getvalue()); tmp.replace(out_path)
    return header


def load(path, dtype='f4'):
    """Consumer reader: {'radiance': array, coordinate arrays..., 'header': dict}."""
    with np.load(path, allow_pickle=False) as z:
        header = json.loads(str(z['header']))
        if header.get('format') != FORMAT: raise TableQError(f"E_TABLE_Q_FORMAT: {header.get('format')}")
        lo, hi, shape, planes = z['log_lo'], z['log_hi'], tuple(int(n) for n in z['shape']), z['q_planes']
        # header and arrays must agree, so a damaged scalar or a swapped header cannot decode silently
        if len(shape) < 2 or lo.shape != (shape[0],) or hi.shape != (shape[0],) or not (np.isfinite(lo).all() and np.isfinite(hi).all()
                and (lo < hi).all() and (hi-lo).max() <= MAX_WIDTH): raise TableQError('E_TABLE_Q_HEADER: log range')
        if (header.get('log_lo'), header.get('log_hi'), tuple(header.get('shape', ())), header.get('levels'), header.get('delta_axis')) != \
                (lo.tolist(), hi.tolist(), shape, LEVELS, delta_axis(len(shape))): raise TableQError('E_TABLE_Q_HEADER: header vs arrays')
        if planes.dtype != np.uint8 or planes.shape != (2, math.prod(shape)): raise TableQError('E_TABLE_Q_HEADER: planes')
        out = {k: z[k] for k in COORDS}
        # (solar, view, azimuth, wavelength) radiance against the coordinate vectors
        if len(shape) == 4 and tuple(len(out[k]) for k in COORDS[:4]) != (shape[3], shape[0], shape[1], shape[2]):
            raise TableQError('E_TABLE_Q_HEADER: coordinate lengths')
        out['radiance'] = decode(lo, hi, planes, shape).astype(dtype)
    if not np.isfinite(out['radiance']).all() or (out['radiance'] <= 0).any(): raise TableQError('E_TABLE_Q_DECODE: nonfinite or nonpositive')
    out['header'] = header
    return out


def round_trip(table_npz, q_path):
    """Read the stored table_q back with the consumer reader and compare it with the float32 original (B2).
    Returns a row with max_rel; raises TableQError on any mismatch."""
    back = load(q_path)
    with np.load(table_npz, allow_pickle=False) as z:
        r = z['radiance']
        for k in COORDS:
            if z[k].shape != back[k].shape or not np.array_equal(z[k], back[k]): raise TableQError(f'E_TABLE_Q_COORDS: {k}')
    if back['radiance'].shape != r.shape: raise TableQError('E_TABLE_Q_SHAPE')
    ref = r.astype('f8')
    rel = float((np.abs(back['radiance'].astype('f8')-ref)/ref).max())
    if not rel <= MAX_REL: raise TableQError(f'E_TABLE_Q_ROUNDTRIP: max rel {rel:.3e} > {MAX_REL}')
    return dict(max_rel=rel, width=back['header']['width'], log_lo=min(back['header']['log_lo']), log_hi=max(back['header']['log_hi']))
