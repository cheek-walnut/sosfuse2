"""Exact accepted angular/path quadrature, compact ray endpoints, no samples.

Small transient packs serve both backends and are independent of the choice
of production geometry residency. All mutable paths are under the caller's cache folder (the work root by default).
"""

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
from .source_solver import Kernel, load_engine
from ..foundation import content_identity as source_identity, paths
from ..backends import numpy_reference
from ..atmosphere import tropo_profile
from ..foundation.timers import stage


def normal_state():
    return next(s for s in json.loads((paths.ASSETS / 'states_v1.json').read_text())['states'] if s['id'] == 'normal')


def pack(p, d, n, s, scope, weights=None, ground=True):
    p = np.broadcast_to(p, d.shape).copy()
    d = np.ascontiguousarray(d, 'f8')
    old = scope.b.sh.old
    edges = old.probe.boundaries(p, d)
    end = edges[:, -1]
    dot = np.sum(p * d, -1)
    pp = np.sum(p * p, -1)
    more = []
    for height in (10.0, 25.0, 40.0):
        disc = dot * dot - pp + (6371.0 + height) ** 2
        for sign in (-1, 1):
            rr = -dot + sign * np.sqrt(np.maximum(disc, 0))
            more.append(np.where((disc > 0) & (rr > 0) & (rr < end), rr, end))
    edges = np.sort(np.column_stack([edges, *more]), axis=-1)
    qa = p[:, None, :] + edges[:, :-1, None] * d[:, None, :]
    qb = p[:, None, :] + edges[:, 1:, None] * d[:, None, :]
    lower = np.sum(qa * qa, axis=-1) <= np.sum(qb * qb, axis=-1)
    B = scope.basis_at(p)
    qe = p + end[:, None] * d
    re = np.linalg.norm(qe, axis=-1)
    ch = np.arccos(np.clip(qe[:, 2] / re, -1, 1))
    j = np.clip(np.searchsorted(s.geo.chi, ch, side='right') - 1, 0, len(s.geo.chi) - 2)
    f = np.clip((ch - s.geo.chi[j]) / (s.geo.chi[j + 1] - s.geo.chi[j]), 0, 1)
    joint = {}
    if ground:
        from ..rays.ray_columns import integrate, replace

        optics = getattr(s, 'density_optics', None)
        profile = getattr(s, 'strato', {}).get('profile')
        col, corrected, strato = integrate(
            p, d, old, 24, legacy=True, optics=optics, profile=profile, edges=edges, main=True
        )
        # Numerically equivalent to old.total_columns(p,d,24) at rtol 1e-6, plus independent sidecars.
        if optics is not None:
            joint['density'] = replace(col, corrected, optics)[:, [0, 2, 3, 4]].astype('f4')
        if strato is not None:
            joint['strato'] = strato.astype('f4')
    else:
        col = np.zeros((len(d), 5), 'f8')
    return dict(
        p=p,
        d=d,
        edges=edges,
        lower=lower,
        target=B,
        dloc=np.einsum('nij,ni->nj', B, d),
        weight=np.ones(len(d), 'f8') if weights is None else weights,
        end_index=j.astype('i4'),
        end_fraction=f,
        end_columns=col,
        hits_ground=re < 6371.0 + 1e-5,
        nq=n,
        **({'_joint': joint} if joint else {}),
    )


def create(output, grid='selftest', wavelengths=(550.0,), backend='numpy', block_size=128, state=None):
    with stage('solver_create'):
        return _create(output, grid, wavelengths, backend, block_size, state)


def _create(output, grid, wavelengths, backend, block_size, state):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    engine = load_engine()
    if state is None:
        import copy

        state = copy.deepcopy(normal_state())
    # Historical unsupported states preserve requested tau but compute no layer.

    from ..atmosphere.density import profile_step_km, kernel_kwargs

    kw = kernel_kwargs(profile_step_km(state))  # Kernel compiles the state's PROFILE node count (none at 0.5).
    initial, scope, settings = engine.create(
        state,
        output,
        grid,
        4,
        float(wavelengths[0]),
        kernel_factory=lambda threads, H, g: Kernel(4, H, g, backend, block_size, **kw),
        aerosol=tropo_profile.engine_hooks(state),
    )
    s = initial
    if len(wavelengths) > 1:
        s = scope.Solver(initial.geo, initial.kernel, wavelengths, 1, 0.03)
    s.active[1::2] = 0
    s.fuse_state = state
    tropo_profile.attach(s)
    from ..atmosphere.density import attach as attach_density
    from ..atmosphere.stratosphere import tau_of

    # attach() immediately replaces all first-ground arrays when the layer is on.
    attach_density(s, scope, defer_ground=tau_of(state) > 0)
    from ..atmosphere.stratosphere import attach

    attach(s, scope)
    s.geo.pack = lambda p, d, n: pack(p, d, n, s, scope, ground=False)
    return s, scope, settings


def subset(g, index):
    return {k: v[index] if isinstance(v, np.ndarray) else v for k, v in g.items()}


def _csr_loop(edges, lower):
    """Per-ray reference implementation of endpoint packing."""
    edge_counts = []
    edge_values = []
    flags_out = []
    for ray in range(len(edges)):
        keep = np.flatnonzero(np.diff(edges[ray]) > 0)
        values = np.r_[edges[ray, keep], edges[ray, keep[-1] + 1]] if len(keep) else np.array([0.0])
        edge_counts.append(len(values))
        edge_values.append(values)
        flags_out.append(sum(int(lower[ray, k]) << i for i, k in enumerate(keep)))
    return edge_counts, edge_values, flags_out


def _csr_vector(edges, lower):
    """Array form of _csr_loop. Returns (counts i8[n], values f8[sum], flags u2[n]).

    A ray keeps every strictly positive interval (diff>0, even 1e-12: the kernel
    skips <=1e-8 itself).  Values are the left endpoints of kept intervals plus
    the right endpoint of the last kept one; a ray with none stores one 0.0.
    Flag bit i is the lower flag of the i-th kept interval.
    """
    edges = np.asarray(edges)
    n, m1 = edges.shape
    m = m1 - 1
    if m > 64:
        raise OverflowError('E_LOWER_BITS: more than 64 intervals per ray')
    mask = np.diff(edges, axis=1) > 0
    has = mask.any(1)
    last = m - 1 - np.argmax(mask[:, ::-1], axis=1) if m > 0 else np.zeros(n, 'i8')
    select = np.zeros((n, m1), bool)
    select[:, :m] = mask
    rows = np.flatnonzero(has)
    select[rows, last[rows] + 1] = True
    empty = np.flatnonzero(~has)
    select[empty, 0] = True
    source = edges.astype('f8', copy=True)
    source[empty, 0] = 0.0
    counts = select.sum(1).astype('i8')
    rank = np.cumsum(mask, axis=1) - 1
    bits = np.where(mask, np.asarray(lower[:, :m], 'u8') << np.maximum(rank, 0).astype('u8'), np.uint64(0))
    flags = bits.sum(1, dtype='u8')
    if (flags > 0xFFFF).any():
        # The loop builds Python ints and np.array(...,'u2') raises OverflowError.
        raise OverflowError('E_LOWER_BITS: kept-interval flags exceed uint16')
    flags = flags.astype('u2')
    return counts, source[select], flags


class Bundle:
    """CSR endpoints; sorted destination cells are the block partition.

    Compression removes duplicate zero-width intervals, preserving the retained
    interval's endpoint orientation flag. No path samples are ever generated.
    _csr_vector packs consecutive groups with equal edge width in one array call.
    """

    # Sidecars are method-scoped even when an existing host Bundle is reused.
    # Properties also cover __new__ bundles and device promotion's setters.
    def _columns(self, kind):
        from ..rays.column_table import method

        return self.__dict__.setdefault('_' + kind + '_methods', {}).setdefault(method(), {})

    @property
    def density_columns(self):
        return self._columns('density')

    @density_columns.setter
    def density_columns(self, value):
        from ..rays.column_table import method

        self._density_methods = dict(self.__dict__.get('_density_methods', {}), **{method(): value})

    @property
    def strato_columns(self):
        return self._columns('strato')

    @strato_columns.setter
    def strato_columns(self, value):
        from ..rays.column_table import method

        self._strato_methods = dict(self.__dict__.get('_strato_methods', {}), **{method(): value})

    def __init__(self, groups, xp=np, loop=False):
        with stage('bundle_pack'):
            self.cells = len(groups)
            self.nq = groups[0]['nq']
            self.rays = sum(len(g['d']) for g in groups)
            if any(g['nq'] != self.nq for g in groups):
                raise ValueError('E_QUADRATURE: mixed path orders')
            starts = []
            targets = []
            counts = []
            edge_counts = []
            edge_values = []
            lower = []
            for g in groups:
                if not np.all(g['p'] == g['p'][0]):
                    raise ValueError('E_START: one destination start per group')
                starts.append(g['p'][0])
                targets.append(g['target'][0])
                counts.append(len(g['d']))
            if loop:
                for g in groups:
                    ec, ev, fl = _csr_loop(g['edges'], g['lower'])
                    edge_counts += ec
                    edge_values += ev
                    lower += fl
            else:
                run = []

                def flush():
                    if not run:
                        return
                    e = np.concatenate([g['edges'] for g in run]) if len(run) > 1 else run[0]['edges']
                    lo = np.concatenate([g['lower'] for g in run]) if len(run) > 1 else run[0]['lower']
                    ec, ev, fl = _csr_vector(e, lo)
                    edge_counts.append(ec)
                    edge_values.append(ev)
                    lower.append(fl)
                    run.clear()

                for g in groups:
                    if run and run[0]['edges'].shape[1] != g['edges'].shape[1]:
                        flush()
                    run.append(g)
                flush()
                edge_counts = np.concatenate(edge_counts) if edge_counts else np.zeros(0, 'i8')
                lower = np.concatenate(lower) if lower else np.zeros(0, 'u2')
            edge_total = int(np.sum(edge_counts, dtype='i8'))
        self._finish(groups, xp, starts, targets, counts, edge_counts, edge_values, lower, edge_total)

    def _finish(self, groups, xp, starts, targets, counts, edge_counts, edge_values, lower, edge_total):
        if edge_total >= 2**32:
            raise RuntimeError('E_EDGE_OFFSET: compact edge count exceeds uint32')
        a = dict(
            cell_off=np.r_[0, np.cumsum(counts)].astype('i8'),
            starts=np.array(starts, 'f8'),
            targets=np.array(targets, 'f8'),
            dirs=np.concatenate([g['d'] for g in groups]),
            weights=np.concatenate([g['weight'] for g in groups]).astype('f4'),
            edge_off=np.r_[0, np.cumsum(edge_counts)].astype('u4'),
            edges=np.concatenate(edge_values).astype('f8'),
            lower=np.array(lower, 'u2'),
            end_j=np.concatenate([g['end_index'] for g in groups]).astype('i4'),
            end_f=np.concatenate([g['end_fraction'] for g in groups]).astype('f4'),
            end_col=np.concatenate([g['end_columns'] for g in groups]).astype('f4'),
            hits=np.concatenate([g['hits_ground'] for g in groups]).astype('u1'),
        )
        with stage('bundle_upload', device=True):
            self.arrays = {k: xp.asarray(np.ascontiguousarray(v)) for k, v in a.items()}
        self.bytes = sum(v.nbytes for v in self.arrays.values())


class TransientGeometry:
    """Small reference/selftest geometry; never enables a full production cache."""

    def __init__(self, s, scope):
        self.s = s
        self.scope = scope
        self.groups = {}
        self.bundle = None
        self.ground_bundle = None
        self.stats = dict(cells=0, rays=0, samples=0, compact_bytes=0, scope='transient reduced validation geometry')

    def cell(self, cell):
        if cell not in self.groups:
            with stage('geometry_cell_build'):
                s = self.s
                b = self.scope.b
                ih, ic = divmod(cell, len(s.geo.chi))
                r = 6371.0 + s.geo.h[ih]
                ch = s.geo.chi[ic]
                loc, w = b.legacy.incoming(r, ch, s.geo.cfg['nmu'], s.geo.cfg['nphi'], 4.0, True)
                p, up, ex = b.sh.old.probe.frame(np.array(r), ch)
                ey = np.cross(up, ex)
                d = loc[:, 0, None] * ex + loc[:, 1, None] * ey + loc[:, 2, None] * up
                g = pack(p, d, s.geo.cfg['path_n'], s, self.scope, w)
                self.groups[cell] = g
                self.stats['cells'] += 1
                self.stats['rays'] += len(d)
                self.stats['samples'] += int((np.diff(g['edges'], axis=1) > 1e-8).sum()) * g['nq']
        return self.groups[cell]

    def evaluate(self, s, field, first=False, phase=0, ground_only=False, boundary=None, timed=False):
        xp = s.kernel.xp
        count = len(s.geo.chi) if ground_only else s.geo.ns
        if xp is not np:
            attr = 'ground_bundle' if ground_only else 'bundle'
            bundle = getattr(self, attr)
            if bundle is None:
                bundle = Bundle([self.cell(i) for i in range(count)], xp)
                setattr(self, attr, bundle)
                self.stats['compact_bytes'] = max(self.stats['compact_bytes'], bundle.bytes)
            moment, flux, _ = s.kernel.launch(bundle, s, field, first, phase, boundary, timed)
            return moment, flux
        moments = np.zeros((count, 4 + s.kernel.K, s.W), 'f8')
        flux = np.empty((count, s.W), 'f8')
        for i in range(count):
            g = self.cell(i)
            c = s.kernel.setup(s, g['nq'])
            with stage('launch_first' if first else 'launch_ground' if phase == 1 else 'launch_warm'):
                value, Y = numpy_reference.transport(g, c, field, first, boundary, s.kernel.metrics, return_basis=True)
                moments[i], flux[i] = numpy_reference.project(g, value, c, Y, ground_only=phase == 1)
        return moments, flux


FIELDS = ('starts', 'targets', 'dirs', 'weights', 'edge_off', 'edges', 'lower', 'end_j', 'end_f', 'end_col', 'hits')
DTYPES = dict(
    cell_off='i8',
    starts='f8',
    targets='f8',
    dirs='f8',
    weights='f4',
    edge_off='u4',
    edges='f8',
    lower='u2',
    end_j='i4',
    end_f='f4',
    end_col='f4',
    hits='u1',
)
TRAIL = dict(starts=(3,), targets=(3, 3), dirs=(3,), end_col=(5,))


class PackedCache:
    """Whole-grid endpoint cache with one contiguous raw file per field.

    Replaces 1500 x (13 .npy + complete.json).  Cells are appended in order;
    journal.jsonl commits each cell after its bytes are flushed, so an
    interrupted build resumes by truncating every field to the last committed
    size.  edge_off is stored already global (+base), exactly the device
    layout of the original prepare(); cell_off is written at completion.
    Loading is one memmap per field (no per-cell file access).
    """

    def __init__(self, folder):
        self.folder = Path(folder)

    def complete(self):
        return (self.folder / 'complete.json').exists()

    def build(self, ncell, cell_arrays, progress=None):
        f = self.folder
        f.mkdir(parents=True, exist_ok=True)
        journal = f / 'journal.jsonl'
        done = []
        if journal.exists():
            for line in journal.read_text().splitlines():
                try:
                    done.append(json.loads(line))
                except json.JSONDecodeError:
                    break
        sizes = done[-1]['sizes'] if done else {k: 0 for k in FIELDS}
        for k in FIELDS:
            (f / (k + '.bin')).touch()
        if any((f / (k + '.bin')).stat().st_size < sizes[k] for k in FIELDS):
            # A field shorter than its committed size: restart the build from zero.
            done = []
            sizes = {k: 0 for k in FIELDS}
        if not done:
            journal.write_text('')
        else:
            journal.write_text(''.join(json.dumps(r) + '\n' for r in done))
        handles = {}
        for k in FIELDS:
            os.truncate(f / (k + '.bin'), sizes[k])
            handles[k] = open(f / (k + '.bin'), 'ab')
        base = done[-1]['edges_total'] if done else 0
        try:
            with journal.open('a') as jl:
                for i in range(len(done), ncell):
                    host, meta = cell_arrays(i)
                    if base + meta['edges'] >= 2**32:
                        raise RuntimeError('E_EDGE_OFFSET: exceeds uint32')
                    rows = dict(
                        starts=host['starts'][:1],
                        targets=host['targets'][:1],
                        edge_off=(np.asarray(host['edge_off'][:-1]).astype('u8') + base).astype('u4'),
                    )
                    for k in FIELDS:
                        handles[k].write(np.ascontiguousarray(rows.get(k, host[k]), DTYPES[k]).tobytes())
                    for h in handles.values():
                        h.flush()
                    base += meta['edges']
                    sizes = {k: handles[k].tell() for k in FIELDS}
                    jl.write(
                        json.dumps(
                            dict(
                                cell=i,
                                rays=meta['rays'],
                                edges=meta['edges'],
                                samples=meta['samples'],
                                nq=meta['nq'],
                                edges_total=base,
                                sizes=sizes,
                            )
                        )
                        + '\n'
                    )
                    jl.flush()
                    if progress:
                        progress(i)
            handles['edge_off'].write(np.array([base], 'u4').tobytes())
        finally:
            for h in handles.values():
                h.close()
        records = [json.loads(line) for line in journal.read_text().splitlines()]
        counts = [r['rays'] for r in records]
        cell_off = np.r_[0, np.cumsum(counts)].astype('i8')
        (f / 'cell_off.bin').write_bytes(cell_off.tobytes())
        rays = int(cell_off[-1])
        shapes = dict(
            cell_off=[len(records) + 1],
            starts=[len(records), 3],
            targets=[len(records), 3, 3],
            edge_off=[rays + 1],
            edges=[base],
        )
        for k in ('dirs', 'weights', 'lower', 'end_j', 'end_f', 'end_col', 'hits'):
            shapes[k] = [rays, *TRAIL.get(k, ())]
        layout = dict(version=3, cells=len(records), rays=rays, edges=base, dtypes=DTYPES, shapes=shapes)
        for k, shape in shapes.items():
            if (f / (k + '.bin')).stat().st_size != int(np.prod(shape)) * np.dtype(DTYPES[k]).itemsize:
                raise RuntimeError(f'E_PACKED_SIZE: {k}')
        (f / 'layout.json').write_text(json.dumps(layout, indent=1) + '\n')
        (f / 'complete.json').write_text(json.dumps(dict(cells=len(records), rays=rays, edges=base)) + '\n')

    def open(self):
        """Host memmaps in the original Bundle key order, plus per-cell records."""
        layout = json.loads((self.folder / 'layout.json').read_text())
        from ..precompute.endpoint_contract import PACKED_LAYOUT_VERSION

        if layout.get('version') != PACKED_LAYOUT_VERSION or layout['shapes'].get('end_col') != [layout['rays'], 5]:
            raise ValueError('E_PACKED_LAYOUT: incompatible optical component layout')
        records = [json.loads(line) for line in (self.folder / 'journal.jsonl').read_text().splitlines()]
        arrays = {}
        for k in ('cell_off',) + FIELDS:
            shape = tuple(layout['shapes'][k])
            arrays[k] = (
                np.memmap(self.folder / (k + '.bin'), dtype=DTYPES[k], mode='r', shape=shape)
                if int(np.prod(shape))
                else np.zeros(shape, DTYPES[k])
            )
        return arrays, records


def identity(s, version):
    """Cache identity.  version 1 is used for read-only legacy lookup;
    version 2 is the sosfuse2 packed cache (hashes sosfuse2/geometry.py);
    version 3 = sosfuse2 two-layer COMMON layer: identical but without the aerosol
    scale height (only end_columns[:,1] depends on it)."""
    from ..precompute.endpoint_contract import UPSTREAM

    upstream = [paths.PKG / n for n in UPSTREAM]
    files = {paths.rel(p): source_identity.file_sha(p) for p in upstream}
    if version == 1:
        # The legacy geometry source is optional. If unavailable, use a sentinel
        # so no legacy folder matches and the cells are generated instead.
        root = paths.sky_sos_root()
        code_file = (root / 'scripts/sosfuse/geometry.py') if root is not None else None
        files['sosfuse/geometry.py'] = (
            source_identity.file_sha(code_file)
            if code_file is not None and code_file.is_file()
            else 'unavailable'
        )
    else:
        files[paths.rel(Path(__file__))] = source_identity.file_sha(Path(__file__))
    content = dict(
        h=s.geo.h.tolist(), chi=s.geo.chi.tolist(), cfg=s.geo.cfg, height=s.kernel.height, version=version, files=files
    )
    from ..rays.column_table import identity as column_identity

    content['column_method'] = column_identity()
    from ..rays.column_table import molecular

    content['minor_profiles'] = [molecular(None, k) for k in (2, 3)]
    if version == 3:
        del content['height']
        content['layer'] = 'common: all endpoint arrays; end_col[:,1] replaced per scale height'
    profile = tropo_profile.of(s)
    # Two-layer aerosol: v1/v2 caches hold this state's end_col[:,1], so the profile joins their identity (absent = unchanged).
    if profile is not None and version != 3:
        content['aerosol_profile'] = profile['canonical']
    return content, hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:20]


def aerosol_column(arrays, height, chunk=50_000, workers=None, profile=None):
    """end_columns[:,1] from cached endpoints, operation-identical to
    the total column calculation used by pack() (the integrate(...) call in this module).

    Same segment endpoints (the CSR keeps every diff>0 interval, and a dropped
    zero-width interval never has diff>1e-8), same per-segment expressions and
    the same (A,24,3) reduction shape, so every per-ray sum sees the same
    operands in the same order.  It skips probe.boundaries/sort (endpoints are
    cached), the unused prefix/cumsum/tau of geometry_segments and the Rayleigh/
    ozone columns.  Rays are grouped in chunks (all operations are per ray), run
    on `workers` threads (NumPy releases the GIL).  Returns f4.
    profile: None = exp(-max(h,0)/height) (historical expression); else the two-layer callable of
    tropo_profile.column_function, the same function used by the engine profiles.
    """
    from ..rays import column_table as ct

    if ct.method() != 'gl24':
        spec = (
            ct.descriptor('exp', height=float(height))
            if profile is None
            else profile.descriptor()
        )
        dirs = arrays['dirs']
        co = arrays['cell_off']
        starts = arrays['starts']
        out = np.empty(len(dirs), 'f4')
        for lo in range(0, len(dirs), 32768):
            hi = min(lo + 32768, len(dirs))
            owner = np.searchsorted(co, np.arange(lo, hi), side='right') - 1
            end = arrays['edges'][arrays['edge_off'][lo + 1 : hi + 1].astype('i8') - 1]
            out[lo:hi] = ct.column(spec, starts[owner], dirs[lo:hi], end)
        return out
    from concurrent.futures import ThreadPoolExecutor

    if workers is None:
        try:
            workers = int(os.environ.get('OMP_NUM_THREADS', '4'))
        except ValueError:
            workers = 4  # an OMP_NUM_THREADS such as '4,2' or '' is not a worker count
    R = 6371.0
    H = float(height)
    cell_off = np.asarray(arrays['cell_off'], 'i8')
    starts = np.asarray(arrays['starts'])
    dirs = np.asarray(arrays['dirs'])
    edge_off = np.asarray(arrays['edge_off']).astype('i8')
    edges = np.asarray(arrays['edges'])
    x, w = np.polynomial.legendre.leggauss(24)
    u, wu = (x + 1) / 2, w / 2  # sos_probe.gl(24)
    rays = len(dirs)
    out = np.empty(rays, 'f4')
    owner = np.repeat(np.arange(len(starts)), np.diff(cell_off))

    def run(lo):
        hi = min(rays, lo + chunk)
        counts = edge_off[lo + 1 : hi + 1] - edge_off[lo:hi]
        m = int(counts.max())
        E = edges[edge_off[lo:hi, None] + np.minimum(np.arange(m)[None, :], counts[:, None] - 1)]
        p = starts[owner[lo:hi]]
        d = np.ascontiguousarray(dirs[lo:hi])
        acc = np.zeros(hi - lo)
        for k in range(m - 1):
            active = np.flatnonzero(E[:, k + 1] - E[:, k] > 1e-8)
            if not len(active):
                continue
            a, b = E[active, k, None], E[active, k + 1, None]
            pa, da = p[active], d[active]
            ra = np.sum((pa + a * da) ** 2, -1)
            rb = np.sum((pa + b * da) ** 2, -1)
            lower = (ra <= rb)[:, None]
            t = np.where(lower, a + (b - a) * u * u, b - (b - a) * (1 - u) ** 2)
            ds = np.where(lower, 2 * (b - a) * u * wu, 2 * (b - a) * (1 - u) * wu)
            q = pa[:, None, :] + t[..., None] * da[:, None, :]
            r = np.sqrt(np.sum(q * q, -1))
            # profiles(h)*ds[...,None] restricted to column 1; columns 0/2 are 0.
            z = np.zeros(r.shape + (3,))
            z[..., 1] = (np.exp(-np.maximum(r - R, 0) / H) if profile is None else profile(r - R)) * ds
            acc[active] += np.sum(z, axis=1)[:, 1]
        out[lo:hi] = acc.astype('f4')

    with ThreadPoolExecutor(max(1, int(workers))) as pool:
        list(pool.map(run, range(0, rays, chunk)))
    return out


def aerosol_column_pack_route(arrays, old):
    """Timing/cross-check only: the pack() route, old.total_columns per cell."""
    cell_off = np.asarray(arrays['cell_off'], 'i8')
    starts = np.asarray(arrays['starts'])
    dirs = np.asarray(arrays['dirs'])
    out = np.empty(len(dirs), 'f4')
    for i in range(len(starts)):
        lo, hi = cell_off[i : i + 2]
        d = np.ascontiguousarray(dirs[lo:hi], 'f8')
        p = np.broadcast_to(starts[i], d.shape).copy()
        out[lo:hi] = old.total_columns(p, d, 24)[:, 1].astype('f4')
    return out


class ResidentGeometry(TransientGeometry):
    """Endpoint-resident option (~0.87 GB), explicitly selected.

    The device arrays come from one PackedCache (few contiguous files). If
    a compatible v1 per-cell cache (endpoints_<v1 signature>/cell_*)
    exists it is read (never written) and converted; otherwise cells are
    generated.  Device input remains resident for every wavelength/iteration.
    """

    _host_test = False  # Numerical-equivalence tests only; production requires CUDA.
    _version = 2

    def __init__(self, s, scope, cache):
        super().__init__(s, scope)
        if s.kernel.xp is np and not self._host_test:
            raise ValueError('E_RESIDENT_BACKEND: production residency requires CUDA')
        content, signature = identity(s, self._version)
        _, legacy = identity(s, 1)
        self.root = Path(cache)
        self.cache = self.root / ('endpoints_' + signature)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.legacy = self.find_legacy(legacy)
        (self.cache / 'identity.json').write_text(json.dumps(content, indent=2) + '\n')
        self.stats.update(
            scope='endpoint-resident option, no samples',
            signature=signature,
            legacy_signature=legacy,
            build_seconds=0.0,
            upload_seconds=0.0,
            load_seconds=0.0,
            legacy_cells_read=0,
        )
        self._host_offsets = None

    def find_legacy(self, legacy):
        return self.root / ('endpoints_' + legacy)

    def cached_cell(self, i):
        """Original v1 cell: read from the legacy folder if complete, else generate (not stored)."""
        folder = self.legacy / f'cell_{i:05d}'
        marker = folder / 'complete.json'
        if marker.exists():
            with stage('geometry_legacy_read'):
                meta = json.loads(marker.read_text())
                arrays = {
                    name: np.load(folder / (name + '.npy'), mmap_mode='r', allow_pickle=False) for name in meta['keys']
                }
            self.stats['legacy_cells_read'] += 1
            return arrays, meta
        g = self.cell(i)
        bundle = Bundle([g], np)
        self.__dict__.setdefault('_joint_cells', {})[i] = g.get('_joint', {})
        meta = dict(
            keys=list(bundle.arrays),
            rays=bundle.rays,
            edges=len(bundle.arrays['edges']),
            bytes=bundle.bytes,
            samples=int((np.diff(g['edges'], axis=1) > 1e-8).sum()) * g['nq'],
            nq=bundle.nq,
        )
        self.groups.clear()
        return bundle.arrays, meta

    def packed(self):
        return PackedCache(self.cache / 'packed')

    def build_packed(self):
        """CPU-only: build (or resume) the packed cache; no device access."""
        packed = self.packed()
        if packed.complete():
            return False
        if shutil.disk_usage(self.cache).free < 2 * 1024**3:
            raise RuntimeError('E_DISK_LOW: need ~0.9 GB packed cache plus 1 GiB reserve')
        started = time.perf_counter()
        last = [started]

        def progress(i):
            if time.perf_counter() - last[0] >= 30:
                print(
                    f'SOSFUSE GEOMETRY cell={i}/{self.s.geo.ns} seconds={time.perf_counter() - started:.1f}', flush=True
                )
                last[0] = time.perf_counter()

        with stage('geometry_cache_build'):
            packed.build(self.s.geo.ns, self.cached_cell, progress)
        self.stats['build_seconds'] = time.perf_counter() - started
        return True

    def host_arrays(self):
        self.build_packed()
        started = time.perf_counter()
        cells = self.__dict__.pop('_joint_cells', {})
        if len(cells) == self.s.geo.ns:
            self._joint = {
                k: np.concatenate([cells[i][k] for i in range(self.s.geo.ns)])
                for k in ('density', 'strato')
                if all(k in cells[i] for i in range(self.s.geo.ns))
            }
        with stage('geometry_cache_load'):
            host, records = self.packed().open()
        if len(records) != self.s.geo.ns:
            raise RuntimeError('E_PACKED_CELLS: cache cell count differs from grid')
        self.stats['load_seconds'] = time.perf_counter() - started
        return host, records

    def install(self, host, records, replace=None):
        """Upload host arrays (replace: dict of host arrays overriding fields)."""
        xp = self.s.kernel.xp
        before = time.perf_counter()
        replace = replace or {}
        with stage('geometry_upload', device=True):
            a = {
                k: (
                    xp.asarray(np.ascontiguousarray(replace.get(k, v)))
                    if xp is not np
                    else np.asarray(replace.get(k, v))
                )
                for k, v in host.items()
            }
            self.s.kernel.sync()
        self.stats['upload_seconds'] = time.perf_counter() - before
        offsets = np.array(host['cell_off'], 'i8')
        self._host_offsets = offsets
        rays = int(offsets[-1])
        bundle = Bundle.__new__(Bundle)
        bundle.arrays = a
        bundle.cells = len(records)
        bundle.rays = rays
        bundle.nq = self.s.geo.cfg['path_n']
        bundle.bytes = sum(v.nbytes for v in a.values())
        self.bundle = bundle
        self.stats.update(
            cells=bundle.cells, rays=rays, samples=sum(m['samples'] for m in records), compact_bytes=bundle.bytes
        )
        self.ground_bundle = self.prefix(int(offsets[len(self.s.geo.chi)]))

    def prepare(self):
        if self.bundle is not None:
            return
        host, records = self.host_arrays()
        self.install(host, records)

    def prefix(self, rays):
        if self.bundle is None:
            self.prepare()
        rays = min(int(rays), self.bundle.rays)
        if rays <= 0:
            raise ValueError('E_RAY_COUNT: positive count required')
        offsets = self._host_offsets
        cells = int(np.searchsorted(offsets, rays, side='left'))
        cells = max(1, min(cells, len(offsets) - 1))
        off = offsets[: cells + 1].copy()
        off[-1] = rays
        source = self.bundle.arrays
        xp = self.s.kernel.xp
        edges = int(self.s.kernel.host(source['edge_off'][rays : rays + 1])[0])
        a = dict(
            cell_off=xp.asarray(off),
            starts=source['starts'][:cells],
            targets=source['targets'][:cells],
            edge_off=source['edge_off'][: rays + 1],
            edges=source['edges'][:edges],
        )
        for name in ('dirs', 'weights', 'lower', 'end_j', 'end_f', 'end_col', 'hits'):
            a[name] = source[name][:rays]
        bundle = Bundle.__new__(Bundle)
        bundle.arrays = a
        bundle.cells = cells
        bundle.rays = rays
        bundle.nq = self.bundle.nq
        bundle.bytes = sum(v.nbytes for v in a.values())
        return bundle

    def evaluate(self, s, field, first=False, phase=0, ground_only=False, boundary=None, timed=False):
        self.prepare()
        bundle = self.ground_bundle if ground_only else self.bundle
        moment, flux, _ = s.kernel.launch(bundle, s, field, first, phase, boundary, timed)
        return moment, flux


class TwoLayerGeometry(ResidentGeometry):
    """COMMON endpoint layer (no scale height in its signature) +
    per-height aerosol column end_col[:,1].

    The common layer is any complete packed build; it may be converted from a compatible v1
    cache of ANY scale height (identity equal except height).  The aerosol column
    of the current height is loaded from common/aerosol/H_<hex>.f4 or recomputed by
    aerosol_column() and stored.  The device end_col is the composite (rays,5)
    contiguous array, so kernel.cu's boundary attenuation reads this state's column.
    """

    _version = 3

    def find_legacy(self, legacy):
        mine = identity(self.s, 1)[0]
        found = []
        for path in sorted(self.root.glob('endpoints_*/identity.json')):
            try:
                other = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if other.get('version') != 1:
                continue
            if {k: v for k, v in other.items() if k != 'height'} != {
                k: v for k, v in mine.items() if k not in ('height', 'aerosol_profile')
            }:
                continue
            if (path.parent / f'cell_{self.s.geo.ns - 1:05d}' / 'complete.json').exists():
                found.append((other['height'] != mine['height'], path.parent, other['height']))
        if not found:
            return self.root / ('endpoints_' + legacy)
        found.sort(key=lambda f: (f[0], str(f[1])))
        self.legacy_height = found[0][2]
        return found[0][1]

    def column_path(self, height):
        # 'H_<float.hex(height)>' without a troposphere_profile block (unchanged), else the profile tag
        return self.cache / 'aerosol' / f'{tropo_profile.column_name(getattr(self, "s", None), height)}.f4'

    def build_packed(self):
        journal = self.cache / 'packed' / 'journal.jsonl'
        resumed = not self.packed().complete() and journal.exists() and journal.stat().st_size > 0
        built = super().build_packed()
        source = self.cache / 'source_height.json'
        if built and resumed:
            source.write_text(json.dumps(dict(height=None)) + '\n')
        elif built or not source.exists():
            # Height that produced the common end_col[:,1], only if one source built
            # every cell in this process (all legacy or all generated); else unknown.
            read = self.stats.get('legacy_cells_read', 0)
            h = None
            if built and read == self.s.geo.ns:
                h = getattr(self, 'legacy_height', None)
            elif built and read == 0:
                h = (
                    self.s.kernel.height if tropo_profile.of(self.s) is None else None
                )  # a two-layer build is no scale height's column
            source.write_text(json.dumps(dict(height=h)) + '\n')
        return built

    def prepare(self):
        if self.bundle is not None:
            return
        host, records = self.host_arrays()
        height = self.s.kernel.height
        path = self.column_path(height)
        source = json.loads((self.cache / 'source_height.json').read_text())['height']
        started = time.perf_counter()
        if path.exists():
            with stage('aerosol_column_load'):
                column = np.fromfile(path, 'f4')
            how = 'cached'
        elif tropo_profile.same_source(self.s, source, height):
            column = np.array(host['end_col'][:, 1])
            how = 'common layer (same height)'
        else:
            with stage('aerosol_column_compute'):
                column = aerosol_column(host, height, profile=tropo_profile.column_function(self.s))
            how = 'computed'
        if how != 'cached':
            path.parent.mkdir(exist_ok=True)
            tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
            column.tofile(tmp)
            os.replace(tmp, path)
        if column.shape != (len(host['dirs']),):
            raise RuntimeError('E_AEROSOL_COLUMN: wrong length')
        end_col = np.array(host['end_col'])
        end_col[:, 1] = column
        self.stats.update(
            aerosol_height=height,
            aerosol_column=how,
            aerosol_column_seconds=time.perf_counter() - started,
            common_source_height=source,
        )
        if tropo_profile.of(self.s) is not None:
            self.stats['aerosol_profile'] = tropo_profile.of(self.s)['tag']
        from ..precompute import host_cache

        optics = getattr(self.s, 'density_optics', None)
        if host_cache.enabled() and optics is not None:
            end_col[:, [0, 2, 3, 4]] = host_cache.endpoint_columns(
                self.s,
                host,
                optics,
                self.cache,
                self.stats['signature'],
                prepared=getattr(self, '_joint', {}).get('density'),
            )
        self.install(host, records, dict(end_col=end_col))
        joint = self.__dict__.pop('_joint', {})
        if 'strato' in joint:
            for bundle in (self.bundle, self.ground_bundle):
                bundle.prepared_strato = joint['strato'][: bundle.rays]
        if host_cache.enabled() and optics is not None:
            # Installed bytes already include the density correction. Prefixes
            # share those exact bytes, including the current state's col1.
            from ..atmosphere.density import profile_identity

            key = profile_identity(optics)['sha256']  # same key as fused.density_end_columns and the IPC check
            self.bundle.density_columns = {key: self.bundle.arrays['end_col']}
            self.ground_bundle.density_columns = {key: self.ground_bundle.arrays['end_col']}


# Intentional same-layer cycle: the adapter replaces PackedCache after its serial definition exists.
from . import ray_geometry_parallel  # noqa: E402,F401
