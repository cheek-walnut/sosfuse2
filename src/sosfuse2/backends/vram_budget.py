"""Device-memory estimates as named byte counts. Each Counter totals one set of simultaneous costs."""

from collections import Counter

from .device_profile import reserved_field_bytes

KSPLIT_STATE_VRAM_GB = 5.1
VPROD_STORE_BPP = 28


def aero_w(W):
    """Aerosol record width in floats, padded for float4 stores."""
    return ((W + 3) // 4) * 4


def aerosol_bytes(points, W):
    return points * 4 * aero_w(W) * 4


def aerosol_cap(mb, W):
    return max(1, int(float(mb) * 2**20) // aerosol_bytes(1, W))


def ksplit_reserve(mode, free, field_bytes):
    if mode == 1:
        return Counter(headroom=free - int(0.9 * free))
    return Counter(source_fields=16 * field_bytes, headroom=max(256 << 20, int(0.03 * free)))


def bundle_fixed_items(column_bytes, cells, phase, copies, grid_cells, K, W):
    return Counter(
        column_records=column_bytes,
        moments_and_split_scratch=int(copies * cells * (4 + K) * W * 8) if phase == 0 else 0,
        sh_sidecar_and_scratch=int(5 * grid_cells * K * W * 4),
    )


def split_peak_items(rows, cap, W):
    return Counter(
        column_records=sum(row['fixed']['column_records'] for row in rows), aerosol_chunk=aerosol_bytes(cap, W)
    )


def vprod_option(fixed, points, pmax, rmax, W, K, mode, store, resident, verify, verify_bytes):
    """Bundle peak, including just the workspace that is live during the more expensive stage."""
    workspace = (21 * 4 + 4) * pmax + (1 << 20)
    records = 8 * (points if resident == 'device' else pmax)
    stored = VPROD_STORE_BPP * points if store else 0
    base = fixed + records + stored
    run = Counter(
        aerosol_chunk=aerosol_bytes(pmax, W),
        ray_basis=rmax * K * 4,
        verify_stage=verify_bytes if verify else 0,
        plan_workspace=0 if store else workspace,
        ray_radiances=4 * W * 4 * rmax * (2 if verify else 1) if mode == 3 else 0,
    )
    build = Counter(plan_workspace=workspace)
    work = max((run, build), key=Counter.total) if store else run
    return dict(
        peak=base + work.total(),
        run=base + run.total(),
        work_items=work,
        workspace=workspace,
        y_bytes=run['ray_basis'],
        records_bytes=records,
        store_bytes=stored,
    )


# Work per second used before the first CUDA event. A point visits four source
# corners; direct contraction also traverses all SH modes. Include projection
# work so a cell with many short or empty rays cannot escape the probe cap.
FAST_WORK_RATE = 1 << 26
PLAN_COUNT_BLOCKS = 256
PLAN_VISITORS = 256
COUNT_SHARED_BYTES = 48 << 10
INT32_MAX = (1 << 31) - 1


def count_segments(records, blocks):
    # A scatter segment is a multiple of four 32-record warp tiles.
    segment = -(-max(4096, -(-records // blocks)) // 128) * 128
    return segment, -(-records // segment)


def transport_scratch(points, rays, cells, K, W, direct, flux_only):
    """Worst simultaneous bytes, including library scan/sort temporaries and block ordering."""
    n = 4 * points
    # Grid tails round up to 256 threads. Target and flux rows use 9*cell
    # and W*cell in the CUDA ABI; sample cells use signed int indices.
    if max(n + 256, rays + 256, cells + 256, cells * max(9, W)) > INT32_MAX:
        return None
    ray_rows = max(1, rays) * (4 * W + (0 if flux_only and not direct else K)) * 4
    if not direct:
        return ray_rows
    nb = min(PLAN_COUNT_BLOCKS, -(-n // 4096))  # monotone upper bound on scatter segments
    blocks = min(n, cells + -(-n // PLAN_VISITORS))
    if max(cells * nb if 4 * cells <= COUNT_SHARED_BYTES else 0, 3 * blocks, n + 1) > INT32_MAX:
        return None
    fixed = ray_rows + max(1, points) * (8 + 4 * aero_w(W) * 4)
    # slot + cell are live across sorting. Counting keeps counts, offsets
    # and scan scratch. The key sort reserves keys and two scratch copies.
    sort = 12 * cells * nb if 4 * cells <= COUNT_SHARED_BYTES else 24 * n
    starts = 8 * (cells + 1)
    sorting = 8 * n + starts + sort
    visitors = (8 + 1 + 4 + 8 + 4) * n + 4 * points + 4 * starts
    ordering = 12 * n + 5 * starts + 112 * blocks
    return fixed + max(sorting, visitors, ordering) + (1 << 20)


def transport_chunks(co, pcum, cells, K, W, direct, flux_only, step, seconds, free, resident, *, budget=None):
    """Return cell boundaries and a cumulative work proxy, or None before allocation.

    Reserve the two independent maxima across *all* chunks. In particular the
    chunk with most rays need not contain the most points.
    """
    import numpy as np

    if len(co) <= 1:
        return None
    available = free - resident - (64 << 20)
    points_max = int(np.diff(pcum).max())
    rays_max = int(np.diff(co).max())

    def fits(p, r):
        size = transport_scratch(p, r, cells, K, W, direct, flux_only)
        return size is not None and size <= available

    if not fits(points_max, rays_max):
        return None
    work = (pcum - pcum[0]).astype('f8') * (4 * (K if direct else 1)) + (co - co[0]) * float(K)
    limit = max(1, int(FAST_WORK_RATE * seconds))
    hard_limit = max(limit, int(FAST_WORK_RATE * (budget or seconds)))
    # The direct visitor plan scans the full source grid even for a ray
    # prefix. The counting branch also clears and scans one row per segment.
    def plan_work(points):
        if not direct:
            return 0
        segments = min(PLAN_COUNT_BLOCKS, -(-(4 * points) // 4096)) if 4 * cells <= COUNT_SHARED_BYTES else 0
        return 4 * cells * (1 + segments)

    worst = float(np.max(np.diff(work)))
    if worst + plan_work(points_max) > hard_limit:
        raise RuntimeError('E_LAUNCH_BUDGET: one transport cell exceeds the pre-launch work cap')
    chunks = []
    a = 0
    while a < len(co) - 1:
        hi = min(len(co) - 1, a + step if step else len(co) - 1,
                 max(a + 1, int(np.searchsorted(work, work[a] + limit, side='right')) - 1))
        lo = a + 1
        while lo <= hi:
            b = (lo + hi) // 2
            p = int(pcum[b] - pcum[a])
            within_target = b == a + 1 or work[b] - work[a] + plan_work(p) <= limit
            if within_target and fits(max(points_max, p), max(rays_max, int(co[b] - co[a]))):
                lo = b + 1
            else:
                hi = b - 1
        points_max = max(points_max, int(pcum[hi] - pcum[a]))
        rays_max = max(rays_max, int(co[hi] - co[a]))
        chunks.append((a, hi))
        a = hi
    return chunks, work


def state_peak_items(rows, plans):
    """Keep columns, records and stored plans for every bundle. Only one bundle runs or builds at a time."""
    retained = Counter()
    transients = []
    for row, plan in zip(rows, plans, strict=True):
        fixed = row['fixed']
        retained.update(
            column_records=fixed['column_records'], vprod_records=plan['records_bytes'], vprod_store=plan['store_bytes']
        )
        work = plan['work_items'].copy()
        work.update({name: size for name, size in fixed.items() if name != 'column_records'})
        transients.append(work)
    retained.update(max(transients, key=Counter.total))
    return retained


def admission_items(state, payload_bytes, ksplit):
    """Persistent admission uses a fixed field width across SH orders."""
    field_bytes = state.geo.ns * (1132 if state.strato is not None else 571) * state.W * 4
    return Counter(
        payload_arrays=payload_bytes,
        source_fields=reserved_field_bytes(field_bytes),
        ksplit_floor=int(KSPLIT_STATE_VRAM_GB * 1e9) if ksplit else 0,
    )
