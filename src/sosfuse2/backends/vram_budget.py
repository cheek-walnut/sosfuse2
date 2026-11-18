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
