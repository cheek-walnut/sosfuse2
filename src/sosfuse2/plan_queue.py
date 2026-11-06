#!/usr/bin/env python3
"""Cache-aware queue planner for the C3 SOS calculator: CPU only, never imports CuPy, writes no cache.

For every state it computes, with the calculator's own functions, the cold CPU work a queue would do and whether a
cache already holds it:
  column   column_cache.cache_key(kernel, bundle, c, 'strato') of the main and ground bundles that
           fused.Kernel.strato_end_columns integrates (rays from the packed geometry cache, installed by the
           calculator's geometry.ResidentGeometry.install/prefix; c = the stratosphere profile and `old`);
  attach   run_gpu_state_v3.attach_lock_name(state) (the group lock) and stratosphere._geometry_key (the file);
  host     the host_cache keys sun02 / ground / endpoint02 of density states, captured from the calculator's own
           corrected_sun / prepare_ground / endpoint_columns with the cache I/O replaced by a recorder;
  aerosol  the per-scale-height aerosol column file of the two-layer geometry.
The state-dependent pieces come from one probe per class (parameters except aod550/stratosphere_tau550/albedo,
density options): run_gpu_state_v3.create_configured with the stratosphere switched off (attach returns before its
ground geometry) and the host-cache ground step skipped.  The table bundles (about 4,000 of 10 million rays per
state, eda logs 2026-10-07) are not keyed here.

Platform: the attach and sun02 keys hash floating-point arrays computed on the host (sun columns).  Mac and rig bytes
differ (2026-10-07: 26,378 of 32,481 solar-grid entries, max_rel 2.7e-15), so their hit prediction is valid only when
the planner runs on the queue's own host.  Column, ground, endpoint02 and aerosol keys hash cached geometry bytes,
profile reprs and sources, and match across hosts (rig and eda keys reproduced on the Mac, test_plan_queue.py).

Order: representatives of the missing items first, spread over GPUs and workers, then states with nothing to
compute, then the remaining members of missing groups (they hit once their representative has published).  With
several GPUs the states are split by estimated seconds (cost profile below).  pipeline.py --plan <plan> --plan-gpu k
runs GPU k's list in that order.

  python -B -m sosfuse2.plan_queue --states STATES.json --geometry-cache GEOMETRY_CACHE --cache-root CACHES \
      --gpus 2 --workers-per-gpu 3 --out plan.json
"""

import argparse
import contextlib
import copy
import hashlib
import json
import os
import platform
import shlex
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from sosfuse2 import paths


FULL_MAIN_RAYS = 9586008  # Packed geometry cache layout.json
# Serial seconds of one cold item.  None = not measured: counted as 0 in the totals and listed as unmeasured.
COST_PROFILES = {
    'eda': dict(
        strato_column_seconds_per_ray=118.0 / (9586008 + 404496),
        density_column_seconds_per_ray=21.0 / 9586008,
        attach_miss_seconds=51.3,
        host_sun02_seconds=None,
        host_ground_seconds=None,
        aerosol_column_seconds=None,
        geometry_build_seconds=None,
        state_seconds=dict(strato=2006.0, off=1030.0),
        sources=dict(
            strato_column='development GPU host CPU profile, 2026-10-03 (research record): stratosphere columns about 118 s; '
            'rays 9,586,008 main + 404,496 ground (bake logs of 2026-10-07)',
            density_column='same profile: density column correction about 21 s; taken as the '
            'main-bundle correction (guess: the profile does not name the code path)',
            attach_miss='bake log of 2026-10-07 (research record), state L19s3_tau0p01_aod0p15_g0p65: attach_cache miss 51.3 s',
            state='mean source_done of the r2 runs in the 2026-10-07 bake logs (two-slot shared GPU): '
            'stratosphere on 2006 s (5 states), off 1030 s (2 states); used only to balance GPUs',
        ),
    ),
    'rig4090': dict(
        strato_column_seconds_per_ray=145.0 / 9586008,
        density_column_seconds_per_ray=9.9 / 450000,
        attach_miss_seconds=77.9,
        host_sun02_seconds=None,
        host_ground_seconds=None,
        aerosol_column_seconds=None,
        geometry_build_seconds=None,
        state_seconds=dict(strato=52.0, off=82.8),
        sources=dict(
            strato_column='leased 4090 run of 2026-10-03 (research record), state B22s5_tau0p005_aod0p15: column_cache miss 145.0 s, 9,586,008 rays',
            density_column='same log: density column_cache miss 9.9 s for a 450,000-ray table bundle',
            attach_miss='same log: attach_cache miss 77.9 s',
            state='same run (research record): 12 stratosphere states in 10 min 24 s (52 s each, overlapped), '
            '5 stratosphere-off states in 6 min 54 s; used only to balance GPUs',
        ),
    ),
    'mac': dict(
        strato_column_seconds_per_ray=39.60 / 9586008,
        density_column_seconds_per_ray=55.68 / 9586008,
        attach_miss_seconds=None,
        host_sun02_seconds=None,
        host_ground_seconds=None,
        aerosol_column_seconds=None,
        geometry_build_seconds=None,
        state_seconds=dict(strato=1.0, off=1.0),
        sources=dict(
            strato_column='Mac column-parallel measurement of 2026-10-07 (research record; serial, 9,586,008 rays)',
            density_column='same file: fused density_end_columns serial, 9,586,008 rays (the host-cache endpoint02 '
            'path runs the same per-chunk correction)',
            state='no GPU on the Mac: equal weights',
        ),
    ),
}


# Column-build speed-up by SOSFUSE2_COLUMN_WORKERS, the lowest of the three kinds measured on the Mac full grid
# (measured during development): strato 1.94/3.36/4.72, density 1.96/3.52/4.98,
# endpoint02 3.50/5.02 at 2/4/8.  Linear between points; held at the 8-worker value above 8 (not measured).
SPEEDUP = ((1, 1.0), (2, 1.94), (4, 3.36), (8, 4.72))
PARALLEL_KINDS = ('column', 'host_endpoint02')  # items the column workers divide; the others stay serial


def speedup(workers):
    points = [(w, s) for w, s in SPEEDUP if w <= workers]
    (w0, s0) = points[-1]
    above = [(w, s) for w, s in SPEEDUP if w > workers]
    if not above:
        return s0
    w1, s1 = above[0]
    return s0 + (s1 - s0) * (workers - w0) / (w1 - w0)


def resources(a, n_states):
    """States per GPU (W) and column workers per state (T) for the commands, each with its basis.
    The planner runs on the queue host, so the cores read here are the queue's."""
    from sosfuse2 import pipeline
    from sosfuse2.host_limits import RIG_MIN_RAM_GB, rss_cap_gb

    try:
        from sosfuse2.host import host_cores

        cores = min(pipeline.cpu_budget(), host_cores.cores()[0])
    except Exception:  # host_cores reads Linux sources; the pipeline's own budget is the check it applies
        cores = pipeline.cpu_budget()
    per_gpu = -(-n_states // a.gpus)
    if a.workers_per_gpu is not None:
        workers, wbasis = a.workers_per_gpu, '--workers-per-gpu'
    else:
        auto = pipeline.auto_workers(per_gpu, 1, 'cupy', True)
        workers = max(1, min(auto, (cores - pipeline.CORES_LEFT_FREE) // a.gpus)) if a.gpus > 1 else auto
        wbasis = 'pipeline.auto_workers(%d states, threads 1, cupy, persistent) = %d' % (per_gpu, auto) + (
            ''
            if a.gpus == 1
            else ', at most (%d cores - %d free) // %d GPUs' % (cores, pipeline.CORES_LEFT_FREE, a.gpus)
        )
    if a.column_workers is not None:
        threads, tbasis = a.column_workers, '--column-workers'
    elif rss_cap_gb() < RIG_MIN_RAM_GB:
        threads, tbasis = (
            1,
            (
                'memory-protected host (host_limits.rss_cap_gb %g GB: Mac, eda): the eda slot PSS budget '
                '(configs/devices.json slot_pss_gb) does not count column worker processes' % rss_cap_gb()
            ),
        )
    else:
        threads = max(1, (cores - pipeline.CORES_LEFT_FREE) // (a.gpus * workers))
        tbasis = (
            '(%d cores - %d free) // (%d GPUs x %d states): every concurrent state may run a cold build at once'
            % (cores, pipeline.CORES_LEFT_FREE, a.gpus, workers)
        )
    if workers * threads > pipeline.cpu_budget():
        raise ValueError(
            'E_PLAN_CPU_BUDGET: workers x column workers exceeds this host (pipeline.cpu_budget %d)'
            % pipeline.cpu_budget()
        )
    return dict(
        cores=cores,
        workers=workers,
        workers_basis=wbasis,
        column_workers=threads,
        column_workers_basis=tbasis,
        column_speedup=speedup(threads),
    )


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--states', type=Path, required=True)
    p.add_argument('--ids', help='comma list in queue order (default: every runnable state in file order)')
    p.add_argument('--geometry-cache', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, help='deployment cache root holding column_cache/ and attach_cache/')
    p.add_argument(
        '--column-cache',
        type=Path,
        help='default <cache-root>/column_cache, else <geometry-cache>/column_cache (pipeline.py)',
    )
    p.add_argument(
        '--attach-cache',
        type=Path,
        help='default <cache-root>/attach_cache, else STRATO_ATTACH_CACHE, else <geometry-cache>/attach_cache',
    )
    p.add_argument('--settings-json', type=Path)
    p.add_argument('--geometry-layers', type=int, choices=(1, 2), default=2)
    p.add_argument('--stratosphere', choices=('on',))
    p.add_argument('--density-profile', choices=('exp8', 'usstd76'))
    p.add_argument('--ozone-profile', choices=('triangle', 'usstd76'))
    p.add_argument('--rayleigh-column', choices=('keep', 'usstd'))
    p.add_argument('--gpus', type=int, default=1)
    p.add_argument(
        '--workers-per-gpu',
        type=int,
        help='states prepared at once per GPU (pipeline --workers); default from '
        'pipeline.auto_workers on this host, shared over --gpus',
    )
    p.add_argument(
        '--column-workers',
        type=int,
        help='column worker processes per state (pipeline --threads); default from '
        'the cores left per concurrent state, 1 on memory-protected hosts (Mac, eda)',
    )
    p.add_argument('--cost-profile', choices=sorted(COST_PROFILES), default='eda')
    p.add_argument('--costs', type=Path, help='JSON object overriding cost-profile fields')
    p.add_argument('--work-root', type=Path, help='work root for the printed commands (default <out dir>/work)')
    p.add_argument('--scratch', type=Path, help='probe folder (default: a temporary folder, removed)')
    p.add_argument('--out', type=Path, required=True, help='plan JSON; the summary goes to <out>.txt and stdout')
    return p


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---- probe: the calculator's own solver constructor, minus the cold work ---------------------------------------


class _Skip(Exception):
    """Not a host_cache.CacheUnavailable, so no uncached fallback runs."""


@contextlib.contextmanager
def recording(records):
    """host_cache.cached records (kind, token, folder) instead of reading or writing; sun02 is computed in memory
    (its corrected sun columns enter the attach key), ground and endpoint02 are skipped."""
    from sosfuse2 import host_cache

    cached, ground = host_cache.cached, host_cache.prepare_ground

    def record(folder, key, build, compressed=False):
        records.append(dict(kind=key['kind'], token=host_cache.digest(key), folder=str(folder)))
        if key['kind'] == 'sun02':
            return build()
        raise _Skip(key['kind'])

    def skip_ground(s, scope, optics, method=None):  # method: density.attach's sun_method (③ 0(b))
        try:
            return ground(s, scope, optics, method)
        except _Skip:
            return None

    host_cache.cached, host_cache.prepare_ground = record, skip_ground
    try:
        yield
    finally:
        host_cache.cached, host_cache.prepare_ground = cached, ground


def probe_class(state):
    params = {
        k: v for k, v in state.get('parameters', {}).items() if k not in ('aod550', 'stratosphere_tau550', 'albedo')
    }
    from sosfuse2.density import options, build, profile_identity, profile_step_km
    from sosfuse2.tropo_profile import config, canonical

    key = dict(params=params, density=options(state))
    # ②b: states with different PROFILE tables/grids must not share one probe (host sun/endpoint/ground records).
    # Ordinary states (no step, no external table) keep their old class key.
    raw = state.get('density', {})
    if isinstance(raw, dict) and ('profile_step_km' in raw or 'external_profile' in raw):
        optics = build(state)
        key['profile'] = dict(
            step_km=profile_step_km(state), identity=None if optics is None else profile_identity(optics)['sha256']
        )
    profile = config(state)
    if profile is not None:
        key['aerosol_profile'] = canonical(
            profile
        )  # two-layer shape depends on aod550 too (absent block: key unchanged)
    return json.dumps(key, sort_keys=True, default=str)


def probe(state, cfg, geometry_cache, folder):
    import numpy as np
    from sosfuse2 import run_gpu_state_v3 as rv

    cfg = rv.resolve_settings(cfg, state)  # A′: per-state spectral block (the shared cfg is not changed)
    geo_state = copy.deepcopy(state)
    layer = geo_state.get('stratosphere')
    if isinstance(layer, dict):
        layer['status'] = 'disabled'  # stratosphere.tau_of -> 0: attach() returns before _ground_geometry
    wl = np.asarray(cfg['wavelengths_nm'])
    chunk = wl if len(wl) in (1, 15) else wl[:15]
    if len(chunk) not in (1, 15):
        chunk = np.r_[chunk, np.repeat(chunk[-1], 15 - len(chunk))]
    records = []
    with recording(records):
        s, scope = rv.create_configured(folder, cfg, chunk, 'numpy', geo_state, 128, geometry_cache=geometry_cache)
    return SimpleNamespace(s=s, scope=scope, records=records, old=scope['b'].sh.old)


def geometry_layer(s, root, layers, kernel):
    """Endpoint folder of this grid and, if its packed cache is complete, the calculator's main and ground bundles."""
    from sosfuse2 import geometry as gm, geometry_contract as gc

    signature = gc.identity(s)[1] if layers == 2 else gm.identity(s, 2)[1]
    folder = Path(root) / ('endpoints_' + signature)
    out = dict(signature=signature, folder=folder, complete=False, bundles={}, host=None)
    packed = gm.PackedCache(folder / 'packed')
    if not packed.complete():
        return out
    host, records = packed.open()
    if len(records) != s.geo.ns:
        return out
    geometry = gm.ResidentGeometry.__new__(
        gm.ResidentGeometry
    )  # install()/prefix() only; __init__ writes identity.json
    geometry.s = SimpleNamespace(kernel=kernel, geo=s.geo)
    geometry.stats = {}
    geometry.bundle = None
    geometry.install(host, records)
    out.update(complete=True, host=host, bundles=dict(main=geometry.bundle, ground=geometry.ground_bundle))
    return out


def endpoint_record(s, host, folder, signature):
    from sosfuse2 import host_cache

    records = []
    with recording(records):
        try:
            host_cache.endpoint_columns(s, host, s.density_optics, folder, signature)
        except _Skip:
            pass
    return records[0]


# ---- items, representatives, order -----------------------------------------------------------------------------


def cost_profile(a):
    values = copy.deepcopy(COST_PROFILES[a.cost_profile])
    if a.costs is not None:
        extra = json.loads(a.costs.read_text())
        if not isinstance(extra, dict) or set(extra) - set(values):
            raise ValueError('E_PLAN_COSTS: unknown cost fields')
        values.update(extra)
        values.setdefault('sources', {})['override'] = str(a.costs)
    return values


def collect(a, cfg, states, columns, attach_root):
    """Per-state item lists and the item table (id -> kind, key, file, hit, cost)."""
    import numpy as np
    from sosfuse2 import column_cache, host_cache, geometry as gm, tropo_profile
    from sosfuse2.fused import Kernel
    from sosfuse2.run_gpu_state_v3 import attach_lock_name
    from sosfuse2.stratosphere import _geometry_key, profile_config, tau_of

    costs = cost_profile(a)
    kernel = Kernel(backend='numpy')
    items = {}
    rows = []
    probes = {}
    geometry = {}
    host_enabled = host_cache.enabled()

    def item(iid, **fields):
        it = items.setdefault(iid, dict(fields, states=[]))
        return it

    def rate(name, rays):
        v = costs[name]
        return None if v is None or rays is None else v * rays

    from sosfuse2.density import profile_identity as density_identity

    scratch = a.scratch or Path(tempfile.mkdtemp(prefix='plan-queue-'))
    try:
        for state in states:
            started = time.monotonic()
            klass = probe_class(state)
            if klass not in probes:
                p = probe(state, cfg, a.geometry_cache, scratch / f'probe_{len(probes)}')
                g = geometry_layer(p.s, a.geometry_cache, a.geometry_layers, kernel)
                p.geometry = g
                p.seconds = time.monotonic() - started
                if (
                    getattr(p.s, 'density_optics', None) is not None
                    and host_enabled
                    and a.geometry_layers == 2
                    and g['complete']
                ):
                    p.records.append(endpoint_record(p.s, g['host'], g['folder'], g['signature']))
                probes[klass] = p
                geometry[g['signature']] = g
            p = probes[klass]
            s = p.s
            g = p.geometry
            row = dict(
                id=state['id'],
                probe=list(probes).index(klass),
                strato=None,
                attach_lock=attach_lock_name(state),
                scale_height_km=float(s.kernel.height),
                density=None
                if getattr(s, 'density_optics', None) is None
                else dict(
                    mask=int(s.density_optics['mask']),
                    asset_sha256=s.density_optics['meta'].get('asset_sha256'),
                    profile_identity=density_identity(s.density_optics)['sha256'],
                    profile_n=int(s.kernel.profile_n),
                ),
                items=[],
            )
            if not g['complete']:
                it = item(
                    'geometry:' + g['signature'],
                    kind='geometry',
                    key=g['signature'],
                    file=str(g['folder'] / 'packed'),
                    hit=False,
                    cost_seconds=costs['geometry_build_seconds'],
                    platform_dependent=False,
                )
                row['items'].append('geometry:' + g['signature'])
            if tau_of(state) > 0:
                profile = profile_config(state)
                row['strato'] = {k: profile[k] for k in ('center', 'sigma', 'norm')}
                c = dict(strato_old=p.old, strato_profile=profile)
                if g['complete']:
                    for name in ('main', 'ground'):
                        bundle = g['bundles'][name]
                        key = column_cache.cache_key(kernel, bundle, c, 'strato')
                        path = Path(columns) / (key + '.npy') if columns else None
                        hit = bool(path is not None and path.is_file() and path.with_suffix('.sha256').is_file())
                        item(
                            'column:' + key,
                            kind='column',
                            bundle=name,
                            key=key,
                            rays=int(bundle.rays),
                            file=str(path),
                            hit=hit,
                            cost_seconds=rate('strato_column_seconds_per_ray', bundle.rays),
                            platform_dependent=False,
                            profile=row['strato'],
                        )
                        row['items'].append('column:' + key)
                else:
                    iid = 'column:unknown:%s:%r,%r' % (g['signature'], profile['center'], profile['sigma'])
                    item(
                        iid,
                        kind='column',
                        bundle='main+ground',
                        key=None,
                        rays=None,
                        file=None,
                        hit=False,
                        cost_seconds=rate('strato_column_seconds_per_ray', FULL_MAIN_RAYS),
                        platform_dependent=False,
                        profile=row['strato'],
                    )
                    row['items'].append(iid)
                s.fuse_state = state  # the attach key reads parameters except stratosphere.ATTACH_OPTICS_PARAMS
                key = _geometry_key(s, p.scope, profile)
                path = Path(attach_root) / f'attach_geometry_{key}.npz'
                it = item(
                    'attach:' + key,
                    kind='attach',
                    key=key,
                    lock=row['attach_lock'],
                    file=str(path),
                    hit=path.is_file(),
                    lock_done=(Path(attach_root) / (str(row['attach_lock']) + '.done')).is_file(),
                    cost_seconds=costs['attach_miss_seconds'],
                    platform_dependent=True,
                )
                row['items'].append('attach:' + key)
            if row['density'] is not None and host_enabled:
                for rec in p.records:
                    path = Path(rec['folder']) / (rec['token'] + '.hcache')
                    name = dict(sun02='host_sun02_seconds', ground='host_ground_seconds').get(rec['kind'])
                    cost = costs[name] if name else rate('density_column_seconds_per_ray', g['bundles']['main'].rays)
                    item(
                        'host:' + rec['token'],
                        kind='host_' + rec['kind'],
                        key=rec['token'],
                        file=str(path),
                        hit=path.is_file(),
                        cost_seconds=cost,
                        platform_dependent=rec['kind'] == 'sun02',
                    )
                    row['items'].append('host:' + rec['token'])
            if a.geometry_layers == 2 and g['complete']:
                own = SimpleNamespace(
                    aerosol_profile=tropo_profile.build(state)
                )  # this state's profile, not the probe's
                column = gm.TwoLayerGeometry.column_path(SimpleNamespace(cache=g['folder'], s=own), s.kernel.height)
                try:
                    source = json.loads((g['folder'] / 'source_height.json').read_text())['height']
                except (OSError, ValueError, KeyError):
                    source = None
                same = tropo_profile.same_source(own, source, s.kernel.height)
                iid = 'aerosol:%s:%s' % (
                    g['signature'],
                    float(s.kernel.height).hex() if own.aerosol_profile is None else own.aerosol_profile['tag'],
                )
                if own.aerosol_profile is not None:
                    row['aerosol_profile'] = own.aerosol_profile['tag']
                item(
                    iid,
                    kind='aerosol',
                    key=column.name,
                    file=str(column),
                    hit=column.is_file() or same,
                    cost_seconds=costs['aerosol_column_seconds'],
                    platform_dependent=False,
                )
                row['items'].append(iid)
            for iid in row['items']:
                items[iid]['states'].append(row['id'])
            rows.append(row)
    finally:
        if a.scratch is None:
            import shutil

            shutil.rmtree(scratch, ignore_errors=True)
    return rows, items, probes, geometry, costs


def order(rows, items, costs, gpus, workers, column_speedup):
    """Greedy representatives (largest uncovered missing cost first), then GPU lists by estimated seconds."""
    weight = lambda iid: items[iid]['cost_seconds'] if items[iid]['cost_seconds'] is not None else 1.0
    uncovered = {i for i, it in items.items() if not it['hit']}
    reps = []
    owner = {}
    while uncovered:
        best = max(rows, key=lambda r: (sum(weight(i) for i in r['items'] if i in uncovered), -rows.index(r)))
        gain = [i for i in best['items'] if i in uncovered]
        if not gain:
            break
        reps.append(best)
        for i in gain:
            owner[i] = best['id']
            uncovered.discard(i)
    for it_id, it in items.items():
        it['owner'] = owner.get(it_id)
    rep_ids = [r['id'] for r in reps]
    for r in rows:
        missing = [i for i in r['items'] if not items[i]['hit']]
        r['missing'] = missing
        own = [items[i] for i in missing if owner.get(i) == r['id']]
        r['cold_cpu_seconds'] = sum(it['cost_seconds'] or 0.0 for it in own)
        r['cold_seconds'] = sum(
            (it['cost_seconds'] or 0.0) / (column_speedup if it['kind'] in PARALLEL_KINDS else 1.0) for it in own
        )
        r['role'] = 'representative' if r['id'] in rep_ids else 'member' if missing else 'warm'
        r['waits_for'] = sorted({owner[i] for i in missing if owner.get(i) not in (None, r['id'])})
        base = costs['state_seconds']['strato' if r['strato'] else 'off']
        r['est_seconds'] = base + r['cold_seconds']
        r['est_basis'] = (
            'state_seconds[%s] %.4g + own cold wall %.4g s (serial %.4g s; column and endpoint02 items / '
            'speed-up %.3g of the column workers)'
            % ('strato' if r['strato'] else 'off', base, r['cold_seconds'], r['cold_cpu_seconds'], column_speedup)
        )
    lists = [dict(gpu=k, ids=[], est_seconds=0.0) for k in range(gpus)]

    def place(r):
        g = min(lists, key=lambda x: (x['est_seconds'], x['gpu']))
        g['ids'].append(r['id'])
        g['est_seconds'] += r['est_seconds']

    by_id = {r['id']: r for r in rows}
    for r in sorted(reps, key=lambda r: -r['cold_seconds']):
        place(r)
    for r in sorted([r for r in rows if r['role'] == 'warm'], key=lambda r: -r['est_seconds']):
        place(r)
    # Members that wait on the least cold work (e.g. only the shared host items) go first.
    for r in sorted(
        [r for r in rows if r['role'] == 'member'],
        key=lambda r: (sum(weight(i) for i in r['missing']), -r['est_seconds'], rows.index(r)),
    ):
        place(r)
    for g in lists:
        g['first_wave'] = g['ids'][:workers]
        g['expected_waits'] = [
            dict(id=i, waits_for=by_id[i]['waits_for']) for i in g['first_wave'] if by_id[i]['role'] == 'member'
        ]
    return lists


def commands(a, lists, columns, attach_root, out, res):
    work = (a.work_root or out.parent / 'work').resolve()
    base = [
        *paths.module_command('pipeline'),
        '--states',
        str(a.states.resolve()),
        '--plan',
        str(out.resolve()),
        '--geometry-cache',
        str(a.geometry_cache.resolve()),
        '--persistent',
        '--geometry-layers',
        str(a.geometry_layers),
    ]
    if columns:
        base += ['--column-cache', str(Path(columns).resolve())]
    for key in ('settings_json', 'stratosphere', 'density_profile', 'ozone_profile', 'rayleigh_column'):
        if getattr(a, key) is not None:
            base += ['--' + key.replace('_', '-'), str(getattr(a, key))]
    # Both fixed: with --threads T the pipeline's own auto_workers would divide the whole host by T for every GPU.
    base += ['--workers', str(res['workers']), '--threads', str(res['column_workers'])]
    env = {}
    if Path(attach_root).resolve() != (a.geometry_cache / 'attach_cache').resolve():
        env['STRATO_ATTACH_CACHE'] = str(Path(attach_root).resolve())
    for g in lists:
        cmd = base + [
            '--plan-gpu',
            str(g['gpu']),
            '--work-root',
            str(work / (f'gpu{g["gpu"]}' if len(lists) > 1 else 'queue')),
        ]
        extra = dict(env)
        if len(lists) > 1:
            cmd += ['--gpu-lock', str(Path(tempfile.gettempdir()) / f'sosfuse2_gpu{g["gpu"]}.lock')]
            extra['CUDA_VISIBLE_DEVICES'] = str(g['gpu'])
        g['command'] = ' '.join([f'{k}={shlex.quote(v)}' for k, v in extra.items()] + [shlex.quote(x) for x in cmd])


def summary(plan):
    rows = {r['id']: r for r in plan['states']}
    items = plan['items']
    lines = []
    inp = plan['inputs']
    geo = plan['geometry']
    lines.append(
        f"QUEUE PLAN  {inp['states']}  states={len(rows)}  gpus={len(plan['gpus'])}  workers/gpu={plan['workers_per_gpu']}"
        f"  column workers={plan['resources']['column_workers']}"
        f"  cost profile={plan['cost_profile']['name']}  host={plan['host']['machine']}"
    )
    for g in geo:
        lines.append(
            f"geometry endpoints_{g['signature']}: "
            + (
                f"complete (main {g['rays']['main']:,} rays, ground {g['rays']['ground']:,})"
                if g['complete']
                else 'NOT built (cold geometry; column keys unknown)'
            )
        )
    fmt = lambda v: 'unmeasured' if v is None else f'{v:.1f} s'
    lines.append('COLUMN GROUPS (stratosphere columns, column_cache)')
    for grp in plan['groups']['column']:
        lines.append(
            f"  {grp['label']:<26} states={len(grp['states']):<3} "
            + '  '.join(
                f"{items[i]['bundle']} {str(items[i]['key'])[:16]} {'hit' if items[i]['hit'] else 'MISS'} {fmt(items[i]['cost_seconds'])}"
                for i in grp['items']
            )
            + f"  rep={grp['representative']}"
        )
    lines.append('ATTACH GROUPS (lock = attach_lock_name, file = _geometry_key; platform-dependent)')
    for grp in plan['groups']['attach']:
        lines.append(
            f"  {grp['lock']}  key {grp['key'][:16]}  {'hit' if grp['hit'] else 'MISS'}  lock_done={grp['lock_done']}"
            f"  states={len(grp['states'])}  rep={grp['representative']}"
        )
    other = [(i, it) for i, it in items.items() if it['kind'] not in ('column', 'attach')]
    if other:
        lines.append('HOST / AEROSOL / GEOMETRY ITEMS')
        for i, it in other:
            lines.append(
                f"  {it['kind']:<16} {str(it['key'])[:24]:<24} {'hit' if it['hit'] else 'MISS'}  {fmt(it['cost_seconds'])}"
                f"  states={len(it['states'])}  rep={it.get('owner')}"
            )
    for g in plan['gpus']:
        lines.append(
            f"GPU {g['gpu']} order (est {g['est_seconds']:.0f} s; first wave = first {plan['workers_per_gpu']}):"
        )
        for k, i in enumerate(g['ids']):
            r = rows[i]
            tag = (
                r['role']
                + (
                    f" cold {r['cold_seconds']:.1f} s wall ({r['cold_cpu_seconds']:.1f} s serial)"
                    if r['role'] == 'representative'
                    else ''
                )
                + (f" after {','.join(r['waits_for'])}" if r['waits_for'] else '')
            )
            lines.append(f"  {k + 1:>3}. {i:<36} {tag}")
        for w in g['expected_waits']:
            lines.append(
                f"  ! {w['id']} starts in the first wave and waits on {','.join(w['waits_for'])} (no warm state to run instead)"
            )
        lines.append('  ' + g['command'])
    cold = plan['cold_cpu_seconds']
    lines.append(
        f"cold CPU (missing items once each, serial, profile {plan['cost_profile']['name']}): {cold['total_known']:.1f} s known"
        + (f" + {len(cold['unmeasured_items'])} unmeasured item(s)" if cold['unmeasured_items'] else '')
        + '  by kind: '
        + ', '.join(f'{k} {v:.1f} s' for k, v in cold['by_kind'].items())
        + f"\n  with {plan['resources']['column_workers']} column worker(s) (speed-up {plan['resources']['column_speedup']:.2f}"
        f" on column/endpoint02): {cold['wall_known_with_column_workers']:.1f} s wall if one state did all of it"
        f"\n  workers: {plan['resources']['workers_basis']}\n  column workers: {plan['resources']['column_workers_basis']}"
    )
    return '\n'.join(lines) + '\n'


def plan_queue(a):
    import numpy as np
    from sosfuse2 import host_cache
    from sosfuse2 import run_gpu_state_v3 as rv

    if a.gpus < 1 or (a.workers_per_gpu is not None and a.workers_per_gpu < 1):
        raise ValueError('E_PLAN_ARGUMENTS: positive --gpus and --workers-per-gpu required')
    options = SimpleNamespace(
        reduced=False,
        settings_json=a.settings_json,
        stratosphere=a.stratosphere,
        density_profile=a.density_profile,
        ozone_profile=a.ozone_profile,
        rayleigh_column=a.rayleigh_column,
    )
    cfg = rv.settings_for(options)
    every = rv.read_states(a.states, True)
    runnable = [s['id'] for s in every if s.get('runnable')]
    ids = a.ids.split(',') if a.ids else runnable
    if len(set(ids)) != len(ids) or any(sum(s['id'] == i and bool(s.get('runnable')) for s in every) != 1 for i in ids):
        raise ValueError('E_PLAN_IDS: unique ids of exactly one runnable state each required')
    states = [rv.effective_state(next(s for s in every if s['id'] == i and s.get('runnable')), options) for i in ids]
    columns = a.column_cache or (a.cache_root / 'column_cache' if a.cache_root else a.geometry_cache / 'column_cache')
    attach_env = os.environ.get('STRATO_ATTACH_CACHE', '').strip()
    if attach_env.lower() == 'off':
        attach_env = ''  # 'off' switches the runner's attach cache off; it is not a folder name
    attach_root = a.attach_cache or (
        a.cache_root / 'attach_cache' if a.cache_root else Path(attach_env or a.geometry_cache / 'attach_cache')
    )
    if a.column_workers is not None and a.column_workers < 1:
        raise ValueError('E_PLAN_ARGUMENTS: positive --column-workers required')
    res = resources(a, len(ids))
    workers = res['workers']
    started = time.monotonic()
    rows, items, probes, geometry, costs = collect(a, cfg, states, columns, attach_root)
    lists = order(rows, items, costs, a.gpus, workers, res['column_speedup'])
    doc = json.loads(a.states.read_text())
    parent = doc.get('parent')
    groups = dict(column=[], attach=[])
    seen = {}
    for r in rows:
        col = tuple(i for i in r['items'] if items[i]['kind'] == 'column')
        if col:
            seen.setdefault(col, []).append(r['id'])
    for col, ids_ in seen.items():
        prof = items[col[0]]['profile']
        owners = [items[i]['owner'] for i in col if items[i]['owner']]
        groups['column'].append(
            dict(
                label='center %g sigma %g' % (prof['center'], prof['sigma']),
                items=list(col),
                states=ids_,
                hit=all(items[i]['hit'] for i in col),
                representative=owners[0] if owners else None,
            )
        )
    for i, it in items.items():
        if it['kind'] == 'attach':
            groups['attach'].append(
                dict(
                    lock=it['lock'],
                    key=it['key'],
                    hit=it['hit'],
                    lock_done=it['lock_done'],
                    states=it['states'],
                    representative=it['owner'],
                )
            )
    missing = {i: it for i, it in items.items() if not it['hit']}
    by_kind = {}
    wall = 0.0
    for it in missing.values():
        by_kind[it['kind']] = by_kind.get(it['kind'], 0.0) + (it['cost_seconds'] or 0.0)
        wall += (it['cost_seconds'] or 0.0) / (res['column_speedup'] if it['kind'] in PARALLEL_KINDS else 1.0)
    plan = dict(
        version=1,
        tool='sosfuse2.plan_queue',
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        planning_seconds=time.monotonic() - started,
        host=dict(platform=platform.platform(), machine=platform.machine(), numpy=np.__version__),
        inputs=dict(
            states=str(a.states.resolve()),
            states_sha256=sha256(a.states),
            parent=parent,
            parent_sha256=sha256(a.states.parent / parent) if parent else None,
            ids=ids,
            ids_explicit=bool(a.ids),
            reduced=False,
            settings_json=str(a.settings_json) if a.settings_json else None,
            settings_sha256=sha256(a.settings_json) if a.settings_json else None,
            geometry_cache=str(a.geometry_cache.resolve()),
            geometry_layers=a.geometry_layers,
            column_cache=str(Path(columns).resolve()),
            attach_cache=str(Path(attach_root).resolve()),
            host_cache_enabled=host_cache.enabled(),
            options={k: getattr(a, k) for k in ('stratosphere', 'density_profile', 'ozone_profile', 'rayleigh_column')},
        ),
        workers_per_gpu=workers,
        resources=res,
        geometry=[
            dict(
                signature=sig,
                folder=str(g['folder']),
                complete=g['complete'],
                rays={k: int(b.rays) for k, b in g['bundles'].items()},
            )
            for sig, g in geometry.items()
        ],
        probes=[dict(index=k, seconds=p.seconds, host_records=p.records) for k, p in enumerate(probes.values())],
        cost_profile=dict(
            name=a.cost_profile, values={k: v for k, v in costs.items() if k != 'sources'}, sources=costs['sources']
        ),
        items=items,
        groups=groups,
        states=[
            {
                k: r[k]
                for k in (
                    'id',
                    'role',
                    'strato',
                    'attach_lock',
                    'scale_height_km',
                    'density',
                    'items',
                    'missing',
                    'waits_for',
                    'cold_cpu_seconds',
                    'cold_seconds',
                    'est_seconds',
                    'est_basis',
                    'probe',
                )
            }
            for r in rows
        ],
        gpus=lists,
        cold_cpu_seconds=dict(
            total_known=sum(by_kind.values()),
            by_kind=by_kind,
            wall_known_with_column_workers=wall,
            unmeasured_items=[i for i, it in missing.items() if it['cost_seconds'] is None],
        ),
        notes=[
            'attach and sun02 hits are valid only on this host (floating-point keys); run the planner on the queue host',
            'table bundles (about 0.04 % of the stratosphere rays per state) are not keyed',
            'cold_cpu_seconds are serial; cold_seconds divide column and endpoint02 items by the measured speed-up '
            'of the column workers (SPEEDUP, Mac column-parallel measurement of 2026-10-07)',
        ],
    )
    commands(a, lists, columns, attach_root, a.out, res)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(plan, indent=1, allow_nan=False) + '\n')
    os.replace(tmp, a.out)
    text = summary(plan)
    a.out.with_suffix(a.out.suffix + '.txt').write_text(text)
    print(text, end='', flush=True)
    return plan


def main(argv=None):
    a = parser().parse_args(argv)
    plan_queue(a)
    return 0


if __name__ == '__main__':
    sys.exit(main())
