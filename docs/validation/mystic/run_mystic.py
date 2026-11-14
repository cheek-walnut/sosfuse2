"""Bounded MYSTIC scheduler; durable per-task results and independent MC manifests."""

import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import math
import resource
import subprocess
import time
from pathlib import Path

from common import HERE, add_paths, dump, sha, work_path
from inputs import text_input

FIELDS = (
    'case_id geometry_id spectrum_id run_id engine_config_id code physics_variant mode state_id sza_deg '
    'view_elev_deg view_azim_deg pixel_i pixel_j wavelength_nm seed rng_seed photons radiance radiance_unit '
    'se cpu_s status error'
).split()
H = None


def init_worker(data, root=None):
    global DATA, H
    DATA = data
    if root is not None:
        H = Path(root)


def shard_arg(value):
    try:
        k, n = map(int, value.split('/'))
        if not 1 <= k <= n:
            raise ValueError()
        return k, n
    except ValueError:
        raise argparse.ArgumentTypeError('E_SHARD: require 1 <= K <= N') from None


def build_jobs(c, data, identity, lrt, smoke=False, shard=(1, 1)):
    jobs = []
    sm = c['smoke']
    row = 0
    for si, sid in enumerate(c['state_ids']):
        for pi, p in enumerate(c['pixels']):
            for zi, sza in enumerate(c['sza_deg']):
                selected = row % shard[1] == shard[0] - 1
                row += 1
                if not selected or (smoke and (si, pi, zi) != (sm['state_index'], sm['pixel_index'], sm['sza_index'])):
                    continue
                case = f'{sid}_i{p["i"]}_j{p["j"]}_sza{sza:g}'
                if (
                    abs(90 - (p['i'] + 0.5) * 0.9 - p['elevation_deg']) > 1e-6
                    or abs(p['j'] + 0.5 - p['azimuth_deg']) > 1e-6
                ):
                    raise ValueError('E_PIXEL_AXIS ' + case)
                for wi in sm['wavelength_indices'] if smoke else range(len(data['states'][sid]['wavelengths_nm'])):
                    for seed_id in [sm['seed_id']] if smoke else c['seed_ids']:
                        # Mixed radix: unique state/pixel/SZA/wavelength streams; reserve +1000 for retry.
                        seed = seed_id + 10000 * (
                            wi + len(c['wavelength_nm']) * (zi + len(c['sza_deg']) * (pi + len(c['pixels']) * si))
                        )
                        if not 0 < seed < 2147482647:
                            raise ValueError('E_RNG_RANGE')
                        jobs.append(
                            dict(
                                group='smoke' if smoke else 'full',
                                identity=identity,
                                case=case,
                                sid=sid,
                                pixel=p,
                                sza=sza,
                                wi=wi,
                                seed_id=seed_id,
                                seed=seed,
                                photons=sm['photons'] if smoke else c['photons'],
                                lrt=str(lrt),
                            )
                        )
    return jobs


def folder_for(job):
    return H / 'runs' / job['group'] / job['case'] / f'w{job["wi"]:02d}' / f's{job["seed_id"]}'


def read_result(job):
    path = folder_for(job) / 'result.json'
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result['identity'] != job['identity']:
        raise ValueError('E_RESUME_SHA ' + str(path))
    return result


def one(job):
    folder = folder_for(job)
    folder.mkdir(parents=True, exist_ok=True)
    old = read_result(job)
    if old is not None:
        return old
    reserve = folder / 'reservation.json'
    if reserve.exists() and json.loads(reserve.read_text())['identity'] != job['identity']:
        raise ValueError('E_RESUME_SHA ' + str(folder))
    dump(reserve, job)
    attempts = []
    for index in range(2):
        dest = folder / f'attempt_{index}'
        dest.mkdir(exist_ok=True)
        saved = dest / 'attempt.json'
        if saved.exists():
            result = json.loads(saved.read_text())
            if result['identity'] != job['identity']:
                raise ValueError('E_RESUME_SHA ' + str(saved))
        else:
            rng = job['seed'] + 1000 * index
            text = text_input(
                DATA, job['sid'], job['pixel'], job['sza'], job['wi'], job['photons'], rng, Path(job['lrt']), work=H
            )
            (dest / 'uvspec.inp').write_text(text)
            for name in ('mc.rad.spc', 'mc.rad.std.spc'):
                (dest / name).unlink(missing_ok=True)
            before = resource.getrusage(resource.RUSAGE_CHILDREN)
            t = time.monotonic()
            rc = None
            try:
                with (
                    (dest / 'uvspec.inp').open() as fi,
                    (dest / 'stdout.txt').open('w') as fo,
                    (dest / 'stderr.txt').open('w') as fe,
                ):
                    rc = subprocess.run(
                        [str(Path(job['lrt']) / 'bin/uvspec')], stdin=fi, stdout=fo, stderr=fe, cwd=dest
                    ).returncode
                if rc:
                    raise ValueError('E_UVSPEC_EXIT_' + str(rc))

                def first(path):
                    rows = [
                        x.split() for x in path.read_text().splitlines() if x.strip() and not x.lstrip().startswith('#')
                    ]
                    return float(rows[0][4])

                intensity = first(dest / 'mc.rad.spc')
                std = first(dest / 'mc.rad.std.spc')
                if not math.isfinite(intensity) or not math.isfinite(std) or intensity <= 0 or std < 0:
                    raise ValueError('E_INVALID_OUTPUT')
                status = 'ok'
                error = None
            except (OSError, ValueError, IndexError) as e:
                intensity = std = None
                status = 'failed'
                error = str(e)
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            result = {k: job[k] for k in ('identity', 'case', 'sid', 'sza', 'pixel', 'wi', 'seed_id', 'photons')}
            result.update(
                seed=rng,
                status=status,
                error=error,
                rc=rc,
                cpu_s=after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
                wall_s=time.monotonic() - t,
                transmittance=intensity,
                mc_std=std,
                input_sha256=hashlib.sha256(text.encode()).hexdigest(),
            )
            dump(saved, result)
        attempts.append(
            dict(
                seed=result['seed'],
                status=result['status'],
                error=result['error'],
                rc=result['rc'],
                cpu_s=result['cpu_s'],
            )
        )
        if result['status'] == 'ok':
            break
    result['attempts'] = attempts
    result['cpu_s'] = sum(a['cpu_s'] for a in attempts)
    dump(folder / 'result.json', result)
    return result


def csv_row(job, result, data, c):
    p = job['pixel']
    s = data['states'][job['sid']]
    wi = job['wi']
    solar = s['solar'][wi]
    r = result or dict(seed=job['seed'], transmittance=None, mc_std=None, cpu_s=0, status='not_run', error='E_NOT_RUN')
    return dict(
        case_id=job['case'],
        geometry_id=f'i{p["i"]}_j{p["j"]}_sza{job["sza"]:g}',
        spectrum_id=job['case'],
        run_id=f'{job["case"]}_w{wi:02d}_s{job["seed_id"]}',
        engine_config_id='',
        code='MYSTIC',
        physics_variant=c['physics_variant'],
        mode='point',
        state_id=job['sid'],
        sza_deg=job['sza'],
        view_elev_deg=p['elevation_deg'],
        view_azim_deg=p['azimuth_deg'],
        pixel_i=p['i'],
        pixel_j=p['j'],
        wavelength_nm=s['wavelengths_nm'][wi],
        seed=job['seed_id'],
        rng_seed=r['seed'],
        photons=job['photons'],
        radiance=r['transmittance'] * solar if r['transmittance'] is not None else '',
        radiance_unit='W m-2 nm-1 sr-1',
        se=r['mc_std'] * solar if r['mc_std'] is not None else '',
        cpu_s=r['cpu_s'],
        status=r['status'],
        error=r['error'],
    )


def collect(plan, data, c):
    with (H / 'out/collect.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _collect(plan, data, c)


def _collect(plan, data, c):
    group = plan['group']
    csvpath = H / 'out' / f'{group}.csv'
    tmp = csvpath.with_suffix('.csv.tmp')
    counts = dict(planned=len(plan['jobs']), success=0, failed=0, not_run=0)
    cpu = 0
    retry_reasons = []
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for job in plan['jobs']:
            r = read_result(job)
            if r is None:
                counts['not_run'] += 1
            else:
                counts['success' if r['status'] == 'ok' else 'failed'] += 1
                cpu += r['cpu_s']
                for a in r.get('attempts', []):
                    if a['status'] != 'ok':
                        retry_reasons.append(dict(case_id=job['case'], wi=job['wi'], seed_id=job['seed_id'], **a))
            writer.writerow(csv_row(job, r, data, c))
    tmp.replace(csvpath)
    manifest = dict(
        plan['metadata'],
        schema_version='v1-mc-1',
        shard=plan['shard'],
        group=group,
        counts=counts,
        cpu_s=cpu,
        retry_reasons=retry_reasons,
        results_sha256=sha(csvpath),
        complete=counts['failed'] == counts['not_run'] == 0,
    )
    dump(H / 'out' / ('manifest.json' if group == 'full' else 'manifest_smoke.json'), manifest)
    return manifest


def schedule(jobs, data, workers, deadline):
    """At most workers outstanding; the deadline never cancels a running task."""
    iterator = iter(jobs)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, initializer=init_worker, initargs=(data, str(H))
    ) as pool:
        pending = set()
        while True:
            while len(pending) < workers and (deadline is None or time.time() < deadline):
                job = next(iterator, None)
                if job is None:
                    break
                if read_result(job) is None:
                    pending.add(pool.submit(one, job))
            if not pending:
                break
            done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                future.result()


def main():
    global H
    ap = argparse.ArgumentParser()
    add_paths(ap)
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--lrt-dir', type=Path)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--shard', type=shard_arg, default=(1, 1))
    ap.add_argument('--deadline-epoch', type=float)
    ap.add_argument(
        '--write-partial', action='store_true', help='collect durable results using saved plan, without starting work'
    )
    a = ap.parse_args()
    H = work_path(a.work)
    c = json.loads(a.config.read_text())
    data = json.loads((H / 'out/optics.json').read_text())
    group = 'smoke' if a.smoke else 'full'
    pp = H / 'out' / f'plan_{group}.json'
    if a.write_partial:
        plan = json.loads(pp.read_text())
        if plan['metadata']['optics_sha256'] != sha(H / 'out/optics.json') or plan['metadata']['config_sha256'] != sha(
            a.config
        ):
            raise ValueError('E_PLAN_SHA')
        manifest = collect(plan, data, c)
    else:
        if a.jobs < 1 or a.lrt_dir is None:
            raise ValueError('E_ARGUMENT: jobs>=1 and --lrt-dir required')
        if sha(a.config) != data['identity']['config']:
            raise ValueError('E_CONFIG_SHA')
        for name in ('tau_ledger.json', 'disort.json'):
            if not json.loads((H / 'out' / name).read_text())['passed']:
                raise ValueError('E_INPUT_GATE ' + name)
        for path, digest in json.loads((H / 'out/input_sha.json').read_text()).items():
            if sha(H / path) != digest:
                raise ValueError('E_INPUT_SHA ' + path)
        lrt = a.lrt_dir.resolve()
        dh = hashlib.sha256()
        for f in sorted((lrt / 'data').rglob('*')):
            if f.is_file():
                dh.update(str(f.relative_to(lrt / 'data')).encode() + bytes.fromhex(sha(f)))
        metadata = dict(
            config_sha256=sha(a.config),
            optics_sha256=sha(H / 'out/optics.json'),
            input_sha256=sha(H / 'out/input_sha.json'),
            uvspec_sha256=sha(lrt / 'bin/uvspec'),
            libradtran_data_sha256=dh.hexdigest(),
            code_sha256={p.name: sha(p) for p in (HERE / 'run_mystic.py', HERE / 'inputs.py')},
            photons=c['photons'],
            assets=data['identity'],
            levels_km=data['levels_km'],
            earth_radius_km=6371,
            observer_km=0.1,
            polarization='polarized Rayleigh, depolarizing aerosols',
            refraction=False,
            solar_disk='point',
            solar_azimuth_deg=0,
            physics_variant=c['physics_variant'],
            mc_std='MYSTIC internal estimate, separate from seed scatter',
        )
        identity = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        plan = dict(
            group=group,
            shard=list(a.shard),
            metadata=metadata,
            jobs=build_jobs(c, data, identity, lrt, a.smoke, a.shard),
        )
        if pp.exists():
            old = json.loads(pp.read_text())
            if old['metadata'] != metadata or old['shard'] != list(a.shard):
                raise ValueError('E_PLAN_RESUME_SHA')
        dump(pp, plan)
        collect(plan, data, c)
        try:
            schedule(plan['jobs'], data, a.jobs, a.deadline_epoch)
        finally:
            manifest = collect(plan, data, c)
    print('MYSTIC', group, json.dumps(manifest['counts']), flush=True)
    if not a.write_partial and manifest['counts']['failed']:
        raise SystemExit(2)
    if not a.write_partial and manifest['counts']['not_run']:
        raise SystemExit(3)


if __name__ == '__main__':
    main()
