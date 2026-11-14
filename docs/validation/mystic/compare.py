"""Join independent MC and engine outputs by optical SHA and report seed-level intervals."""

import argparse
import csv
import json
import math

import numpy as np

from common import add_paths, dump, sha, work_path
from run_mystic import FIELDS

# Two-sided 95% Student critical values, df 1..15, rounded to 9 decimals.
# NIST/SEMATECH e-Handbook 1.3.6.7.2 (t distribution table); at most 16 configured seeds.
T975 = (
    None,
    12.706204736,
    4.302652730,
    3.182446305,
    2.776445105,
    2.570581836,
    2.446911851,
    2.364624252,
    2.306004135,
    2.262157163,
    2.228138852,
    2.200985160,
    2.178812830,
    2.160368656,
    2.144786688,
    2.131449546,
)
LONG_FIELDS = (
    'case_id engine_config_id metric wavelength_nm mean se n ci_low ci_high t975 status discarded_seeds'.split()
)


def color(spec, w):
    if not np.isfinite(spec).all() or np.any(np.asarray(spec) < 0):
        raise ValueError('E_COLOR')
    xyz = np.asarray(spec) @ np.asarray(w)[:, :3]
    den = xyz[..., 0] + 15 * xyz[..., 1] + 3 * xyz[..., 2]
    if np.any(den <= 0) or np.any(xyz[..., 1] <= 0):
        raise ValueError('E_COLOR')
    return np.stack((4 * xyz[..., 0] / den, 9 * xyz[..., 1] / den, xyz[..., 1]), axis=-1)


def estimate(values):
    n = len(values)
    status = 'OK' if n >= 8 else 'UNRESOLVED'
    if n == 0:
        return dict(mean=None, se=None, n=0, ci_low=None, ci_high=None, t975=None, status=status, per_seed=[])
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(n)) if n > 1 else None
    if n > 16:
        raise ValueError('E_T_TABLE_RANGE')
    t = T975[n - 1] if n > 1 else None
    return dict(
        mean=mean,
        se=se,
        n=n,
        ci_low=mean - t * se if t else None,
        ci_high=mean + t * se if t else None,
        t975=t,
        status=status,
        per_seed=np.asarray(values).tolist(),
    )


def seed_spectra(rows, c, wl):
    spectra = []
    seeds = []
    discarded = []
    unexpected = {int(r['seed']) for r in rows} - set(c['seed_ids'])
    if unexpected:
        raise ValueError('E_UNEXPECTED_SEED')
    for seed in c['seed_ids']:
        selected = [r for r in rows if int(r['seed']) == seed]
        reason = None
        if len(selected) != len(wl):
            reason = 'E_SPECTRUM_COUNT'
        elif any(r['status'] != 'ok' for r in selected):
            reason = 'E_SPECTRUM_STATUS'
        else:
            try:
                selected = sorted(selected, key=lambda r: float(r['wavelength_nm']))
                if not np.allclose([float(r['wavelength_nm']) for r in selected], wl, rtol=1e-10, atol=0):
                    reason = 'E_SPECTRUM_WAVELENGTH'
                else:
                    spec = np.asarray([float(r['radiance']) for r in selected])
                    if not np.isfinite(spec).all() or np.any(spec <= 0):
                        reason = 'E_SPECTRUM_RADIANCE'
            except (KeyError, TypeError, ValueError):
                reason = 'E_SPECTRUM_VALUE'
        if reason:
            discarded.append(dict(seed=seed, reason=reason))
        else:
            spectra.append(spec)
            seeds.append(seed)
    return np.asarray(spectra).reshape(len(seeds), len(wl)), seeds, discarded


def write_csv(path, fields, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    add_paths(ap, arms=True)
    ap.add_argument('--arms', default='A0')
    ap.add_argument('--allow-partial', action='store_true')
    a = ap.parse_args()
    H = work_path(a.work)
    data = json.loads((H / 'out/optics.json').read_text())
    manifest = json.loads((H / 'out/manifest.json').read_text())
    c = json.loads(a.config.read_text())
    expected_arms = json.loads(a.arms_file.read_text())
    if manifest.get('complete') is not True and not a.allow_partial:
        raise ValueError('E_MC_INCOMPLETE')
    digest = sha(H / 'out/optics.json')
    if manifest['optics_sha256'] != digest:
        raise ValueError('E_OPTICS_SHA')
    if manifest['results_sha256'] != sha(H / 'out/full.csv') or manifest['config_sha256'] != sha(a.config):
        raise ValueError('E_RESULT_SHA')
    with (H / 'out/full.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    requested = a.arms.split(',')
    engines = {}
    engine_sha = {}
    for arm in requested:
        if arm not in ('A0', 'A1', 'A2') or arm not in expected_arms:
            raise ValueError('E_UNKNOWN_ARM')
        path = H / 'out' / f'engine_pixels.{arm}.json'
        if not path.exists():
            continue
        e = json.loads(path.read_text())
        if e['optics_sha256'] != digest:
            raise ValueError('E_OPTICS_SHA ' + arm)
        if e['engine_config_id'] != arm:
            raise ValueError('E_ENGINE_ARM')
        expected_cases = {
            f'{sid}_i{p["i"]}_j{p["j"]}_sza{sza:g}'
            for sid in c['state_ids']
            for p in c['pixels']
            for sza in c['sza_deg']
        }
        if set(e['cases']) != expected_cases:
            raise ValueError('E_ENGINE_CASE_SET')
        engines[arm] = e
        engine_sha[arm] = sha(path)
    results = []
    long = []
    engine_rows = []
    ratios = []
    ratio_long = []
    for arm, eng in engines.items():
        for case, e in eng['cases'].items():
            s = data['states'][e['state_id']]
            wl = np.asarray(s['wavelengths_nm'])
            gpu = np.asarray(e['radiance'])
            w = s['weights']
            if (
                not np.allclose(e['wavelength_nm'], wl, rtol=1e-6, atol=0)
                or gpu.shape != wl.shape
                or not np.isfinite(gpu).all()
                or np.any(gpu <= 0)
            ):
                raise ValueError('E_ENGINE_SPECTRUM')
            gc = color(gpu, w)
            spectra, seeds, discarded = seed_spectra([r for r in rows if r['case_id'] == case], c, wl)
            colors = color(spectra, w) if len(seeds) else np.empty((0, 3))
            metrics = {
                key: estimate(values)
                for key, values in [
                    ('Y_ratio', colors[:, 2] / gc[2]),
                    ('delta_u_prime', colors[:, 0] - gc[0]),
                    ('delta_v_prime', colors[:, 1] - gc[1]),
                ]
            }
            spectral = [estimate(spectra[:, i] / gpu[i]) for i in range(len(wl))]
            results.append(
                dict(
                    case_id=case,
                    engine_config_id=arm,
                    status='OK' if len(seeds) >= 8 else 'UNRESOLVED',
                    n=len(seeds),
                    seeds=seeds,
                    discarded=discarded,
                    metrics=metrics,
                    wavelength_nm=wl.tolist(),
                    spectral_ratio=spectral,
                )
            )
            for key, metric in list(metrics.items()) + [(f'spectral_ratio:{i}', v) for i, v in enumerate(spectral)]:
                long.append(
                    dict(
                        case_id=case,
                        engine_config_id=arm,
                        metric='spectral_ratio' if key.startswith('spectral_ratio:') else key,
                        wavelength_nm=wl[int(key.split(':')[1])] if ':' in key else '',
                        discarded_seeds=len(discarded),
                        **{k: v for k, v in metric.items() if k != 'per_seed'},
                    )
                )
            for wi, value in enumerate(gpu):
                engine_rows.append(
                    dict(
                        case_id=case,
                        geometry_id=f'i{e["i"]}_j{e["j"]}_sza{e["sza_deg"]:g}',
                        spectrum_id=case,
                        run_id=case + '_' + arm,
                        engine_config_id=arm,
                        code='sosfuse2',
                        physics_variant=c['physics_variant'],
                        mode='point',
                        state_id=e['state_id'],
                        sza_deg=e['sza_deg'],
                        view_elev_deg=e['elevation_deg'],
                        view_azim_deg=e['azimuth_deg'],
                        pixel_i=e['i'],
                        pixel_j=e['j'],
                        wavelength_nm=wl[wi],
                        radiance=value,
                        radiance_unit='W m-2 nm-1 sr-1',
                        status='ok',
                    )
                )
    if 'A0' in engines:
        for arm in ('A1', 'A2'):
            if arm not in engines:
                continue
            for case, base in engines['A0']['cases'].items():
                if case not in engines[arm]['cases']:
                    continue
                alt = engines[arm]['cases'][case]
                if alt['wavelength_nm'] != base['wavelength_nm']:
                    raise ValueError('E_ENGINE_RATIO_AXIS')
                br = np.asarray(base['radiance'])
                ar = np.asarray(alt['radiance'])
                w = data['states'][base['state_id']]['weights']
                bc = color(br, w)
                ac = color(ar, w)
                entry = dict(
                    case_id=case,
                    engine_config_id=arm + '/A0',
                    Y_ratio=float(ac[2] / bc[2]),
                    delta_u_prime=float(ac[0] - bc[0]),
                    delta_v_prime=float(ac[1] - bc[1]),
                    wavelength_nm=base['wavelength_nm'],
                    spectral_ratio=(ar / br).tolist(),
                )
                ratios.append(entry)
                for key in ('Y_ratio', 'delta_u_prime', 'delta_v_prime'):
                    ratio_long.append(
                        dict(case_id=case, engine_config_id=arm + '/A0', metric=key, wavelength_nm='', value=entry[key])
                    )
                for wl, v in zip(base['wavelength_nm'], entry['spectral_ratio'], strict=True):
                    ratio_long.append(
                        dict(
                            case_id=case,
                            engine_config_id=arm + '/A0',
                            metric='spectral_ratio',
                            wavelength_nm=wl,
                            value=v,
                        )
                    )
    write_csv(H / 'out/engine.csv', FIELDS, engine_rows)
    write_csv(H / 'out/compare_long.csv', LONG_FIELDS, long)
    write_csv(
        H / 'out/engine_ratios.csv', ['case_id', 'engine_config_id', 'metric', 'wavelength_nm', 'value'], ratio_long
    )
    method = (
        'Mean of seed-level MC/engine Y and spectral ratios, and seed-level chromaticity differences; '
        'two-sided approximate 95% t interval, sample SD/sqrt(n); MC sampling error only. '
        'Independent batch normal approximation; no accuracy threshold.'
    )
    dump(
        H / 'out/compare.json',
        dict(
            results=results,
            engine_ratios=ratios,
            estimator=method,
            missing_arms=[x for x in requested if x not in engines],
            config_file=a.config.name,
            config_sha256=sha(a.config),
            status='PARTIAL'
            if manifest.get('complete') is not True
            else 'WAITING_ENGINE'
            if not engines
            else 'PARTIAL'
            if any(x not in engines for x in requested)
            else 'JOINED',
        ),
    )
    dump(
        H / 'out/manifest_compare.json',
        dict(
            schema_version='v1-join-1',
            mc_manifest_sha256=sha(H / 'out/manifest.json'),
            optics_sha256=digest,
            mystic_csv_sha256=sha(H / 'out/full.csv'),
            engine_pixels_sha256=engine_sha,
            outputs_sha256={
                name: sha(H / 'out' / name)
                for name in ('compare.json', 'compare_long.csv', 'engine.csv', 'engine_ratios.csv')
            },
            estimator=method,
        ),
    )
    print('COMPARE', len(results), 'rows', len(engines), 'configurations')


if __name__ == '__main__':
    main()
