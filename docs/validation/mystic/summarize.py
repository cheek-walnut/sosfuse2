"""Summarize luminance and chromaticity comparisons."""

import argparse
import csv
import json
import statistics as st
from collections import defaultdict

from common import DEFAULT_CONFIG, add_paths, read


def main():
    parser = argparse.ArgumentParser()
    add_paths(parser)
    args = parser.parse_args()
    # The summary is defined for the eight states, six pixels and two SZA of config.json.
    if read(args.config) != read(DEFAULT_CONFIG):
        raise ValueError('E_PUBLIC_CONFIG: summarize.py needs the full configuration')
    out = args.work.resolve() / 'out'
    rows = [r for r in csv.DictReader((out / 'compare_long.csv').open()) if r['wavelength_nm'] == '']

    def parts(case):
        sid, i, j, sza = case.rsplit('_', 3)
        return sid, f'{i}_{j}', int(sza[3:])

    PIX = {
        'i98_j0': '1.35/0.5',
        'i98_j89': '1.35/89.5',
        'i88_j179': '10.35/179.5',
        'i77_j0': '20.25/0.5',
        'i66_j90': '30.15/90.5',
        'i32_j179': '60.75/179.5',
    }
    table = defaultdict(dict)
    res = {}
    for r in rows:
        sid, pix, sza = parts(r['case_id'])
        table[(r['engine_config_id'], r['metric'], sza, pix)][sid] = (
            float(r['mean']),
            float(r['se']),
            float(r['ci_low']),
            float(r['ci_high']),
            r['status'],
            int(r['n']),
        )
    print(
        'status counts',
        {s: sum(1 for r in rows if r['status'] == s) for s in {r['status'] for r in rows}},
        'n',
        {r['n'] for r in rows},
    )
    for arm in (x for x in ('A0', 'A1', 'A2') if x in {r['engine_config_id'] for r in rows}):
        print(
            f'\n== {arm}: MYSTIC/engine Y - 1 (%) over 8 states: mean [min..max], '
            'rows whose 95% CI excludes 1, median SE'
        )
        for sza in (93, 96):
            for pix, label in PIX.items():
                d = table[(arm, 'Y_ratio', sza, pix)]
                v = [100 * (x[0] - 1) for x in d.values()]
                out_ci = sum(1 for x in d.values() if not (x[2] <= 1 <= x[3]))
                se = st.median(100 * x[1] for x in d.values())
                res[f'{arm}|Y|{sza}|{label}'] = dict(
                    mean=st.mean(v),
                    min=min(v),
                    max=max(v),
                    ci_excludes=out_ci,
                    median_se=se,
                    per_state={k: round(100 * (x[0] - 1), 3) for k, x in d.items()},
                )
                print(
                    f'  SZA {sza} {label:12s} {st.mean(v):+.3f} [{min(v):+.3f}..{max(v):+.3f}]'
                    f'  CI!=1: {out_ci}/8  SE {se:.3f}'
                )
        for m, key in (('delta_u_prime', 'du'), ('delta_v_prime', 'dv')):
            for sza in (93, 96):
                vals = [x[0] for pix in PIX for x in table[(arm, m, sza, pix)].values()]
                ses = [x[1] for pix in PIX for x in table[(arm, m, sza, pix)].values()]
                res[f'{arm}|{key}|{sza}'] = dict(
                    mean=st.mean(vals), min=min(vals), max=max(vals), median_se=st.median(ses)
                )
                print(
                    f'  {m} SZA {sza}: mean {st.mean(vals):+.5f} [{min(vals):+.5f}..{max(vals):+.5f}]'
                    f' median SE {st.median(ses):.5f}'
                )
        allv = [
            100 * (x[0] - 1) for (a, m, s, p), d in table.items() if a == arm and m == 'Y_ratio' for x in d.values()
        ]
        res[f'{arm}|Y|all'] = dict(
            mean=st.mean(allv),
            rms=(sum(x * x for x in allv) / len(allv)) ** 0.5,
            maxabs=max(abs(x) for x in allv),
            n=len(allv),
        )
        print(
            f'  all 96 rows: mean {st.mean(allv):+.3f} rms {res[f"{arm}|Y|all"]["rms"]:.3f}'
            f' max|.| {max(abs(x) for x in allv):.3f}'
        )
    er = [
        r
        for r in csv.DictReader((out / 'engine_ratios.csv').open())
        if r['wavelength_nm'] == '' and r['metric'] == 'Y_ratio'
    ]
    for pair in sorted({r['engine_config_id'] for r in er}):
        v = [100 * (float(r['value']) - 1) for r in er if r['engine_config_id'] == pair]
        res[f'{pair}|Y'] = dict(mean=st.mean(v), min=min(v), max=max(v))
        print(f'\nengine/engine {pair} Y - 1 (%): mean {st.mean(v):+.4f} [{min(v):+.4f}..{max(v):+.4f}]')
    (out / 'summary.json').write_text(json.dumps(res, indent=2) + '\n')


if __name__ == "__main__":
    main()
