"""Two-stage numerical diagnostics; run with python -m sosfuse2.gate_flags.

Since gate v2 the blocking gate is gate.py; E1 (bootstrap <=1e-6)
and E2 (15-lane observer <=1e-5) here are diagnostic records. CUDA must be invoked through the site's guarded GPU queue.
Exit 1 preserves a failed check, exit 2 an exception; gate.json is written either way and contains every result. --display-atol is a radiance-unit
bound derived from the caller's display transform, never an arbitrary 1/255.
Without it, stage 2 uses the stricter atol=0 sufficient condition.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from . import reference
from . import paths
from .fused import Solver, STAGE1_FLAGS
from .geometry import Bundle, TransientGeometry, create, normal_state
from .stratosphere import source_size
from .validate import moment_comparison, reduced, relative


def controls(name, approximate=False):
    flags = json.loads((paths.CONFIGS / f'{name}.json').read_text())['kernel_flags']
    os.environ['SOSFUSE2_KERNEL_FLAGS'] = ','.join(flags)
    for flag in STAGE1_FLAGS:
        os.environ['SOSFUSE2_' + flag] = str(int(approximate and flag in ('FMAD', 'FASTMATH')))
    os.environ['SOSFUSE2_STRATO_ASYNC'] = '0'
    return flags


def evaluate(out, name, backend, approximate=False, check=False):
    controls(name, approximate)
    state = copy.deepcopy(normal_state())
    if name == 'C3':
        state['stratosphere']['status'] = 'enabled'
        state['parameters']['stratosphere_tau550'] = 0.003
    wl = (360 + (np.arange(15) + 0.5) * 470 / 15).tolist()
    s, scope, _ = create(out, 'selftest', wl, backend, 256, state)
    geometry = TransientGeometry(s, scope)
    solver = Solver(s, geometry)
    kernel = s.kernel
    field = kernel.xp.zeros((s.geo.ns, source_size(s), 15), 'f4')
    for _ in range(2):
        field = solver.apply(field)
    host = kernel.host(field)
    full = np.zeros((*host.shape[:2], 30), 'f4')
    full[..., ::2] = host
    # Fixed final observer radiance, including both direct and multiple light.
    el = np.array([2.0, 10.0, 30.0, 89.0])
    az = np.array([0.0, 0.0, 90.0, 180.0])
    radiance = np.stack(
        [(s.query(h, el, az, full, True) + s.query(h, el, az, full, False))[..., ::2] for h in (-6.0, -9.0, -12.0)]
    )
    np.savez(out / 'values.npz', source=host, radiance=radiance)
    rows = []
    if check:
        g = geometry.cell(12)
        c = kernel.setup(s, g['nq'])
        bundle = Bundle([g], kernel.xp)
        for first in (True, False):
            ref, Y = reference.transport(g, c, host, first, return_basis=True)
            moment, flux = reference.project(g, ref, c, Y)
            _, _, query = kernel.launch(bundle, s, field, first, phase=2)
            gm, gf, _ = kernel.launch(bundle, s, field, first, phase=0)
            m = kernel.host(gm)[0]
            ray = relative(kernel.host(query), ref)
            ferr = relative(kernel.host(gf)[0], flux)
            merr = relative(m, moment)
            projected, _ = reference.project(g, kernel.host(query), c, Y)
            # Keep the old diagnostic too; never silently relax its criterion.
            rows.append(
                dict(
                    first=first,
                    ray_max_relative=ray,
                    ground_max_relative=ferr,
                    moment_raw_relative=merr,
                    projection_same_rays=moment_comparison(m, projected),
                    **moment_comparison(m, moment),
                    pass_=bool(np.isfinite([ray, ferr, merr]).all() and max(ray, ferr, merr) <= 1e-6),
                )
            )
    return radiance, dict(
        flags=list(kernel.flags),
        stage1={f: os.environ['SOSFUSE2_' + f] for f in STAGE1_FLAGS},
        bootstrap=rows,
        bootstrap_pass=all(r['pass_'] for r in rows),
        scope='40 cells, 15 wavelengths, two fixed applies; bootstrap cell 12',
        launches=kernel.metrics['launches'],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--display-atol', type=float)
    args = p.parse_args()
    if args.display_atol is not None and (not np.isfinite(args.display_atol) or args.display_atol < 0):
        p.error('--display-atol must be finite and nonnegative')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = dict(
        schema=1,
        exact_tolerance=1e-6,
        selftest_tolerance=1e-5,
        code_sha256={
            n: hashlib.sha256((paths.PKG / n).read_bytes()).hexdigest()
            for n in ('kernel.cu', 'fused.py', 'ksplit.py', 'reference.py', 'validate.py')
        },
        exact={},
        approximate={},
    )
    try:
        return finish(out, report, args)
    except Exception as exc:  # noqa: BLE001 - the reason is kept in gate.json (gate v2 rule)
        report.update(code='E_GATE_EXCEPTION', error=f'{type(exc).__name__}: {exc}', **{'pass': False})
        (out / 'gate.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
        print(json.dumps(dict(code=report['code'], error=report['error'])), flush=True)
        return 2


def finish(out, report, args):
    # The historical fixture uses disabled stratosphere and one wavelength.
    controls('C3_nostrato')
    fixture = out / 'fixture'
    fixture.mkdir(exist_ok=True)
    report['historical_selftest'], _ = reduced(fixture, 'cupy', 256)
    for name in ('C3', 'C3_nostrato'):
        try:
            actual, row = evaluate(out / (name + '_cuda'), name, 'cupy', check=True)
            expected, _ = evaluate(out / (name + '_numpy'), name, 'numpy')
            error = relative(actual, expected)
            row.update(selftest_max_relative=error, selftest_pass=bool(np.isfinite(error) and error <= 1e-5))
            row['pass'] = row['bootstrap_pass'] and row['selftest_pass'] and report['historical_selftest']['pass_']
            report['exact'][name] = row
            if name == 'C3':
                fast, fastrow = evaluate(out / 'C3_S2', name, 'cupy', approximate=True)
                delta = abs(fast - actual)
                atol = args.display_atol or 0.0
                bound = 0.001 * abs(actual) + atol
                report['approximate'] = dict(
                    rtol=0.001,
                    atol=atol,
                    baseline_exact_gate_pass=row['pass'],
                    atol_source='caller display bound'
                    if args.display_atol is not None
                    else 'conservative zero; display transform unspecified',
                    max_absolute=float(delta.max()),
                    max_relative=relative(fast, actual),
                    minimum_atol_required=float(np.maximum(delta - 0.001 * abs(actual), 0).max()),
                    pass_=bool(np.isfinite(fast).all() and np.isfinite(actual).all() and np.all(delta <= bound)),
                    solar_elevation_deg=[-6, -9, -12],
                    flags=fastrow['flags'],
                    stage1=fastrow['stage1'],
                )
        except Exception as exc:
            report['exact'][name] = dict(
                error=f'{type(exc).__name__}: {exc}', code='E_GATE_EXCEPTION', **{'pass': False}
            )
    report['pass'] = all(v['pass'] for v in report['exact'].values()) and report['approximate'].get('pass_', False)
    failed = any(v.get('code') == 'E_GATE_EXCEPTION' for v in report['exact'].values())
    report['code'] = 'E_GATE_EXCEPTION' if failed else None
    (out / 'gate.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return 2 if failed else 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
