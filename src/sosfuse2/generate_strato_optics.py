#!/usr/bin/env python3
"""Reproducible local sulfate Mie table and convergence/self-check report."""

import argparse, json, sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
from sosfuse2.mie import ensemble, self_checks

DEFAULT = Path(__file__).with_name('assets') / 'strato_sulfate_rg0.08_sg1.6_n1.43.json'


def build(k=0.0, rg=0.08, sg=1.6):
    wl = np.sort(np.r_[360.0, 360 + (np.arange(15) + 0.5) * 470 / 15, 550.0, 830.0])
    rows = [ensemble(w, rg=rg, sg=sg, k=k, size_nodes=1024) for w in wl]
    checks = self_checks()
    tighter = ensemble(550, rg=rg, sg=sg, k=k, size_nodes=1536, angle_nodes=768)
    base = next(r for r in rows if r['wavelength_nm'] == 550)
    convergence = {
        key: abs(tighter[key] - base[key]) / max(abs(tighter[key]), 1e-30) for key in ('C_ext_um2', 'C_sca_um2', 'g')
    }
    convergence['chi_max_absolute'] = float(np.max(abs(np.array(tighter['chi_l']) - base['chi_l'])))
    checks['ensemble_convergence'] = dict(**convergence, pass_=max(convergence.values()) < 1e-5)
    checks['ensemble_normalization'] = dict(
        max_absolute=max(abs(r['phase_mean'] - 1) for r in rows),
        max_g_error=max(abs(r['g'] - r['g_angular']) for r in rows),
        pass_=max(abs(r['phase_mean'] - 1) for r in rows) < 1e-10
        and max(abs(r['g'] - r['g_angular']) for r in rows) < 1e-10,
    )
    return dict(
        version=1,
        model='Bohren-Huffman homogeneous sphere; number-lognormal ensemble',
        particle=dict(r_g_um=rg, sigma_g=sg, n_real=1.43, n_imag=k),
        polarization='fully depolarizing scalar Mie; amplitudes not used for Stokes transport',
        phase_convention='P=sum chi_l P_l; mean(P)=1; SH eigenvalue=chi_l/(2l+1)',
        quadrature=dict(size_nodes=1024, log_radius_sigma_extent=9, angle_nodes=512, L=32),
        wavelengths_nm=wl.tolist(),
        rows=rows,
        checks=checks,
        pass_=all(c['pass_'] for c in checks.values()),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--out',
        type=Path,
        required=True,
        help=f'output JSON; the packaged asset is {DEFAULT.name} (regenerating it is a maintenance step, 2026-11-03 release review)',
    )
    p.add_argument('--imag-index', type=float, default=0.0)
    p.add_argument('--rg', type=float, default=0.08, help='lognormal median radius, um')
    p.add_argument('--sigma-g', type=float, default=1.6)
    a = p.parse_args()
    d = build(a.imag_index, a.rg, a.sigma_g)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(d, indent=2, allow_nan=False) + '\n')
    print(json.dumps(d['checks'], indent=2))
    print('MIE PASS' if d['pass_'] else 'MIE FAIL')
    if not d['pass_']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
