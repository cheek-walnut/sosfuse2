#!/usr/bin/env python3
"""Tropospheric rural aerosol Mie table (sosfuse2/assets/tropo_rural_sf79_rh70.json) and its checks.

Particle model = LOWTRAN7/MODTRAN boundary-layer RURAL aerosol (Shettle & Fenn 1979, AFGL-TR-79-0214), the model
Lee 2003 used for his LOWTRAN purple-light runs (results/lit_lee/lee_2003_purple.txt:446-451, IHAZE rural, V 23 km,
midlatitude winter, Mie phase functions). Bimodal number-lognormal, sigma in log10:
  small (water-soluble/dust mix) N 0.999875, r 0.02700 um dry -> 0.02846 um at RH 70 %, sigma 0.35
  large                          N 0.000125, r 0.4300 um dry -> 0.4571 um at RH 70 %, sigma 0.40
Dry index 1.53-0.0066i (70 % water-soluble 1.53-0.006i + 30 % dust-like 1.53-0.008i at 0.55 um), spectrally flat
here; wet index = volume mix with water 1.333 by the radius growth of each mode. Radius 0.001-20 um.
RH 70 %: the MLW profile of LOWTRAN7 gives RH 77 / 70.5 / 65.5 % at 0 / 1 / 2 km (lowtran7.f MLATMB, RELHUM),
so the aerosol-weighted boundary layer sits at the 70 % table. The mode radii are the S&F table values as recalled
(the report itself was not retrievable, DTIC 403); they are checked against LOWTRAN7's own rural RH 70 % tables
(lowtran7.f RURE2/RURA2/RURG2, PHSF 4) in checks.lowtran7 below.
usage: OMP_NUM_THREADS=1 python -B -m sosfuse2.generate_tropo_optics --out FILE [--jobs 4]   (--out is required)
"""
import argparse, json, math, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
from sosfuse2.mie import coefficients, phase

DEFAULT = Path(__file__).with_name('assets')/'tropo_rural_sf79_rh70.json'
WATER = complex(1.333, 0.)
DRY = complex(1.53, 0.0066)
MODES = (dict(N=0.999875, r_dry=0.02700, r_wet=0.02846, sigma_log10=0.35),
         dict(N=0.000125, r_dry=0.4300, r_wet=0.4571, sigma_log10=0.40))
R_RANGE = (0.001, 20.)
WAVELENGTHS = sorted([360.]+[360+(i+.5)*470/15 for i in range(15)]+[550., 830.])
L_STORE = 64
# LOWTRAN7 rural RH 70 % (lowtran7.f BLOCK DATA EXTDTA RURE2/RURA2/RURG2, normalized to 1 at 0.55 um; PHSFNC 4 per sr)
LOWTRAN = dict(wavelength_um=[0.3, 0.3371, 0.55, 0.6943, 1.06], ext=[1.7416, 1.5998, 1.0, 0.7532, 0.4217],
               ssa=[0.9379, 0.9521, 0.9462, 0.9378, 0.8735], g=[0.693, 0.686, 0.664, 0.650, 0.631],
               phsf4_deg=[0, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 50, 60, 70, 80, 90, 100, 110, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180])


def index(mode):
    grow = mode['r_wet']/mode['r_dry']
    return WATER+(DRY-WATER)/grow**3


def quadrature():
    """Gauss-Legendre panels in theta (dense near forward) for the Legendre moments."""
    panels = [(0, .5, 48), (.5, 2, 48), (2, 5, 48), (5, 10, 48), (10, 20, 48), (20, 40, 64), (40, 90, 96), (90, 180, 128)]
    th, w = [], []
    for a, b, n in panels:
        t, wt = np.polynomial.legendre.leggauss(n); th.append((a+b)/2+(b-a)/2*t); w.append((b-a)/2*wt)
    th = np.deg2rad(np.concatenate(th)); w = np.deg2rad(np.concatenate(w))*np.sin(th)
    return th, w


def output_grid():
    return np.unique(np.round(np.r_[np.arange(0, 2, .02), np.arange(2, 20, .1), np.arange(20, 180.0001, .25)], 6))


def size_grid(lam_um, dlog=.02, dx=.25):
    """ln r nodes: step min(dlog, dx/x) so large spheres are sampled every dx in size parameter (ripple average)."""
    k = 2*np.pi/lam_um; r = [math.log(R_RANGE[0])]
    while r[-1] < math.log(R_RANGE[1]):
        r.append(r[-1]+min(dlog, dx/(k*math.exp(r[-1]))))
    lr = np.array(r); lr[-1] = math.log(R_RANGE[1])
    w = np.zeros_like(lr); d = np.diff(lr); w[:-1] += d/2; w[1:] += d/2
    return np.exp(lr), w


def ensemble(lam_nm, dlog=.02, dx=.25, extra_deg=()):
    lam = lam_nm*1e-3
    qth, qw = quadrature(); grid = output_grid(); extra = np.asarray(extra_deg, float)
    mu = np.cos(np.r_[qth, np.deg2rad(grid), np.deg2rad(extra)])
    r, wr = size_grid(lam, dlog, dx)
    modes = []
    for m in MODES:
        ls = m['sigma_log10']*math.log(10)
        modes.append((m['N']/(math.sqrt(2*math.pi)*ls)*np.exp(-.5*((np.log(r)-math.log(m['r_wet']))/ls)**2)*wr, index(m)))
    ce = cs = 0.; ang = np.zeros_like(mu)
    for i, ri in enumerate(r):
        x = 2*np.pi*ri/lam
        for dn, m in modes:
            if dn[i] < 1e-300:
                continue
            a, b = coefficients(x, m); n = np.arange(1, len(a)+1)
            qe = 2/x**2*np.sum((2*n+1)*(a+b).real); qs = 2/x**2*np.sum((2*n+1)*(abs(a)**2+abs(b)**2))
            area = np.pi*ri*ri; ce += dn[i]*area*qe; cs += dn[i]*area*qs
            ang += dn[i]*area*qs*phase(mu, x, qs, a, b)
    P = ang/cs
    nq = len(qth); Pq = P[:nq]
    PL = np.polynomial.legendre.legvander(np.cos(qth), L_STORE)
    chi = (qw[:, None]*PL*Pq[:, None]).sum(0)/2*(2*np.arange(L_STORE+1)+1)
    return dict(wavelength_nm=float(lam_nm), C_ext_um2=float(ce), C_sca_um2=float(cs), SSA=float(cs/ce), g=float(chi[1]/3),
                normalization=float(qw@Pq/2), chi=chi, phase=P[nq:nq+len(grid)], extra=P[nq+len(grid):], size_nodes=len(r))


def job(args):
    lam, kw = args
    t = time.perf_counter(); e = ensemble(lam, **kw); e['seconds'] = time.perf_counter()-t
    return e


def hg_sr(theta_deg, g):
    c = np.cos(np.deg2rad(theta_deg))
    return (1-g*g)/(4*np.pi*(1+g*g-2*g*c)**1.5)


def build(jobs):
    lt = LOWTRAN; tasks = [(w, {}) for w in WAVELENGTHS]
    tasks += [(w*1000, dict(extra_deg=lt['phsf4_deg'])) for w in lt['wavelength_um']]
    tasks += [(550., dict(dlog=.01, dx=.125))]
    with ProcessPoolExecutor(jobs) as ex:
        res = list(ex.map(job, tasks))
    rows = res[:len(WAVELENGTHS)]; ltr = res[len(WAVELENGTHS):-1]; fine = res[-1]
    base = next(r for r in rows if r['wavelength_nm'] == 550.)
    grid = output_grid()
    win = (grid >= 10) & (grid <= 40)
    conv = dict(C_ext=abs(fine['C_ext_um2']/base['C_ext_um2']-1), g=abs(fine['g']-base['g']),
                chi_l32_max_abs=float(np.max(abs(fine['chi'][:33]-base['chi'][:33]))),
                phase_10_40_max_rel=float(np.max(abs(fine['phase'][win]/base['phase'][win]-1))))
    conv['pass_'] = conv['C_ext'] < 2e-3 and conv['g'] < 1e-3 and conv['phase_10_40_max_rel'] < 5e-3
    norm = dict(max_abs=max(abs(r['normalization']-1) for r in rows))
    norm['pass_'] = norm['max_abs'] < 1e-4
    e550 = next(r for r in ltr if abs(r['wavelength_nm']-550) < 1e-9)
    lowtran = dict(model='LOWTRAN7 rural RH 70 % (lowtran7.f)', rows=[])
    for r, w, ext, ssa, g in zip(ltr, lt['wavelength_um'], lt['ext'], lt['ssa'], lt['g']):
        lowtran['rows'].append(dict(wavelength_um=w, ext_rel_mie=r['C_ext_um2']/e550['C_ext_um2'], ext_rel_lowtran=ext,
                                    ssa_mie=r['SSA'], ssa_lowtran=ssa, g_mie=r['g'], g_lowtran=g))
    lowtran['phase_550_per_sr'] = dict(theta_deg=lt['phsf4_deg'], mie=[float(x/(4*np.pi)) for x in e550['extra']],
                                       note='LOWTRAN PHSFNC 4 (rural 0/70 % at 0.55/0.69 um) is a binned average; compare shape only')
    th = np.array([10, 15, 20, 25, 30, 40, 90, 180.])
    comp = dict(theta_deg=th.tolist(), hg_065=hg_sr(th, .65).tolist(), hg_080=hg_sr(th, .8).tolist())
    for r in rows:
        if r['wavelength_nm'] in (375.6666666666667, 438.3333333333333, 550., 657.6666666666667, 814.3333333333333):
            comp[f"mie_{r['wavelength_nm']:.0f}"] = (np.exp(np.interp(th, grid, np.log(r['phase'])))/(4*np.pi)).tolist()
    trunc = {f"{r['wavelength_nm']:.1f}": dict(eig33=float(r['chi'][33]/67), eig48=float(r['chi'][48]/97), eig64=float(r['chi'][64]/129))
             for r in rows}
    checks = dict(normalization=norm, size_angle_convergence_550=conv, lowtran7=lowtran)
    return dict(version=1, model='Shettle & Fenn (1979) rural aerosol, RH 70 %, bimodal number-lognormal; Bohren-Huffman spheres (mie.py)',
                particle=dict(modes=[dict(m, m_real=index(m).real, m_imag=index(m).imag) for m in MODES], dry_index=[DRY.real, DRY.imag],
                              water_index=[WATER.real, WATER.imag], radius_range_um=list(R_RANGE), relative_humidity_percent=70),
                polarization='fully depolarizing scalar Mie',
                phase_convention='phase = mean-one phase function (integral over the sphere / 4 pi = 1) on theta_deg; chi_l coefficients of P_l (chi_0=1); SH eigenvalue chi_l/(2l+1)',
                quadrature=dict(size='ln r step min(0.02, 0.25/x)', theta='Gauss-Legendre panels 0-0.5-2-5-10-20-40-90-180 deg (528 nodes) for chi_l', L_stored=L_STORE),
                wavelengths_nm=[r['wavelength_nm'] for r in rows], theta_deg=grid.tolist(),
                rows=[dict(wavelength_nm=r['wavelength_nm'], C_ext_um2=r['C_ext_um2'], C_sca_um2=r['C_sca_um2'], SSA=r['SSA'], g=r['g'],
                           chi_l=[float(x) for x in r['chi'][:33]], chi_l_33_64=[float(x) for x in r['chi'][33:]], size_nodes=r['size_nodes'],
                           phase=[float('%.9g' % x) for x in r['phase']]) for r in rows],
                checks=checks, diagnostics=dict(phase_per_sr_vs_hg=comp, truncation_eigenvalues=trunc,
                                                seconds=[round(r['seconds'], 1) for r in res]),
                pass_=all(c.get('pass_', True) for c in (norm, conv)))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True, help=f'output JSON; the packaged asset is {DEFAULT.name} (regenerating it is a maintenance step, 2026-11-03 release review)'); p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args(); d = build(a.jobs); a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(d, indent=1, allow_nan=False)+'\n')
    print(json.dumps(dict(checks=dict(normalization=d['checks']['normalization'], convergence=d['checks']['size_angle_convergence_550'],
                                      lowtran7=d['checks']['lowtran7']['rows']), comparison=d['diagnostics']['phase_per_sr_vs_hg']), indent=1))
    print('TROPO MIE PASS' if d['pass_'] else 'TROPO MIE FAIL')
    if not d['pass_']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
