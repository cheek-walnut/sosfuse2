"""Generate self-contained libRadtran species and molecular input files."""

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

from common import add_paths, dump, sha, work_path

H = None


def files(data, c):
    levels = np.asarray(data['levels_km'])
    lo = levels[1:]
    hi = levels[:-1]
    dz = hi - lo
    checks = {}
    ledger = []
    for sid, s in data['states'].items():
        a = s['aerosol']
        p = s['state']['parameters']
        root = H / 'inputs' / sid
        lib = root / 'speclib'
        lib.mkdir(parents=True, exist_ok=True)
        (lib / 'aerosol_species_names.dat').write_text('tropo\nstrato\n')
        edges = np.asarray(a['edges_km'])
        j = np.clip(np.searchsorted(edges, (lo + hi) / 2, side='right') - 1, 0, len(edges) - 2)
        tro = np.asarray(a['tropo_ext550_shell'])
        sd = np.asarray(a['strato_density_shell'])
        aw = np.asarray(a['wavelengths_nm'])
        i550 = int(np.flatnonzero(np.isclose(aw, 550))[0])
        strat = sd * a['strato_sca'][i550]
        np.savetxt(
            root / 'species_profile.dat',
            np.column_stack([levels, np.r_[0.0, tro[j]], np.r_[0.0, strat[j]]]),
            fmt='%.16e',
        )
        profile = np.loadtxt(root / 'species_profile.dat')[1:]
        for k, (_z, t, v) in enumerate(profile):
            for component, expected, actual in (
                ('tropo', tro[j[k]] * dz[k], t * dz[k]),
                ('strato', strat[j[k]] * dz[k], v * dz[k]),
            ):
                error = abs(actual / expected - 1) if expected else abs(actual)
                ledger.append(
                    dict(
                        state=sid,
                        wavelength_nm=550.0,
                        component=component,
                        layer=k,
                        engine=float(expected),
                        input=float(actual),
                        relative_error=float(error),
                    )
                )
        theta = np.linspace(180, 0, 1801)
        mu = np.cos(np.deg2rad(theta))
        n = c['legendre_order'] + 1
        m = np.arange(n)
        for name in ('tropo', 'strato'):
            phase = []
            pm = []
            ext = []
            ssa = []
            gg = []
            for i, w in enumerate(aw):
                if name == 'tropo':
                    g = p['g']
                    v = (1 - g * g) / (1 + g * g - 2 * g * mu) ** 1.5
                    moment = (2 * m + 1) * g**m
                    ex = (550 / w) ** p['angstrom']
                    al = p['ssa']
                else:
                    chi = np.asarray(a['strato_chi'][i])
                    moment = np.zeros(n)
                    moment[: len(chi)] = chi
                    v = np.polynomial.legendre.legval(mu, moment)
                    g = chi[1] / 3
                    ex = a['strato_sca'][i] / a['strato_sca'][i550]
                    al = 1.0
                phase.append(v)
                pm.append(moment)
                ext.append(ex)
                ssa.append(al)
                gg.append(g)
            vals = dict(
                wavelen=aw / 1000,
                hum=[0],
                ntheta=np.full(len(aw), len(theta)),
                theta=np.tile(theta, len(aw)),
                phase=phase,
                nmom=np.full(len(aw), n),
                pmom=pm,
                ext=ext,
                ssa=ssa,
                gg=gg,
                refre=np.full(len(aw), 1.43),
                refim=np.zeros(len(aw)),
                rho=np.ones(len(aw)),
            )
            head = f'''netcdf {name} {{
dimensions: nlam={len(aw)}; nhum=1; nphamat=1; nthetamax={len(theta)}; nmommax={n};
variables:
 double wavelen(nlam); double hum(nhum);
 int ntheta(nlam,nhum,nphamat); float theta(nlam,nhum,nphamat,nthetamax); theta:_FillValue=-999.f;
 float phase(nlam,nhum,nphamat,nthetamax); phase:_FillValue=-999.f;
 int nmom(nlam,nhum,nphamat); float pmom(nlam,nhum,nphamat,nmommax); pmom:_FillValue=3.402823e38f;
 double ext(nlam,nhum); double ssa(nlam,nhum); double gg(nlam,nhum); double refre(nlam); double refim(nlam);\
 double rho(nlam,nhum);
:version=20110916; :complete_moments=0;
data:
'''
            body = '\n'.join(
                k + ' = ' + ', '.join(format(float(v), '.12g') for v in np.asarray(x).ravel()) + ';'
                for k, x in vals.items()
            )
            cdl = lib / f'{name}.cdl'
            cdl.write_text(head + body + '\n}\n')
            subprocess.run(['ncgen', '-k', 'classic', '-o', str(lib / f'{name}.mie.cdf'), str(cdl)], check=True)
            checks[sid + '_' + name] = dict(
                min_phase=float(np.min(phase)),
                max_integral_error=float(np.max(abs(np.trapezoid(phase, mu, axis=1) / 2 - 1))),
            )
        for i, w in enumerate(s['wavelengths_nm']):
            folder = root / f'wl_{i:02d}'
            folder.mkdir(exist_ok=True)
            (folder / 'atmosphere.dat').write_text(
                ''.join(f'{z:.6f} {1013.25 * math.exp(-z / 8):.12e} 288.15\n' for z in levels)
            )
            (folder / 'solar.dat').write_text(f'300 1\n{w:.17g} 1\n900 1\n')
            tc = s['tau_components']
            sca = np.asarray(tc['rayleigh'][i])
            absorb = sum(np.asarray(tc[x][i]) for x in ('o3', 'no2', 'o4'))
            for name, t in [('tau_sca.dat', sca), ('tau_abs.dat', absorb)]:
                (folder / name).write_text(
                    ''.join(f'{z:.6f} {v:.17e}\n' for z, v in zip(levels, np.r_[0.0, t], strict=True))
                )
            for component in ('rayleigh', 'o3', 'no2', 'o4'):
                values = np.asarray(tc[component][i])
                path = folder / f'component_{component}.dat'
                path.write_text(''.join(f'{z:.6f} {v:.17e}\n' for z, v in zip(levels, np.r_[0.0, values], strict=True)))
                read = np.loadtxt(path)[:, 1][1:]
                errors = np.divide(abs(read - values), abs(values), out=np.zeros_like(values), where=values != 0)
                ledger.extend(
                    dict(
                        state=sid,
                        wavelength_nm=w,
                        component=component,
                        layer=k,
                        engine=float(v),
                        input=float(x),
                        relative_error=float(e),
                    )
                    for k, (v, x, e) in enumerate(zip(values, read, errors, strict=True))
                )
            serialized_sca = np.loadtxt(folder / 'tau_sca.dat')[:, 1][1:]
            serialized_abs = np.loadtxt(folder / 'tau_abs.dat')[:, 1][1:]
            if not np.allclose(serialized_sca, sca, rtol=1e-12, atol=0) or not np.allclose(
                serialized_abs, absorb, rtol=1e-12, atol=0
            ):
                raise ValueError('E_SERIALIZED_TAU')
        (root / 'zero_tau.dat').write_text(''.join(f'{z:.6f} 0\n' for z in levels))
    worst = max(x['relative_error'] for x in ledger)
    # Same tolerance as the E_SERIALIZED_TAU check above; the ledger is written before the refusal.
    dump(H / 'out/layer_ledger.json', dict(passed=worst <= 1e-12, rows=ledger, max_relative_error=worst))
    if worst > 1e-12:
        raise ValueError(f'E_LAYER_LEDGER: relative error {worst:.3e} between engine and written layer values')
    dump(H / 'out/phase_checks.json', checks)
    if any(x['max_integral_error'] > 1e-4 or x['min_phase'] < -1e-8 for x in checks.values()):
        raise ValueError('E_PHASE_CHECK')
    return checks


def text_input(data, sid, p, sza, wi, photons, seed, lrt, disort=False, vacuum=False, work=None):
    s = data['states'][sid]
    root = Path(work if work is not None else H) / 'inputs' / sid
    folder = root / f'wl_{wi:02d}'
    w = s['wavelengths_nm'][wi]
    sca = root / 'zero_tau.dat' if vacuum else folder / 'tau_sca.dat'
    ab = root / 'zero_tau.dat' if vacuum else folder / 'tau_abs.dat'
    txt = f'''data_files_path {lrt / 'data'}
atmosphere_file {folder / 'atmosphere.dat'}
mol_abs_param crs
mol_tau_file sca {sca}
mol_tau_file abs {ab}
rayleigh_depol 0.03
source solar {folder / 'solar.dat'}
wavelength {w:.17g} {w:.17g}
earth_radius 6371
albedo {s['state']['parameters']['albedo']:.17g}
sza {sza}
phi0 0
mol_modify NO2 0 DU
ck_lowtran_absorption O4 off
no_absorption mol
'''
    if not vacuum:
        txt += (
            f'aerosol_default\naerosol_species_library {root / "speclib"}/\n'
            f'aerosol_species_file {root / "species_profile.dat"} tropo strato\n'
        )
    if disort:
        return txt + 'rte_solver disort\nnumber_of_streams 16\nzout 0\noutput_user lambda edir\n'
    return (
        txt
        + f'''zout 0.1
umu {-math.sin(math.radians(p['elevation_deg'])):.17g}
phi {p['azimuth_deg']:.17g}
rte_solver mystic
mc_spherical 1D
mc_backward
mc_escape on
mc_vroom on
mc_photons {photons}
mc_randomseed {seed}
mc_std
mc_basename mc
output_quantity transmittance
quiet
mc_polarisation
'''
    )


def verify_disort(data, c, lrt):
    rows = []
    for sid, s in data['states'].items():
        wi = 14
        a = s['aerosol']
        w = s['wavelengths_nm'][wi]
        ia = int(np.argmin(abs(np.asarray(a['wavelengths_nm']) - w)))
        aerosol = (
            a['tropo_column'] * (550 / w) ** s['state']['parameters']['angstrom']
            + a['strato_column'] * a['strato_ext'][ia]
        )
        for vacuum in (False, True):
            folder = H / 'runs' / 'disort' / sid / ('vacuum' if vacuum else 'full')
            folder.mkdir(parents=True, exist_ok=True)
            inp = text_input(data, sid, c['pixels'][0], 0, wi, 1, 1, lrt, True, vacuum)
            (folder / 'uvspec.inp').write_text(inp)
            with (
                (folder / 'uvspec.inp').open() as fi,
                (folder / 'stdout.txt').open('w') as fo,
                (folder / 'stderr.txt').open('w') as fe,
            ):
                proc = subprocess.run([str(lrt / 'bin/uvspec')], stdin=fi, stdout=fo, stderr=fe, cwd=folder)
            if proc.returncode:
                raise RuntimeError('E_DISORT_EXIT ' + (folder / 'stderr.txt').read_text()[-1000:])
            measured = -math.log(float((folder / 'stdout.txt').read_text().split()[-1]))
            expected = (
                0.0
                if vacuum
                else aerosol + sum(sum(s['tau_components'][x][wi]) for x in ('rayleigh', 'o3', 'no2', 'o4'))
            )
            error = abs(measured / expected - 1) if expected else abs(measured)
            rows.append(
                dict(state=sid, vacuum=vacuum, wavelength_nm=w, measured=measured, expected=expected, error=error)
            )
            if error > (1e-4 if expected else 1e-7):
                raise ValueError('E_DISORT_TAU ' + str(rows[-1]))
    dump(H / 'out/disort.json', dict(passed=True, rows=rows))
    print('DISORT_PASS', rows)


def main():
    global H
    ap = argparse.ArgumentParser()
    add_paths(ap)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--lrt-dir', type=Path)
    group.add_argument('--no-disort', action='store_true')
    a = ap.parse_args()
    if not shutil.which('ncgen'):
        raise SystemExit('E_NCGEN_MISSING: install netcdf-bin and make ncgen available on PATH')
    H = work_path(a.work)
    data = json.loads((H / 'out/optics.json').read_text())
    c = json.loads(a.config.read_text())
    if data['identity']['config'] != sha(a.config):
        raise ValueError('E_CONFIG_SHA')
    print('PHASE', files(data, c))
    if a.no_disort:
        dump(H / 'out/disort.json', dict(passed=False, skipped=True, reason='DISORT was not run'))
    else:
        verify_disort(data, c, a.lrt_dir.resolve())
    checks = {str(p.relative_to(H)): sha(p) for p in (H / 'inputs').rglob('*') if p.is_file()}
    dump(H / 'out/input_sha.json', checks)


if __name__ == '__main__':
    main()
