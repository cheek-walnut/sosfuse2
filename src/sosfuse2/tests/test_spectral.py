#!/usr/bin/env python3
"""Spectral sampling modes (spectral.py): Mac checks.

  (default)  unit checks: legacy settings unchanged, band = bin integral, point = monochromatic, edge/mode errors,
             weights split invariance (§4-3), provenance vs assets/spectra.json (§4-4), `contract` keys and hashes
  --runs     reduced NumPy solves (rig/states_v2.json clear_33, 3 wavelengths) through run_gpu_state_v3:
               legacy     this tree == 6a5c3f5 tree (git archive), table bytes (§6-3)
               E          band with E of bin 1 doubled: that wavelength x2 exactly, the others bit-equal (§4-1)
               sigma      band with sigma of bin 1 x1.5: the other wavelengths bit-equal (§4-2)
               cache      legacy then band on one work root (geometry/attach/column caches kept) == band on a fresh root (§4-6),
                          without and with --stratosphere on (attach cache hit)
               n>15       16 band wavelengths (2 chunks): M3 extra-elevation row from the stored source == direct run row
usage: python -B -m sosfuse2.tests.test_spectral [--runs --out <dir> --jobs 4]
"""
import argparse, json, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
sys.dont_write_bytecode = True
import numpy as np
from sosfuse2 import paths, testing
from sosfuse2 import spectral as SP
from sosfuse2 import run_gpu_state_v3 as V3
ROOT = testing.sky_sos_root()   # sky-sos checkout (None stand-alone): the legacy-tree runs and the builder rerun need it
CUR = paths.SRC                 # 'current tree' marker for run(): the installed package
try:
    from sosfuse2.numerical_checks import assert_equivalent, within_tol
except ImportError:  # standalone research harness
    from numerical_checks import assert_equivalent, within_tol
LEGACY_BASELINE = '6a5c3f5'
TREE_FILES = ('scripts', 'rig', 'results/sos_full/a2_driver.py', 'results/sos_full/b2_driver.py', 'assets')
FAILS = []
ON = ('--stratosphere', 'on')    # attach cache in use (clear states have the layer off)


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ')+name+(' '+detail if detail else ''), flush=True)
    if not ok: FAILS.append(name)


def settings(raw, tmp):
    path = None
    if raw is not None:
        path = Path(tmp)/'s.json'; path.write_text(json.dumps(raw))
    return V3.settings_for(SimpleNamespace(reduced=False, settings_json=path))


def expect(code, fn):
    try: fn()
    except ValueError as e: return str(e).startswith(code)
    return False


def units():
    with tempfile.TemporaryDirectory() as tmp:
        base = settings({'spectral_mode': 'legacy'}, tmp)
        check('legacy: explicit legacy adds no spectral key, 15-point linspace grid', 'spectral' not in base and
              base['wavelengths_nm'] == (360+(np.arange(15)+.5)*470/15).tolist())
        prod = settings(None, tmp)
        check('product default {} = band BM45 (45 bins, SP.PRODUCT_EDGES, midpoints)', settings({}, tmp) == prod and
              prod['spectral']['mode'] == 'band' and prod['spectral']['bin_edges_nm'] == SP.PRODUCT_EDGES and
              prod['wavelengths_nm'] == SP.PRODUCT['wavelengths_nm'] and len(prod['wavelengths_nm']) == 45)
        check('product default ozone = A′ profile_mean (P2, user 2026-10-31), resolved per state',
              prod['spectral']['ozone_temperature'] == 'profile_mean' and 'ozone_column' not in prod['spectral'])
        fixed = settings({'ozone_temperature': '233K'}, tmp)
        check('ozone_temperature 233K keeps the fixed column on BM45', fixed['wavelengths_nm'] == prod['wavelengths_nm'] and
              fixed['spectral']['ozone_column'] == 'ozone_233K' and fixed['spectral']['bin_edges_nm'] == SP.PRODUCT_EDGES)
        temp = settings({'ozone_temperature_K': 213}, tmp)
        check('temperature-only settings preserve 213 K on BM45', temp['wavelengths_nm'] == prod['wavelengths_nm'] and
              temp['spectral'] == dict(fixed['spectral'], ozone_column='ozone_213K'))
        check('E_SPECTRAL_MODE: wavelengths or bins changed without a mode',
              expect('E_SPECTRAL_MODE', lambda: settings({'wavelengths': 30}, tmp)) and
              expect('E_SPECTRAL_MODE', lambda: settings({'bin_edges_nm': SP.PRODUCT_EDGES}, tmp)))
        red = V3.settings_for(SimpleNamespace(reduced=True, settings_json=None))
        check('reduced smoke grid stays legacy 550 nm', 'spectral' not in red and red['wavelengths_nm'] == [550.])
        band = settings({'spectral_mode': 'band', 'wavelengths': 40}, tmp)['spectral']
        point = settings({'spectral_mode': 'point', 'wavelengths': 40}, tmp)['spectral']
        check('band default bins linspace(360,830,41)', np.array_equal(band['bin_edges_nm'], np.linspace(360, 830, 41)))
        check('E_SPECTRAL_EDGES explicit wavelengths without edges',
              expect('E_SPECTRAL_EDGES', lambda: settings({'spectral_mode': 'band', 'wavelengths': 2, 'wavelengths_nm': [500., 600.]}, tmp)))
        check('E_SPECTRAL_EDGES wavelength outside its bin',
              expect('E_SPECTRAL_EDGES', lambda: settings({'spectral_mode': 'band', 'wavelengths': 2, 'wavelengths_nm': [500., 600.],
                                                         'bin_edges_nm': [400., 550., 580.]}, tmp)))
        check('E_SPECTRAL_MODE / E_SPECTRAL_OZONE_T',
              expect('E_SPECTRAL_MODE', lambda: settings({'spectral_mode': 'bands'}, tmp)) and
              expect('E_SPECTRAL_MODE', lambda: settings({'spectral_mode': 'legacy', 'ozone_temperature_K': 213}, tmp)) and
              expect('E_SPECTRAL_OZONE_T', lambda: settings({'spectral_mode': 'band', 'ozone_temperature_K': 240}, tmp)))
        state = json.loads((paths.ASSETS/'states_v2.json').read_text())['states'][0]
        sig = {k: V3.signed_contract(state, c, 'numpy', False, 'depolarizing') for k, c in
               (('legacy', base), ('band', settings({'spectral_mode': 'band'}, tmp)), ('point', settings({'spectral_mode': 'point'}, tmp)),
                ('band213', settings({'spectral_mode': 'band', 'ozone_temperature_K': 213}, tmp)))}
        check('contract: asset hash only off legacy, four distinct signatures',
              'assets/spectra_fine.json' not in sig['legacy'][1] and all('assets/spectra_fine.json' in sig[k][1] for k in ('band', 'point', 'band213'))
              and len({v[0] for v in sig.values()}) == 4)
    A = SP.asset(); wl = 360+(np.arange(40)+.5)*470/40
    s_b, e_b = SP.representative(band, wl)
    fine = np.linspace(360, 830, 470_001)                        # 0.001 nm, linear interpolant of the asset
    dense = lambda y, lo, hi: np.trapezoid(np.interp(f := fine[(fine >= lo-1e-9) & (fine <= hi+1e-9)], A['wl'], y), f)/(hi-lo)
    e = band['bin_edges_nm']; ref = np.array([dense(A['solar'], lo, hi) for lo, hi in zip(e[:-1], e[1:])])
    check('band E = bin integral of the asset interpolant', np.max(abs(e_b/ref-1)) < 1e-6, '%.2e' % np.max(abs(e_b/ref-1)))
    s_p, e_p = SP.representative(point, wl)
    dense_wl = np.sort(np.r_[wl, 360+np.arange(41)*470/40])
    s_p2, e_p2 = SP.representative(point, dense_wl); at = np.searchsorted(dense_wl, wl)
    check('point: asset values, unchanged by a denser grid', within_tol(e_p, np.interp(wl, A['wl'], A['solar'])) and
          within_tol(s_p, np.interp(wl, A['wl'], A['ozone_233K'])) and within_tol(s_p2[at], s_p) and within_tol(e_p2[at], e_p))
    # §4-3 split invariance: R = 1 -> L_i = E_rep,i, so sum_i W_ic L_i = K int c E over the union of bins
    for mode in ('band', 'point'):
        b15 = dict(band if mode == 'band' else point, bin_edges_nm=np.linspace(360, 830, 16).tolist())
        w15 = 360+(np.arange(15)+.5)*470/15; e30 = np.sort(np.r_[b15['bin_edges_nm'], w15])   # each bin split at l_i
        w30 = (e30[:-1]+e30[1:])/2; b30 = dict(b15, bin_edges_nm=e30.tolist())
        x15 = SP.representative(b15, w15)[1] @ SP.channel_weights(b15, w15)
        x30 = SP.representative(b30, w30)[1] @ SP.channel_weights(b30, w30)
        check(f'§4-3 split invariance ({mode}) XYZS', np.max(abs(x30/x15-1)) < 1e-12, '%.1e' % np.max(abs(x30/x15-1)))
    # Provenance (the check uses the pre-relabel columns):
    # the vacuum-axis ozone columns the builder makes before the air relabel vs assets/spectra.json [l, l+10) means, and the
    # solar checks (TSIS-1 HSRS: Jacobian energy, column integral, BM45 vs the HSRS p1nm product). With the
    # gitignored sources present the builder is rerun (its columns must match the asset within the 7-digit rounding);
    # without them the self-check the builder stored in the asset is read.
    doc = json.loads(SP.ASSET.read_text()); sc = doc['self_check']
    RAW = testing.sky_sos_optional('results/sigma_tz_20261028/raw/serdyuchenkogorshelev5digits.dat')
    HSRS = testing.sky_sos_optional('results/d1_axis_20261031/raw/tsis1_hsrs_355_845.csv')
    if RAW is not None and HSRS is not None:
        import importlib.util
        spec = importlib.util.spec_from_file_location('build_spectra_fine', testing.sky_sos('scripts/build_spectra_fine.py'))
        B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
        H = B.read_hsrs(HSRS); cols = B.columns(H, B.read_serdyuchenko(RAW)); sc = B.provenance(cols, H)
        # tol, not bytes: prepared inputs compare at rtol 1e-6; 6e-7 = 7-digit rounding
        check('§4-4 builder rerun reproduces the asset columns (rtol 6e-7)',
              all(np.allclose(cols[k], doc[k], rtol=6e-7, atol=0) for k in ('solar', 'ozone_213K', 'ozone_233K', 'ozone_253K')))
        st, rt = doc['self_check']['solar'], sc['solar']
        check('§4-4 recomputed solar self-check equals the stored one (rtol 1e-6)',
              all(np.isclose(rt[k], st[k], rtol=1e-6, atol=1e-15) for k in ('jacobian_energy_relative', 'integral_360_840_relative', 'vacuum_domain_max_relative'))
              and np.isclose(rt['bm45_vs_p1nm']['max_relative'], st['bm45_vs_p1nm']['max_relative'], rtol=1e-6, atol=0))
    else:
        print('§4-4: raw Serdyuchenko or HSRS file absent (gitignored), reading the stored self-check')
    ro = sc['ozone_vacuum_233K']
    # l = 360: the full 0.01 nm source integral itself is 0.99824 x the old value (old table provenance), not a sampling error.
    check('§4-4 provenance vacuum ozone 370-820 in 0.999-1.001, 360 in 0.998-1.001',
          .999 < ro['ratio_370_820'][0] and ro['ratio_370_820'][1] < 1.001 and .998 < ro['ratio_360'] < 1.001,
          f"{ro['ratio_370_820'][0]:.5f}..{ro['ratio_370_820'][1]:.5f}, 360 {ro['ratio_360']:.5f}")
    ss = sc['solar']
    check('§4-4 solar (HSRS): Jacobian energy 1e-12, column integral 2e-5, BM45 vs p1nm 2e-3, vacuum-domain cell means 6e-7',
          abs(ss['jacobian_energy_relative']) <= 1e-12 and abs(ss['integral_360_840_relative']) <= 2e-5 and ss['bm45_vs_p1nm']['max_relative'] <= 2e-3
          and ss['vacuum_domain_max_relative'] <= 6e-7,
          f"energy {ss['jacobian_energy_relative']:.1e}, integral {ss['integral_360_840_relative']:.2e}, p1nm max {ss['bm45_vs_p1nm']['max_relative']:.2e}, "
          f"vacuum {ss['vacuum_domain_max_relative']:.2e}")
    check('§4-4 solar source is the pinned TSIS-1 HSRS V2', doc['sources']['solar']['sha256'] == 'ac751973cbe95ab7a604a8781912d8496f65b4e94a81fab2d412a4857fe37a3e'
          and 'c2022-11-30' in doc['sources']['solar']['version'] and 'cell mean' in doc['units']['solar'])
    check('D1 axis declared air', doc['wavelength_reference'].startswith('air') and doc['sources']['solar']['wavelength_reference'].startswith('air')
          and doc['sources']['ozone']['wavelength_reference'].startswith('air'))
    # n-wavelength production checks: bake_run.check_table takes the state's settings count (sky-sos rig consumer, rig/bake)
    if ROOT is not None:
        import importlib.util
        spec = importlib.util.spec_from_file_location('bake_run', testing.sky_sos('rig/bake/bake_run.py'))
        BR = importlib.util.module_from_spec(spec); spec.loader.exec_module(BR)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp); z = dict(radiance=np.ones((25, 100, 180, 30), 'f4'), wavelength_nm=np.arange(30, dtype='f4'), solar_elevation_deg=np.zeros(25, 'f4'),
                                    view_elevation_deg=np.zeros(100, 'f4'), relative_azimuth_deg=np.zeros(180, 'f4'), units=np.array('x'))
            np.savez(d/'table.npz', **z); ok_none = BR.check_table(d/'table.npz')[0]
            (d/'meta.json').write_text(json.dumps(dict(settings=dict(wavelengths_nm=list(range(30)))))); ok30 = BR.check_table(d/'table.npz')[0]
            (d/'meta.json').write_text(json.dumps(dict(settings=dict(wavelengths_nm=list(range(15)))))); ok15 = BR.check_table(d/'table.npz')[0]
        check('bake_run.check_table: 30-wavelength table passes with its 30-wavelength settings only', ok30 and not ok15 and not ok_none)
    else:
        print(f'bake_run.check_table: sky-sos rig/bake not available ({testing.ENV} unset), skipped')
    other = dict(band, mode='point')
    check('check_contract: equal passes, mode/legacy mismatch refused', SP.check_contract(band, dict(band)) is None and
          SP.check_contract(None, None) is None and expect('E_SPECTRAL_CONTRACT', lambda: SP.check_contract(band, other)) and
          expect('E_SPECTRAL_CONTRACT', lambda: SP.check_contract(None, band)))


CHILD = r'''
import sys,json
from pathlib import Path
sys.path[:0]=[sys.argv[1]]
from sosfuse2 import spectral as SP
from sosfuse2 import run_gpu_state_v3 as V3
kind,k=sys.argv[2],1
if kind!='none':
    orig=SP.spec_table
    def patched(b,w,state=None):
        t=orig(b,w,state);key={'E':'solar','sigma':'ozone'}[kind];t[key][k]*= {'E':2.,'sigma':1.5}[kind];return t
    SP.spec_table=patched
sys.argv=[sys.argv[0]]+sys.argv[3:];V3.main()
'''


def run(tree, work, out_name, raw, kind='none', extra=()):
    s = Path(work).parent/f'{Path(work).name}.{out_name}.settings.json'; s.write_text(json.dumps(raw))
    args = ['--id', 'clear_33', '--out-name', out_name, '--work-root', str(work), '--backend', 'numpy', '--reduced', '--settings-json', str(s), *extra]
    if Path(tree) == CUR: cmd, cwd = [sys.executable, '-B', '-c', CHILD, str(paths.SRC), kind]+args, paths.SRC
    else: cmd, cwd = [sys.executable, '-B', str(Path(tree)/'scripts/sosfuse2/run_gpu_state_v3.py')]+args, Path(tree)/'scripts'   # legacy layout
    log = Path(work).parent/f'{Path(work).name}.{out_name}.log'
    with log.open('w') as f: rc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc: raise RuntimeError(f'{out_name}: exit {rc}, see {log}')
    with np.load(Path(work)/'out'/out_name/'table.npz') as z: return z['radiance'].copy()


def load_hs(work, name):
    with np.load(Path(work)/'out'/name/'table.npz') as z: return z['solar_elevation_deg'].astype(float)


def runs(out, jobs):
    if ROOT is None: raise SystemExit(f'E_SKY_SOS_ROOT: --runs extracts the legacy tree {LEGACY_BASELINE} from the sky-sos checkout ({testing.ENV})')
    out = Path(out)
    if (out.exists() or out.is_symlink()) and not out.is_dir(): raise SystemExit(f'E_SPECTRAL_OUT_NOT_DIRECTORY: {out}')
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): raise SystemExit(f'E_SPECTRAL_OUT_NOT_EMPTY: {out}')
    tree = out/'legacy_tree'; tree.mkdir()
    with subprocess.Popen(['git', '-C', str(ROOT), 'archive', LEGACY_BASELINE, *TREE_FILES], stdout=subprocess.PIPE) as archive:
        try: subprocess.run(['tar', '-x', '-C', str(tree)], stdin=archive.stdout, check=True)
        finally: archive.stdout.close()
        if archive.wait(): raise subprocess.CalledProcessError(archive.returncode, archive.args)
    leg, band = {'wavelengths': 3, 'spectral_mode': 'legacy'}, {'wavelengths': 3, 'spectral_mode': 'band'}
    tasks = dict(head=(tree, out/'w_head', 'leg', leg), cur=(CUR, out/'w_cur', 'leg', leg), band=(CUR, out/'w_band', 'band', band),
                 E=(CUR, out/'w_E', 'bandE', band, 'E'), sigma=(CUR, out/'w_sig', 'bandS', band, 'sigma'),
                 shared_leg=(CUR, out/'w_shared', 'leg', leg),
                 sband=(CUR, out/'w_sband', 'band', band, 'none', ON), sshared_leg=(CUR, out/'w_sshared', 'leg', leg, 'none', ON))
    with ThreadPoolExecutor(jobs) as ex: R = {k: f.result() for k, f in {k: ex.submit(run, *v) for k, v in tasks.items()}.items()}
    R['shared_band'] = run(CUR, out/'w_shared', 'band', band)        # after the legacy state on the same work root
    # More than one FW=15 chunk (16 band wavelengths, 450-800 nm: the reduced grid converges there): a direct run with
    # -4.5 deg in its rows vs M3 rows from the stored chunked source (run_extra)
    e16 = np.linspace(450, 800, 17); w16 = ((e16[:-1]+e16[1:])/2).tolist()
    b16 = dict(wavelengths=16, wavelengths_nm=w16, bin_edges_nm=e16.tolist(), spectral_mode='band')
    with ThreadPoolExecutor(2) as ex:
        f1 = ex.submit(run, CUR,out/'w_16', 'src16', b16)
        f2 = ex.submit(run, CUR,out/'w_16d', 'dir16', dict(b16, solar_elevation_deg=[-3., -4.5, -6.]))
        R['src16'], R['dir16'] = f1.result(), f2.result()
    last = dict(wavelengths=1, wavelengths_nm=w16[-1:], bin_edges_nm=e16[-2:].tolist(), spectral_mode='band')
    R['last16'] = run(CUR, out/'w_last16', 'last16', last)
    check('n=16 last wavelength == standalone band bin (within tol)', (within_tol(R['src16'][..., -1:], R['last16'])))
    R['x16'] = run(CUR, out/'w_16', 'x16', b16, 'none', ('--source-out-name', 'src16', '--extra-elevations=-4.5'))
    R['sshared_band'] = run(CUR, out/'w_sshared', 'band', band, 'none', ON)
    hits = [l for l in (out/'w_sshared.band.log').read_text().splitlines() if '"attach_cache"' in l]
    check(f'§6-3 legacy table within tol of {LEGACY_BASELINE} tree', (within_tol(R['head'], R['cur'])))
    check('band differs from legacy (the mode is live)', R['band'].tobytes() != R['cur'].tobytes())
    r = R['E'][..., 1]/np.where(R['band'][..., 1] != 0, R['band'][..., 1], 1)
    check('§4-1 E post-multiply: bin 1 x2, bins 0/2 within tol', within_tol(R['E'][..., [0, 2]], R['band'][..., [0, 2]]) and
          np.allclose(r[R['band'][..., 1] != 0], 2, rtol=1e-6, atol=0), f'ratio {r.min():.7f}..{r.max():.7f}')
    check('§4-2 wavelength independence: sigma of bin 1 only moves bin 1', within_tol(R['sigma'][..., [0, 2]], R['band'][..., [0, 2]])
          and not np.array_equal(R['sigma'][..., 1], R['band'][..., 1]),
          'max |d| others %.3g' % np.max(abs(R['sigma'][..., [0, 2]]-R['band'][..., [0, 2]])))
    check('§4-6 cache: band after legacy on one work root == fresh band', (within_tol(R['shared_band'], R['band'])))
    check('n>15 wavelengths: M3 row from the 2-chunk stored source == direct run row (within tol)',
          R['src16'].shape[-1] == 16 and within_tol(R['x16'][list(np.round(xs, 3)).index(-4.5)] if (xs := load_hs(out/'w_16', 'x16')) is not None else None,
                                                         R['dir16'][1]), f"shapes {R['src16'].shape} {R['x16'].shape} {R['dir16'].shape}")
    check('§4-6 cache with the stratospheric layer (attach cache hit) == fresh', (within_tol(R['sshared_band'], R['sband']))
          and any('"result": "hit"' in l for l in hits), hits[0][hits[0].find('"result"'):][:40] if hits else 'no attach_cache phase')


def test_units():
    """Run the existing CLI unit checks under pytest as well."""
    FAILS.clear()
    units()
    assert not FAILS, ', '.join(FAILS)


def test_check_meta_contract():
    """spectral.check_meta_contract: only a *signed* table (meta has `signature`) must name `contract` v3/v4; every table's physics
    must be known, and a table without physics (written before B2) reads as the legacy identity."""
    import pytest      # not at module level: this file is also a CLI harness (`python -m sosfuse2.tests.test_spectral`)
    from sosfuse2.components import physics, optics_state
    for contract in ('v3', 'v4'):
        assert SP.check_meta_contract(dict(signature='s', contract=contract)) == contract
        assert SP.check_meta_contract(dict(signature='s', contract=contract, physics=physics(), optics_state=optics_state({}))) == contract
    for bad in (None, 'v2', 'merge49 contract text'):
        with pytest.raises(ValueError, match='E_PROVENANCE_CONTRACT'):
            SP.check_meta_contract(dict(signature='s', contract=bad))
    with pytest.raises(ValueError, match='E_PROVENANCE_CONTRACT'):
        SP.check_meta_contract(dict(signature='s'))                            # signed, `contract` key absent
    assert SP.check_meta_contract(dict(contract='derived table: merge of two signed tables')) == 'derived table: merge of two signed tables'
    assert SP.check_meta_contract(dict(id='x')) is None                        # unsigned and without a `contract` field
    assert SP.table_block(dict(contract='free text', settings=dict(spectral=None))) is None
    for meta in (dict(contract='free text', physics='lambda4_v0'), dict(signature='s', contract='v4', physics=dict(physics(), minor_absorbers='on'))):
        with pytest.raises(ValueError, match='E_PHYSICS_CONTRACT'):
            SP.check_meta_contract(meta)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', action='store_true'); p.add_argument('--out', type=Path); p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args()
    units()
    if a.runs: runs(a.out or Path(tempfile.mkdtemp(prefix='test_spectral_')), a.jobs)
    print('FAILED: '+', '.join(FAILS) if FAILS else 'ALL PASS'); sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
