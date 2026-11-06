#!/usr/bin/env python3
"""NO2 and O2-O2 (O4) absorption cross sections on the fine spectral grid.

NO2 is packaged (sosfuse2/assets/no2_vandaele1998_fine.json, CC BY 4.0 source). O4 is not shipped (HITRAN states no
redistribution terms): the prepare step writes it into the work root with build_o4().

Grid: the wavelengths of assets/spectra_fine.json (standard air, 360.0-840.0 nm, 0.1 nm). Values are 0.1 nm cell means
(the mean over [x-0.05, x+0.05] of the source's linear interpolant, a filtered spectrum like the solar column), because
the NO2 structure is finer than 0.1 nm: point values would move BM45 band means by up to 6.5e-3, cell means by 9.2e-4.
Rules:
  NO2 220 K, Vandaele et al. 1998 (BIRA-IASB): no2c_97.dat below 660.0 nm, no2cp_97.dat column 6 (NO2 0.16 Torr, the
    lower-pressure one of the two columns reaching 830 nm) above 666.0 nm, unscaled, and a linear blend C_97 -> CP6 over
    660-666 nm (CP6 reads 8-16 % below C_97 in the overlap; no rescaling of the published column).
    Column 1 of both files is 1/lambda in air (Edlen, 15 C, 101325 Pa; BIRA dataset page): used as the air wavelength.
  O4, HITRAN2024 CIA main O2-O2_2024.cia (vacuum wavenumbers, converted with air_wavelength.standard_n):
    ref 54 (Finkenzeller & Volkamer 2022) 293 K up to its last point (499.861 nm air = 20000 cm^-1), ref 32 (Thalman &
    Volkamer 2013) 293 K from there to 600.0 nm (HITRAN recommends ref 32 from 509.33 nm: 499.86-509.33 nm is this
    product's connection rule), ref 32 287 K from 600.0 nm to the end of that set (653.84 nm); zero elsewhere.
    The O2 A- and B-band CIA sets (refs 4, 30) are left to the O2 line work (R4b). Negative cell means (baseline noise)
    are set to 0 and counted.
Raw files are not shipped; their SHA-256 are checked. Units: NO2 m^2 molecule^-1, O4 m^5 molecule^-2.
usage: python -B -m sosfuse2.generate_minor_absorbers --bira DIR --out FILE   [--cia FILE --o4-out FILE]
"""
import argparse, hashlib, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
from sosfuse2.air_wavelength import standard_n

ASSETS = Path(__file__).with_name('assets')
DEFAULT = ASSETS/'no2_vandaele1998_fine.json'
RULES_VERSION = 'b2-s2-v2'   # v2: missing-marker policy, interior gaps and unknown markers refused
BIRA_SHA = {'no2c_97.dat': '8fa43c685c070b7b1d19b24e4b8cbbc002fc4df2d6128feb8e8ddc2b4864f43d',
            'no2cp_97.dat': 'ce734d755bcd5953374e1157859247e469613ba7f7f212edfbf1c140456f57d3'}
CIA_SHA = '0146acfe5c1414b8deaa56be9a63159154479d5804d1615855888a2e1efd4961'
CIA_URL = 'https://hitran.org/data/CIA/main/O2-O2_2024.cia'
NO2_BLEND_NM = (660.0, 666.0)
O4_SEGMENTS = (('54', 293.0, 'blue', -np.inf, 'end54'), ('32', 293.0, 'blue', 'end54', 600.0), ('32', 287.0, 'red', 600.0, np.inf))
CELL = 0.1


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path, want):
    got = sha256(path)
    if got != want:
        raise SystemExit(f'E_SOURCE_SHA: {Path(path).name} sha256 {got} != {want}')
    return got


class Source:
    """Linear interpolant of (lam, sig) on [lam[0], lam[-1]]; integral over any [a, b] (zero outside the support)."""
    def __init__(self, lam, sig):
        o = np.argsort(lam); self.x = np.asarray(lam, float)[o]; self.y = np.asarray(sig, float)[o]
        if np.any(np.diff(self.x) <= 0) or not np.isfinite(self.y).all():
            raise SystemExit('E_SOURCE_GRID: increasing wavelengths and finite values required')
        self.F = np.concatenate(([0.], np.cumsum(np.diff(self.x)*.5*(self.y[1:]+self.y[:-1]))))

    def cum(self, t):
        t = np.clip(np.asarray(t, float), self.x[0], self.x[-1])
        i = np.clip(np.searchsorted(self.x, t, side='right')-1, 0, len(self.x)-2)
        yt = self.y[i]+(self.y[i+1]-self.y[i])*(t-self.x[i])/(self.x[i+1]-self.x[i])
        return self.F[i]+(t-self.x[i])*.5*(self.y[i]+yt)

    def integral(self, a, b):
        return self.cum(b)-self.cum(a)           # cum clips to the support: zero contribution outside it


def piecewise_cells(grid, pieces):
    """Cell means of a spectrum made of (source, lo, hi) pieces: each piece contributes its integral over [lo, hi) only."""
    a, b = grid-CELL/2, grid+CELL/2; total = np.zeros_like(grid)
    for src, lo, hi in pieces:
        lo_, hi_ = np.maximum(a, lo), np.minimum(b, hi)
        m = hi_ > lo_
        total[m] += src.integral(lo_[m], hi_[m])
    return total/CELL


MISSING = {'no2c_97.dat': (-999.,), 'no2cp_97.dat': (0., -999.)}   # BIRA: -999 = missing (dataset page); CP_97 pads with 0


def contiguous(lam, val, missing, what):
    """Drop missing markers at the ends only: an interior missing row would otherwise be bridged by the interpolant
    (E_SOURCE_GAP). Any value <= -1e-10 left after that is a marker this policy does not know (E_SOURCE_SENTINEL);
    small negative values are measurement noise and are kept."""
    lam, val = np.asarray(lam, float), np.asarray(val, float)
    ok = np.isfinite(val) & ~np.isin(val, missing)
    idx = np.flatnonzero(ok)
    if not idx.size:
        raise SystemExit(f'E_SOURCE_EMPTY: {what}')
    if np.any(np.diff(idx) > 1):
        k = idx[np.flatnonzero(np.diff(idx) > 1)[0]]
        raise SystemExit(f'E_SOURCE_GAP: {what} has missing rows inside its valid range after {lam[k]:.4f} nm')
    lam, val = lam[idx[0]:idx[-1]+1], val[idx[0]:idx[-1]+1]
    if np.any(val <= -1e-10):
        raise SystemExit(f'E_SOURCE_SENTINEL: {what} holds {val.min():g} (not a cross section, not a declared missing marker)')
    return lam, val


def bira(path, column):
    d = np.loadtxt(path)
    lam, val = contiguous(1e7/d[:, 0], d[:, column-1], MISSING[Path(path).name], f'{Path(path).name} column {column}')
    return Source(lam, val)


def cia_blocks(path):
    lines = Path(path).read_text().splitlines(); i = 0
    while i < len(lines):
        f = lines[i].split(); n = int(f[3]); ref = lines[i].rstrip().split()[-1]
        data = np.array([[float(x) for x in s.split()[:2]] for s in lines[i+1:i+1+n]])
        yield dict(lo=float(f[1]), hi=float(f[2]), T=float(f[4]), ref=ref, nu=data[:, 0], sig=data[:, 1]); i += n+1


def o4_sources(path):
    out = {}
    for b in cia_blocks(path):
        if b['ref'] not in ('32', '54'):
            continue
        part = 'red' if b['hi'] < 16700 else 'blue'
        lam_vac = 1e7/b['nu']; key = (b['ref'], b['T'], part)
        lam_air, sig = contiguous(lam_vac/standard_n(lam_vac), b['sig'], (), f'HITRAN ref {key}')
        if key in out:
            raise SystemExit(f'E_CIA_BLOCKS: duplicate set {key}')
        out[key] = Source(lam_air, sig)
    return out


def blend(a, b, lo, hi):
    """Linear blend a -> b over [lo, hi] on the union of both sample grids (weight (x-lo)/(hi-lo))."""
    x = np.union1d(a.x[(a.x > lo) & (a.x < hi)], b.x[(b.x > lo) & (b.x < hi)]); x = np.concatenate(([lo], x, [hi]))
    if x[0] < max(a.x[0], b.x[0]) or x[-1] > min(a.x[-1], b.x[-1]):
        raise SystemExit('E_NO2_BLEND: both sources must cover the blend window')
    w = (x-lo)/(hi-lo)
    return Source(x, (1-w)*np.interp(x, a.x, a.y)+w*np.interp(x, b.x, b.y))


def no2_pieces_of(c97, cp6):
    lo, hi = NO2_BLEND_NM
    return [(c97, -np.inf, lo), (blend(c97, cp6, lo, hi), lo, hi), (cp6, hi, np.inf)]


def o4_pieces_of(src):
    end54 = float(src[('54', 293.0, 'blue')].x[-1])
    val = lambda v: end54 if v == 'end54' else v
    return [(src[(r, T, part)], val(lo), val(hi)) for r, T, part, lo, hi in O4_SEGMENTS]


def band_check(grid, cells, pieces, edges):
    """BM45 band means: trapezoid of the cell-mean grid (the engine's band rule) vs the exact integral of the pieces."""
    rel = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        x = np.concatenate(([lo], grid[(grid > lo) & (grid < hi)], [hi])); y = np.interp(x, grid, cells)
        eng = np.trapezoid(y, x)/(hi-lo)
        exact = sum(float(s.integral(max(lo, l), min(hi, h))) for s, l, h in pieces if min(hi, h) > max(lo, l))/(hi-lo)
        rel.append((eng, exact))
    rel = np.array(rel); big = rel[:, 1] > 1e-2*rel[:, 1].max()
    return float(np.max(np.abs(rel[big, 0]/rel[big, 1]-1))), int(big.sum())


def fine_grid():
    fine = json.loads((ASSETS/'spectra_fine.json').read_text())
    grid = np.asarray(fine['wavelength_nm'], float)
    if grid.shape != (4801,) or grid[0] != 360.0 or grid[-1] != 840.0:
        raise SystemExit('E_FINE_GRID: spectra_fine.json must hold 360.0-840.0 nm at 0.1 nm')
    return fine, grid


def finish(grid, pieces, scale):
    """Cell means, negative cells set to 0 (counted), BM45 self-check, and the stored 7-digit SI values."""
    from sosfuse2.spectral import PRODUCT_EDGES
    cells = piecewise_cells(grid, pieces); neg = int((cells < 0).sum()); cells = np.maximum(cells, 0.)
    rel, n = band_check(grid, cells, pieces, np.asarray(PRODUCT_EDGES, float))
    return [float(f'{v*scale:.7g}') for v in cells], cells, neg, dict(bm45_band_vs_exact_max_rel=rel, bands_checked=n,
                                                                   note='bands above 1 % of the largest band mean')


def common(fine):
    gen = Path(__file__).read_bytes()
    return dict(wavelength_reference=fine['wavelength_reference'], wavelength_nm=fine['wavelength_nm'],
                definition=('0.1 nm cell means [x-0.05, x+0.05] of the source linear interpolant on the spectra_fine.json grid, '
                            '7 significant digits; band means integrate this grid with the trapezoid rule (spectral.py), '
                            'point values interpolate it (a filtered spectrum, like the solar column).'),
                generator=dict(module='sosfuse2.generate_minor_absorbers', rules_version=RULES_VERSION,
                               sha256=hashlib.sha256(gen).hexdigest()))


def build_no2(bira_dir):
    """The packaged NO2 asset (CC BY 4.0 source, shipped)."""
    fine, grid = fine_grid()
    shas = {n: checked(Path(bira_dir)/n, s) for n, s in BIRA_SHA.items()}
    c97, cp6 = bira(Path(bira_dir)/'no2c_97.dat', 3), bira(Path(bira_dir)/'no2cp_97.dat', 6)
    values, cells, neg, chk = finish(grid, no2_pieces_of(c97, cp6), 1e-4)
    return dict(format='no2-fine-v1', **common(fine), no2_220K=values,
                units={'wavelength_nm': 'nm (air)', 'no2_220K': 'm^2 molecule^-1 (0.1 nm cell mean)'},
                rules=dict(temperature_K=220, below_nm=NO2_BLEND_NM[0], below='no2c_97.dat column 3',
                           above_nm=NO2_BLEND_NM[1], above='no2cp_97.dat column 6 (0.16 Torr NO2), unscaled',
                           blend='linear C_97 -> CP6 over the window, on the union of both sample grids',
                           columns_doc='BIRA dataset page: CP_97 columns 3-7 = p < 0.015, 0.03, 0.04, 0.16, 0.25 Torr',
                           negative_cells_set_to_zero=neg),
                source=dict(name='Vandaele et al. 1998, NO2 220 K (BIRA-IASB)', files=shas,
                            dataset='Robert, S., Vandaele, A. C., Hermans, C. (2026). NO2 absorption cross sections at different '
                                    'temperatures and pressures (Version 1). Royal Belgian Institute for Space Aeronomy. '
                                    'https://doi.org/10.18758/6b80spv8',
                            page='https://data.aeronomie.be/dataset/cross-sections-and-spectroscopic-molecular-parameters-about-no2',
                            licence='CC BY 4.0', reference='Vandaele, A. C. et al. (1998). JQSRT 59, 171-184.',
                            wavelength='column 1 = 1/lambda in air (Edlen 15 C, 101325 Pa), used as the air wavelength',
                            unit_conversion='cm^2 -> m^2 (x 1e-4)'),
                self_check=dict(chk, max_cm2=float(cells.max())))


def build_o4(cia):
    """The O4 table, generated in the user's work root by the prepare step (HITRAN terms unconfirmed: not shipped)."""
    fine, grid = fine_grid()
    cia_sha = checked(cia, CIA_SHA)
    src = o4_sources(cia); pieces = o4_pieces_of(src)
    values, cells, neg, chk = finish(grid, pieces, 1e-10)
    i577 = int(np.argmin(np.abs(grid-577.0)))
    return dict(format='o4-fine-v1', **common(fine), o4=values,
                units={'wavelength_nm': 'nm (air)', 'o4': 'm^5 molecule^-2 (0.1 nm cell mean; absorption coefficient = o4 * n_O2^2)'},
                rules=dict(segments=[dict(hitran_ref=r, temperature_K=T, block=part, from_nm=None if not np.isfinite(lo) else lo,
                                          to_nm=None if not np.isfinite(hi) else hi)
                                     for (r, T, part, _, _), (_, lo, hi) in zip(O4_SEGMENTS, pieces)],
                           connection='ref 32 over 499.861-509.334 nm (HITRAN table 1 recommends ref 32 from 19628 cm^-1)',
                           end_nm=float(src[('32', 287.0, 'red')].x[-1]), outside='zero (no extrapolation)',
                           excluded='O2-O2 A-band (HITRAN ref 4) and B-band (ref 30) CIA: with the O2 line work',
                           negative_cells_set_to_zero=neg),
                source=dict(name='HITRAN2024 collision-induced absorption, O2-O2 main', file='O2-O2_2024.cia', sha256=cia_sha,
                            url=CIA_URL,
                            licence='no licence text on hitran.org/cia or /citepolicy (checked 2026-11-04); not redistributed',
                            references=['Gordon, I. E. et al. (2026). The HITRAN2024 molecular spectroscopic database. JQSRT 353, 109807.',
                                        'Terragni, J. et al. (2025). Collision induced absorption in HITRAN2024. JQSRT 347, 109631.',
                                        'Finkenzeller, H., Volkamer, R. (2022). JQSRT 279, 108063 (HITRAN ref 54).',
                                        'Thalman, R., Volkamer, R. (2013). PCCP 15, 15371 (HITRAN ref 32).'],
                            wavelength='vacuum wavenumber -> air (air_wavelength.standard_n)',
                            unit_conversion='cm^5 molecule^-2 -> m^5 molecule^-2 (x 1e-10)'),
                self_check=dict(chk, cell_577nm_cm5=float(cells[i577])))


def main(argv=None):
    p = argparse.ArgumentParser(description='maintenance: regenerate the packaged NO2 asset; --cia/--o4-out writes the O4 '
                                            'table (the prepare step calls build_o4 for users)')
    p.add_argument('--bira', type=Path, help='folder with no2c_97.dat and no2cp_97.dat')
    p.add_argument('--out', type=Path, help=f'NO2 output JSON; the packaged asset is {DEFAULT.name}')
    p.add_argument('--cia', type=Path, help='HITRAN O2-O2_2024.cia')
    p.add_argument('--o4-out', type=Path, help='O4 output JSON (never inside the package)')
    a = p.parse_args(argv)
    if not ((a.bira and a.out) or (a.cia and a.o4_out)):
        p.error('give --bira and --out, or --cia and --o4-out')
    for src, dst, fn in ((a.bira, a.out, build_no2), (a.cia, a.o4_out, build_o4)):
        if src and dst:
            d = fn(src)
            dst.write_text(json.dumps(d, indent=1, allow_nan=False)+'\n')
            print(json.dumps(dict(format=d['format'], rules=d['rules'], self_check=d['self_check'], sha256=sha256(dst)), indent=1))


if __name__ == '__main__':
    main()
