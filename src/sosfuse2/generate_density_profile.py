#!/usr/bin/env python3
"""US1976 hydrostatics + IPRT ozone-shape asset, deterministically generated from the IPRT phase-E inputs.

The IPRT inputs are not redistributed: iprt.fetch() downloads them from libradtran.org into the work root and checks
pinned SHA-256 digests. The engine calls density.asset_path(), which generates the
asset into the work root on first use and checks density.ASSET_SHA256. Run as `python -m sosfuse2.generate_density_profile
[out.json]` to write the text explicitly (default: the work-root asset location).

Platform-independent arithmetic: the float64/numpy version of this generator produced
different bytes on arm64 and x86_64 — 6 values differed by <= 1.9e-15 relative, the last ulp of exp/log/pow
and of SIMD summation order — so a pinned sha256 could only match the platform that made the pin. Every transcendental
function, sum and interpolation below runs in `decimal` (libmpdec: exp/ln correctly rounded, power "almost always", and in
any case the same C algorithm in every CPython build, so the result does not depend on libm or SIMD width) at 40 digits and
is rounded to float64 once, at the end; json float repr is shortest-round-trip. Measured identical on arm64/CPython 3.14 and
x86_64/CPython 3.12; other interpreters (PyPy's _pydecimal) are not covered by that measurement.
The expressions match the original asset; measured against it the
values move by at most 4.7e-15 relative (air ratio, the float64 version accumulated several ulp over the layer products)
and 1.2e-15 (ozone), far below the rtol 1e-6 of prepared quantities."""

import argparse
import json
from decimal import Context, Decimal as D, DivisionByZero, InvalidOperation, Overflow, ROUND_HALF_EVEN, localcontext
from . import iprt, paths

PRECISION = 40
# An explicit, self-contained context (precision, rounding, exponent range, traps) used only inside build(): the caller's
# decimal context is neither consulted nor modified; a caller trapping Inexact must not break build().
CONTEXT = Context(
    prec=PRECISION,
    rounding=ROUND_HALF_EVEN,
    Emin=-999999,
    Emax=999999,
    traps=[InvalidOperation, DivisionByZero, Overflow],
)


def _rows(path):
    """(z_km, tau) rows of an IPRT two-column text file as exact decimals, top-down order as in the file."""
    rows = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        a, b = s.split()[:2]
        rows.append((D(a), D(b)))
    return rows


def _interp_log_linear(x, xs, ys):
    """numpy.interp semantics (xs increasing, clamp outside) on ln(ys), exponentiated."""
    lys = [y.ln() for y in ys]
    if x <= xs[0]:
        return lys[0].exp()
    if x >= xs[-1]:
        return lys[-1].exp()
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    t = (x - xs[lo]) / (xs[hi] - xs[lo])
    return (lys[lo] + t * (lys[hi] - lys[lo])).exp()


def _trapezoid(ys, xs):
    return sum(((ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i]) for i in range(len(xs) - 1)), D(0)) / 2


def build(offline=None):
    with localcontext(CONTEXT):
        return _build(offline)


def _build(offline):
    z = [D(i) / 2 for i in range(161)]  # 0..80 km, 0.5 km
    R0 = D('6356.766')
    H = [R0 * zi / (R0 + zi) for zi in z]  # geopotential height
    bounds = [D(b) for b in ('0', '11', '20', '32', '47', '51', '71', '84.852')]
    lapse = [D(L) for L in ('-6.5', '0', '1', '2.8', '0', '-2.8', '-2')]
    A = D(1000) * D('9.80665') * D('.0289644') / D('8.31432')
    Tbase = [D('288.15')]
    pbase = [D('101325')]
    for i, L in enumerate(lapse):
        dH = bounds[i + 1] - bounds[i]
        Tb = Tbase[-1]
        T = Tb + L * dH
        p = pbase[-1] * ((-A * dH / Tb).exp() if L == 0 else (Tb / T) ** (A / L))
        Tbase.append(T)
        pbase.append(p)
    k = D('1.380649e-23')
    number = []
    for h in H:
        i = max(j for j in range(len(lapse)) if bounds[j] <= h)  # searchsorted(side='right') - 1
        L = lapse[i]
        Tb = Tbase[i]
        T = Tb + L * (h - bounds[i])
        p = pbase[i] * ((-A * (h - bounds[i]) / Tb).exp() if L == 0 else (Tb / T) ** (A / L))
        number.append(p / (k * T))
    air = [n / number[0] for n in number]
    files, digests = iprt.inputs(offline)
    ray = _rows(files['tau_rayleigh_450nm_usstd.dat'])[::-1]
    oz = _rows(files['tau_absorption_320nm_usstd.dat'])[::-1]
    mids = [(oz[i][0] + oz[i + 1][0]) / 2 for i in range(len(oz) - 1)]
    beta = [oz[i][1] / (oz[i + 1][0] - oz[i][0]) for i in range(len(oz) - 1)]
    shape = [_interp_log_linear(zi, mids, beta) for zi in z]
    ozone_scale = D(300) * D('2.687e20') / (D(1000) * _trapezoid(shape, z))
    ozone = [s * ozone_scale for s in shape]
    tau550 = sum((t for zz, t in ray if zz < 80), D(0)) * (D(450) / D(550)) ** 4
    f = float
    return dict(
        version=1,
        altitude_km=[f(v) for v in z],
        air_number_density_ratio=[f(v) for v in air],
        ozone_number_density_m3=[f(v) for v in ozone],
        air_sea_level_number_density_m3=f(number[0]),
        rayleigh_tau550_0_80=f(tau550),
        ozone_reference_du=300,
        ozone_semantics='IPRT molecular absorption shape proxy normalized to 300 DU; not measured absolute US1976 ozone',
        source=dict(
            air='US Standard Atmosphere 1976 hydrostatic layer equations; docs/DENSITY_PROFILE.md',
            ozone='local IPRT E2 tau_absorption_320nm_usstd.dat lower-boundary layer tau / dz; midpoint log-linear interpolation',
            hashes={
                'results/iprt_ref/' + n: digests[n]
                for n in ('tau_rayleigh_450nm_usstd.dat', 'tau_absorption_320nm_usstd.dat')
            },
        ),
    )  # keys as in the 2026-10 asset


def text(offline=None):
    return json.dumps(build(offline), indent=2, allow_nan=False) + '\n'


if __name__ == '__main__':
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('out', nargs='?', type=Path, help='output JSON (default: the work-root asset location)')
    out = ap.parse_args().out or paths.work_root() / 'assets' / 'usstd76_profile.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text())
    print(out)
