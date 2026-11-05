#!/usr/bin/env python3
"""D0 CPU successive-orders probe; no product imports or writes.

Rayleigh's phase is exactly quadratic.  The source at (r, chi) is therefore
represented without angular truncation by Q = integral L(omega) omega omega^T
and J(v) = beta * 3/(16*pi) * (trace(Q) + v^T Q v). Reflection symmetry leaves
Qxx,Qyy,Qzz,Qxz in the local (sun projection, perpendicular, radial) frame.
The discrete transport operator is assembled once, then reused for all orders.
"""
import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

import numpy as np
from sasktran_tiebreak import R, Z, H, B0, H0, DIRS, SunTable, method_a

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/sos_probe'
HS = (2, -3, -6, -9, -12)
C = 3.0 / (16.0 * np.pi)
ALBEDO = 0.0  # canonical tiebreak: Lambert boundary contributes exactly zero


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=1, allow_nan=False) + '\n')


def gl(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1) / 2, w / 2


def density(r, wavelength=550.0):
    return B0 * (550.0 / wavelength) ** 4 * np.exp(-np.maximum(r - R, 0) / H)


def grid(scale):
    h = np.unique(np.r_[np.linspace(0, np.sqrt(Z), 28 * scale + 1) ** 2, H0])
    chi = np.unique(np.r_[np.linspace(0, 80, 16 * scale + 1),
                           np.linspace(80, 85, 5 * scale + 1),
                           np.linspace(85, 108, 58 * scale + 1),
                           np.linspace(108, 120, 12 * scale + 1),
                           np.linspace(120, 180, 12 * scale + 1)])
    return h, np.deg2rad(chi)


def frame(r, chi):
    up = np.stack([np.sin(chi), np.zeros_like(chi), np.cos(chi)], -1)
    ex = np.stack([-np.cos(chi), np.zeros_like(chi), np.sin(chi)], -1)
    return r[..., None] * up, up, ex


def angular(r, chi, nmu, nphi):
    """Split at the actual Earth limb and local horizon, never a fixed mu grid.
    Squared Gauss coordinates resolve grazing rays on either side of the limb.
    Reflection in the solar plane lets phi in [0,pi] carry twice its weight.
    """
    u, w = gl(nmu)
    mg = -np.sqrt(max(0.0, 1 - (R / r) ** 2))
    mu = np.r_[mg - (1 + mg) * u ** 2, mg * (1 - u), u ** 2]
    wm = np.r_[2 * (1 + mg) * u * w, -mg * w, 2 * u * w]
    phi = (np.arange(nphi) + .5) * np.pi / nphi
    mu, phi = np.meshgrid(mu, phi, indexing='ij')
    wm = np.broadcast_to(wm[:, None], mu.shape).ravel() * (2 * np.pi / nphi)
    mu, phi = mu.ravel(), phi.ravel()
    dx = np.sqrt(np.maximum(1 - mu * mu, 0)) * np.cos(phi)
    dy = np.sqrt(np.maximum(1 - mu * mu, 0)) * np.sin(phi)
    p, up, ex = frame(np.array(r), np.array(chi))
    ey = np.cross(up, ex)
    d = mu[:, None] * up + dx[:, None] * ex + dy[:, None] * ey
    moment = np.stack([dx * dx, dy * dy, mu * mu, dx * mu], -1) * wm[:, None]
    return np.broadcast_to(p, d.shape), d, moment


def boundaries(p, d):
    """Exact shell exit, opaque-ground hit, tangent, and cylindrical shadow hits.
    Sun is +z; only the back half of the Earth cylinder is shadow.
    """
    b = np.sum(p * d, -1)
    pp = np.sum(p * p, -1)
    end = -b + np.sqrt(np.maximum(b * b - pp + (R + Z) ** 2, 0))
    disc = b * b - pp + R * R
    hit = -b - np.sqrt(np.maximum(disc, 0))
    ground = (disc >= 0) & (b < 0) & (hit >= -1e-7)
    end = np.where(ground, np.maximum(hit, 0), end)
    end = np.maximum(end, 0)
    tangent = np.where((-b > 0) & (-b < end), -b, end)
    aa = d[:, 0] ** 2 + d[:, 1] ** 2
    bb = p[:, 0] * d[:, 0] + p[:, 1] * d[:, 1]
    cc = p[:, 0] ** 2 + p[:, 1] ** 2 - R * R
    ds = bb * bb - aa * cc
    roots = []
    for sign in (-1, 1):
        root = (-bb + sign * np.sqrt(np.maximum(ds, 0))) / np.maximum(aa, 1e-30)
        valid = (aa > 1e-20) & (ds >= 0) & (root > 0) & (root < end) & (p[:, 2] + root * d[:, 2] < 0)
        roots.append(np.where(valid, root, end))
    return np.sort(np.stack([np.zeros_like(end), tangent, *roots, end], -1), axis=-1)


def segments(p, d, n, wavelength):
    """Gauss quadrature, clustered at the lower endpoint of each split interval.
    Transmission uses the ordered quadrature optical-depth prefix on that ray.
    No source interpolation across a direct-sun shadow discontinuity occurs.
    """
    edges = boundaries(p, d)
    u, wu = gl(n)
    tau = np.zeros(len(d))
    for k in range(edges.shape[1] - 1):
        a, b = edges[:, k], edges[:, k + 1]
        active = np.flatnonzero(b - a > 1e-9)
        if not len(active):
            continue
        a, b = a[active, None], b[active, None]
        pa, da = p[active], d[active]
        ra = np.sum((pa + a * da) ** 2, -1)
        rb = np.sum((pa + b * da) ** 2, -1)
        low_start = (ra <= rb)[:, None]
        t = np.where(low_start, a + (b - a) * u ** 2,
                     b - (b - a) * (1 - u) ** 2)
        weight = np.where(low_start, 2 * (b - a) * u * wu,
                          2 * (b - a) * (1 - u) * wu)
        q = pa[:, None, :] + t[..., None] * da[:, None, :]
        r = np.sqrt(np.sum(q * q, -1))
        optical = density(r, wavelength) * weight
        prefix = tau[active, None] + np.cumsum(optical, -1) - optical / 2
        att = optical * np.exp(-prefix)
        tau[active] += np.sum(optical, -1)
        yield active, q, r, att


def source_coeff(q, r, d):
    mu = np.sum(q * d[:, None, :], -1) / r
    cs = np.clip(q[..., 2] / r, -1, 1)
    st = np.sqrt(np.maximum(1 - cs * cs, 0))
    dx = np.divide(d[:, None, 2] - cs * mu, st, out=np.zeros_like(mu), where=st > 1e-10)
    dy2 = np.maximum(1 - mu * mu - dx * dx, 0)
    return C * np.stack([1 + dx * dx, 1 + dy2, 1 + mu * mu, 2 * dx * mu], -1)


def corners(q, r, h, chi):
    altitude = np.clip(r - R, h[0], h[-1])
    angle = np.arccos(np.clip(q[..., 2] / r, -1, 1))
    i = np.clip(np.searchsorted(h, altitude, side='right') - 1, 0, len(h) - 2)
    j = np.clip(np.searchsorted(chi, angle, side='right') - 1, 0, len(chi) - 2)
    a = (altitude - h[i]) / (h[i + 1] - h[i])
    b = np.clip((angle - chi[j]) / (chi[j + 1] - chi[j]), 0, 1)
    base = i * len(chi) + j
    ids = np.stack([base, base + len(chi), base + 1, base + len(chi) + 1], -1)
    weights = np.stack([(1-a)*(1-b), a*(1-b), (1-a)*b, a*b], -1)
    return ids, weights


class Solver:
    def __init__(self, scale=1, wavelength=550.0):
        self.scale, self.wavelength = scale, float(wavelength)
        self.h, self.chi = grid(scale)
        self.nmu, self.nphi, self.npath = 8 * scale, 16 * scale, 24 * scale
        self.size = len(self.h) * len(self.chi)
        digest = kernel_hash()
        self.folder = OUT / f'kernel_{digest}_x{scale}_{wavelength:g}'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.folder / 'meta.json'
        self.sun = None
        self.cache_hit = self.meta_path.exists()
        if self.cache_hit:
            self.meta = json.loads(self.meta_path.read_text())
            self.q1 = np.load(self.folder / 'q1.npy')
        else:
            self.build()

    def sun_table(self):
        if self.sun is None:
            t = time.perf_counter()
            self.sun = SunTable(nh=321 * self.scale, nx=1201 * self.scale,
                                n=256 * self.scale)
            self.sun.tau *= (550.0 / self.wavelength) ** 4
            print(f'SUN x{self.scale} {time.perf_counter()-t:.3f}s', flush=True)
        return self.sun

    def build(self):
        t0 = time.perf_counter()
        sun = self.sun_table()
        q1 = np.zeros((self.size, 4))
        blocks = []
        block_rows, block_cols, block_values = [], [], []
        block_start = 0
        last = time.perf_counter()
        for row in range(self.size):
            ih, ic = divmod(row, len(self.chi))
            p, d, mom = angular(R + self.h[ih], self.chi[ic], self.nmu, self.nphi)
            dense = np.zeros((self.size, 16))
            for active, q, r, att in segments(p, d, self.npath, self.wavelength):
                # L1 is always evaluated analytically with the exact shadow mask.
                t_sun = sun.T(q, np.array([0., 0., 1.]))
                l1seg = np.sum(att * t_sun, -1) * C * (1 + d[active, 2] ** 2)
                q1[row] += np.sum(mom[active] * l1seg[:, None], 0)
                ids, interp = corners(q, r, self.h, self.chi)
                coeff = source_coeff(q, r, d[active]) * att[..., None]
                outer = (mom[active, None, :, None] * coeff[..., None, :]).reshape(-1, 16)
                ids = ids.ravel()
                interp = interp.reshape(-1, 4)
                for j in range(16):
                    dense[:, j] += np.bincount(ids, weights=(outer[:, j, None] * interp).ravel(), minlength=self.size)
            cols = np.flatnonzero(np.any(dense != 0, axis=-1))
            block_rows.append(np.full(len(cols), row - block_start, dtype=np.int32))
            block_cols.append(cols.astype(np.int32))
            block_values.append(dense[cols].reshape(-1, 4, 4).astype(np.float32))
            if (row + 1) % 128 == 0 or row + 1 == self.size:
                name = f'block_{len(blocks):04d}.npz'
                np.savez(self.folder / name, row=np.concatenate(block_rows),
                         col=np.concatenate(block_cols), value=np.concatenate(block_values))
                blocks.append(dict(file=name, start=block_start, count=row+1-block_start))
                block_rows, block_cols, block_values = [], [], []
                block_start = row + 1
            if time.perf_counter() - last > 30 or row + 1 == self.size:
                print(f'BUILD x{self.scale} {row+1}/{self.size} {time.perf_counter()-t0:.1f}s', flush=True)
                last = time.perf_counter()
        np.save(self.folder / 'q1.npy', q1)
        self.q1 = q1
        self.meta = dict(scale=self.scale, wavelength_nm=self.wavelength,
                         nr=len(self.h), nchi=len(self.chi), nmu_per_piece=self.nmu,
                         nphi_half=self.nphi, npath_per_segment=self.npath,
                         h_km=self.h.tolist(), chi_deg=np.rad2deg(self.chi).tolist(),
                         build_seconds=time.perf_counter()-t0, blocks=blocks)
        save_json(self.meta_path, self.meta)

    def apply(self, field):
        out = np.zeros_like(field)
        for block in self.meta['blocks']:
            z = np.load(self.folder / block['file'])
            values = np.einsum('kab,kb->ka', z['value'], field[z['col']], optimize=False)
            local = np.stack([np.bincount(z['row'], weights=values[:, j], minlength=block['count'])
                              for j in range(4)], -1)
            out[block['start']:block['start']+block['count']] = local
            z.close()
        return out

    def query(self, hs, elevations, azimuths, field=None):
        elevations, azimuths = np.broadcast_arrays(elevations, azimuths)
        shape = elevations.shape
        el, az = np.deg2rad(elevations.ravel()), np.deg2rad(azimuths.ravel())
        p0, up, ex = frame(np.array(R + H0), np.deg2rad(np.array(90 - hs)))
        ey = np.cross(up, ex)
        d = (np.sin(el)[:, None] * up + np.cos(el)[:, None] *
             (np.cos(az)[:, None] * ex + np.sin(az)[:, None] * ey))
        result = np.zeros(len(d))
        for start in range(0, len(d), 256):
            dd = d[start:start+256]
            pp = np.broadcast_to(p0, dd.shape)
            part = np.zeros(len(dd))
            for active, q, r, att in segments(pp, dd, self.npath * 2, self.wavelength):
                if field is None:
                    sun = self.sun_table()
                    source = C * (1 + dd[active, 2] ** 2)[:, None] * sun.T(q, np.array([0., 0., 1.]))
                else:
                    ids, weights = corners(q, r, self.h, self.chi)
                    qq = np.sum(field[ids] * weights[..., None], axis=-2)
                    source = np.sum(qq * source_coeff(q, r, dd[active]), -1)
                part[active] += np.sum(att * source, -1)
            result[start:start+len(dd)] = part
        return result.reshape(shape)

    def solve(self, hs_values=HS):
        t0 = time.perf_counter()
        el = np.array([e for _, e, _ in DIRS])
        az = np.array([a for _, _, a in DIRS])
        orders = {str(hs): [self.query(hs, el, az)] for hs in hs_values}
        field = self.q1.copy()
        sum_source = np.zeros_like(field)
        increments = []
        beta_grid = np.repeat(density(R+self.h, self.wavelength), len(self.chi))
        for order in range(2, 13):
            sum_source += field
            worst = 0.0
            for hs in hs_values:
                arr = orders[str(hs)]
                arr.append(self.query(hs, el, az, field))
                fraction = arr[-1] / np.maximum(np.sum(arr, axis=0), 1e-300)
                worst = max(worst, float(np.max(fraction)))
            # Infinity norm of the entire angular source, using its exact largest
            # eigenvalue; also protect the much fainter twilight observer rays.
            def source_peak(q):
                trace = q[:, :3].sum(-1)
                eigen = np.maximum(q[:, 1], (q[:, 0]+q[:, 2]+
                                   np.sqrt((q[:, 0]-q[:, 2])**2+4*q[:, 3]**2))/2)
                return float(np.max(beta_grid * C * (trace + eigen)))
            source_relative = source_peak(field) / max(source_peak(sum_source), 1e-300)
            increments.append(dict(order=order, maximum_observer_increment=worst,
                                   global_source_norm_increment=source_relative))
            print(f'ORDER x{self.scale} n={order} increment={worst:.6g} source={source_relative:.6g} {time.perf_counter()-t0:.2f}s', flush=True)
            if order >= 3 and max(worst, source_relative) < .005:
                break
            if order < 12:
                field = self.apply(field)
        np.save(self.folder / 'sum_source.npy', sum_source)
        return orders, dict(iteration_and_query_seconds=time.perf_counter()-t0,
                            increments=increments, orders_used=order,
                            capped=max(worst, source_relative) >= .005, cache_hit=self.cache_hit)


def kernel_hash():
    functions = (gl, density, grid, frame, angular, boundaries, segments,
                 source_coeff, corners, Solver.build, Solver.sun_table)
    payload = '\n'.join(inspect.getsource(fn) for fn in functions)
    payload += repr((R, Z, H, B0, H0, C, ALBEDO, 8, 16, 24, 'float32-operator'))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def ratios(numerator, denominator):
    return [float(a / b) if b != 0 else None for a, b in zip(numerator, denominator)]


def rel_change(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.divide(np.abs(a-b), np.maximum(np.abs(b), 1e-300)).tolist()


def reference(sun):
    old = json.loads((ROOT / 'results/sasktran_probe/tiebreak.json').read_text())
    deep_path = OUT / 'method_a_deep.json'
    if deep_path.exists():
        deep = json.loads(deep_path.read_text())
    else:
        deep = {}
        for hs in (-9, -12):
            t = time.perf_counter()
            small, large = [], []
            for _, el, az in DIRS:
                small.append(method_a(hs, el, az, sun))
                large.append(method_a(hs, el, az, sun, nv=128, nmu=32, nphi=64, ni=128))
            deep[str(hs)] = dict(L1=np.array(large)[:, 0].tolist(), L2=np.array(large)[:, 1].tolist(),
                                 base=np.array(small).tolist(), seconds=time.perf_counter()-t,
                                 self_conv=rel_change(small, large))
            print(f'A hs={hs} {deep[str(hs)]["seconds"]:.2f}s', flush=True)
            save_json(deep_path, deep)
    if not all(deep[str(hs)].get('L1_refinement_version') == 3 for hs in (-9, -12)):
        # A's original midpoint rule can miss the very short illuminated portion
        # at TOA. Keep A itself as referee, increasing only its view resolution.
        # Angular parameters do not affect method_a's L1; discard the cheap L2.
        t = time.perf_counter()
        critical = [(-9, 5), (-9, 6), (-12, 1)]
        history = deep['-9'].get('L1_critical_history', {}).copy()
        for hs in (-9, -12):
            key = str(hs)
            deep[key].setdefault('L1_original_nv128', deep[key]['L1'])
            if not deep[key].get('L1_refined'):
                deep[key]['L1'] = [method_a(hs, el, az, sun, nv=2048, nmu=1, nphi=1, ni=1)[0]
                                   for _, el, az in DIRS]
        previous = None
        stable = 0
        for nv in (2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144):
            if str(nv) in history:
                values = history[str(nv)]['values']
            else:
                values = []
                for hs, i in critical:
                    _, el, az = DIRS[i]
                    values.append(method_a(hs, el, az, sun, nv=nv, nmu=1, nphi=1, ni=1)[0])
            delta = max(rel_change(previous, values)) if previous is not None else None
            history[str(nv)] = dict(values=values, relative_change=delta)
            stable = stable + 1 if delta is not None and delta < .002 else 0
            print(f'A1 REFINE nv={nv} relative_change={delta}', flush=True)
            previous = values
            if stable >= 2:
                break
        for (hs, i), value in zip(critical, values):
            deep[str(hs)]['L1'][i] = value
        for hs in (-9, -12):
            deep[str(hs)].update(L1_refined=True, L1_nv_noncritical=2048,
                                L1_nv_critical=nv, L1_critical_history=history,
                                L1_critical_cases=[[h, DIRS[i][0]+str(DIRS[i][1])] for h,i in critical],
                                L1_refinement_seconds=time.perf_counter()-t,
                                L1_refinement_converged=stable >= 2, L1_refinement_version=3)
        save_json(deep_path, deep)
    refs = {}
    for hs in HS:
        key = str(hs)
        if key in old['hs']:
            a = old['hs'][key]['A']
            refs[key] = dict(L1=a.get('L1_double') or a['L1'], L2=a.get('L2_double') or a['L2'],
                             pt=old['hs'][key]['pt'])
        else:
            refs[key] = deep[key]
    return refs


def numeric(wavelength=550.0):
    if wavelength != 550:
        raise ValueError('--numeric comparison references are only defined at 550 nm')
    t0 = time.perf_counter()
    base = Solver(1, wavelength)
    low, low_time = base.solve()
    # Persist base output before the larger deterministic run.
    save_json(OUT / 'base_orders.json', {k: [a.tolist() for a in v] for k, v in low.items()})
    fine = Solver(2, wavelength)
    high, high_time = fine.solve()
    refs = reference(fine.sun_table())
    result = dict(dirs=[f'{n}{e}/{a}' for n, e, a in DIRS], wavelength_nm=wavelength,
                  units='radiance / incident solar irradiance; sr^-1',
                  atmosphere=dict(R_km=R, top_km=Z, H_km=H, B0_per_km=B0, observer_km=H0,
                                  albedo=ALBEDO, aerosol=0, ozone=0),
                  discretization='exact Rayleigh tensor moments; local Earth-limb angular split; exact shadow ray split',
                  grids={name: s.meta for name, s in [('base', base), ('double', fine)]},
                  timing=dict(base=low_time, double=high_time), hs={}, failures=[], excluded=[],
                  zero_reference_policy='0/0 ratio=null; agreement requires both values exactly zero')
    for hs in HS:
        key = str(hs)
        hi, lo, ref = high[key], low[key], refs[key]
        o1, o2, o3 = hi[:3]
        ms = np.sum(hi[1:], axis=0)
        ms3 = o2 + o3
        low_ms = np.sum(lo[1:], axis=0)
        conv = dict(o1=rel_change(lo[0], o1), o2=rel_change(lo[1], o2),
                    o3=rel_change(lo[2], o3), total_ms=rel_change(low_ms, ms))
        maxconv = np.max(np.array(list(conv.values())), axis=0).tolist()
        table = dict(L1_over_A=ratios(o1, ref['L1']), L2_over_A=ratios(o2, ref['L2']),
                     MS_over_pt=None, MS3_over_pt=None, self_conv=maxconv,
                     self_conv_by_order=conv, MS_status=None, MS3_status=None)
        for order, values, targets in ((1, o1, ref['L1']), (2, o2, ref['L2'])):
            for i, (v, target) in enumerate(zip(values, targets)):
                good = (v == 0) if target == 0 else abs(v / target - 1) <= .03
                if not good:
                    result['failures'].append(dict(hs=hs, direction=result['dirs'][i], check=f'L{order}',
                                                   ratio=float(v/target) if target else None, value=float(v), reference=target))
        for i, value in enumerate(maxconv):
            if value > .02:
                result['failures'].append(dict(hs=hs, direction=result['dirs'][i], check='self_conv', relative=value))
        if 'pt' in ref:
            for tag, values, pt_key, status_key in [('MS', ms, 'd0', 'MS_status'), ('MS3', ms3, 'd3', 'MS3_status')]:
                pt = ref['pt'][pt_key]
                mean = np.array(pt['multiple'])
                se = np.array(pt['multiple_se']) / mean
                rr = np.array(values) / mean
                table['MS_over_pt' if tag == 'MS' else 'MS3_over_pt'] = rr.tolist()
                table[tag + '_se_relative'] = se.tolist()
                table[tag + '_tolerance'] = np.maximum(.05, 2 * se).tolist()
                statuses = []
                for i in range(len(rr)):
                    if se[i] > .15:
                        statuses.append('판정 불가')
                        result['excluded'].append(dict(hs=hs, direction=result['dirs'][i], check=tag, se_relative=float(se[i])))
                    elif abs(rr[i] - 1) <= max(.05, 2 * se[i]):
                        statuses.append('합격')
                    else:
                        statuses.append('불합격')
                        result['failures'].append(dict(hs=hs, direction=result['dirs'][i], check=tag,
                                                       ratio=float(rr[i]), tolerance=float(max(.05, 2 * se[i]))))
                table[status_key] = statuses
        result['hs'][key] = dict(sos=dict(o1=o1.tolist(), o2=o2.tolist(), o3=o3.tolist(),
                                         total_ms=ms.tolist(), orders_used=len(hi),
                                         orders=[x.tolist() for x in hi]),
                                 base_sos=dict(o1=lo[0].tolist(), o2=lo[1].tolist(), o3=lo[2].tolist(),
                                               total_ms=low_ms.tolist(), orders_used=len(lo)),
                                 A=dict(L1=ref['L1'], L2=ref['L2']), table=table)
        fmt = lambda x: ['—' if v is None else round(v, 5) for v in x]
        print(f'TABLE hs={hs} L1={fmt(table["L1_over_A"])} L2={fmt(table["L2_over_A"])} '
              f'MS={fmt(table["MS_over_pt"]) if table["MS_over_pt"] else "—"} '
              f'self_max={max(maxconv):.6g}', flush=True)
    result['elapsed_seconds'] = time.perf_counter()-t0
    result['verdict'] = '합격' if not result['failures'] else '불합격'
    save_json(OUT / 'sos_tiebreak.json', result)
    report(result)
    print(f'NUMERIC seconds={result["elapsed_seconds"]:.3f} failures={len(result["failures"])} excluded={len(result["excluded"])}', flush=True)
    print('VERDICT ' + result['verdict'], flush=True)
    return result


def report(result):
    lines = ['# 자체 구형 successive-orders 탐침', f'판정: **{result["verdict"]}**. D0, 제품 코드 변경 없음.',
             '정본: scripts/sasktran_tiebreak.py의 순수 레일리·R6371 km·상단80 km·H8 km·관측자0.1 km·알베도0.',
             '550 nm, B0=0.013558 /km; 출력은 태양 입사량으로 나눈 복사휘도(sr^-1).',
             '방향 순서: sun2 / sun10 / sun30 / anti10 / anti30 / side10 / zenith89.',
             'A는 기존 +2/−3/−6의 두 배 구적; −9/−12 L2는 새 method_a 두 배 구적.',
             '깊은 박명 A L1은 nv2048, 작은 양수3칸은 최대262144까지 재측정; 내역 method_a_deep.json.',
             '레일리 원천은 Qxx/Qyy/Qzz/Qxz로 정확히 표현; (r,χ) 쌍선형 보간, J1은 직달광에서 직접.',
             '각 구적은 실제 지구 접선 μg와 μ=0에서 분할; φ는 태양면 반사 대칭으로 0…π.',
             '시선은 지표·접점·정확한 태양 그림자 원통 경계에서 분할, 각 구간 하단 쪽 제곱 Gauss 좌표.',
             '시선 투과율은 구적 광학두께 prefix. Lambert 알베도0이므로 지면 원천은 정확히0.',
             '파장은 함수 인자; 레일리 λ⁻⁴. 에어로졸·오존·편광은 이 기준 대기에 없음.',
             '|격자|r|χ|μ(3구간 각각)|φ(반구)|시선(분할구간 각각)|사전계산 s|반복+35조회 s|',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('base', 'double'):
        g, t = result['grids'][name], result['timing'][name]
        lines.append(f'|{name}|{g["nr"]}|{g["nchi"]}|{g["nmu_per_piece"]}|{g["nphi_half"]}|{g["npath_per_segment"]}|{g["build_seconds"]:.3f}|{t["iteration_and_query_seconds"]:.3f}|')
    lines += ['χ85…108° 간격: 기본23/58°, 두 배23/116°. 고도는 제곱 격자에0.1 km 추가.',
              '각 셀 = L1/A / L2/A. 0/0은 —이며 두 값이0일 때만 일치로 판정.',
              '|태양고도|sun2|sun10|sun30|anti10|anti30|side10|zenith89|', '|---|---|---|---|---|---|---|---|']
    f = lambda v: '—' if v is None else f'{v:.4f}'
    for hs in HS:
        t = result['hs'][str(hs)]['table']
        cells = [f(a)+'/'+f(b) for a,b in zip(t['L1_over_A'], t['L2_over_A'])]
        lines.append('|'+str(hs)+'|'+'|'.join(cells)+'|')
    lines += ['|태양고도|MS/PT (7방향)|MS3/PT (7방향)|자기수렴 최대|차수|', '|---|---|---|---:|---:|']
    for hs in HS:
        row = result['hs'][str(hs)]
        t = row['table']
        ms = ' / '.join(f(v)+('†' if status == '판정 불가' else '') for v,status in zip(t['MS_over_pt'],t['MS_status'])) if t['MS_over_pt'] else '정답 없음'
        ms3 = ' / '.join(f(v)+('†' if status == '판정 불가' else '') for v,status in zip(t['MS3_over_pt'],t['MS3_status'])) if t['MS3_over_pt'] else '정답 없음'
        lines.append(f'|{hs}|{ms}|{ms3}|{100*max(t["self_conv"]):.3f}%|{row["sos"]["orders_used"]}|')
    lines += [f'합격선 고정: L1/L2±3%, MS/MS3 max(5%,2×상대SE), 모든 차수 자기수렴≤2%. 실패 {len(result["failures"])}칸/항목.',
              '자기수렴은 L1·L2·L3·전체MS의 기본/두 배 상대차 중 최댓값; 세부 값은 JSON.',
              '† 판정 불가, SE>15% 제외: '+ ('; '.join(f'{x["hs"]}° {x["direction"]} {x["check"]} SE={100*x["se_relative"]:.2f}%' for x in result['excluded']) or '없음')+'.',
              '실패: '+('; '.join(f'{x["hs"]}° {x["direction"]} {x["check"]}={x.get("ratio",x.get("relative"))}' for x in result['failures']) or '없음')+'.',
              '정지: 전 공간·전 방향 원천의 최대값 노름 및 5고도×7방향 증분/전체 모두<0.5%; 최소3차·상한12.',
              '원천 최대값은 Rayleigh 텐서의 최대 고유값으로 계산. 상한 도달 여부는 JSON timing.capped.',
              '추측: 잔여 PT 차이는 PT 두 시드 SE 추정 또는 SOS 3차 이상 구적 오차일 수 있음; 원인 미분리.',
              f'수치 단 실행 실측 {result["elapsed_seconds"]:.3f}s; 현재 캐시 사용 base={result["timing"]["base"]["cache_hit"]}, double={result["timing"]["double"]["cache_hit"]}.',
              'sparse CPU 연산자 캐시는 results/sos_probe/kernel_*; 사전계산 시간은 최초 조립 실측을 유지.',
              '한계: 순수 레일리 점광원 기준; 에어로졸 전방 피크와 유한 태양 원반은 미확인.',
              '미확인: −9/−12° 전 차수 외부 정답, 15파장 비용, 제품 LUT 통합.',
              '파노라마: 수치 단 합격 때만 --pano 실행. 불합격이면 산출하지 않음.']
    original = json.loads((ROOT / 'results/sasktran_probe/tiebreak.json').read_text())['hs']['-6']
    a2 = original['A']['L2_double'][5]
    pt2 = original['pt']['d2']['multiple'][5]
    lines.append(f'진단: 기존 −6° 측면10° PT 2차/A={pt2/a2:.6f}, SOS 2차/A={result["hs"]["-6"]["table"]["L2_over_A"][5]:.6f}; PT3차 이상 원인 단정 안 함.')
    deep = json.loads((OUT / 'method_a_deep.json').read_text())
    last = list(deep['-9']['L1_critical_history'].values())[-1]['relative_change']
    lines.append(f'A 작은 L1 3칸 마지막 단계 상대 변화={100*last:.6f}%; 연속2단계<0.2% 확인={deep["-9"]["L1_refinement_converged"]}.')
    box_path = OUT / 'box_diagnostic.json'
    if box_path.exists():
        box = json.loads(box_path.read_text())['-6']
        lines.append(f'−6° 측면10° 2×2픽셀/점시선: MS={box["MS_box_over_point"][5]:.8f}, MS3={box["MS3_box_over_point"][5]:.8f} (box_diagnostic.json).')
    cold_path = OUT / 'cold_sos_tiebreak.json'
    if cold_path.exists():
        cold = json.loads(cold_path.read_text())
        times = cold['timing']
        lines.append(f'최초 조립 후 반복+35조회: 기본{times["base"]["iteration_and_query_seconds"]:.3f}s, 두 배{times["double"]["iteration_and_query_seconds"]:.3f}s; 전체명령{cold["elapsed_seconds"]:.3f}s.')
    lines += ['캐시 재호출의 반복+조회 시간에는 태양 투과율 표 재생성도 포함. 일부 조립 중 A 재측정 병행.',
              '하늘 한 장의 시간은 미측정: 수치 단 불합격으로 파노라마 실행을 생략.']
    (ROOT / 'docs/SOS_PROBE.md').write_text('\n'.join(lines)+'\n')


def pano(wavelength=550.0):
    path = OUT / 'sos_tiebreak.json'
    if not path.exists() or json.loads(path.read_text())['verdict'] != '합격':
        print('PANO skipped: 수치 단 합격 필요', flush=True)
        return
    solver = Solver(2, wavelength)
    # Row centers cover 90...0 degrees with 0.9-degree spacing; solar azimuth at center.
    el = 90 - (np.arange(100) + .5) * .9
    az = np.arange(360) + .5 - 180
    el, az = np.meshgrid(el, az, indexing='ij')
    timings = {}
    for hs in (0, -3, -6, -9, -12):
        t0 = time.perf_counter()
        solver.solve(hs_values=(hs,))
        source = np.load(solver.folder / 'sum_source.npy')
        t1 = time.perf_counter()
        radiance = solver.query(hs, el, az) + solver.query(hs, el, az, source)
        name = f'pano{wavelength:g}_e{hs}.npy'
        np.save(OUT / name, radiance.astype(np.float32))
        timings[str(hs)] = dict(source_iteration_seconds=t1-t0, lookup_seconds=time.perf_counter()-t1,
                                total_seconds=time.perf_counter()-t0, file=name)
        print(f'PANO hs={hs} {timings[str(hs)]["total_seconds"]:.3f}s', flush=True)
    save_json(OUT / f'pano{wavelength:g}_timing.json', timings)
    doc = ROOT / 'docs/SOS_PROBE.md'
    with doc.open('a') as stream:
        stream.write(f'파노라마 360×100 float32, 0.9°×1°, 태양 가운데; {wavelength:g} nm 단색, 태양 입사량당.\n')
        stream.write('장당 원천반복+조회 실측(s): '+', '.join(f'{hs}°={v["total_seconds"]:.3f}' for hs,v in timings.items())+'.\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--numeric', action='store_true')
    ap.add_argument('--pano', action='store_true')
    ap.add_argument('--wavelength', type=float, default=550.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.numeric:
        numeric(args.wavelength)
    if args.pano:
        pano(args.wavelength)
    if not args.numeric and not args.pano:
        ap.error('choose --numeric and/or --pano')


if __name__ == '__main__':
    main()
