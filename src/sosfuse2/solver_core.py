"""Solver, ray packing and cell construction of the CPU reference engine; state is read from the engine context."""

import time

import numpy as np


def basis_at(q):
    r = np.linalg.norm(q, axis=-1)
    up = q / r[..., None]
    cos = up[..., 2]
    ex = np.array([0.0, 0.0, 1.0]) - cos[..., None] * up
    l = np.linalg.norm(ex, axis=-1)
    ex = np.divide(ex, l[..., None], out=np.zeros_like(ex), where=l[..., None] > 1e-12)
    ex = np.where(
        (l < 1e-12)[..., None],
        np.stack([-np.where(cos >= 0, 1.0, -1.0), np.zeros_like(cos), np.zeros_like(cos)], -1),
        ex,
    )
    return np.stack([ex, np.cross(up, ex), up], -1)


def pack(ctx, p, d, h, ch, sun, n):
    rows = []
    pts = []
    bases = []
    lits = []
    for active, q, r, ds, prefix in ctx.b.sh.old.geometry_segments(p, d, n):
        base, ah, ac, mu, cp = ctx.b.sh.spatial_lookup(q, r, d[active, None, :], h, ch)
        sc, lit = sun.lookup(q, r)
        rho = np.linalg.norm(q[..., :2], axis=-1)
        cx = np.divide(q[..., 0], rho, out=np.ones_like(rho), where=rho > 1e-12)
        sx = np.divide(q[..., 1], rho, out=np.zeros_like(rho), where=rho > 1e-12)
        z = np.stack(
            [
                ah,
                ac,
                mu,
                cp,
                ds,
                r - ctx.b.R,
                *[prefix[..., k] for k in range(5)],
                *[sc[..., k] for k in range(5)],
                cx,
                sx,
            ],
            -1,
        )
        rows.append(np.repeat(active, ds.shape[1]))
        pts.append(z.reshape(-1, 18).astype('f4'))
        bases.append(base.ravel())
        lits.append(lit.ravel())
    B = basis_at(p)
    if not rows:
        return dict(
            off=np.zeros(len(d) + 1, 'i8'),
            base=np.empty(0, 'i4'),
            point=np.empty((0, 18), 'f4'),
            lit=np.empty(0, 'u1'),
            nu=d[:, 2].copy(),
            target=np.ascontiguousarray(B),
            dloc=np.einsum('nij,ni->nj', B, d),
        )
    row = np.concatenate(rows)
    order = np.argsort(row, kind='stable')
    return dict(
        off=np.r_[0, np.cumsum(np.bincount(row, minlength=len(d)))].astype('i8'),
        base=np.concatenate(bases)[order].astype('i4'),
        point=np.concatenate(pts)[order],
        lit=np.concatenate(lits)[order].astype('u1'),
        nu=d[:, 2].copy(),
        target=np.ascontiguousarray(B),
        dloc=np.einsum('nij,ni->nj', B, d),
    )


class Solver:
    def __init__(self, ctx, g, kernel, wavelengths, mode=0, delta=0.03):
        self.ctx = ctx
        self.geo, self.kernel, self.mode, self.delta = g, kernel, mode, delta
        self.folder = g.folder
        self.wl = np.asarray(wavelengths, float)
        self.W = len(self.wl)
        self.V = 2 * self.W
        self.active = np.ones(self.V, 'i4')
        self.ext = np.ascontiguousarray(self.ctx.b.sh.old.coefficients(self.wl, self.ctx.b.sh.NORMAL))
        self.asca = np.ascontiguousarray(self.ext[1] * self.ctx.b.sh.NORMAL['ssa'])
        self.hr = np.exp(-g.h / 8)
        self.ha = self.ctx.AER_PROFILE(g.h)
        self.solar = np.interp(self.wl, self.ctx.b.sh.old.SPEC['wavelength_nm'], self.ctx.b.sh.old.SPEC['solar'])
        self.levels = np.array([l for l in range(33) for m in range(l + 1)])
        self.rf = (self.levels[:6] == 0) + 0.1 * (self.levels[:6] == 2)
        self.af = self.ctx.AER_G**self.levels
        self.dummy = np.empty(0, 'f4')
        t = time.perf_counter()
        if self.ctx.prepare_ground:
            self.prepare_ground()
        self.prepare_seconds = time.perf_counter() - t
        self.timings = dict(transport_seconds=0.0, projection_seconds=0.0, ground_seconds=0.0)

    def prepare_ground(self):
        ds = []
        vs = []
        off = [0]
        g = self.geo
        self.gm = np.zeros((g.ns, 4, self.W), 'f4')
        for cell in range(g.ns):
            ih, ic = divmod(cell, len(g.chi))
            loc, w = self.ctx.b.sh.incoming(self.ctx.b.R + g.h[ih], g.chi[ic], 1)
            p, up, ex = self.ctx.b.sh.old.probe.frame(np.array(self.ctx.b.R + g.h[ih]), g.chi[ic])
            ey = np.cross(up, ex)
            d = loc[:, 0, None] * ex + loc[:, 1, None] * ey + loc[:, 2, None] * up
            p = np.broadcast_to(p, d.shape)
            end = self.ctx.b.sh.old.probe.boundaries(p, d)[:, -1]
            q = p + end[:, None] * d
            r = np.linalg.norm(q, axis=-1)
            col = self.ctx.b.sh.old.total_columns(p, d, 24)
            sc, lit = g.sun.lookup(q, np.maximum(self.ctx.b.R, r))
            ch = np.arccos(np.clip(q[:, 2] / r, -1, 1))
            value = (
                self.ctx.b.sh.NORMAL['albedo']
                / np.pi
                * np.maximum(0, np.cos(ch))[:, None]
                * np.exp(-(sc + col) @ self.ext)
                * lit[:, None]
                * (r < self.ctx.b.R + 1e-5)[:, None]
                * w[:, None]
            )
            take = np.any(value > 0, axis=1)
            ds.append(loc[take])
            vs.append(value[take])
            off.append(off[-1] + take.sum())
            factor = np.stack(
                [
                    (1 - loc[:, 0] ** 2) / 2,
                    (1 - loc[:, 1] ** 2) / 2,
                    (1 - loc[:, 2] ** 2) / 2,
                    -loc[:, 0] * loc[:, 2] / 2,
                ],
                -1,
            )
            self.gm[cell] = (factor.T @ value) * (2 * self.ctx.C * self.ext[0])[None, :]
        self.goff = np.array(off, 'i8')
        self.gdir = np.ascontiguousarray(np.concatenate(ds), 'f4')
        self.gval = np.ascontiguousarray(np.concatenate(vs), 'f4')

    def radiance(self, i, field, boundary=None):
        z = self.geo.cell(i, self)
        t = time.perf_counter()
        L = self.kernel.transport(z, self, field) + z['first']
        self.timings['transport_seconds'] += time.perf_counter() - t
        if boundary is not None:
            E = (1 - z['end_fraction'][:, None, None]) * boundary[z['end_index']] + z['end_fraction'][
                :, None, None
            ] * boundary[z['end_index'] + 1]
            v = z['loc']
            P = np.stack([1 - v[:, 0] ** 2, 1 - v[:, 1] ** 2, 1 - v[:, 2] ** 2, -v[:, 0] * v[:, 2]], -1) / 2
            L += E[..., None] * z['ground_tr'][:, :, None, None] * P[:, None, None, :]
        return z, L

    def boundary(self, field):
        t = time.perf_counter()
        out = np.empty((len(self.geo.chi), self.W, 2))
        for i in range(len(out)):
            z, L = self.radiance(i, field)
            I = L[..., :3].sum(-1)
            out[i] = (
                self.ctx.b.sh.NORMAL['albedo']
                / np.pi
                * np.einsum('n,nwl->wl', z['weight'] * np.maximum(0, z['loc'][:, 2]), I)
            )
        self.timings['ground_seconds'] += time.perf_counter() - t
        return out

    def apply(self, field):
        boundary = self.boundary(field)
        out = np.empty_like(field)
        t = time.perf_counter()
        last = t
        for i in range(self.geo.ns):
            z, L = self.radiance(i, field, boundary)
            tt = time.perf_counter()
            I = L[..., :3].sum(-1)
            ch = self.geo.chi[i % len(self.geo.chi)]
            v = z['loc']
            glob = np.column_stack(
                [v[:, 2] * np.sin(ch) - v[:, 0] * np.cos(ch), -v[:, 1], v[:, 2] * np.cos(ch) + v[:, 0] * np.sin(ch)]
            )
            Y = self.kernel.basis(glob, 32)
            moment = Y.T @ (I.reshape(len(I), self.V) * z['weight'][:, None])
            matrix = np.einsum('n,nwlp->pwl', z['weight'], L).reshape(4, self.V)
            out[i, :4] = matrix * np.repeat(2 * self.ctx.C * self.ext[0], 2)[None, :]
            out[i, 4:10] = moment[:6] * self.rf[:, None] * np.repeat(self.ext[0], 2)[None, :]
            out[i, 10:] = moment * self.af[:, None] * np.repeat(self.asca, 2)[None, :]
            inactive = self.active == 0
            out[i, :, inactive] = field[i, :, inactive]
            self.timings['projection_seconds'] += time.perf_counter() - tt
            if time.perf_counter() - last > 30:
                print(
                    f'FULL STREAM {self.geo.name} cell={i}/{self.geo.ns} seconds={time.perf_counter() - t:.1f} RSS={self.ctx.b.peak() / 1e9:.3f}GB',
                    flush=True,
                )
                last = time.perf_counter()
        return out

    def query(self, h, el, az, field=None, first=False):
        ee, aa = np.broadcast_arrays(el, az)
        shape = ee.shape
        e, z = np.deg2rad(ee.ravel()), np.deg2rad(aa.ravel())
        p, up, ex = self.ctx.b.sh.old.probe.frame(np.array(self.ctx.b.R + self.ctx.b.sh.OBS), np.deg2rad(90 - h))
        ey = np.cross(up, ex)
        d = np.sin(e)[:, None] * up + np.cos(e)[:, None] * (np.cos(z)[:, None] * ex + np.sin(z)[:, None] * ey)
        out = np.empty((len(d), self.W, 2))
        field = self.ms if field is None else field
        active = self.active.copy()
        self.active[:] = 1
        for j in range(0, len(d), 64):
            dd = d[j : j + 64]
            raw = self.geo.pack(np.broadcast_to(p, dd.shape), dd, getattr(self.geo, 'query_n', 24))
            out[j : j + len(dd)] = self.kernel.transport(raw, self, field, first)[..., :3].sum(-1)
        self.active = active
        return out.reshape(*shape, self.W, 2) * self.solar[:, None]

    def boxes(self, field, first=False):
        u, w = np.polynomial.legendre.leggauss(4)
        ww = (w[:, None] * w[None, :] / 4).ravel()
        el = np.array([e + de for _, e, z in self.ctx.b.DIRS for de in u for dz in u])
        az = np.array([z + dz for _, e, z in self.ctx.b.DIRS for de in u for dz in u])
        return np.stack(
            [
                np.einsum('dnwl,n->dwl', self.query(h, el, az, field, first).reshape(7, 16, self.W, 2), ww)
                for h in self.ctx.b.HS
            ]
        )


def build(self, i, s):
    t = time.perf_counter()
    ih, ic = divmod(i, len(self.chi))
    loc, w = self.ctx.b.legacy.incoming(
        self.ctx.b.R + self.h[ih], self.chi[ic], self.cfg['nmu'], self.cfg['nphi'], 4.0, True
    )
    self.meta['solid_angle_max_relative'] = max(self.meta['solid_angle_max_relative'], abs(w.sum() / (4 * np.pi) - 1))
    p, up, ex = self.ctx.b.sh.old.probe.frame(np.array(self.ctx.b.R + self.h[ih]), self.chi[ic])
    ey = np.cross(up, ex)
    d = loc[:, 0, None] * ex + loc[:, 1, None] * ey + loc[:, 2, None] * up
    p = np.broadcast_to(p, d.shape)
    end = self.ctx.b.sh.old.probe.boundaries(p, d)[:, -1]
    qe = p + end[:, None] * d
    re = np.linalg.norm(qe, axis=-1)
    col = self.ctx.b.sh.old.total_columns(p, d, 24)
    tr = np.exp(-col @ s.ext) * (re < self.ctx.b.R + 1e-05)[:, None]
    ec = np.arccos(np.clip(qe[:, 2] / re, -1, 1))
    j = np.clip(np.searchsorted(self.chi, ec, side='right') - 1, 0, len(self.chi) - 2)
    frac = np.clip((ec - self.chi[j]) / (self.chi[j + 1] - self.chi[j]), 0, 1)
    self.meta['geometry_seconds'] += time.perf_counter() - t
    chunks = []
    first = []
    off = [0]
    for start in range(0, len(loc), 256):
        t = time.perf_counter()
        z = self.pack(p[start : start + 256], d[start : start + 256], self.cfg['path_n'])
        self.meta['geometry_seconds'] += time.perf_counter() - t
        t = time.perf_counter()
        first.append(s.kernel.transport(z, s, s.dummy, True).astype('f4'))
        self.meta['first_seconds'] += time.perf_counter() - t
        if self.compact:
            t = time.perf_counter()
            q = z['point']
            ds = q[:, 4] * np.exp(-q[:, 6:11].astype(float) @ s.ext[:, 0])
            z['point'] = np.column_stack([q[:, 0], q[:, 1], ds, q[:, 5], q[:, 16], q[:, 17]]).astype('f4')
            self.meta['fold_seconds'] += time.perf_counter() - t
        off.extend((z['off'][1:] + off[-1]).tolist())
        chunks.append(z)
    z = dict(
        off=np.array(off, 'i8'),
        **{key: np.concatenate([v[key] for v in chunks]) for key in ('point', 'base', 'lit', 'nu', 'target', 'dloc')},
        loc=loc,
        weight=w,
        first=np.concatenate(first),
        ground_tr=tr,
        end_index=j.astype('i4'),
        end_fraction=frac,
        end_columns=col,
        hits_ground=re < self.ctx.b.R + 1e-05,
        wavelengths=s.wl.copy(),
        delta=np.array(s.delta),
    )
    self.meta['incoming_rays'] += len(w)
    self.meta['points'] += len(z['point'])
    self.meta['largest_block_bytes'] = max(self.meta['largest_block_bytes'], sum((v.nbytes for v in z.values())))
    return z
