"""Solver definitions loaded into per-call scopes by accepted.engine.definitions."""
# ruff: noqa: F821 -- globals are injected by the per-call loader.


def incoming(r, chi, qscale):
    mg = -np.sqrt(max(0, 1 - (R / r) ** 2))
    a = 4.0
    sa = np.sinh(a)
    nphi = 32 * qscale
    u, wu = old.probe.gl(3 * qscale)
    allmu = []
    allphi = []
    allw = []
    for j, phi in enumerate(np.linspace(0, np.pi, nphi + 1)):
        edges = [-1.0, 1.0, mg, 0.0, mg - 0.05234, mg + 0.05234, mg - 0.0061, mg + 0.0061]
        rp = r * np.sin(chi)
        if rp > R + 1e-8 and abs(np.sin(chi)) > 1e-8:
            kk = R / np.sqrt(rp * rp - R * R)
            for sign in (-1, 1):
                t = np.cos(phi) * np.cos(chi) / np.sin(chi) + sign * np.sin(phi) / (kk * np.sin(chi))
                mu = t / np.sqrt(1 + t * t)
                st = np.sqrt(1 - mu * mu)
                dx = mu * np.sin(chi) - st * np.cos(phi) * np.cos(chi)
                dy = -st * np.sin(phi)
                dz = mu * np.cos(chi) + st * np.cos(phi) * np.sin(chi)
                tmin = -rp * dx / max(dx * dx + dy * dy, 1e-25)
                if abs(dx) + abs(dy) < 1e-10 or (tmin > 0 and r * np.cos(chi) + tmin * dz < 0):
                    edges.append(mu)
        if phi < np.pi / 6:
            mu = np.cos(chi) / np.sqrt(np.cos(chi) ** 2 + np.cos(phi) ** 2 * np.sin(chi) ** 2 + 1e-30)
            edges.extend([mu - 0.012, mu, mu + 0.012])
        edges = np.unique(np.clip(edges, -1, 1))
        delta = edges - mg
        tt = np.sign(delta) * np.arcsinh(np.abs(delta) / np.where(delta < 0, 1 + mg, 1 - mg) * sa) / a
        mus = []
        weights = []
        for left, right in zip(tt[:-1], tt[1:]):
            if right - left < 1e-10:
                continue
            t = left + (right - left) * u
            den = np.where(t < 0, 1 + mg, 1 - mg)
            mus.extend(mg + np.sign(t) * den * np.sinh(a * np.abs(t)) / sa)
            weights.extend((right - left) * wu * den * a * np.cosh(a * np.abs(t)) / sa)
        wp = 2 * np.pi / nphi * (0.5 if j in (0, nphi) else 1.0)
        allmu.extend(mus)
        allphi.extend([phi] * len(mus))
        allw.extend(np.array(weights) * wp)
    mu = np.array(allmu)
    phi = np.array(allphi)
    st = np.sqrt(np.maximum(0, 1 - mu * mu))
    return np.column_stack([st * np.cos(phi), st * np.sin(phi), mu]), np.array(allw)


def spatial_lookup(q, r, d, h, chi):
    height = np.clip(r - R, 0, Z)
    ch = np.arccos(np.clip(q[..., 2] / r, -1, 1))
    ih = np.clip(np.searchsorted(h, height, side='right') - 1, 0, len(h) - 2)
    ic = np.clip(np.searchsorted(chi, ch, side='right') - 1, 0, len(chi) - 2)
    ah = np.clip((height - h[ih]) / (h[ih + 1] - h[ih]), 0, 1)
    ac = np.clip((ch - chi[ic]) / (chi[ic + 1] - chi[ic]), 0, 1)
    mu = np.clip(np.sum(q * d, -1) / r, -1, 1)
    cs = q[..., 2] / r
    den = np.sqrt(np.maximum(0, (1 - cs * cs) * (1 - mu * mu)))
    cp = np.divide(d[..., 2] - cs * mu, den, out=np.ones_like(mu), where=den > 1e-12)
    return (ih * len(chi) + ic).astype(np.int32), ah, ac, mu, np.clip(cp, -1, 1)
