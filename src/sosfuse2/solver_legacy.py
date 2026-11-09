"""Solver definitions loaded into per-call scopes by accepted.engine.definitions.

The loader executes only the top-level def/class nodes, in a fresh scope per state (filled by accepted.engine.create). The imports and
placeholders below never run there; they name the injected globals for static analysis (ruff, pyright)."""

from typing import Any

import numpy as np

R: Any = None  # Earth radius, km (engine.R)
sh: Any = None  # namespace of solver_sky.incoming/spatial_lookup (engine.create)


def ring_edges(r, chi, phi, heights, prepared=None):
    """View-plane intersection with the exact shadow-cylinder/shell ring.

    The night-side solar shadow has x²+y²=R², z<0. At shell R+h
    its boundary is the ring z=-sqrt((R+h)²-R²). Project each of its
    two intersections with the ray's azimuth plane onto local mu.
    These are real endpoints of illumination in each density layer.
    """
    if prepared is None:
        p, up, ex = sh.old.probe.frame(np.array(r), np.array(chi))
        ey = np.cross(up, ex)
        levels = [-np.sqrt(max(0, (R + h) ** 2 - R**2)) for h in heights]
    else:
        p, up, ex, ey, levels = prepared
    horizontal = np.cos(phi) * ex + np.sin(phi) * ey
    # Three scalar products in precisely np.cross's original order.
    normal = np.array(
        [
            up[1] * horizontal[2] - up[2] * horizontal[1],
            up[2] * horizontal[0] - up[0] * horizontal[2],
            up[0] * horizontal[1] - up[1] * horizontal[0],
        ]
    )
    amp = R * np.hypot(normal[0], normal[1])
    if amp < 1e-10:
        return []
    angle = np.arctan2(normal[1], normal[0])
    answer = []
    for z in levels:
        target = -normal[2] * z / amp
        if abs(target) > 1:
            continue
        delta = np.arccos(target)
        for az in (angle - delta, angle + delta):
            q = np.array([R * np.cos(az), R * np.sin(az), z])
            d = q - p
            length = np.sqrt(d @ d)
            if length < 1e-8:
                continue
            d /= length
            if d @ horizontal <= 1e-10:
                continue
            # Reject rings hidden behind the opaque Earth.
            b = p @ d
            disc = b * b - r * r + R * R
            hit = -b - np.sqrt(max(0, disc))
            if disc >= 0 and b < 0 and hit >= -1e-7 and hit < length - 1e-5:
                continue
            answer.append(float(d @ up))
    return answer


def incoming(r, chi, nmu, nphi, a=4.0, rings=True):
    mg = -np.sqrt(max(0, 1 - (R / r) ** 2))
    sa = np.sinh(a)
    u, wu = sh.old.probe.gl(nmu)
    mus = []
    phis = []
    weights = []
    sinchi = np.sin(chi)
    coschi = np.cos(chi)
    rp = r * sinchi
    prepared = None
    if rings:
        p, up, ex = sh.old.probe.frame(np.array(r), np.array(chi))
        ey = np.cross(up, ex)
        levels = [-np.sqrt(max(0, (R + h) ** 2 - R**2)) for h in (0.0, 1.2, 8.0, 25.0, 40.0, 80.0)]
        prepared = (p, up, ex, ey, levels)
    for j, phi in enumerate(np.linspace(0, np.pi, nphi + 1)):
        edges = [-1.0, 1.0, mg, 0.0, mg - 0.05234, mg + 0.05234, mg - 0.0061, mg + 0.0061]
        if rp > R + 1e-8 and abs(sinchi) > 1e-8:
            kk = R / np.sqrt(rp * rp - R * R)
            for sign in (-1, 1):
                t = np.cos(phi) * coschi / sinchi + sign * np.sin(phi) / (kk * sinchi)
                mu = t / np.sqrt(1 + t * t)
                st = np.sqrt(1 - mu * mu)
                dx = mu * sinchi - st * np.cos(phi) * coschi
                dy = -st * np.sin(phi)
                dz = mu * coschi + st * np.cos(phi) * sinchi
                tmin = -rp * dx / max(dx * dx + dy * dy, 1e-25)
                if abs(dx) + abs(dy) < 1e-10 or (tmin > 0 and r * coschi + tmin * dz < 0):
                    edges.append(mu)
        if phi < np.pi / 6:
            mu = coschi / np.sqrt(coschi**2 + np.cos(phi) ** 2 * sinchi**2)
            edges.extend([mu, mu - 0.012, mu + 0.012])
        if rings:
            edges += ring_edges(r, chi, phi, (0.0, 1.2, 8.0, 25.0, 40.0, 80.0), prepared)
        edges = np.unique(np.clip(edges, -1, 1))
        delta = edges - mg
        tt = np.sign(delta) * np.arcsinh(np.abs(delta) / np.where(delta < 0, 1 + mg, 1 - mg) * sa) / a
        # Every interval is independent. Keep interval-major / GL-node-minor
        # ordering and the exact scalar operation order of the accepted loop.
        keep = tt[1:] - tt[:-1] >= 1e-10
        left = tt[:-1][keep, None]
        right = tt[1:][keep, None]
        t = left + (right - left) * u
        den = np.where(t < 0, 1 + mg, 1 - mg)
        mus.append((mg + np.sign(t) * den * np.sinh(a * np.abs(t)) / sa).ravel())
        weights.append(
            (
                (right - left)
                * wu
                * den
                * a
                * np.cosh(a * np.abs(t))
                / sa
                * (2 * np.pi / nphi)
                * (0.5 if j in (0, nphi) else 1.0)
            ).ravel()
        )
        phis.append(np.full(t.size, phi))
    mu = np.concatenate(mus)
    phi = np.concatenate(phis)
    st = np.sqrt(np.maximum(0, 1 - mu * mu))
    return np.column_stack([st * np.cos(phi), st * np.sin(phi), mu]), np.concatenate(weights)
