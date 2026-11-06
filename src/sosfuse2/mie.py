"""Bohren–Huffman sphere coefficients and scalar lognormal ensemble, NumPy only.

Units: radius/wavelength in micrometres, cross sections in um^2. m=n+i*k
uses the passive (k>=0) convention. chi_l are coefficients of P_l, whereas
SH scattering eigenvalues are chi_l/(2l+1). No polarization matrix is used by
the ensembles; scattering_matrix() serves independent polarization checks only.
"""

import numpy as np


def coefficients(x, m=1.43 + 0j):
    x = float(x)
    m = complex(m)
    if not np.isfinite(x) or x <= 0 or not np.isfinite([m.real, m.imag]).all() or m.real <= 0 or m.imag < 0:
        raise ValueError('E_MIE_INPUT: positive x,n and passive k>=0 required')
    stop = int(x + 4 * np.cbrt(x) + 2)
    top = max(stop, int(abs(m * x))) + 30
    z = m * x
    D = np.zeros(top + 1, complex)
    for n in range(top, 0, -1):
        D[n - 1] = n / z - 1 / (D[n] + n / z)
    p0 = np.sin(x)
    p1 = x * x * (1 / 3 - x * x / 30 + x**4 / 840 - x**6 / 45360) if x < 0.01 else np.sin(x) / x - np.cos(x)
    c0 = np.cos(x)
    c1 = np.cos(x) / x + np.sin(x)
    a = np.empty(stop, complex)
    b = np.empty(stop, complex)
    for n in range(1, stop + 1):
        xi = p1 - 1j * c1
        xi0 = p0 - 1j * c0
        da = D[n] / m + n / x
        db = m * D[n] + n / x
        a[n - 1] = (da * p1 - p0) / (da * xi - xi0)
        b[n - 1] = (db * p1 - p0) / (db * xi - xi0)
        p0, p1 = p1, (2 * n + 1) * p1 / x - p0
        c0, c1 = c1, (2 * n + 1) * c1 / x - c0
    return a, b


def efficiencies(x, m=1.43 + 0j):
    a, b = coefficients(x, m)
    n = np.arange(1, len(a) + 1, dtype=float)
    qe = 2 / x**2 * np.sum((2 * n + 1) * (a + b).real)
    qs = 2 / x**2 * np.sum((2 * n + 1) * (abs(a) ** 2 + abs(b) ** 2))
    adjacent = np.sum(n[:-1] * (n[:-1] + 2) / (n[:-1] + 1) * (a[:-1] * a[1:].conj() + b[:-1] * b[1:].conj()).real)
    crossed = np.sum((2 * n + 1) / (n * (n + 1)) * (a * b.conj()).real)
    g = 4 / (x * x * qs) * (adjacent + crossed)
    return float(qe), float(qs), float(g), a, b


def phase(mu, x, qs, a, b):
    """Unpolarized phase with mean over 4pi equal to one (not per sr)."""
    mu = np.asarray(mu, float)
    p0 = np.zeros_like(mu)
    p1 = np.ones_like(mu)
    s1 = np.zeros(mu.shape, complex)
    s2 = np.zeros(mu.shape, complex)
    for n, (an, bn) in enumerate(zip(a, b), 1):
        tn = n * mu * p1 - (n + 1) * p0
        fac = (2 * n + 1) / (n * (n + 1))
        s1 += fac * (an * p1 + bn * tn)
        s2 += fac * (an * tn + bn * p1)
        p0, p1 = p1, ((2 * n + 1) * mu * p1 - (n + 1) * p0) / n
    return 2 * (abs(s1) ** 2 + abs(s2) ** 2) / (x * x * qs)


def scattering_matrix(mu, x, qs, a, b):
    """Sphere scattering (Mueller) matrix elements (F11, F12, F33, F34); F22=F11, F44=F33.

    Same S1/S2 series as phase() (Bohren & Huffman 1983, ch. 4 sphere series and
    ch. 3 amplitude/Mueller matrices): S1 is the perpendicular amplitude, S2 the
    parallel one; scattered E_par = S2 E_par, E_perp = S1 E_perp up to a common
    factor.  Stokes vectors are referred to the scattering plane with
    Q = I_par - I_perp and U = 2 Re<E_par E_perp*>, so (BH Mueller elements)
        S11 = (|S2|^2+|S1|^2)/2      S12 = (|S2|^2-|S1|^2)/2
        S33 = Re(S2 S1*)             S34 = Im(S2 S1*)
    and the 4x4 matrix is [[F11,F12,0,0],[F12,F11,0,0],[0,0,F33,F34],[0,0,-F34,F33]].
    Normalization is phase()'s: every element is divided by the same constant
    x^2 Qs/4, so F11 is the mean-one phase (integral over 4 pi / 4 pi = 1) and is
    the identical expression phase() returns.  Rayleigh limit (x->0):
    F12/F11 -> -(1-mu^2)/(1+mu^2) (negative = perpendicular polarization),
    F33/F11 -> 2 mu/(1+mu^2).  The sign of F34 follows BH and only matters for V.
    Divide by 4 pi for a per-steradian matrix with integral of F11 over dOmega = 1.
    """
    mu = np.asarray(mu, float)
    p0 = np.zeros_like(mu)
    p1 = np.ones_like(mu)
    s1 = np.zeros(mu.shape, complex)
    s2 = np.zeros(mu.shape, complex)
    for n, (an, bn) in enumerate(zip(a, b), 1):
        tn = n * mu * p1 - (n + 1) * p0
        fac = (2 * n + 1) / (n * (n + 1))
        s1 += fac * (an * p1 + bn * tn)
        s2 += fac * (an * tn + bn * p1)
        p0, p1 = p1, ((2 * n + 1) * mu * p1 - (n + 1) * p0) / n
    f11 = 2 * (abs(s1) ** 2 + abs(s2) ** 2) / (x * x * qs)
    f12 = 2 * (abs(s2) ** 2 - abs(s1) ** 2) / (x * x * qs)
    cross = s2 * np.conj(s1)
    f33 = 4 * cross.real / (x * x * qs)
    f34 = 4 * cross.imag / (x * x * qs)
    return f11, f12, f33, f34


def ensemble(wavelength_nm, rg=0.08, sg=1.6, n=1.43, k=0.0, size_nodes=96, angle_nodes=512, L=32):
    if rg <= 0 or sg <= 1 or not np.isfinite([rg, sg, n, k]).all():
        raise ValueError('E_SIZE_INPUT')
    # Number distribution in log radius; +/-9 sigma includes the r^6-weighted tail.
    t, wt = np.polynomial.legendre.leggauss(size_nodes)
    t = t * 9
    wt = wt * 9 * np.exp(-t * t / 2) / np.sqrt(2 * np.pi)
    radii = rg * np.exp(np.log(sg) * t)
    mu, wm = np.polynomial.legendre.leggauss(angle_nodes)
    PL = np.polynomial.legendre.legvander(mu, L)
    ce = cs = gg = 0.0
    angular = np.zeros_like(mu)
    for r, w in zip(radii, wt):
        x = 2 * np.pi * r / (wavelength_nm * 0.001)
        qe, qs, g, a, b = efficiencies(x, complex(n, k))
        area = np.pi * r * r
        ce += w * area * qe
        sca = w * area * qs
        cs += sca
        gg += sca * g
        angular += sca * phase(mu, x, qs, a, b)
    angular /= cs
    bl = (wm[:, None] * PL * angular[:, None]).sum(0) / 2
    chi = bl * (2 * np.arange(L + 1) + 1)
    return dict(
        wavelength_nm=float(wavelength_nm),
        C_ext_um2=float(ce),
        C_sca_um2=float(cs),
        SSA=float(cs / ce),
        g=float(gg / cs),
        chi_l=chi.tolist(),
        phase_mean=float(bl[0]),
        g_angular=float(bl[1]),
    )


def self_checks():
    m = 1.43 + 0j
    xs = np.array([0.001, 0.002])
    q = np.array([efficiencies(x, m)[1] for x in xs])
    theory = (8 / 3) * xs**4 * abs((m * m - 1) / (m * m + 2)) ** 2
    rayerr = float(np.max(abs(q / theory - 1)))
    slope = float(np.log(q[1] / q[0]) / np.log(2))
    large = {str(x): efficiencies(x, m)[0] for x in (1000.0, 2000.0, 4000.0)}
    mu, w = np.polynomial.legendre.leggauss(512)
    qe, qs, g, a, b = efficiencies(2.5, m)
    p = phase(mu, 2.5, qs, a, b)
    norm = float(w @ p / 2)
    ga = float((w * mu) @ p / 2)
    chi1 = 3 * ga
    return dict(
        rayleigh=dict(max_relative=rayerr, x4_slope=slope, pass_=rayerr < 1e-5 and abs(slope - 4) < 1e-5),
        large_x=dict(
            Q_ext=large,
            max_absolute_from_2=max(abs(v - 2) for v in large.values()),
            pass_=all(abs(v - 2) < 0.04 for v in large.values()),
        ),
        normalization=dict(mean=norm, absolute_error=abs(norm - 1), pass_=abs(norm - 1) < 1e-11),
        asymmetry=dict(
            g_coefficients=g, g_angular=ga, chi_1_over_3=chi1 / 3, absolute_error=abs(g - ga), pass_=abs(g - ga) < 1e-11
        ),
    )
