"""FP32 primitives used verbatim by kernel.cu (no implicit FMA).

Coordinates use two FP32 words. Only the NumPy emulation of the hardware
FP32 FMA uses an FP64 temporary; the CUDA coordinate path has no FP64 ALU.
SH is evaluated once per ray in FP64 then rounded, as in the accepted solver.
"""
import math
import numpy as np

F = np.float32
R = F(6371)


def split(x):
    x = np.asarray(x, dtype=np.float64)
    hi = x.astype('f4')
    return hi, (x-hi.astype('f8')).astype('f4')


def add(a, b):
    s = a[0]+b[0]
    v = s-a[0]
    e = ((a[0]-(s-v))+(b[0]-v))+(a[1]+b[1])
    h = s+e
    return h, e-(h-s)


def neg(a):
    return -a[0], -a[1]


def sub(a, b):
    return add(a, neg(b))


def mul(a, b):
    p = a[0]*b[0]
    # Correctly rounded fmaf(a.hi,b.hi,-p), including its one rounding.
    e = (a[0].astype('f8')*b[0].astype('f8')-p.astype('f8')).astype('f4')
    e = (e+a[0]*b[1])+a[1]*b[0]
    h = p+e
    return h, e-(h-p)


def root(a):
    h = np.sqrt(np.maximum(a[0], F(0)))
    hh = mul((h, np.zeros_like(h)), (h, np.zeros_like(h)))
    e = sub(a, hh)
    lo = (e[0]+e[1])/np.maximum(F(2)*h, F(1e-30))
    return add((h, np.zeros_like(h)), (lo, np.zeros_like(lo)))


def divide(a, b):
    h = a[0]/b[0]
    e = sub(a, mul((h, np.zeros_like(h)), b))
    lo = (e[0]+e[1])/b[0]
    return add((h,np.zeros_like(h)),(lo,np.zeros_like(lo)))


def quotient(a, b):
    return rounded(divide(a,b))


def rounded(a):
    return a[0]+a[1]


def norm_constants(L=32):
    from .sh_layout import count
    aa = np.zeros(count(L), 'f8'); bb = np.zeros_like(aa)
    dd = np.zeros(33, 'f8')
    for m in range(1, L+1):
        dd[m] = -math.sqrt((2.*m+1)/(2*m))
    for l in range(1, L+1):
        for m in range(l):
            k = l*(l+1)//2+m
            aa[k] = math.sqrt((4.*l*l-1)/(l*l-m*m))
            if l>m+1:
                bb[k] = math.sqrt((2.*l+1)*((l-1.)**2-m*m)/((2.*l-3)*(l*l-m*m)))
    return aa, bb, dd


def basis(mu, L=32):
    """Source SH at cos(phi)=1; destination SH reuses it with cos(m phi)."""
    from .sh_layout import count
    mu = np.clip(np.asarray(mu, 'f8'), -1, 1)
    st = np.sqrt(np.maximum(0., 1-mu*mu))
    aa, bb, dd = norm_constants(L)
    out = np.empty((len(mu), count(L)), 'f4')
    diag = np.full_like(mu, 1/math.sqrt(4*math.pi))
    for m in range(L+1):
        if m:
            diag = (diag*dd[m])*st
        mult = math.sqrt(2.) if m else 1.
        z0 = diag
        out[:, m*(m+1)//2+m] = mult*z0
        if m<L:
            z1 = (math.sqrt(2.*m+3)*mu)*z0
            out[:, (m+1)*(m+2)//2+m] = mult*z1
            for l in range(m+2, L+1):
                k = l*(l+1)//2+m
                z2 = (aa[k]*mu)*z1-bb[k]*z0
                out[:, k] = mult*z2
                z0, z1 = z1, z2
    return out


def cosine(cp):
    cp = np.clip(cp, F(-1), F(1)).astype('f4')
    out = np.empty((len(cp), 33), 'f4')
    out[:, 0] = F(1); out[:, 1] = cp
    for m in range(2, 33):
        out[:, m] = (F(2)*cp)*out[:, m-1]-out[:, m-2]
    return out


def warp_sum(x):
    """CUDA shfl_down tree: m=32 belongs to lane zero, added before tree."""
    x = x.copy()
    x[:, 0] = x[:, 0]+x[:, 32]
    x = x[:, :32]
    for stride in (16, 8, 4, 2, 1):
        old = x.copy()
        x[:, :32-stride] = old[:, :32-stride]+old[:, stride:]
    return x[:, 0]


def kahan(total, correction, value):
    y = value-correction
    t = total+y
    return t, (t-total)-y
