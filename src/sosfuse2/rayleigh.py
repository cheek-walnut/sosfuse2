"""Bodhaine (1999) molecular cross section, anchored at 450 nm in standard air.

Equations 18, 19, 5, 6, 23 and 22, CO2=360 ppm. Wavelengths in those
equations are vacuum wavelengths; the engine uses Ciddor standard air.
"""
import numpy as np

from .air_wavelength import standard_n


def vacuum_nm(air_nm):
    air = np.asarray(air_nm, dtype='f8')
    vac = air.copy()
    for _ in range(6):
        vac = air * standard_n(vac)
    return vac


def sigma(vac_nm):
    """Cross section in cm² per molecule (FP64)."""
    vac = np.asarray(vac_nm, dtype='f8')
    u = vac / 1000.
    n = 1 + (8060.51 + 2480990 / (132.274-u**-2) + 17455.7 / (39.32957-u**-2))*1e-8*(1+.54*(360e-6-.0003))
    fn = 1.034 + 3.17e-4/u**2
    fo = 1.096 + 1.385e-3/u**2 + 1.448e-4/u**4
    king = (78.084*fn + 20.946*fo + .934 + .036*1.15)/(78.084+20.946+.934+.036)
    return 24*np.pi**3*(n*n-1)**2 / ((vac*1e-7)**4 * 2.546899e19**2 * (n*n+2)**2)*king


def coefficient(air_nm):
    """Unscaled Rayleigh extinction AND scattering coefficient, km⁻¹."""
    return .013558*(550./450.)**4 * sigma(vacuum_nm(air_nm))/sigma(vacuum_nm(450.))
