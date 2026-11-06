"""Vacuum -> air wavelength: λ_air = λ_vac / n(λ_vac) in standard dry air.

Standard air is a coordinate convention (15 °C, 101 325 Pa, dry, CO2 450 µmol/mol; Ciddor 1996, Appl. Opt. 35, 1566),
not the local medium of a twilight path. standard_n is Ciddor's standard-air dispersion (eq. 1); ciddor_dry adds the
dry-air density ratio (BIPM-91 compressibility, Ciddor eqs. 2, 4, A) so the chain can be checked against NIST's
published Ciddor values at other temperatures and pressures (test_air_wavelength.py). edlen1966_n is kept for the
record only (the 10-30 shift measurement used it; Ciddor - Edlén (n-1) is 1.1-1.4e-8 over 360-840 nm).
"""
import numpy as np

K0, K1, K2, K3 = 238.0185, 5792105., 57.362, 167917.        # Ciddor 1996 eq. 1, µm^-2 and 1e-8 units
R_GAS = 8.314510                                             # J mol^-1 K^-1 (Ciddor 1996)
BIPM = dict(a0=1.58123e-6, a1=-2.9331e-8, a2=1.1043e-10, d=1.83e-11)   # dry-air terms of the compressibility Z
STANDARD = dict(t_c=15., p_pa=101325., co2=450.)


def _sigma2(lam_vac_nm):
    lam = np.asarray(lam_vac_nm, float)*1e-3                # µm
    return 1./lam**2


def standard_n(lam_vac_nm, co2=450.):
    """n of standard dry air (15 °C, 101 325 Pa) at the vacuum wavelength, Ciddor eq. 1 with the CO2 scaling."""
    s2 = _sigma2(lam_vac_nm)
    n1 = (K1/(K0-s2) + K3/(K2-s2))*1e-8
    return 1. + n1*(1. + 0.534e-6*(co2-450.))


def _density(t_c, p_pa, co2):
    T = t_c+273.15
    M = 1e-3*(28.9635 + 12.011e-6*(co2-400.))
    Z = 1. - p_pa/T*(BIPM['a0'] + BIPM['a1']*t_c + BIPM['a2']*t_c**2) + (p_pa/T)**2*BIPM['d']
    return p_pa*M/(Z*R_GAS*T)


def ciddor_dry(lam_vac_nm, t_c, p_pa, co2=450.):
    """n of dry air at temperature t_c (°C), pressure p_pa (Pa) and CO2 (µmol/mol): standard n scaled by density."""
    ratio = _density(t_c, p_pa, co2)/_density(STANDARD['t_c'], STANDARD['p_pa'], co2)
    return 1. + ratio*(standard_n(lam_vac_nm, co2)-1.)


def edlen1966_n(lam_vac_nm):
    """Edlén 1966 standard air (15 °C, 101 325 Pa, CO2 300 µmol/mol) — record only."""
    s2 = _sigma2(lam_vac_nm)
    return 1. + (8342.13 + 2406030./(130.-s2) + 15997./(38.9-s2))*1e-8


def vacuum_to_air(lam_vac_nm):
    """λ_air = λ_vac / n_standard(λ_vac), nm."""
    lam = np.asarray(lam_vac_nm, float)
    return lam/standard_n(lam)
