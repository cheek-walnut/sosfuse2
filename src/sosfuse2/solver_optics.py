"""Solver definitions loaded into per-call scopes by accepted.engine.definitions."""
# ruff: noqa: F821 -- globals are injected by the per-call loader.

def coefficients(wavelengths, p, no2=None, o4=None):
    from sosfuse2.components import reference_active
    from sosfuse2.rayleigh import coefficient
    w = np.asarray(wavelengths, dtype='f8')
    ray = .013558*(550/w)**4 if reference_active() else coefficient(w)
    # The per-state normalisation and SI values stay FP64 until device setup.
    if no2 is None:
        no2 = 0. if reference_active() else np.interp(w, SPEC['wavelength_nm'], SPEC['no2'])*2.687e20*p['no2_du']/p['no2_H_f_km']
    if o4 is None:
        o4 = 0. if reference_active() or not p['o4'] else 1e3*np.interp(w, SPEC['wavelength_nm'], SPEC['o4'])*(.20946*p['n0'])**2
    return np.stack([ray, p['aer']*(550/w)**p['ang'],
                     np.interp(w,SPEC['wavelength_nm'],SPEC['ozone'])*p['ozone_DU']*2.687e20/15,
                     np.broadcast_to(no2, w.shape), np.broadcast_to(o4, w.shape)])
