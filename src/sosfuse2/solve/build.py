"""Assemble a solver, its per-state engine context and geometry callbacks."""

import os
from functools import partial
from pathlib import Path

from .configuration import execution_domain, refined_chi, refined_heights

# Solar column table resolution multiplier. 4 = production (measured:
# scale 1 gives -2.38 % first order at -10 deg AOD .6, scale 4 +0.20 %); 1 = the accepted engine table.
SUN_SCALE = 4


def create_configured(
    output, cfg, wavelengths, backend, state, block_size=128, geometry_cache=None, *, share_sun_geometry=True, sun_scale=SUN_SCALE
):
    """Configure the accepted engine's create() for the production settings through its keyword hooks (observer, kernel,
    grid, spectrum, sun geometry, deferred ground, aerosol profile). Returns the solver and its scope; the solver is
    built over `wavelengths` (a second Solver in the same scope when more than one).
    Wavelengths are a one-dimensional host sequence in nm (normally a NumPy float64
    array, any stride). The input is read only. The grid hook receives one-dimensional
    host float64 height (km) and solar zenith (rad) arrays. It returns the original
    axes or refined arrays without modifying them in place."""
    from sosfuse2.atmosphere import spectral

    domain = execution_domain(cfg, state)
    spectral.check_state(
        cfg.get('spectral'), state
    )  # A′: a block resolved for another (or no) state never reaches SPEC
    if cfg.get('spectral') is not None:
        from ..atmosphere.components import check_current

        check_current(cfg['spectral'].get('physics'), what='solver spectral block')
    from sosfuse2.solve import ray_geometry as gm
    from sosfuse2.solve.source_solver import Kernel, load_engine

    engine = load_engine()
    threads_budget = int(os.environ.get('OMP_NUM_THREADS', str(os.cpu_count() or 1)))
    from sosfuse2.atmosphere.density import profile_step_km, kernel_kwargs

    kw = kernel_kwargs(profile_step_km(state))
    kw['domain'] = domain

    kernel_factory = partial(Kernel, backend=backend, block_size=block_size, L=cfg['L'], **kw)

    def grid(mode, h, ch, nmu, nphi, path_n):
        # The settings own the grid: full mode refines the accepted h/chi, every mode takes the quadrature counts from cfg.
        if mode == 'full':
            h = refined_heights(h, cfg['h_refine'])
            if domain.upper_heights_km:
                import numpy as np

                h = np.r_[h, domain.upper_heights_km]
            ch = refined_chi(ch, cfg['chi_step_deg'], cfg.get('chi_refine_deg'))
        if len(h) != cfg['nr'] or len(ch) != cfg['nchi']:
            raise ValueError('E_SETTINGS_GRID_COUNT: actual grid disagrees with nr/nchi')
        return h, ch, cfg['nmu'], cfg['nphi'], cfg['path_n']

    from sosfuse2.precompute import host_cache

    fast_host = host_cache.enabled()
    sun_geometry = None
    if fast_host or sun_scale != 1:
        # host_cache.sun_geometry keys its folder by height and scale; the accepted constructor keys neither.
        def sun_geometry(old, cache, H):
            return host_cache.sun_geometry(old, sun_scale, cache, H, shared=geometry_cache if share_sun_geometry else None)

    spec = None
    if cfg.get('spectral') is not None:
        # The engine's only spectral input is this SPEC dict (a fresh table per call; create() completes it in place).
        # Representative values belong to the settings bins; a solver over a subset (geometry-only passes the first
        # wavelength) reads its own rows of the full table.
        if any(float(w) not in cfg['wavelengths_nm'] for w in wavelengths):
            raise ValueError('E_SPECTRAL_CONTRACT: solver wavelengths outside the settings')
        spec = spectral.spec_table(cfg['spectral'], cfg['wavelengths_nm'], state)
    from sosfuse2.atmosphere import tropo_profile

    output.mkdir(parents=True, exist_ok=True)
    # With fast_host, density.attach prepares the ground below, after the wavelength Solver.
    initial, scope, _ = engine.create(
        state,
        output,
        'selftest' if cfg['nr'] == 4 else 'full',
        threads_budget,
        float(wavelengths[0]),
        observer_km=cfg['observer_km'],
        kernel_factory=kernel_factory,
        grid=grid,
        spec=spec,
        sun_geometry=sun_geometry,
        prepare_ground=not fast_host,
        aerosol=tropo_profile.engine_hooks(state),
        domain=domain,
    )
    s = initial if len(wavelengths) == 1 else scope.Solver(initial.geo, initial.kernel, wavelengths, 1, 0.03)
    s.levels = s.levels[: s.kernel.K]
    s.af = s.af[: s.kernel.K]
    s.active[1::2] = 0
    s.fuse_state = state
    s.export_first_order = cfg.get('export_first_order', False)
    tropo_profile.attach(s)
    if fast_host:
        scope.prepare_ground = True
    if geometry_cache is not None:
        s.host_cache_root = Path(geometry_cache)
    from sosfuse2.atmosphere.density import attach as attach_density
    from sosfuse2.atmosphere.stratosphere import tau_of

    # attach() immediately replaces all first-ground arrays when the layer is on.
    attach_density(s, scope, defer_ground=tau_of(state) > 0)
    from sosfuse2.atmosphere.stratosphere import attach

    attach(s, scope)
    from sosfuse2.atmosphere.tropo_mie import attach as attach_tropo

    attach_tropo(s)
    s.geo.pack = lambda p, d, n: gm.pack(p, d, n, s, scope, ground=False)
    return s, scope
