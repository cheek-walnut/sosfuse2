"""Export CPU optical profiles from installed sosfuse2; no GPU required."""

import argparse
from types import SimpleNamespace

import numpy as np

from common import add_paths, configure_root, dump, make_states, package_identity, read, sha, work_path


def main():
    parser = argparse.ArgumentParser()
    add_paths(parser)
    args = parser.parse_args()
    H = work_path(args.work)
    configure_root(H)
    from sosfuse2.atmosphere import components, density, extinction, minor_absorbers, rayleigh, spectral, stratosphere
    from sosfuse2.jobs import run_state as engine

    if not minor_absorbers.o4_path().is_file():
        raise ValueError('E_O4_ASSET_MISSING: run python -m sosfuse2.asset_tools.prepare_minor_absorbers --download')
    c = read(args.config)
    states_parent = make_states(H, c)
    settings = H / 'settings.json'
    dump(settings, c['settings'])
    cfg = engine.settings_for(SimpleNamespace(reduced=False, settings_json=settings))
    states_file = H / 'states.json'
    states = {s['id']: s for s in read(states_file)['states'] if s['id'] in c['state_ids']}
    if set(states) != set(c['state_ids']):
        raise ValueError('E_STATE_MISSING')
    # Integrate the molecular profiles on the layer grid.
    wl = np.asarray(cfg['wavelengths_nm'])
    levels = np.unique(np.round(np.r_[np.arange(31) * 0.1, np.arange(6, 161) * 0.5], 9))[::-1]
    if not np.allclose(wl, c['wavelength_nm'], rtol=0, atol=1e-12):
        raise ValueError('E_CONFIG_WAVELENGTH')
    lo, hi = levels[1:], levels[:-1]
    dz = hi - lo
    edges = levels[::-1].copy()
    h = edges[:-1, None] + (np.arange(16)[None, :] + 0.5) / 16 * np.diff(edges)[:, None]
    js = np.clip(np.searchsorted(edges, (lo + hi) / 2, side='right') - 1, 0, len(edges) - 2)
    out = {}
    rows = []
    for sid, st in states.items():
        opt = density.build(st)
        p = st['parameters']
        block = engine.resolve_settings(cfg, st)['spectral']
        spec = spectral.spec_table(block, wl, st)
        weights, meta = spectral.table_weights(block, wl)
        pp = dict(
            aer=p['aod550'] / (p['scale_height_km'] * -np.expm1(-80 / p['scale_height_km'])),
            ang=p['angstrom'],
            ozone_DU=p['ozone_du'],
            **components.optics_state(st, opt),
            no2_H_f_km=opt['meta']['no2']['H_f_km'],
        )
        k = extinction.coefficients(spec, wl, pp)[[0, 2, 3, 4]]
        k[0] *= opt['scale']
        layer = np.empty((len(dz), 4))
        grid = density.grid(len(opt['table']))
        for li, (left, right) in enumerate(zip(lo, hi, strict=True)):
            cuts = np.unique(np.r_[left, grid[(grid > left) & (grid < right)], right])
            widths = np.diff(cuts)
            gn = (cuts[:-1, None] + cuts[1:, None]) / 2 + widths[:, None] / (2 * np.sqrt(3)) * np.array([-1.0, 1.0])
            layer[li] = (density.molecular_profiles(gn, opt).mean(axis=1) * widths[:, None]).sum(axis=0)
        tau = layer[:, :, None] * k[None, :, :]
        # Integrate analytic and tabulated density columns.
        exact = density.exact_profile_columns(
            np.array([[0.0, 0.0, 6371.0]]), np.array([[0.0, 0.0, 1.0]]), np.array([80.0]), opt
        )[0]
        air = exact[0] if opt['mask'] & 1 else 8 * -np.expm1(-10)
        o4col = exact[3] if opt['mask'] & 1 else 4 * -np.expm1(-20)
        expected = np.stack(
            [
                rayleigh.coefficient(wl) * opt['scale'] * air,
                np.asarray(spec['ozone']) * p['ozone_du'] * 2.687e20,
                np.asarray(spec['no2']) * pp['no2_du'] * 2.687e20,
                k[3] * o4col,
            ]
        )
        for q, name in enumerate(('rayleigh', 'o3', 'no2', 'o4')):
            for i, w in enumerate(wl):
                m = float(tau[:, q, i].sum())
                e = float(expected[q, i])
                err = abs(m / e - 1) if e else (0.0 if m == 0 else 1e300)
                rows.append(
                    dict(state=sid, component=name, wavelength_nm=float(w), engine=e, layer_sum=m, relative_error=err)
                )
        aw = np.unique(np.r_[350.0, wl, 550.0, 850.0])
        tw = stratosphere.load_table(stratosphere.table_path(st))[0]
        strat = stratosphere.build_optics(st, edges, np.clip(aw, tw[0], tw[-1]))
        tro = (pp['aer'] * np.exp(-h / p['scale_height_km'])).mean(axis=1)
        sd = stratosphere.density(h, strat['profile']).mean(axis=1)
        out[sid] = dict(
            state=st,
            density=opt['meta'],
            wavelengths_nm=wl.tolist(),
            solar=spec['solar'],
            weights=np.asarray(weights).tolist(),
            weights_meta=meta,
            spectral_block=block,
            tau_components={n: tau[:, j].T.tolist() for j, n in enumerate(('rayleigh', 'o3', 'no2', 'o4'))},
            aerosol=dict(
                wavelengths_nm=aw.tolist(),
                edges_km=edges.tolist(),
                tropo_ext550_shell=tro.tolist(),
                strato_density_shell=sd.tolist(),
                strato_ext=strat['ext'].tolist(),
                strato_sca=strat['sca'].tolist(),
                strato_chi=strat['chi'].T.tolist(),
                tropo_column=float(tro[js] @ dz),
                strato_column=float(sd[js] @ dz),
            ),
        )
        for w in wl:
            ai = int(np.flatnonzero(np.isclose(aw, w, rtol=0, atol=1e-9))[0])
            for name, observed, ref in [
                ('tropo', float(tro[js] @ dz) * (550 / w) ** p['angstrom'], p['aod550'] * (550 / w) ** p['angstrom']),
                ('strato', float(sd[js] @ dz) * strat['ext'][ai], strat['ext'][ai]),
            ]:
                rows.append(
                    dict(
                        state=sid,
                        component=name,
                        wavelength_nm=float(w),
                        engine=float(ref),
                        layer_sum=float(observed),
                        relative_error=float(abs(observed / ref - 1)),
                    )
                )
    identity = {
        'states_asset': sha(states_file),
        'o4_asset': sha(minor_absorbers.o4_path()),
        'no2_asset': sha(minor_absorbers.NO2),
        'config': sha(args.config),
        'states_parent': sha(states_parent),
        'settings': sha(settings),
        'package': package_identity(),
    }
    identity.update(
        {
            m.__name__: sha(m.__file__)
            for m in (density, rayleigh, extinction, spectral, stratosphere, engine, components, minor_absorbers)
        }
    )
    (H / 'out').mkdir(exist_ok=True)
    dump(H / 'out/optics.json', dict(settings=cfg, levels_km=levels.tolist(), states=out, identity=identity))
    worst = max(r['relative_error'] for r in rows)
    dump(H / 'out/tau_ledger.json', dict(rows=rows, max_relative_error=worst, passed=worst <= 1e-4, identity=identity))
    print('TAU_LEDGER', len(rows), worst, 'PASS' if worst <= 1e-4 else 'FAIL')
    if worst > 1e-4:
        raise SystemExit('E_TAU_LEDGER')


if __name__ == '__main__':
    main()
