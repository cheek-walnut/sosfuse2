"""Validate metadata and extract table pixels with immutable identities."""

import argparse
import json

import numpy as np

from common import add_paths, configure_root, dump, sha, work_path


def validate(meta, st, expected, contract):
    from sosfuse2.atmosphere import density, stratosphere
    from sosfuse2.jobs import run_state as engine

    if not meta.get('complete'):
        raise ValueError('E_ENGINE_INCOMPLETE')
    if meta['parameters'] != st['parameters']:
        raise ValueError('E_ENGINE_PARAMETERS')
    if meta['density'] != density.build(st)['meta']:
        raise ValueError('E_ENGINE_DENSITY')
    # Includes all grids, geometry, spectral block, wavelengths, L, path_n and ground SH.
    if meta['settings'] != expected:
        raise ValueError('E_ENGINE_SETTINGS')
    if meta['package_version'] != contract['engine_version']:
        raise ValueError('E_ENGINE_VERSION')
    if contract.get('engine_commit') is not None and meta.get('package_commit') != contract['engine_commit']:
        raise ValueError('E_ENGINE_COMMIT')
    if meta['polarization_model'] != 'depolarizing':
        raise ValueError('E_ENGINE_POLARIZATION')
    strato_path = stratosphere.table_path(st)
    expected_strato = dict(
        st['stratosphere'],
        optics_table=stratosphere.path_label(strato_path),
        optics_sha256=sha(strato_path),
        polarization='fully depolarizing',
        status='enabled',
    )
    if meta['stratosphere'] != expected_strato:
        raise ValueError('E_ENGINE_STRATOSPHERE')
    if not isinstance(meta.get('kernel'), dict) or not meta['kernel'].get('deterministic'):
        raise ValueError('E_ENGINE_DETERMINISTIC')
    if expected.get('kernel', {}).get('ground_sh', {}).get('order') != 24:
        raise ValueError('E_ENGINE_GROUND_SH')
    # Check the tolerance included in the signed contract.
    if engine.ANDERSON_TOLERANCE != contract['tolerance'] or contract['tolerance'] != 0.001:
        raise ValueError('E_ENGINE_TOLERANCE')
    sig, _ = engine.signed_contract(
        st, expected, meta['backend'], False, meta['polarization_model'], meta.get('derived')
    )
    if sig != meta['signature']:
        raise ValueError('E_ENGINE_SIGNED_CONTRACT')
    stats = meta['source_solve']
    chunks = stats if isinstance(stats, list) else [stats]
    if not chunks or any(not isinstance(x, dict) or not x.get('converged') or not all(x['converged']) for x in chunks):
        raise ValueError('E_ENGINE_CONVERGENCE')
    return dict(tolerance=contract['tolerance'], convergence=stats, signature=sig)


def extract(H, c, data, arms, arm, state_ids):
    from sosfuse2.foundation import paths
    from sosfuse2.jobs import run_state as engine

    overrides = arms[arm]
    contract = json.loads((H / 'engine' / arm / 'contract.json').read_text())
    if (
        contract['optics_sha256'] != sha(H / 'out/optics.json')
        or contract['states_sha256'] != sha(H / 'states.json')
        or contract['states_parent_sha256'] != sha(paths.ASSETS / 'states_v2.json')
    ):
        raise ValueError('E_ARM_INPUT_SHA')
    expected = dict(data['settings'], **overrides)
    expected['kernel'] = contract['settings']['kernel']
    if expected != contract['settings']:
        raise ValueError('E_ARM_SETTINGS')
    if expected['path_n'] != dict(A0=12, A1=24, A2=12)[arm] or expected['L'] != dict(A0=16, A1=16, A2=24)[arm]:
        raise ValueError('E_ARM_DEFINITION')
    cases = {}
    tables = {}
    metadata = {}
    for sid in state_ids:
        folder = H / 'engine' / arm / 'work/out' / sid
        meta = json.loads((folder / 'meta.json').read_text())
        st = data['states'][sid]['state']
        metadata[sid] = dict(
            validate(meta, st, engine.resolve_settings(expected, st), contract),
            meta_sha256=sha(folder / 'meta.json'),
            density=meta['density'],
            parameters=meta['parameters'],
            settings=meta['settings'],
            package_version=meta['package_version'],
        )
        file = folder / 'table.npz'
        tables[sid] = sha(file)
        with np.load(file) as z:
            wl = z['wavelength_nm']
            el = z['view_elevation_deg']
            az = z['relative_azimuth_deg']
            hs = z['solar_elevation_deg']
            rad = z['radiance']
            if not np.allclose(wl, data['states'][sid]['wavelengths_nm'], rtol=1e-6, atol=0):
                raise ValueError('E_WAVELENGTH_AXIS')
            for p in c['pixels']:
                i, j = p['i'], p['j']
                if not np.isclose(el[i], p['elevation_deg'], atol=1e-6, rtol=0) or not np.isclose(
                    az[j], p['azimuth_deg'], atol=1e-6, rtol=0
                ):
                    raise ValueError('E_VIEW_AXIS')
                for sza in c['sza_deg']:
                    k = np.flatnonzero(np.isclose(hs, 90 - sza, atol=1e-6, rtol=0))
                    if len(k) != 1:
                        raise ValueError('E_SOLAR_AXIS')
                    spectrum = np.asarray(rad[int(k[0]), i, j], float)
                    if not np.isfinite(spectrum).all() or np.any(spectrum <= 0):
                        raise ValueError('E_RADIANCE')
                    case = f'{sid}_i{i}_j{j}_sza{sza:g}'
                    cases[case] = dict(
                        state_id=sid,
                        i=i,
                        j=j,
                        elevation_deg=float(el[i]),
                        azimuth_deg=float(az[j]),
                        sza_deg=float(90 - hs[int(k[0])]),
                        wavelength_nm=wl.tolist(),
                        radiance=spectrum.tolist(),
                    )
    dump(
        H / 'out' / f'engine_pixels.{arm}.json',
        dict(
            engine_config_id=arm,
            contract=contract,
            cases=cases,
            table_sha256=tables,
            metadata=metadata,
            optics_sha256=sha(H / 'out/optics.json'),
        ),
    )
    print('ENGINE_PIXELS', arm, len(cases))


def main():
    ap = argparse.ArgumentParser()
    add_paths(ap, arms=True)
    ap.add_argument('--arms', default='A0')
    ap.add_argument('--states')
    a = ap.parse_args()
    H = work_path(a.work)
    configure_root(H)
    c = json.loads(a.config.read_text())
    data = json.loads((H / 'out/optics.json').read_text())
    arms = json.loads(a.arms_file.read_text())
    state_ids = a.states.split(',') if a.states else c['state_ids']
    if not state_ids or not set(state_ids) <= set(c['state_ids']):
        raise ValueError('E_STATE_MISSING')
    for arm in a.arms.split(','):
        if arm not in ('A0', 'A1', 'A2') or arm not in arms:
            raise ValueError('E_UNKNOWN_ARM')
        extract(H, c, data, arms, arm, state_ids)


if __name__ == '__main__':
    main()
