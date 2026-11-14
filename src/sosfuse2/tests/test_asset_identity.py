"""Numeric asset boundaries and compatibility with published byte digests."""

import copy
import json

import numpy as np
import pytest

from sosfuse2.atmosphere import components, density, minor_absorbers as MA
from sosfuse2.foundation import content_identity as I
from sosfuse2.atmosphere import ozone_temperature as OT, spectral as SP
from sosfuse2.foundation import paths


INPUTS = {
    'spectra_fine.json': ('format', 'wavelength_nm', 'solar', 'ozone_213K', 'ozone_233K', 'ozone_253K', 'units'),
    'spectra_fine_temperature.json': ('format', 'wavelength_nm', 'solar', 'temperatures_K', 'units',
                                    'frozen_source_sha256', *(f'ozone_{t}K' for t in range(193, 294, 10))),
    'temperature_usstd76.json': ('altitude_km', 'temperature_K', 'units', 'interpolation',
                               'geopotential_radius_km', 'geopotential_bounds_km', 'lapse_K_per_km'),
    'no2_afglus_profile.json': ('altitude_km', 'no2_number_density_m3', 'source_sha256', 'interpolation'),
    'no2_vandaele1998_fine.json': ('format', 'wavelength_nm', 'no2_220K', 'units', 'rules'),
}


def encoded(doc):
    return json.dumps(doc, allow_nan=False).encode()


@pytest.mark.parametrize('name', INPUTS)
def test_published_asset_equivalence_and_notes(name):
    raw = (paths.ASSETS / name).read_bytes()
    old, expected = next(iter(I.ASSET_SUPERSEDED[name].items()))
    assert I.asset_identity(raw, name) == expected
    assert I.asset_matches(name, old, raw)
    doc = json.loads(raw)
    for key in I.ASSET_NOTES[name]:
        changed = copy.deepcopy(doc)
        changed[key] = 'Documentary revision'
        assert I.asset_identity(encoded(changed), name) == expected
    assert not I.asset_matches(name, '0' * 64, raw)


@pytest.mark.parametrize('name,key', [(n, k) for n, keys in INPUTS.items() for k in keys])
def test_computation_fields_cannot_disappear(name, key):
    doc = json.loads((paths.ASSETS / name).read_bytes())
    original = I.asset_identity(encoded(doc), name)
    value = doc[key]
    if isinstance(value, list):
        value[0] += 0.125
    elif isinstance(value, dict):
        value[next(iter(value))] = 'changed'
    elif isinstance(value, (float, int)):
        doc[key] += 0.125
    else:
        doc[key] += ' changed'
    raw = encoded(doc)
    assert I.asset_identity(raw, name) != original
    assert not I.asset_matches(name, next(iter(I.ASSET_SUPERSEDED[name])), raw)


def current_block():
    raw = dict(spectral_mode='band', ozone_temperature='profile_mean', bin_edges_nm=SP.PRODUCT_EDGES)
    return SP.resolve(SP.block(raw, SP.PRODUCT['wavelengths_nm'], True), {})


def published_block(block):
    block = copy.deepcopy(block)
    for key, name in [('asset_sha256', 'spectra_fine.json'),
                      ('temperature_asset_sha256', 'temperature_usstd76.json'),
                      ('ozone_sigma_asset_sha256', 'spectra_fine_temperature.json')]:
        block[key] = next(iter(I.ASSET_SUPERSEDED[name]))
    for kind, name in [('no2', MA.NO2.name), ('o4', MA.O4_NAME)]:
        if block['minor_absorber_sha256'][kind] is not None:
            digest = next(iter(I.ASSET_SUPERSEDED[name])) if kind == 'no2' else MA.load(kind)['raw_sha256']
            block['minor_absorber_sha256'][kind] = digest
            block['physics']['minor_absorbers'][kind]['asset_sha256'] = digest
    return block


@pytest.mark.parametrize('version', ['v3', 'v4', 'v5'])
def test_old_table_and_source_physics(version):
    new = current_block()
    old = published_block(new)
    meta = dict(signature='test', contract=version, settings=dict(spectral=old), physics=old['physics'],
                optics_state=dict(no2_du=0.207, o4=False, n0=2.5469e25))
    assert SP.table_block(meta) == old
    SP.check_asset(old)
    SP.check_contract(old, new)
    components.check_current(old['physics'], what='source')
    for a, b in zip(SP.representative(old, SP.PRODUCT['wavelengths_nm']),
                    SP.representative(new, SP.PRODUCT['wavelengths_nm'])):
        np.testing.assert_array_equal(a, b)
    got = MA.representative(old, SP.PRODUCT['wavelengths_nm'], o4=False)
    expected = MA.representative(new, SP.PRODUCT['wavelengths_nm'], o4=False)
    np.testing.assert_array_equal(got['no2'], expected['no2'])


@pytest.mark.parametrize('field,code', [('asset_sha256', 'E_SPECTRAL_CONTRACT'),
                                     ('temperature_asset_sha256', 'E_OZONE_T_ASSET_SHA'),
                                     ('ozone_sigma_asset_sha256', 'E_OZONE_T_ASSET_SHA')])
def test_unknown_digest_refused(field, code):
    b = published_block(current_block())
    b[field] = '0' * 64
    with pytest.raises(ValueError, match=code):
        SP.check_asset(b)


def test_generator_records_are_not_cross_section_identity(tmp_path):
    raw = MA.NO2.read_bytes()
    doc = json.loads(raw)
    doc['generator']['sha256'] = '1' * 64
    doc['generator']['identity_sha256'] = '2' * 64
    doc['source']['licence'] = 'Updated licence description'
    path = tmp_path / 'generated.json'
    path.write_bytes(encoded(doc))
    got = MA.load('no2', path=path)
    original = MA.load('no2')
    assert got['raw_sha256'] != original['raw_sha256']
    assert got['sha256'] == original['sha256']
    for section, key in [('generator', 'rules_version'), ('source', 'files')]:
        changed = copy.deepcopy(doc)
        changed[section][key] = 'changed'
        assert I.asset_identity(encoded(changed), MA.NO2.name) != original['sha256']


def test_o4_validation_and_generator_boundary(tmp_path):
    from sosfuse2.asset_tools.generate_minor_absorbers import CIA_SHA, RULES_VERSION

    doc = dict(format='o4-fine-v1', wavelength_nm=SP.asset()['wl'].tolist(), o4=[1e-46] * len(SP.asset()['wl']),
               units=dict(wavelength_nm='nm (air)', o4=MA.UNITS['o4']),
               generator=dict(rules_version=RULES_VERSION, sha256='1' * 64), source=dict(sha256=CIA_SHA))
    path = tmp_path / MA.O4_NAME
    path.write_bytes(encoded(doc))
    initial = MA.load('o4', path=path)
    doc['generator']['sha256'] = '2' * 64
    path.write_bytes(encoded(doc))
    assert MA.load('o4', path=path)['sha256'] == initial['sha256']
    doc['o4'][0] *= 2
    path.write_bytes(encoded(doc))
    assert MA.load('o4', path=path)['sha256'] != initial['sha256']
    doc['source']['sha256'] = '0' * 64
    path.write_bytes(encoded(doc))
    with pytest.raises(ValueError, match='E_O4_ASSET_STALE'):
        MA.load('o4', path=path)


def test_frozen_spectral_reference_is_checked(tmp_path, monkeypatch):
    doc = json.loads(OT.FINE.read_bytes())
    doc['frozen_source_sha256'] = '0' * 64
    path = tmp_path / OT.FINE.name
    path.write_bytes(encoded(doc))
    monkeypatch.setattr(OT, 'FINE', path)
    with pytest.raises(ValueError, match='E_OZONE_T_ASSET_SHA'):
        OT.ProfileMean(SP.PRODUCT_EDGES, 'triangle')


def test_unknown_asset_fields_and_profile_cache_identity():
    name = 'no2_afglus_profile.json'
    raw = (paths.ASSETS / name).read_bytes()
    doc = json.loads(raw)
    doc['new_input'] = 7
    assert I.asset_identity(encoded(doc), name) != I.asset_identity(raw, name)
    assert density.no2_asset()[2] == I.asset_identity(raw, name)
    assert I.package_hashes()[paths.rel(paths.ASSETS / name)] == density.no2_asset()[2]


def test_asset_note_boundary_is_explicit():
    expected = {
        'spectra_fine.json': ('definition', 'wavelength_reference', 'sources', 'generator', 'previous_sha256', 'self_check'),
        'spectra_fine_temperature.json': ('wavelength_reference', 'sources', 'generator', 'previous_sha256'),
        'temperature_usstd76.json': ('generator_sha256', 'formula_source'),
        'no2_afglus_profile.json': ('source', 'generator', 'semantics'),
        'no2_vandaele1998_fine.json': ('definition', 'wavelength_reference', 'self_check'),
        'o4_hitran2024_fine.json': ('definition', 'wavelength_reference', 'self_check'),
    }

    for name, excluded in expected.items():
        doc = {key: {'note_gain': 1} for key in excluded}
        original = I.asset_identity(encoded(doc), name)
        for key in excluded:
            changed = copy.deepcopy(doc)
            del changed[key]
            assert I.asset_identity(encoded(changed), name) == original, (name, key)
        for key in ('note_gain', 'description', 'new_input'):
            changed = dict(doc, **{key: 123})
            assert I.asset_identity(encoded(changed), name) != original, (name, key)

    for name in ('no2_vandaele1998_fine.json', 'o4_hitran2024_fine.json'):
        for section, keys in {
            'generator': ('sha256', 'identity_sha256'),
            'source': ('name', 'dataset', 'page', 'licence', 'reference', 'references',
                       'url', 'wavelength', 'unit_conversion'),
        }.items():
            doc = {section: {key: 'prose' for key in keys}}
            original = I.asset_identity(encoded(doc), name)
            for key in keys:
                changed = copy.deepcopy(doc)
                del changed[section][key]
                assert I.asset_identity(encoded(changed), name) == original, (name, section, key)
            changed = copy.deepcopy(doc)
            changed[section]['note_gain'] = 123
            assert I.asset_identity(encoded(changed), name) != original


@pytest.mark.parametrize('section', [None, 'rules', 'source'])
def test_no2_unknown_note_fields_are_inputs(section):
    name = 'no2_vandaele1998_fine.json'
    raw = (paths.ASSETS / name).read_bytes()
    doc = json.loads(raw)
    target = doc if section is None else doc[section]
    target['note_gain'] = 123
    assert I.asset_identity(encoded(doc), name) != I.asset_identity(raw, name)


@pytest.mark.parametrize('raw', [b'{"note":NaN}', b'{"definition":1e400}', b'{"x":1,"x":2}'])
def test_asset_strict_json_before_exclusions(raw):
    with pytest.raises(ValueError, match='E_IDENTITY_SOURCE'):
        I.asset_identity(raw, 'spectra_fine.json')


@pytest.mark.parametrize('version', ['v3', 'v4'])
def test_saved_source_restores_registered_assets(tmp_path, version):
    from types import SimpleNamespace
    from sosfuse2.jobs import run_state as runner
    from sosfuse2.outputs import radiance_table

    state = {'id': 'asset-test', 'parameters': {'aod550': 0.1, 'o4': False}}
    # Build full settings through the same public adapter used by restoration.
    raw = dict(ozone_temperature='profile_mean', spectral_mode='band',
               wavelengths_nm=SP.PRODUCT['wavelengths_nm'], wavelengths=45, bin_edges_nm=SP.PRODUCT_EDGES)
    cfg = runner.resolve_settings(runner.settings_for(SimpleNamespace(
        reduced=True, settings_json=SimpleNamespace(read_text=lambda: json.dumps(raw)))), state)
    old = copy.deepcopy(cfg)
    old['spectral'] = published_block(cfg['spectral'])
    contract = dict(contract=version, state=components.canonical_state(state), settings=old,
                    backend='numpy', polarization_model='depolarizing')
    path = tmp_path / 'source.npz'
    solver = SimpleNamespace(fuse_state=state, geo=SimpleNamespace(h=np.array([0.]), chi=np.array([0.])))
    payload = dict(wavelength_nm=cfg['wavelengths_nm'], h=solver.geo.h, chi=solver.geo.chi,
                   source=np.ones((1, 1, 45), 'f4'), physics=json.dumps(old['spectral']['physics']),
                   optics_state=json.dumps(components.optics_state(state)))
    np.savez(path, **payload, source_contract=json.dumps(contract))
    assert radiance_table.restored_settings(path, state, True, 'numpy') == old
    np.testing.assert_array_equal(radiance_table.load_source(path, solver, cfg['wavelengths_nm'], settings=cfg,
                                                       backend='numpy', full_settings=True), payload['source'])
    assert SP.same_settings(old, cfg)
    changed = copy.deepcopy(old)
    changed['spectral']['unexpected_input'] = 1
    assert not SP.same_settings(changed, cfg)
    changed = copy.deepcopy(old)
    changed['spectral']['temperature_asset_sha256'] = '0' * 64
    contract['settings'] = changed
    np.savez(path, **payload, source_contract=json.dumps(contract))
    with pytest.raises(ValueError, match='E_SOURCE_CONTRACT'):
        radiance_table.restored_settings(path, state, True, 'numpy')
