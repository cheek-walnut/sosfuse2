"""A′ P1 opt-in checks, groups ①–⑨ (numbers in the test names).

Fast (plain pytest; no GPU, no solve):
  ① test_01_default_path_unchanged        233K / 213 / 253 / point: same block fields, block_key and SPEC arithmetic
     test_01b_product_default_is_aprime    P2: {} on the product grid = BM45 + profile_mean; named treatments and modes kept
  ② test_02_research_sigma_bar            sigma_bar == research Model.averages(shape)[1], rtol 1e-12, both q shapes
  ③ test_03_identity_and_conflicts        block_key / check_contract refuse cross consumption; settings conflicts
     test_03b_contract_hashes              signed_contract code_hashes and gate.code_identity carry the A′ assets
  ④ test_04_resolver                      per-state q shape, pinned mismatch, external tables, cfg never shared
     test_04b_resolver_call_sites          create_configured / plan_queue.probe / gate worker (FX-AP) go through the resolver
     test_04c_scheduler_presign            gpu_sched_v3 expected settings + pre-signature use the resolved cfg
  ⑤ test_05_commutation                   bin mean then layer mean == layer mean then bin mean (1e-12)
     test_05b_band_definitions             band vs point sampling (diagnostic, comparison-only definition)
  ⑦ test_07_asset_sha_and_temperature_range
  ⑨ test_09_constant_temperature_reduction  constant 213/233/253 K T(z) reduces to the fixed columns (1e-12)
Slow (SOSFUSE2_APRIME_RUNS=1; CPU-heavy, skipped otherwise):
  ⑥ test_06_reduced_transport             reduced NumPy solve, 3 wavelengths: production A′ == research A′ injection
  ⑧ test_08_cold_warm_alternation         D -> A′ -> D on one work root (caches kept) == fresh roots; M3; geometry-only

No GPU; the research modules are imported read-only.
"""

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from sosfuse2 import spectral as SP, ozone_temperature as OT, density, paths, testing
from sosfuse2 import run_gpu_state_v3 as V3

RESEARCH_REL = 'results/sigma_tz_20261028'  # sky-sos research record (A′ reference model); tests that need it skip without SOSFUSE2_SKY_SOS_ROOT
WL = np.asarray(SP.PRODUCT['wavelengths_nm'])
SHAPES = ('triangle', 'usstd76')
SELECTED = [492.1875, 600.875, 650.8125]  # research transport wavelengths (model.SELECTED)
SLOW = pytest.mark.skipif(
    os.environ.get('SOSFUSE2_APRIME_RUNS') != '1', reason='reduced NumPy solves, CPU-heavy: set SOSFUSE2_APRIME_RUNS=1'
)


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


def settings(raw, reduced=False):
    # Exercise the actual settings adapter without writing a settings file.
    path = SimpleNamespace(read_text=lambda: json.dumps(raw))
    return V3.settings_for(SimpleNamespace(reduced=reduced, settings_json=path))


def state(shape='triangle', **density_extra):
    return {'density': dict(ozone_profile=shape, **density_extra)}


FIXED = {'ozone_temperature': '233K'}  # the fixed 233 K product (D); {} is A′ since P2


def unresolved(**extra):
    return settings(dict(ozone_temperature='profile_mean', **extra))['spectral']


def block(shape='triangle', **extra):
    """profile_mean block resolved for a state of this q shape."""
    return SP.resolve(unresolved(**extra), state(shape))


def raises(code, call):
    with pytest.raises(ValueError, match='^' + code):
        call()


@pytest.fixture(scope='module')
def research():
    relative = Path(RESEARCH_REL)
    ROOT = testing.sky_sos(RESEARCH_REL, 'model.py').parents[
        2
    ]  # the sky-sos checkout; its assets/ are the research inputs
    spec = importlib.util.spec_from_file_location('aprime_reference', ROOT / relative / 'model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Only the ignored raw source may be absent in a worktree. Resolve that one manifest input from the main
    # checkout; still verify its full expected SHA. Model code and all numerical assets remain those of this tree.
    raw = relative / 'raw/serdyuchenkogorshelev5digits.dat'
    if not (ROOT / raw).is_file():
        common = subprocess.check_output(
            ['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'], cwd=ROOT, text=True
        )
        main_root = Path(common.strip()).parent
        original_sha = module.sha
        module.sha = lambda p: original_sha(main_root / raw if Path(p) == ROOT / raw else p)
    # D1: the research manifest pins the vacuum-axis assets it was frozen on. When the
    # tree carries a production build of the air-axis assets (build_spectra_fine.py, frozen_source_sha256 consistent), the
    # research code is used as an independent reference on those bytes; any other mismatch still raises.
    rebuilt = ('assets/spectra_fine.json', 'assets/spectra_fine_temperature.json')
    pinned = json.loads((ROOT / relative / 'out/assets_manifest.json').read_text())['files']
    t_asset = json.loads((ROOT / rebuilt[1]).read_text())
    if OT.sha(ROOT / rebuilt[0]) != pinned[rebuilt[0]]:
        assert t_asset['generator']['script'] == 'scripts/build_spectra_fine.py'
        assert t_asset['frozen_source_sha256'] == OT.sha(ROOT / rebuilt[0])
        checked_sha = module.sha
        module.sha = lambda p: (
            pinned[str(Path(p).relative_to(ROOT))] if Path(p) in [ROOT / r for r in rebuilt] else checked_sha(p)
        )
    model = module.Model()
    for name, expected in model.manifest['files'].items():
        if (name.startswith('assets/') or name.startswith('scripts/sosfuse2/assets/')) and name not in rebuilt:
            assert OT.sha(ROOT / name) == expected
    return model


def band_means(column, edges):
    """Independent restatement of the bin mean: trapezoid of the linear interpolant on the fine nodes + bin ends."""
    g = SP.asset()['wl']
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        x = np.unique(np.r_[lo, g[(g > lo) & (g < hi)], hi])
        out.append(np.trapezoid(np.interp(x, g, column), x) / (hi - lo))
    return np.array(out)


# ① ------------------------------------------------------------------------------------------------------------------
def test_01_default_path_unchanged(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('default path opened temperature assets')

    for name in ('identity', 'sigma_bar', 'state_shape', 'profile_identity'):
        monkeypatch.setattr(OT, name, forbidden)
    explicit = settings(FIXED)
    named = settings(
        {
            'spectral_mode': 'band',
            'bin_edges_nm': SP.PRODUCT_EDGES,
            'wavelengths': 45,
            'wavelengths_nm': SP.PRODUCT['wavelengths_nm'],
        }
    )
    assert named == explicit and set(explicit['spectral']) == {
        'mode',
        'asset',
        'asset_sha256',
        'ozone_column',
        'bin_edges_nm',
        'bin_edges_sha256',
        'physics',
        'minor_absorber_sha256',
        'point_definition',
    }
    A = SP.asset()
    # The fixed-233 K product (settings naming 233K, or naming spectral_mode like the gate fixtures): band mode, the
    # fixed 233 K column, and the SPEC table built from that very block.
    default = explicit['spectral']
    assert (default['mode'], default['ozone_column'], default['bin_edges_nm']) == (
        'band',
        'ozone_233K',
        list(SP.PRODUCT_EDGES),
    )
    table = SP.spec_table(default, WL)
    np.testing.assert_allclose(table['ozone'], band_means(A['ozone_233K'], SP.PRODUCT_EDGES), rtol=1e-12, atol=0)
    np.testing.assert_allclose(table['solar'], band_means(A['solar'], SP.PRODUCT_EDGES), rtol=1e-12, atol=0)
    for mode, temp in (('band', 233), ('band', 213), ('band', 253), ('point', 253), ('point', 233)):
        raw = dict(spectral_mode=mode, bin_edges_nm=SP.PRODUCT_EDGES, ozone_temperature_K=temp)
        b = SP.block(dict(raw), WL, True)
        b233 = SP.block(dict(raw, ozone_temperature='233K'), WL, True)
        from sosfuse2.components import physics

        assert (
            b
            == b233
            == dict(
                physics=physics(),
                mode=mode,
                asset='assets/spectra_fine.json',
                asset_sha256=A['sha'],
                ozone_column=f'ozone_{temp}K',
                bin_edges_nm=list(SP.PRODUCT_EDGES),
                bin_edges_sha256=SP.edges_sha(SP.PRODUCT_EDGES),
                minor_absorber_sha256=b['minor_absorber_sha256'],
                point_definition=b['point_definition'],
            )
        )
        assert (
            SP.block_key(b)
            == SP.block_key(b233)
            == (
                mode,
                A['sha'],
                f'ozone_{temp}K',
                SP.edges_sha(SP.PRODUCT_EDGES),
                json.dumps(b['physics'], sort_keys=True),
                json.dumps(b['minor_absorber_sha256'], sort_keys=True),
            )
        )
        table = SP.spec_table(b, WL)
        if mode == 'band':
            sigma, solar = band_means(A[f'ozone_{temp}K'], SP.PRODUCT_EDGES), band_means(A['solar'], SP.PRODUCT_EDGES)
        else:
            sigma, solar = np.interp(WL, A['wl'], A[f'ozone_{temp}K']), np.interp(WL, A['wl'], A['solar'])
        np.testing.assert_allclose(table['ozone'], sigma, rtol=1e-12, atol=0)
        np.testing.assert_allclose(table['solar'], solar, rtol=1e-12, atol=0)
        assert table['wavelength_nm'] == WL.tolist() and table['source'].endswith(f'ozone_{temp}K)')
        # The resolver and its consumer checks leave fixed-column blocks alone (same object, no hashes added).
        st = state('usstd76')
        assert SP.resolve(b, st) is b and SP.check_state(b, st) is None and SP.asset_hashes(b) == {}
    cfg = settings(FIXED)
    assert V3.resolve_settings(cfg, state('usstd76')) is cfg
    assert SP.block({}, WL, False) is None and SP.block({'ozone_temperature': '233K'}, WL, False) is None
    assert SP.resolve(None, state()) is None and SP.block_key(None) is None and SP.asset_hashes(None) == {}
    legacy = settings({'spectral_mode': 'legacy'})
    assert 'spectral' not in legacy and V3.resolve_settings(legacy, state()) is legacy


def test_01b_product_default_is_aprime():
    """P2: settings {} (the product, rig bakes, G-F) = BM45 bins + ozone profile_mean; a
    settings file naming ozone_temperature or a fixed ozone_temperature_K column, or naming spectral_mode, keeps it."""
    product = settings({})
    assert product == settings({'ozone_temperature': 'profile_mean'})
    b = product['spectral']
    assert b['ozone_temperature'] == 'profile_mean' and b['ozone_profile'] is None and 'ozone_column' not in b
    assert (b['mode'], b['bin_edges_nm'], product['wavelengths_nm']) == (
        'band',
        list(SP.PRODUCT_EDGES),
        SP.PRODUCT['wavelengths_nm'],
    )
    raises('E_SPECTRAL_UNRESOLVED', lambda: SP.spec_table(b, WL))  # consumed only after the per-state resolver
    for shape in SHAPES:
        resolved = V3.resolve_settings(product, state(shape))['spectral']
        assert resolved == block(shape)
        np.testing.assert_allclose(
            SP.spec_table(resolved, WL)['ozone'], OT.sigma_bar(SP.PRODUCT_EDGES, shape)[0], rtol=1e-12, atol=0
        )
    assert settings({'ozone_profile': 'usstd76'})['spectral'] == unresolved(
        ozone_profile='usstd76'
    )  # pinned q shape, A′
    # Named treatments: the fixed columns stay fixed columns on the product bins.
    for raw, column in (
        (FIXED, 'ozone_233K'),
        ({'ozone_temperature_K': 213}, 'ozone_213K'),
        ({'ozone_temperature_K': 253}, 'ozone_253K'),
        ({'ozone_temperature': '233K', 'ozone_temperature_K': 233}, 'ozone_233K'),
    ):
        fixed = settings(raw)['spectral']
        assert (
            fixed['ozone_column'] == column
            and 'ozone_temperature' not in fixed
            and fixed['bin_edges_nm'] == list(SP.PRODUCT_EDGES)
        )
    # A settings file that names its spectral_mode keeps the per-mode default (fixed 233 K); the reduced grid stays legacy.
    for mode in ('band', 'point'):
        assert settings({'spectral_mode': mode, 'wavelengths': 3})['spectral']['ozone_column'] == 'ozone_233K'
    assert 'spectral' not in settings({}, reduced=True) and 'spectral' not in settings({'spectral_mode': 'legacy'})
    raises('E_SPECTRAL_CONTRACT', lambda: settings({'ozone_profile': 'usstd76', 'ozone_temperature': '233K'}))


# ② ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('shape', SHAPES)
def test_02_research_sigma_bar(shape, research):
    got, meta = OT.sigma_bar(SP.PRODUCT_EDGES, shape)
    ref = research.averages(shape)[1]
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0)
    for path, key in ((OT.FINE, 'ozone_sigma_asset_sha256'), (OT.TEMP, 'temperature_asset_sha256')):
        assert meta[key] == OT.sha(path)  # the bytes read (pinned SHAs: research fixture; D1 rebuilds rebound there)
    profile_sha = research.manifest['files'][
        'scripts/sosfuse2/assets/usstd76_profile.json'
    ]  # the research manifest's own (sky-sos) key
    # The research manifest may still name an earlier generation of the same logical asset (density.ASSET_SHA256_SUPERSEDED).
    assert profile_sha == density.ASSET_SHA256 or profile_sha in density.ASSET_SHA256_SUPERSEDED
    assert OT.sha(density.ASSET) == density.ASSET_SHA256
    if shape == 'usstd76':
        assert meta['ozone_profile']['asset_sha256'] == density.ASSET_SHA256
    assert meta['knot_count'] == len(research.knots()) and meta['quadrature_n'] == 64
    # The production path: settings -> per-state resolver -> SPEC table.
    table = SP.spec_table(block(shape), WL)
    np.testing.assert_allclose(table['ozone'], ref, rtol=1e-12, atol=0)
    assert table['solar'] == SP.spec_table(settings(FIXED)['spectral'], WL)['solar']
    print(f'research {shape}: max_relative={np.max(abs(got / ref - 1)):.17g}, knots={meta["knot_count"]}, n=64')


# ③ ------------------------------------------------------------------------------------------------------------------
def test_03_identity_and_conflicts():
    default = settings(FIXED)['spectral']
    mean = block()
    assert 'ozone_column' not in mean and mean['ozone_temperature_rule'] == OT.RULE
    assert SP.block_key(default) != SP.block_key(mean) != SP.block_key(block('usstd76'))
    SP.check_contract(mean, copy.deepcopy(mean))
    usstd = block('usstd76')
    changed = copy.deepcopy(usstd)
    changed['ozone_profile']['asset_sha256'] = '0' * 64
    for table, weights in (
        (mean, default),
        (default, mean),
        (mean, block('usstd76')),
        (usstd, changed),
        (mean, dict(mean, temperature_asset_sha256='0' * 64)),
        (mean, dict(mean, ozone_sigma_asset_sha256='0' * 64)),
        (mean, dict(mean, ozone_temperature_rule='v0')),
        (None, mean),
    ):
        raises('E_SPECTRAL_CONTRACT', lambda: SP.check_contract(table, weights))
    # An unresolved settings block is never a consumable definition.
    loose = unresolved()
    assert loose['ozone_profile'] is None
    for call in (
        lambda: SP.block_key(loose),
        lambda: SP.spec_table(loose, WL),
        lambda: SP.check_contract(loose, loose),
        lambda: SP.table_weights(loose, WL),
        lambda: SP.check_asset(loose),
        lambda: SP.check_state(loose, state()),
    ):
        raises('E_SPECTRAL_UNRESOLVED', call)
    # Settings conflicts.
    for raw in (
        {'ozone_column': 'ozone_233K'},
        {'ozone_temperature_K': 233},
        {'spectral_mode': 'legacy'},
        {'spectral_mode': 'point'},
    ):
        raises('E_SPECTRAL_CONTRACT', lambda: settings(dict(ozone_temperature='profile_mean', **raw)))
    raises('E_SPECTRAL_CONTRACT', lambda: settings({'ozone_temperature': 'invalid'}))
    raises(
        'E_SPECTRAL_CONTRACT', lambda: settings(dict(FIXED, ozone_profile='usstd76'))
    )  # 233K takes no spectral q shape
    raises('E_SETTINGS_KEYS', lambda: settings(dict(FIXED, ozone_column='ozone_233K')))  # as before A′
    raises('E_SPECTRAL_CONTRACT', lambda: settings({'ozone_column': 'ozone_233K'}))  # {} is A′ since P2: refused there
    raises('E_OZONE_SHAPE', lambda: unresolved(ozone_profile='invalid'))
    raises(
        'E_SPECTRAL_CONTRACT',
        lambda: SP.block(dict(spectral_mode='legacy', ozone_temperature='profile_mean'), [550.0], False),
    )
    forged = dict(mean, ozone_column='ozone_233K')
    for call in (lambda: SP.block_key(forged), lambda: SP.spec_table(forged, WL)):
        raises('E_SPECTRAL_CONTRACT', call)
    # A′ changes sigma only: E and the colour weights equal the fixed-column definition on the same bins.
    weights, info = SP.table_weights(mean, WL)
    base, _ = SP.table_weights(default, WL)
    assert weights.shape == (45, 4) and info['spectral'] == mean and info['sources']['ozone'] == 'profile_mean'
    np.testing.assert_allclose(weights, base, rtol=1e-12, atol=0)


def test_03c_v4_fixed_provenance_lists(monkeypatch, tmp_path):
    """The v4 provenance inputs are fixed lists, kept equal to the tree, and a missing file is an error (not a gap)."""
    tree = {p.stem for p in paths.PKG.glob('*.py') if not p.name.startswith(('test_', '._'))} - {'testing', 'conftest'}
    assert tree == set(paths.PRODUCTION_MODULES), tree ^ set(
        paths.PRODUCTION_MODULES
    )  # add new modules to paths.PRODUCTION_MODULES
    assert len(set(paths.PRODUCTION_MODULES)) == len(paths.PRODUCTION_MODULES)
    assert {p.name for p in (paths.ACCEPTED).glob('*.py')} == set(V3.ACCEPTED_SOURCES)
    assert {p.name for p in (paths.PKG / 'host').glob('*.py')} == set(V3.HOST_MODULES)
    h = V3.package_hashes()
    assert set(h) >= {f'sosfuse2/{n}.py' for n in paths.PRODUCTION_MODULES} | {
        f'sosfuse2/assets/{n}' for n in V3.DATA_ASSETS
    } | {'sosfuse2/kernel.cu', 'sosfuse2/native/sh_cpu.cpp'}
    assert 'sosfuse2/testing.py' not in h and not any('/test_' in k for k in h)
    # version/commit are recorded but not signed: the signature is a function of the content hashes and the job only.
    st = json.loads((paths.ASSETS / 'states_v2.json').read_text())['states'][0]
    cfg = V3.resolve_settings(settings(FIXED), st)
    sig_a, _ = V3.signed_contract(st, cfg, 'numpy', False, 'depolarizing')
    monkeypatch.setattr(V3, 'package_info', lambda: dict(package_version='9.9.9', package_commit='deadbeef'))
    sig_b, _ = V3.signed_contract(st, cfg, 'numpy', False, 'depolarizing')
    assert sig_a == sig_b
    # a listed file that is missing raises E_PACKAGE_FILE_MISSING instead of shrinking the hash set.
    monkeypatch.setattr(paths, 'PRODUCTION_MODULES', paths.PRODUCTION_MODULES + ('no_such_module',))
    with pytest.raises(RuntimeError, match='^E_PACKAGE_FILE_MISSING'):
        V3.package_hashes()
    from sosfuse2 import gate

    with pytest.raises(gate.GateError, match='^E_PACKAGE_FILE_MISSING'):
        gate.code_identity()


def test_03b_contract_hashes(monkeypatch):
    st = json.loads((paths.ASSETS / 'states_v2.json').read_text())['states'][0]
    rows = {}
    for name, raw, shape in (
        ('default', FIXED, 'triangle'),
        ('mean_tri', dict(ozone_temperature='profile_mean'), 'triangle'),
        ('mean_us', dict(ozone_temperature='profile_mean'), 'usstd76'),
    ):
        current = dict(
            copy.deepcopy(st), density=dict(density_profile='exp8', ozone_profile=shape, rayleigh_column='keep')
        )
        cfg = V3.resolve_settings(settings(raw), current)
        rows[name] = V3.signed_contract(current, cfg, 'numpy', False, 'depolarizing')
    # v4 provenance keys are package-relative (paths.rel): 'sosfuse2/assets/...'. Under v4 package_hashes() lists every
    # data asset for every `contract` version (v3 hashed the A′ inputs only when the block used them), so the A′ keys are present in
    # all three rows; the A′-specific part is spectral.asset_hashes (test_01: {} for fixed blocks) and the signatures differ.
    extra = {paths.rel(OT.FINE), paths.rel(OT.TEMP)}
    assert extra == {'sosfuse2/assets/spectra_fine_temperature.json', 'sosfuse2/assets/temperature_usstd76.json'}
    assert extra <= set(rows['default'][1]) and extra <= set(rows['mean_tri'][1]) and extra <= set(rows['mean_us'][1])
    assert (
        density.ASSET_KEY in rows['mean_us'][1] and density.ASSET_KEY not in rows['mean_tri'][1]
    )  # logical key, state-dependent
    for p in (OT.FINE, OT.TEMP):
        assert rows['mean_tri'][1][paths.rel(p)] == OT.sha(p)
    assert len({sig for sig, _ in rows.values()}) == 3
    # signed_contract refuses a block that is unresolved or resolved for another state.
    current = dict(
        copy.deepcopy(st), density=dict(density_profile='exp8', ozone_profile='usstd76', rayleigh_column='keep')
    )
    for cfg, code in (
        (settings(dict(ozone_temperature='profile_mean')), 'E_SPECTRAL_UNRESOLVED'),
        (dict(settings(FIXED), spectral=block('triangle')), 'E_SPECTRAL_OZONE_PROFILE_MISMATCH'),
    ):
        raises(code, lambda: V3.signed_contract(current, cfg, 'numpy', False, 'depolarizing'))
    # gate.code_identity (staged bless identity) = package_hashes() minus gate.py + fixtures/configs + the ozone asset: the A′
    # assets and ozone_temperature.py enter through package_hashes, the ozone asset through its logical key.
    from sosfuse2 import gate

    before = gate.code_identity()
    h = V3.package_hashes()
    assert {paths.rel(OT.FINE), paths.rel(OT.TEMP), 'sosfuse2/ozone_temperature.py'} <= set(
        h
    ) and 'sosfuse2/gate.py' in h
    extra = {}
    monkeypatch.setattr(gate, 'file_sha', lambda p: extra.setdefault(Path(p), 'x'))
    gate.code_identity()
    assert (
        density.asset_path() in extra and paths.GATE / 'fixtures.json' in extra
    )  # asset under its logical key + gate inputs
    monkeypatch.undo()
    monkeypatch.setattr(
        V3, 'package_hashes', lambda: {k: v for k, v in h.items() if k != 'sosfuse2/ozone_temperature.py'}
    )
    assert gate.code_identity() != before  # the identity depends on package_hashes


# ④ ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('shape', SHAPES)
def test_04_resolver(shape):
    other = next(s for s in SHAPES if s != shape)
    loose = unresolved()
    before = copy.deepcopy(loose)
    b = SP.resolve(loose, state(shape))
    assert loose == before and b is not loose and b['ozone_profile'] == OT.profile_identity(shape)
    assert SP.resolve(unresolved(ozone_profile=shape), state(shape)) == b  # pinned and matching
    raises('E_SPECTRAL_OZONE_PROFILE_MISMATCH', lambda: SP.resolve(unresolved(ozone_profile=other), state(shape)))
    SP.check_state(b, state(shape))
    raises('E_SPECTRAL_OZONE_PROFILE_MISMATCH', lambda: SP.check_state(b, state(other)))
    assert SP.spec_table(b, WL) == SP.spec_table(SP.resolve(unresolved(), state(shape)), WL)
    # The state's options decide: no density block = triangle; CLI options through effective_state.
    assert SP.resolve(loose, {})['ozone_profile']['name'] == 'triangle'
    cli = V3.effective_state({'id': 'x', 'parameters': {}}, SimpleNamespace(ozone_profile=shape))
    assert SP.resolve(loose, cli)['ozone_profile']['name'] == shape
    # A finer PROFILE step keeps the same piecewise-linear q (density.build linear refinement): same identity, same SPEC.
    fine = SP.resolve(loose, state(shape, profile_step_km=0.125))
    assert fine == b and SP.spec_table(fine, WL) == SP.spec_table(b, WL)
    if shape == 'usstd76':
        optics = density.build(state(shape, profile_step_km=0.125))
        zq = np.linspace(0, 80, 1601)
        np.testing.assert_allclose(
            np.interp(zq, density.grid(641), optics['table'][:, 1]),
            OT.ProfileMean(SP.PRODUCT_EDGES, shape).q(zq),
            rtol=1e-12,
            atol=1e-15,
        )
    # The shared settings dict is copied, never changed; each state gets its own block.
    cfg = settings(dict(ozone_temperature='profile_mean'))
    snapshot = copy.deepcopy(cfg)
    one, two = V3.resolve_settings(cfg, state(shape)), V3.resolve_settings(cfg, state(other))
    assert cfg == snapshot and one is not cfg and one['spectral'] is not cfg['spectral']
    assert one['spectral']['ozone_profile']['name'] == shape and two['spectral']['ozone_profile']['name'] == other
    assert {k: v for k, v in one.items() if k != 'spectral'} == {k: v for k, v in cfg.items() if k != 'spectral'}


def test_04c_superseded_profile_asset_digest():
    """A block made with the 2026-10 float64 usstd76 asset (a7ccf7b1…, density.ASSET_SHA256_SUPERSEDED) names the same logical
    profile: check_asset/representative/resolve accept it and re-resolving records the current digest; an unknown digest stays
    refused (public split S4: tables baked before the decimal asset must stay readable)."""
    b = SP.resolve(unresolved(), state('usstd76'))
    old = copy.deepcopy(b)
    old['ozone_profile']['asset_sha256'] = density.ASSET_SHA256_SUPERSEDED[0]
    assert old['ozone_profile'] != b['ozone_profile'] and OT.same_profile_identity(
        old['ozone_profile'], b['ozone_profile']
    )
    SP.check_asset(old)
    np.testing.assert_array_equal(SP.representative(old, WL)[0], SP.representative(b, WL)[0])
    assert SP.resolve(old, state('usstd76'))['ozone_profile'] == b['ozone_profile']
    bad = copy.deepcopy(b)
    bad['ozone_profile']['asset_sha256'] = '0' * 64
    raises('E_OZONE_T_ASSET_SHA', lambda: SP.check_asset(bad))
    raises('E_OZONE_T_ASSET_SHA', lambda: SP.resolve(bad, state('usstd76')))
    assert not OT.same_profile_identity(dict(name='triangle'), b['ozone_profile'])
    assert OT.same_profile_identity(dict(name='triangle'), dict(name='triangle'))


def test_04_resolver_external_tables():
    z = density.grid(161)
    base = state('usstd76', density_profile='usstd76')
    air_only = density.with_profile(
        base,
        air_surface_number_density_m3=2.5469e25,
        z_km=z,
        table=np.column_stack([np.exp(-z / 8), np.zeros_like(z)]),
        mask=1,
    )
    ozone = density.with_profile(
        base,
        air_surface_number_density_m3=2.5469e25,
        z_km=z,
        table=np.column_stack([np.exp(-z / 8), np.maximum(0, 1 - abs(z - 22) / 12)]),
        mask=3,
        normalize_ozone=True,
    )
    # mask bit 2 off: the kernel keeps the analytic triangle although the option string still says usstd76.
    assert density.options(air_only)['ozone_profile'] == 'usstd76' and OT.state_shape(air_only) == 'triangle'
    assert SP.resolve(unresolved(), air_only)['ozone_profile'] == {'name': 'triangle'}
    raises('E_SPECTRAL_OZONE_PROFILE_MISMATCH', lambda: SP.resolve(unresolved(ozone_profile='usstd76'), air_only))
    raises('E_SPECTRAL_OZONE_PROFILE_MISMATCH', lambda: SP.resolve(unresolved(), ozone))
    raises('E_SPECTRAL_OZONE_PROFILE_MISMATCH', lambda: SP.check_state(block('usstd76'), ozone))


def test_04b_resolver_call_sites(tmp_path, monkeypatch):
    st = state('usstd76')
    loose = settings(dict(ozone_temperature='profile_mean'))
    # create_configured refuses before any work (no output folder made).
    for cfg, code in (
        (loose, 'E_SPECTRAL_UNRESOLVED'),
        (V3.resolve_settings(loose, state('triangle')), 'E_SPECTRAL_OZONE_PROFILE_MISMATCH'),
    ):
        raises(code, lambda: V3.create_configured(tmp_path / 'never', cfg, np.array(SELECTED), 'numpy', st))
    assert not (tmp_path / 'never').exists()
    captured = []

    class Stop(Exception):
        pass

    def capture(output, cfg, wavelengths, backend, s, *args, **kwargs):
        captured.append((cfg, s))
        raise Stop

    monkeypatch.setattr(V3, 'create_configured', capture)
    from sosfuse2 import plan_queue, gate

    snapshot = copy.deepcopy(loose)
    with pytest.raises(Stop):
        plan_queue.probe(st, loose, tmp_path / 'geo', tmp_path / 'probe')
    cfg, _ = captured.pop()
    assert loose == snapshot and cfg['spectral'] == SP.resolve(loose['spectral'], st)
    # gate worker: the A′ fixture FX-AP (P2) is resolved for the fixture state; the other fixtures keep the fixed 233 K
    # column (their shared settings name spectral_mode), and the shared settings are never changed.
    F = gate.load_fixtures()
    shared = copy.deepcopy(F['settings'])
    assert 'ozone_temperature' not in shared and gate.fixture_settings(F, 'FX-AP') == dict(
        shared, ozone_temperature='profile_mean'
    )
    assert all(gate.fixture_settings(F, fx) == shared for fx in F['fixtures'] if fx != 'FX-AP')
    assert F['fixtures']['FX-AP']['golden'] == 'FX-AP' and ['FX-AP', 'golden'] in F['ratchet_pairs']
    assert {k: v for k, v in F['fixtures']['FX-AP'].items() if k not in ('golden', 'settings', 'note')} == {
        k: v for k, v in F['fixtures']['FX-C3'].items() if k not in ('golden', 'note')
    }  # FX-C3 + A′, nothing else
    monkeypatch.setenv('SOSFUSE2_BLOCK_SIZE', '128')
    monkeypatch.delenv('STRATO_ATTACH_CACHE', raising=False)
    for fx in ('FX-AP', 'FX-C3'):
        with pytest.raises(Stop):
            gate.worker('fixed', fx, tmp_path / f'gate_{fx}', 'numpy')
        cfg, s = captured.pop()
        assert json.loads((tmp_path / f'gate_{fx}/settings.json').read_text()) == gate.fixture_settings(F, fx)
        if fx == 'FX-AP':
            assert (
                cfg['spectral']['ozone_profile'] == OT.profile_identity('usstd76')
                and s['density']['ozone_profile'] == 'usstd76'
            )
        else:
            assert cfg['spectral']['ozone_column'] == 'ozone_233K' and 'ozone_temperature' not in cfg['spectral']
    assert F['settings'] == shared
    # The reduced solve (path ①) gets the same per-fixture settings file.
    calls = []

    def fake_child(cmd, log, env, cwd=None, timeout=None):
        if '--settings-json' in cmd:
            calls.append(json.loads(Path(cmd[cmd.index('--settings-json') + 1]).read_text()))
        return 0, [], None

    monkeypatch.setattr(gate, 'child', fake_child)
    for fx in ('FX-AP', 'FX-N'):
        with pytest.raises(OSError):
            gate.run_fixture(F, fx, tmp_path / f'solve_{fx}', 'numpy', True)  # no pipeline output: stops after the call
    assert calls == [gate.fixture_settings(F, 'FX-AP'), shared]
    # check_config / evaluate refuse a run whose recorded block is not the fixture's ozone treatment (D for FX-AP, A′ for FX-C3).
    mean, fixed = block('usstd76'), settings(FIXED)['spectral']
    assert gate.ozone_mismatch(F, 'FX-AP', mean) == '' and gate.ozone_mismatch(F, 'FX-C3', fixed) == ''
    for fx, b in (('FX-AP', fixed), ('FX-C3', mean), ('FX-AP', unresolved())):
        assert gate.ozone_mismatch(F, fx, b).startswith('ozone_temperature requested')


def test_04c_scheduler_presign(tmp_path):
    """rig/gpu_sched_v3.py signs and checks prefetch settings with the cfg the runner consumes (resolved per state)."""
    spec = importlib.util.spec_from_file_location(
        'aprime_sched', testing.sky_sos('rig/gpu_sched_v3.py')
    )  # sky-sos rig scheduler (consumer)
    sched = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sched)
    st = next(s for s in V3.read_states(paths.ASSETS / 'states_v1_usstd.json', True) if s.get('runnable'))
    assert st['density']['ozone_profile'] == 'usstd76'
    path = tmp_path / 'aprime.json'
    path.write_text(json.dumps(dict(ozone_temperature='profile_mean')))
    raws = V3.settings_for(SimpleNamespace(reduced=False, settings_json=path))
    raises(
        'E_SPECTRAL_UNRESOLVED', lambda: V3.signed_contract(st, raws, 'cupy', False, 'depolarizing')
    )  # the old pre-sign
    cfg = sched.job_settings(V3, str(path), st, False)
    assert cfg == V3.resolve_settings(raws, V3.effective_state(st, SimpleNamespace()))
    assert (
        cfg['spectral']['ozone_profile'] == OT.profile_identity('usstd76')
        and raws['spectral'] == settings(dict(ozone_temperature='profile_mean'))['spectral']
    )
    # The runner signs (effective_state, resolve_settings(settings_for)) — verify_pack needs the same signature here.
    eff = V3.effective_state(st, SimpleNamespace(stratosphere=None))
    runner = V3.signed_contract(eff, V3.resolve_settings(raws, eff), 'cupy', False, 'depolarizing')[0]
    assert V3.signed_contract(st, cfg, 'cupy', False, 'depolarizing')[0] == runner
    # The default job ({} = A′ since P2) is resolved for the state like the A′ file; a fixed-column job keeps the plain
    # settings_for cfg; a runner without a resolver (sosfuse) is untouched.
    plain_default = V3.settings_for(SimpleNamespace(reduced=False, settings_json=None))
    assert sched.job_settings(V3, None, st, False) == cfg and plain_default['spectral'] == raws['spectral']
    fixed = tmp_path / 'fixed.json'
    fixed.write_text(json.dumps(FIXED))
    assert sched.job_settings(V3, str(fixed), st, False) == V3.settings_for(
        SimpleNamespace(reduced=False, settings_json=fixed)
    )
    plain = SimpleNamespace(settings_for=lambda a: {'nr': 1})
    assert sched.job_settings(plain, None, st, False) == {'nr': 1}


# ⑤ ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('shape', SHAPES)
def test_05_commutation(shape):
    """Layer-average the fine 0.1 nm sigma(T) columns first, then take bin means: equal to the production order."""
    model = OT.ProfileMean(SP.PRODUCT_EDGES, shape)
    fine = json.loads(OT.FINE.read_bytes())
    columns = np.array([fine[f'ozone_{int(t)}K'] for t in OT.TEMPS])
    z, w = model.quadrature()
    qw = model.q(z) * w
    t = model.temp(z)
    i = np.minimum(np.searchsorted(OT.TEMPS, t, side='right') - 1, len(OT.TEMPS) - 2)
    f = (t - OT.TEMPS[i]) / 10
    layer = np.empty(columns.shape[1])
    for lo in range(0, len(layer), 256):
        hi = lo + 256
        layer[lo:hi] = qw @ (columns[i, lo:hi] * (1 - f[:, None]) + columns[i + 1, lo:hi] * f[:, None]) / qw.sum()
    layer_then_band = band_means(layer, SP.PRODUCT_EDGES)
    band_then_layer, _ = OT.sigma_bar(SP.PRODUCT_EDGES, shape)
    err = np.max(abs(layer_then_band / band_then_layer - 1))
    print(f'commutation {shape}: max relative {err:.3g}')
    assert err <= 1e-12


@pytest.mark.parametrize('shape', SHAPES)
def test_05b_band_definitions(shape, research):
    b = block(shape)
    band, solar = SP.representative(b, WL)
    point, point_solar = SP.representative(b, WL, band_mean_of='point')
    np.testing.assert_array_equal(solar, point_solar)
    # Independently use the research vertical integral with wavelength point columns.
    fine = json.loads(OT.FINE.read_bytes())
    reference = copy.copy(research)
    reference.sigmas = np.array([np.interp(WL, fine['wavelength_nm'], fine[f'ozone_{int(t)}K']) for t in OT.TEMPS])
    np.testing.assert_allclose(point, reference.averages(shape)[1], rtol=1e-12, atol=0)
    relative = abs(band / point - 1)
    assert len(relative) == 45 and np.isfinite(relative).all() and np.any(relative > 0)
    print(f'band_mean_of {shape}: abs(band/point-1) max={relative.max():.17g} mean={relative.mean():.17g}')


# ⑦ ------------------------------------------------------------------------------------------------------------------
def test_07_asset_sha_and_temperature_range(monkeypatch):
    b = block('usstd76')
    for key in ('temperature_asset_sha256', 'ozone_sigma_asset_sha256', 'ozone_profile'):
        changed = copy.deepcopy(b)
        if key == 'ozone_profile':
            changed[key]['asset_sha256'] = '0' * 64
        else:
            changed[key] = '0' * 64
        raises('E_OZONE_T_ASSET_SHA', lambda: SP.spec_table(changed, WL))
    pinned = unresolved(ozone_profile='usstd76')
    pinned['ozone_profile']['asset_sha256'] = '0' * 64
    raises('E_OZONE_T_ASSET_SHA', lambda: SP.resolve(pinned, state('usstd76')))
    original = Path.read_bytes
    for target in (OT.FINE, OT.TEMP, density.ASSET):
        with monkeypatch.context() as patch:
            patch.setattr(Path, 'read_bytes', lambda p: original(p) + (b' ' if p == target else b''))
            raises('E_OZONE_T_ASSET_SHA', lambda: SP.spec_table(b, WL))
    model = OT.ProfileMean(SP.PRODUCT_EDGES, 'triangle')
    for value in (192.999, 293.001, np.nan, np.inf, -np.inf):
        raises('E_OZONE_T_RANGE', lambda: model.sigma(value))
    np.testing.assert_array_equal(model.sigma(193.0), model.sigmas[0])
    np.testing.assert_array_equal(model.sigma(293.0), model.sigmas[-1])
    damaged = json.loads(original(OT.TEMP))
    damaged['temperature_K'][0] = 294.0
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'read_bytes', lambda p: encoded(damaged) if p == OT.TEMP else original(p))
        raises('E_OZONE_T_RANGE', lambda: OT.sigma_bar(SP.PRODUCT_EDGES, 'triangle'))


# ⑨ ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('temperature', (213, 233, 253))
def test_09_constant_temperature_reduction(shape, temperature):
    """A constant T(z) at a fixed column: the layer mean is that column's bin mean (the 213/233/253 K blocks)."""
    model = OT.ProfileMean(SP.PRODUCT_EDGES, shape)
    model.t = np.full_like(model.t, float(temperature))
    fixed = SP.spec_table(
        SP.block(dict(spectral_mode='band', bin_edges_nm=SP.PRODUCT_EDGES, ozone_temperature_K=temperature), WL, True),
        WL,
    )['ozone']
    err = np.max(abs(model.average() / np.asarray(fixed) - 1))
    print(f'constant {temperature} K {shape}: max relative {err:.3g}')
    assert err <= 1e-12


# ⑥ ------------------------------------------------------------------------------------------------------------------
def research_transport():
    sys.path.insert(0, str(testing.sky_sos(RESEARCH_REL)))
    import transport

    return transport


def reduced_solve(T, cfg, st, out):
    """One monochromatic solve per SELECTED wavelength (the kernel specializes to 1 or 15 wavelengths; lanes are
    independent), stacked on the wavelength axis."""
    from sosfuse2 import fused
    from sosfuse2.geometry import TransientGeometry

    rows, ext, evaluations = [], [], []
    for i, w in enumerate(SELECTED):
        s, scope = V3.create_configured(out / f'w{i}', cfg, [w], 'numpy', st)
        geo = TransientGeometry(s, scope)

        def observer(field):
            f = T.full_source(s, field)
            return np.stack(
                [
                    s.query(h, np.array([2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 85.0]), np.zeros(7), f)[..., 0]
                    for h in (-1.0, -3.0, -6.0, -9.0, -12.0)
                ]
            )

        field, stats = fused.Solver(s, geo, observer=observer).solve(64, 4e-5)
        rows.append(T.queries(s, field))
        ext.append(np.array(s.ext, float)[:, 0])
        evaluations.append(stats['evaluations'])
    return np.stack(rows, -1), np.stack(ext, -1), dict(evaluations=evaluations)


@SLOW
@pytest.mark.parametrize('shape', SHAPES)
def test_06_reduced_transport(shape, tmp_path):
    """Research mac_case grid (states_v2[0], albedo .3, 300 DU, 3 wavelengths, tolerance 4e-5): the production
    resolver path and the research Aprime injection (transport.research) give the same solve."""
    T = research_transport()
    m = T.Model()
    st = copy.deepcopy(json.loads((paths.ASSETS / 'states_v2.json').read_text())['states'][0])
    st['parameters'].update(ozone_du=300.0, albedo=0.3)
    st['density'] = dict(density_profile='exp8', ozone_profile=shape, rayleigh_column='keep')
    base = T.configuration()  # reduced grid + BM45 233 K block, 45 wavelengths
    mean = SP.block(
        dict(spectral_mode='band', bin_edges_nm=SP.PRODUCT_EDGES, ozone_temperature='profile_mean'), WL, True
    )
    production = V3.resolve_settings(dict(base, spectral=mean), st)
    with T.research(m, 'D', shape, state=st):  # same scoped environment (disk caches off), no SPEC swap
        ours, ext_ours, stats_ours = reduced_solve(T, production, st, tmp_path / 'production')
    with T.research(m, 'Aprime', shape, state=st):
        ref, ext_ref, stats_ref = reduced_solve(T, base, st, tmp_path / 'research')
    np.testing.assert_allclose(ext_ours, ext_ref, rtol=1e-12, atol=0)
    fixed = np.asarray(SP.spec_table(base['spectral'], WL)['ozone'])[[list(WL).index(w) for w in SELECTED]]
    live = np.max(abs(ext_ours[2] / (fixed * st['parameters']['ozone_du'] * 2.687e20 / 15) - 1))
    rel_rms = float(np.sqrt(np.mean((ours - ref) ** 2)) / max(float(np.sqrt(np.mean(ref**2))), 1e-30))
    max_rel = float(np.max(abs(ours - ref) / np.maximum(abs(ref), 1e-30)))
    print(
        f'⑥ {shape}: A′ ozone coefficient vs 233 K max relative {live:.3g}; production vs research radiance rel RMS '
        f'{rel_rms:.3g}, max relative {max_rel:.3g}; evaluations {stats_ours["evaluations"]} / {stats_ref["evaluations"]}'
    )
    assert live > 1e-3  # the opt-in is live (sigma_bar, not the 233 K column)
    assert rel_rms <= 3e-5  # research reduced radiance target (transport.compare_mac)


# ⑧ ------------------------------------------------------------------------------------------------------------------
USSTD = (
    '--density-profile',
    'usstd76',
    '--ozone-profile',
    'usstd76',
    '--rayleigh-column',
    'usstd',
    '--stratosphere',
    'on',
)
D_SETTINGS = {'wavelengths': 3, 'spectral_mode': 'band'}
A_SETTINGS = dict(D_SETTINGS, ozone_temperature='profile_mean')


def runner(work, out_name, raw, *extra):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    s = work.parent / f'{work.name}.{out_name}.settings.json'
    s.write_text(json.dumps(raw))
    cmd = [
        *paths.module_command('run_gpu_state_v3'),
        '--id',
        'clear_33',
        '--out-name',
        out_name,
        '--work-root',
        str(work),
        '--backend',
        'numpy',
        '--reduced',
        '--settings-json',
        str(s),
        *USSTD,
        *extra,
    ]
    log = work.parent / f'{work.name}.{out_name}.log'
    keep = (
        'SOSFUSE2_WORK_ROOT',
        'SOSFUSE2_OFFLINE',
    )  # location of the work root / the IPRT offline switch must reach the child
    env = paths.child_env(
        {
            k: v
            for k, v in os.environ.items()
            if (not k.startswith('SOSFUSE2_') or k in keep) and k != 'STRATO_ATTACH_CACHE'
        }
    )
    with log.open('w') as f:
        rc = subprocess.run(cmd, cwd=paths.SRC, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    if rc:
        raise RuntimeError(f'{out_name}: exit {rc}, see {log}: {log.read_text()[-2000:]}')
    return log


def table(work, name):
    with np.load(Path(work) / 'out' / name / 'table.npz') as z:
        return z['radiance'].astype(float), z['solar_elevation_deg'].astype(float)


def max_rel(a, b):
    return float(np.max(abs(a - b) / np.maximum(abs(b), 1e-30)))


@SLOW
def test_08_cold_warm_alternation(tmp_path):
    """D (233 K) -> A′ -> D on one work root (geometry, attach, column and host caches kept; usstd76 density and ozone,
    stratospheric layer on) against fresh work roots, final table tol rtol 1e-5 (cold/warm); plus the M3
    extra-elevation path from the A′ source and the geometry-only path, both through the resolver."""
    with ThreadPoolExecutor(3) as ex:
        cold = {
            k: ex.submit(runner, tmp_path / f'cold_{k}', k, raw) for k, raw in (('d0', D_SETTINGS), ('a0', A_SETTINGS))
        }
        geo = ex.submit(runner, tmp_path / 'geo_only', 'g0', A_SETTINGS, '--geometry-only')
        shared = tmp_path / 'shared'
        logs = {k: runner(shared, k, raw) for k, raw in (('d1', D_SETTINGS), ('a1', A_SETTINGS), ('d2', D_SETTINGS))}
        logs['x1'] = runner(shared, 'x1', A_SETTINGS, '--source-out-name', 'a1', '--extra-elevations=-4.5')
        for k, f in cold.items():
            logs[k] = f.result()
        geo.result()
    R = {k: table(shared, k)[0] for k in ('d1', 'a1', 'd2')}
    R.update(d0=table(tmp_path / 'cold_d0', 'd0')[0], a0=table(tmp_path / 'cold_a0', 'a0')[0])
    diffs = dict(
        a1_vs_a0=max_rel(R['a1'], R['a0']),
        d1_vs_d0=max_rel(R['d1'], R['d0']),
        d2_vs_d0=max_rel(R['d2'], R['d0']),
        a0_vs_d0=max_rel(R['a0'], R['d0']),
    )

    def caches(k):
        out = {}
        for line in logs[k].read_text().splitlines():
            e = json.loads(line) if line.startswith('{"event": "PHASE"') else {}
            if e.get('phase') == 'attach_cache':
                out['attach'] = e.get('result')
            if e.get('phase') == 'geometry_done':
                out.update(geometry=e.get('cache'), common=e.get('common'), column=e.get('column'))
        return out

    hits = {k: caches(k) for k in ('d1', 'a1', 'd2')}
    hcache = {
        k: len(list(Path(root).rglob('*.hcache')))
        for k, root in (('shared', shared), ('cold_d0', tmp_path / 'cold_d0'), ('cold_a0', tmp_path / 'cold_a0'))
    }
    print(
        '⑧ max relative', json.dumps(diffs), 'cache events', json.dumps(hits), 'host_cache entries', json.dumps(hcache)
    )
    assert diffs['a1_vs_a0'] <= 1e-5 and diffs['d1_vs_d0'] <= 1e-5 and diffs['d2_vs_d0'] <= 1e-5
    assert diffs['a0_vs_d0'] > 1e-6  # A′ is live in the solve
    # The warm runs reused the caches built by the other definition (attach, geometry, column, host sun02 entries).
    assert hits['d1']['attach'] == 'miss' and hits['d1']['geometry'] == 'miss'
    for k in ('a1', 'd2'):
        assert hits[k]['attach'] == 'hit' and hits[k]['geometry'] == 'hit' and hits[k]['column'] == 'cached'
    assert hcache['shared'] == hcache['cold_d0'] == hcache['cold_a0'] >= 1
    meta = {k: json.loads((shared / 'out' / k / 'meta.json').read_text()) for k in ('d1', 'a1', 'd2', 'x1')}
    assert meta['a1']['settings']['spectral']['ozone_profile'] == OT.profile_identity('usstd76')
    assert {paths.rel(OT.FINE), paths.rel(OT.TEMP)} <= set(meta['a1']['code_hashes'])
    assert meta['d1']['signature'] == meta['d2']['signature'] != meta['a1']['signature']
    assert meta['x1']['settings'] == meta['a1']['settings']
    x, hx = table(shared, 'x1')
    a, ha = table(shared, 'a1')
    for h in (-3.0, -6.0):
        np.testing.assert_allclose(x[list(hx).index(h)], a[list(ha).index(h)], rtol=1e-5, atol=0)
    ready = json.loads((tmp_path / 'geo_only' / 'prefetch' / 'g0' / 'geometry_ready.json').read_text())
    assert ready['settings']['spectral']['ozone_profile'] == OT.profile_identity('usstd76')
