"""Spectral sampling modes of the solar spectrum and ozone cross section.

  legacy  assets/spectra.json read as point values (the accepted behaviour; settings carry no 'spectral' block)
  point   values sigma(l_i), E(l_i) of assets/spectra_fine.json (adding a wavelength never moves another); sigma is a point
          value, E the value of the filtered solar spectrum (0.1 nm cell means of TSIS-1 HSRS)
  band    arithmetic means over explicit bins [a_i, b_i): trapezoid of the 0.1 nm fine grid; default bins
          linspace(360, 830, n+1); explicit wavelengths_nm + band requires bin_edges_nm (E_SPECTRAL_EDGES)

The engine reads its spectra only through the SPEC dict injected in accepted/engine.py create() (coefficients() and Solver.solar,
both np.interp). spec_table() returns a table whose nodes are exactly the engine wavelengths, so the interpolation returns
the representative values unchanged and the accepted solver code is untouched. The block returned by block() goes into
the settings (and so into the signed inputs, the table meta and every consumer check); legacy returns None so legacy
settings bytes do not change.

ozone_temperature: '233K' = the fixed ozone_temperature_K column (block,
block_key and SPEC unchanged; the default of every settings file that names its spectral_mode); 'profile_mean' = A′, band
mode only: sigma_bar of ozone_temperature.py with the temperature/sigma(T) asset SHAs, the averaging rule and the q identity
in the block. Product default since P2: settings {} on the product grid mean BM45 + profile_mean;
naming ozone_temperature or ozone_temperature_K keeps the named treatment (opt-out: {"ozone_temperature": "233K"}). A settings block has no state, so
its q shape is resolved per state by resolve(block, state) (run_gpu_state_v3.resolve_settings); an unresolved block is
refused by every consumer (E_SPECTRAL_UNRESOLVED) and a pinned settings ozone_profile that is not the state's shape is
E_SPECTRAL_OZONE_PROFILE_MISMATCH.
"""

import copy
import hashlib
import json
import math

import numpy as np

from . import paths, identity as source_identity

ASSET = paths.ASSETS / 'spectra_fine.json'
CMF_1NM = paths.ASSETS / 'cmf_1nm.json'  # Mitsuba 3.9.1 CIE 1931 XYZ 1 nm
LEF = paths.ASSETS / 'lef_cie.json'  # colour-science 0.4.7 V_scotopic (was assets/lef_cie.json)
SPECTRA_48 = paths.ASSETS / 'spectra.json'  # 48-band legacy solar spectrum (engine SPEC dict)
MODES = ('legacy', 'point', 'band')
OZONE_COLUMNS = (213, 233, 253)
KEYS = ('spectral_mode', 'bin_edges_nm', 'ozone_temperature_K', 'ozone_temperature', 'ozone_profile', 'ozone_column')
KM, KS = 683.0, 1700.0
_CACHE = {}
# Product default: band BM45 = W40 bins with 489-501, 512.75-524.5, 524.5-536.25, 559.75-571.5 and
# 642-653.75 nm halved, ozone A′ profile_mean since P2. Settings {} on the product grid mean this;
# a settings file that changes the wavelengths must name its spectral_mode (legacy included).
PRODUCT_EDGES = [
    360.0,
    371.75,
    383.5,
    395.25,
    407.0,
    418.75,
    430.5,
    442.25,
    454.0,
    465.75,
    477.5,
    489.25,
    495.125,
    501.0,
    512.75,
    518.625,
    524.5,
    530.375,
    536.25,
    548.0,
    559.75,
    565.625,
    571.5,
    583.25,
    595.0,
    606.75,
    618.5,
    630.25,
    642.0,
    647.875,
    653.75,
    665.5,
    677.25,
    689.0,
    700.75,
    712.5,
    724.25,
    736.0,
    747.75,
    759.5,
    771.25,
    783.0,
    794.75,
    806.5,
    818.25,
    830.0,
]
PRODUCT = dict(
    spectral_mode='band',
    wavelengths=45,
    bin_edges_nm=PRODUCT_EDGES,
    wavelengths_nm=[(a + b) / 2 for a, b in zip(PRODUCT_EDGES[:-1], PRODUCT_EDGES[1:])],
)
WAVELENGTH_KEYS = ('wavelengths', 'wavelengths_nm')


def product_default(spectral_raw, raw):
    """Fill the product BM45 block into spectral_raw/raw when neither names a mode nor a wavelength grid (in place); the
    ozone treatment is A′ unless the settings name one (ozone_temperature, or a fixed ozone_temperature_K column)."""
    if 'spectral_mode' in spectral_raw:
        return
    if 'bin_edges_nm' in spectral_raw or any(k in raw for k in WAVELENGTH_KEYS):
        raise ValueError(
            'E_SPECTRAL_MODE: settings that change the wavelengths or bins must name spectral_mode (default band BM45)'
        )
    spectral_raw.update(spectral_mode='band', bin_edges_nm=list(PRODUCT_EDGES))
    if 'ozone_temperature' not in spectral_raw and 'ozone_temperature_K' not in spectral_raw:
        spectral_raw['ozone_temperature'] = 'profile_mean'
    raw.update(wavelengths=PRODUCT['wavelengths'], wavelengths_nm=list(PRODUCT['wavelengths_nm']))


def asset():
    data = ASSET.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if _CACHE.get('raw_sha256') != digest:
        doc = json.loads(data)
        _CACHE.update(
            sha=source_identity.asset_identity(data, 'spectra_fine.json'),
            raw_sha256=digest,
            wl=np.asarray(doc['wavelength_nm'], float),
            solar=np.asarray(doc['solar'], float),
            **{f'ozone_{t}K': np.asarray(doc[f'ozone_{t}K'], float) for t in OZONE_COLUMNS},
        )
    return _CACHE


def edges_sha(edges):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(edges, float)).tobytes()).hexdigest()


def block(raw, wavelengths, explicit_wavelengths):
    """Pop the spectral keys from a settings dict and return the spectral settings block (None for legacy).

    profile_mean: an optional settings ozone_profile pins the q shape (checked against every state by resolve());
    omitted, the block stays unresolved (ozone_profile None) until resolve() fills it from the state."""
    temperature = raw.pop('ozone_temperature', '233K')
    if temperature not in ('233K', 'profile_mean'):
        raise ValueError('E_SPECTRAL_CONTRACT: ozone_temperature in 233K, profile_mean')
    if temperature == '233K':
        if 'ozone_column' in raw:
            raise ValueError('E_SETTINGS_KEYS: unknown settings')  # never a settings key outside A′
        if 'ozone_profile' in raw:
            raise ValueError('E_SPECTRAL_CONTRACT: spectral ozone_profile requires profile_mean')
    elif 'ozone_column' in raw or 'ozone_temperature_K' in raw:
        raise ValueError(
            'E_SPECTRAL_CONTRACT: profile_mean takes no ozone_column/ozone_temperature_K; fixed columns use ozone_temperature_K'
        )
    shape = raw.pop('ozone_profile', None)
    mode = raw.pop('spectral_mode', 'legacy')
    edges = raw.pop('bin_edges_nm', None)
    temp = raw.pop('ozone_temperature_K', 233)
    if mode not in MODES:
        raise ValueError('E_SPECTRAL_MODE: spectral_mode in ' + ','.join(MODES))
    if temperature == 'profile_mean' and mode != 'band':
        raise ValueError('E_SPECTRAL_CONTRACT: profile_mean requires band mode (point and legacy are out of scope)')
    if mode == 'legacy':
        if edges is not None or temp != 233:
            raise ValueError('E_SPECTRAL_MODE: legacy takes no bin_edges_nm/ozone_temperature_K')
        return None
    if temp not in OZONE_COLUMNS:
        raise ValueError('E_SPECTRAL_OZONE_T: ozone_temperature_K in 213, 233, 253')
    wl = np.asarray(wavelengths, float)
    n = len(wl)
    if edges is None:
        if mode == 'band' and explicit_wavelengths:
            raise ValueError('E_SPECTRAL_EDGES: explicit wavelengths_nm in band mode need bin_edges_nm')
        edges = np.linspace(360, 830, n + 1)
    edges = np.asarray(edges, float)
    a = asset()['wl']
    if (
        edges.shape != (n + 1,)
        or not np.isfinite(edges).all()
        or np.any(np.diff(edges) <= 0)
        or edges[0] < a[0]
        or edges[-1] > a[-1]
    ):
        raise ValueError('E_SPECTRAL_EDGES: n+1 increasing bin edges inside the fine asset (360-840 nm) required')
    if np.any(wl < edges[:-1]) or np.any(wl >= edges[1:]):
        raise ValueError('E_SPECTRAL_EDGES: each wavelength must lie in its bin [a_i, b_i)')
    from .components import physics

    result = dict(
        physics=physics(),
        mode=mode,
        asset='assets/spectra_fine.json',
        asset_sha256=asset()['sha'],
        ozone_column=f'ozone_{temp}K',
        bin_edges_nm=edges.tolist(),
        bin_edges_sha256=edges_sha(edges),
    )
    from .minor_absorbers import hashes

    result.update(
        minor_absorber_sha256=hashes(),
        point_definition={'o3': 'point', 'no2': '0.1 nm cell-mean interpolant', 'o4': '0.1 nm cell-mean interpolant'},
    )
    if temperature == 'profile_mean':
        from . import ozone_temperature as OT

        del result['ozone_column']
        result.update(ozone_temperature=temperature, **OT.identity(shape))
    return result


def _profile_mean(b):
    """True for an A′ block; refuses malformed or unresolved ones (consumers need the resolved q identity)."""
    if b is None or b.get('ozone_temperature') != 'profile_mean':
        return False
    from . import ozone_temperature as OT

    if 'ozone_column' in b:
        raise ValueError('E_SPECTRAL_CONTRACT: profile_mean takes no ozone_column')
    if b.get('mode') != 'band':
        raise ValueError('E_SPECTRAL_CONTRACT: profile_mean requires band mode')
    if b.get('ozone_temperature_rule') != OT.RULE:
        raise ValueError('E_SPECTRAL_CONTRACT: profile_mean averaging rule differs from this code')
    if not isinstance(b.get('ozone_profile'), dict):
        raise ValueError(
            'E_SPECTRAL_UNRESOLVED: profile_mean block without a state q identity (spectral.resolve(block, state) first)'
        )
    return True


def check_asset(b):
    from .components import check_physics

    if b is not None:
        check_physics(b.get('physics'))
    if b is not None:
        _minor_contract(b)
    if b is not None and source_identity.asset_digest('spectra_fine.json', b['asset_sha256']) != asset()['sha']:
        raise ValueError('E_SPECTRAL_CONTRACT: assets/spectra_fine.json changed since the block was made')
    if _profile_mean(b):
        from . import ozone_temperature as OT

        expected = OT.identity(b['ozone_profile'].get('name'))
        if not OT.identity_matches(
            b, expected
        ):  # ozone_profile: equal up to superseded asset digests (density.ASSET_SHA256_SUPERSEDED)
            raise ValueError('E_OZONE_T_ASSET_SHA: profile_mean assets changed since the block was made')


def resolve(b, state):
    """The block for one state (A′ resolver): a profile_mean block gets the
    q identity of the shape the engine uses for this state (ozone_temperature.state_shape), as a new dict; a pinned
    settings ozone_profile that differs is E_SPECTRAL_OZONE_PROFILE_MISMATCH. Any other block (None, fixed 213/233/253 K
    columns) is returned as the same object, so the default path is untouched."""
    if b is None or b.get('ozone_temperature') != 'profile_mean':
        return b
    from . import ozone_temperature as OT

    shape = OT.state_shape(state)
    pinned = b.get('ozone_profile')
    if pinned is not None and (not isinstance(pinned, dict) or pinned.get('name') != shape):
        raise ValueError(f"E_SPECTRAL_OZONE_PROFILE_MISMATCH: settings ozone_profile {pinned!r} vs state {shape}")
    out = copy.deepcopy(b)
    out['ozone_profile'] = OT.profile_identity(shape)
    if pinned is not None and not OT.same_profile_identity(
        pinned, out['ozone_profile']
    ):  # a superseded digest re-resolves to the current one
        raise ValueError('E_OZONE_T_ASSET_SHA: ozone profile asset changed since the settings block was made')
    check_asset(out)
    return out


def check_state(b, state):
    """Refuse a profile_mean block not resolved for this state (create_configured, signed_contract); no-op otherwise."""
    if b is None or b.get('ozone_temperature') != 'profile_mean':
        return
    _profile_mean(b)
    from . import ozone_temperature as OT

    shape = OT.state_shape(state)
    if b['ozone_profile'].get('name') != shape:
        raise ValueError(
            f"E_SPECTRAL_OZONE_PROFILE_MISMATCH: block ozone_profile {b['ozone_profile'].get('name')} vs state {shape}"
        )


def asset_hashes(b):
    """{repository path: sha256} of the A′ inputs of a resolved profile_mean block ({} otherwise), for code_hashes."""
    if not _profile_mean(b):
        return {}
    from . import ozone_temperature as OT

    return {
        key: source_identity.asset_sha(p) for key, p in OT.asset_items(b['ozone_profile'].get('name'))
    }


def _fine_grid(lo, hi):
    """Fine nodes inside [lo, hi] plus the two ends (trapezoid of the linear interpolant, exact for non-grid edges)."""
    g = asset()['wl']
    return np.unique(np.r_[lo, g[(g > lo) & (g < hi)], hi])


def band_mean(y, lo, hi):
    x = _fine_grid(lo, hi)
    return np.trapezoid(np.interp(x, asset()['wl'], y), x) / (hi - lo)


def representative(b, wavelengths, *, band_mean_of='band'):
    """(sigma_ozone [m^2], E_sun [W m^-2 nm^-1]) at the engine wavelengths for a non-legacy block."""
    check_asset(b)
    A = asset()
    wl = np.asarray(wavelengths, float)
    if _profile_mean(b):
        from . import ozone_temperature as OT

        sigma, meta = OT.sigma_bar(
            b['bin_edges_nm'], b['ozone_profile']['name'], band_mean_of=band_mean_of, wavelengths=wl
        )
        if not OT.identity_matches(b, {k: meta[k] for k in OT.identity(b['ozone_profile']['name'])}):
            raise ValueError('E_OZONE_T_ASSET_SHA: profile_mean assets read differ from the block')
        e = b['bin_edges_nm']
        return sigma, np.array([band_mean(A['solar'], lo, hi) for lo, hi in zip(e[:-1], e[1:])])
    oz = A[b['ozone_column']]
    if b['mode'] == 'point':
        return np.interp(wl, A['wl'], oz), np.interp(wl, A['wl'], A['solar'])
    e = b['bin_edges_nm']
    return (
        np.array([band_mean(oz, lo, hi) for lo, hi in zip(e[:-1], e[1:])]),
        np.array([band_mean(A['solar'], lo, hi) for lo, hi in zip(e[:-1], e[1:])]),
    )


def spec_table(b, wavelengths, state=None):
    """SPEC dict for accepted/engine.py: nodes = engine wavelengths, so np.interp returns the representative values exactly.
    A profile_mean block must already be resolved for its state (resolve, check_state). `state` only decides whether O4 is
    on (components.o4_enabled; omitted = {} = the default, O4 on); NO2 and O4 are zero for the reference/legacy physics."""
    sigma, E = representative(b, wavelengths)
    from .components import o4_enabled, reference_active, identity, LEGACY
    from .minor_absorbers import representative as minor

    values = (
        {k: np.zeros_like(wavelengths, dtype='f8') for k in ('no2', 'o4')}
        if reference_active() or identity(b.get('physics')) == LEGACY
        else minor(b, wavelengths, o4=o4_enabled(state or {}))
    )
    ozone = b.get('ozone_column', 'profile_mean')
    return dict(
        wavelength_nm=np.asarray(wavelengths, float).tolist(),
        ozone=sigma.tolist(),
        solar=E.tolist(),
        **{k: v.tolist() for k, v in values.items()},
        source=f"spectral {b['mode']} ({b['asset']} {b['asset_sha256'][:12]}, {ozone})",
    )


def _samples(lo, hi):
    """Integration nodes: the fine grid plus integer nm (CMF and V' are tabulated at 1 nm and linear in between)."""
    return np.unique(np.r_[_fine_grid(lo, hi), np.arange(np.ceil(lo), hi)])


def _product_integral(x, f, g):
    """Exact integral of the product of two functions linear between the nodes x, so adding a node changes nothing (§4-3)."""
    h = np.diff(x)
    return float(np.sum(h / 6 * (2 * f[:-1] * g[:-1] + f[:-1] * g[1:] + f[1:] * g[:-1] + 2 * f[1:] * g[1:])))


def channel_weights(b, wavelengths):
    """W[n, 4] (X, Y, Z, S'): W_ic = K int_bin c(l) E(l) dl / E_rep,i with E_rep,i the value the engine used
    (point: E(l_i), band: bin mean), bins from the block; exact product integral of the piecewise-linear CMF and E.
    Generalizes the legacy 'esun' weight set (legacy_channel_weights, fixed 15 bins) to n bins; V' is restricted
    to 380-780 nm."""
    check_asset(b)
    A = asset()
    mi = json.loads(CMF_1NM.read_text())
    w1 = np.array(mi['wavelength_nm'], float)
    xyz = np.array(mi['xyz'], float)
    lef = json.loads(LEF.read_text())
    lw = np.array(lef['wavelength_nm'], float)
    vs = np.array(lef['V_scotopic'], float)
    _, Erep = representative(b, wavelengths)
    e = b['bin_edges_nm']
    W = np.zeros((len(e) - 1, 4))
    for i, (lo, hi) in enumerate(zip(e[:-1], e[1:])):
        x = _samples(lo, hi)
        E = np.interp(x, A['wl'], A['solar'])
        for c in range(3):
            W[i, c] = KM * _product_integral(x, np.interp(x, w1, xyz[:, c]), E) / Erep[i]
        lo2, hi2 = max(lo, 380.0), min(hi, 780.0)
        if hi2 > lo2:
            x2 = _samples(lo2, hi2)
            W[i, 3] = KS * _product_integral(x2, np.interp(x2, lw, vs), np.interp(x2, A['wl'], A['solar'])) / Erep[i]
    return W


LEGACY_EDGES = np.linspace(360, 830, 16)  # bin edges of the legacy 15-bin tables


def _isamples(lo, hi):  # integration nodes of a legacy bin: its two edges and the integer wavelengths between them
    return np.unique(np.r_[lo, np.arange(math.ceil(lo), hi), hi])


def legacy_channel_weights(wl):
    """The 15-bin legacy weight set ('esun'), computed the way the legacy tables were made: the arithmetic
    (np.interp to 1 nm, unique integer samples per bin, trapezoid) and the sources (Mitsuba CIE 1931 1 nm,
    colour-science V_scotopic, 48-band spectra.json 'solar') are fixed. Returns (W[15, 4], info)."""
    wl = np.asarray(wl, float)
    mi = json.loads(CMF_1NM.read_text())
    wl1 = np.array(mi['wavelength_nm'])
    xyz1 = np.array(mi['xyz'])
    lef = json.loads(LEF.read_text())
    lw = np.array(lef['wavelength_nm'])
    vs = np.array(lef['V_scotopic'])
    sp = json.loads(SPECTRA_48.read_text())
    sw = np.array(sp['wavelength_nm'], float)
    se = np.array(sp['solar'], float)
    Ec = np.interp(wl, sw, se)  # solar irradiance at the bin centres, as the legacy solver sampled it
    We = np.zeros((15, 4))
    for i, (lo, hi) in enumerate(zip(LEGACY_EDGES[:-1], LEGACY_EDGES[1:])):
        x = _isamples(lo, hi)
        E = np.interp(x, sw, se)
        for c in range(3):
            cm = np.interp(x, wl1, xyz1[:, c])
            We[i, c] = KM * np.trapezoid(cm * E, x) / Ec[i]
        lo2, hi2 = max(lo, 380), min(hi, 780)  # V' is tabulated on 380-780 nm
        if hi2 > lo2:
            x2 = _isamples(lo2, hi2)
            v = np.interp(x2, lw, vs)
            E2 = np.interp(x2, sw, se)
            We[i, 3] = KS * np.trapezoid(v * E2, x2) / Ec[i]
    sha = hashlib.sha256(SPECTRA_48.read_bytes()).hexdigest()
    info = dict(
        edges_nm=LEGACY_EDGES.tolist(),
        wavelength_nm=wl.tolist(),
        E_sun_at_centres=Ec.tolist(),
        W_esun=We.tolist(),
        sources=dict(
            cmf='mitsuba 3.9.1 mi.cie1931_xyz 1 nm (sosfuse2/assets/cmf_1nm.json)',
            V_scotopic='sosfuse2/assets/lef_cie.json V_scotopic (CIE 1951, 1 nm, 380-780; 0 outside)',
            E_sun=f"sosfuse2/assets/spectra.json 'solar' W m^-2 nm^-1, 10 nm 360-830, source {sp.get('source')} (file sha256 {sha}); "
            "linear interpolation (np.interp) to 1 nm and to bin centres as sos_sky.py:301",
            bins='EDGES=linspace(360,830,16), samples unique([lo, ceil(lo)..hi-1, hi]) + trapezoid',
        ),
        definition=dict(
            esun='w_i = 683 * int_bin cmf(l) E_sun(l) dl / E_sun(lambda_i)  (R=L/E_sun piecewise constant in bin)'
        ),
    )
    return We, info


def table_weights(block, wavelengths):
    """(W[n, 4] X, Y, Z, S', info) for tables made with `block` (None = legacy). Legacy is the 15-bin E_sun set
    (legacy_channel_weights); another legacy count has no defined weights."""
    wl = np.asarray(wavelengths, float)
    if block is None:
        if len(wl) != 15:
            raise ValueError(
                f'E_SPECTRAL_CONTRACT: legacy weights are the lutcv2 15-bin set, table has {len(wl)} wavelengths'
            )
        return legacy_channel_weights(wl)
    W = channel_weights(block, wl)
    info = dict(
        edges_nm=list(block['bin_edges_nm']),
        wavelength_nm=wl.tolist(),
        spectral=block,
        definition=dict(
            esun="w_i = 683 int_bin cmf E_sun dl / E_rep,i (S': 1700 V' over 380-780); E_rep = the engine's value "
            "(point: E(l_i), band: bin mean); exact piecewise-linear product integral (sosfuse2/spectral.py)"
        ),
        sources=dict(
            cmf='sosfuse2/assets/cmf_1nm.json (CIE 1931 2 deg)',
            V_scotopic='sosfuse2/assets/lef_cie.json V_scotopic',
            E_sun=f"{block['asset']} solar, sha256 {block['asset_sha256']}",
            ozone=block.get('ozone_column', 'profile_mean'),
        ),
    )
    return W, info


def _minor_contract(b):
    """B2 asset fields cannot disappear in a copied block, even without a signature."""
    from .components import identity, LEGACY, S1_REFERENCE

    physical = identity(b.get('physics'))
    if physical == LEGACY and ('minor_absorber_sha256' in b or 'point_definition' in b):
        raise ValueError('E_PHYSICS_LOST: B2 spectral fields without B2 physics')
    if physical in (LEGACY, S1_REFERENCE):
        return
    minor = physical['minor_absorbers']
    from .minor_absorbers import canonical_hashes

    if canonical_hashes(b.get('minor_absorber_sha256')) != {k: minor[k]['asset_sha256'] for k in ('no2', 'o4')}:
        raise ValueError('E_PHYSICS_CONTRACT: missing or inconsistent minor absorber block SHA')
    if b.get('point_definition') != dict(
        o3='point', no2='0.1 nm cell-mean interpolant', o4='0.1 nm cell-mean interpolant'
    ):
        raise ValueError('E_PHYSICS_CONTRACT: missing component point definitions')


def canonical_block(b):
    """Copy a block with registered equivalent asset digests; retain every other field."""
    if b is None:
        return None
    _minor_contract(b)
    b = copy.deepcopy(b)
    for key, name in (('asset_sha256', 'spectra_fine.json'),
                      ('temperature_asset_sha256', 'temperature_usstd76.json'),
                      ('ozone_sigma_asset_sha256', 'spectra_fine_temperature.json')):
        if key in b:
            b[key] = source_identity.asset_digest(name, b[key])
    if 'minor_absorber_sha256' in b:
        from .minor_absorbers import canonical_hashes

        b['minor_absorber_sha256'] = canonical_hashes(b['minor_absorber_sha256'])
    if 'physics' in b:
        from .components import identity

        b['physics'] = identity(b['physics'])
    return b


def same_settings(a, b):
    """Full settings equality, allowing only registered asset-digest replacements."""
    a, b = copy.deepcopy(a), copy.deepcopy(b)
    for cfg in (a, b):
        if 'spectral' in cfg:
            cfg['spectral'] = canonical_block(cfg['spectral'])
    return a == b


def block_key(b):
    b = canonical_block(b)
    if b is None:
        return None
    from .components import identity  # a block without physics (before B2) keys as the legacy identity

    physical = json.dumps(identity(b.get('physics')), sort_keys=True)
    minor = json.dumps(b.get('minor_absorber_sha256'), sort_keys=True)
    if _profile_mean(b):
        return (
            b['mode'],
            b['asset_sha256'],
            'profile_mean',
            b['bin_edges_sha256'],
            b['temperature_asset_sha256'],
            b['ozone_sigma_asset_sha256'],
            b['ozone_temperature_rule'],
            tuple(sorted(b['ozone_profile'].items())),
            physical,
            minor,
        )
    return (b['mode'], b['asset_sha256'], b['ozone_column'], b['bin_edges_sha256'], physical, minor)


CONTRACTS = (
    'v3',
    'v4',
    'v5',
)  # docs/PROVENANCE_V3.md "Reader rule": the `contract` fields of signed tables (run_gpu_state_v3 meta.json)


def check_meta_contract(meta):
    """Returns meta['contract']. A *signed* table (meta carries `signature`, i.e. a run_gpu_state_v3 output) must name a known
    `contract` version, E_PROVENANCE_CONTRACT otherwise (SHA comparisons are only meaningful within a known version:
    v3 and v4 hash file bytes, v5 hashes the identity text of sosfuse2.identity under the same keys).
    A derived table without a signature (merge tools write their own
    descriptive `contract` text) is accepted as is (and may be paired with a signed table, as the product twilight/day pair is).
    Physics applies to both (B2): the table's identity must be known, a table without one (written before B2) reads as the
    legacy identity (components.identity), and its own spectral block must agree with it.
    Combining tables is the caller's check (components.check_same, check_contract)."""
    c = meta.get('contract')
    if 'signature' in meta and c not in CONTRACTS:
        raise ValueError(f'E_PROVENANCE_CONTRACT: signed table with unsupported contract {c!r}')
    from .components import check_same, check_optics, identity, LEGACY

    if identity(meta.get('physics')) == LEGACY and 'optics_state' in meta:
        raise ValueError('E_PHYSICS_LOST: optics_state without B2 physics')
    b = (meta.get('settings') or {}).get('spectral')
    if b is None:
        check_same(meta.get('physics'), what='table')
    else:
        block_key(b)
        check_same(meta.get('physics'), b.get('physics'), what='table meta and its spectral block')
    if identity(meta.get('physics')) != LEGACY:
        check_optics(meta.get('optics_state'))
    return c


def table_block(meta):
    """The spectral block a table was made with (None = legacy) from its meta.json dict; refuses an unsupported `contract` version."""
    check_meta_contract(meta)
    return (meta.get('settings') or {}).get('spectral')


def check_contract(table, weights):
    """Refuse weights made for another spectral definition than the table (E_SPECTRAL_CONTRACT) or for other physics
    (E_PHYSICS_MIXED: a legacy block with a current one, components.check_same)."""
    if table is not None and weights is not None:
        from .components import check_same

        check_same(table.get('physics'), weights.get('physics'), what='table and weights')
    key = block_key
    if key(table) != key(weights):
        raise ValueError(f'E_SPECTRAL_CONTRACT: table {key(table)} vs weights {key(weights)}')
