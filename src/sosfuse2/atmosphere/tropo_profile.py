"""Tropospheric aerosol vertical profile: opt-in two layers (boundary layer + free troposphere).

A state enables it with a top-level block (next to troposphere_phase):
    "troposphere_profile": {"status": "enabled", "model": "two_layer",
                            "bl_top_km": 2.0, "bl_transition_km": 1.0, "bl_shape": "exponential",
                            "bl_scale_height_km": 2.14, "ft_aod550": 0.0797, "ft_scale_height_km": 1.85}
Omitted numbers take DEFAULTS. Absent block or status "disabled" = the historical exp(-h/scale_height_km) path:
nothing below runs, engine_hooks() is None, no kernel flag or constant is added, every cache key is unchanged.

Profile (extinction at 550 nm per unit total aod550, km^-1, integral over 0..80 km = 1):
    p(h) = a_bl*b(h)*(1-s(h)) + a_ft*f(h)*s(h)
    b = 1 (uniform) or exp(-h/bl_scale_height_km);  f = exp(-(h-bl_top)/ft_scale_height_km)
    s = smoothstep x*x*(3-2x), x = clip((h-(bl_top-w/2))/w, 0, 1), w = bl_transition_km (C1; no step inside a GL segment)
    a_bl = (aod_bl/aod550)/Ib, Ib = integral of b*(1-s) on [0, 80];  a_ft = (aod_ft/aod550)/If, If = integral of f*s on [0, 80]
    (closed form outside the transition, Gauss-Legendre 64 inside it: both layer integrals are exact to rounding).
    aod_bl = parameters.aod550 - ft_aod550 (>= 0; the free troposphere is fixed, the boundary layer takes the rest).
The engine normalization becomes NORMAL['aer'] = aod550/COLUMN (COLUMN = 1). parameters.scale_height_km stays a
required state key but no longer shapes the aerosol (it remains the Kernel 'height' scalar that every path overrides).
One parameter vector P = [bl_top-w/2, 1/w, a_bl, k_bl, a_ft, k_ft, bl_top] (k = 1/H, 0 for uniform), evaluated as
    x = clip((h-P0)*P1, 0, 1); s = x*x*(3-2*x); p = (P2*exp(-h*P3))*(1-s) + (P4*exp(-(max(h,P0)-P6)*P5))*s
(max(h,P0) only keeps the free-troposphere exponent finite where s = 0) by: shell_segments.profiles / engine_core.Solver.ha /
NORMAL (through solve/engine.py create(aerosol=engine_hooks(state)), see engine_hooks below), ray_geometry.aerosol_column,
native/sh_cpu.cpp full_transport (aerosol_height < 0, parameters after asc[0..W)), kernel.cu -DAEROSOL_2L (parameters after
sun_h[0..nsh)) and numpy_reference.sample (FP32 mirror of the kernel line).

DEFAULTS. Source: LOWTRAN7 lowtran7.f, https://raw.githubusercontent.com/space-physics/lowtran/main/src/lowtran/fortran/lowtran7.f
(875,546 bytes, sha256 25e83d94e24b...): BLOCK DATA PRFDTA :13587-13630, AERPRF :3900-3938, season default :1068,
EXPINT interpolation :4057.
  bl_top_km 2.0          boundary layer table HZ2K covers ZHT <= 2 km (:13592-13593, AERPRF :3913). Verified.
  bl_scale_height_km 2.14  HZ2K 23 km column (rural V 23 km = Lee 2003 reference case):
                         0.158 / 0.0991 / 0.0621 km^-1 at 0 / 1 / 2 km (:13619-13624, verified); log least squares H = 2.142 km
                         (the 50 km column 0.0662/0.0415/0.026 gives the same H; the 10/5/2 km columns are flat 0-1 km).
  ft_aod550 0.0797, ft_scale_height_km 1.85  = lowtran_free_troposphere(V <= 23 km) below. LOWTRAN's own class rule
                         (AERPRF, fall/winter branch :3933-3938): VIS <= 23 km -> FAWI23 (0.0272 0.0120 0.00485 ... 0.000787 at
                         3..10 km, :13628-13630); VIS > 23 km -> 1/V interpolation between FAWI23 and FAWI50 at 3-4 km (:3936),
                         FAWI50 above 4 km (:3938; identical to FAWI23 there). The 2 km anchor is the boundary-layer value
                         HZ2K(3,.) interpolated in 1/V (:3914-3920): 0.0621 for every VIS <= 23 km. MODEL 3 (midlatitude winter,
                         Lee 2003 reference atmosphere) selects fall/winter (:1068); levels are joined exponentially (EXPINT
                         :4057). V <= 23 km: 2-10 km integral 0.0797, log least squares H 1.85 km. The Lee comparison states (aod
                         .2 / .4 ~ V 23 / 10 km by an approximate estimate) are
                         in this class, so it is the default; a state with V > 23 km (aod below about .2) should carry its own
                         ft values from lowtran_free_troposphere(V) (V 50 km: 0.0429, H 2.39 km).
  bl_transition_km 1.0   smoothing width (not a LOWTRAN quantity). A real boundary-layer top is an entrainment zone a few
                         hundred m to ~1 km thick; 1 km also resolves the blend for the GL-24 column quadrature (vertical column
                         vs a dense integral, these defaults, aod .15: +1.40 % at 0.2 km, -0.12 % at 1.0 km).
"""

import hashlib, json, math, types
from dataclasses import dataclass
import numpy as np

KEY = 'troposphere_profile'
STATUS = ('enabled', 'disabled')
KEYS = {
    'status',
    'model',
    'bl_top_km',
    'bl_transition_km',
    'bl_shape',
    'bl_scale_height_km',
    'ft_aod550',
    'ft_scale_height_km',
}
SHAPES = ('uniform', 'exponential')
DEFAULTS = dict(
    bl_top_km=2.0,
    bl_transition_km=1.0,
    bl_shape='exponential',
    bl_scale_height_km=2.14,
    ft_aod550=0.0797,
    ft_scale_height_km=1.85,
)
TOP_KM = 80.0  # shell_segments.Z (atmosphere top); the kernel clips sample heights to [0, 80]
FLAG = 'AEROSOL_2L'

# lowtran7.f BLOCK DATA PRFDTA (:13619-13630): 0.55 um extinction, km^-1.
LOWTRAN_VS = (50.0, 23.0, 10.0, 5.0, 2.0)
LOWTRAN_HZ2K_2KM = (2.60e-2, 6.21e-2, 6.21e-2, 6.21e-2, 6.21e-2)  # HZ2K(3, 1..5), ZHT 2 km
LOWTRAN_FAWI50 = (1.14e-2, 6.43e-3, 4.85e-3, 3.54e-3, 2.31e-3, 1.41e-3, 9.80e-4, 7.87e-4)  # ZHT 3..10 km
LOWTRAN_FAWI23 = (2.72e-2, 1.20e-2, 4.85e-3, 3.54e-3, 2.31e-3, 1.41e-3, 9.80e-4, 7.87e-4)


def lowtran_free_troposphere(vis_km):
    """(ft_aod550, ft_scale_height_km) of LOWTRAN7's 2-10 km fall/winter profile at visibility vis_km: AERPRF's class
    rule (:3914-3920 for the 2 km anchor, :3933-3938 above), exponential joins between levels (EXPINT), the 2-10 km
    integral and a log least-squares scale height over 2..10 km."""
    v = float(vis_km)
    j = next((k for k in range(1, 5) if v >= LOWTRAN_VS[k]), 4)
    c = 1.0 / (1.0 / LOWTRAN_VS[j] - 1.0 / LOWTRAN_VS[j - 1])
    h2 = LOWTRAN_HZ2K_2KM
    anchor = c * ((h2[j] - h2[j - 1]) / v + h2[j - 1] / LOWTRAN_VS[j] - h2[j] / LOWTRAN_VS[j - 1])
    c = 1.0 / (1.0 / 23.0 - 1.0 / 50.0)
    upper = []
    for z, (f23, f50) in enumerate(zip(LOWTRAN_FAWI23, LOWTRAN_FAWI50), 3):
        upper.append(f23 if v <= 23.0 else c * ((f23 - f50) / v + f50 / 23.0 - f23 / 50.0) if z <= 4 else f50)
    b = np.array([anchor] + upper)
    tau = float(np.sum((b[:-1] - b[1:]) / np.log(b[:-1] / b[1:])))
    return tau, float(-1.0 / np.polyfit(np.arange(2.0, 11.0) - 2.0, np.log(b), 1)[0])


def _number(block, key, low, high, low_open=True):
    v = block[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f'E_TROPO_PROFILE_VALUE: {key} must be a finite number')
    v = float(v)
    if (v <= low if low_open else v < low) or v >= high:
        raise ValueError(
            f'E_TROPO_PROFILE_VALUE: {key}={v!r} outside ({low}, {high})'
            if low_open
            else f'E_TROPO_PROFILE_VALUE: {key}={v!r} outside [{low}, {high})'
        )
    return v


LEGACY_BLOCKS = ('troposphere_profile', 'troposphere_phase')


def reject_legacy(state, runner):
    """Legacy entry points that build the solver without the state configuration hooks know
    neither block and would compute the exponential/HG troposphere silently. They call this on the state they read:
    a block that is not exactly {"status": "disabled", ...} raises E_LEGACY_UNSUPPORTED_BLOCK (absent key = fine)."""
    for key in LEGACY_BLOCKS:
        if isinstance(state, dict) and key in state:
            block = state[key]
            if not (isinstance(block, dict) and block.get('status') == 'disabled'):
                raise ValueError(
                    f'E_LEGACY_UNSUPPORTED_BLOCK: {runner} cannot honour state {key} (use '
                    'sosfuse2.jobs.run_state or sosfuse2.jobs.pipeline)'
                )
    return state


def config(state):
    """None = historical exponential profile: only when the key is absent, or status "disabled". An enabled block is
    validated and completed with DEFAULTS; anything else (null, {}, wrong status, ...) is rejected, never read as the
    exponential profile."""
    if not isinstance(state, dict) or KEY not in state:
        return None
    block = state[KEY]
    if not isinstance(block, dict):
        raise ValueError('E_TROPO_PROFILE: troposphere_profile must be an object')
    status = block.get('status')
    if status == 'disabled':
        return None
    if status != 'enabled':
        raise ValueError('E_TROPO_PROFILE_STATUS: status must be "enabled" or "disabled"')
    if set(block) - KEYS:
        raise ValueError('E_TROPO_PROFILE_KEYS: unknown keys ' + ','.join(sorted(set(block) - KEYS)))
    if block.get('model') != 'two_layer':
        raise ValueError('E_TROPO_PROFILE_MODEL: only model "two_layer" is supported')
    shape = block.get('bl_shape', DEFAULTS['bl_shape'])
    if shape not in SHAPES:
        raise ValueError('E_TROPO_PROFILE_SHAPE: bl_shape must be "uniform" or "exponential"')
    if shape == 'uniform' and 'bl_scale_height_km' in block:
        raise ValueError('E_TROPO_PROFILE_SHAPE: bl_scale_height_km is only for bl_shape "exponential"')
    full = dict({k: v for k, v in DEFAULTS.items() if not (k == 'bl_scale_height_km' and shape == 'uniform')}, **block)
    out = dict(
        model='two_layer',
        bl_shape=shape,
        bl_top_km=_number(full, 'bl_top_km', 0.0, TOP_KM),
        bl_scale_height_km=_number(full, 'bl_scale_height_km', 0.0, math.inf) if shape == 'exponential' else None,
        ft_aod550=_number(full, 'ft_aod550', 0.0, math.inf, low_open=False),
        ft_scale_height_km=_number(full, 'ft_scale_height_km', 0.0, math.inf),
    )
    # The transition [bl_top-w/2, bl_top+w/2] must lie inside (0, 80).
    out['bl_transition_km'] = _number(
        full, 'bl_transition_km', 0.0, 2 * min(out['bl_top_km'], TOP_KM - out['bl_top_km'])
    )
    try:
        aod = float(state['parameters']['aod550'])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError('E_TROPO_PROFILE_AOD: parameters.aod550 required') from e
    if not math.isfinite(aod) or aod <= 0:
        raise ValueError('E_TROPO_PROFILE_AOD: parameters.aod550 must be positive')
    if out['ft_aod550'] > aod:
        raise ValueError(
            f'E_TROPO_PROFILE_AOD: ft_aod550 {out["ft_aod550"]!r} exceeds aod550 {aod!r} (boundary layer aod < 0)'
        )
    out['aod550'] = aod
    return out


def smoothstep(x):
    return x * x * (3 - 2 * x)


def basis_integrals(z, w, kb, kf):
    """(Ib, If) = integrals over [0, 80] of exp(-h*kb)*(1-s) and exp(-(h-z)*kf)*s: closed form outside the transition
    [z-w/2, z+w/2], Gauss-Legendre 64 inside (smooth there; cubic times exponential)."""
    z0, z1 = z - w / 2, z + w / 2
    x, g = np.polynomial.legendre.leggauss(64)
    h = z0 + (x + 1) * (w / 2)
    g = g * (w / 2)
    s = smoothstep((h - z0) / w)
    ib = (z0 if kb == 0 else -math.expm1(-z0 * kb) / kb) + float(np.sum(np.exp(-h * kb) * (1 - s) * g))
    jf = (
        float(np.sum(np.exp(-(h - z) * kf) * s * g))
        + math.exp(-(z1 - z) * kf) * (-math.expm1(-(TOP_KM - z1) * kf)) / kf
    )
    return ib, jf


def parameters(cfg):
    """P = [bl_top-w/2, 1/w, a_bl, k_bl, a_ft, k_ft, bl_top] (float64) and the layer aods; profile integral on [0,80] = 1."""
    z, w, aod, aft = cfg['bl_top_km'], cfg['bl_transition_km'], cfg['aod550'], cfg['ft_aod550']
    abl = aod - aft
    kb = 0.0 if cfg['bl_shape'] == 'uniform' else 1.0 / cfg['bl_scale_height_km']
    kf = 1.0 / cfg['ft_scale_height_km']
    # The kernel evaluates the FT exponential even where its blend is zero.
    # Reject before the FP64 integral can overflow or produce a bad FP32 vector.
    limit = math.log(np.finfo('f4').max) - 1e-4
    if w / 2 * kf >= limit:
        raise ValueError('E_TROPO_PROFILE_FP32: free-troposphere exponent overflows')
    ib, jf = basis_integrals(z, w, kb, kf)
    if not math.isfinite(kb) or ib <= 0 or jf <= 0:
        raise ValueError('E_TROPO_PROFILE_FP32: boundary-layer scale height too small')
    vector = np.array([z - w / 2, 1.0 / w, abl / aod / ib, kb, aft / aod / jf, kf, z], 'f8')
    with np.errstate(over='ignore', invalid='ignore'):
        k = vector.astype('f4')
    if not np.isfinite(k).all():
        raise ValueError('E_TROPO_PROFILE_FP32: nonfinite kernel parameter')
    exponent = (float(k[6]) - float(k[0])) * float(k[5])
    if exponent >= limit or (k[4] > 0 and math.log(float(k[4])) + exponent >= limit):
        raise ValueError('E_TROPO_PROFILE_FP32: free-troposphere term overflows')
    return dict(vector=vector, aod_bl=abl, aod_ft=aft, column=1.0, basis=(ib, jf))


def profile(h, vector):
    """Host profile (float64), h in km; negative heights clamp to 0 like the historical np.maximum(h,0)."""
    h = np.maximum(np.asarray(h, 'f8'), 0.0)
    z0, iw, ab, kb, af, kf, z = (float(x) for x in vector)
    x = np.clip((h - z0) * iw, 0.0, 1.0)
    s = x * x * (3.0 - 2.0 * x)
    return (ab * np.exp(-h * kb)) * (1.0 - s) + (af * np.exp(-(np.maximum(h, z0) - z) * kf)) * s


@dataclass(eq=False)
class TwoLayerProfile:
    """Explicit profile parameters, pickled by module reference for column workers."""

    vector: np.ndarray

    def __call__(self, h):
        return profile(h, self.vector)

    def __eq__(self, other):
        if not isinstance(other, TwoLayerProfile):
            return NotImplemented
        return np.array_equal(self.vector, other.vector)

    def descriptor(self):
        return dict(kind='two_layer', vector=np.asarray(self.vector).tolist())


@dataclass
class ExponentialProfile:
    """Keep the column clamp and the solver's unclamped default separate."""

    height: float
    clamp: bool

    def __call__(self, h):
        if self.clamp:
            return np.exp(-np.maximum(h, 0) / self.height)
        return np.exp(-h / self.height)

    def descriptor(self):
        return dict(kind='exp', height=float(self.height))


@dataclass
class ProfileFolder:
    tag: str

    def __call__(self, cache):
        out = cache / 'aerosol_profile' / self.tag
        out.mkdir(parents=True, exist_ok=True)
        return out


def profile32(h, k32):
    """FP32 mirror of kernel.cu aerosol_two_layer() (h already clipped to [0,80] by the caller, as in point())."""
    h = np.asarray(h, 'f4')
    k = np.asarray(k32, 'f4')
    one, two, three = np.float32(1), np.float32(2), np.float32(3)
    x = np.clip((h - k[0]) * k[1], np.float32(0), one)
    s = x * x * (three - two * x)
    return ((k[2] * np.exp(-h * k[3])) * (one - s) + (k[4] * np.exp(-(np.maximum(h, k[0]) - k[6]) * k[5])) * s).astype(
        'f4'
    )


def layer_integrals(cfg):
    """Integrals of the two profile components times aod550 (= aod_bl, aod_ft)."""
    p = parameters(cfg)
    v = p['vector']
    aod = cfg['aod550']
    ib, jf = p['basis']
    return dict(bl=aod * v[2] * ib, ft=aod * v[4] * jf)


def canonical(cfg):
    """Exact text identity of the profile function (aod550 included: the layer split depends on it)."""
    keys = (
        'model',
        'bl_top_km',
        'bl_transition_km',
        'bl_shape',
        'bl_scale_height_km',
        'ft_aod550',
        'ft_scale_height_km',
        'aod550',
    )
    return json.dumps({k: cfg[k] for k in keys}, sort_keys=True, allow_nan=False)


def tag(cfg):
    return 'A2L_' + hashlib.sha256(canonical(cfg).encode()).hexdigest()[:16]


def build(state):
    cfg = config(state)
    if cfg is None:
        return None
    p = parameters(cfg)
    return dict(
        config=cfg,
        vector=p['vector'],
        kernel=np.ascontiguousarray(p['vector'], 'f4'),
        tag=tag(cfg),
        canonical=canonical(cfg),
        meta=dict(
            model='two_layer',
            canonical=json.loads(canonical(cfg)),
            tag=tag(cfg),
            aod_bl=p['aod_bl'],
            aod_ft=p['aod_ft'],
            kernel_vector=[float(x) for x in p['vector']],
            note='scale_height_km is not used for the aerosol profile when this block is enabled',
        ),
    )


def of(s):
    """The solver's profile (attach() result), computed from s.fuse_state if not attached yet; None = exponential."""
    if s is None:
        return None
    if hasattr(s, 'aerosol_profile'):
        return s.aerosol_profile
    state = getattr(s, 'fuse_state', None)
    return None if state is None else build(state)


def attach(s):
    """Idempotent: s.aerosol_profile = build(s.fuse_state) or None."""
    if not hasattr(s, 'aerosol_profile'):
        s.aerosol_profile = build(getattr(s, 'fuse_state', {}))
    return s.aerosol_profile


def column_name(s, height):
    """Aerosol column file stem: historical 'H_<float.hex>' (unchanged without the block) or the profile tag."""
    p = of(s)
    return f'H_{float(height).hex()}' if p is None else p['tag']


def same_source(s, source, height):
    """True if the common layer's end_col[:,1] (built for source_height.json 'height') is this state's column.
    Two-layer states never reuse it (a two-layer build records height None)."""
    return of(s) is None and source is not None and float(source) == float(height)


def column_function(s):
    """profile callable for ray_geometry.aerosol_column (None = historical exponential expression)."""
    p = of(s)
    return None if p is None else TwoLayerProfile(p['vector'])


def metadata(s):
    p = of(s)
    return None if p is None else dict(p['meta'])


# ---- solve/engine.py create(aerosol=...): the enabled block as explicit engine hooks ------------------------------


def engine_hooks(state):
    """create()'s `aerosol` argument for this state: None without an enabled block (the historical exponential profile,
    nothing added anywhere), else a namespace the engine reads at four points:
      profile(h)     the two-layer profile, used by every optical column and by the solver's ha vector (AER_PROFILE),
      column         the aod normalisation (NORMAL['aer'] = aod550/column, 1 for this model),
      folder(cache)  cache/aerosol_profile/<tag>: the sun-column disk caches are keyed by folder (and scale height) only,
                     so every profile gets its own,
      vector         the parameter vector the native kernel passes after asc[0..W) (Kernel.aerosol_vector)."""
    p = build(state)
    if p is None:
        return None
    return types.SimpleNamespace(
        profile=TwoLayerProfile(p['vector']),
        column=parameters(p['config'])['column'],
        folder=ProfileFolder(p['tag']),
        vector=p['vector'],
        tag=p['tag'],
    )
