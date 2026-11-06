"""Opt-in density tables; immutable endpoint caches retain historical columns.

PROFILE grid: PROFILE nodes 0..80 km at density.profile_step_km in {0.5, 0.25, 0.125}
(161/321/641 rows, linear interpolation, air/ozone plus NO2, mask bits 1/2 select air/ozone). Omitted = 0.5 = the historical air/ozone table,
built by the unchanged expressions (no interpolation, no renormalization). The asset (v1, 161 x 0.5 km) is normalized on
its own grid first; 0.25/0.125 are the same piecewise-linear function sampled on the finer grid (no new information).

External tables (with_profile) carry their own grid, mask and Rayleigh scale; normalize_ozone=False uses the ozone/q_eff
column as given. A caller that replaces only q_eff keeps the air table, air mask and scale of the state it adapts:

    base = build(dict(state, density=dict(state.get('density', {}), profile_step_km=.125)))   # asset air on the grid
    z = grid(len(base['table']))   # 641 nodes, 0.125 km
    q = <q_eff evaluated on z>     # e.g. Model.effective(shape, i, step=.125); never an upsampled 0.5 km table
    state2 = with_profile(state, z_km=z, table=np.column_stack([base['table'][:, 0], q]),
                          mask=base['mask'] | 2, normalize_ozone=False, scale=base['scale'],
                          air_surface_number_density_m3=base['meta']['o4']['n0'])   # O4 (on by default) needs it: E_O4_N0

The external API remains two columns; NO2 is supplied from its packaged shape. Errors are E_PROFILE_* codes.
"""
import copy
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
import numpy as np

from . import paths
from .components import NCOMP, MOLECULAR_COLUMNS

# The US1976 + IPRT ozone-shape profile asset is not redistributed with the package: it is generated once, on first use, from the IPRT inputs (iprt.fetch, pinned sha256) into the work root and
# checked against the pinned digest. The pin is the platform-independent (decimal) generator's output, measured identical on
# arm64/Python 3.14 and x86_64/Python 3.12 (2026-11-03); the float64 2026-10 asset a7ccf7b1… differed per platform in the last
# ulp and is superseded (values within 4.7e-15 relative, generate_density_profile docstring). Provenance (code_hashes, gate
# identity, block identity) always uses the logical key ASSET_KEY, wherever the file physically lives. `density.ASSET`
# stays usable as a Path (module __getattr__).
ASSET_KEY = 'sosfuse2/assets/usstd76_profile.json'
ASSET_SHA256 = '05dfeb6e158e97fcccd2af30e3fd55ad1ea11fa0c8a58be51bd214de506c5795'
# Digests of earlier generations of the same logical asset (same formulas, inputs and normalization; only the arithmetic
# implementation differed). Readers of blocks and tables made with them accept them as the same ozone profile
# (ozone_temperature.same_profile_identity, docs/PROVENANCE_V3.md "Superseded asset digests"); they are never produced again.
# a7ccf7b1…: the 2026-10 float64/numpy generator, values within 4.7e-15 of the decimal asset and platform dependent in the
# last ulp, which is why the decimal generator replaced it.
ASSET_SHA256_SUPERSEDED = ('a7ccf7b152de5d82c3c47759e55b92cd558e64f8d4e8584623d92e78670c3020',)
_PACKAGED = Path(__file__).with_name('assets')/'usstd76_profile.json'   # present only in a tree that still carries the file


def asset_path(offline=None):
    """Path of usstd76_profile.json: the packaged copy if the tree has one (checked against the pin like every other copy), else the work-root copy, generated from the IPRT inputs when missing and refused (E_DENSITY_ASSET_SHA256)
    unless it reproduces the pinned bytes."""
    if _PACKAGED.is_file():
        got = hashlib.sha256(_PACKAGED.read_bytes()).hexdigest()
        if got != ASSET_SHA256: raise RuntimeError(f'E_DENSITY_ASSET_SHA256: packaged usstd76_profile.json is {got}, expected {ASSET_SHA256}')
        return _PACKAGED
    target = paths.work_root()/'assets'/'usstd76_profile.json'
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == ASSET_SHA256: return target
    from . import generate_density_profile
    text = generate_density_profile.text(offline)
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != ASSET_SHA256:
        raise RuntimeError(f'E_DENSITY_ASSET_SHA256: regenerated usstd76_profile.json is {digest}, expected {ASSET_SHA256}')
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f'.{os.getpid()}.tmp'); tmp.write_text(text); os.replace(tmp, target)
    return target


def __getattr__(name):   # PEP 562: density.ASSET materialises the asset on first use
    if name == 'ASSET': return asset_path()
    raise AttributeError(name)
DEFAULTS = dict(density_profile='exp8', ozone_profile='triangle', rayleigh_column='keep')
CHOICES = dict(density_profile=('exp8','usstd76'), ozone_profile=('triangle','usstd76'), rayleigh_column=('keep','usstd'))
LEGACY_COLUMN = 8*(-np.expm1(-10.))
# Numeric fields beside the string options, validated separately (the string whitelist and CLI stay DEFAULTS/CHOICES).
EXTRA = ('profile_step_km', 'external_profile')
DEFAULT_STEP = .5
PROFILE_STEPS = {.5: 161, .25: 321, .125: 641}
STEP_OF_N = {n: step for step, n in PROFILE_STEPS.items()}
# kernel.cu profile_value literals (inverse spacing, last interval) per node count.
KERNEL_LITERALS = {161: (2, 159), 321: (4, 319), 641: (8, 639)}
EXTERNAL_KEYS = ('version', 'z_km', 'table', 'mask', 'normalize_ozone', 'scale')


def options(state):
    raw=state.get('density',{})
    if not isinstance(raw,dict) or set(raw)-set(DEFAULTS)-set(EXTRA):raise ValueError('E_DENSITY_OPTIONS')
    opt=dict(DEFAULTS,**{k:v for k,v in raw.items() if k in DEFAULTS})
    if any(opt[k] not in CHOICES[k] for k in opt):raise ValueError('E_DENSITY_OPTIONS')
    return opt


def profile_n(step):
    """Node count of an allowed step (exactly 0.5, 0.25 or 0.125 km); E_PROFILE_STEP otherwise."""
    if isinstance(step,(bool,np.bool_)) or not isinstance(step,(int,float,np.integer,np.floating)) or float(step) not in PROFILE_STEPS:
        raise ValueError(f'E_PROFILE_STEP: profile_step_km must be 0.5, 0.25 or 0.125 (got {step!r})')
    return PROFILE_STEPS[float(step)]


def kernel_kwargs(step):
    """fused.Kernel keyword for a resolved step: none at 0.5 km, so the default construction is the historical call."""
    return {} if profile_n(step)==PROFILE_STEPS[DEFAULT_STEP] else dict(profile_step_km=float(step))


def grid(n):
    """PROFILE nodes for n rows; n=161 is np.arange(161)*.5 byte for byte."""
    if n not in STEP_OF_N:raise ValueError(f'E_PROFILE_N: {n} rows (161, 321 or 641 allowed)')
    return np.arange(n)*STEP_OF_N[n]


def _validated(z_km, table, mask, normalize_ozone, scale):
    """Checked FP64 copies of an injected profile; every failure is a stable E_PROFILE_* code."""
    try:z=np.asarray(z_km)
    except (TypeError,ValueError) as e:raise ValueError('E_PROFILE_GRID: z_km not numeric') from e
    if z.ndim!=1 or z.dtype.kind not in 'fiu' or len(z) not in STEP_OF_N:
        raise ValueError('E_PROFILE_GRID: z_km must be 161, 321 or 641 real nodes')
    z=np.array(z,'f8')
    if not np.isfinite(z).all() or not np.array_equal(z,grid(len(z))):
        raise ValueError(f'E_PROFILE_GRID: z_km must equal arange(N)*{STEP_OF_N[len(z)]} km (0..80)')
    try:t=np.asarray(table)
    except (TypeError,ValueError) as e:raise ValueError('E_PROFILE_TABLE: table not numeric') from e
    if t.dtype.kind not in 'fiu' or t.shape!=(len(z),2):
        raise ValueError(f'E_PROFILE_TABLE: real ({len(z)}, 2) table required (got {t.dtype.str} {t.shape})')
    t=np.array(t,'f8')
    if not np.isfinite(t).all() or np.any(t<0):raise ValueError('E_PROFILE_TABLE: finite non-negative values required')
    with np.errstate(over='ignore'):
        if not np.isfinite(t.astype('f4')).all():raise ValueError('E_PROFILE_TABLE: values overflow the FP32 upload')
    if isinstance(mask,(bool,np.bool_)) or not isinstance(mask,(int,np.integer)) or not 0<=int(mask)<=3:
        raise ValueError('E_PROFILE_MASK: integer 0..3 required (bit 1 air, bit 2 ozone)')
    mask=int(mask)
    if mask&1 and np.any(t[:,0]<=0):raise ValueError('E_PROFILE_AIR: an active air column must be positive')
    if not isinstance(normalize_ozone,(bool,np.bool_)):raise ValueError('E_PROFILE_NORMALIZE: normalize_ozone must be bool')
    if isinstance(scale,(bool,np.bool_)) or not isinstance(scale,(int,float,np.integer,np.floating)) or not np.isfinite(scale) or scale<=0:
        raise ValueError('E_PROFILE_SCALE: finite positive Rayleigh scale required')
    return z,t,mask,bool(normalize_ozone),float(scale)


def external(state):
    """Validated external payload (z, FP64 table, mask, normalize_ozone, scale) or None."""
    raw=state.get('density',{})
    if not isinstance(raw,dict) or 'external_profile' not in raw:return None
    ext=raw['external_profile']
    if not isinstance(ext,dict) or set(ext)-{'air_surface_number_density_m3'}!=set(EXTERNAL_KEYS) or ext['version']!=1:
        raise ValueError('E_PROFILE_PAYLOAD: external_profile needs exactly '+', '.join(EXTERNAL_KEYS)+' (version 1)')
    z,t,mask,normalize,scale=_validated(ext['z_km'],ext['table'],ext['mask'],ext['normalize_ozone'],ext['scale'])
    return dict(z=z,table=t,mask=mask,normalize_ozone=normalize,scale=scale,
                air_surface_number_density_m3=ext.get('air_surface_number_density_m3'))


def profile_step_km(state):
    """Resolved PROFILE step: an explicit density.profile_step_km, else the external grid, else 0.5 (omitted)."""
    raw=state.get('density',{})
    if not isinstance(raw,dict):raise ValueError('E_DENSITY_OPTIONS')
    explicit='profile_step_km' in raw
    n=profile_n(raw['profile_step_km']) if explicit else None
    if 'external_profile' in raw:
        ext=raw['external_profile']
        z=ext.get('z_km') if isinstance(ext,dict) else None
        m=len(z) if isinstance(z,(list,tuple,np.ndarray)) else None
        if m not in STEP_OF_N:raise ValueError('E_PROFILE_GRID: z_km must be 161, 321 or 641 real nodes')
        if explicit and m!=n:raise ValueError(f'E_PROFILE_STEP_CONFLICT: profile_step_km {raw["profile_step_km"]} vs external grid {STEP_OF_N[m]} km')
        return STEP_OF_N[m]
    return STEP_OF_N[n] if explicit else DEFAULT_STEP


def with_options(state,args):
    """CLI string options over the state; profile_step_km/external_profile are kept as given (an omitted step stays
    omitted, so ordinary states keep their signed input bytes). Only an explicit conflict is an error."""
    state=copy.deepcopy(state);opt=options(state)
    for key in DEFAULTS:
        value=getattr(args,key,None)
        if value is not None:opt[key]=value
    raw=state.get('density',{})
    extra={k:raw[k] for k in EXTRA if k in raw}
    step=getattr(args,'profile_step_km',None)
    if step is not None:
        if 'profile_step_km' in extra and profile_n(extra['profile_step_km'])!=profile_n(step):
            raise ValueError(f'E_PROFILE_STEP_CONFLICT: state {extra["profile_step_km"]} vs requested {step}')
        extra['profile_step_km']=float(step)
    if opt!=DEFAULTS or 'density' in state or extra:state['density']=dict(opt,**extra)
    profile_step_km(state);external(state)   # E_PROFILE_* before any expensive work
    return state


def with_profile(state,*,z_km,table,mask,normalize_ozone=False,scale=1.0,air_surface_number_density_m3=None):
    """Deep-copied JSON-serializable state whose density.external_profile is the given table (see module doc).

    The grid fixes profile_step_km; an explicitly different step already in the state is E_PROFILE_STEP_CONFLICT.
    normalize_ozone=False uses column 2 as given (q_eff); True scales it to the production 15 on the injected grid.
    scale is the Rayleigh factor applied to ext[0] (build()['scale'] of the adapted state keeps it)."""
    z,t,mask,normalize,scale=_validated(z_km,table,mask,normalize_ozone,scale)
    if normalize and not np.trapezoid(t[:,1],z)>0:raise ValueError('E_PROFILE_OZONE_ZERO: normalize_ozone needs a positive ozone integral')
    state=copy.deepcopy(state)
    raw=state.get('density',{})
    if not isinstance(raw,dict):raise ValueError('E_DENSITY_OPTIONS')
    step=STEP_OF_N[len(z)]
    if 'profile_step_km' in raw and profile_n(raw['profile_step_km'])!=len(z):
        raise ValueError(f'E_PROFILE_STEP_CONFLICT: profile_step_km {raw["profile_step_km"]} vs external grid {step} km')
    raw=dict(raw,profile_step_km=step,external_profile=dict(version=1,z_km=z.tolist(),table=t.tolist(),mask=mask,
             normalize_ozone=normalize,scale=scale))
    if air_surface_number_density_m3 is not None:
        raw['external_profile']['air_surface_number_density_m3']=air_surface_number_density_m3
    state['density']=raw
    options(state);external(state)
    json.dumps(state,allow_nan=False)
    return state


def add_arguments(parser):
    for key in DEFAULTS:
        parser.add_argument('--'+key.replace('_','-'),choices=CHOICES[key],default=None,
                            help='default '+DEFAULTS[key]+'; honors state density metadata')


def load_asset():
    path=asset_path()   # outside the try: a download or digest failure keeps its own E_IPRT_*/E_DENSITY_ASSET_SHA256 reason
    try:
        data=json.loads(path.read_text());z=np.asarray(data['altitude_km'],float)
        air=np.asarray(data['air_number_density_ratio'],float);oz=np.asarray(data['ozone_number_density_m3'],float)
        if data['version']!=1 or not np.array_equal(z,np.arange(161)*.5):raise ValueError()
        if air.shape!=z.shape or oz.shape!=z.shape or not np.isfinite([air,oz]).all() or np.any(air<=0) or np.any(oz<0) or not np.any(oz>0) or air[0]!=1:raise ValueError()
        tau=float(data['rayleigh_tau550_0_80'])
        if not np.isfinite(tau) or tau<=0:raise ValueError()
    except (KeyError,ValueError,TypeError,OSError) as e:raise ValueError('E_DENSITY_ASSET') from e
    return data,z,air,oz


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _freeze(optics):
    optics['table'].flags.writeable=False   # attach() consumers share it; a changed table needs a new optics/Solver
    return optics


@lru_cache(None)
def no2_asset():
    path = Path(__file__).with_name('assets')/'no2_afglus_profile.json'
    raw = path.read_bytes()
    data = json.loads(raw)
    z = np.asarray(data['altitude_km'], 'f8')
    v = np.asarray(data['no2_number_density_m3'], 'f8')
    if (not np.array_equal(z, grid(161)) or v.shape != z.shape or
            not np.isfinite(v).all() or np.any(v < 0) or not v.max() > 0):
        raise ValueError('E_NO2_PROFILE_ASSET')
    f = v/v.max()
    if not np.isfinite(f.astype('f4')).all() or np.any((f > 0) & (f.astype('f4') == 0)):
        raise ValueError('E_NO2_PROFILE_FP32')
    z.flags.writeable = f.flags.writeable = False
    return z, f, hashlib.sha256(raw).hexdigest()


def no2_profile(h):
    z, f, _ = no2_asset()
    return np.interp(h, z, f)


def _minor_columns(optics, step, state):
    z = grid(PROFILE_STEPS[step])
    f = no2_profile(z)
    optics['table'] = np.column_stack([optics['table'], f])
    optics['air_surface'] = float(optics['table'][0, 0]) if optics['mask'] & 1 else 1.
    if not np.isfinite(optics['air_surface']) or optics['air_surface'] <= 0:
        raise ValueError('E_PROFILE_AIR: air_surface must be finite and positive')
    from .components import o4_enabled, reference_active
    ext = external(state)
    if ext is not None:
        n0 = ext.get('air_surface_number_density_m3')
    elif options(state)['density_profile'] == 'usstd76':
        n0 = load_asset()[0]['air_sea_level_number_density_m3']
    else:
        n0 = 2.5469e25
    if n0 is not None and (isinstance(n0, bool) or not isinstance(n0, (int, float)) or not np.isfinite(n0) or n0 <= 0):
        raise ValueError('E_O4_N0: air_surface_number_density_m3 must be finite and positive')
    if o4_enabled(state) and n0 is None and not reference_active():
        raise ValueError('E_O4_N0: external profile needs air_surface_number_density_m3 when O4 is enabled')
    from .rayleigh import coefficient
    air_column = float(np.trapezoid(optics['table'][:,0], z)) if optics['mask'] & 1 else LEGACY_COLUMN
    optics['meta']['rayleigh_tau550'] = air_column*optics['scale']*float(.013558 if reference_active() else coefficient(550.))
    optics['meta'].update(no2=dict(asset_sha256=no2_asset()[2], normalization='maximum=1',
        H_f_km=float(np.trapezoid(f, z)), source='afglus asset (also for external air/ozone)',
        grid_step_km=step), o4=dict(shape='square of interpolated air / air_surface',
        air_surface=optics['air_surface'], n0=n0))
    optics['meta']['profile_identity'] = profile_identity(optics)['sha256']
    return _freeze(optics)


def build(state):
    from .components import no2_du
    no2_du(state)  # Validate optical amount; deliberately excluded from shape/cache identity.
    opt=options(state);step=profile_step_km(state);ext=external(state)
    if ext is not None:return _minor_columns(_external(opt,step,ext),step,state)
    if opt==DEFAULTS:
        z=grid(PROFILE_STEPS[step])
        optics=dict(table=np.column_stack([np.exp(-z/8),np.maximum(0,1-np.abs(z-25)/15)]),mask=0,scale=1.,
                    meta=dict(**opt,asset_sha256=None,origin='analytic',profile_step_km=step,
                              normalization='analytic exp8/triangle',rayleigh_tau550=.013558*LEGACY_COLUMN))
        return _minor_columns(optics,step,state)
    data,z,air,oz=load_asset()
    table=np.column_stack([air*LEGACY_COLUMN/np.trapezoid(air,z),oz*15/np.trapezoid(oz,z)])
    mask=(1 if opt['density_profile']=='usstd76' else 0)|(2 if opt['ozone_profile']=='usstd76' else 0)
    scale=1. if opt['rayleigh_column']=='keep' else data['rayleigh_tau550_0_80']/(.013558*LEGACY_COLUMN)
    optics=dict(table=table,mask=mask,scale=scale,meta=dict(**opt,asset_sha256=hashlib.sha256(asset_path().read_bytes()).hexdigest(),
                rayleigh_tau550=.013558*LEGACY_COLUMN*scale,ozone_column='state ozone_du preserved',interpolation='linear 0.5 km'))
    if step!=DEFAULT_STEP:
        # Normalized on the asset's own 161 nodes above; linear refinement keeps the function (and its integrals).
        fine=grid(PROFILE_STEPS[step])
        optics['table']=np.column_stack([np.interp(fine,z,table[:,k]) for k in (0,1)])
        optics['meta'].update(interpolation=f'linear {step} km (asset 0.5 km nodes, linear refinement)',profile_step_km=step,
                              origin='asset',normalization=NORMALIZATION_ASSET)
    return _minor_columns(optics,step,state)


NORMALIZATION_ASSET='asset: air*LEGACY_COLUMN/trapezoid, ozone*15/trapezoid on the 0.5 km asset nodes'


def _external(opt,step,ext):
    z,table=ext['z'],np.array(ext['table'],'f8')
    if ext['normalize_ozone']:
        total=np.trapezoid(table[:,1],z)
        if not total>0:raise ValueError('E_PROFILE_OZONE_ZERO: normalize_ozone needs a positive ozone integral')
        table[:,1]=table[:,1]*15/total
    scale=ext['scale']
    optics=dict(table=table,mask=ext['mask'],scale=scale,meta=dict(**opt,asset_sha256=None,origin='external',
                profile_step_km=step,interpolation=f'linear {step} km (external nodes)',
                normalization='ozone column scaled to 15 on the injected grid' if ext['normalize_ozone'] else 'none: injected values used as given',
                input_table_sha256=_sha(ext['table']),air_integral_km=float(np.trapezoid(table[:,0],z)),
                ozone_integral_km=float(np.trapezoid(table[:,1],z)),rayleigh_tau550=.013558*LEGACY_COLUMN*scale,
                ozone_column='state ozone_du times ozone_integral_km/15',
                note='density/ozone/rayleigh strings are metadata only; mask and scale come from the external payload'))
    return optics


def profile_identity(optics):
    """What every PROFILE consumer (host sun/endpoint/ground, device memo, column tables, strato memo) must key on:
    grid, interpolation, mask, scale, normalization, origin and the effective FP64 table bytes (after validation and
    normalization). Two tables with equal integrals but different shapes, or two wavelengths' q_eff, differ here."""
    check_consumed(optics,where='profile_identity')
    t=np.ascontiguousarray(optics['table'],'f8');n=int(t.shape[0]);meta=optics.get('meta',{})
    fields=dict(version=1,n=n,step_km=STEP_OF_N[n],top_km=80.,interpolation='linear',mask=int(optics['mask']),
                scale=float(optics['scale']).hex(),origin=meta.get('origin','asset'),
                normalization=meta.get('normalization',NORMALIZATION_ASSET),source_sha256=meta.get('asset_sha256'),
                table=dict(shape=list(t.shape),dtype=t.dtype.str,sha256=_sha(t)))
    return dict(fields,sha256=hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest())


def column_identity(optics):
    """Only integrated shapes/grid/normalization, without optical coefficients."""
    from .column_table import molecular
    fields = dict(components=[molecular(optics, k) for k in range(4)])
    return dict(fields, sha256=hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest())


def check_consumed(optics,compiled_n=None,rows=None,where='consume'):
    """The one PROFILE N check before any consumer: the declared step (meta.profile_step_km, 0.5 when unstated)
    == table rows (shape (N,3)) [== uploaded rows] [== compiled PROFILE_N]. Every path calls it, whether or not setup()
    ran and on either backend: fused.Kernel.check_profile (setup hit/miss, upload, launch, payload, installed constants),
    reference.profile_at, profiles, profile_identity. Returns N; E_PROFILE_N_MISMATCH otherwise."""
    meta=optics.get('meta',{});declared=meta.get('profile_step_km',DEFAULT_STEP)
    n=PROFILE_STEPS.get(float(declared)) if isinstance(declared,(int,float,np.integer,np.floating)) and not isinstance(declared,bool) else None
    shape=np.shape(optics['table']);table_rows=int(shape[0]) if len(shape) else -1
    bad=n is None or shape!=(n,3) or (rows is not None and int(rows)!=n) or (compiled_n is not None and int(compiled_n)!=n)
    if bad:
        raise RuntimeError(f'E_PROFILE_N_MISMATCH: {where}: declared step {declared!r} km (N={n}), table {shape} ({table_rows} rows)'
                           +('' if rows is None else f', uploaded rows {rows}')+('' if compiled_n is None else f', compiled PROFILE_N={compiled_n}'))
    return n


optics_n=check_consumed   # Earlier name


def upload_buffer(optics):
    """The exact FP32 PROFILE upload (fused.constants_device): the f4 audit SHA is taken from this buffer."""
    return np.ascontiguousarray(optics['table'],'f4')


def evidence(optics,compiled_n=None,uploaded=None):
    """Run metadata: compiled N, uploaded N, consumed mask, effective f8/f4 table SHA, identity, normalization, origin."""
    if optics is None:return dict(compiled_n=compiled_n,uploaded_n=None,mask=0,profile_identity=None,origin='analytic')
    ident=profile_identity(optics);buf=upload_buffer(optics) if uploaded is None else uploaded
    return dict(compiled_n=compiled_n,uploaded_n=int(buf.shape[0]),optics_n=optics_n(optics),mask=int(optics['mask']),
                step_km=ident['step_km'],origin=ident['origin'],normalization=ident['normalization'],scale=float(optics['scale']),
                table_f8_sha256=ident['table']['sha256'],table_f4_sha256=_sha(buf),profile_identity=ident['sha256'])


def interpolate32(h,table):
    """Operation order shared with kernel.cu; no FMA. Node count from the table (161/321/641)."""
    t=np.asarray(table,'f4');inv,last=KERNEL_LITERALS[t.shape[0]]
    h=np.clip(np.asarray(h,'f4'),np.float32(0),np.float32(80));x=h*np.float32(inv)
    i=np.minimum(x.astype('i4'),last);f=x-i.astype('f4')
    return t[i]+f[...,None]*(t[i+1]-t[i])


def profiles(h,optics):
    z=grid(check_consumed(optics,where='profiles'))
    h=np.clip(np.asarray(h,float),0,80);t=optics['table']
    return np.stack([np.interp(h,z,t[:,k]) for k in (0,1,2)],-1)


def molecular_profiles(h, optics):
    """Air, O3, NO2, O4 values; O4 squares the interpolated active air."""
    tab = profiles(h, optics)
    air = tab[..., 0] if optics['mask'] & 1 else np.exp(-np.maximum(h, 0)/8)
    oz = tab[..., 1] if optics['mask'] & 2 else np.maximum(0, 1-np.abs(h-25)/15)
    q = air/(optics['table'][0, 0] if optics['mask'] & 1 else 1.)
    return np.stack([air, oz, tab[..., 2], q*q], -1)


SUN_METHOD_ENV='SOSFUSE2_SUN_METHOD'
SUN_METHODS=('gl','exact')


def sun_method():
    """The one reading of SOSFUSE2_SUN_METHOD.

    gl (unset/empty = default) = the accepted rules: GL32 solar-table rows, GL24 cell->ground rays.
    exact = the selected PROFILE columns of those two tables by the FP64 closed form (exact_profile_columns).
    Main-ray endpoint columns (correct_columns, column_table.method) never read it. Callers resolve it once and pass
    the value to their workers; host_cache (sun02/ground) and stratosphere._geometry_key key on it."""
    import os
    value=os.environ.get(SUN_METHOD_ENV,'').strip() or 'gl'
    if value not in SUN_METHODS:raise ValueError('E_SUN_METHOD: expected gl or exact')
    return value


def exact_profile_columns(p,d,end,optics,selected=(0,1,2,3)):
    """(n, 4) air/O3/NO2/O4 columns over the straight path [0, end]: integral of the GL rules' integrand
    q(clip(h,0,80)) (profiles). Only mask-selected columns are filled (replace() reads no other).

    = column_table.exact_columns (closed shell integrals of the piecewise-linear table, both tangent sides; q=0 outside
    0..80 km) + q(0) times the path length inside the ground sphere + q(80) times the length outside the 80 km sphere.
    Real paths have neither (they end on the ground or the top), but the ground-shell half chord sqrt((R-rho)(R+rho))
    of a ray leaving the ground almost horizontally loses digits (rho~R): without the q(0) term such a ray dropped up to
    4e-5 km of air (stage 0(b) exact_check). The extra lengths use the closed form's own first/last shell chords, so the
    two partition the path consistently. The path is the caller's: unchanged ray (x=0 mu=mg+1e-9) and
    end=probe.boundaries(p,d)[:,-1] (ground end, zero length at the top). Base/unselected columns, strato columns and the
    Rayleigh scale are not touched here."""
    from .column_table import R as RADIUS, Z as TOP, exact_columns, molecular
    d=np.asarray(d,'f8');p=np.broadcast_to(p,d.shape);end=np.asarray(end,'f8');out=np.zeros((len(d),4))
    sa=np.sum(p*d,axis=-1);rho=np.sqrt(np.maximum(np.sum(p*p,axis=-1)-sa*sa,0));sb=sa+end   # as exact_columns
    def inside(radius):
        c=np.sqrt(np.maximum((radius-rho)*(radius+rho),0));return np.maximum(np.minimum(sb,c)-np.maximum(sa,-c),0)
    below=inside(RADIUS);above=np.maximum(end-inside(RADIUS+TOP),0)
    t=molecular_profiles(np.array([0.,80.]),optics)
    for k in selected:
        if k>=2 or optics['mask']&(1<<k):
            out[:,k]=exact_columns(molecular(optics,k),p,d,end)+t[0,k]*below+t[-1,k]*above
    if not np.isfinite(out).all():raise ValueError('E_SUN_METHOD_NONFINITE')
    return out


def correct_columns(p,d,base,old,optics,n=24,*,end=None):
    """Only selected columns change; accepted segment decomposition and GL order."""
    out=np.array(base,copy=True)
    selected=[(col,k) for k,col in enumerate(MOLECULAR_COLUMNS) if k>=2 or optics['mask']&(1<<k)]
    sums=np.zeros((len(d),4));p=np.broadcast_to(p,np.shape(d))
    for active,q,r,ds,prefix in old.geometry_segments(p,d,n):
        sums[active]+=np.sum(molecular_profiles(r-6371.,optics)*ds[...,None],axis=1)
    # New absorbers split at their profile knots even in the historical GL mode;
    # the original air/ozone quadrature and its rounding remain unchanged.
    if end is None:end=old.probe.boundaries(p,d)[:,-1]
    sums[:,2:]=exact_profile_columns(p,d,end,optics,selected=(2,3))[:,2:]
    for col,k in selected:out[:,col]=sums[:,k]
    return out


def attach(s,scope,defer_ground=False):
    if hasattr(s,'density_optics'):return s.density_optics
    optics=build(s.fuse_state)
    if optics is None:return None
    old=scope['b'].sh.old;sun=s.geo.sun;optics['old']=old
    s.ext[0]*=optics['scale']
    if optics['mask']&1:s.hr=profiles(s.geo.h,optics)[:,0]
    method=sun_method()   # once for the sun table and the first-ground rays, cached or not
    s.density_sun_method=method   # run record (run_gpu_state_v3 meta.json profile.sun_method); not part of optics
    from . import host_cache
    if host_cache.enabled():
        try:
            sun.col=host_cache.corrected_sun(s,optics,method=method)
            if not defer_ground:host_cache.prepare_ground(s,scope,optics,method=method)
        except host_cache.CacheUnavailable:
            pass  # Reuse the accepted uncached sun/ground path below.
        else:
            s.density_optics=optics
            return optics
    grid=sun.col.copy()
    for i,h in enumerate(sun.h):
        r=6371.+h;mg=-np.sqrt(max(0.,1-(6371./r)**2));mu=mg+(1-mg)*sun.x**2;mu[0]=mg+1e-9
        p=np.column_stack([np.zeros(len(mu)),np.zeros(len(mu)),np.full(len(mu),r)])
        d=np.column_stack([np.sqrt(np.maximum(1-mu*mu,0)),np.zeros(len(mu)),mu])
        if method=='exact':
            from .ray_columns import replace
            grid[i]=replace(grid[i],exact_profile_columns(p,d,old.probe.boundaries(p,d)[:,-1],optics),optics)
        else:grid[i]=correct_columns(p,d,grid[i],old,optics,32)
    sun.col=grid  # In-memory only: saved legacy sun_columns.npz is untouched.
    original=old.total_columns
    if method=='exact':
        from .ray_columns import replace
        old.total_columns=lambda p,d,n:replace(original(p,d,n),exact_profile_columns(p,d,old.probe.boundaries(p,d)[:,-1],optics),optics)
    else:old.total_columns=lambda p,d,n:correct_columns(p,d,original(p,d,n),old,optics,n)
    try:
        if not defer_ground:s.prepare_ground()
    finally:old.total_columns=original
    s.density_optics=optics
    return optics


def metadata(s):
    return dict(DEFAULTS) if getattr(s,'density_optics',None) is None else dict(s.density_optics['meta'])
