"""Fixed separable optical components; stratospheric aerosol is a separate field.

CUDA's compile-time NCOMP is checked against this order by test_b2_components.
B2 uses Bodhaine Rayleigh and the NO2/O4 cross sections on those columns.

Physics identity of a product (table meta, spectral block, source.npz, q16 header):
- physics() is what this code writes.
- A product without the field was written before B2 and reads as LEGACY: it is
  read on its own, never refused for the missing field.
- Combining products (table + weights, a source with this solver, merges, comparisons) needs one identity:
  check_same refuses legacy with current as E_PHYSICS_MIXED.
- A declared value this code does not know is E_PHYSICS_CONTRACT.
"""

import copy
from contextlib import contextmanager
from contextvars import ContextVar
import math
import re

from ..foundation.content_identity import state_identity

COMPONENTS = ('air', 'tropo_aerosol', 'o3', 'no2', 'o4')
NCOMP = len(COMPONENTS)
MOLECULAR_COLUMNS = (0, 2, 3, 4)
SINV_F = 1 + NCOMP + 1  # path weight, separable columns, stratosphere

LEGACY = dict(rayleigh='lambda4_v0', minor_absorbers='off')
S1_REFERENCE = dict(rayleigh='lambda4_v0', components=list(COMPONENTS), minor_absorbers='off', fixture='s1_reference')
_REFERENCE = ContextVar('sosfuse2_s1_reference', default=False)


@contextmanager
def _s1_reference():
    """Private in-process test/gate fixture; never a production environment switch."""
    token = _REFERENCE.set(True)
    try:
        yield
    finally:
        _REFERENCE.reset(token)


def reference_active():
    return _REFERENCE.get()


def production_only():
    if reference_active():
        raise ValueError('E_PHYSICS_REFERENCE: s1_reference is only for test/gate fixtures')


def _b2(o4_sha):
    from .minor_absorbers import CIA_SHA, RULES_VERSION, load

    return dict(
        rayleigh='bodhaine_iprt450air_vac_v1',
        phase_depolarization=0.03,
        components=list(COMPONENTS),
        minor_absorbers=dict(
            no2=dict(asset_sha256=load('no2')['sha256'], temperature_K=220),
            o4=dict(source_sha256=CIA_SHA, rules_version=RULES_VERSION, asset_sha256=o4_sha),
            o4_density=dict(o2_vmr=0.20946, shape='q^2', n0_rule='usstd76 asset; exp8 2.5469e25; external explicit'),
            no2_normalization='maximum=1, coefficient divided by actual H_f_km',
        ),
    )


def physics():
    if reference_active():
        return copy.deepcopy(S1_REFERENCE)
    from .minor_absorbers import hashes

    return _b2(hashes()['o4'])


def identity(value):
    """The physics identity a product declares: None (no field) = LEGACY; otherwise one this code knows."""
    if value is None:
        return dict(LEGACY)
    if isinstance(value, dict):
        if value == LEGACY or value == S1_REFERENCE:
            return copy.deepcopy(value)
        try:
            from .minor_absorbers import canonical_hashes

            value = copy.deepcopy(value)
            minors = value['minor_absorbers']
            for kind, digest in canonical_hashes({k: minors[k]['asset_sha256'] for k in ('no2', 'o4')}).items():
                minors[kind]['asset_sha256'] = digest
            sha = value['minor_absorbers']['o4']['asset_sha256']
            if (sha is None or isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha)) and value == _b2(sha):
                return copy.deepcopy(value)
        except (KeyError, TypeError):
            pass
    raise ValueError(f'E_PHYSICS_CONTRACT: unknown optical physics identity {value!r}')


def check_physics(value):
    """Reading one product on its own: any known identity, legacy included. Returns the identity."""
    return identity(value)


def check_same(*values, what='products'):
    """Combining products: all must have one identity (missing = LEGACY). Returns it."""
    ids = [identity(v) for v in values]
    if any(i != ids[0] for i in ids[1:]):
        raise ValueError(f'E_PHYSICS_MIXED: {what} of different optical physics {ids}')
    return ids[0]


def check_current(value, what='product'):
    """A product used together with this code (a source re-queried by this solver): its identity must be physics()."""
    return check_same(value, physics(), what=f'{what} and this code')


def no2_du(state):
    value = state.get('parameters', {}).get('no2_du', 0.207)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('E_NO2_DU: finite non-negative no2_du required')
    return float(value)


def o4_enabled(state):
    value = state.get('parameters', {}).get('o4', True)
    if type(value) is not bool:
        raise ValueError('E_O4_SWITCH: o4 must be a bool')
    return value


def optics_state(state, optics=None):
    """Resolved optical amounts, separate from shape/cache identities."""
    if optics is None:
        from .density import build

        optics = build(state)
    return dict(no2_du=no2_du(state), o4=o4_enabled(state), n0=optics['meta']['o4']['n0'])


def canonical_state(state):
    """The state as contracts and signatures carry it: state_identity (no note or label keys) with the
    resolved density options and NO2 and O4 amounts filled in."""
    from .density import options

    result = state_identity(state)
    result['density'] = dict(result.get('density', {}), **options(state))
    result.setdefault('parameters', {}).update(no2_du=no2_du(state), o4=o4_enabled(state))
    return result


def check_optics(*values):
    """Same-state source reuse/merge/resume must retain all resolved amounts."""
    for value in values:
        if not isinstance(value, dict) or set(value) != {'no2_du', 'o4', 'n0'}:
            raise ValueError('E_OPTICS_STATE_MIXED: missing or incomplete optics_state')
        no2_du(dict(parameters=value))
        o4_enabled(dict(parameters=value))
        n0 = value['n0']
        if n0 is not None and (
            isinstance(n0, bool) or not isinstance(n0, (int, float)) or not math.isfinite(n0) or n0 <= 0
        ):
            raise ValueError('E_OPTICS_STATE_MIXED: invalid n0')
        if value['o4'] and n0 is None:
            raise ValueError('E_OPTICS_STATE_MIXED: enabled O4 needs n0')
    if any(value != values[0] for value in values[1:]):
        raise ValueError('E_OPTICS_STATE_MIXED: optical amounts differ')
    return copy.deepcopy(values[0])


def check_same_state(*metas):
    """Combining same-state products, including unsigned derived tables."""
    result = check_same(*(meta.get('physics') for meta in metas))
    if result != LEGACY:
        check_optics(*(meta.get('optics_state') for meta in metas))
    return result
