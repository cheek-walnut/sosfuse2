"""Atmospheric calculation boundary and independent sampling policies, in km."""

import math
from dataclasses import dataclass

TOP_KM = 80.0
NORMALIZATION_TOP_KM = 80.0
SUN_HEIGHT_INTERVALS = 80
COLUMN_TOP_MARGIN_KM = 1.0
UPPER_AIR_ASSET_SHA256 = '0bc815afb9fca955b9b9df345824e0dd8148e7d73b47e3d548de6654061e5f88'


@dataclass(frozen=True)
class Domain:
    """Calculation boundary, profile spacing and component support of one execution."""

    top_km: float = TOP_KM
    step_km: float = 0.5
    upper_heights_km: tuple = ()
    upper_air: str | None = None

    @property
    def n(self):
        return profile_n(self.step_km, self.top_km)

    @property
    def air_end(self):
        return self.top_km if self.upper_air == 'us76_neff' else min(self.top_km, NORMALIZATION_TOP_KM)

    @property
    def lower_end(self):
        return min(self.top_km, NORMALIZATION_TOP_KM)

    @property
    def extra_shells(self):
        return (10.0, 25.0, 40.0) + ((NORMALIZATION_TOP_KM,) if self.top_km > NORMALIZATION_TOP_KM else ())

    @property
    def ring_heights(self):
        return (0.0, 1.2, 8.0, 25.0, 40.0) + (
            (NORMALIZATION_TOP_KM, self.top_km) if self.top_km > NORMALIZATION_TOP_KM else (self.top_km,)
        )

    def identity(self):
        return dict(top_km=self.top_km, step_km=self.step_km, upper_heights_km=list(self.upper_heights_km),
                    upper_air=self.upper_air, upper_asset_sha256=UPPER_AIR_ASSET_SHA256 if self.upper_air == 'us76_neff' else None,
                    air_end_km=self.air_end, lower_end_km=self.lower_end)

    def defines(self):
        return profile_kernel_defines(self.step_km, self.top_km) + (
            '-DAIR_END_KM=' + _float_literal(self.air_end),
            '-DLOWER_END_KM=' + _float_literal(self.lower_end),
        )

    def monitors(self):
        stop = [2, -3, -6, -9, -12] + ([-15] if self.top_km > NORMALIZATION_TOP_KM else [])
        return dict(stop=stop, acceleration=list(stop), record=[-18] if self.top_km > NORMALIZATION_TOP_KM else [])


def resolve(settings, step=0.5, *, reduced=False):
    """Validate top settings without changing an absent default or shared input."""
    value = settings.get('top_km', TOP_KM)
    if type(value) not in (int, float) or not math.isfinite(value) or not TOP_KM <= value <= 200:
        raise ValueError('E_SETTINGS_TOP: top_km must be a finite number in [80, 200]')
    try:
        profile_n(step, value)
    except ValueError as exc:
        raise ValueError('E_SETTINGS_TOP: top_km must be an integral multiple of the PROFILE spacing') from exc
    if value == TOP_KM:
        if any(k in settings for k in ('upper_heights_km', 'upper_air')):
            raise ValueError('E_SETTINGS_TOP: upper settings require top_km > 80')
        return Domain(float(value), float(step))
    if reduced:
        raise ValueError('E_SETTINGS_TOP: extended top cannot use the reduced grid')
    nodes = settings.get('upper_heights_km')
    if 'upper_heights_km' not in settings:
        if value % 10:
            raise ValueError('E_SETTINGS_TOP: upper_heights_km is required for a top not divisible by 10')
        nodes = list(range(90, int(value) + 1, 10))
    if (not isinstance(nodes, list) or not nodes
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in nodes)
            or any(b <= a for a, b in zip([TOP_KM, *nodes[:-1]], nodes)) or nodes[-1] != value):
        raise ValueError('E_SETTINGS_TOP: upper heights must increase strictly above 80 and end at top_km')
    air = settings.get('upper_air', 'us76_neff')
    if air not in ('vacuum', 'us76_neff'):
        raise ValueError('E_SETTINGS_TOP: upper_air must be vacuum or us76_neff')
    return Domain(float(value), float(step), tuple(float(x) for x in nodes), air)


def profile_n(step, top=None):
    """Number of nodes on a uniform profile grid, including both endpoints."""
    top = TOP_KM if top is None else top
    try:
        if isinstance(step, bool) or isinstance(top, bool):
            raise ValueError
        step, top = float(step), float(top)
        if not math.isfinite(step) or not math.isfinite(top) or step <= 0 or top <= 0:
            raise ValueError
        intervals = top / step
        nearest = round(intervals)
        if nearest < 1 or not math.isclose(intervals, nearest, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError('E_PROFILE_GRID: top must be a positive integral multiple of step') from exc
    return nearest + 1


def profile_kernel_constants(step, top=None):
    """Return the inverse spacing and last interval for the same grid."""
    n = profile_n(step, top)
    return 1.0 / float(step), n - 2


def _float_literal(value):
    text = format(float(value), '.9g')
    if '.' not in text and 'e' not in text.lower():
        text += '.0'
    return text + 'f'


def profile_kernel_defines(step, top=None):
    """NVRTC definitions for the boundary and its uniform profile grid."""
    top = TOP_KM if top is None else top
    n = profile_n(step, top)
    inverse, last = profile_kernel_constants(step, top)
    return (
        f'-DPROFILE_N={n}',
        f'-DPROFILE_INV={_float_literal(inverse)}',
        f'-DPROFILE_LAST={last}',
        f'-DTOP_KM={_float_literal(top)}',
    )
