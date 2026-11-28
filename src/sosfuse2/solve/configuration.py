"""State selection, input validation and the grid rules used by solver assembly."""

import contextlib
import copy
import hashlib
import json
import os
import re

from sosfuse2.foundation import top
from sosfuse2.foundation.top import TOP_KM

# Production kernel SH order. 16 = production with SH_COMBINE in backends/configs/C3.json (measured:
# CUDA measurement: L16+SH_COMBINE+nphi16+sun4 vs L32+nphi32+sun1 3.3-5.2x faster, order 2 vs direct L8 equal or 0.03 %p
# better, full radiance within 0.40 %). settings_json may still choose 24 or 32.
SH_L = 16
# Production Anderson fixed-point tolerance (observer and source relative residual). .001 = production (measured:
# .005 left up to 0.75 % iteration error at -9 deg, .001 -> .0002
# changes <= 0.16 %); was .005. The persistent service (pipeline_execution.py) uses the same validated setting.
ANDERSON_TOLERANCE = 0.001


def effective_tol(settings):
    value = settings.get('anderson_tol', ANDERSON_TOLERANCE)
    if type(value) not in (int, float) or not 1e-6 <= value <= 1e-2:
        raise ValueError('E_SETTINGS_TOL: anderson_tol must be a number in [1e-6, 1e-2]')
    return float(value)


AUTO_FLAGS = ('STRATO', 'TROPO_MIE', 'AEROSOL_2L')


def kernel_settings(block_size=None):
    """Resolve the kernel variant and block size. None selects default transport with block 128."""
    from sosfuse2.backends.flags import KERNEL_FLAGS, resolve_config

    flags = sorted({f for f in os.environ.get('SOSFUSE2_KERNEL_FLAGS', '').replace(' ', '').split(',') if f})
    flags = list(resolve_config(flags).flags)
    allowed = [f for f in KERNEL_FLAGS if f not in AUTO_FLAGS]
    if set(flags) - set(allowed):
        raise ValueError(
            'E_KERNEL_FLAGS: SOSFUSE2_KERNEL_FLAGS from ' + ','.join(allowed) + ' (physical selectors are automatic)'
        )
    text = os.environ.get('SOSFUSE2_BLOCK_SIZE', '').strip()
    if text not in ('', '128', '256'):
        raise ValueError('E_KERNEL_BLOCK_SIZE: SOSFUSE2_BLOCK_SIZE must be 128 or 256')
    if text and block_size is not None and block_size != int(text):
        raise ValueError('E_KERNEL_BLOCK_SIZE: --block-size disagrees with SOSFUSE2_BLOCK_SIZE')
    block = int(text or 128) if block_size is None else block_size
    from sosfuse2.solve.ground_sh import configured_order, GRID_INTERVALS

    ground_order = configured_order()
    if not flags and block == 128 and not ground_order:
        return None
    settings = dict(flags=flags, block_size=block, source='env SOSFUSE2_KERNEL_FLAGS/SOSFUSE2_BLOCK_SIZE')
    if ground_order:
        settings['ground_sh'] = dict(order=ground_order, grid_intervals=GRID_INTERVALS, interpolation='cubic')
    return settings


# Height grid rules (settings key h_refine). The 20 base heights of shell_segments.grids have 14 intervals at or below
# 13 km and 5 above. 'b<m>a<k>' splits every interval at or below 13 km into m parts and every interval above into k parts;
# 'b<m>a<k1>.<k2>.<k3>.<k4>.<k5>' gives the five intervals above 13 km their own counts; m and k are 1-12. The names of
# 1.3.4 and earlier are aliases and give the same heights.
H_REFINE_ALIASES = {'none': 'b1a1', 'mid': 'b2a2', 'below13mid_above13thirds': 'b2a3'}
_H_RULE = re.compile(r'b([1-9]|1[0-2])a([1-9]|1[0-2])((?:\.(?:[1-9]|1[0-2])){4})?')
CHI_STEPS_DEG = (0.5, 0.25, 0.125)


def height_rule(rule):
    """(parts below 13 km, (parts of the five intervals above)) for an h_refine value, or None when it is not a rule."""
    match = _H_RULE.fullmatch(H_REFINE_ALIASES.get(rule, rule)) if isinstance(rule, str) else None
    if match is None:
        return None
    above = (int(match.group(2)),) * 5
    if match.group(3):
        above = (above[0], *(int(x) for x in match.group(3)[1:].split('.')))
    return int(match.group(1)), above


def refined_heights(h, rule):
    """Base heights h refined by an h_refine rule, sorted. Two parts add the midpoint (u + v) / 2, n parts add
    u + (v - u) * (j / n), so the aliases reproduce the 1.3.4 grids; 'none' returns h itself.
    The solver's h is a 1-D CPU float64 array of heights in km. Reads any stride without
    modifying h. New nodes are returned in a contiguous array when refinement is requested."""
    import numpy as np

    below, above = height_rule(rule)
    if below == 1 and set(above) == {1}:
        return h
    pairs = list(zip(h[:-1], h[1:]))
    low = [(u, v) for u, v in pairs if v <= 13 + 1e-9]
    high = [(u, v) for u, v in pairs if u >= 13 - 1e-9]
    if len(low) + len(high) != len(pairs) or len(high) != len(above):
        raise ValueError('E_SETTINGS_GRID: height rule %r does not fit the base heights' % rule)

    def inner(u, v, n):
        return [(u + v) / 2] if n == 2 else [u + (v - u) * (j / n) for j in range(1, n)]

    return np.sort(
        np.r_[
            h,
            [x for u, v in low for x in inner(u, v, below)],
            [x for (u, v), n in zip(high, above) for x in inner(u, v, n)],
        ]
    )


def refined_chi(ch, step, window=None):
    """Solar zenith nodes in radians: the base nodes outside 85-110 deg, `step` deg inside (0.5 = the base nodes), and the
    midpoint of every pair of neighbouring nodes inside `window` = [lo, hi] deg. Built in degrees, converted once.
    The solver's ch is a 1-D CPU float64 array in radians. Reads any stride without modifying
    ch. Refinement produces a contiguous array. Step None or 0.5 without a window returns ch itself."""
    import numpy as np

    if step in (None, 0.5) and window is None:
        return ch
    degrees = np.round(np.rad2deg(ch), 12)
    if step not in (None, 0.5):
        degrees = np.unique(np.r_[degrees[(degrees < 85) | (degrees > 110)], np.arange(85, 110 + step / 2, step)])
    if window is not None:
        windows = window if isinstance(window[0], list) else [window]
        for lo, hi in windows:
            middle = [(u + v) / 2 for u, v in zip(degrees[:-1], degrees[1:]) if u >= lo - 1e-9 and v <= hi + 1e-9]
            degrees = np.unique(np.r_[degrees, middle])
    return np.deg2rad(degrees)


def _chi_window(value, step, base_chi):
    """Checked chi_refine_deg as [lo, hi] floats: two numbers within 1e-9 deg of two different nodes of the step grid in
    85-110 deg, lo below hi; each is replaced by its node."""
    import numpy as np

    if isinstance(value, list) and value and isinstance(value[0], list):
        nodes = refined_chi(base_chi, step)
        windows = []
        for window in value:
            checked = _chi_window(window, None, nodes)
            windows.append(checked)
            nodes = refined_chi(nodes, None, checked)
        return windows
    if isinstance(value, list) and len(value) == 2 and all(type(x) in (int, float) for x in value):
        nodes = np.round(np.rad2deg(refined_chi(base_chi, step)), 12)
        nodes = nodes[(nodes >= 85) & (nodes <= 110)]
        with contextlib.suppress(OverflowError):  # a JSON integer too large for a float is refused like any other value
            near = [nodes[np.argmin(np.abs(nodes - float(x)))] for x in value]
            if all(abs(n - float(x)) < 1e-9 for n, x in zip(near, value)) and near[0] < near[1]:
                return [float(near[0]), float(near[1])]
    raise ValueError(
        'E_SETTINGS_GRID: chi_refine_deg must be [lo, hi] with 85 <= lo < hi <= 110 on the chi_step_deg nodes'
    )


def settings_for(a):
    import numpy as np

    # V14 production defaults; overrides are checked before any expensive work.
    cfg = dict(
        nr=4 if a.reduced else 44,
        h_refine='none' if a.reduced else 'below13mid_above13thirds',
        chi_step_deg=None if a.reduced else 0.5,
        nchi=10 if a.reduced else 75,
        nmu=1 if a.reduced else 12,
        nphi=2 if a.reduced else 16,
        path_n=4 if a.reduced else 12,
        L=SH_L,
        anderson_tol=effective_tol({}),
        interpolation='log',
        observer_km=0.1,
        wavelengths=1 if a.reduced else 15,
        wavelengths_nm=[550.0] if a.reduced else (360 + (np.arange(15) + 0.5) * 470 / 15).tolist(),
        solar_elevation_deg=[-3.0, -6.0] if a.reduced else np.arange(6.0, -19.0, -1.0).tolist(),
        view_shape=[4, 6] if a.reduced else [100, 180],
    )
    raw = json.loads(a.settings_json.read_text()) if a.settings_json else {}
    from sosfuse2.atmosphere import spectral

    # Spectral keys leave raw before the key check; legacy adds nothing, so explicit-legacy settings bytes are unchanged.
    spectral_raw = {k: raw.pop(k) for k in spectral.KEYS if isinstance(raw, dict) and k in raw}
    # chi_refine_deg has no default entry: absent (or null) means no refinement, so settings without it stay as they were.
    if not isinstance(raw, dict) or set(raw) - set(cfg) - {'chi_refine_deg', 'top_km', 'upper_heights_km', 'upper_air', 'export_first_order'}:
        raise ValueError('E_SETTINGS_KEYS: unknown settings')
    raw['anderson_tol'] = effective_tol(raw)
    domain = top.resolve(raw, 0.125, reduced=a.reduced)
    if 'export_first_order' in raw:
        if type(raw['export_first_order']) is not bool:
            raise ValueError('E_SETTINGS_VALUE: export_first_order must be bool')
        if not raw['export_first_order']:
            del raw['export_first_order']
    if domain.top_km > TOP_KM:
        raw.update(top_km=domain.top_km, upper_heights_km=list(domain.upper_heights_km), upper_air=domain.upper_air)
    # Product grid default band BM45; the reduced 550 nm smoke grid stays legacy unless a mode is named.
    if not a.reduced:
        spectral.product_default(spectral_raw, raw)
    cfg.update(raw)
    pairs = {20: 'none', 39: 'mid', 44: 'below13mid_above13thirds'}
    for k in ('nr', 'nchi', 'nmu', 'nphi', 'path_n', 'L', 'wavelengths'):
        if type(cfg[k]) is not int or cfg[k] < 1:
            raise ValueError('E_SETTINGS_VALUE: ' + k)
    if 'nr' in raw and 'h_refine' not in raw and domain.top_km == TOP_KM:
        cfg['h_refine'] = 'none' if a.reduced else pairs.get(cfg['nr'])
    if 'wavelengths' in raw and 'wavelengths_nm' not in raw:
        n = cfg['wavelengths']
        cfg['wavelengths_nm'] = (360 + (np.arange(n) + 0.5) * 470 / n).tolist()
    if len(cfg['wavelengths_nm']) != cfg['wavelengths']:
        raise ValueError('E_SETTINGS_WAVELENGTHS: inconsistent count')
    wl = np.asarray(cfg['wavelengths_nm'], float)
    if not np.isfinite(wl).all() or np.any(wl <= 0) or np.any(np.diff(wl) <= 0):
        raise ValueError('E_SETTINGS_WAVELENGTHS: positive ordered wavelengths required')
    if cfg['L'] not in (16, 24, 32) or cfg['interpolation'] != 'log':
        raise ValueError('E_SETTINGS_FIXED: L in {16,24,32} and log interpolation required')
    if cfg.get('chi_refine_deg', 0) is None:
        del cfg['chi_refine_deg']
    if a.reduced:
        # The reduced smoke grid is fixed: 4 heights, no refinement, its own 10 chi nodes.
        fixed = (cfg['nr'], cfg['h_refine'], cfg['nchi'], cfg['chi_step_deg']) == (4, 'none', 10, None)
        if not fixed or 'chi_refine_deg' in cfg:
            raise ValueError(
                'E_SETTINGS_GRID: received=%r; the reduced grid is nr 4, h_refine none, nchi 10, no chi_step_deg or '
                'chi_refine_deg' % ((cfg['nr'], cfg['h_refine']),)
            )
    else:
        from sosfuse2.rays.shell_segments import grids

        base_h, base_chi = grids(1)
        if height_rule(cfg['h_refine']) is None:
            raise ValueError(
                'E_SETTINGS_GRID: received=%r; h_refine must be none, mid, below13mid_above13thirds or '
                'b<m>a<k>[.<k2>.<k3>.<k4>.<k5>] with m, k in 1-12 (nr 20, 39, 44 alone name the first three)'
                % ((cfg['nr'], cfg['h_refine']),)
            )
        want = len(refined_heights(base_h, cfg['h_refine'])) + len(domain.upper_heights_km)
        if ('h_refine' in raw or domain.top_km > TOP_KM) and 'nr' not in raw:
            cfg['nr'] = want
        if cfg['nr'] != want:
            raise ValueError(
                'E_SETTINGS_GRID: received=%r; h_refine gives nr %d' % ((cfg['nr'], cfg['h_refine']), want)
            )
        step = cfg['chi_step_deg']
        if type(step) not in (int, float) or step not in CHI_STEPS_DEG:
            raise ValueError('E_SETTINGS_GRID: chi_step_deg must be 0.5, 0.25 or 0.125')
        if 'chi_refine_deg' in cfg:
            cfg['chi_refine_deg'] = _chi_window(cfg['chi_refine_deg'], step, base_chi)
        want = len(refined_chi(base_chi, step, cfg.get('chi_refine_deg')))
        if 'nchi' in raw and cfg['nchi'] != want:
            raise ValueError(
                'E_SETTINGS_GRID: received nchi %d; chi_step_deg and chi_refine_deg give %d' % (cfg['nchi'], want)
            )
        cfg['nchi'] = want
    if not 0 <= cfg['observer_km'] <= domain.top_km:
        raise ValueError('E_SETTINGS_OBSERVER')
    if (
        not isinstance(cfg['view_shape'], list)
        or len(cfg['view_shape']) != 2
        or any(type(x) is not int or x < 1 for x in cfg['view_shape'])
    ):
        raise ValueError('E_SETTINGS_VIEW')
    hs = np.asarray(cfg['solar_elevation_deg'], float)
    if hs.ndim != 1 or not np.isfinite(hs).all() or len(set(hs)) != len(hs) or -3 not in hs or -6 not in hs:
        raise ValueError('E_SETTINGS_SOLAR: unique finite elevations including -3,-6 required')
    # After the settings_json key check, so a settings file can never set or override it.
    kernel = kernel_settings(getattr(a, 'block_size', None))
    if kernel is not None:
        cfg['kernel'] = kernel
    block = spectral.block(spectral_raw, wl, 'wavelengths_nm' in raw)
    if block is not None:
        cfg['spectral'] = block
    json.dumps(cfg, allow_nan=False)
    return cfg


def execution_domain(cfg, state):
    """Resolve a calculation domain and reject unsupported density inputs before preparation."""
    from sosfuse2.atmosphere import density

    domain = top.resolve(cfg, 0.125, reduced=cfg.get('nr') == 4)
    if domain.top_km > TOP_KM:
        raw = state.get('density', {})
        if not isinstance(raw, dict) or raw.get('density_profile') != 'usstd76' or 'external_profile' in raw:
            raise ValueError('E_SETTINGS_TOP: extended top requires an explicit usstd76 air profile without external PROFILE')
        density.options(state)
    return top.resolve(cfg, density.profile_step_km(state), reduced=cfg.get('nr') == 4)


def resolve_settings(cfg, state):
    """Per-state settings (A′ resolver): a profile_mean spectral block takes
    its q identity from this (effective) state; the shared cfg is never changed (a new copy is returned). Without a
    profile_mean block the same cfg object comes back. Callers: run, run_extra, geometry_only, plan_queue.probe, reference-run worker."""
    from sosfuse2.atmosphere import spectral

    execution_domain(cfg, state)
    block = cfg.get('spectral')
    resolved = spectral.resolve(block, state)
    if resolved is block:
        return cfg
    out = copy.deepcopy(cfg)
    out['spectral'] = resolved
    return out


def grid_metadata(cfg, h=None):
    """Sign the height bytes without constructing solar tables or a solver."""
    import numpy as np

    if h is None:
        from sosfuse2.rays.shell_segments import grids, small_heights

        h = refined_heights(grids(1)[0], cfg['h_refine']) if cfg['nr'] != 4 else small_heights()
        if cfg.get('top_km', TOP_KM) > TOP_KM:
            h = np.r_[h, cfg['upper_heights_km']]
    result = dict(h_refine=cfg['h_refine'], h_sha256=hashlib.sha256(np.ascontiguousarray(h).tobytes()).hexdigest())
    result.update({k: cfg[k] for k in ('top_km', 'upper_heights_km', 'upper_air') if k in cfg})
    return result


def effective_state(state, a):
    state = copy.deepcopy(state)
    if getattr(a, 'stratosphere', None) == 'on':
        state.setdefault('stratosphere', {})['status'] = 'enabled'
    from sosfuse2.atmosphere.density import with_options

    return with_options(state, a)


def parse_elevations(text):
    values = [float(x) for x in str(text).split(',') if x.strip()]
    if not values or len(set(values)) != len(values) or not all(abs(v) < 90 for v in values):
        raise ValueError('E_EXTRA_ELEVATIONS: unique finite |h|<90 list required')
    return values


def extra_contract(cfg, extra, source_out_name):
    """M3 (shared by scheduler and runner): rows = extra elevations + the -3/-6
    anchors (previews, pack format and a bit-level reuse check against the
    source job's table).  cfg stays the SOURCE job's settings unchanged."""
    if not re.fullmatch('[A-Za-z0-9_-]+', str(source_out_name or '')):
        raise ValueError('E_EXTRA_SOURCE: safe source_out_name required')
    extra = parse_elevations(extra) if isinstance(extra, str) else [float(x) for x in extra]
    if set(extra) & set(float(x) for x in cfg['solar_elevation_deg']):
        raise ValueError('E_EXTRA_ELEVATIONS: already a source table row')
    rows = sorted(set(extra) | {-3.0, -6.0}, reverse=True)
    return dict(
        mode='extra_elevations',
        extra_elevations=sorted(extra, reverse=True),
        table_elevations=rows,
        source_out_name=source_out_name,
    )


def read_states(path, include_parent=False):
    doc = json.loads(path.read_text())
    states = list(doc['states'])
    if include_parent and doc.get('parent'):
        parent = json.loads((path.parent / doc['parent']).read_text())['states']
        own = {s['id'] for s in states}
        states.extend(s for s in parent if s['id'] not in own)
    return states



def state_settings(a, *, geometry_only=False, resolve=None):
    """Return the effective state and its cfg using only the input options and assets.

    Geometry prefetch validates grid settings before state overrides, preserving its CLI
    error order. The runner can supply its public per-state resolver at call time.
    """
    found = [s for s in read_states(a.states, True) if s['id'] == a.id and s.get('runnable')]
    if len(found) != 1:
        raise ValueError(f'E_STATE_ID: exactly one runnable state {a.id!r} required in {a.states}')
    if geometry_only:
        cfg = settings_for(a)
        state = effective_state(found[0], a)
    else:
        state = effective_state(found[0], a)
        cfg = settings_for(a)
    if resolve is None:
        resolve = resolve_settings
    return state, resolve(cfg, state)
