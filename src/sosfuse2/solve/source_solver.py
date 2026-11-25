"""CuPy RawKernel wrapper and NumPy reference, with no sample cache.

Only contiguous arrays are passed to NVRTC. Source layout is cell,10+K,wave
or cell,10+2*K,wave for an explicitly enabled stratosphere, where K is the SH mode count;
the inactive scalar diagnostic lane is omitted and never substitutes for the
polarized calculation. Boundary and source outputs retain all K SH modes.
"""

from dataclasses import replace
import importlib.util
import json
import os
import math
import time
import warnings
from pathlib import Path
import numpy as np
from ..backends import numpy_reference
from sosfuse2.backends import vram_budget, vram_trace
from ..atmosphere.components import NCOMP, STRATO_COLUMN_SLOT
from ..backends.arithmetic import F, norm_constants
from ..backends.sh_layout import (
    count,
    padded,
    workspace,
    source_size,
    kernel_source,
    combined_aux,
    combined_aux_waves,
    FastTransport,
)
from ..foundation.timers import TIMERS, stage
from ..foundation import top
from ..backends.flags import STAGE1_FLAGS, KSPLIT_VARIANTS, resolve_config, deterministic
from .ground_sh import GRID_INTERVALS
from .source_iteration import Solver  # noqa: F401


from ..foundation import paths  # noqa: E402  (public split: package-relative resources, work root for caches)

# R1 split (SOSFUSE2_KSPLIT=1): warm/ground launches become a column
# record pass once per bundle (SINV mode 1), then per chunk a producer (PROD: SH contraction -> own_aero) and a consumer
# (SLIM: transport from own_aero and the columns). Internal compile variants, never user kernel flags.
KSPLIT_DEFAULT = '3'  # SOSFUSE2_KSPLIT when unset (Kernel.__init__)
from .ksplit import (  # noqa: E402,F401 (re-exported: KSPLIT host side, ksplit.py)
    KsplitMixin,
    SLIM_PROJ_THREADS,
    SLIM_PROJ_MOMENTS,
    KSPLIT_BLOCK,
    VPROD_VISITORS,
    VPROD_ORDER,
    VPROD_THREADS,
    aero_w,
    VPROD_SENTINEL,
    VPROD_VERIFY_STAGE,
    VPROD_VERIFY_WINDOW,
    VPROD_STORE_BPP,
    VPROD_COUNT_BLOCKS,
    VPROD_COUNT_SHARED,
    VPROD_PLAN_SRC,
)
# Kernel variants (kernel.cu header); each independent, all off = original kernel.
# Stage-1 compile options: SOSFUSE2_<NAME>=1/0 overrides,
# unset = device profile stage1_defaults (backends/configs/devices.json: cuda FMAD+FASTMATH, split/mac none).
# None active = the exact pre-r17 options, key and kernel tokens.
KPAD = padded()
SECTION_CLOCK_NAMES = ('point', 'ground', 'sh_basis', 'transport', 'attenuation', 'projection', 'barrier', 'total')


def section_clock_summary(counters):
    """Lossless sums of disjoint lane-0 observer intervals, never wall seconds."""
    counters = np.asarray(counters)
    if (
        counters.dtype != np.dtype('uint64')
        or counters.ndim != 4
        or counters.shape[-1] != len(SECTION_CLOCK_NAMES)
        or any(n <= 0 for n in counters.shape[:-1])
    ):
        raise ValueError('E_SECTION_CLOCK_BUFFER: uint64 [cell,tile,warp,8] required')
    rows = counters.reshape(-1, len(SECTION_CLOCK_NAMES))
    if any(sum(map(int, row[:-1])) > int(row[-1]) for row in rows):
        raise ValueError('E_SECTION_CLOCK_OVERLAP: section intervals exceed observer total')
    sums = {name: sum(map(int, rows[:, i])) for i, name in enumerate(SECTION_CLOCK_NAMES)}
    total = sums['total']
    residual = total - sum(sums[name] for name in SECTION_CLOCK_NAMES[:-1])
    return dict(
        shape=list(counters.shape),
        observer_cycles=sums,
        residual_observer_cycles=residual,
        fractions={name: sums[name] / total if total else None for name in SECTION_CLOCK_NAMES[:-1]},
        units='uint64 clock64 cycles per warp lane-0 observer',
        accounting='disjoint sections; ground includes internal warp sync; '
        'barrier is observed synchronization interval, not pure waiting; '
        'concurrent observer sums are not wall time',
    )


def default_flags():
    import os

    names = tuple(f for f in os.environ.get('SOSFUSE2_KERNEL_FLAGS', '').replace(' ', '').split(',') if f)
    return names


def sh_aux(src, xp=np, L=32):
    """SH source as a new contiguous [cell][wave][KPAD] array (not a view):
    sh[c, w, k] = src[c, 10+k, w] for k<561, zero padding to KPAD."""
    K = count(L)
    KPAD = padded(L)
    out = xp.zeros((src.shape[0], src.shape[2], KPAD), 'f4')
    out[:, :, :K] = xp.transpose(src[:, 10 : 10 + K, :], (0, 2, 1))
    if not out.flags.c_contiguous:
        raise RuntimeError('E_SH_AUX_LAYOUT')
    return out


def sh_aux_pair(src, xp=np, L=32):
    """Exact FP32 [cell,ceil(W/2),KPAD,2], adjacent wavelengths per float2.

    Only the optional SH sidecar changes. src, outputs, and bundle stay intact.
    The padded final wavelength is zero and is never accumulated.
    """
    cells, _, waves = src.shape
    K = count(L)
    KPAD = padded(L)
    out = xp.zeros((cells, (waves + 1) // 2, KPAD, 2), 'f4')
    out[:, :, :K, 0] = xp.transpose(src[:, 10 : 10 + K, 0::2], (0, 2, 1))
    if waves > 1:
        out[:, : waves // 2, :K, 1] = xp.transpose(src[:, 10 : 10 + K, 1::2], (0, 2, 1))
    if not out.flags.c_contiguous:
        raise RuntimeError('E_SH_AUX_PAIR_LAYOUT')
    return out


def sh_aux_waves(src, layout, xp=np):
    """FP32 [cell,mode,wave], padded to whole pairs for the transport variant."""
    cells, _, waves = src.shape
    K = count(layout.L)
    out = xp.zeros((cells, K, layout.stride), 'f4')
    out[:, :, :waves] = src[:, 10 : 10 + K, :]
    if not out.flags.c_contiguous:
        raise RuntimeError('E_SH_AUX_WAVE_LAYOUT')
    return out


def sh_spline_basis(layout):
    """Cubic cardinal spline coefficients on the two periodic angular axes.

    For the polar axis extend sin(theta)**m with its sign beyond pi.
    At integer knots the cubic B-spline has weights [1, 4, 1]/6, whose
    periodic convolution is inverted once per basis function in Fourier space.
    """
    from ..backends.arithmetic import basis

    order, bins = layout.L, layout.bins
    a = np.arange(2 * bins) * (np.pi / bins)
    m = np.array([m for l in range(order + 1) for m in range(l + 1)])
    polar = basis(np.cos(a), order).astype('f8')
    polar[bins + 1 :] *= np.where(m % 2, -1.0, 1.0)
    response = (2.0 + np.cos(a[: bins + 1])) / 3.0
    coeff = np.fft.irfft(np.fft.rfft(polar, axis=0) / response[:, None], axis=0)
    knots = np.arange(-1, bins + 2)
    orders = np.arange(order + 1)
    azimuth = np.cos(knots[:, None] * (np.pi / bins) * orders) / response[orders]
    return np.ascontiguousarray(coeff[knots], 'f4'), np.ascontiguousarray(azimuth, 'f4')


def load_engine():
    """A fresh, private copy of the accepted engine module (callers patch its namespace; never the shared import).
    The module name keeps the package context so its relative imports resolve; the solver modules stay shared."""
    spec = importlib.util.spec_from_file_location('sosfuse2.solve.engine_readonly', paths.ACCEPTED / 'engine.py')
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    return engine


from ..backends.device_profile import first_light_split_error  # noqa: E402 (stdlib-only module; pipeline preflight uses it too)


def ksplit_setting(backend, flags, L, W=15):
    """Resolve KSPLIT for the given wavelength count.
    The default needs CuPy, W=15, STRATO, L<=16 and uninstrumented transport.
    """
    return _ksplit_request(backend, resolve_config(flags, W=W), L)[0]


def _ksplit_request(backend, config, L):
    if config.instrumented:
        return '0', 'instrumented'
    text = os.environ.get('SOSFUSE2_KSPLIT', '').strip()
    if text:
        return text, 'explicit_zero' if text == '0' else None
    if backend != 'cupy':
        return '0', 'numpy'
    if config.W == 1:
        return '0', 'fw1'
    if L > 16:
        return '0', 'sh_order_gt_16'
    if not config.ksplit_supported:
        return '0', 'no_strato' if not config.strato else 'warm_raysplit'
    return KSPLIT_DEFAULT, None


class _FastResourceError(RuntimeError):
    """The compiled variant cannot launch on this device."""


def _transport_resource_error(exc):
    # Both CUDA APIs use 2 for allocation failure and 701 for launch resource
    # exhaustion. An invalid address, bad argument or budget error must propagate.
    return isinstance(exc, (MemoryError, _FastResourceError)) or (
        type(exc).__name__ in ('CUDADriverError', 'CUDARuntimeError') and getattr(exc, 'status', None) in (2, 701)
    )


class _TransportLaunch:
    """Owns arguments and scratch arrays until Kernel.launch returns."""


def _section_clock_directory(kernel):
    section_dir = None
    if kernel.config().instrumented:
        text = os.environ.get('SOSFUSE2_SECTION_CLOCKS_DIR', '').strip()
        if not text:
            raise ValueError('E_SECTION_CLOCK_OUTPUT: set SOSFUSE2_SECTION_CLOCKS_DIR')
        section_dir = Path(text).resolve()
        section_dir.mkdir(parents=True, exist_ok=True)
    return section_dir


def _prepare_transport_constants(kernel, launch, bundle, s, field, first, phase, boundary, _ksplit):
    launch.xp = kernel.xp
    launch.c = kernel.setup(s, bundle.nq)
    launch.dc = kernel.constants_device(s, bundle.nq)
    launch.W = s.W
    launch.ncell = bundle.cells
    launch.nr = bundle.rays
    launch.ks = kernel.ksplit and not first and phase in (0, 1)
    if launch.ks:
        if launch.W != 15 or 'strato_ext' not in launch.c:
            raise ValueError(
                'E_KSPLIT_STATE: SOSFUSE2_KSPLIT=1 needs 15 wavelengths and an enabled stratosphere '
                '(five separable columns plus the strato profile, kernel.cu SINV)'
            )
        launch.kb = bundle.ksplit if _ksplit else kernel.ksplit_columns(bundle, s, field, phase, boundary)
        if _ksplit == 'record':
            # Each recorded wavelength integrates one unit-basis extinction column.
            launch.e = launch.xp.zeros(launch.dc['ext'].size, launch.dc['ext'].dtype)
            for k in range(NCOMP):
                launch.e[k * launch.W + k] = 1
            launch.se = launch.xp.zeros(launch.dc['strato_ext'].size, launch.dc['strato_ext'].dtype)
            launch.se[STRATO_COLUMN_SLOT] = 1
            launch.dc = dict(
                launch.dc,
                ext=launch.e.reshape(launch.dc['ext'].shape),
                strato_ext=launch.se.reshape(launch.dc['strato_ext'].shape),
            )
        else:
            launch.kb['hdr32'][0:1].fill(2)


def _allocate_transport_outputs(kernel, launch, s, field, first, phase, boundary):
    with vram_trace.item('source.contiguous'):
        launch.src = launch.xp.ascontiguousarray(field, dtype='f4')
    with vram_trace.item('moments.first' if first else 'moments.transport'):
        launch.moments = (
            launch.xp.empty((launch.ncell, 4 + kernel.K, launch.W), 'f4') if phase == 0 else launch.xp.empty(1, 'f4')
        )
    with vram_trace.item('transport.flux'):
        launch.flux = launch.xp.empty((launch.ncell, launch.W), 'f4')
    with vram_trace.item('query.radiance'):
        launch.query = (
            launch.xp.empty((launch.nr, launch.W, 4), 'f4')
            if phase == 2
            else (launch.kb['buf'] if launch.ks else launch.xp.empty(1, 'f4'))
        )
    launch.bd = (
        launch.xp.ascontiguousarray(boundary, dtype='f4')
        if boundary is not None
        else launch.xp.zeros((len(s.geo.chi), launch.W), 'f4')
    )


def _assemble_transport_arguments(kernel, launch, bundle, s, first, phase, boundary):
    launch.names = (
        'cell_off',
        'starts',
        'targets',
        'dirs',
        'weights',
        'edge_off',
        'edges',
        'lower',
        'end_j',
        'end_f',
        'end_col',
        'hits',
    )
    launch.args = [
        np.int32(v)
        for v in (
            launch.ncell,
            len(s.geo.h),
            len(s.geo.chi),
            bundle.nq,
            len(launch.c['sun_h']),
            launch.c['sun_col'].shape[1],
            int(first),
            phase,
            int(boundary is not None),
        )
    ]
    launch.args += [bundle.arrays[name] for name in launch.names]
    if 'density_optics' in launch.c:
        launch.args[9 + launch.names.index('end_col')] = kernel.density_end_columns(bundle, launch.c)
    launch.args += [
        launch.dc[name]
        for name in (
            'u',
            'wu',
            'aa',
            'bb',
            'dd',
            'h',
            'chi',
            'hr',
            'ha',
            'ext',
            'asca',
            'sun_h',
            'sun_col',
            'gm',
            'goff',
            'gdir',
            'gval',
        )
    ]
    launch.goff_index = len(launch.args) - 3
    launch.ground_sh_l = kernel.ground_sh_l if first and phase in (0, 2) and 'ground_sh' in launch.dc else 0
    if launch.ground_sh_l:
        # Only the SH specialization interprets gval as its synthesized
        # convolution table. Without it the launch uses the
        # original binary, argument arrays, and shared-memory size.
        launch.args[-1] = launch.dc['ground_sh']
    launch.args += [
        launch.c['height'],
        launch.c['g'],
        launch.c['mix'],
        F(launch.c['albedo']),
        launch.src,
        launch.bd,
        launch.moments,
        launch.flux,
        launch.query,
    ]
    launch.sinv_index = len(launch.args) if launch.ks else None
    if launch.ks:
        launch.args += kernel.ksplit_pointer_args(launch.kb)
    launch.sh_index = len(launch.args)  # the SH sidecar appended below (VPROD reads the same combined plane)
    with stage('sh_aux_build', device=True), vram_trace.item('sh.sidecar'):
        cfg = kernel.config(launch.W)
        layout = FastTransport.select(kernel.L, cfg, first, phase, split=launch.ks)
        if 'strato_ext' in launch.c and cfg.strato:
            if launch.src.shape != (len(s.geo.h) * len(s.geo.chi), 10 + 2 * kernel.K, launch.W):
                raise ValueError(f'E_STRATO_SH_SOURCE: expected [nh*nc,{10 + 2 * kernel.K},W]')
            if cfg.c3_fw15 and not first:
                if layout.role == 'warm':
                    launch.args.append(combined_aux_waves(launch.src, launch.c, layout, launch.xp, skip_thin=True))
                else:
                    launch.args.append(combined_aux(launch.src, launch.c, kernel.L, launch.xp, skip_thin=True))
            else:
                from .strato_sh_aux import strato_sh_aux

                launch.args.append(strato_sh_aux(launch.src, launch.xp, pair=True, L=kernel.L))
        else:
            if layout.role == 'warm':
                launch.args.append(sh_aux_waves(launch.src, layout, launch.xp))
            else:
                launch.args.append(sh_aux_pair(launch.src, launch.xp, kernel.L))
    if 'strato_ext' in launch.c:
        # No optical sidecar is persisted in legacy/common endpoint files.
        launch.endst = kernel.strato_end_columns(bundle, launch.c)
        launch.profile = launch.c['strato_profile']
        launch.args += [
            launch.dc[name] for name in ('strato_ext', 'strato_sca', 'strato_chi', 'strato_density', 'strato_sun_col')
        ]
        launch.args += [
            launch.endst,
            F(launch.profile['center']),
            F(launch.profile['sigma']),
            F(launch.profile['norm']),
        ]
    kernel.check_upload(launch.c, launch.dc, 'launch')  # N and upload identity (E_PROFILE_N_MISMATCH / E_PROFILE_STALE)
    launch.args += [
        launch.dc.get('profile_table', np.uint64(0)),
        np.int32(launch.c.get('density_optics', {}).get('mask', 0)),
        F(launch.c.get('density_optics', {}).get('air_surface', 1.0)),
        launch.xp.empty_like(launch.moments),  # reusable per-chunk Neumaier corrections
        bundle.arrays['ray_geometry'],
    ]


def _use_fast_vprod(kernel, first, phase, W, ks):
    layout = FastTransport.select(kernel.L, kernel.config(W), first, phase, split=ks)
    return kernel.backend == 'cupy' and phase in (0, 1) and layout.role == 'warm'


def _budget_seconds(profile):
    return Kernel.PREDICT_CAP * (profile['launch_budget_seconds'] or 30.0)


def _work_seconds(profile):
    return min(profile['launch_target_seconds'] or float('inf'), _budget_seconds(profile))


TABLE_WORK_RATE = 1 << 29


def _hg_fits(layout, counts, free, seconds):
    if not len(counts) or not layout.hg:
        return False
    owners = int(np.count_nonzero(counts))
    nodes = layout.hg_nodes
    peak = int(counts.max())
    return (owners > 0 and peak <= 512 and peak * nodes**2 * layout.W <= TABLE_WORK_RATE * seconds
            and owners * (nodes**2 * layout.stride * 4 + 4) + 8 * len(counts) + (256 << 20) <= free)


def _transport_geometry(kernel, bundle):
    meta = getattr(bundle, '_fast_vprod_meta', None)
    if meta is None:
        co = np.asarray(kernel.host(bundle.arrays['cell_off']), 'i8')
        edges = np.asarray(kernel.host(bundle.arrays['edge_off'][co]), 'i8')
        meta = bundle._fast_vprod_meta = co, (edges - co) * int(bundle.nq)
    return meta


def _transport_limits(kernel, launch, bundle, angular, resident):
    profile = kernel.device()[1]
    return vram_budget.transport_chunks(
        *_transport_geometry(kernel, bundle), len(launch.c['h']) * len(launch.c['chi']), kernel.K, launch.W,
        not angular, launch.flux_only, launch.step if launch.mode == 'fixed' else None,
        _work_seconds(profile), kernel.ksplit_memory(), resident, budget=_budget_seconds(profile),
    )


def _live_sh_cells(c, strato, nh, nc, W):
    cells = nh * nc
    if W == 1 or nh > 128 or 'ha' not in c or 'hr' not in c:
        return np.arange(cells, dtype='i4')
    cutoff = np.float32(1e-7) * np.asarray(c['hr'], dtype='f4')
    live = np.asarray(c['ha'], dtype='f4') > cutoff
    if strato and 'strato_density' in c:
        live = live | (np.asarray(c['strato_density'], dtype='f4') > cutoff)
    return np.flatnonzero(np.repeat(live, nc)).astype('i4')


def _prepare_fast_transport(kernel, launch, bundle, phase, allow_table):
    nh, nc = len(launch.c['h']), len(launch.c['chi'])
    cells = nh * nc
    _, pcum = _transport_geometry(kernel, bundle)
    profile = kernel.device()[1]
    cfg = kernel.config(launch.W)
    layout = FastTransport.select(kernel.L, cfg, False, phase)
    nodes = layout.sh_nodes
    owners = _live_sh_cells(launch.c, cfg.strato, nh, nc, launch.W)
    n_owners = len(owners)
    src_rm_bytes = launch.src.shape[0] * 4 * layout.stride * 4
    table_bytes = (
        4 * nodes * (n_owners * nodes * layout.stride + kernel.K + kernel.L + 1)
        + 8 * cells + 4 * n_owners
    )
    for angular in (True, False):
        zero = False
        resident = 0
        if angular:
            if not allow_table or kernel.deterministic or launch.nr < cells:
                continue
            # Test the smallest workspace before launching even the zero-source reduction.
            limits = _transport_limits(kernel, launch, bundle, True, 0)
            if limits is None:
                continue
            zero = kernel.source_is_zero(launch.src)
            if not zero:
                # Segments along one ray repeat its direction. With fewer rays per
                # origin cell than bins on one angular axis, contract the sampled
                # directions exactly instead of interpolating the whole sphere.
                if launch.nr < (len(pcum) - 1) * layout.bins:
                    continue
                # A table covers every live cell, including cells the bundle never visits.
                dots = nodes * (kernel.K + nodes * (layout.L + 1))
                if (n_owners * dots > 4 * int(pcum[-1] - pcum[0]) * kernel.K
                        or dots * layout.stride > TABLE_WORK_RATE * _work_seconds(profile)):
                    continue
                resident = table_bytes
        if not zero:
            # Both layouts retain the packed source across all chunks.
            resident += src_rm_bytes
            limits = _transport_limits(kernel, launch, bundle, angular, resident)
        if limits is not None:
            launch.fast_resident_bytes = resident
            launch.live_cells = owners
            launch.fast_layout = replace(layout, role='angular' if angular else 'direct')
            launch.fast_chunks, launch.fast_work = limits
            launch.angular, launch.zero_volume = angular, zero
            return True
    return False


def _prepare_launch_order(kernel, launch, bundle, first, phase, allow_fast=True, allow_table=True):
    launch.mode, launch.step = kernel.launch_mode(launch.ncell)
    launch.lpt_index = None
    launch.split_count = None
    launch.flux_only = phase == 1
    launch.fast_vprod = False
    if allow_fast and _use_fast_vprod(kernel, first, phase, launch.W, launch.ks):
        try:
            launch.fast_vprod = _prepare_fast_transport(kernel, launch, bundle, phase, allow_table)
        except Exception as exc:
            if not _transport_resource_error(exc):
                raise
            warnings.warn(f'Transport preflight allocation failed: {exc}', RuntimeWarning)
    if launch.fast_vprod:
        launch.section_buffer = None
        return
    if not first and not (kernel.config().split_warm and phase == 0):
        # R4 LPT_ORDER: per-launch cell order by descending segment count (block work proxy).
        cfg = kernel.config(launch.W)
        S = 2 if (cfg.c3_fw15 or (launch.W == 15 and not cfg.instrumented)) else 1

        xp = launch.xp

        def lpt_order(c0, c1):
            co = bundle.arrays['cell_off'][c0 : c1 + 1]
            eo = bundle.arrays['edge_off']
            work = xp.diff(eo[co]) - xp.diff(co)
            if S > 1:  # units = cell*S+part, parts of a cell adjacent, cells by descending work
                o = xp.argsort(-work).astype('i4')
                return xp.ascontiguousarray((o[:, None] * S + xp.arange(S, dtype='i4')[None, :]).reshape(-1))
            return xp.ascontiguousarray(xp.argsort(-work).astype('i4'))

        launch.lpt_order = lpt_order
        launch.lpt_index = len(launch.args)
        launch.args.append(lpt_order(0, launch.ncell))
        if S > 1:
            with vram_trace.item('moments.split'):
                launch.args.append(
                    launch.xp.empty(
                        max(1, 2 * (S - 1) * launch.ncell * (4 + kernel.K) * launch.W) if phase == 0 else 1, 'f4'
                    )
                )
            launch.args.append(launch.xp.empty(2 * S * launch.ncell * launch.W, 'f4'))
            launch.split_count = launch.xp.zeros(launch.ncell, 'i4')
            launch.args.append(launch.split_count)
    launch.section_buffer = None
    if launch.section_dir is not None:
        launch.section_buffer = launch.xp.zeros(
            (
                launch.ncell,
                launch.W // kernel.tile_width(launch.W, first, phase),
                kernel.bs(first) // 32,
                len(SECTION_CLOCK_NAMES),
            ),
            dtype='uint64',
        )
        launch.args.append(launch.section_buffer)


def _select_transport_kernels(kernel, launch, bundle, first, phase, boundary, _ksplit):
    if launch.ks and launch.mode == 'single':
        launch.mode, launch.step = 'fixed', launch.ncell  # KSPLIT always goes through the chunk loop
    launch.warm_work = None
    launch.fast_first = False
    launch.ray_sync = None
    launch.ray_mode = False
    if phase == 0 and ((kernel.config().split_first and first) or (kernel.config().split_warm and not first)):
        # FL_RAYSPLIT private scratch: per-ray rad rows and [ray queue, per-cell done] counters (zeroed).
        # Cell-subset launches (split profile): the kernel reads rays cell_off[0]..cell_off[cells] and indexes ray_rads
        # and the counters chunk-locally, so the counters are re-zeroed before every chunk.
        # WARM_RAYSPLIT is not ported to cell-subset launches.
        if launch.mode != 'single' and (not first):
            raise ValueError('E_FL_RAYSPLIT_LAUNCH_CELLS: WARM_RAYSPLIT needs one launch per cell grid')
        # Split launches and, with SOSFUSE2_FL_PROJECT=1, single launches project first light after processing the
        # ray ranges.
        launch.ray_mode = launch.mode != 'single' or os.environ.get('SOSFUSE2_FL_PROJECT', '').strip() == '1'
        launch.ray_sync = launch.xp.zeros(
            launch.ncell + 2, 'i4'
        )  # [queue head, cell counters, ray-range end (LAUNCH_RAY_LIMIT)]
        with vram_trace.item('transport.ray_radiance'):
            launch.ray_rads = launch.xp.empty((max(launch.nr, 1), 4 * launch.W), 'f4')
        launch.args += [launch.ray_rads, launch.ray_sync]
    launch.rs = bool(kernel.device()[1].get('requires_split'))
    if first and phase == 0 and (not launch.ray_mode):
        err = first_light_split_error(kernel.flags, 'strato_ext' in launch.c, kernel.device()[0], kernel.device()[1])
        if err:
            raise RuntimeError(err)
    launch.variant = _ksplit or ('prod' if launch.ks else None)
    if launch.fast_vprod:
        launch.fv_kernels = kernel.fast_vprod_kernels(launch.fast_layout, boundary is not None)
        # Loading a module can change free memory. Recheck the selected layout
        # with the compiled per-thread local footprint before any fast allocation.
        limits = _transport_limits(kernel, launch, bundle, launch.angular,
                                   launch.fast_resident_bytes + launch.fv_kernels['local_reserve_bytes'])
        if limits is None:
            raise _FastResourceError('Compiled transport workspace exceeds available memory')
        launch.fast_chunks, launch.fast_work = limits
        launch.raws = []
        launch.raw = None
        return
    launch.raws = (
        [
            ('_prod', kernel.raw_kernel(launch.W, first, phase, boundary is not None, variant='prod')),
            ('_slim', kernel.raw_kernel(launch.W, first, phase, boundary is not None, variant='slim')),
        ]
        if launch.ks and not _ksplit
        else [
            (
                '',
                kernel.raw_kernel(
                    launch.W,
                    first,
                    phase,
                    boundary is not None,
                    ray_limit=launch.ray_mode,
                    variant=_ksplit,
                    ground_sh_l=launch.ground_sh_l,
                ),
            )
        ]
    )
    launch.raw = launch.raws[0][1]
    cfg = kernel.config(launch.W)
    layout = FastTransport.select(kernel.L, cfg, first, phase, boundary is not None, launch.ray_mode)
    launch.fast_first = layout.role == 'first'
    if layout.role == 'warm' and kernel.backend == 'cupy' and not launch.ks:
        co, pc = _transport_geometry(kernel, bundle)
        launch.warm_work = (pc - pc[0]).astype('f8') * (4 * kernel.K) + (co - co[0]) * float(kernel.K)
        if launch.mode == 'single':
            launch.mode, launch.step = 'fixed', launch.ncell
    if layout.hg and not kernel.deterministic and abs(launch.c['g']) <= 0.8:
        try:
            kernel.prepare_hg_table(launch, layout)
        except Exception as exc:
            if not _transport_resource_error(exc):
                raise
            launch.hg_pages = None
            warnings.warn(f'HG table fallback ({type(exc).__name__}): {exc}', RuntimeWarning)


def _bounded_end(work, a, end, cost, rate, profile, *, kernel=None, kind=None):
    """Limit even a cold probe by work. A single indivisible unit must fit the hard budget."""
    unit_rate = rate / max(1, cost)
    target = _work_seconds(profile)
    room = unit_rate * target
    span = work[end] - work[a]
    if span <= room:
        return end
    hard = unit_rate * _budget_seconds(profile)
    if work[a + 1] - work[a] > hard:
        raise RuntimeError(f'E_LAUNCH_BUDGET: work of unit {a} exceeds the pre-launch cap')
    # A measured fast kernel may use up to the static hard work limit only
    # when its conservative CUDA-event prediction still meets the split target.
    if kernel is not None:
        prediction = kernel.predict_seconds(kind, span)
        if prediction is not None and prediction <= target:
            room = hard
    if span <= room:
        return end
    return max(a + 1, int(np.searchsorted(work, work[a] + room, side='right')) - 1)


def _launch_ray_ranges(kernel, launch, bundle):
    launch.cum = kernel.ray_work(bundle)
    nrays = len(launch.cum) - 1
    launch.kind += '_rays'
    # Adaptive launches start with half the estimated rays in the initial cells.
    initial = max(
        1,
        (launch.step if launch.mode == 'fixed' else kernel.device()[1]['launch_cells_initial'] or 1)
        * nrays
        // max(launch.ncell, 1)
        // (1 if launch.mode == 'fixed' else 2),
    )
    launch.report = dict(chunks=0, gpu_seconds=0.0, max_seconds=0.0, max_cells=launch.ncell, max_rays=0)
    profile = kernel.device()[1]
    ground_cost = None
    if launch.fast_first:
        owners = np.diff(launch.c['goff'])
        samples = 36 if getattr(launch, 'hg_pages', None) else 2 * int(owners.max(initial=0))
        ground_cost = bundle.nq * (1 + 4 * samples)
        if launch.mode == 'fixed':
            cell_off = kernel.host(bundle.arrays['cell_off'])
    q0 = 0
    while q0 < nrays:
        q1 = (
            nrays
            if launch.mode == 'single'
            else min(nrays, q0 + initial)
            if (launch.mode == 'fixed' and not launch.rs)
            else kernel.next_chunk_end(
                launch.kind, q0, nrays, initial if launch.mode == 'fixed' else None, launch.cum, initial
            )
        )
        if ground_cost is not None:
            if launch.mode == 'fixed':
                # A ray range starting inside a cell still touches at most step cells.
                first_cell = np.searchsorted(cell_off, cell_off[0] + q0, side='right') - 1
                q1 = min(q1, int(cell_off[min(launch.ncell, first_cell + launch.step)] - cell_off[0]))
            q1 = _bounded_end(launch.cum, q0, q1, ground_cost, TABLE_WORK_RATE, profile,
                              kernel=kernel, kind=launch.kind)
        launch.ray_sync[0:1].fill(q0)
        launch.ray_sync[launch.ncell + 1 : launch.ncell + 2].fill(q1)
        seconds = kernel.timed_launch(
            launch.kind,
            launch.raw,
            (launch.ncell, launch.grid_y),
            launch.block,
            tuple(launch.args),
            launch.shared,
            q1 - q0,
            launch.cum,
            q0,
            q1,
        )
        launch.report['chunks'] += 1
        launch.report['gpu_seconds'] += seconds
        launch.report['max_seconds'] = max(launch.report['max_seconds'], seconds)
        launch.report['max_rays'] = max(launch.report['max_rays'], q1 - q0)
        q0 = q1


def _project_ray_ranges(kernel, launch, bundle, boundary, first):
    # With LAUNCH_RAY_LIMIT the ray launches only publish ray_rads; project the cells now (fl_project,
    # one block per cell, compensated FP32 ray batches).
    proj = kernel.fl_project_kernel(launch.W, boundary is not None, ground_sh_l=launch.ground_sh_l, first=first)
    pkind = launch.kind[:-5] + '_project'
    launch.co_h = np.asarray(kernel.host(bundle.arrays['cell_off']), 'i8')
    launch.pcum = (launch.co_h - launch.co_h[0]).astype('f8')
    ci = 9 + launch.names.index('cell_off')
    ti = 9 + launch.names.index('targets')
    launch.fixed_args = (
        launch.args[9 + launch.names.index('dirs')],
        launch.args[9 + launch.names.index('weights')],
        launch.dc['aa'],
        launch.dc['bb'],
        launch.dc['dd'],
        launch.ray_rads,
        F(launch.c['albedo']),
    )
    c0 = 0
    while c0 < launch.ncell:
        c1 = (
            launch.ncell
            if launch.mode == 'single'
            else min(launch.ncell, c0 + launch.step)
            if (launch.mode == 'fixed' and not launch.rs)
            else kernel.next_chunk_end(
                pkind, c0, launch.ncell, launch.step if launch.mode == 'fixed' else None, launch.pcum
            )
        )
        if launch.fast_first:
            c1 = _bounded_end(launch.pcum, c0, c1, kernel.K, vram_budget.FAST_WORK_RATE, kernel.device()[1],
                              kernel=kernel, kind=pkind)
        seconds = kernel.timed_launch(
            pkind,
            proj,
            (c1 - c0,),
            (256,),
            (np.int32(c1 - c0), np.int64(launch.co_h[0]), launch.args[ci][c0:], launch.args[ti][c0:])
            + launch.fixed_args
            + (launch.moments[c0:c1], launch.flux[c0:c1]),
            0,
            c1 - c0,
            launch.pcum,
            c0,
            c1,
        )
        launch.report['chunks'] += 1
        launch.report['gpu_seconds'] += seconds
        launch.report['max_seconds'] = max(launch.report['max_seconds'], seconds)
        c0 = c1


def _launch_cell_ranges(kernel, launch, bundle, s, phase, boundary, _ksplit):
    launch.cum = kernel.cell_work(bundle) if (launch.mode == 'adaptive' or launch.rs) else None
    cell_index = {n: 9 + launch.names.index(n) for n in ('cell_off', 'starts', 'targets')}
    mi = next(i for i, a in enumerate(launch.args) if a is launch.moments)
    fi = next(i for i, a in enumerate(launch.args) if a is launch.flux)
    launch.report = dict(chunks=0, gpu_seconds=0.0, max_seconds=0.0, max_cells=0)
    if launch.warm_work is not None:
        launch.cum = launch.warm_work
    c0 = launch.ncell if (launch.ks and not _ksplit and kernel.ksplit_mode in (2, 3)) else 0
    if c0:

        def cell_part(a, b):
            part = list(launch.args)
            part[0] = np.int32(b - a)
            part[launch.sinv_index : launch.sinv_index + 3] = kernel.ksplit_pointer_args(launch.kb)
            for n, i in cell_index.items():
                part[i] = launch.args[i][a:]
            if phase == 0:
                part[mi] = launch.moments[a:b]
            part[fi] = launch.flux[a:b]
            if launch.lpt_index is not None:
                part[launch.lpt_index] = launch.lpt_order(a, b)
            if launch.split_count is not None:
                launch.split_count.fill(0)
            return tuple(part)

        launch.ctx = dict(
            args=launch.args,
            sinv_index=launch.sinv_index,
            lpt_index=launch.lpt_index,
            cell_index=cell_index,
            mi=mi,
            fi=fi,
            moments=launch.moments,
            flux=launch.flux,
            albedo=launch.c['albedo'],
            ray_cum=kernel.ray_work(bundle) if launch.cum is not None else None,
        )
        kernel.vprod_launch(
            launch.kb,
            bundle,
            s,
            launch.dc,
            launch.args[launch.sh_index],
            launch.W,
            phase,
            boundary is not None,
            launch.kind,
            launch.raws,
            cell_part,
            ((launch.nsplit, launch.grid_y), launch.block, launch.shared),
            launch.step,
            launch.cum,
            launch.report,
            launch.ctx,
        )
    while c0 < launch.ncell:
        c1 = kernel.next_chunk_end(launch.kind + launch.raws[0][0], c0, launch.ncell, launch.step, launch.cum)
        if launch.warm_work is not None:
            c1 = _bounded_end(launch.cum, c0, c1, 1, vram_budget.FAST_WORK_RATE, kernel.device()[1],
                              kernel=kernel, kind=launch.kind)
        if launch.ks and not _ksplit:
            c1 = kernel.ksplit_chunk(launch.kb, c0, c1)
        launch.part = list(launch.args)
        launch.part[0] = np.int32(c1 - c0)
        if launch.ks:
            launch.part[launch.sinv_index : launch.sinv_index + 3] = kernel.ksplit_pointer_args(launch.kb)
        for n, i in cell_index.items():
            launch.part[i] = launch.args[i][c0:]
        if phase == 0:
            launch.part[mi] = launch.moments[c0:c1]
        launch.part[fi] = launch.flux[c0:c1]
        if launch.section_buffer is not None:
            launch.part[-1] = launch.section_buffer[c0:c1]
        if launch.lpt_index is not None:
            launch.part[launch.lpt_index] = launch.lpt_order(c0, c1)
        if launch.ray_sync is not None and c0 > 0:
            launch.ray_sync.fill(0)
        if launch.split_count is not None and c0 > 0:
            launch.split_count.fill(0)
        for suffix, r in launch.raws:  # KSPLIT: producer then consumer on the same chunk (one stream, in order)
            seconds = kernel.timed_launch(
                launch.kind + suffix,
                r,
                ((c1 - c0) * launch.nsplit, launch.grid_y),
                launch.block,
                tuple(launch.part),
                launch.shared,
                c1 - c0,
                launch.cum,
                c0,
                c1,
            )
            launch.report['gpu_seconds'] += seconds
            launch.report['max_seconds'] = max(launch.report['max_seconds'], seconds)
        launch.report['chunks'] += 1
        launch.report['max_cells'] = max(launch.report['max_cells'], c1 - c0)
        c0 = c1


def _dispatch_transport(kernel, launch, bundle, s, first, phase, boundary, _ksplit):
    name = (
        'launch_first'
        if first
        else ('launch_query' if phase == 2 else 'launch_ground' if phase == 1 else 'launch_warm')
    )
    with stage(name, device=True):
        # Cell-subset launches (split profile thermal/watchdog cap, device_profile.py): blocks are
        # independent, so only the cell-indexed pointers move: cell_off/starts/targets (inputs) and
        # moments/flux (outputs); per-chunk counters (FL_RAYSPLIT ray_sync, CELL_SPLIT split_count) are
        # re-zeroed. Numerically equivalent to one launch within the reference tolerance. 'adaptive' sizes each chunk from the last
        # measured seconds per unit of work of the same kernel kind; 'fixed' is SOSFUSE2_LAUNCH_CELLS.
        cfg = kernel.config(launch.W)
        launch.nsplit = (
            (2 if (cfg.c3_fw15 or (launch.W == 15 and not cfg.instrumented)) else 1)
            if launch.lpt_index is not None
            else 1
        )
        launch.grid_y = launch.W // kernel.tile_width(launch.W, first, phase)
        launch.block = (kernel.bs(first, launch.variant),)
        launch.shared = kernel.shared_bytes(launch.W, phase, first=first, variant=launch.variant,
                                            boundary=boundary is not None, ray_limit=launch.ray_mode)
        launch.kind = f'first{int(first)}_phase{int(phase)}_W{launch.W}' + (
            '_record' if _ksplit else '_split' if launch.ks else ''
        )
        launch.report = None
        if launch.fast_vprod:
            kernel.fast_vprod_launch(launch, bundle, s, boundary is not None)
        elif launch.mode == 'single' and not launch.ray_mode:
            seconds = kernel.timed_launch(
                launch.kind,
                launch.raw,
                (launch.ncell * launch.nsplit, launch.grid_y),
                launch.block,
                tuple(launch.args),
                launch.shared,
                launch.ncell,
            )
            launch.report = dict(chunks=1, gpu_seconds=seconds, max_seconds=seconds, max_cells=launch.ncell)
        elif launch.ray_mode:
            _launch_ray_ranges(kernel, launch, bundle)
            _project_ray_ranges(kernel, launch, bundle, boundary, first)
        else:
            _launch_cell_ranges(kernel, launch, bundle, s, phase, boundary, _ksplit)
        if launch.report is not None:
            launch.report.update(
                kind=launch.kind,
                mode=launch.mode,
                cells=launch.ncell,
                profile=kernel.launch_log['profile'],
                budget_seconds=kernel.launch_log['budget_seconds'],
            )
            print('LAUNCH_REPORT ' + json.dumps(launch.report, allow_nan=False), flush=True)


def _record_transport_clocks(kernel, launch, first, phase, timed):
    if timed or launch.section_buffer is not None:
        launch.end.record()
        launch.end.synchronize()
        cuda_seconds = float(launch.xp.cuda.get_elapsed_time(launch.begin, launch.end) / 1000)
        if timed:
            kernel.cuda_events.append(cuda_seconds)
    if launch.section_buffer is not None:
        launch.counters = kernel.host(launch.section_buffer)
        launch.report = section_clock_summary(launch.counters)
        path = launch.section_dir / f'clock_{os.getpid()}_{time.time_ns()}_{kernel.metrics["launches"]}.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, cycles=launch.counters)
        launch.report.update(
            schema='sosfuse2-section-clocks-v1',
            flags=list(kernel.flags),
            gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
            pid=os.getpid(),
            launch=kernel.metrics['launches'],
            first=bool(first),
            phase=int(phase),
            wavelengths=launch.W,
            block_size=kernel.block_size,
            grid=[launch.ncell, launch.W // kernel.tile_width(launch.W, first, phase)],
            launch_cells=kernel.launch_cells(),
            cuda_event_seconds=cuda_seconds,
            raw_path=str(path),
        )
        print('SECTION_CLOCKS ' + json.dumps(launch.report, allow_nan=False), flush=True)


class Kernel(KsplitMixin):
    def __init__(
        self, threads=4, height=1.2, g=0.8, backend='numpy', block_size=128, flags=None, L=32, *, profile_step_km=0.5, domain=None
    ):
        if backend not in ('numpy', 'cupy'):
            raise ValueError('E_BACKEND: select numpy or cupy explicitly')
        if block_size not in (32, 64, 128, 256):
            raise ValueError('E_BLOCK_SIZE: select 32,64,128,256')
        self.deterministic = deterministic()
        self.L = L
        self.K = count(L)
        self.height, self.g, self.backend, self.block_size = height, g, backend, block_size
        # PROFILE boundary and spacing are compile constants.
        from ..atmosphere.density import profile_n

        profile_n(profile_step_km)
        self.domain = domain or top.Domain(step_km=float(profile_step_km))
        self.profile_n = self.domain.n
        self.profile_step_km = self.domain.step_km
        self.flags = resolve_config(default_flags() if flags is None else flags).flags
        from .ground_sh import configured_order

        self.ground_sh_l = configured_order()
        # Production default:
        # unset means KSPLIT=3 (VPROD + warm ray consumer + projection) where it applies (CuPy, STRATO/FW15, L<=16)
        # and the unsplit kernel elsewhere; SOSFUSE2_KSPLIT=0 forces the unsplit kernel, and an
        # explicit 1-3 still fails loudly on an unsupported configuration.
        ks_text, self._ksplit_zero_reason = _ksplit_request(backend, self.config(15), L)
        if ks_text not in ('0', '1', '2', '3'):
            raise ValueError(f'E_KSPLIT_FLAGS: SOSFUSE2_KSPLIT must be 0, 1, 2 or 3 (got {ks_text!r})')
        self.ksplit = ks_text in ('1', '2', '3')
        self.ksplit_mode = (
            int(ks_text) if self.ksplit else 0
        )  # 2: VPROD instead of PROD; 3: + warm ray consumer and projection
        if self.ksplit_mode >= 2 and self.L > 16:
            raise ValueError(
                'E_KSPLIT_FLAGS: SOSFUSE2_KSPLIT=2 or 3 (VPROD) is compiled for SH order L<=16 only (kernel.cu sos_vprod)'
            )
        self.ksplit_verify = os.environ.get('SOSFUSE2_KSPLIT_VERIFY', '').strip() == '1'
        if self.ksplit_verify and self.ksplit_mode not in (2, 3) and os.environ.get('SOSFUSE2_KSPLIT', '').strip():
            raise ValueError(
                'E_KSPLIT_VERIFY: SOSFUSE2_KSPLIT_VERIFY=1 compares PROD and VPROD and needs SOSFUSE2_KSPLIT=2 or 3'
            )
        self.xp = np
        if backend == 'cupy':
            import cupy

            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('E_CUDA_DEVICE: no CUDA device')
            self.xp = cupy
            trace = vram_trace.current.get()
            if trace is not None:
                trace.bind(cupy)
        self.raw = {}
        self.constants = {}
        self.device_constants = {}
        self.ground_sh_tables = {}
        self.metrics = dict(samples=0, angular_pairs=0, launches=0, query_launches=0)
        self.attributes = {}
        self.cuda_events = []
        self.pool_peaks = dict(used_bytes_sampled=0, reserved_bytes_sampled=0)
        TIMERS.bind(self.xp)

    def config(self, W=None):
        return resolve_config(self.flags, W=W if W is not None else getattr(self, '_W', 15))

    def memory(self):
        if self.xp is not np:
            pool = self.xp.get_default_memory_pool()
            self.pool_peaks['used_bytes_sampled'] = max(self.pool_peaks['used_bytes_sampled'], int(pool.used_bytes()))
            self.pool_peaks['reserved_bytes_sampled'] = max(
                self.pool_peaks['reserved_bytes_sampled'], int(pool.total_bytes())
            )

    def sync(self):
        if self.xp is not np:
            self.xp.cuda.get_current_stream().synchronize()

    def host(self, array):
        return np.asarray(array) if self.xp is np else self.xp.asnumpy(array)

    def basis(self, dirs, L):
        count(L)
        if L != self.L:
            raise ValueError('E_SH_L_MISMATCH: basis and kernel degree differ')
        y = numpy_reference.basis(np.asarray(dirs)[:, 2], L)
        st = np.sqrt(np.maximum(0.0, 1 - np.asarray(dirs)[:, 2] ** 2))
        cp = np.where(st > 1e-12, dirs[:, 0] / np.maximum(st, 1e-30), 1.0).astype('f4')
        cm = numpy_reference.cosine(cp)
        for l in range(L + 1):
            for m in range(l + 1):
                y[:, l * (l + 1) // 2 + m] *= cm[:, m]
        return y.astype('f8')

    def _configure_wavelengths(self, s, W):
        cfg = self.config(W)
        if self.ground_sh_l and not cfg.ground_sh_supported:
            raise ValueError('E_GROUND_SH_FLAGS: requires fused strato ground, split first light and HG')
        ks, self._ksplit_zero_reason = _ksplit_request(self.backend, cfg, self.L)
        if ks not in ('0', '1', '2', '3') or (ks != '0' and not cfg.ksplit_supported):
            raise ValueError('E_KSPLIT_FLAGS: needs STRATO, FW15 and an uninstrumented transport role')
        decision = getattr(s, 'ksplit_decision', None)
        mode = decision['effective'] if decision is not None else int(ks)
        memory_fallback = (
            decision is not None and decision.get('requested') in (2, 3) and mode == 0
            and decision.get('reason') == 'state_minimum_exceeds_solo_budget'
        )
        if memory_fallback:
            self._ksplit_zero_reason = decision['reason']
        if self.ksplit_verify and mode not in (2, 3) and not memory_fallback:
            if self.xp is not np and W == 15:
                raise ValueError('E_KSPLIT_VERIFY: requires KSPLIT=2 or 3')
            warnings.warn(
                'W_KSPLIT_VERIFY_INACTIVE: SOSFUSE2_KSPLIT_VERIFY=1 does not apply to this kernel (NumPy backend or FW1)',
                RuntimeWarning,
                stacklevel=3,
            )
        self._W = W
        self.ksplit_mode = mode
        self.ksplit = bool(mode)

    def setup(self, s, nq):
        key = (id(s), nq)
        c = self.constants.get(key)
        if c is None:
            with stage('kernel_setup'):
                from ..atmosphere.stratosphere import attach
                from ..atmosphere.density import attach as attach_density

                density = attach_density(s, s.ctx)
                if density is not None:
                    self.check_profile(density, 'setup')
                optics = attach(s, s.ctx)
                if optics is not None:
                    self.flags = tuple(sorted(set(self.flags) | {'STRATO'}))
                elif 'STRATO' in self.flags:
                    raise ValueError('E_STRATO_FLAG_ZERO: STRATO is automatic for nonzero tau')
                from ..atmosphere.tropo_mie import attach as attach_tropo

                tropo = attach_tropo(s)
                if self.ground_sh_l and (tropo is not None or optics is None):
                    raise ValueError('E_GROUND_SH_PHASE: requires tropospheric HG and enabled stratosphere')
                if tropo is not None:
                    self.flags = tuple(sorted(set(self.flags) | {'TROPO_MIE'}))
                elif 'TROPO_MIE' in self.flags:
                    raise ValueError('E_TROPO_FLAG_DEFAULT: TROPO_MIE is automatic for an enabled troposphere_phase')
                from ..atmosphere.tropo_profile import attach as attach_profile

                aer = attach_profile(s)
                if aer is not None:
                    self.flags = tuple(sorted(set(self.flags) | {'AEROSOL_2L'}))
                elif 'AEROSOL_2L' in self.flags:
                    raise ValueError(
                        'E_AEROSOL_2L_FLAG_DEFAULT: AEROSOL_2L is automatic for an enabled troposphere_profile'
                    )
                self._configure_wavelengths(s, s.W)
                u, wu = np.polynomial.legendre.leggauss(nq)
                u = (u + 1) / 2
                wu = wu / 2
                ng = int(max(np.diff(s.goff), default=0))
                gd = np.zeros((s.geo.ns, ng, 3), 'f4')
                gv = np.zeros((s.geo.ns, ng, s.W), 'f4')
                for cell in range(s.geo.ns):
                    a, b = s.goff[cell : cell + 2]
                    gd[cell, : b - a] = s.gdir[a:b]
                    gv[cell, : b - a] = s.gval[a:b]
                c = dict(
                    domain=self.domain,
                    L=self.L,
                    W=s.W,
                    height=F(self.height),
                    g=F(self.g),
                    mix=F((1 - s.delta) / (1 + s.delta / 2)),
                    albedo=float(s.fuse_state['parameters']['albedo']),
                    u=u,
                    wu=wu,
                    h=s.geo.h.astype('f4'),
                    chi=s.geo.chi.astype('f4'),
                    hr=s.hr.astype('f4'),
                    ha=s.ha.astype('f4'),
                    ext=s.ext.astype('f4'),
                    asca=s.asca.astype('f4'),
                    sun_h=s.geo.sun.h.astype('f4'),
                    sun_col=s.geo.sun.col.astype('f4'),
                    gm=s.gm.astype('f4'),
                    gd=gd,
                    gv=gv,
                    goff=s.goff,
                    gdir=s.gdir,
                    gval=s.gval,
                )
                if self.config().c3_fw15:
                    c.update(sh_combine=True, sh_combine_skip=True)
                if optics is not None:
                    c.update(
                        strato_ext=optics['ext'].astype('f4'),
                        strato_sca=optics['sca'].astype('f4'),
                        strato_chi=optics['chi'].astype('f4'),
                        strato_density=optics['source_density'].astype('f4'),
                        strato_sun_col=optics['sun_col'].astype('f4'),
                        strato_profile=optics['profile'],
                        strato_old=optics['old'],
                    )
                if density is not None:
                    c['density_optics'] = density
                if tropo is not None:
                    c.update(tropo_lut=np.ascontiguousarray(tropo['lut'], 'f4').ravel(), tropo_bins=int(tropo['bins']))
                if aer is not None:
                    c['aer2'] = np.ascontiguousarray(aer['kernel'], 'f4')
                self.constants[key] = c
        elif getattr(self, '_W', None) != c['W']:
            # Prepared constants can be the first setup on a new service kernel.
            self._configure_wavelengths(s, c['W'])
        # Every setup() call, cache hit or miss and installed constants alike (NumPy transport, geometry,
        # radiance_table and the persistent service all read prepared constants through here).
        if 'density_optics' in c:
            self.check_profile(c['density_optics'], 'setup')
        return c

    def tile_width(self, W, first=False, phase=0):
        return W

    def launch_cells(self):
        text = os.environ.get('SOSFUSE2_LAUNCH_CELLS', '').strip()
        if not text:
            return 0
        if not text.isdigit():
            raise ValueError('E_LAUNCH_CELLS: SOSFUSE2_LAUNCH_CELLS must be a non-negative integer')
        return int(text)

    def device(self):
        """(name, profile, reason) from backends/configs/devices.json, read once per Kernel (device_profile.py)."""
        if getattr(self, '_device', None) is None:
            from ..backends.device_profile import select, cuda_device_name

            self._device = select(cuda_device_name(self.xp) if self.xp is not np else None)
            self.launch_log = dict(
                profile=self._device[0],
                reason=self._device[2],
                budget_seconds=self._device[1]['launch_budget_seconds'],
                target_seconds=self._device[1]['launch_target_seconds'],
                launches=0,
                gpu_seconds=0.0,
                max_seconds=0.0,
                by_kind={},
            )
        return self._device

    def launch_mode(self, ncell):
        """('single', ncell) | ('fixed', cells) | ('adaptive', None). SOSFUSE2_LAUNCH_CELLS keeps the old fixed split."""
        name, p, _ = self.device()
        fixed = p['launch_cells_fixed']
        if p.get('requires_split') and ncell > 1:
            # split profile: never a whole-grid launch; SOSFUSE2_LAUNCH_CELLS is only an upper bound under the predictive cap.
            if fixed:
                return 'fixed', min(fixed, ncell)
            if p['launch_target_seconds']:
                return 'adaptive', None
            raise RuntimeError(
                f'E_LAUNCH_BUDGET_CONFIG: the {name} profile requires split launches; set '
                'SOSFUSE2_LAUNCH_TARGET_SECONDS > 0 (or unset it) or SOSFUSE2_LAUNCH_CELLS. Not launched.'
            )
        if fixed is not None:
            return ('fixed', fixed) if 0 < fixed < ncell else ('single', ncell)
        if p['launch_target_seconds']:
            return 'adaptive', None
        return 'single', ncell

    def cell_work(self, bundle):
        """Host cumulative work proxy over cells (edges of the cell's rays), cached on the Bundle."""
        cum = getattr(bundle, 'cell_work_cum', None)
        if cum is None:
            co = bundle.arrays['cell_off']
            eo = bundle.arrays['edge_off']
            e = np.asarray(self.host(eo[co]), 'i8')
            cum = np.r_[0, np.cumsum(np.maximum(np.diff(e), 1))].astype('f8')
            bundle.cell_work_cum = cum
        return cum

    def ray_work(self, bundle):
        """Host cumulative work proxy over the bundle's rays (edges + 1 per ray), cached on the Bundle."""
        cum = getattr(bundle, 'ray_work_cum', None)
        if cum is None:
            co = self.host(bundle.arrays['cell_off'])
            r0, r1 = int(co[0]), int(co[-1])
            e = np.asarray(self.host(bundle.arrays['edge_off'][r0 : r1 + 1]), 'i8')
            cum = (e - e[0]).astype('f8') + np.arange(r1 - r0 + 1, dtype='f8')
            bundle.ray_work_cum = cum
        return cum

    PREDICT_MARGIN = 1.3  # predicted launch seconds are inflated by this before any comparison
    PREDICT_CAP = 0.8  # of launch_budget_seconds: a chunk predicted above it is halved, never launched
    PROBE_DIVISOR = 4  # first chunk of a kernel kind = initial size / 4 (nothing measured yet)

    def predict_seconds(self, kind, work):
        """Conservative launch time for [work] from this kind's recent (work, seconds): an affine fit
        t = L + r*w (L = per-launch latency floor, r = seconds per unit of work),
        times PREDICT_MARGIN. With chunks of one size only, L is taken as 0.8 of the fastest launch."""
        obs = self.__dict__.get('_launch_obs', {}).get(kind)
        if not obs:
            return None
        w = np.array([o[0] for o in obs], 'f8')
        t = np.array([o[1] for o in obs], 'f8')
        fit = None
        if len(obs) >= 2 and w.max() > 1.5 * w.min():
            (L, r), *_ = np.linalg.lstsq(np.vstack([np.ones_like(w), w]).T, t, rcond=None)
            if r > 0 and L >= 0:
                fit = (L, max(r, float(np.max((t - L) / w))))  # slope never below any single observation's
        if fit is None:
            L = 0.8 * float(t.min())
            fit = (L, float(np.max((t - L) / w)))
        return self.PREDICT_MARGIN * (fit[0] + fit[1] * float(work))

    def next_chunk_end(self, kind, c0, n, step, cum, initial=None):
        """Fixed: c0+step. Adaptive (cells, or rays for ray-range launches): the largest chunk whose predicted time
        (predict_seconds) fits the profile target, at most 1.5x the previous chunk; the first chunk of a kind is a
        probe (initial/PROBE_DIVISOR). Before returning, a chunk predicted above PREDICT_CAP of the budget is halved
        down to one unit; if even one unit is predicted above it, E_LAUNCH_BUDGET is raised before launching
        (the earlier size floor could exceed the 5 s budget). Blocks/rays are independent, so
        the split never changes the numbers."""
        if (
            cum is None
        ):  # no launch budget (cuda profile): fixed step, or the whole rest when the caller has none (KSPLIT=3 projection)
            return n if step is None else min(n, c0 + step)
        _, p, _ = self.device()
        initial = max(1, int(initial or p['launch_cells_initial'] or 1))
        obs = self.__dict__.setdefault('_launch_obs', {}).get(kind)
        if not obs:
            c1 = min(n, c0 + max(1, initial // self.PROBE_DIVISOR), c0 + step if step else n)
        else:
            last = obs[-1][2]
            hi = min(
                n, c0 + max(int(1.5 * last), last + 1), c0 + step if step else n
            )  # a fixed step is an upper bound here
            target = p['launch_target_seconds'] or self.PREDICT_CAP * (p['launch_budget_seconds'] or 0) or float('inf')
            lo, best = c0 + 1, c0 + 1
            while lo <= hi:  # largest c1 in (c0, hi] whose prediction fits the target (monotone in work)
                mid = (lo + hi) // 2
                if self.predict_seconds(kind, cum[mid] - cum[c0]) <= target:
                    best, lo = mid, mid + 1
                else:
                    hi = mid - 1
            c1 = best
        budget = p['launch_budget_seconds']
        if budget and obs:
            cap = self.PREDICT_CAP * budget
            while c1 > c0 + 1 and self.predict_seconds(kind, cum[c1] - cum[c0]) > cap:
                c1 = c0 + (c1 - c0) // 2
            predicted = self.predict_seconds(kind, cum[c1] - cum[c0])
            if predicted > cap:
                raise RuntimeError(
                    f'E_LAUNCH_BUDGET: predicted {predicted:.3f}s > {cap:.3f}s for one unit of {kind} '
                    f'(unit {c0}); not launched'
                )
        return c1

    def timed_launch(self, kind, raw, grid, block, args, shared, units, cum=None, c0=0, c1=None):
        """One RawKernel launch timed by CUDA events; post-launch budget check (detects, never preempts)."""
        xp = self.xp
        _, p, _ = self.device()
        if not p['launch_budget_seconds'] and cum is None:
            # No budget and nothing to adapt (mac profile, host test doubles): launch exactly as before.
            raw(grid, block, args, shared_mem=shared)
            self.launch_log['launches'] += 1
            return 0.0
        begin, end = xp.cuda.Event(), xp.cuda.Event()
        begin.record()
        raw(grid, block, args, shared_mem=shared)
        end.record()
        end.synchronize()
        seconds = float(xp.cuda.get_elapsed_time(begin, end) / 1000.0)
        if not math.isfinite(seconds) or seconds < 0:
            raise RuntimeError('E_LAUNCH_EVENT: invalid CUDA event duration')
        log = self.launch_log
        k = log['by_kind'].setdefault(kind, dict(launches=0, gpu_seconds=0.0, max_seconds=0.0, max_units=0))
        k['launches'] += 1
        k['gpu_seconds'] += seconds
        k['max_seconds'] = max(k['max_seconds'], seconds)
        k['max_units'] = max(k['max_units'], int(units))
        log['launches'] += 1
        log['gpu_seconds'] += seconds
        log['max_seconds'] = max(log['max_seconds'], seconds)
        if cum is not None and c1 is not None and c1 > c0:
            work = max(float(cum[c1] - cum[c0]), 1.0)
            predicted = self.predict_seconds(kind, work)
            obs = self.__dict__.setdefault('_launch_obs', {}).setdefault(kind, [])
            obs.append((work, seconds, int(c1 - c0)))
            del obs[:-12]
            print(
                f'LAUNCH_CHUNK {kind} {c0}:{c1} {seconds:.3f}s predicted='
                f'{"none" if predicted is None else format(predicted, ".3f")}',
                flush=True,
            )
        budget = p['launch_budget_seconds']
        if budget and seconds > budget:
            raise RuntimeError(
                f'E_LAUNCH_BUDGET: {seconds:.3f}s > {budget:.3f}s ({log["profile"]} profile); kind={kind} '
                f'units={c0}:{c0 + units}; no further launches issued; this launch already completed'
            )
        if p['launch_pause_ms']:
            time.sleep(p['launch_pause_ms'] / 1000.0)
        return seconds

    def bs(self, first=False, variant=None):
        if variant:
            return KSPLIT_BLOCK
        return self.block_size

    def shared_bytes(self, W, phase, first=False, variant=None, *, boundary=False, ray_limit=False):
        W = self.tile_width(W, first, phase)
        warps = self.bs(first, variant) // 32
        cfg = self.config(W)
        layout = FastTransport.select(self.L, cfg, first, phase, boundary, ray_limit, bool(variant))
        if layout.role in ('warm', 'first'):
            return layout.shared_bytes(warps)
        # ACC_GLOBAL: the first-light phase-0 moment accumulator is the moments output (no shared rows).
        acc = (
            (4 + self.K)
            if phase == 0 and not ((first or cfg.split_warm) or (W == 15 and not first))
            else 0
        )
        size = (acc + 1) * W * 8 + warps * ((workspace(self.L) + 8 * W + 33) * 4 + 300 + 40 * W + 4)
        size += warps * (8 + 16 * W) if 'STRATO' in self.flags else 0
        # Ray constants occupy 15 floats per warp.
        size += warps * 15 * 4
        size += warps * 128
        # GROUND_FUSED (first-light phase 0): per-warp scratch 32 phases + 65 int2 LUT nodes (162 floats).
        # (with FL_RAYSPLIT the scratch lives in the warp's Y rows: no extra shared)
        size += (
            warps * 520 if (self.config(W).strato and not self.config(W).split_first and first and phase == 0) else 0
        )
        # PT_BATCH (R4): 16 PointData slots per warp, non-first-light kernels only; RAY_SORT keys+perm.
        # KSPLIT=3 consumer (variant slimray): kernel.cu PointM2, 112 B (16-byte aligned) per batched sample instead of
        # PointData, plus 16 B slack for the aligned batch base
        pt_bytes = 112 if (variant or '').startswith('slimray') else 300 + (8 if 'STRATO' in self.flags else 0)
        size += (
            warps * 16 * pt_bytes + (16 if (variant or '').startswith('slimray') else 0)
            if self.config(W).strato and (not first)
            else 0
        )
        size += 512 if (self.config(W).strato and not first) else 0
        return size

    def stage1_flags(self):
        _, profile, _ = self.device()
        defaults = tuple(profile.get('stage1_defaults') or ())
        if set(defaults) - set(STAGE1_FLAGS):
            raise ValueError(f'E_DEVICE_PROFILE: unknown stage1_defaults {sorted(set(defaults) - set(STAGE1_FLAGS))}')
        out = []
        for name in STAGE1_FLAGS:
            text = os.environ.get('SOSFUSE2_' + name, '').strip()
            if text not in ('', '0', '1'):
                raise ValueError(f'E_STAGE1_FLAG: SOSFUSE2_{name} must be 0 or 1')
            if text == '1' or (not text and name in defaults):
                out.append(name)
        return tuple(out)

    def kernel_key(self, flags, W, width, first, phase, boundary, *, ground_sh_l=0, ray_limit=False, stage1=()):
        """The one RawKernel memo-key rule (raw_kernel, fl_project_kernel, vprod_kernels, slim_proj_kernel): shared raw dicts
        (persistent service, payload restore) must not mix SH orders or PROFILE grids."""
        key = (tuple(flags), W, width, int(first), int(phase), int(boundary), ('L', self.L))
        key += (('PROFILE_N', self.profile_n), ('TOP_KM', self.domain.top_km), ('PROFILE_STEP', self.profile_step_km),
                ('AIR_END_KM', self.domain.air_end), ('LOWER_END_KM', self.domain.lower_end))
        cfg = self.config(W)
        layout = FastTransport.select(self.L, cfg, first, phase, boundary, ray_limit, 'SINV' in flags)
        if layout.defines():
            key += (layout.defines(),)
        if self.deterministic:
            key += ('DETERMINISTIC',)
        if ground_sh_l:
            key += (('GROUND_SH_L', ground_sh_l),)
        if ray_limit:
            key += ('LAUNCH_RAY_LIMIT',)
        return (key, stage1) if stage1 else key

    def attribute_suffix(self):
        return f'_profileN{self.profile_n}_top{float(self.domain.top_km).hex()}_step{self.profile_step_km.hex()}_air{float(self.domain.air_end).hex()}_lower{float(self.domain.lower_end).hex()}'

    def check_profile(self, optics, where, rows=None):
        """compiled N == declared optics N == table rows (== uploaded rows): density.check_consumed with this Kernel's
        PROFILE_N. Also on restored/persistent kernels whose setup() never ran (installed constants)."""
        from ..atmosphere.density import check_consumed

        return check_consumed(optics, self.profile_n, rows, where, domain=self.domain)

    def check_upload(self, c, dc, where):
        """The uploaded PROFILE buffer still is the optics' table: N as check_profile, and the identity
        recorded at upload equals the current optics identity; a rebound table/mask/policy is E_PROFILE_STALE."""
        if 'density_optics' not in c:
            return
        from ..atmosphere.density import profile_identity

        optics = c['density_optics']
        self.check_profile(optics, where, dc['profile_table'].shape[0])
        # Keyed by the uploaded buffer object: launch-local dict copies (KSPLIT record) share it.
        uploaded = self.__dict__.get('device_profile_identity', {}).get(id(dc['profile_table']))
        current = profile_identity(optics)['sha256']
        if uploaded != current:
            raise RuntimeError(
                f'E_PROFILE_STALE: {where}: uploaded PROFILE identity {str(uploaded)[:12]} != current optics '
                f'{current[:12]}; a changed table needs a new optics/Solver (no rebinding after attach)'
            )

    def transport_options(self, layout, flags, stage1, first, phase, boundary):
        return (
            '-std=c++17',
            '--fmad=true' if 'FMAD' in stage1 else '--fmad=false',
            '--prec-div=false' if 'FASTMATH' in stage1 else '--prec-div=true',
            '--prec-sqrt=false' if 'FASTMATH' in stage1 else '--prec-sqrt=true',
            f'-DFW={layout.W}', f'-DSOSFUSE2_DETERMINISTIC={int(self.deterministic)}',
            *(f'-DSOSFUSE2_{f}' for f in stage1 if f != 'FMAD'),
            *(f'-D{f}' for f in flags),
            f'-DK_FIRST={int(first)}', f'-DK_PHASE={int(phase)}', f'-DK_BOUNDARY={int(boundary)}',
            *self.domain.defines(), *layout.defines(),
        )

    def raw_kernel(self, W, first=0, phase=0, boundary=0, ray_limit=False, variant=None, ground_sh_l=0):
        if W not in (1, 15):
            raise ValueError('E_WAVELENGTHS: kernel specializes to 1 or 15 wavelengths')
        width = self.tile_width(W, first, phase)
        flags = (
            tuple(sorted(set(self.config(W).defines) | set(KSPLIT_VARIANTS[variant])))
            if variant
            else self.config(W).defines
        )
        stage1 = self.stage1_flags()
        # kernel_source(L) differs per SH order and PROFILE_N per node count; shared raw dicts must not mix them
        key = self.kernel_key(
            flags, W, width, first, phase, boundary, ground_sh_l=ground_sh_l, ray_limit=ray_limit, stage1=stage1
        )
        akey = ('+'.join(flags) or 'default') + ':' + f'{W}_first{int(first)}_phase{int(phase)}_boundary{int(boundary)}'
        akey += f'_tile{width}' + ('_raylimit' if ray_limit else '')
        if ground_sh_l:
            akey += f'_groundsh{ground_sh_l}'
        akey += self.attribute_suffix()
        programs = self.__dict__.setdefault('raw_options', {})
        if key not in programs:
            code = kernel_source(self.L, combine=(self.config(W).c3_fw15 and not first))
            layout = FastTransport.select(self.L, self.config(W), first, phase, boundary, ray_limit, bool(variant))
            options = self.transport_options(layout, flags, stage1, first, phase, boundary)
            if ray_limit:
                options += ('-DLAUNCH_RAY_LIMIT',)
            if ground_sh_l:
                options += (f'-DGROUND_SH_L={ground_sh_l}', f'-DGROUND_SH_GRID={GRID_INTERVALS}')
            programs[key] = (code, options)
        required = self.shared_bytes(width, phase, first=first, variant=variant, boundary=boundary, ray_limit=ray_limit)
        if key not in self.raw:
            code, options = programs[key]
            with stage('raw_compile'):
                raw = self.xp.RawKernel(code, 'sos_fused', options=options, backend='nvrtc')
                raw.compile()
            props = self.xp.cuda.runtime.getDeviceProperties(self.xp.cuda.runtime.getDevice())
            maximum = int(props.get('sharedMemPerBlockOptin', props['sharedMemPerBlock']))
            if required > maximum:
                raise RuntimeError(
                    f'E_SHARED_MEMORY: need {required} bytes, device supports {maximum}; reduce --block-size'
                )
            raw.preferred_shared_memory_carveout = 100
            self.raw[key] = raw
            self.attributes[akey] = dict(
                raw.attributes,
                compiler='NVRTC',
                implicit_fma='--fmad=true' in options,
                stage1=list(stage1),
                coordinate_precision='FP32',
                flags=list(flags),
                profile_n=self.profile_n,
            )
            if ground_sh_l:
                self.attributes[akey]['ground_sh'] = dict(order=ground_sh_l, grid_intervals=GRID_INTERVALS)
        self.raw[key].max_dynamic_shared_size_bytes = required
        self.attributes[akey].update(
            block_size=self.bs(first, variant) if variant else self.block_size, shared_bytes=required
        )
        if not variant:
            self.attributes[str(W)] = self.attributes[akey]
        return self.raw[key]

    def fl_project_kernel(self, W, boundary=0, ground_sh_l=0, first=True):
        """kernel.cu fl_project from the same source and options as the ray-range kernel of this pass: the first-light
        kernel, or with first=False the warm kernel of WARM_RAYSPLIT, which compiles its own specialization."""
        stage1 = self.stage1_flags()
        key = self.kernel_key(
            self.config(W).defines,
            W,
            self.tile_width(W, first, 0),
            int(bool(first)),
            0,
            boundary,
            ground_sh_l=ground_sh_l,
            ray_limit=True,
            stage1=stage1,
        )  # same key as raw_kernel(W, first, 0, boundary, ray_limit=True)
        pkey = key + ('fl_project',)
        if pkey not in self.raw:
            code, options = self.raw_options[key]
            with stage('raw_compile'):
                raw = self.xp.RawKernel(code, 'fl_project', options=options, backend='nvrtc')
                raw.compile()
            self.raw[pkey] = raw
        return self.raw[pkey]

    def source_is_zero(self, src):
        """Search bounded spans of source rows and stop at the first nonzero value."""
        if not src.size:
            return True
        if self.xp is np:
            return not bool(np.any(src))
        cells = len(src)
        row = int(src.size) // cells
        # Keep temporary reduction storage inside the preflight headroom. A larger
        # single source row can simply take the ordinary nonzero transport path.
        span = (16 << 20) // (row * src.itemsize)
        if not span:
            return False
        found = self.xp.zeros(1, 'i4')
        work = np.arange(cells + 1, dtype='i8') * row
        # CuPy's generic reduction can compile for multiple input shapes within
        # the timed probe. This fixed kernel has the same binary for every span.
        scan = self.vprod_plan_module().get_function('vp_source_nonzero')
        threads = 256
        name = 'transport_zero_source'
        for a, b in self.table_ranges(name, 0, cells, work, max_span=span):
            n = (b - a) * row
            self.timed_launch(name, scan, (-(-n // threads),), (threads,), (np.int32(n), src[a:b], found),
                              0, b - a, work, a, b)
            if bool(found[0]):
                return False
        return True

    def table_ranges(self, name, a, end, work, *, max_span=None):
        # Count scalar quadrature/SH operations before the first observation,
        # including pages with just one very expensive owner.
        profile = self.device()[1]
        step = profile['launch_cells_fixed'] or max_span
        if max_span is not None:
            step = min(step, max_span)
        while a < end:
            limit = _bounded_end(work, a, end, 1, TABLE_WORK_RATE, profile)
            b = self.budget_end(name, a, limit, step, work)
            yield a, b
            a = b

    def prepare_hg_table(self, launch, layout):
        """Replace repeated first-light ground sums with the same quadrature sampled in angle."""
        nodes = layout.hg_nodes
        counts = np.diff(launch.c['goff'])
        stride = nodes * nodes * layout.stride * 4
        if not _hg_fits(layout, counts, self.ksplit_memory(), _work_seconds(self.device()[1])):
            return
        owners = np.flatnonzero(counts).astype('i4')
        key = self.kernel_key(
            self.config(layout.W).defines, layout.W, self.tile_width(layout.W, True, 0), 1, 0, False,
            ray_limit=True, stage1=self.stage1_flags(),
        )
        name = 'hg_direction_table'
        if key + (name,) not in self.raw:
            code, options = self.raw_options[key]
            self.raw[key + (name,)] = self.xp.RawKernel(code, name, options=options, backend='nvrtc')
        generate = self.raw[key + (name,)]
        ids = self.xp.asarray(owners)
        pointers = np.zeros(len(counts), dtype='u8')
        launch.hg_pages = [
            self.xp.empty((min(owners.size - start, layout.HG_PAGE_CELLS), nodes, nodes, layout.stride), 'f4')
            for start in range(0, owners.size, layout.HG_PAGE_CELLS)
        ]
        dc = launch.dc
        work = np.r_[0, np.cumsum(counts[owners], dtype='i8')] * nodes**2 * launch.W
        rows = layout.THREADS // layout.stride
        for i, page in enumerate(launch.hg_pages):
            start = i * layout.HG_PAGE_CELLS
            end = start + len(page)
            pointers[owners[start:end]] = page.data.ptr + np.arange(len(page), dtype='u8') * stride
            for a, b in self.table_ranges(name, start, end, work):
                self.timed_launch(
                    name, generate, (b - a, nodes, -(-nodes // rows)), (layout.THREADS,),
                    (ids[a:], dc['goff'], dc['gdir'], dc['gval'], launch.c['g'], page[a - start:]),
                    0, b - a, work, a, b,
                )
        # A null offset pointer selects the table in the first-light kernel.
        launch.args[launch.goff_index + 2] = self.xp.asarray(pointers)
        launch.args[launch.goff_index] = np.uint64(0)

    def fast_vprod_kernels(self, layout, boundary):
        """Compile and check the actual functions before choosing coefficient staging."""
        W = layout.W
        flags = self.config(W).defines
        stage1 = self.stage1_flags()
        base = self.kernel_key(flags, W, self.tile_width(W, False, 0), 0, 0, boundary, stage1=stage1)
        names = ['sos_fast_slim', 'sos_slim_proj']
        names += ['fl_project', 'sh_direction_table'] if layout.role == 'angular' else [
            'sos_vprod_y', 'vp_c0cp_chunk', 'sos_vprod_direct'
        ]
        free = self.ksplit_memory()
        out = {'local_reserve_bytes': 0}
        for name in names:
            threads = 256 if name == 'fl_project' else layout.THREADS
            candidates = [layout]
            if name == 'sos_vprod_direct':
                # The second variant reads the coefficient plane from global memory;
                # it retains only the visitor tile in shared memory.
                candidates.append(replace(layout, stage_plane=False))
            for candidate in candidates:
                options = self.transport_options(candidate, flags, stage1, False, 0, boundary)
                key = base + (candidate.defines(), name)
                akey = (f'{name}:{W}_L{layout.L}_{candidate.role}_plane{int(candidate.stage_plane)}'
                        f'_boundary{int(boundary)}_det{int(self.deterministic)}'
                        + ''.join('_' + f for f in (*flags, *stage1)) + self.attribute_suffix())
                dynamic = candidate.direct_shared if name == 'sos_vprod_direct' else 0
                if key not in self.raw:
                    with stage('raw_compile'):
                        raw = self.xp.RawKernel(
                            kernel_source(self.L, combine=self.config(W).c3_fw15), name, options=options, backend='nvrtc'
                        )
                        raw.compile()
                    props = self.xp.cuda.runtime.getDeviceProperties(self.xp.cuda.runtime.getDevice())
                    attr = raw.attributes
                    static = attr['shared_size_bytes']
                    shared_limit = props.get('sharedMemPerBlockOptin') or props['sharedMemPerBlock']
                    active = 0
                    if (static + dynamic <= shared_limit and attr['max_threads_per_block'] >= threads
                            and attr['num_regs'] * threads <= props['regsPerBlock']):
                        if static + dynamic > props['sharedMemPerBlock']:
                            raw.max_dynamic_shared_size_bytes = dynamic
                        active = self.xp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
                            raw.kernel.ptr, threads, dynamic)
                    self.attributes[akey] = dict(
                        attr, block_size=threads, shared_bytes=dynamic, resident_blocks_per_sm=active,
                        local_reserve_bytes=attr['local_size_bytes'] * props['maxThreadsPerMultiProcessor']
                                            * props['multiProcessorCount'],
                        compiler='NVRTC', stage1=list(stage1), implicit_fma='--fmad=true' in options,
                        coordinate_precision='FP32', profile_n=self.profile_n, flags=list(flags),
                    )
                    self.raw[key] = raw
                attrs = self.attributes[akey]
                if attrs['resident_blocks_per_sm'] and attrs['local_reserve_bytes'] <= free:
                    out[name] = self.raw[key]
                    out['local_reserve_bytes'] = max(out['local_reserve_bytes'], attrs['local_reserve_bytes'])
                    if name == 'sos_vprod_direct':
                        out['shared_bytes'] = dynamic
                    if name == 'sos_fast_slim':
                        self.attributes.setdefault(str(W), attrs)
                    break
            else:
                raise _FastResourceError(f'No resident block for {name} at L{layout.L}, FW{W}')
        return out

    def fast_vprod_launch(self, launch, bundle, s, boundary):
        """Project each bounded cell span, evaluating aerosol from a table or direct SH contraction."""
        xp = self.xp
        kmap = launch.fv_kernels
        c0cp_kern = kmap.get('vp_c0cp_chunk')
        ykern = kmap.get('sos_vprod_y')
        vkern = kmap.get('sos_vprod_direct')
        layout = launch.fast_layout
        skim = kmap['sos_fast_slim']
        pkern = kmap['sos_slim_proj']
        flpkern = kmap.get('fl_project')
        nh, nc = len(s.geo.h), len(s.geo.chi)
        dc = launch.dc
        src_sh = launch.args[launch.sh_index]
        end_col_arr = launch.args[9 + launch.names.index('end_col')]
        profile_table = dc.get('profile_table', np.uint64(0))
        profile_mask = np.int32(launch.c.get('density_optics', {}).get('mask', 0))
        air_surface = F(launch.c.get('density_optics', {}).get('air_surface', 1.0))
        nsh = np.int32(len(launch.c['sun_h']))
        co, pcum = bundle._fast_vprod_meta
        chunks = launch.fast_chunks
        max_npt = max((int(pcum[b] - pcum[a]) for a, b in chunks), default=1)
        max_nray = max((int(co[b] - co[a]) for a, b in chunks), default=1)
        AW = aero_w(launch.W)
        t_table = table_max = 0.0
        zero_volume = launch.zero_volume
        if zero_volume and launch.flux_only and not boundary:
            work = np.arange(launch.ncell + 1, dtype='i8') * launch.W
            name = launch.kind + '_zero_flux'
            report = dict(chunks=0, gpu_seconds=0.0, max_seconds=0.0, max_cells=0)

            def clear(grid, block, args, shared_mem):
                args[0].fill(0)

            for a, b in self.table_ranges(name, 0, launch.ncell, work):
                elapsed = self.timed_launch(name, clear, (), (), (launch.flux[a:b],), 0, b - a, work, a, b)
                report['gpu_seconds'] += elapsed
                report['max_seconds'] = max(report['max_seconds'], elapsed)
                report['max_cells'] = max(report['max_cells'], b - a)
                report['chunks'] += 1
            launch.report = report
            return
        is_strato = 'strato_ext' in launch.c and self.config(launch.W).strato
        st_c0cp = (dc['strato_density'],) if is_strato else ()
        st_vprod = (dc['strato_density'],) if is_strato else ()
        st_slim = (
            (
                dc['strato_ext'],
                dc['strato_density'],
                launch.endst,
                F(launch.profile['center']),
                F(launch.profile['sigma']),
                F(launch.profile['norm']),
            )
            if is_strato
            else ()
        )
        with vram_trace.item('fast_vprod.scratch'):
            if zero_volume:
                aero = np.uint64(0)
                c0cp = None
                src_rm = np.uint64(0)
            else:
                src_rm = xp.zeros((launch.src.shape[0], 4, layout.stride), 'f4')
                src_rm[:, :, :launch.W] = launch.src[:, :4, :]
                if launch.angular:
                    nodes = layout.sh_nodes
                    cells = nh * nc
                    owners = getattr(launch, 'live_cells', None)
                    if owners is None:
                        owners = _live_sh_cells(launch.c, is_strato, nh, nc, launch.W)
                    stride = nodes * nodes * layout.stride * 4
                    pages = [xp.empty((min(len(owners) - i, layout.PAGE_CELLS), nodes, nodes, layout.stride), 'f4')
                             for i in range(0, len(owners), layout.PAGE_CELLS)]
                    pointers = np.zeros(cells, dtype='u8')
                    if len(owners):
                        if not hasattr(self, '_sh_spline_basis'):
                            self._sh_spline_basis = tuple(xp.asarray(v) for v in sh_spline_basis(layout))
                        owners_dev = xp.asarray(owners)
                        per_cell = nodes * layout.stride * (self.K + nodes * (self.L + 1))
                        work = np.arange(len(owners) + 1, dtype='i8') * per_cell
                        for i, page in enumerate(pages):
                            start = i * layout.PAGE_CELLS
                            end = start + len(page)
                            pointers[owners[start:end]] = page.data.ptr + np.arange(len(page), dtype='u8') * stride
                            for a, b in self.table_ranges('sh_direction_table', start, end, work):
                                elapsed = self.timed_launch(
                                    'sh_direction_table', kmap['sh_direction_table'], (b - a, nodes), (layout.THREADS,),
                                    (owners_dev[a:], src_sh, *self._sh_spline_basis, page[a - start:]), 0, b - a, work, a, b,
                                )
                                t_table += elapsed
                                table_max = max(table_max, elapsed)
                    aero = xp.asarray(pointers)
                    c0cp = None
                else:
                    aero = xp.empty(max(max_npt, 1) * 4 * AW, 'f4')
                    c0cp = xp.empty(max(max_npt, 1) * 2, 'i4')
            yb = np.uint64(0) if launch.angular else xp.empty((max(max_nray, 1), self.K), 'f4')
            ray_rads = xp.empty(max(max_nray, 1) * 4 * launch.W, 'f4')
        report = dict(chunks=0, gpu_seconds=t_table, max_seconds=table_max, max_cells=0)
        kind = launch.kind
        tiles = 1 if launch.flux_only else -(-(5 + self.K) // layout.PROJ_MOMENTS)
        work = launch.fast_work
        warps = layout.THREADS // 32

        def run_chunk(c0, c1):
            base = int(pcum[c0])
            npt = int(pcum[c1]) - base
            r0, r1 = int(co[c0]), int(co[c1])
            nray = r1 - r0
            c0cp_view = None if launch.angular else c0cp[: max(2 * npt, 2)]
            if not launch.angular and nray and npt:
                nb_c = -(-nray // warps)
                c0cp_kern(
                    (nb_c,),
                    (layout.THREADS,),
                    (
                        np.int32(c1 - c0),
                        np.int32(nh),
                        np.int32(nc),
                        np.int32(bundle.nq),
                        np.int64(r0),
                        np.int32(nray),
                        np.int64(base),
                        bundle.arrays['cell_off'][c0:],
                        bundle.arrays['starts'][c0:],
                        bundle.arrays['dirs'],
                        bundle.arrays['edge_off'],
                        bundle.arrays['edges'],
                        bundle.arrays['lower'],
                        bundle.arrays['ray_geometry'],
                        dc['u'],
                        dc['h'],
                        dc['chi'],
                        dc['hr'],
                        dc['ha'],
                        c0cp_view,
                        *st_c0cp,
                    ),
                )
            plan = None
            if not launch.angular:
                # The enclosing timed launch includes the plan kernels and the CuPy scan/sort.
                kb = dict(chunks=[(c0, c1)], pcum=pcum, co=co)
                plan = self.vprod_build_plan(kb, 0, bundle, c0cp_view, nh, nc,
                                             count_blocks=vram_budget.PLAN_COUNT_BLOCKS, measure_blocks=False)
            if nray and not launch.angular:
                nb_y = -(-nray // warps)
                ykern(
                    (nb_y,),
                    (layout.THREADS,),
                    (
                        np.int32(nray),
                        bundle.arrays['dirs'].reshape(-1)[3 * r0 :],
                        dc['aa'],
                        dc['bb'],
                        dc['dd'],
                        yb,
                    ),
                )
            nblk = 0 if plan is None else plan['nblk']
            if nblk:
                vargs = (
                    plan['blk'],
                    plan['vis_rbeg'],
                    plan['vis_ray'],
                    plan['slot'],
                    c0cp_view,
                    yb,
                    src_sh,
                    dc['hr'],
                    dc['ha'],
                    np.int32(nh),
                    np.int32(nc),
                    aero,
                    *st_vprod,
                )
                vkern((nblk,), (layout.THREADS,), vargs, shared_mem=kmap['shared_bytes'])
            del plan
            if nray:
                nb_s = -(-nray // warps)
                sargs = (
                    np.int32(c1 - c0),
                    np.int32(nh),
                    np.int32(nc),
                    np.int32(bundle.nq),
                    np.int64(r0),
                    np.int32(0),
                    np.int32(nray),
                    np.int64(base),
                    bundle.arrays['cell_off'][c0:],
                    bundle.arrays['starts'][c0:],
                    bundle.arrays['targets'][c0:],
                    bundle.arrays['dirs'],
                    bundle.arrays['edge_off'],
                    bundle.arrays['edges'],
                    bundle.arrays['lower'],
                    bundle.arrays['ray_geometry'],
                    bundle.arrays['end_j'],
                    bundle.arrays['end_f'],
                    end_col_arr,
                    bundle.arrays['hits'],
                    dc['u'],
                    dc['wu'],
                    dc['h'],
                    dc['chi'],
                    dc['hr'],
                    dc['ha'],
                    dc['ext'],
                    dc['sun_h'],
                    nsh,
                    launch.c['height'],
                    launch.c['mix'],
                    src_rm,
                    launch.bd,
                    aero,
                    profile_table,
                    profile_mask,
                    air_surface,
                    ray_rads,
                    *st_slim,
                )
                skim((nb_s,), (layout.THREADS,), sargs)
            if launch.angular and not launch.flux_only:
                flpargs = (
                    np.int32(c1 - c0),
                    np.int64(r0),
                    bundle.arrays['cell_off'][c0:],
                    bundle.arrays['targets'][c0:],
                    bundle.arrays['dirs'],
                    bundle.arrays['weights'],
                    dc['aa'],
                    dc['bb'],
                    dc['dd'],
                    ray_rads,
                    F(launch.c['albedo']),
                    launch.moments[c0:c1],
                    launch.flux[c0:c1],
                )
                flpkern((c1 - c0,), (256,), flpargs)
            else:
                pargs = (
                    np.int32(c1 - c0),
                    np.int32(-1 if launch.flux_only else 4),
                    np.int64(r0),
                    bundle.arrays['cell_off'][c0:],
                    bundle.arrays['targets'][c0:],
                    bundle.arrays['dirs'],
                    bundle.arrays['weights'],
                    bundle.arrays['edge_off'],
                    yb,
                    ray_rads,
                    F(launch.c['albedo']),
                    launch.moments if launch.flux_only else launch.moments[c0:c1],
                    launch.flux[c0:c1],
                )
                pkern((c1 - c0, tiles), (layout.THREADS,), pargs)

        # Time the whole dependency chain as one launch group. This charges the
        # scan/sort library kernels too, and splits their input *before* planning.
        group = kind + ('_angular' if launch.angular else '_direct')
        initial = self.device()[1]['launch_cells_initial']

        def execute(grid, block, args, shared_mem):
            run_chunk(*args)

        for start, limit in chunks:
            a = start
            while a < limit:
                b = self.budget_end(group, a, limit, launch.step, work, initial)
                seconds = self.timed_launch(group, execute, (), (), (a, b), 0, b - a, work, a, b)
                report['gpu_seconds'] += seconds
                report['max_seconds'] = max(report['max_seconds'], seconds)
                report['max_cells'] = max(report['max_cells'], b - a)
                report['chunks'] += 1
                a = b
        launch.report = report

    def constants_device(self, s, nq):
        key = (id(s), nq)
        if key not in self.device_constants:
            c = self.setup(s, nq)
            with stage('constants_device', device=True):
                names = (
                    'h',
                    'chi',
                    'hr',
                    'ha',
                    'ext',
                    'asca',
                    'sun_h',
                    'sun_col',
                    'gm',
                    'goff',
                    'gdir',
                    'gval',
                    'u',
                    'wu',
                )
                dc = {}
                for name in names:
                    with vram_trace.item(f'constants.{name}'):
                        dc[name] = self.xp.asarray(np.ascontiguousarray(c[name], dtype='f4' if name in ('u', 'wu') else None))
                if 'strato_ext' in c:
                    for name in ('strato_ext', 'strato_sca', 'strato_chi', 'strato_density', 'strato_sun_col'):
                        with vram_trace.item(f'constants.{name}'):
                            dc[name] = self.xp.asarray(np.ascontiguousarray(c[name]))
                if 'strato_ext' in c and self.config().strato:
                    with vram_trace.item('lut.strato'):
                        dc['strato_chi'] = self.strato_lut_device(dc['strato_chi'], c['W'])
                if self.ground_sh_l and c['gval'].size:
                    from .ground_sh import lut_eigenvalues, project, synthesize

                    # The accepted state is immutable, and ground illumination
                    # does not depend on the ray quadrature order nq. Observer,
                    # first-light and table transport share this one allocation.
                    ground_key = (id(s), self.ground_sh_l)
                    if ground_key not in self.ground_sh_tables:
                        lut = self.host(dc['strato_chi']).reshape(-1, c['W'])[33:]
                        eigen = lut_eigenvalues(lut, self.ground_sh_l)
                        coeff = project(c['goff'], c['gdir'], c['gval'], c['g'], eigen, self.ground_sh_l)
                        with vram_trace.item('lut.ground_sh'):
                            self.ground_sh_tables[ground_key] = synthesize(coeff, self.ground_sh_l, self.xp)
                    dc['ground_sh'] = self.ground_sh_tables[ground_key]
                    self.metrics['ground_sh_order'] = self.ground_sh_l
                    self.metrics['ground_sh_bytes'] = int(dc['ground_sh'].nbytes)
                if 'density_optics' in c:
                    from ..atmosphere.density import upload_buffer, evidence

                    buf = upload_buffer(c['density_optics'])
                    self.check_profile(c['density_optics'], 'upload', buf.shape[0])
                    dc['profile_table'] = self.xp.asarray(buf)
                    self.profile_consumed = evidence(c['density_optics'], self.profile_n, buf)
                    # The identity this upload belongs to (checked on every reuse and at launch).
                    self.__dict__.setdefault('device_profile_identity', {})[id(dc['profile_table'])] = (
                        self.profile_consumed['profile_identity']
                    )
                if ('tropo_lut' in c) != ('TROPO_MIE' in self.flags):
                    raise RuntimeError('E_TROPO_LUT_FLAGS: table and TROPO_MIE must come together')
                if 'tropo_lut' in c:
                    # kernel.cu TROPO_LUT = asca+FW: asca[0:W] unchanged, the per-wavelength phase table follows.
                    if c['tropo_lut'].size != (c['tropo_bins'] + 1) * c['W'] or c['tropo_bins'] != 16384:
                        raise ValueError('E_TROPO_LUT_SHAPE')
                    dc['asca'] = self.xp.asarray(
                        np.ascontiguousarray(np.concatenate([np.asarray(c['asca'], 'f4'), c['tropo_lut']]), 'f4')
                    )
                if ('aer2' in c) != ('AEROSOL_2L' in self.flags):
                    raise RuntimeError('E_AEROSOL_2L_FLAGS: profile vector and AEROSOL_2L must come together')
                if 'aer2' in c:
                    # kernel.cu aerosol_two_layer reads sun_h+nsh: sun_h[0:nsh] unchanged (nsh = len(c['sun_h'])), the 7 parameters follow.
                    if c['aer2'].shape != (7,):
                        raise ValueError('E_AEROSOL_2L_SHAPE')
                    dc['sun_h'] = self.xp.asarray(
                        np.ascontiguousarray(np.concatenate([np.asarray(c['sun_h'], 'f4'), c['aer2']]), 'f4')
                    )
                constants = norm_constants(self.L)
                dc.update(zip(('aa', 'bb', 'dd'), [self.xp.asarray(v, dtype='f4') for v in constants]))
            self.device_constants[key] = dc
        self.check_upload(self.setup(s, nq), self.device_constants[key], 'device_constants')
        if self.config().strato and 'strato_chi' in self.device_constants[key]:
            bins = 16384
            expected = (33 + bins + 1) * self.setup(s, nq)['W']
            if self.device_constants[key]['strato_chi'].size != expected:
                raise RuntimeError('E_STRATO_LUT_CACHE: phase flags changed after setup; create a fresh Kernel')
        return self.device_constants[key]

    def strato_lut_kernel(self, W):
        """strato_lut_build, compiled before its first launch (precompile.py compiles the same source/options)."""
        code = paths.KERNEL_CU.read_text()
        options = ('-std=c++17', '--fmad=false', '--prec-div=true', '--prec-sqrt=true', f'-DFW={W}')
        options += (f'-DSOSFUSE2_DETERMINISTIC={int(self.deterministic)}',)
        options += tuple('-D' + f for f in resolve_config(self.flags, role='lut', W=W).flags)
        options += self.domain.defines()
        with stage('raw_compile'):
            builder = self.xp.RawKernel(code, 'strato_lut_build', options=options, backend='nvrtc')
            builder.compile()
        return builder

    def strato_lut_device(self, coeff, W):
        """Same pointer argument, enlarged private allocation; nodes built on GPU."""
        bins = 16384
        xp = self.xp
        if xp is np:
            raise RuntimeError('E_STRATO_LUT_BACKEND: this builder requires CuPy; use native phase tests on CPU')
        if coeff.size != 33 * W:
            raise ValueError('E_STRATO_CHI_SHAPE: 33*W coefficients required')
        out = xp.empty((33 + bins + 1) * W, dtype='f4')
        out[: 33 * W] = coeff.ravel()
        builder = self.strato_lut_kernel(W)
        builder((((bins + 1) * W + 255) // 256,), (256,), (out,))
        # Same stream orders the table producer before all consumers.
        self.metrics['strato_lut_bytes'] = int(out.nbytes)
        self.metrics['strato_lut_intervals'] = bins
        return out

    def strato_end_columns(self, bundle, c):
        """Bounded-memory sidecar on immutable Bundle; never part of cache schema."""
        from ..atmosphere.stratosphere import columns

        profile = c['strato_profile']
        key = tuple(profile[k] for k in ('center', 'sigma', 'norm'))
        if not hasattr(bundle, 'strato_columns'):
            bundle.strato_columns = {}
        if key not in bundle.strato_columns:
            from ..precompute.column_cache import cached_columns

            def build():
                original = getattr(bundle, 'host_coordinates', bundle.arrays)
                starts, co, dirs = (original[k] for k in ('starts', 'cell_off', 'dirs'))
                out = np.empty(bundle.rays, 'f4')
                from ..precompute.column_parallel import parallel_columns

                if parallel_columns(out, 'strato', starts, co, dirs, c['strato_old'], profile):
                    return out  # parallel preparation, rtol 1e-6
                # Do not generate full-grid [ray,GLnode,xyz] arrays on the host.
                for start in range(0, bundle.rays, 2048):
                    end = min(start + 2048, bundle.rays)
                    index = np.searchsorted(co, np.arange(start, end), side='right') - 1
                    out[start:end] = columns(starts[index], dirs[start:end], c['strato_old'], profile, 24)
                return out

            out = cached_columns(self, bundle, c, 'strato', build)
            with vram_trace.item('geometry.strato_columns'):
                bundle.strato_columns[key] = self.xp.asarray(out)
        return bundle.strato_columns[key]

    def density_end_columns(self, bundle, c):
        from ..atmosphere.density import correct_columns
        from ..atmosphere.density import profile_identity

        optics = c['density_optics']
        key = profile_identity(optics)['sha256']  # effective table, grid, mask, policy
        if not hasattr(bundle, 'density_columns'):
            bundle.density_columns = {}
        if key not in bundle.density_columns:
            from ..precompute.column_cache import cached_columns

            def build():
                original = getattr(bundle, 'host_coordinates', bundle.arrays)
                starts, co, dirs = (original[k] for k in ('starts', 'cell_off', 'dirs'))
                out = self.host(bundle.arrays['end_col']).copy()
                from ..precompute.column_parallel import parallel_columns

                if parallel_columns(out, 'density', starts, co, dirs, optics['old'], optics):
                    return out  # parallel preparation, rtol 1e-6
                for start in range(0, bundle.rays, 2048):
                    end = min(start + 2048, bundle.rays)
                    index = np.searchsorted(co, np.arange(start, end), side='right') - 1
                    out[start:end] = correct_columns(
                        starts[index], dirs[start:end], out[start:end], optics['old'], optics, 24
                    )
                return out

            out = cached_columns(self, bundle, c, 'density', build)
            with vram_trace.item('geometry.density_columns'):
                bundle.density_columns[key] = self.xp.asarray(np.ascontiguousarray(out, 'f4'))
        return bundle.density_columns[key]

    def launch(self, bundle, s, field, first=False, phase=0, boundary=None, timed=False, _ksplit=None):
        launch = _TransportLaunch()
        launch.section_dir = _section_clock_directory(self)
        _prepare_transport_constants(self, launch, bundle, s, field, first, phase, boundary, _ksplit)
        _allocate_transport_outputs(self, launch, s, field, first, phase, boundary)
        _assemble_transport_arguments(self, launch, bundle, s, first, phase, boundary)
        allow_table = allow_fast = True
        while True:
            _prepare_launch_order(self, launch, bundle, first, phase, allow_fast, allow_table)
            try:
                _select_transport_kernels(self, launch, bundle, first, phase, boundary, _ksplit)
                if timed or launch.section_buffer is not None:
                    launch.begin = launch.xp.cuda.Event()
                    launch.end = launch.xp.cuda.Event()
                    launch.begin.record()
                _dispatch_transport(self, launch, bundle, s, first, phase, boundary, _ksplit)
                break
            except Exception as exc:
                if not launch.fast_vprod or not _transport_resource_error(exc):
                    raise
                # Retry from cell zero; all outputs stay private until the complete pass returns.
                allow_fast, allow_table = launch.angular, False
                warnings.warn(f'Transport fallback ({type(exc).__name__}): {exc}', RuntimeWarning)
        _record_transport_clocks(self, launch, first, phase, timed)
        self.metrics['launches'] += 1
        if phase == 2:
            self.metrics['query_launches'] += 1
        self.memory()
        return launch.moments, launch.flux, launch.query

    def transport(self, geometry, s, field, first=False):
        """Accepted query adapter, but with regenerated geometry, not raw points."""
        if 'edges' not in geometry:
            raise ValueError('E_SAMPLE_CACHE: fused path requires starts/directions/segments, not legacy point arrays')
        c = self.setup(s, geometry['nq'])
        packed = (
            np.ascontiguousarray(self.host(field)[..., ::2], 'f4')
            if np.size(field)
            else np.zeros((s.geo.ns, source_size(s), s.W), 'f4')
        )
        if self.xp is np:
            value = numpy_reference.transport(geometry, c, packed, first, statistics=self.metrics)
        else:
            from .ray_geometry import Bundle

            bundle = Bundle([geometry], self.xp)
            _, _, value = self.launch(bundle, s, self.xp.asarray(packed), first, 2)
            value = self.host(value)
        out = np.zeros((len(value), s.W, 2, 4), 'f8')
        out[:, :, 0] = value
        return out
