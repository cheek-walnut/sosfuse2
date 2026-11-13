"""CuPy RawKernel wrapper and NumPy reference, with no sample cache.

Only contiguous arrays are passed to NVRTC. Source layout is cell,10+K,wave
or cell,10+2*K,wave for an explicitly enabled stratosphere, where K is the SH mode count;
the inactive scalar diagnostic lane is omitted and never substitutes for the
polarized calculation. Boundary and source outputs retain all K SH modes.
"""

import importlib.util
import json
import os
import math
import time
import warnings
from pathlib import Path
import numpy as np
from . import reference
from .components import NCOMP
from .arithmetic import F, norm_constants
from .sh_layout import count, padded, workspace, source_size, kernel_source, combined_aux
from .timers import TIMERS, stage
from .kernel_config import STAGE1_FLAGS, KSPLIT_VARIANTS, resolve_config, deterministic

PROFILE_N_DEFAULT = 161  # density.PROFILE_STEPS[0.5]; kernel.cu #ifndef PROFILE_N

from . import paths  # noqa: E402  (public split: package-relative resources, work root for caches)

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
# A6 kernel variants (kernel.cu header); each independent, all off = original kernel.
# Stage-1 compile options: SOSFUSE2_<NAME>=1/0 overrides,
# unset = device profile stage1_defaults (configs/devices.json: rig FMAD+FASTMATH, eda/mac none).
# None active = the exact pre-r17 options, key and kernel tokens.
KPAD = 576
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
    """A6(c): SH source as a new contiguous [cell][wave][KPAD] array (not a view):
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


def load_engine():
    """A fresh, private copy of the accepted engine module (callers patch its namespace; never the shared import).
    The module name keeps the package context so its relative imports resolve; the solver modules stay shared."""
    spec = importlib.util.spec_from_file_location('sosfuse2.accepted.engine_readonly', paths.ACCEPTED / 'engine.py')
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    return engine


from .device_profile import first_light_split_error  # noqa: E402 (stdlib-only module; pipeline preflight uses it too)


def ksplit_setting(backend, flags, L):
    """Resolve KSPLIT for an FW15 physical configuration; setup defers FW1 to mode 0.
    The default needs CuPy, STRATO, L<=16 and uninstrumented transport.
    """
    config = resolve_config(flags)
    if config.instrumented:
        return '0'
    text = os.environ.get('SOSFUSE2_KSPLIT', '').strip()
    if text:
        return text
    fits = backend == 'cupy' and config.ksplit_supported and L <= 16
    return KSPLIT_DEFAULT if fits else '0'


class Kernel(KsplitMixin):
    def __init__(
        self, threads=4, height=1.2, g=0.8, backend='numpy', block_size=128, flags=None, L=32, *, profile_step_km=0.5
    ):
        if backend not in ('numpy', 'cupy'):
            raise ValueError('E_BACKEND: select numpy or cupy explicitly')
        if block_size not in (32, 64, 128, 256):
            raise ValueError('E_BLOCK_SIZE: select 32,64,128,256')
        self.deterministic = deterministic()
        self.L = L
        self.K = count(L)
        self.height, self.g, self.backend, self.block_size = height, g, backend, block_size
        # ②b: PROFILE node count is a compile constant (kernel.cu PROFILE_N); 0.5 km (161) adds no option and no key field.
        from .density import profile_n

        self.profile_n = profile_n(profile_step_km)
        self.profile_step_km = float(profile_step_km)
        self.flags = resolve_config(default_flags() if flags is None else flags).flags
        from .ground_sh import configured_order

        self.ground_sh_l = configured_order()
        # Production default:
        # unset means KSPLIT=3 (VPROD + warm ray consumer + projection) where it applies (CuPy, STRATO/FW15, L<=16, FP64
        # projection accumulator) and the unsplit kernel elsewhere; SOSFUSE2_KSPLIT=0 forces the unsplit kernel, and an
        # explicit 1-3 still fails loudly on an unsupported configuration.
        ks_text = ksplit_setting(backend, self.flags, L)
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
        y = reference.basis(np.asarray(dirs)[:, 2], L)
        st = np.sqrt(np.maximum(0.0, 1 - np.asarray(dirs)[:, 2] ** 2))
        cp = np.where(st > 1e-12, dirs[:, 0] / np.maximum(st, 1e-30), 1.0).astype('f4')
        cm = reference.cosine(cp)
        for l in range(L + 1):
            for m in range(l + 1):
                y[:, l * (l + 1) // 2 + m] *= cm[:, m]
        return y.astype('f8')

    def setup(self, s, nq):
        key = (id(s), nq)
        if key not in self.constants:
            with stage('kernel_setup'):
                from .stratosphere import attach
                from .density import attach as attach_density

                density = attach_density(s, s.ctx)
                if density is not None:
                    self.check_profile(density, 'setup')
                optics = attach(s, s.ctx)
                if optics is not None:
                    self.flags = tuple(sorted(set(self.flags) | {'STRATO'}))
                elif 'STRATO' in self.flags:
                    raise ValueError('E_STRATO_FLAG_ZERO: STRATO is automatic for nonzero tau')
                from .tropo_mie import attach as attach_tropo

                tropo = attach_tropo(s)
                if self.ground_sh_l and (tropo is not None or optics is None):
                    raise ValueError('E_GROUND_SH_PHASE: requires tropospheric HG and enabled stratosphere')
                if tropo is not None:
                    self.flags = tuple(sorted(set(self.flags) | {'TROPO_MIE'}))
                elif 'TROPO_MIE' in self.flags:
                    raise ValueError('E_TROPO_FLAG_DEFAULT: TROPO_MIE is automatic for an enabled troposphere_phase')
                from .tropo_profile import attach as attach_profile

                aer = attach_profile(s)
                if aer is not None:
                    self.flags = tuple(sorted(set(self.flags) | {'AEROSOL_2L'}))
                elif 'AEROSOL_2L' in self.flags:
                    raise ValueError(
                        'E_AEROSOL_2L_FLAG_DEFAULT: AEROSOL_2L is automatic for an enabled troposphere_profile'
                    )
                self._W = s.W
                cfg = self.config()
                if self.ground_sh_l and not cfg.ground_sh_supported:
                    raise ValueError('E_GROUND_SH_FLAGS: requires fused strato ground, split first light and HG')
                ks = (
                    ksplit_setting(self.backend, self.flags, self.L)
                    if s.W == 15
                    else os.environ.get('SOSFUSE2_KSPLIT', '').strip() or '0'
                )
                if cfg.instrumented:
                    ks = '0'
                if ks not in ('0', '1', '2', '3') or (ks != '0' and not cfg.ksplit_supported):
                    raise ValueError('E_KSPLIT_FLAGS: needs STRATO, FW15 and an uninstrumented transport role')
                decision = getattr(s, 'ksplit_decision', None)
                self.ksplit_mode = decision['effective'] if decision is not None else int(ks)
                self.ksplit = bool(self.ksplit_mode)
                if self.ksplit_verify and self.ksplit_mode not in (2, 3):
                    # Only a CuPy FW15 kernel can run KSPLIT: the persistent pipeline's CPU-prepare Kernel (NumPy backend) and FW1 kernels
                    # have nothing to verify, so VERIFY=1 is inactive there with a warning (SOSFUSE2_KSPLIT_VERIFY=1
                    # could not reach the persistent service at all; the service's CuPy Kernel still enforces it).
                    if self.xp is not np and s.W == 15:
                        raise ValueError('E_KSPLIT_VERIFY: requires KSPLIT=2 or 3')
                    warnings.warn(
                        'W_KSPLIT_VERIFY_INACTIVE: SOSFUSE2_KSPLIT_VERIFY=1 does not apply to this kernel (NumPy backend or FW1)',
                        RuntimeWarning,
                        stacklevel=2,
                    )
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
        c = self.constants[key]
        # Every setup() call, cache hit or miss and installed constants alike (NumPy transport, geometry,
        # table_fast and the persistent service all read prepared constants through here).
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
        """(name, profile, reason) from configs/devices.json, read once per Kernel (device_profile.py)."""
        if getattr(self, '_device', None) is None:
            from .device_profile import select, cuda_device_name

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
            # eda: never a whole-grid launch; SOSFUSE2_LAUNCH_CELLS is only an upper bound under the predictive cap.
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
        t = L + r*w (L = per-launch latency floor, one cell ~0.8 s on the 3080; r = seconds per unit of work),
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
        ):  # no launch budget (rig profile): fixed step, or the whole rest when the caller has none (KSPLIT=3 projection)
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

    def shared_bytes(self, W, phase, first=False, variant=None):
        W = self.tile_width(W, first, phase)
        warps = self.bs(first, variant) // 32
        # ACC_GLOBAL: the first-light phase-0 moment accumulator is the moments output (no shared rows).
        acc = (
            (4 + self.K)
            if phase == 0 and not ((first or self.config(W).split_warm) or (self.config(W).c3_fw15 and not first))
            else 0
        )
        size = (acc + 1) * W * 8 + warps * ((workspace(self.L) + 8 * W + 33) * 4 + 300 + 40 * W + 4)
        size += warps * (8 + 16 * W) if 'STRATO' in self.flags else 0
        # Match RAY_CONST_F: six two-word coordinates cached by PATH_DS kernels (phase-0 first light and warm).
        size += warps * (108 if (not first or phase == 0) else 60)
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
        (persistent service, payload restore) must not mix SH orders or PROFILE node counts. N=161 adds no field."""
        key = (tuple(flags), W, width, int(first), int(phase), int(boundary), ('L', self.L))
        if self.profile_n != PROFILE_N_DEFAULT:
            key += (('PROFILE_N', self.profile_n),)
        if ground_sh_l:
            key += (('GROUND_SH_L', ground_sh_l),)
        if ray_limit:
            key += ('LAUNCH_RAY_LIMIT',)
        return (key, stage1) if stage1 else key

    def attribute_suffix(self):
        return '' if self.profile_n == PROFILE_N_DEFAULT else f'_profileN{self.profile_n}'

    def check_profile(self, optics, where, rows=None):
        """compiled N == declared optics N == table rows (== uploaded rows): density.check_consumed with this Kernel's
        PROFILE_N. Also on restored/persistent kernels whose setup() never ran (installed constants)."""
        from .density import check_consumed

        return check_consumed(optics, self.profile_n, rows, where)

    def check_upload(self, c, dc, where):
        """The uploaded PROFILE buffer still is the optics' table: N as check_profile, and the identity
        recorded at upload equals the current optics identity; a rebound table/mask/policy is E_PROFILE_STALE."""
        if 'density_optics' not in c:
            return
        from .density import profile_identity

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
        if key not in self.raw:
            code = kernel_source(self.L, combine=(self.config(W).c3_fw15 and not first))
            options = (
                '-std=c++17',
                '--fmad=true' if 'FMAD' in stage1 else '--fmad=false',
                '--prec-div=false' if 'FASTMATH' in stage1 else '--prec-div=true',
                '--prec-sqrt=false' if 'FASTMATH' in stage1 else '--prec-sqrt=true',
                f'-DFW={W}',
            )  # no --ftz: CuPy 14 RawKernel always adds -ftz=true (NVRTC 12.8.93 'defined more than once')
            options += (f'-DSOSFUSE2_DETERMINISTIC={int(self.deterministic)}',)
            options += tuple('-DSOSFUSE2_' + f for f in stage1 if f != 'FMAD')
            options += tuple(f'-D{f}' for f in flags)
            options += (f'-DK_FIRST={int(first)}', f'-DK_PHASE={int(phase)}', f'-DK_BOUNDARY={int(boundary)}')
            if ray_limit:
                options += ('-DLAUNCH_RAY_LIMIT',)
            if ground_sh_l:
                from .ground_sh import GRID_INTERVALS

                options += (f'-DGROUND_SH_L={ground_sh_l}', f'-DGROUND_SH_GRID={GRID_INTERVALS}')
            if self.profile_n != PROFILE_N_DEFAULT:
                options += (f'-DPROFILE_N={self.profile_n}',)
            with stage('raw_compile'):
                raw = self.xp.RawKernel(code, 'sos_fused', options=options, backend='nvrtc')
                raw.compile()
            self.__dict__.setdefault('raw_options', {})[key] = (code, options)
            required = self.shared_bytes(width, 0, first=first, variant=variant)
            props = self.xp.cuda.runtime.getDeviceProperties(self.xp.cuda.runtime.getDevice())
            maximum = int(props.get('sharedMemPerBlockOptin', props['sharedMemPerBlock']))
            if required > maximum:
                raise RuntimeError(
                    f'E_SHARED_MEMORY: need {required} bytes, device supports {maximum}; reduce --block-size'
                )
            raw.max_dynamic_shared_size_bytes = required
            raw.preferred_shared_memory_carveout = 100
            self.raw[key] = raw
            self.attributes[akey] = dict(
                raw.attributes,
                block_size=self.bs(first, variant) if variant else self.block_size,
                shared_bytes=required,
                compiler='NVRTC',
                implicit_fma='--fmad=true' in options,
                stage1=list(stage1),
                coordinate_precision='two-word FP32',
                flags=list(flags),
                **({} if self.profile_n == PROFILE_N_DEFAULT else dict(profile_n=self.profile_n)),
            )
            if ground_sh_l:
                self.attributes[akey]['ground_sh'] = dict(order=ground_sh_l, grid_intervals=GRID_INTERVALS)
        required = self.shared_bytes(width, 0, first=first, variant=variant)
        self.raw[key].max_dynamic_shared_size_bytes = required
        self.attributes[akey].update(
            block_size=self.bs(first, variant) if variant else self.block_size, shared_bytes=required
        )
        if not variant:
            self.attributes[str(W)] = self.attributes[akey]
        return self.raw[key]

    def fl_project_kernel(self, W, boundary=0, ground_sh_l=0):
        """kernel.cu fl_project from the same source and options as the ray-range first-light kernel."""
        stage1 = self.stage1_flags()
        key = self.kernel_key(
            self.config(W).defines,
            W,
            self.tile_width(W, True, 0),
            1,
            0,
            boundary,
            ground_sh_l=ground_sh_l,
            ray_limit=True,
            stage1=stage1,
        )  # same key as raw_kernel(..., ray_limit=True)
        pkey = key + ('fl_project',)
        if pkey not in self.raw:
            code, options = self.raw_options[key]
            with stage('raw_compile'):
                raw = self.xp.RawKernel(code, 'fl_project', options=options, backend='nvrtc')
                raw.compile()
            self.raw[pkey] = raw
        return self.raw[pkey]

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
                dc = {name: self.xp.asarray(np.ascontiguousarray(c[name])) for name in names}
                if 'strato_ext' in c:
                    for name in ('strato_ext', 'strato_sca', 'strato_chi', 'strato_density', 'strato_sun_col'):
                        dc[name] = self.xp.asarray(np.ascontiguousarray(c[name]))
                if 'strato_ext' in c and self.config().strato:
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
                        self.ground_sh_tables[ground_key] = synthesize(coeff, self.ground_sh_l, self.xp)
                    dc['ground_sh'] = self.ground_sh_tables[ground_key]
                    self.metrics['ground_sh_order'] = self.ground_sh_l
                    self.metrics['ground_sh_bytes'] = int(dc['ground_sh'].nbytes)
                if 'density_optics' in c:
                    from .density import upload_buffer, evidence

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
                dc.update(zip(('aa', 'bb', 'dd'), [self.xp.asarray(v) for v in norm_constants(self.L)]))
                # sos_vprod_y (sh_basis_ds) reads the same constants as (hi, lo) float pairs
                for name, v in zip(('aa2', 'bb2', 'dd2'), norm_constants(self.L)):
                    hi = v.astype('f4')
                    dc[name] = self.xp.asarray(np.stack([hi, (v - hi.astype('f8')).astype('f4')], 1))
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
        code = Path(__file__).with_name('kernel.cu').read_text()
        options = ('-std=c++17', '--fmad=false', '--prec-div=true', '--prec-sqrt=true', f'-DFW={W}')
        options += (f'-DSOSFUSE2_DETERMINISTIC={int(self.deterministic)}',)
        options += tuple('-D' + f for f in resolve_config(self.flags, role='lut', W=W).flags)
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
        from .stratosphere import columns

        profile = c['strato_profile']
        key = tuple(profile[k] for k in ('center', 'sigma', 'norm'))
        if not hasattr(bundle, 'strato_columns'):
            bundle.strato_columns = {}
        if key not in bundle.strato_columns:
            from .column_cache import cached_columns

            def build():
                starts = self.host(bundle.arrays['starts'])
                co = self.host(bundle.arrays['cell_off'])
                dirs = self.host(bundle.arrays['dirs'])
                out = np.empty(bundle.rays, 'f4')
                from .column_parallel import parallel_columns

                if parallel_columns(out, 'strato', starts, co, dirs, c['strato_old'], profile):
                    return out  # parallel preparation, rtol 1e-6
                # Do not generate full-grid [ray,GLnode,xyz] arrays on the host.
                for start in range(0, bundle.rays, 2048):
                    end = min(start + 2048, bundle.rays)
                    index = np.searchsorted(co, np.arange(start, end), side='right') - 1
                    out[start:end] = columns(starts[index], dirs[start:end], c['strato_old'], profile, 24)
                return out

            out = cached_columns(self, bundle, c, 'strato', build)
            bundle.strato_columns[key] = self.xp.asarray(out)
        return bundle.strato_columns[key]

    def density_end_columns(self, bundle, c):
        from .density import correct_columns
        from .density import profile_identity

        optics = c['density_optics']
        key = profile_identity(optics)['sha256']  # effective table, grid, mask, policy
        if not hasattr(bundle, 'density_columns'):
            bundle.density_columns = {}
        if key not in bundle.density_columns:
            from .column_cache import cached_columns

            def build():
                starts = self.host(bundle.arrays['starts'])
                co = self.host(bundle.arrays['cell_off'])
                dirs = self.host(bundle.arrays['dirs'])
                out = self.host(bundle.arrays['end_col']).copy()
                from .column_parallel import parallel_columns

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
            bundle.density_columns[key] = self.xp.asarray(np.ascontiguousarray(out, 'f4'))
        return bundle.density_columns[key]

    def launch(self, bundle, s, field, first=False, phase=0, boundary=None, timed=False, _ksplit=None):
        section_dir = None
        if self.config().instrumented:
            text = os.environ.get('SOSFUSE2_SECTION_CLOCKS_DIR', '').strip()
            if not text:
                raise ValueError('E_SECTION_CLOCK_OUTPUT: set SOSFUSE2_SECTION_CLOCKS_DIR')
            section_dir = Path(text).resolve()
            section_dir.mkdir(parents=True, exist_ok=True)
        xp = self.xp
        c = self.setup(s, bundle.nq)
        dc = self.constants_device(s, bundle.nq)
        W = s.W
        ncell = bundle.cells
        nr = bundle.rays
        ks = self.ksplit and not first and phase in (0, 1)
        if ks:
            if W != 15 or 'strato_ext' not in c:
                raise ValueError(
                    'E_KSPLIT_STATE: SOSFUSE2_KSPLIT=1 needs 15 wavelengths and an enabled stratosphere '
                    '(five separable columns plus the strato profile, kernel.cu SINV)'
                )
            kb = bundle.ksplit if _ksplit else self.ksplit_columns(bundle, s, field, phase, boundary)
            if _ksplit == 'record':
                # Unit-basis extinction: five profile columns, then the separate stratosphere column.
                e = xp.zeros(dc['ext'].size, dc['ext'].dtype)
                for k in range(NCOMP):
                    e[k * W + k] = 1
                se = xp.zeros(dc['strato_ext'].size, dc['strato_ext'].dtype)
                se[NCOMP] = 1
                dc = dict(dc, ext=e.reshape(dc['ext'].shape), strato_ext=se.reshape(dc['strato_ext'].shape))
            else:
                kb['hdr32'][0:1].fill(2)
        src = xp.ascontiguousarray(field, dtype='f4')
        moments = xp.empty((ncell, 4 + self.K, W), 'f8') if phase == 0 else xp.empty(1, 'f8')
        flux = xp.empty((ncell, W), 'f8')
        query = xp.empty((nr, W, 4), 'f4') if phase == 2 else (kb['buf'] if ks else xp.empty(1, 'f4'))
        bd = xp.ascontiguousarray(boundary, dtype='f8') if boundary is not None else xp.zeros((len(s.geo.chi), W), 'f8')
        names = (
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
        args = [
            np.int32(v)
            for v in (
                ncell,
                len(s.geo.h),
                len(s.geo.chi),
                bundle.nq,
                len(c['sun_h']),
                c['sun_col'].shape[1],
                int(first),
                phase,
                int(boundary is not None),
            )
        ]
        args += [bundle.arrays[name] for name in names]
        if 'density_optics' in c:
            args[9 + names.index('end_col')] = self.density_end_columns(bundle, c)
        args += [
            dc[name]
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
        ground_sh_l = self.ground_sh_l if first and phase in (0, 2) and 'ground_sh' in dc else 0
        if ground_sh_l:
            # Only the SH specialization interprets gval as its synthesized
            # convolution table. Empty A0 uses the
            # original binary, argument arrays, and shared-memory size.
            args[-1] = dc['ground_sh']
        args += [c['height'], c['g'], c['mix'], np.float64(c['albedo']), src, bd, moments, flux, query]
        sinv_index = len(args) if ks else None
        if ks:
            args += self.ksplit_pointer_args(kb)
        sh_index = len(args)  # the SH sidecar appended below (VPROD reads the same combined plane)
        with stage('sh_aux_build', device=True):
            if 'strato_ext' in c and self.config().strato:
                if src.shape != (len(s.geo.h) * len(s.geo.chi), 10 + 2 * self.K, W):
                    raise ValueError(f'E_STRATO_SH_SOURCE: expected [nh*nc,{10 + 2 * self.K},W]')
                if self.config().c3_fw15 and not first:
                    args.append(combined_aux(src, c, self.L, xp, skip_thin=True))
                else:
                    from .strato_parallel import strato_sh_aux

                    args.append(strato_sh_aux(src, xp, pair=True, L=self.L))
            else:
                args.append(sh_aux_pair(src, xp, self.L))
        if 'strato_ext' in c:
            # No optical sidecar is persisted in legacy/common endpoint files.
            endst = self.strato_end_columns(bundle, c)
            profile = c['strato_profile']
            args += [
                dc[name] for name in ('strato_ext', 'strato_sca', 'strato_chi', 'strato_density', 'strato_sun_col')
            ]
            args += [endst, F(profile['center']), F(profile['sigma']), F(profile['norm'])]
        self.check_upload(c, dc, 'launch')  # N and upload identity (E_PROFILE_N_MISMATCH / E_PROFILE_STALE)
        args += [
            dc.get('profile_table', np.uint64(0)),
            np.int32(c.get('density_optics', {}).get('mask', 0)),
            F(c.get('density_optics', {}).get('air_surface', 1.0)),
        ]
        lpt_index = None
        split_count = None
        if not first and not (self.config().split_warm and phase == 0):
            # R4 LPT_ORDER: per-launch cell order by descending segment count (block work proxy).
            S = 2 if self.config().c3_fw15 else 1

            def lpt_order(c0, c1):
                co = bundle.arrays['cell_off'][c0 : c1 + 1]
                eo = bundle.arrays['edge_off']
                work = xp.diff(eo[co]) - xp.diff(co)
                if S > 1:  # units = cell*S+part, parts of a cell adjacent, cells by descending work
                    o = xp.argsort(-work).astype('i4')
                    return xp.ascontiguousarray((o[:, None] * S + xp.arange(S, dtype='i4')[None, :]).reshape(-1))
                return xp.ascontiguousarray(xp.argsort(-work).astype('i4'))

            lpt_index = len(args)
            args.append(lpt_order(0, ncell))
            if S > 1:
                args.append(xp.empty(max(1, (S - 1) * ncell * (4 + self.K) * W) if phase == 0 else 1, 'f8'))
                args.append(xp.empty(S * ncell * W, 'f8'))
                split_count = xp.zeros(ncell, 'i4')
                args.append(split_count)
        section_buffer = None
        if section_dir is not None:
            section_buffer = xp.zeros(
                (ncell, W // self.tile_width(W, first, phase), self.bs(first) // 32, len(SECTION_CLOCK_NAMES)),
                dtype='uint64',
            )
            args.append(section_buffer)
        mode, step = self.launch_mode(ncell)
        if ks and mode == 'single':
            mode, step = 'fixed', ncell  # KSPLIT always goes through the chunk loop
        ray_sync = None
        ray_mode = False
        if phase == 0 and ((self.config().split_first and first) or (self.config().split_warm and not first)):
            # FL_RAYSPLIT private scratch: per-ray rad rows and [ray queue, per-cell done] counters (zeroed).
            # Cell-subset launches (eda): the kernel reads rays cell_off[0]..cell_off[cells] and indexes ray_rads
            # and the counters chunk-locally, so the counters are re-zeroed before every chunk.
            # WARM_RAYSPLIT is not ported to cell-subset launches.
            if mode != 'single' and (not first):
                raise ValueError('E_FL_RAYSPLIT_LAUNCH_CELLS: WARM_RAYSPLIT needs one launch per cell grid')
            # Split FL_RAYSPLIT first light goes by ray ranges over the whole cell grid (eda: 68 cells of first light
            # took 6.3 s, and fewer cells leave SMs idle because a block is a cell).
            # SOSFUSE2_FL_PROJECT=1 also runs one-launch first light as ray launch + fl_project (rig experiment: the
            # single-warp projections may dominate its 11.9 s first light on a 4090; not measured yet).
            ray_mode = mode != 'single' or os.environ.get('SOSFUSE2_FL_PROJECT', '').strip() == '1'
            ray_sync = xp.zeros(ncell + 2, 'i4')  # [queue head, cell counters, ray-range end (LAUNCH_RAY_LIMIT)]
            ray_rads = xp.empty((max(nr, 1), 4 * W), 'f4')
            args += [ray_rads, ray_sync]
        rs = bool(self.device()[1].get('requires_split'))
        if first and phase == 0 and (not ray_mode):
            err = first_light_split_error(self.flags, 'strato_ext' in c, self.device()[0], self.device()[1])
            if err:
                raise RuntimeError(err)
        variant = _ksplit or ('prod' if ks else None)
        raws = (
            [
                ('_prod', self.raw_kernel(W, first, phase, boundary is not None, variant='prod')),
                ('_slim', self.raw_kernel(W, first, phase, boundary is not None, variant='slim')),
            ]
            if ks and not _ksplit
            else [
                (
                    '',
                    self.raw_kernel(
                        W,
                        first,
                        phase,
                        boundary is not None,
                        ray_limit=ray_mode,
                        variant=_ksplit,
                        ground_sh_l=ground_sh_l,
                    ),
                )
            ]
        )
        raw = raws[0][1]
        if timed or section_buffer is not None:
            begin = xp.cuda.Event()
            end = xp.cuda.Event()
            begin.record()
        name = (
            'launch_first'
            if first
            else ('launch_query' if phase == 2 else 'launch_ground' if phase == 1 else 'launch_warm')
        )
        with stage(name, device=True):
            # Cell-subset launches (eda thermal/watchdog cap; device profile, device_profile.py): blocks are
            # independent, so only the cell-indexed pointers move: cell_off/starts/targets (inputs) and
            # moments/flux (outputs); per-chunk counters (FL_RAYSPLIT ray_sync, CELL_SPLIT split_count) are
            # re-zeroed. Numerically equivalent to one launch within gate tolerance. 'adaptive' sizes each chunk from the last
            # measured seconds per unit of work of the same kernel kind; 'fixed' is SOSFUSE2_LAUNCH_CELLS.
            nsplit = (2 if self.config().c3_fw15 else 1) if lpt_index is not None else 1
            grid_y = W // self.tile_width(W, first, phase)
            block = (self.bs(first, variant),)
            shared = self.shared_bytes(W, phase, first=first, variant=variant)
            kind = f'first{int(first)}_phase{int(phase)}_W{W}' + ('_record' if _ksplit else '_split' if ks else '')
            report = None
            if mode == 'single' and not ray_mode:
                seconds = self.timed_launch(kind, raw, (ncell * nsplit, grid_y), block, tuple(args), shared, ncell)
                report = dict(chunks=1, gpu_seconds=seconds, max_seconds=seconds, max_cells=ncell)
            elif ray_mode:
                cum = self.ray_work(bundle)
                nrays = len(cum) - 1
                kind += '_rays'
                # Ray ranges start (and are floored) at half the rays of the initial cells: first-light rays cost about
                # 27 us each on the 3080 next to another GPU job (eda r4), so 24 cells' rays would be about 4 s.
                initial = max(
                    1,
                    (step if mode == 'fixed' else self.device()[1]['launch_cells_initial'] or 1)
                    * nrays
                    // max(ncell, 1)
                    // (1 if mode == 'fixed' else 2),
                )
                report = dict(chunks=0, gpu_seconds=0.0, max_seconds=0.0, max_cells=ncell, max_rays=0)
                q0 = 0
                while q0 < nrays:
                    q1 = (
                        nrays
                        if mode == 'single'
                        else min(nrays, q0 + initial)
                        if (mode == 'fixed' and not rs)
                        else self.next_chunk_end(kind, q0, nrays, initial if mode == 'fixed' else None, cum, initial)
                    )
                    ray_sync[0:1].fill(q0)
                    ray_sync[ncell + 1 : ncell + 2].fill(q1)
                    seconds = self.timed_launch(
                        kind, raw, (ncell, grid_y), block, tuple(args), shared, q1 - q0, cum, q0, q1
                    )
                    report['chunks'] += 1
                    report['gpu_seconds'] += seconds
                    report['max_seconds'] = max(report['max_seconds'], seconds)
                    report['max_rays'] = max(report['max_rays'], q1 - q0)
                    q0 = q1
                # With LAUNCH_RAY_LIMIT the ray launches only publish ray_rads; project the cells now (fl_project,
                # one block per cell, FP32 ray batches combined in FP64).
                proj = self.fl_project_kernel(W, boundary is not None, ground_sh_l=ground_sh_l)
                pkind = kind[:-5] + '_project'
                co_h = np.asarray(self.host(bundle.arrays['cell_off']), 'i8')
                pcum = (co_h - co_h[0]).astype('f8')
                ci = 9 + names.index('cell_off')
                ti = 9 + names.index('targets')
                fixed_args = (
                    args[9 + names.index('dirs')],
                    args[9 + names.index('weights')],
                    dc['aa'],
                    dc['bb'],
                    dc['dd'],
                    ray_rads,
                    np.float64(c['albedo']),
                )
                c0 = 0
                while c0 < ncell:
                    c1 = (
                        ncell
                        if mode == 'single'
                        else min(ncell, c0 + step)
                        if (mode == 'fixed' and not rs)
                        else self.next_chunk_end(pkind, c0, ncell, step if mode == 'fixed' else None, pcum)
                    )
                    seconds = self.timed_launch(
                        pkind,
                        proj,
                        (c1 - c0,),
                        (256,),
                        (np.int32(c1 - c0), np.int64(co_h[0]), args[ci][c0:], args[ti][c0:])
                        + fixed_args
                        + (moments[c0:c1], flux[c0:c1]),
                        0,
                        c1 - c0,
                        pcum,
                        c0,
                        c1,
                    )
                    report['chunks'] += 1
                    report['gpu_seconds'] += seconds
                    report['max_seconds'] = max(report['max_seconds'], seconds)
                    c0 = c1
            else:
                cum = self.cell_work(bundle) if (mode == 'adaptive' or rs) else None
                cell_index = {n: 9 + names.index(n) for n in ('cell_off', 'starts', 'targets')}
                mi = next(i for i, a in enumerate(args) if a is moments)
                fi = next(i for i, a in enumerate(args) if a is flux)
                report = dict(chunks=0, gpu_seconds=0.0, max_seconds=0.0, max_cells=0)
                c0 = ncell if (ks and not _ksplit and self.ksplit_mode in (2, 3)) else 0
                if c0:

                    def cell_part(a, b):
                        part = list(args)
                        part[0] = np.int32(b - a)
                        part[sinv_index : sinv_index + 3] = self.ksplit_pointer_args(kb)
                        for n, i in cell_index.items():
                            part[i] = args[i][a:]
                        if phase == 0:
                            part[mi] = moments[a:b]
                        part[fi] = flux[a:b]
                        if lpt_index is not None:
                            part[lpt_index] = lpt_order(a, b)
                        if split_count is not None:
                            split_count.fill(0)
                        return tuple(part)

                    ctx = dict(
                        args=args,
                        sinv_index=sinv_index,
                        lpt_index=lpt_index,
                        cell_index=cell_index,
                        mi=mi,
                        fi=fi,
                        moments=moments,
                        flux=flux,
                        albedo=c['albedo'],
                        ray_cum=self.ray_work(bundle) if cum is not None else None,
                    )
                    self.vprod_launch(
                        kb,
                        bundle,
                        s,
                        dc,
                        args[sh_index],
                        W,
                        phase,
                        boundary is not None,
                        kind,
                        raws,
                        cell_part,
                        ((nsplit, grid_y), block, shared),
                        step,
                        cum,
                        report,
                        ctx,
                    )
                while c0 < ncell:
                    c1 = self.next_chunk_end(kind + raws[0][0], c0, ncell, step, cum)
                    if ks and not _ksplit:
                        c1 = self.ksplit_chunk(kb, c0, c1)
                    part = list(args)
                    part[0] = np.int32(c1 - c0)
                    if ks:
                        part[sinv_index : sinv_index + 3] = self.ksplit_pointer_args(kb)
                    for n, i in cell_index.items():
                        part[i] = args[i][c0:]
                    if phase == 0:
                        part[mi] = moments[c0:c1]
                    part[fi] = flux[c0:c1]
                    if section_buffer is not None:
                        part[-1] = section_buffer[c0:c1]
                    if lpt_index is not None:
                        part[lpt_index] = lpt_order(c0, c1)
                    if ray_sync is not None and c0 > 0:
                        ray_sync.fill(0)
                    if split_count is not None and c0 > 0:
                        split_count.fill(0)
                    for suffix, r in raws:  # KSPLIT: producer then consumer on the same chunk (one stream, in order)
                        seconds = self.timed_launch(
                            kind + suffix,
                            r,
                            ((c1 - c0) * nsplit, grid_y),
                            block,
                            tuple(part),
                            shared,
                            c1 - c0,
                            cum,
                            c0,
                            c1,
                        )
                        report['gpu_seconds'] += seconds
                        report['max_seconds'] = max(report['max_seconds'], seconds)
                    report['chunks'] += 1
                    report['max_cells'] = max(report['max_cells'], c1 - c0)
                    c0 = c1
            if report is not None:
                report.update(
                    kind=kind,
                    mode=mode,
                    cells=ncell,
                    profile=self.launch_log['profile'],
                    budget_seconds=self.launch_log['budget_seconds'],
                )
                print('LAUNCH_REPORT ' + json.dumps(report, allow_nan=False), flush=True)
        if timed or section_buffer is not None:
            end.record()
            end.synchronize()
            cuda_seconds = float(xp.cuda.get_elapsed_time(begin, end) / 1000)
            if timed:
                self.cuda_events.append(cuda_seconds)
        if section_buffer is not None:
            counters = self.host(section_buffer)
            report = section_clock_summary(counters)
            path = section_dir / f'clock_{os.getpid()}_{time.time_ns()}_{self.metrics["launches"]}.npz'
            with path.open('xb') as stream:
                np.savez_compressed(stream, cycles=counters)
            report.update(
                schema='sosfuse2-section-clocks-v1',
                flags=list(self.flags),
                gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                pid=os.getpid(),
                launch=self.metrics['launches'],
                first=bool(first),
                phase=int(phase),
                wavelengths=W,
                block_size=self.block_size,
                grid=[ncell, W // self.tile_width(W, first, phase)],
                launch_cells=self.launch_cells(),
                cuda_event_seconds=cuda_seconds,
                raw_path=str(path),
            )
            print('SECTION_CLOCKS ' + json.dumps(report, allow_nan=False), flush=True)
        self.metrics['launches'] += 1
        if phase == 2:
            self.metrics['query_launches'] += 1
        self.memory()
        return moments, flux, query

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
            value = reference.transport(geometry, c, packed, first, statistics=self.metrics)
        else:
            from .geometry import Bundle

            bundle = Bundle([geometry], self.xp)
            _, _, value = self.launch(bundle, s, self.xp.asarray(packed), first, 2)
            value = self.host(value)
        out = np.zeros((len(value), s.W, 2, 4), 'f8')
        out[:, :, 0] = value
        return out


class Solver:
    def __init__(
        self, accepted, geometry, fixed_boxes=False, box_group=32, prepared=False, observer=None, box_evaluate=None
    ):
        from .stratosphere import attach

        if not prepared:
            attach(accepted, accepted.ctx)
        self.s = accepted
        self.geometry = geometry
        self.kernel = accepted.kernel
        self.xp = self.kernel.xp
        self.first_moments = None
        self.first_flux = None
        self.observer = observer
        self.box_evaluate = box_evaluate
        # A5: opt-in observer path (fixed captured rays, device source read directly).
        self.fixed_boxes = fixed_boxes
        self.box_group = box_group
        self._box_rays = None
        self.timings = dict(apply_seconds=[], gpu_kernel_seconds=[], setup_seconds=0.0)

    def initialize(self):
        if self.first_moments is not None:
            return
        tt = time.perf_counter()
        s = self.s
        xp = self.xp
        with stage('first_light_total'):
            zero = xp.zeros((s.geo.ns, source_size(s), s.W), 'f4')
            first_moments, first_flux = self.geometry.evaluate(s, zero, first=True, phase=0)
            self.kernel.sync()
        del zero
        if xp is not np:
            if self.geometry.bundle is None:
                if hasattr(self.geometry, 'prepare'):
                    self.geometry.prepare()
                else:
                    from .geometry import Bundle

                    self.geometry.bundle = Bundle([self.geometry.cell(i) for i in range(s.geo.ns)], xp)
            if self.geometry.ground_bundle is None:
                from .geometry import Bundle

                self.geometry.ground_bundle = Bundle([self.geometry.cell(i) for i in range(len(s.geo.chi))], xp)
            bundles = [(self.geometry.ground_bundle, 1), (self.geometry.bundle, 0)]
            from .pipeline_execution import prepare_bundle

            for bundle, _ in bundles:
                prepare_bundle(self.kernel, s, bundle)
                self.kernel.constants_device(s, bundle.nq)
            self.timings['ksplit'] = self.kernel.ksplit_preflight(s, bundles)
        self.first_moments, self.first_flux = first_moments, first_flux
        self.timings['setup_seconds'] = time.perf_counter() - tt

    def apply(self, field):
        self.initialize()
        tt = time.perf_counter()
        s = self.s
        xp = self.xp
        with stage('apply_ground'):
            _, flux = self.geometry.evaluate(s, field, phase=1, ground_only=True)
        boundary = flux + self.first_flux[: len(s.geo.chi)]
        with stage('apply_warm'):
            moment, _ = self.geometry.evaluate(s, field, phase=0, boundary=boundary)
        moment += self.first_moments
        if os.environ.get('SOSFUSE2_EXPORT_MOMENT') == '1':
            self.last_moment = (
                moment  # Opt-in: incident-radiance SH moments of this evaluation
            )
        out = xp.empty_like(field)
        out[:, :4] = (moment[:, :4] * xp.asarray(F(3 / (8 * math.pi)) * s.ext[0].astype('f4'))[None, None, :]).astype(
            'f4'
        )
        out[:, 4:10] = (
            moment[:, 4:10]
            * xp.asarray(s.rf.astype('f4'))[None, :, None]
            * xp.asarray(s.ext[0].astype('f4'))[None, None, :]
        ).astype('f4')
        if getattr(s, 'tropo', None) is not None:
            from .tropo_mie import eigen_by_sh as tropo_eigen

            out[:, 10 : 10 + self.kernel.K] = (
                moment[:, 4:]
                * xp.asarray(tropo_eigen(s.tropo, self.kernel.L).astype('f4'))[None, :, :]
                * xp.asarray(s.asca.astype('f4'))[None, None, :]
            ).astype('f4')
        else:
            out[:, 10 : 10 + self.kernel.K] = (
                moment[:, 4:]
                * xp.asarray(s.af.astype('f4'))[None, :, None]
                * xp.asarray(s.asca.astype('f4'))[None, None, :]
            ).astype('f4')
        if getattr(s, 'strato', None) is not None:
            from .stratosphere import eigen_by_sh

            out[:, 10 + self.kernel.K :] = (
                moment[:, 4:]
                * xp.asarray(eigen_by_sh(s.strato, self.kernel.L).astype('f4'))[None, :, :]
                * xp.asarray(s.strato['sca'].astype('f4'))[None, None, :]
            ).astype('f4')
        if xp is np:
            out[np.abs(out) < reference.TRACE_MIN] = (
                0  # FP32 flush-to-zero of the device composition (no subnormal source on CUDA, reference.flush_to_zero)
            )
        self.kernel.sync()
        self.kernel.memory()
        self.timings['apply_seconds'].append(time.perf_counter() - tt)
        return out

    def boxes(self, field):
        with stage('observer'):
            full = np.zeros((self.s.geo.ns, field.shape[1], 2 * self.s.W), 'f4')
            full[..., ::2] = self.kernel.host(field)
            return self.s.boxes(full)[..., 0].reshape(5, 7, self.s.W)

    def boxes_fixed(self, field):
        """A5: same 35 boxes as boxes(); rays are captured once from the accepted
        s.boxes, packed into one fixed Bundle, and every call launches phase 2 on
        the device field (no host round trip of the source), then replays the
        accepted reshape/quadrature/solar factor with the returned ray values."""
        from .table_fast import _bundle, _evaluate

        s = self.s
        kernel = self.kernel
        dummy = np.empty(0, 'f4')
        with stage('observer_fixed'):
            if self._box_rays is None:
                raws = []

                def record(raw, solver, _field, first=False):
                    raws.append(raw)
                    return np.zeros((len(raw['d']), solver.W, 2, 4), 'f8')

                kernel.transport = record
                try:
                    s.boxes(dummy, False)
                finally:
                    del kernel.transport
                self._box_rays = (raws, _bundle(s, raws, self.box_group))
            raws, bundle = self._box_rays
            values = (self.box_evaluate or _evaluate)(s, raws, field, False, self.box_group, bundle)
            index = iter(range(len(raws)))
            pack = s.geo.pack

            def token(p, d, n):
                i = next(index)
                if len(d) != len(raws[i]['d']) or not np.array_equal(d, raws[i]['d']):
                    raise RuntimeError('E_REPLAY_DRIFT: observer ray order changed')
                return i

            def fetch(i, solver, _field, first=False):
                if first:
                    raise RuntimeError('E_REPLAY_FLAG')
                out = np.zeros((len(values[i]), solver.W, 2, 4), 'f8')
                out[:, :, 0] = values[i]
                return out

            s.geo.pack = token
            kernel.transport = fetch
            try:
                result = s.boxes(dummy, False)
            finally:
                s.geo.pack = pack
                del kernel.transport
            if next(index, None) is not None:
                raise RuntimeError('E_REPLAY_COUNT')
            return result[..., 0].reshape(5, 7, s.W)

    def solve(self, max_eval=16, tolerance=0.005):
        """Same Anderson4 normalization, regularization and candidate guard."""
        s = self.s
        xp = self.xp
        x = xp.zeros((s.geo.ns, source_size(s), s.W), 'f4')
        gs = [[] for _ in range(s.W)]
        fs = [[] for _ in range(s.W)]
        done = np.zeros(s.W, bool)
        history = []
        started = time.perf_counter()
        boxes = self.observer or (self.boxes_fixed if self.fixed_boxes else self.boxes)
        for iteration in range(1, max_eval + 1):
            evaluation_start = time.perf_counter()
            old = boxes(x)
            old_query_seconds = time.perf_counter() - evaluation_start
            apply_start = time.perf_counter()
            g = self.apply(x)
            apply_seconds = time.perf_counter() - apply_start
            # A NaN trace takes the kernel's linear path and propagates instead of vanishing: count and stop.
            nonfinite = int(xp.count_nonzero(~xp.isfinite(g)))
            if nonfinite:
                raise RuntimeError(f'E_NONFINITE_SOURCE: {nonfinite} values after evaluation {iteration}')
            g = xp.where(xp.asarray(done)[None, None, :], x, g)
            query_start = time.perf_counter()
            new = boxes(g)
            query_seconds = old_query_seconds + time.perf_counter() - query_start
            rr = np.max(abs(new - old) / np.maximum(abs(new), 1e-30), axis=(0, 1))
            ss = self.kernel.host(xp.max(abs(g - x), axis=(0, 1)) / xp.maximum(xp.max(abs(g), axis=(0, 1)), 1e-30))
            if iteration >= 2:
                done |= np.maximum(rr, ss) <= tolerance
            history.append(
                dict(
                    evaluation=iteration,
                    observer_relative=rr.tolist(),
                    source_relative=ss.tolist(),
                    converged=done.tolist(),
                    apply_seconds=apply_seconds,
                    observer_seconds=query_seconds,
                    seconds=time.perf_counter() - started,
                )
            )
            print(
                f'SOSFUSE EVAL {iteration} obs={max(rr):.6g} source={max(ss):.6g} done={done.sum()}/{s.W}', flush=True
            )
            if done.all():
                x = g
                history[-1]['evaluation_seconds'] = time.perf_counter() - evaluation_start
                history[-1]['Anderson_seconds'] = 0.0
                break
            anderson_start = time.perf_counter()
            nxt = g.copy()  # A1: Anderson time recorded below as stage 'anderson'
            for w in range(s.W):
                if done[w]:
                    continue
                norm = xp.linalg.norm(g[:, :, w], axis=1)
                scale = xp.maximum(norm, norm.max() * 1e-8)
                feature = ((g[:, :, w] - x[:, :, w]) / scale[:, None]).astype('f8').ravel()
                obs = xp.asarray(
                    ((new[:, :, w] - old[:, :, w]) / np.maximum(abs(new[:, :, w]), 1e-20)).ravel()
                    * math.sqrt(len(feature) / 35)
                )
                gs[w].append(g[:, :, w].copy())
                fs[w].append(xp.concatenate([feature, obs]))
                if len(fs[w]) > 4:
                    gs[w].pop(0)
                    fs[w].pop(0)
                if len(fs[w]) > 1:
                    f = xp.stack(fs[w], 1)
                    gram = f.T @ f
                    gram += xp.eye(len(gram)) * max(float(self.kernel.host(xp.trace(gram))) * 1e-10, 1e-30)
                    self.kernel.memory()
                    try:
                        alpha = xp.linalg.solve(gram, xp.ones(len(gram)))
                        alpha /= alpha.sum()
                        ah = self.kernel.host(alpha)
                        if np.isfinite(ah).all() and max(abs(ah)) < 8:
                            nxt[:, :, w] = sum(c * q for c, q in zip(alpha, gs[w]))
                    except np.linalg.LinAlgError:
                        pass
            # A converged wavelength is frozen exactly as the accepted active mask.
            x = xp.where(xp.asarray(done)[None, None, :], g, nxt)
            self.kernel.sync()
            history[-1]['Anderson_seconds'] = time.perf_counter() - anderson_start
            TIMERS.add('anderson', history[-1]['Anderson_seconds'])
            history[-1]['evaluation_seconds'] = time.perf_counter() - evaluation_start
        self.kernel.sync()
        if not done.all():
            raise RuntimeError(f'E_NOT_CONVERGED: {done.sum()}/{s.W} within {tolerance}')
        return x, dict(
            seconds=time.perf_counter() - started,
            evaluations=iteration,
            tolerance=tolerance,
            converged=done.tolist(),
            history=history,
        )
