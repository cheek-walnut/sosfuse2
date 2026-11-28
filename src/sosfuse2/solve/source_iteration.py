"""Fixed-point iteration of the atmospheric source field."""

import math
import os
import time
import warnings

import numpy as np

from ..backends import numpy_reference
from sosfuse2.backends import vram_trace
from ..backends.arithmetic import F
from ..backends.sh_layout import source_size
from ..foundation.timers import TIMERS, stage
from ..foundation import top


def _use_gpu_history(host_history, on_device, deterministic, strato, order, waves, features, free):
    # Stage one wavelength's residual ring and FP32 mixing vectors. Preserve
    # the pre-existing stratospheric dispatch; new orders apply to clear transport.
    supported = (order in (16, 24, 32) and waves in (1, 15)) if not strato else (order, waves) == (24, 15)
    return host_history and on_device and supported and not deterministic and 32 * features + (32 << 20) < free


def _anderson_weights(products):
    """Solve the small regularized Gram system on the host before rounding the weights."""
    gram = np.array(products, dtype=np.float64, copy=True)
    gram[np.diag_indices_from(gram)] += max(float(np.trace(gram)) * 1e-10, 1e-30)
    weights = np.linalg.solve(gram, np.ones(len(gram)))
    weights /= weights.sum()
    if np.isfinite(weights).all() and np.max(np.abs(weights)) < 8:
        return weights.astype(np.float32)
    return None


def _history_row(xp, shape, dtype, pin):
    if pin:
        # The persistent worker solves several spectral groups. Own these large
        # pages directly so dropping the history releases them to the OS.
        cuda = xp.cuda
        try:
            nbytes = math.prod(shape) * np.dtype(dtype).itemsize
            pointer = cuda.PinnedMemoryPointer(cuda.PinnedMemory(nbytes), 0)
            return np.ndarray(shape, dtype=dtype, buffer=pointer)
        except (MemoryError, xp.cuda.runtime.CUDARuntimeError) as exc:
            if not isinstance(exc, MemoryError) and exc.status != 2:  # cudaErrorMemoryAllocation
                raise
            warnings.warn(f'Pinned Anderson allocation failed ({exc}), using pageable host memory', RuntimeWarning)
    return np.empty(shape, dtype=dtype)


_SOURCE_METRICS = r'''
extern "C" __global__ void source_metrics(const float *g, const float *x, long long rows,
                                          unsigned frozen, unsigned *partials) {
    __shared__ unsigned maxima[2][32];
    __shared__ unsigned invalid[32];
    const unsigned w = threadIdx.x;
    const unsigned waves = blockDim.x;
    if (threadIdx.y == 0) {
        maxima[0][w] = maxima[1][w] = invalid[w] = 0;
    }
    __syncthreads();
    unsigned bad = 0, magnitude = 0, difference = 0;
    const long long start = (long long)blockIdx.x * 4096;
    for (long long row = start + threadIdx.y; row < rows && row < start + 4096; row += blockDim.y) {
        const long long i = row * waves + w;
        const float current = g[i], old = x[i];
        bad += !isfinite(current);
        const float value = (frozen & (1u << w)) ? old : current;
        // Positive IEEE words have numeric order, with NaNs above infinity.
        magnitude = max(magnitude, __float_as_uint(fabsf(value)));
        difference = max(difference, __float_as_uint(fabsf(value - old)));
    }
    atomicAdd(&invalid[w], bad);
    atomicMax(&maxima[0][w], magnitude);
    atomicMax(&maxima[1][w], difference);
    __syncthreads();
    if (threadIdx.y == 0) {
        unsigned *out = partials + blockIdx.x * 3 * waves + w;
        out[0] = invalid[w];
        out[waves] = maxima[0][w];
        out[2 * waves] = maxima[1][w];
    }
}
'''


def _device_source_metrics(xp, reduce, g, x, done):
    waves = len(done)
    rows = g.size // waves
    blocks = (rows + 4095) // 4096
    partials = xp.empty((blocks, 3, waves), 'u4')
    frozen = sum(1 << int(w) for w in np.flatnonzero(done))
    reduce((blocks,), (waves, 256 // waves), (g, x, np.int64(rows), np.uint32(frozen), partials))
    parts = xp.asnumpy(partials)
    magnitude, difference = parts[:, 1:].max(axis=0).view('f4')
    return int(parts[:, 0].sum()), difference / np.maximum(magnitude, np.float32(1e-30))


class Solver:
    def __init__(
        self, accepted, geometry, fixed_boxes=False, box_group=32, prepared=False, observer=None, box_evaluate=None
    ):
        from ..atmosphere.stratosphere import attach

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
        # Opt-in observer path: fixed captured rays that read the device source directly.
        self.fixed_boxes = fixed_boxes
        self.box_group = box_group
        self._box_rays = None
        self.monitor_sets = getattr(self.kernel, 'domain', top.Domain()).monitors()
        self.timings = dict(apply_seconds=[], gpu_kernel_seconds=[], setup_seconds=0.0)

    def initialize(self):
        if self.first_moments is not None:
            return
        tt = time.perf_counter()
        s = self.s
        xp = self.xp
        with stage('first_light_total'):
            with vram_trace.item('source.first_zero'):
                zero = xp.zeros((s.geo.ns, source_size(s), s.W), 'f4')
            first_moments, first_flux = self.geometry.evaluate(s, zero, first=True, phase=0)
            self.kernel.sync()
        del zero
        if xp is not np:
            xp.get_default_memory_pool().free_all_blocks()
        bundles = []
        if xp is not np:
            if self.geometry.bundle is None:
                if hasattr(self.geometry, 'prepare'):
                    self.geometry.prepare()
                else:
                    from .ray_geometry import Bundle

                    self.geometry.bundle = Bundle([self.geometry.cell(i) for i in range(s.geo.ns)], xp)
            if self.geometry.ground_bundle is None:
                from .ray_geometry import Bundle

                self.geometry.ground_bundle = Bundle([self.geometry.cell(i) for i in range(len(s.geo.chi))], xp)
            bundles = [(self.geometry.ground_bundle, 1), (self.geometry.bundle, 0)]
            from ..jobs.pipeline_execution import prepare_bundle

            for bundle, _ in bundles:
                prepare_bundle(self.kernel, s, bundle)
                self.kernel.constants_device(s, bundle.nq)
        if xp is not np or not self.kernel.ksplit:
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
        with vram_trace.item('source.output'):
            out = xp.empty_like(field)
        with vram_trace.item('source.composition_scratch'):
            out[:, :4] = (moment[:, :4] * xp.asarray(F(3 / (8 * math.pi)) * s.ext[0].astype('f4'))[None, None, :]).astype(
                'f4'
            )
            out[:, 4:10] = (
                moment[:, 4:10]
                * xp.asarray(s.rf.astype('f4'))[None, :, None]
                * xp.asarray(s.ext[0].astype('f4'))[None, None, :]
            ).astype('f4')
            if getattr(s, 'tropo', None) is not None:
                from ..atmosphere.tropo_mie import eigen_by_sh as tropo_eigen

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
                from ..atmosphere.stratosphere import eigen_by_sh

                out[:, 10 + self.kernel.K :] = (
                    moment[:, 4:]
                    * xp.asarray(eigen_by_sh(s.strato, self.kernel.L).astype('f4'))[None, :, :]
                    * xp.asarray(s.strato['sca'].astype('f4'))[None, None, :]
                ).astype('f4')
        if xp is np:
            out[np.abs(out) < numpy_reference.TRACE_MIN] = (
                0  # FP32 flush-to-zero of the device composition (no subnormal source on CUDA, numpy_reference.flush_to_zero)
            )
        self.kernel.sync()
        self.kernel.memory()
        self.timings['apply_seconds'].append(time.perf_counter() - tt)
        return out

    def boxes(self, field):
        with stage('observer'):
            full = np.zeros((self.s.geo.ns, field.shape[1], 2 * self.s.W), 'f4')
            full[..., ::2] = self.kernel.host(field)
            return self.s.boxes(full)[..., 0].reshape(-1, 7, self.s.W)

    def boxes_fixed(self, field):
        """Return (solar elevation, seven views, wavelength) as in boxes().

        The default domain uses 5 × 7 stopping boxes. The extended domain uses
        6 × 7 stopping boxes followed by 1 × 7 recorded for diagnostics. Rays are captured once from the accepted
        s.boxes, packed into one fixed Bundle, and every call launches phase 2 on
        the device field (no host round trip of the source), then replays the
        accepted reshape/quadrature/solar factor with the returned ray values."""
        from ..outputs.radiance_table import _bundle, _evaluate

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
            return result[..., 0].reshape(-1, 7, s.W)

    def ground_sh_order(self):
        """Order of the ground convolution table in this state's device constants, which the first-light launches
        use; 0 without one. Only CUDA builds device constants; NumPy convolves the ground exactly."""
        constants = getattr(self.kernel, 'device_constants', {})
        used = any('ground_sh' in dc for (state, _), dc in constants.items() if state == id(self.s))
        return getattr(self.kernel, 'ground_sh_l', 0) if used else 0

    def _apply_vram_reserve(self, x):
        """Conservative device-memory reserve for warm transport and apply outputs."""
        s = self.s
        h = getattr(s.geo, 'h', None)
        chi = getattr(s.geo, 'chi', None)
        cells = len(h) * len(chi) if (h is not None and chi is not None) else s.geo.ns
        L = getattr(self.kernel, 'L', 32)
        K = getattr(self.kernel, 'K', (L + 1) * (L + 2) // 2)
        stride = 2 * ((s.W + 1) // 2)
        nodes = (10 * L + 2) // 3 + 3
        bundle = getattr(self.geometry, 'bundle', None)
        c = getattr(self.kernel, 'constants', {}).get((id(s), getattr(bundle, 'nq', 0)))
        if c is not None and h is not None and chi is not None:
            from ..backends.sh_layout import live_sh_cells as _live_sh_cells

            n_owners = len(_live_sh_cells(c, getattr(s, 'strato', None) is not None, len(h), len(chi), s.W))
        else:
            n_owners = cells
        table_bytes = 4 * nodes * (n_owners * nodes * stride + K + L + 1) + 8 * cells + 4 * n_owners
        src_rm_bytes = cells * 4 * stride * 4
        return table_bytes + src_rm_bytes + 3 * x.nbytes + (512 << 20)

    def solve(self, max_eval=16, tolerance=0.005):
        """Same Anderson4 normalization, regularization and candidate guard."""
        s = self.s
        xp = self.xp
        history_storage = os.environ.get('SOSFUSE2_ANDERSON_HISTORY', 'device')
        if history_storage not in ('host', 'device'):
            raise ValueError('E_ANDERSON_HISTORY: expected device or host')
        host_history = history_storage == 'host'
        self.timings['anderson_history'] = history_storage
        # where the history rings actually live: 'host', or with SOSFUSE2_ANDERSON_HISTORY=host on CUDA 'device-sources'
        # or 'device' when the first evaluation found enough free device memory (set below, kept per chunk)
        self.timings['anderson_ring'] = 'device' if not host_history and xp is not np else 'host'
        with vram_trace.item('source.iterate'):
            x = xp.zeros((s.geo.ns, source_size(s), s.W), 'f4')
        gs = [[] for _ in range(s.W)]
        fs = [[] for _ in range(s.W)]
        grams = [np.empty((0, 0), 'f8') for _ in range(s.W)] if host_history else None
        tmp_buf = None
        src_pin = None
        mix_f4 = None
        res_pin = None
        fs_mat = [None for _ in range(s.W)] if host_history else None
        ring_len = [0 for _ in range(s.W)] if host_history else None
        ring_head = [0 for _ in range(s.W)] if host_history else None
        device_sources = False
        device_residuals = False
        done = np.zeros(s.W, bool)
        history = []
        metrics_kernel = None
        started = time.perf_counter()
        boxes = self.observer or (self.boxes_fixed if self.fixed_boxes else self.boxes)
        try:
            for iteration in range(1, max_eval + 1):
                trace = vram_trace.current.get()
                if trace is not None:
                    trace.iteration = iteration
                evaluation_start = time.perf_counter()
                old = boxes(x)
                old_query_seconds = time.perf_counter() - evaluation_start
                apply_start = time.perf_counter()
                g = self.apply(x)
                apply_seconds = time.perf_counter() - apply_start
                # A NaN trace takes the kernel's linear path and propagates instead of vanishing: count and stop.
                ss = None
                if (getattr(self.kernel, 'backend', None) == 'cupy' and 0 < s.W <= 32 and g.size
                        and g.flags.c_contiguous and x.flags.c_contiguous and g.dtype == x.dtype == np.float32):
                    with vram_trace.item('source.residual_test'):
                        if metrics_kernel is None:
                            metrics_kernel = xp.RawKernel(_SOURCE_METRICS, 'source_metrics')
                        nonfinite, ss = _device_source_metrics(xp, metrics_kernel, g, x, done)
                else:
                    nonfinite = int(xp.count_nonzero(~xp.isfinite(g)))
                if nonfinite:
                    raise RuntimeError(f'E_NONFINITE_SOURCE: {nonfinite} values after evaluation {iteration}')
                if done.any():
                    with vram_trace.item('source.masked'):
                        g = xp.where(xp.asarray(done)[None, None, :], x, g)
                query_start = time.perf_counter()
                new = boxes(g)
                query_seconds = old_query_seconds + time.perf_counter() - query_start
                stop_count = len(self.monitor_sets['stop'])
                stop_old, stop_new = old[:stop_count], new[:stop_count]
                rr = np.max(abs(stop_new - stop_old) / np.maximum(abs(stop_new), 1e-30), axis=(0, 1))
                if ss is None:
                    with vram_trace.item('source.residual_test'):
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
                if self.monitor_sets['record']:
                    history[-1]['record_observer_relative'] = np.max(
                        abs(new[stop_count:] - old[stop_count:]) / np.maximum(abs(new[stop_count:]), 1e-30), axis=1
                    ).tolist()
                print(
                    f'SOSFUSE EVAL {iteration} obs={max(rr):.6g} source={max(ss):.6g} done={done.sum()}/{s.W}', flush=True
                )
                if done.all():
                    x = g
                    del g
                    history[-1]['evaluation_seconds'] = time.perf_counter() - evaluation_start
                    history[-1]['Anderson_seconds'] = 0.0
                    break
                for w in range(s.W):
                    if done[w]:
                        gs[w] = []
                        if host_history:
                            fs_mat[w] = None
                        else:
                            fs[w] = []
                anderson_start = time.perf_counter()
                with vram_trace.item('source.candidate'):
                    nxt = g.copy()  # the Anderson time is recorded below as stage 'anderson'
                on_device = host_history and xp is not np
                features = x.shape[0] * x.shape[1] + stop_new[:, :, 0].size
                if iteration == 1 and on_device and not getattr(self.kernel, 'deterministic', False):
                    # Keep Anderson rings in free device memory when warm transport retains headroom.
                    free_during_apply = self.kernel.ksplit_memory() + 2 * x.nbytes
                    room = free_during_apply - self._apply_vram_reserve(x)
                    source_ring_bytes = 4 * x.nbytes
                    residual_ring_bytes = 4 * s.W * features * 4
                    if room >= source_ring_bytes + residual_ring_bytes:
                        device_sources = True
                        device_residuals = True
                    elif room >= source_ring_bytes:
                        device_sources = True
                    self.timings['anderson_ring'] = 'device' if device_residuals else 'device-sources' if device_sources else 'host'
                pin_history = 12 * x.nbytes <= (2 << 30)
                strato = getattr(s, 'strato', None) is not None
                gpu_history_math = _use_gpu_history(
                    host_history, on_device, getattr(self.kernel, 'deterministic', False), strato,
                    getattr(self.kernel, 'L', None), s.W, features, self.kernel.ksplit_memory() if on_device else 0,
                ) and (not strato or pin_history)
                gw = norm = scale = feature = obs = res_view = None
                for w in range(s.W):
                    if done[w]:
                        continue
                    gw = g[:, :, w]
                    norm = xp.linalg.norm(gw, axis=1)
                    scale = xp.maximum(norm, norm.max() * 1e-8)
                    with vram_trace.item('anderson.feature'):
                        feature = ((gw - x[:, :, w]) / scale[:, None]).astype('f4', copy=False).ravel()
                    obs = xp.asarray(
                        ((stop_new[:, :, w] - stop_old[:, :, w]) / np.maximum(abs(stop_new[:, :, w]), 1e-20)).ravel()
                        * math.sqrt(len(feature) / (stop_new.shape[0] * stop_new.shape[1])), dtype=xp.float32,
                    )
                    if host_history and device_sources and device_residuals:
                        nfeat = len(feature) + len(obs)
                        cur_len = ring_len[w]
                        try:
                            if fs_mat[w] is None:
                                with vram_trace.item('anderson.residual_history'):
                                    fs_mat[w] = xp.empty((4, nfeat), dtype=xp.float32)
                            if cur_len < 4:
                                with vram_trace.item('anderson.source_history'):
                                    new_src = gw.copy()
                            else:
                                new_src = None
                        except MemoryError as exc:
                            warnings.warn(f'Anderson device ring fallback ({exc})', RuntimeWarning)
                            device_sources = False
                            device_residuals = False
                            self.timings['anderson_ring'] = 'host'
                            for wi in range(s.W):
                                if gs[wi] and not isinstance(gs[wi][0], np.ndarray):
                                    gs[wi] = [self.kernel.host(arr) for arr in gs[wi]]
                                if fs_mat[wi] is not None and not isinstance(fs_mat[wi], np.ndarray):
                                    fs_mat[wi] = self.kernel.host(fs_mat[wi])
                        else:
                            if cur_len < 4:
                                slot = cur_len
                                gs[w].append(new_src)
                                order_prev = list(range(cur_len))
                                ring_len[w] = cur_len + 1
                            else:
                                slot = ring_head[w]
                                ring_head[w] = (slot + 1) & 3
                                grams[w] = grams[w][1:, 1:]
                                order_prev = [(slot + 1 + j) & 3 for j in range(3)]
                                xp.copyto(gs[w][slot], gw)
                            del new_src
                            order_new = order_prev + [slot]
                            res_view = fs_mat[w][slot]
                            res_view[: len(feature)] = feature
                            res_view[len(feature) :] = obs
                            del feature, obs
                            diag = float(self.kernel.host(xp.dot(res_view, res_view)))
                            m = len(order_prev)
                            if m:
                                products = self.kernel.host(fs_mat[w][:cur_len] @ res_view)
                                col = products[order_prev]
                                G = np.empty((m + 1, m + 1), dtype='f8')
                                G[:m, :m] = grams[w]
                                G[:m, m] = col
                                G[m, :m] = col
                                G[m, m] = diag
                            else:
                                G = np.array([[diag]], dtype='f8')
                            grams[w] = G
                            if len(order_new) > 1:
                                self.kernel.memory()
                                try:
                                    ah = _anderson_weights(G)
                                    if ah is not None:
                                        mixed = xp.multiply(gs[w][order_new[0]], ah[0], dtype=xp.float32)
                                        for c_val, s_idx in zip(ah[1:], order_new[1:]):
                                            mixed += xp.multiply(gs[w][s_idx], c_val, dtype=xp.float32)
                                        nxt[:, :, w] = mixed
                                        del mixed
                                except np.linalg.LinAlgError:
                                    pass
                            continue
                    with vram_trace.item('anderson.source_history'):
                        source = gw.copy()
                    with vram_trace.item('anderson.residual_history'):
                        residual = xp.concatenate([feature, obs])
                    if host_history:
                        diag = float(self.kernel.host(xp.dot(residual, residual)))
                        nfeat = int(residual.size)
                        cur_len = ring_len[w]
                        if fs_mat[w] is None:
                            fs_mat[w] = _history_row(xp, (4, nfeat), 'f4', gpu_history_math and pin_history)
                        if cur_len < 4:
                            slot = cur_len
                            if device_sources:
                                gs[w].append(source)
                            else:
                                gs[w].append(_history_row(xp, source.shape, source.dtype, gpu_history_math and pin_history))
                            order_prev = list(range(cur_len))
                            ring_len[w] = cur_len + 1
                        else:
                            slot = ring_head[w]
                            ring_head[w] = (slot + 1) & 3
                            grams[w] = grams[w][1:, 1:]
                            order_prev = [(slot + 1 + j) & 3 for j in range(3)]
                            if device_sources:
                                xp.copyto(gs[w][slot], source)
                        order_new = order_prev + [slot]
                        res_view = fs_mat[w][slot]
                        if gpu_history_math:
                            if not device_sources:
                                pinned = isinstance(gs[w][slot].base, xp.cuda.PinnedMemoryPointer)
                                source.get(out=gs[w][slot], blocking=not pinned)
                            residual.get(out=res_view)
                        elif xp is not np:
                            if not device_sources and src_pin is None:
                                src_pin = _history_row(xp, (s.geo.ns, source_size(s)), np.float32, True)
                            if res_pin is None or res_pin.size != nfeat:
                                res_pin = _history_row(xp, (nfeat,), np.float32, True)
                            if not device_sources:
                                source.get(out=src_pin)
                                np.copyto(gs[w][slot], src_pin)
                            residual.get(out=res_pin)
                            np.copyto(res_view, res_pin)
                        else:
                            np.copyto(gs[w][slot], self.kernel.host(source))
                            np.copyto(res_view, self.kernel.host(residual))
                        del source
                        m = len(order_prev)
                        if m:
                            previous = fs_mat[w][:cur_len]
                            if gpu_history_math:
                                try:
                                    products = self.kernel.host(xp.asarray(previous) @ residual)
                                except MemoryError as exc:
                                    warnings.warn(f'Anderson Gram fallback ({exc})', RuntimeWarning)
                                    gpu_history_math = False
                            if not gpu_history_math:
                                products = previous @ res_view
                            col = products[order_prev]
                            G = np.empty((m + 1, m + 1), dtype='f8')
                            G[:m, :m] = grams[w]
                            G[:m, m] = col
                            G[m, :m] = col
                            G[m, m] = diag
                        else:
                            G = np.array([[diag]], dtype='f8')
                        del residual
                        grams[w] = G
                        if len(order_new) > 1:
                            self.kernel.memory()
                            try:
                                ah = _anderson_weights(G)
                                if ah is not None:
                                    if device_sources:
                                        mixed = xp.multiply(gs[w][order_new[0]], ah[0], dtype=xp.float32)
                                        for c_val, s_idx in zip(ah[1:], order_new[1:]):
                                            mixed += xp.multiply(gs[w][s_idx], c_val, dtype=xp.float32)
                                        nxt[:, :, w] = mixed
                                        del mixed
                                        continue
                                    if gpu_history_math:
                                        mixed = None
                                        try:
                                            mixed = xp.multiply(xp.asarray(gs[w][order_new[0]]), ah[0], dtype=xp.float32)
                                            for c_val, s_idx in zip(ah[1:], order_new[1:]):
                                                mixed += xp.multiply(xp.asarray(gs[w][s_idx]), c_val, dtype=xp.float32)
                                            nxt[:, :, w] = mixed
                                            continue
                                        except MemoryError as exc:
                                            warnings.warn(f'Anderson mixing fallback ({exc})', RuntimeWarning)
                                            gpu_history_math = False
                                        finally:
                                            del mixed
                                    if mix_f4 is None:
                                        shape = (s.geo.ns, source_size(s))
                                        mix_f4 = (
                                            _history_row(xp, shape, np.float32, True)
                                            if xp is not np
                                            else np.empty(shape, 'f4')
                                        )
                                    if tmp_buf is None:
                                        tmp_buf = np.empty((s.geo.ns, source_size(s)), 'f4')
                                    np.multiply(gs[w][order_new[0]], ah[0], out=mix_f4)
                                    for c_val, s_idx in zip(ah[1:], order_new[1:]):
                                        np.multiply(gs[w][s_idx], c_val, out=tmp_buf)
                                        mix_f4 += tmp_buf
                                    nxt[:, :, w] = xp.asarray(mix_f4)
                            except np.linalg.LinAlgError:
                                pass
                    else:
                        gs[w].append(source)
                        fs[w].append(residual)
                        del source, residual
                        if len(fs[w]) > 4:
                            gs[w].pop(0)
                            fs[w].pop(0)
                        if len(fs[w]) > 1:
                            with vram_trace.item('anderson.stack'):
                                f = xp.stack(fs[w], 1)
                            products = self.kernel.host(f.T @ f)
                            self.kernel.memory()
                            try:
                                ah = _anderson_weights(products)
                                if ah is not None:
                                    nxt[:, :, w] = sum(c * q for c, q in zip(ah, gs[w]))
                            except np.linalg.LinAlgError:
                                pass
                # nxt starts as a copy of g and the mixing loop only writes active wavelengths.
                x = nxt
                gw = norm = scale = feature = obs = res_view = None
                del g, nxt
                self.kernel.sync()
                history[-1]['Anderson_seconds'] = time.perf_counter() - anderson_start
                TIMERS.add('anderson', history[-1]['Anderson_seconds'])
                history[-1]['evaluation_seconds'] = time.perf_counter() - evaluation_start
        finally:
            gs.clear()
            fs.clear()
            if fs_mat is not None:
                fs_mat.clear()
            tmp_buf = src_pin = mix_f4 = res_pin = None
            if xp is not np:
                xp.get_default_memory_pool().free_all_blocks()
        self.kernel.sync()
        if not done.all():
            raise RuntimeError(f'E_NOT_CONVERGED: {done.sum()}/{s.W} within {tolerance}')
        self.timings['ground_sh_order'] = self.ground_sh_order()
        return x, dict(
            seconds=time.perf_counter() - started,
            evaluations=iteration,
            tolerance=tolerance,
            converged=done.tolist(),
            history=history,
            **(dict(monitor_sets=self.monitor_sets, convergence_path='extended-top-observer') if self.monitor_sets['record'] else {}),
        )
