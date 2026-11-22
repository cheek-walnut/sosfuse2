"""Fixed-point iteration of the atmospheric source field."""

import math
import os
import time

import numpy as np

from ..backends import numpy_reference
from sosfuse2.backends import vram_trace
from ..backends.arithmetic import F
from ..backends.sh_layout import source_size
from ..foundation.timers import TIMERS, stage
from ..foundation import top


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
        # A5: opt-in observer path (fixed captured rays, device source read directly).
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

    def solve(self, max_eval=16, tolerance=0.005):
        """Same Anderson4 normalization, regularization and candidate guard."""
        s = self.s
        xp = self.xp
        history_storage = os.environ.get('SOSFUSE2_ANDERSON_HISTORY', 'device')
        if history_storage not in ('host', 'device'):
            raise ValueError('E_ANDERSON_HISTORY: expected device or host')
        host_history = history_storage == 'host'
        history_xp = np if host_history else xp
        self.timings['anderson_history'] = history_storage
        with vram_trace.item('source.iterate'):
            x = xp.zeros((s.geo.ns, source_size(s), s.W), 'f4')
        gs = [[] for _ in range(s.W)]
        fs = [[] for _ in range(s.W)]
        done = np.zeros(s.W, bool)
        history = []
        started = time.perf_counter()
        boxes = self.observer or (self.boxes_fixed if self.fixed_boxes else self.boxes)
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
            nonfinite = int(xp.count_nonzero(~xp.isfinite(g)))
            if nonfinite:
                raise RuntimeError(f'E_NONFINITE_SOURCE: {nonfinite} values after evaluation {iteration}')
            with vram_trace.item('source.masked'):
                g = xp.where(xp.asarray(done)[None, None, :], x, g)
            query_start = time.perf_counter()
            new = boxes(g)
            query_seconds = old_query_seconds + time.perf_counter() - query_start
            stop_count = len(self.monitor_sets['stop'])
            stop_old, stop_new = old[:stop_count], new[:stop_count]
            rr = np.max(abs(stop_new - stop_old) / np.maximum(abs(stop_new), 1e-30), axis=(0, 1))
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
                history[-1]['evaluation_seconds'] = time.perf_counter() - evaluation_start
                history[-1]['Anderson_seconds'] = 0.0
                break
            anderson_start = time.perf_counter()
            with vram_trace.item('source.candidate'):
                nxt = g.copy()  # A1: Anderson time recorded below as stage 'anderson'
            for w in range(s.W):
                if done[w]:
                    continue
                norm = xp.linalg.norm(g[:, :, w], axis=1)
                scale = xp.maximum(norm, norm.max() * 1e-8)
                with vram_trace.item('anderson.feature'):
                    feature = ((g[:, :, w] - x[:, :, w]) / scale[:, None]).astype('f8').ravel()
                obs = xp.asarray(
                    ((stop_new[:, :, w] - stop_old[:, :, w]) / np.maximum(abs(stop_new[:, :, w]), 1e-20)).ravel()
                    * math.sqrt(len(feature) / (stop_new.shape[0] * stop_new.shape[1]))
                )
                with vram_trace.item('anderson.source_history'):
                    source = g[:, :, w].copy()
                with vram_trace.item('anderson.residual_history'):
                    residual = xp.concatenate([feature, obs])
                gs[w].append(self.kernel.host(source) if host_history else source)
                fs[w].append(self.kernel.host(residual) if host_history else residual)
                del source, residual
                if len(fs[w]) > 4:
                    gs[w].pop(0)
                    fs[w].pop(0)
                if len(fs[w]) > 1:
                    # Stage one wavelength at a time. Gram and mixing arithmetic stay on the solve backend.
                    with vram_trace.item('anderson.stack'):
                        f = xp.asarray(history_xp.stack(fs[w], 1))
                    gram = f.T @ f
                    gram += xp.eye(len(gram)) * max(float(self.kernel.host(xp.trace(gram))) * 1e-10, 1e-30)
                    self.kernel.memory()
                    if host_history:
                        del f
                    try:
                        alpha = xp.linalg.solve(gram, xp.ones(len(gram)))
                        alpha /= alpha.sum()
                        ah = self.kernel.host(alpha)
                        if np.isfinite(ah).all() and max(abs(ah)) < 8:
                            nxt[:, :, w] = sum(c * xp.asarray(q) for c, q in zip(alpha, gs[w]))
                    except np.linalg.LinAlgError:
                        pass
            # A converged wavelength is frozen exactly as the accepted active mask.
            with vram_trace.item('source.iterate'):
                x = xp.where(xp.asarray(done)[None, None, :], g, nxt)
            self.kernel.sync()
            history[-1]['Anderson_seconds'] = time.perf_counter() - anderson_start
            TIMERS.add('anderson', history[-1]['Anderson_seconds'])
            history[-1]['evaluation_seconds'] = time.perf_counter() - evaluation_start
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
