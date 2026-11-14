#!/usr/bin/env python3
"""Portable host tests for the new flags; no engine/CUDA/import side effects."""

import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np
from sosfuse2.tests.test_support_strato import C, C2, BYTE, NEW, load_wrapper
from sosfuse2.solve.strato_sh_aux import strato_sh_aux

from sosfuse2.checks.numerical_checks import within_tol
F = np.float32


def assert_raises(call, text):
    try:
        call()
    except (ValueError, RuntimeError) as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError('expected rejection: ' + text)


def check_flags():
    wrapper = load_wrapper()
    Kernel = wrapper.Kernel
    bad = [(('STRATO_MIE_LUT_TYPO',), 'E_KERNEL_FLAG')]
    for flags, error in bad:
        assert_raises(lambda: Kernel(flags=flags), error)
    count = 0
    for bits in itertools.product((False, True), repeat=len(NEW)):
        extra = tuple(f for f, bit in zip(NEW, bits) if bit)
        k = Kernel(flags=C2 + extra)
        assert k.flags == ()  # retired optimization lists all select default transport
        count += 1
    for W, first, phase, boundary in itertools.product((1, 15), (0, 1), (0, 1, 2), (0, 1)):
        # No new specialization key is introduced by the patch.
        assert Kernel(flags=C2).tile_width(W, first, phase) == W
    # Enabling or resizing a LUT on an already populated Kernel cache must
    # reject stale coefficient-only storage rather than read beyond its end.
    dummy = object()
    k = Kernel(flags=C2 + ('STRATO', 'STRATO_MIE_LUT'))
    key = (id(dummy), 12)
    k.constants[key] = {'W': 15}
    k.device_constants[key] = {'strato_chi': np.zeros(33 * 15, 'f4')}
    assert_raises(lambda: k.constants_device(dummy, 12), 'E_STRATO_LUT_CACHE')
    k.device_constants[key]['strato_chi'] = np.zeros((33 + 8192 + 1) * 15, 'f4')
    assert_raises(lambda: k.constants_device(dummy, 12), 'E_STRATO_LUT_CACHE')
    k.device_constants[key]['strato_chi'] = np.zeros((33 + 16384 + 1) * 15, 'f4')
    assert k.constants_device(dummy, 12) is k.device_constants[key]
    return dict(valid_combinations=count, expected_conflicts=len(bad), stale_lut_rejected=True, pass_=True)


def check_layout():
    rng = np.random.default_rng(82)
    checked = 0
    for W, cells, paired in itertools.product((1, 15), (1, 3, 7), (False, True)):
        src = rng.normal(size=(cells, 1132, W)).astype('f4')
        # Include signed zero and subnormals: storage must preserve their bits.
        src.ravel()[:8] = np.array(
            [0.0, -0.0, np.nextafter(F(0), F(1)), -np.nextafter(F(0), F(1)), 1.0, -1.0, 1e-20, -1e-20], 'f4'
        )
        src[:, 10, :] = src.ravel()[:W] if W == 1 else F(-0.0)
        src[:, 571, :] = np.nextafter(F(0), F(1))
        saved = src.tobytes()
        aux = strato_sh_aux(src, pair=paired)
        assert src.tobytes() == saved and aux.flags.c_contiguous
        for plane, start in enumerate((10, 571)):
            for cc in range(cells):
                for w in range(W):
                    if paired:
                        # Exactly the float2 index used in kernel.cu.
                        base = ((plane * cells + cc) * ((W + 1) // 2) + w // 2) * 576
                        flat = aux.reshape(-1, 2)
                        got = flat[base : base + 561, w % 2]
                    else:
                        base = ((plane * cells + cc) * W + w) * 576
                        got = aux.ravel()[base : base + 561]
                    assert got.tobytes() == src[cc, start : start + 561, w].tobytes()
                    checked += 561
        if paired:
            assert not np.any(aux[:, :, 561:, :])
            assert not np.any(aux[:, -1, :, 1])  # both supported W are odd
        else:
            assert not np.any(aux[:, :, 561:])
    return dict(fp32_values_bit_checked=checked, layouts='plain and float2; both SH planes; W1/W15', pass_=True)


def check_ground_and_first():
    rng = np.random.default_rng(1042)
    ground_cases = 0
    source_values = 0
    for W, tile in ((1, 1), (15, 15), (15, 3), (15, 5)):
        for trial in range(24):
            # Already evaluated +/- phases have no cross-wave dependency.
            # All products and sums below explicitly round to binary32.
            counts = [0, 1, 17, 33] if trial % 3 == 0 else rng.integers(0, 35, 4).tolist()
            sw = rng.random(4).astype('f4')
            sw = sw / F(sw.sum())
            pp = [rng.uniform(0.001, 1, (count, W)).astype('f4') for count in counts]
            pn = [rng.uniform(0.001, 1, (count, W)).astype('f4') for count in counts]
            gv = [rng.uniform(-0.1, 0.1, (count, W)).astype('f4') for count in counts]
            serial = np.zeros(W, 'f4')
            for corner, count in enumerate(counts):
                for j in range(count):
                    for w in range(W):
                        ph = F(F(pp[corner][j, w] + pn[corner][j, w]) / F(2))
                        serial[w] = F(serial[w] + F(F(sw[corner] * ph) * gv[corner][j, w]))
            lanes = np.zeros(W, 'f4')
            for group in range(W // tile):
                for lane in range(tile):
                    w = group * tile + lane
                    value = F(0)
                    for corner, count in enumerate(counts):
                        for j in range(count):
                            ph = F(F(pp[corner][j, w] + pn[corner][j, w]) / F(2))
                            value = F(value + F(F(sw[corner] * ph) * gv[corner][j, w]))
                    lanes[w] = value
            assert within_tol(serial, lanes)
            ground_cases += W
            # Same source expression, wave loop versus lane ownership; sun
            # phase is identical whether evaluated here or once per ray.
            prof = rng.random(3).astype('f4')
            ext = rng.random((3, W)).astype('f4')
            sc = rng.random(3).astype('f4')
            st_sc = F(rng.random())
            ste = rng.random(W).astype('f4')
            stsca = rng.random(W).astype('f4')
            asca = rng.random(W).astype('f4')
            pu = rng.uniform(-1, 1, 4).astype('f4')
            direct = rng.uniform(-1, 1, 4).astype('f4')
            mix = F(0.96)
            sunphase = F(0.45)
            miesun = rng.uniform(0.01, 1, W).astype('f4')
            stprof = F(0.03)

            def value_at(w, sun_lookup):
                od = F(0)
                for k in range(3):
                    od = F(od + F(sc[k] * ext[k, w]))
                od = F(od + F(st_sc * ste[w]))
                sun = F(np.exp(-od))
                beam = F(F(prof[1] * asca[w]) * F(F(sun * sunphase) + serial[w]))
                ray = F(F(sun * prof[0]) * ext[0, w])
                out = np.empty(4, 'f4')
                for k in range(4):
                    out[k] = F(
                        F(
                            ray
                            * F(
                                F(F(mix * F(3 / (8 * math.pi))) * direct[k])
                                + F(F(F(1) - mix) * F(1 / (8 * math.pi)) * pu[k])
                            )
                        )
                        + F(F(beam * pu[k]) / F(2))
                    )
                sb = F(F(stprof * stsca[w]) * F(F(sun * sun_lookup(w)) + serial[w]))
                for k in range(4):
                    out[k] = F(out[k] + F(F(sb * pu[k]) / F(2)))
                return out

            serial_values = np.stack([value_at(w, lambda i: miesun[i]) for w in range(W)])
            lane_values = np.empty_like(serial_values)
            for group in range(W // tile):
                for lane in range(tile):
                    w = group * tile + lane
                    cached = F(miesun[w])
                    lane_values[w] = value_at(w, lambda _: cached)
            assert within_tol(serial_values, lane_values)
            source_values += 4 * W
    return dict(
        ground_wave_cases=ground_cases,
        first_value_scalars=source_values,
        pass_=True,
        scope='FP32 arithmetic/scheduling emulation; not CUDA expf, shuffles or warp scheduling',
    )


class FakeEvent:
    def record(self):
        pass

    def synchronize(self):
        pass


def occupancy_tests():
    module = load_wrapper()
    Kernel = module.Kernel
    rows = []
    extras = [
        (),
        BYTE,
        BYTE + ('STRATO_SH_AUX',),
        BYTE + ('STRATO_MIE_LUT',),
        BYTE + ('STRATO_MIE_LUT', 'STRATO_MIE_LUT_FINE'),
    ]
    checks = 0
    for W, block, phase, st, base, extra in itertools.product(
        (1, 15), (32, 64, 128, 256), (0, 1, 2), (False, True), (C, C2), extras
    ):
        flags = base + ('STRATO',) * st
        k = Kernel(block_size=block, flags=flags + extra)
        old = Kernel(block_size=block, flags=flags)
        assert k.shared_bytes(W, phase) == old.shared_bytes(W, phase)
        assert k.shared_bytes(W, phase) <= 101376
        checks += 1
    for name, flags in [('C2', C2), ('C2_STRATO', C2 + ('STRATO',))]:
        k = Kernel(block_size=256, flags=flags)
        for phase in (0, 1, 2):
            shared = k.shared_bytes(15, phase)
            # Resource model assumptions made explicit, never measured occupancy.
            shared_blocks = 102400 // (((shared + 255) // 256) * 256 + 1024)
            for regs in (96, 128, 129, 160, 192, 255):
                register_block = ((regs * 32 + 255) // 256) * 256 * 8
                predicted = min(shared_blocks, 65536 // register_block, 1536 // 256)
                rows.append(
                    dict(
                        variant=name,
                        phase=phase,
                        shared_bytes=shared,
                        regs_assumed=regs,
                        resource_blocks_upper_bound=predicted,
                        warps_upper_bound=8 * predicted,
                    )
                )
    assert Kernel(block_size=256, flags=C2 + ('STRATO',)).shared_bytes(15, 0) == 74008
    assert Kernel(block_size=256, flags=C2 + ('STRATO',)).shared_bytes(15, 1) == 74008
    return dict(
        pass_=True,
        shared_comparisons=checks,
        block256_limit=101376,
        remaining_C3_STRATO_bytes=101376 - 74008,
        resources=rows,
        scope='Analytic limits only. Register counts assumed; 256-register/warp allocation, 1KiB/block runtime shared reserve assumed. Use CUDA occupancy API on each target.',
    )


def run(out, occupancy_only=False):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    r = dict(occupancy=occupancy_tests())
    if not occupancy_only:
        r.update(flags=check_flags(), sidecar=check_layout(), arithmetic=check_ground_and_first())
    r['pass_'] = all(v.get('pass_', False) for v in r.values())
    r['scope'] = 'Host layout, arithmetic and control-flow checks only; no GPU parity/performance assertion.'
    out.write_text(json.dumps(r, indent=2) + '\n')
    print('HOST PASS' if r['pass_'] else 'HOST FAIL')
    return r


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--occupancy-only', action='store_true')
    a = p.parse_args()
    raise SystemExit(0 if run(a.out, a.occupancy_only)['pass_'] else 1)
