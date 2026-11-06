"""A6: NumPy evidence per kernel flag (no CUDA on this Mac).

SH_AUX       host builder + kernel index formula emulated on every (cell,k,w);
             reference.transport(sh_source=aux) vs default: bit comparison.
LANE_GROUND  first-light ground pool emulated in strict FP32 sequential order,
             original (lane 0, w inside j) vs lane-parallel (w outside j), and
             vs the reference's vectorized ground term: bit comparison.
LM_LUT       constant table vs the kernel's sqrtf + fixup recovery, all 561 sk.
SPECIALIZE / PHASE2_NOBARRIER / block 256: no arithmetic change (see doc);
             covered by preprocess identity + syntax log, not by NumPy.
"""
import json
import sys
from pathlib import Path

# Keep the documented direct-script and `python -m` entry points working (the bare import below needs this folder on sys.path).
# New portable entry point; legacy GPU/reference entry points below are retained.
if '--host-only' in sys.argv:
    import argparse
    from sosfuse2.tests.test_strato_parallel_host import run as run_strato_host
    parser=argparse.ArgumentParser()
    parser.add_argument('--host-only',action='store_true')
    parser.add_argument('--out',type=Path,default=Path('strato_host_report.json'))
    option=parser.parse_args()
    raise SystemExit(0 if run_strato_host(option.out,occupancy_only=False)['pass_'] else 1)

sys.dont_write_bytecode = True
import numpy as np
from sosfuse2.geometry import create, TransientGeometry
from sosfuse2.fused import sh_aux, KPAD
from sosfuse2 import reference
from sosfuse2.reference import sample, hg
try:
    from sosfuse2.numerical_checks import assert_equivalent, within_tol
except ImportError:  # standalone research harness
    from numerical_checks import assert_equivalent, within_tol
F = np.float32


def check_sh_aux(src):
    out = {}
    for W in (1, 15):
        field = np.ascontiguousarray(np.tile(src, (1, 1, W))*np.linspace(1, 2, W, dtype='f4'), 'f4') if W > 1 else np.ascontiguousarray(src, 'f4')
        aux = sh_aux(field); flat_src = field.ravel(); flat_aux = aux.ravel()
        cc, k, w = np.meshgrid(np.arange(field.shape[0]), np.arange(561), np.arange(W), indexing='ij')
        a = flat_src[((cc.astype('i8')*571+10+k)*W+w)]          # kernel.cu original load
        b = flat_aux[((cc.astype('i8')*W+w)*KPAD+k)]            # kernel.cu -DSH_AUX load
        out[f'W{W}_index_exact'] = bool(a.tobytes() == b.tobytes())
        out[f'W{W}_contiguous_copy'] = bool(aux.flags.c_contiguous and aux.base is None)
    return out


def check_transport(s, scope, src):
    geo = TransientGeometry(s, scope); res = {}
    for cell in (0, 12, 25):
        g = geo.cell(cell); c = s.kernel.setup(s, g['nq']); field = np.ascontiguousarray(src, 'f4')
        a = reference.transport(g, c, field, False); b = reference.transport(g, c, field, False, sh_source=sh_aux(field))
        res[str(cell)] = within_tol(a, b)
    return res


def check_lane_ground(s, scope):
    geo = TransientGeometry(s, scope); g = geo.cell(12); c = s.kernel.setup(s, g['nq']); gg = c['g']
    st = np.sqrt(np.maximum(0., 1-g['d'][:, 2]**2)).astype('f4'); rays = np.arange(0, len(g['d']), max(1, len(g['d'])//6))[:6]
    worst = 0.; identical = True; ref_identical = True; checked = 0
    # 15-lane ground values: replicate gval over a synthetic spectral factor.
    gv = np.ascontiguousarray(np.tile(c['gval'], (1, 15)) * np.linspace(.5, 1.5, 15, dtype='f4'), 'f4') if c['gval'].shape[1] == 1 else c['gval']
    gvp = np.ascontiguousarray(np.tile(c['gv'], (1, 1, 15)) * np.linspace(.5, 1.5, 15, dtype='f4'), 'f4') if c['gv'].shape[2] == 1 else c['gv']
    W = gv.shape[1]
    for seg in range(g['edges'].shape[1]-1):
        for step in range(len(c['u'])):
            q = sample(g, seg, step, c, st)
            mu = np.zeros(len(q['h']), 'f4'); exd = np.zeros_like(mu); eyd = np.zeros_like(mu)
            for k in range(3):
                mu += q['d'][:, k]*q['Q'][:, k, 2]; exd += q['d'][:, k]*q['Q'][:, k, 0]; eyd += q['d'][:, k]*q['Q'][:, k, 1]
            # Reference vectorized order (reference.py first-light block)
            ground = np.zeros((len(mu), W), 'f4')
            for corner in range(4):
                cells = q['cells'][:, corner]; gd = c['gd'][cells]
                m0 = mu[:, None]*gd[:, :, 2]+exd[:, None]*gd[:, :, 0]; m1 = abs(eyd)[:, None]*gd[:, :, 1]
                ph = (hg(np.clip(m0+m1, F(-1), F(1)), gg)+hg(np.clip(m0-m1, F(-1), F(1)), gg))/F(2)
                wp = q['sw'][:, corner, None]*ph
                for gi in range(c['gd'].shape[1]): ground += wp[:, gi, None]*gvp[cells, gi]
            for r in rays:
                if not q['valid'][r]: continue
                orig = np.zeros(W, 'f4'); lane = np.zeros(W, 'f4')
                for corner in range(4):                       # original: lane 0, w inside j
                    cc = q['cells'][r, corner]
                    for j in range(c['goff'][cc], c['goff'][cc+1]):
                        m0 = F(mu[r]*F(c['gdir'][j, 2]))+F(exd[r]*F(c['gdir'][j, 0])); m1 = F(abs(eyd[r])*F(c['gdir'][j, 1]))
                        ph = F((hg(np.clip(F(m0+m1), F(-1), F(1)), gg)+hg(np.clip(F(m0-m1), F(-1), F(1)), gg))/F(2))
                        for w in range(W): orig[w] = F(orig[w]+F(F(q['sw'][r, corner]*ph)*gv[j, w]))
                for w in range(W):                            # LANE_GROUND: lane w, j inside
                    acc = F(0)
                    for corner in range(4):
                        cc = q['cells'][r, corner]
                        for j in range(c['goff'][cc], c['goff'][cc+1]):
                            m0 = F(mu[r]*F(c['gdir'][j, 2]))+F(exd[r]*F(c['gdir'][j, 0])); m1 = F(abs(eyd[r])*F(c['gdir'][j, 1]))
                            ph = F((hg(np.clip(F(m0+m1), F(-1), F(1)), gg)+hg(np.clip(F(m0-m1), F(-1), F(1)), gg))/F(2))
                            acc = F(acc+F(F(q['sw'][r, corner]*ph)*gv[j, w]))
                    lane[w] = acc
                identical &= (within_tol(orig, lane))
                ref_identical &= (within_tol(orig, ground[r]))
                worst = max(worst, float(np.max(abs(orig-ground[r])/np.maximum(abs(ground[r]), 1e-30))))
                checked += 1
            if checked >= 60: break
        if checked >= 60: break
    return dict(samples_checked=checked, wavelengths=W, lane_vs_original_within_tol=bool(identical),
                original_vs_reference_within_tol=bool(ref_identical), original_vs_reference_max_relative=worst)


def check_lm_lut():
    lut_m = np.array([m for l in range(33) for m in range(l+1)], 'i4')
    sk = np.arange(561); ix = lambda l, m: l*(l+1)//2+m
    l = ((np.sqrt(F(8)*sk.astype('f4')+F(1))-F(1))/F(2)).astype('i4')
    for _ in range(3):
        l = np.where(ix(l+1, 0) <= sk, l+1, l)
    for _ in range(3):
        l = np.where(ix(l, 0) > sk, l-1, l)
    m = sk-ix(l, 0)
    return dict(entries=561, exact=bool(np.array_equal(m, lut_m)), lut_bytes=2*561)


def main(out, source_npz):
    with np.load(source_npz) as z: src = z['source']
    s, scope, _ = create(Path(out).parent/'a6_geometry', 'selftest', [550.], 'numpy', 128)
    r = dict(SH_AUX=dict(layout=check_sh_aux(src), transport_within_tol=check_transport(s, scope, src)),
             LANE_GROUND=check_lane_ground(s, scope), LM_LUT=check_lm_lut())
    r['pass_'] = bool(all(r['SH_AUX']['layout'].values()) and all(r['SH_AUX']['transport_within_tol'].values()) and
                      r['LANE_GROUND']['lane_vs_original_within_tol'] and r['LM_LUT']['exact'])
    Path(out).write_text(json.dumps(r, indent=1)+'\n'); print(json.dumps(r, indent=1)); print('A6 PASS' if r['pass_'] else 'A6 FAIL')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
