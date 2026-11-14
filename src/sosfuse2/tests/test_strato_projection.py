#!/usr/bin/env python3
"""Sequential CUDA-order projection oracle, separate from production np.sum.

No projection implementation/precision/quadrature is changed. Compile the
kernel's FP32 product and sequential per-ray FP64 accumulation rules as host
C++; compare NumPy loop and lane decomposition. GPU execution remains untested.
"""

import argparse, ctypes, json, subprocess, sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
from sosfuse2.backends import numpy_reference
from sosfuse2.solve.ray_geometry import create, TransientGeometry
from sosfuse2.solve.source_solver import Solver
from sosfuse2.asset_tools.strato_states import build

try:
    from sosfuse2.checks.numerical_checks import assert_equivalent, within_tol
except ImportError:  # standalone research harness
    from numerical_checks import assert_equivalent, within_tol
CPP = r'''
#include <cmath>
extern "C" void project_seq(int N,int W,const float*rad,const float*wt,const float*y,const float*cm,double*out){
 for(int moment=0;moment<565;moment++)for(int w=0;w<W;w++){
  double acc=0.;for(int ray=0;ray<N;ray++){
   const float*r=rad+(ray*W+w)*4;float v;
   if(moment<4)v=r[moment]*wt[ray];
   else {int sk=moment-4,l=(int)((sqrtf(8.f*sk+1.f)-1.f)/2.f);
    while((l+1)*(l+2)/2<=sk)l++;while(l*(l+1)/2>sk)l--;int m=sk-l*(l+1)/2;
    float tr=(r[0]+r[1])+r[2];v=(tr*wt[ray])*(y[ray*561+sk]*cm[ray*33+m]);}
   acc+=(double)v;
  }out[moment*W+w]=acc;
 }
}
'''


def numpy_seq(g, rad, Y):
    weight = g['weight'].astype('f4')
    st = np.sqrt(np.maximum(0.0, 1 - g['d'][:, 2] ** 2))
    cp = np.where(st > 1e-12, g['d'][:, 0] / np.maximum(st, 1e-30), 1.0).astype('f4')
    cm = numpy_reference.cosine(cp)
    levels = np.array([m for l in range(33) for m in range(l + 1)])
    out = np.zeros((565, rad.shape[1]), 'f8')
    for ray in range(len(rad)):
        out[:4] += (rad[ray] * weight[ray]).T
        tr = (rad[ray, :, 0] + rad[ray, :, 1]) + rad[ray, :, 2]
        out[4:] += (tr * weight[ray])[None, :] * (Y[ray] * cm[ray, levels])[:, None]
    return out, cm


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    cpp = out / 'projection_host.cpp'
    lib = out / ('projection_host.dylib' if sys.platform == 'darwin' else 'projection_host.so')
    cpp.write_text(CPP)
    subprocess.run(
        [
            'clang++',
            '-std=c++17',
            '-O2',
            '-ffp-contract=off',
            '-dynamiclib' if sys.platform == 'darwin' else '-shared',
            '-fPIC',
            str(cpp),
            '-o',
            str(lib),
        ],
        check=True,
    )
    fn = ctypes.CDLL(str(lib)).project_seq
    fn.argtypes = [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 5
    wl = 360 + (np.arange(15) + 0.5) * 470 / 15
    s, scope, _ = create(out / 'geometry', 'selftest', wl, state=build()['states'][5])
    sv = Solver(s, TransientGeometry(s, scope))
    g = sv.geometry.cell(12)
    c = s.kernel.setup(s, g['nq'])
    rng = np.random.default_rng(20261001)
    field = rng.normal(0, 1e-5, (s.geo.ns, 1132, 15)).astype('f4')
    field[:, :3] = abs(field[:, :3])
    rows = []
    for first in (True, False):
        rad, Y = numpy_reference.transport(g, c, field, first, return_basis=True)
        seq, cm = numpy_seq(g, rad, Y)
        native = np.empty_like(seq)
        wt = np.ascontiguousarray(g['weight'], 'f4')
        Y = np.ascontiguousarray(Y)
        cm = np.ascontiguousarray(cm)
        rad = np.ascontiguousarray(rad)
        fn(len(rad), 15, rad.ctypes.data, wt.ctypes.data, Y.ctypes.data, cm.ctypes.data, native.ctypes.data)
        per = np.concatenate([numpy_seq(g, rad[:, w : w + 1], Y)[0] for w in range(15)], axis=1)
        production, _ = numpy_reference.project(g, rad, c, Y)
        rows.append(
            dict(
                first=first,
                modeled_cuda_numpy_within_tol=(within_tol(seq, native)),
                modeled_cuda_lane_within_tol=(within_tol(per, native)),
                production_numpy_vs_sequential_max_absolute=float(np.max(abs(production - native))),
                production_numpy_vs_sequential_max_relative=float(
                    np.max(abs(production - native) / np.maximum(abs(native), 1e-30))
                ),
            )
        )
    report = dict(
        pass_=all(r['modeled_cuda_numpy_within_tol'] and r['modeled_cuda_lane_within_tol'] for r in rows),
        cases=rows,
        scope='FP32 products and natural per-ray FP64 sum modeled from kernel.cu; no GPU/NVRTC projection execution',
    )
    (out / 'projection.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    raise SystemExit(0 if run(a.out)['pass_'] else 1)
