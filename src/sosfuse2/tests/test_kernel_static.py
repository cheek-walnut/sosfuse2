#!/usr/bin/env python3
"""C++ syntax / preprocessing gates, with explicit non-CUDA scope.

Runs without the full repository on macOS/Linux (clang++, Python >=3.9).
Syntax and unsupported flag conflicts are checked; numerical tolerance is the
requirement. Historical token identity is not a requirement.
"""

from sosfuse2 import paths, testing
import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys

# Keep the documented direct-script entry point and pytest collection working.
from sosfuse2.tests.test_support_strato import C, C2, BYTE

STUB = r'''// Deliberately header-free syntax stubs; no host math implementation is used.
template<class T> inline T min(T a,T b){return a<b?a:b;}
template<class T> inline T max(T a,T b){return a>b?a:b;}
float fmaf(float,float,float); float sqrtf(float);float expf(float);float logf(float);
float fabsf(float);float cosf(float);float sinf(float);float atan2f(float,float);
float fminf(float,float);float fmaxf(float,float);
double sqrt(double);double fmin(double,double);double fmax(double,double);
#define __device__
#define __global__
#define __constant__
#define __shared__
#define __forceinline__ inline
#define __noinline__
struct Dim { int x,y,z; };extern Dim blockIdx,threadIdx,blockDim;
struct float2 { float x,y; };
inline void __syncwarp(){} inline void __syncthreads(){}
inline void __trap(){}   // kernel.cu: sos_fused traps at entry without LANE_ATTEN (task B Simpson prefix)
inline unsigned __ballot_sync(unsigned,bool){return 0;}
inline unsigned long long clock64(){return 0;}
template<class T>T __shfl_sync(unsigned,T v,int){return v;}
template<class T>T __shfl_down_sync(unsigned,T v,int){return v;}
'''


def defines(W, flags, spec=(1, 2, 1)):
    opts = [f'-DFW={W}'] + [f'-D{flag}' for flag in sorted(set(flags))]
    opts += [f'-DK_FIRST={spec[0]}', f'-DK_PHASE={spec[1]}', f'-DK_BOUNDARY={spec[2]}']
    return opts


def run(out, jobs=8, baseline=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which('clang++')
    if compiler is None:
        raise RuntimeError('E_CLANG: clang++ required')
    from sosfuse2.tests.test_tropo_mie import SYNTAX_STUB_EXTRA

    stub = out / 'cuda_builtins_stub.h'
    stub.write_text(STUB + SYNTAX_STUB_EXTRA + 'float __expf(float);float __sinf(float);float __cosf(float);\n')
    code = paths.PKG / 'kernel.cu'
    syntax = []
    legacy = [
        (),
        ('SH_AUX',),
        ('LANE_GROUND',),
        ('LM_LUT',),
        ('PHASE2_NOBARRIER',),
        ('SPECIALIZE',),
        ('SPECIALIZE', 'SH_AUX', 'LANE_GROUND', 'LM_LUT', 'PHASE2_NOBARRIER'),
        ('SH_AUX', 'LANE_GROUND'),
    ]
    for W, st, pro, flags in itertools.product((1, 15), (False, True), (False, True), legacy):
        syntax.append((W, flags + ('STRATO',) * st + ('PROFILE',) * pro, (1, 2, 1)))
    extras = [
        (),
        ('STRATO_LANE_GROUND',),
        ('STRATO_LANE_FIRSTVAL',),
        BYTE,
        BYTE + ('STRATO_SH_AUX',),
        BYTE + ('STRATO_MIE_LUT',),
        BYTE + ('STRATO_MIE_LUT', 'STRATO_MIE_LUT_FINE'),
    ]
    specs = list(itertools.product((0, 1), (0, 1, 2), (0, 1)))
    for W, pro, base, extra, spec in itertools.product((1, 15), (False, True), (C, C2), extras, specs):
        syntax.append((W, base + extra + ('STRATO',) + ('PROFILE',) * pro, spec))
    for W, st, extra in itertools.product((1, 15), (False, True), extras):
        syntax.append((W, tuple(f for f in C2 if f != 'SPECIALIZE') + extra + ('STRATO',) * st, (1, 2, 1)))
    for W, spec in itertools.product((1, 15), specs):
        syntax.append((W, C2 + BYTE + ('STRATO', 'PROFILE', 'EXP_SECTION_CLOCKS'), spec))

    def check_syntax(case):
        W, flags, spec = case
        cmd = [
            compiler,
            '-std=c++17',
            '-ffp-contract=off',
            '-fsyntax-only',
            '-x',
            'c++',
            '-include',
            str(stub),
            *defines(W, flags, spec),
            str(code),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return dict(
            W=W,
            flags=list(flags),
            specialization=list(spec),
            pass_=result.returncode == 0,
            stderr=result.stderr if result.returncode else '',
        )

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(check_syntax, syntax))
    # Optimization dependencies disappeared; physical ground-SH exclusions remain.
    conflicts = [
        (('GROUND_SH_L=16', 'GROUND_SH_GRID=32', 'TROPO_MIE'), 'E_GROUND_SH_PATH'),
        (('GROUND_SH_L=16', 'GROUND_SH_GRID=32', 'LUT_ONLY'), 'E_GROUND_SH_PATH'),
    ]
    errors = []
    for flags, expected_error in conflicts:
        row = check_syntax((15, flags + ('STRATO',), (1, 2, 1)))
        row['pass_'] = not row['pass_'] and expected_error in row['stderr']
        row['expected_error'] = expected_error
        errors.append(row)
    report = dict(
        pass_=all(r['pass_'] for r in rows + errors),
        syntax_count=len(rows),
        conflict_count=len(errors),
        compiler=subprocess.check_output([compiler, '--version'], text=True).splitlines()[0],
        cases=rows,
        conflicts=errors,
        scope='Host Clang syntax only; CUDA builtins stubbed. NOT NVRTC, PTX/SASS, races, registers or GPU parity.',
    )
    (out / 'kernel_static.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"CLANG {len(rows)} syntax + {len(errors)} expected conflicts: " + ('PASS' if report['pass_'] else 'FAIL'))
    for row in [r for r in rows + errors if not r['pass_']][:5]:
        print(json.dumps(row, indent=2))
    return report


def test_run(tmp_path):
    testing.require_tool('clang++')
    assert run(tmp_path, jobs=2)['pass_']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--baseline', type=Path)
    a = p.parse_args()
    raise SystemExit(0 if run(a.out, max(1, a.jobs), a.baseline)['pass_'] else 1)
