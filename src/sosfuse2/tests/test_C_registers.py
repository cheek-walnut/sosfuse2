"""Query CUDA driver occupancy for one supported configuration; no transport run.
The historical A/C labels use retired flags and therefore measure the same configuration.
CLI harness: python -m sosfuse2.tests.test_C_registers --out <dir>  (CUDA driver + CuPy; not a pytest module)."""

import argparse, ctypes, json, sys
from pathlib import Path

if __name__ != '__main__':
    import pytest

    pytest.skip(
        'GPU occupancy CLI harness (python -m sosfuse2.tests.test_C_registers --out <dir>)', allow_module_level=True
    )
from sosfuse2.solve.source_solver import Kernel
import cupy as cp

A = 'LANE_GROUND,LM_LUT,PHASE2_NOBARRIER,SH_AUX,SPECIALIZE'
C = A + ',RAY_CONST_SHARED,POINT_NOINLINE,LANE_ATTEN,GROUND_PHASE_BATCH,PHASE1_TRACE3'
p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
args = p.parse_args()
lib = ctypes.CDLL('libcuda.so.1')
occupancy = lib.cuOccupancyMaxActiveBlocksPerMultiprocessor
occupancy.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
occupancy.restype = ctypes.c_int
rows = []
for label, flags in [('A', A), ('C', C)]:
    for W in (1, 15):
        for first, phase, boundary in [(1, 0, 0), (0, 0, 1), (0, 1, 0), (1, 2, 0), (0, 2, 0), (0, 2, 1)]:
            kernel = Kernel(backend='cupy', block_size=256, flags=flags.split(',') + ['PROFILE'])
            raw = kernel.raw_kernel(W, first, phase, boundary)
            dyn = kernel.shared_bytes(W, phase, first)
            blocks = ctypes.c_int()
            ptr = raw.kernel.ptr
            rc = occupancy(ctypes.byref(blocks), ctypes.c_void_p(ptr), 256, dyn)
            if rc:
                raise RuntimeError(f'E_OCCUPANCY: CUDA driver rc={rc}')
            a = raw.attributes
            rows.append(
                dict(
                    label=label,
                    W=W,
                    first=first,
                    phase=phase,
                    boundary=boundary,
                    num_regs=a['num_regs'],
                    local_size_bytes=a['local_size_bytes'],
                    shared_size_bytes=a['shared_size_bytes'],
                    dynamic_shared_bytes=dyn,
                    total_shared_bytes=dyn + a['shared_size_bytes'],
                    max_threads_per_block=a['max_threads_per_block'],
                    theoretical_blocks_per_SM=blocks.value,
                    occupancy_source='cuOccupancyMaxActiveBlocksPerMultiprocessor',
                    flags=list(kernel.flags),
                )
            )
props = cp.cuda.runtime.getDeviceProperties(0)
result = dict(
    compute_capability=[int(props['major']), int(props['minor'])],
    rows=rows,
    note='RawKernel attributes report static shared; occupancy includes actual phase dynamic shared at block256.',
)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
