"""CPU syntax and scratch-allocation checks for both first-light ground paths.

Radiance parity and timing require the A0/A3/B GPU benchmark; these checks do
not emulate warp execution or claim GPU numerical coverage.
"""
from sosfuse2 import paths
import json
from pathlib import Path
import subprocess
import sys

import pytest

HERE = paths.PKG   # the package directory (kernel, configs, assets)
from sosfuse2 import testing
from sosfuse2.fused import Kernel
from sosfuse2.sh_layout import kernel_source, workspace
from sosfuse2.tests.test_kernel_static import STUB, defines
from sosfuse2.tests.test_tropo_mie import SYNTAX_STUB_EXTRA


@pytest.mark.parametrize('L,W', [(16, 1), (16, 15), (32, 1), (32, 15)])
def test_first_ground_syntax_and_scratch(tmp_path, L, W):
    testing.require_tool('clang++')
    flags = json.loads((HERE/'configs/C3.json').read_text())['kernel_flags']
    flags += ['STRATO', 'PROFILE']
    stub = tmp_path/'stub.h'
    stub.write_text(STUB+SYNTAX_STUB_EXTRA)
    code = tmp_path/'kernel.cpp'
    code.write_text(kernel_source(L))
    assert workspace(L) >= 162  # Y rows must hold the phase/LUT-index scratch.
    for raysplit in (False, True):
        selected = flags + ([] if raysplit else ['EXP_SECTION_CLOCKS'])
        fused = Kernel(flags=selected, L=L, block_size=256)
        plain = Kernel(flags=['STRATO'], L=L,
                       block_size=fused.bs(first=True))
        # Phase 2 reuses the existing Y rows, including the 162-float L16 pad.
        assert fused.shared_bytes(W, 2, first=True) == plain.shared_bytes(W, 2, first=True)
        for phase in (0, 2):
            for mie in (False, True):
                options = defines(W, selected+(['TROPO_MIE'] if mie else []), (1, phase, 0))
                result = subprocess.run(
                    ['clang++', '-std=c++17', '-fsyntax-only', '-x', 'c++',
                     '-include', str(stub), *options, str(code)],
                    capture_output=True, text=True)
                assert result.returncode == 0, result.stderr

