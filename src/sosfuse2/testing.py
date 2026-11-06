"""Test-only helpers for the optional sky-sos research extras.

The package tests run on their own. A test that reads a sky-sos record (results/, rig/, the v1 `sosfuse` oracle, the
legacy selftest fixture) asks for it through sky_sos(): with SOSFUSE2_SKY_SOS_ROOT unset the test is skipped, with the
variable set but the record missing the test FAILS (a wrong root or a moved record must not look like a pass).
Both unittest and pytest honour unittest.SkipTest, so the same helper serves the dual-mode harnesses.
"""
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from . import paths

ENV = 'SOSFUSE2_SKY_SOS_ROOT'
KNOWN_FAILURES_DOC = 'docs/KNOWN_FAILURES.md'


def require_tool(name):
    """Skip the calling test when an external executable (clang++, clang) is not on PATH (docs/CHECKS.md tier 1).

    Both unittest and pytest honour unittest.SkipTest. Inside a CLI harness (`python -m sosfuse2.test_strato_order --out …`)
    the exception propagates: a research run without the compiler stops with a readable reason instead of FileNotFoundError.
    """
    path = shutil.which(name)
    if path is None:
        raise unittest.SkipTest(f'E_TOOL_MISSING: {name!r} not on PATH')
    return path


def numpy_dispatch():
    """NumPy's runtime SIMD dispatch targets enabled on this CPU (the 'found' list of `numpy.show_runtime()`), e.g.
    ('X86_V3', 'X86_V4'). NPY_DISABLE_CPU_FEATURES is honoured. Empty when NumPy does not expose the tables."""
    try:
        from numpy._core._multiarray_umath import __cpu_dispatch__, __cpu_features__
    except ImportError:
        return ()
    return tuple(f for f in __cpu_dispatch__ if __cpu_features__.get(f))


def numpy_avx512():
    """True when NumPy dispatches AVX-512 kernels here (X86_V4 or an AVX512_* target)."""
    return any(f == 'X86_V4' or f.startswith('AVX512') for f in numpy_dispatch())


def known_failure(ident, reason, when=None):
    """Decorator for a case in the known failure set (entries in docs/KNOWN_FAILURES.md).

    Without `when` it is `xfail(strict=True)`: the case still runs, a failure is the expected outcome, and a pass turns the
    run red so the entry gets retired instead of lingering. With `when` (a bool) the entry is platform-dependent: where
    `when` is true the case is `xfail(strict=False)` (a pass is reported as XPASS, not an error, because the outcome there
    depends on the CPU), and elsewhere it runs as a normal test that must pass. Under the research CLI harnesses
    (`unittest.main()`, no pytest import) the decorator is a no-op and the case fails as before.
    """
    try:
        import pytest
    except ImportError:  # unittest.main() research harness without the test extra
        return lambda f: f
    text = f'{ident}: {reason} (see {KNOWN_FAILURES_DOC})'
    if when is None:
        return pytest.mark.xfail(strict=True, reason=text)
    return pytest.mark.xfail(bool(when), strict=False, reason=text)


def sky_sos_root():
    """The sky-sos checkout or None (unset). An env value that is not a directory is an error (paths.sky_sos_root
    raises E_SKY_SOS_ROOT), never a silent skip."""
    return paths.sky_sos_root()


def sky_sos(*relative):
    """Path of a sky-sos record: SkipTest when the root is unset, AssertionError when it is set and the record is missing."""
    root = sky_sos_root()
    rel = '/'.join(relative)
    if root is None:
        raise unittest.SkipTest(f'needs sky-sos record {rel} (set {ENV})')
    p = root.joinpath(*relative)
    if not p.exists():
        raise AssertionError(f'E_SKY_SOS_RECORD: {ENV} is set but {p} is missing')
    return p


def sky_sos_optional(*relative):
    """Path of a gitignored local copy inside sky-sos, or None when the root or the file is absent (the caller skips)."""
    root = sky_sos_root()
    if root is None:
        return None
    p = root.joinpath(*relative)
    return p if p.exists() else None


def scratch(prefix):
    """Fresh output folder for a harness run; under $TMPDIR, never inside the package (the old scripts/sosfuse2/scratchpad)."""
    return Path(tempfile.mkdtemp(prefix=prefix+'-', dir=scratch_parent()))


def scratch_parent():
    p = Path(os.environ.get('SOSFUSE2_TEST_SCRATCH', '').strip() or Path(tempfile.gettempdir())/f'sosfuse2-tests-{os.getuid()}')
    p.mkdir(parents=True, exist_ok=True)
    return p


def outside_package(path):
    """Harness --out folders must not land inside the installed package."""
    p = Path(path).resolve()
    try:
        p.relative_to(paths.PKG)
    except ValueError:
        return p
    raise ValueError(f'E_TEST_OUTPUT: {p} is inside the sosfuse2 package; use a folder under {scratch_parent()}')


def synthetic_o4(root):
    """Test-only, visibly synthetic positive spectrum; no HITRAN source or data."""
    import json
    import numpy as np
    from . import generate_minor_absorbers as gen
    from .minor_absorbers import O4_NAME, UNITS
    fine, grid = gen.fine_grid()
    values = np.where(grid <= 653.842, 1e-56*(1+np.exp(-((grid-577)/12)**2)), 0.)
    doc = dict(format='o4-fine-v1', **gen.common(fine), o4=values.tolist(),
               units=dict(wavelength_nm='nm (air)', o4=UNITS['o4']),
               source=dict(sha256=gen.CIA_SHA, note='SYNTHETIC TEST FIXTURE; not HITRAN measurements'))
    path = Path(root)/'assets'/O4_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, allow_nan=False)+'\n')
    return path
