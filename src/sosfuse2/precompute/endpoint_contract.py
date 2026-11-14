"""Geometry rules for the sosfuse2 two-layer COMMON endpoint cache.

The common-layer signature hashes THIS file instead of the whole
sosfuse2/solve/ray_geometry.py, so edits to timers, packing speed-ups or
documentation in solve/ray_geometry.py no longer invalidate the ~0.87 GB common layer.

Bump VERSION (and re-run the local test) whenever solve/ray_geometry.py changes any of:
  * endpoint generation in pack(): probe.boundaries, the extra shell crossings
    PACK_EXTRA_SHELLS_KM, the sort, the lower flag, end cell index/fraction,
    hits_ground (EARTH_RADIUS_KM + GROUND_TOLERANCE_KM);
  * the incoming quadrature call in TransientGeometry.cell (radius factor 4.);
  * the CSR rules (keep diff>0, empty ray = single 0.0, uint16 lower bits,
    uint32 edge offsets) or the packed layout (FIELDS/DTYPES/TRAIL/layout v3);
  * the column quadrature (Gauss-Legendre COLUMN_GL_ORDER) or which end_col
    column depends on the scale height.
Pure refactors preserving the cached layout and values within preparation tolerance do not need a bump.
The pack metadata also records product source identities.

check() is a cheap drift guard: it compares these constants with the live
geometry module and fails with E_GEOMETRY_CONTRACT_DRIFT; it cannot prove
numerical equivalence (checked separately at rtol 1e-6), so the bump rule above still applies.
"""

import hashlib
import inspect
import re
import json
from pathlib import Path

from ..foundation import content_identity as source_identity, paths

VERSION = 'sosfuse2-geometry-contract-4'
SIGNATURE_VERSION = 7  # v7 also identifies all endpoint-generation sources
EARTH_RADIUS_KM = 6371.0
GROUND_TOLERANCE_KM = 1e-5
PACK_EXTRA_SHELLS_KM = (10.0, 25.0, 40.0)
INCOMING_RADIUS_FACTOR = 4.0
COLUMN_GL_ORDER = 24
HEIGHT_DEPENDENT_END_COLUMNS = (1,)  # aerosol; 0 Rayleigh, 2 ozone, 3 NO2 and 4 O4 are fixed across scale heights
CSR = dict(
    keep='diff>0',
    empty_ray='single 0.0',
    lower='uint16, bit i = lower of i-th kept interval',
    edge_offset='uint32 global',
    kernel_skip='<=1e-8',
)
PACKED_LAYOUT_VERSION = 3
FIELDS = ('starts', 'targets', 'dirs', 'weights', 'edge_off', 'edges', 'lower', 'end_j', 'end_f', 'end_col', 'hits')
DTYPES = dict(
    cell_off='i8',
    starts='f8',
    targets='f8',
    dirs='f8',
    weights='f4',
    edge_off='u4',
    edges='f8',
    lower='u2',
    end_j='i4',
    end_f='f4',
    end_col='f4',
    hits='u1',
)
TRAIL = dict(starts=(3,), targets=(3, 3), dirs=(3,), end_col=(5,))
AEROSOL_COLUMN = dict(path='aerosol/H_<float.hex(height)>.f4', dtype='f4', replaces='end_col[:,1]')
# Upstream inputs that still generate the endpoints.
UPSTREAM = (
    'solve/engine.py',
    'rays/shell_segments.py',
    'rays/incoming_rings.py',
    'solve/engine_core.py',
    'atmosphere/tropo_profile.py',
    'rays/ray_columns.py',
    'atmosphere/density.py',
    'atmosphere/components.py',
    'solve/ray_geometry_parallel.py',
    'precompute/column_parallel.py',
    'jobs/run_state.py',
)
ROOT = paths.PKG  # package dir (public split: upstream definitions are editable sources)


def check(gm):
    """Fail if solve/ray_geometry.py visibly drifted from the constants above."""
    problems = []
    if tuple(gm.FIELDS) != FIELDS:
        problems.append('FIELDS')
    if dict(gm.DTYPES) != DTYPES:
        problems.append('DTYPES')
    if dict(gm.TRAIL) != TRAIL:
        problems.append('TRAIL')

    # The needles are compared with all whitespace removed, so a formatter pass is not drift; their numeric spellings are
    # the formatted ones (10.0, 6371.0, 4.0).
    def squeeze(source):
        return re.sub(r'\s+', '', source)

    pack = squeeze(inspect.getsource(gm.pack))
    cell = squeeze(inspect.getsource(gm.TransientGeometry.cell))
    column = squeeze(inspect.getsource(gm.aerosol_column))
    csr = squeeze(inspect.getsource(gm._csr_vector))
    for needle, text, name in (
        ('forheightin(10.0,25.0,40.0):', pack, 'pack shells'),
        ('integrate(p,d,old,24,', pack, 'pack columns'),
        ('re<6371.0+1e-5', pack, 'hits_ground'),
        ('4.0,True)', cell, 'incoming radius factor'),
        ('leggauss(24)', column, 'column quadrature'),
        ('np.diff(edges,axis=1)>0', csr, 'CSR keep'),
        ("flags.astype('u2')", csr, 'CSR flags'),
    ):
        if needle not in text:
            problems.append(name)
    if problems:
        raise RuntimeError(
            'E_GEOMETRY_CONTRACT_DRIFT: '
            + ', '.join(problems)
            + ' (update sosfuse2/precompute/endpoint_contract.py and bump VERSION)'
        )


def identity(s):
    """Common-layer identity: grid + quadrature counts + geometry rules file + upstream hashes; no scale height."""
    files = {'sosfuse2/' + name: source_identity.file_sha(ROOT / name) for name in UPSTREAM}
    files[paths.rel(__file__)] = source_identity.file_sha(Path(__file__))
    content = dict(
        h=s.geo.h.tolist(),
        chi=s.geo.chi.tolist(),
        cfg=s.geo.cfg,
        version=SIGNATURE_VERSION,
        contract=VERSION,
        layer='common: all endpoint arrays; end_col[:,1] replaced per scale height',
        files=files,
    )
    from ..rays.column_table import identity as column_identity

    content['column_method'] = column_identity()
    from ..rays.column_table import molecular

    content['minor_profiles'] = [molecular(None, k) for k in (2, 3)]
    return content, hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:20]
