"""CPU reference engine with an explicit context for each state."""

import collections, ctypes, hashlib, json, os, resource, shutil, subprocess, sys, tempfile, types, warnings
from pathlib import Path
from collections.abc import Mapping
from functools import partial
import numpy as np
from ..foundation import paths
from ..rays import (
    shell_segments,
    incoming_quadrature,
    incoming_rings,
    zenith_interpolation,
)
from ..atmosphere import extinction
from . import engine_core
from ..foundation import content_identity as source_identity
from ..atmosphere.tropo_profile import ExponentialProfile

ROOT = paths.PKG  # package sources and assets
HERE = Path(__file__).resolve().parent
R, Z = shell_segments.R, shell_segments.Z
OBS = 0.1
SOURCE_FILES = [
    'rays/shell_segments.py',
    'atmosphere/extinction.py',
    'rays/incoming_quadrature.py',
    'rays/incoming_rings.py',
    'rays/zenith_interpolation.py',
    'solve/engine_core.py',
    'outputs/preview_functions.py',
    'atmosphere/tropo_profile.py',
    'assets/spectra.json',
]
SPECTRA = ROOT / 'assets/spectra.json'


def cell_folder(cache, h, chi, cfg):
    """Reuse cells only for the code, layout and grid that wrote them.

    Native transport requires 18-float point rows. Optics stay the caller's, with one folder per state.
    """
    from sosfuse2.atmosphere import components, density

    files = fingerprint()
    files.update(
        {
            paths.rel(p): source_identity.file_sha(p)
            for p in (ROOT / 'atmosphere/density.py', ROOT / 'atmosphere/components.py')
        }
    )
    key = dict(
        version=2,
        components=list(components.COMPONENTS),
        no2_asset_sha256=density.no2_asset()[2],
        files=files,
        h=np.asarray(h, float).tolist(),
        chi=np.asarray(chi, float).tolist(),
        cfg=cfg,
    )
    out = cache / 'blocks' / hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:20]
    out.mkdir(parents=True, exist_ok=True)
    return out


def peak():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == 'darwin' else 1024)


def fingerprint():
    """Package-relative keys ('sosfuse2/rays/shell_segments.py') of everything the accepted CPU engine reads or compiles."""
    files = [ROOT / p for p in SOURCE_FILES] + [
        HERE / 'engine.py',
        ROOT / 'outputs/pack_state.py',
        ROOT / 'backends/native/sh_cpu.cpp',
        ROOT / 'assets/rgb_weights.npy',
    ]
    return {paths.rel(p): source_identity.file_sha(p) for p in files}


def compile_kernel(threads=2, force=False):
    source = paths.NATIVE / 'sh_cpu.cpp'
    digest = hashlib.sha256(source.read_bytes() + str(threads).encode()).hexdigest()
    # build products are keyed by the source bytes and the thread count only, not by host, although built with
    # -march=native; they live in the work root, never inside the package
    target = paths.work_root() / 'build' / f'kernel_t{threads}_{digest[:12]}.so'
    target.parent.mkdir(parents=True, exist_ok=True)
    stamp = target.with_suffix('.json')
    if not force and target.exists():
        try:
            stored = json.loads(stamp.read_text())
        except (OSError, ValueError):
            stored = None
        if isinstance(stored, dict) and stored.get('source_hash') == digest:
            return target
    compiler = shutil.which('g++') or shutil.which('clang++')
    if not compiler:
        raise RuntimeError(
            'E_COMPILER_MISSING: integrated full_transport has no NumPy fallback; install g++ or clang++'
        )
    tmp = target.with_suffix(f'.{os.getpid()}.tmp')
    args = [
        compiler,
        '-std=c++17',
        '-O3',
        '-march=native',
        '-fPIC',
        '-shared',
        '-pthread',
        f'-DRIG_THREADS={threads}',
        str(source),
        '-o',
        str(tmp),
    ]
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('E_COMPILE_FAILED: no integrated NumPy fallback exists; compilation is required') from e
    os.replace(tmp, target)
    fd, name = tempfile.mkstemp(dir=stamp.parent, prefix=stamp.name + '.')
    stamp_tmp = Path(name)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(json.dumps(dict(source_hash=digest, command=args)) + '\n')
        os.replace(stamp_tmp, stamp)
    finally:
        stamp_tmp.unlink(missing_ok=True)
    return target


class Kernel:
    # Two-layer aerosol (tropo_profile): the parameter vector follows the per-wavelength aerosol scattering coefficients and
    # the native transport reads it only when it is handed aerosol_height < 0 (backends/native/sh_cpu.cpp). None = exponential profile.
    aerosol_vector = None

    def __init__(self, threads, height, g):
        self.lib = ctypes.CDLL(str(compile_kernel(threads)))
        self.lib.basis.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.full_transport.argtypes = (
            [ctypes.c_int] * 7 + [ctypes.c_void_p] * 20 + [ctypes.c_double] * 3 + [ctypes.c_void_p]
        )
        self.height, self.g = height, g

    def basis(self, dirs, L):
        dirs = np.ascontiguousarray(dirs, 'f8')
        out = np.empty((len(dirs), (L + 1) * (L + 2) // 2))
        self.lib.basis(len(dirs), L, dirs.ctypes.data, out.ctypes.data)
        return out

    def transport(self, z, s, field, first=False):
        if self.aerosol_vector is None:
            asca, height = s.asca, self.height
        else:
            asca, height = np.r_[s.asca, self.aerosol_vector], -1.0
        out = np.empty((len(z['nu']), s.W, 2, 4))
        arrays = [
            s.active,
            z['off'],
            z['base'],
            z['lit'],
            z['point'],
            z['nu'],
            z['target'],
            z['dloc'],
            field,
            s.ext,
            asca,
            s.geo.chi,
            s.hr,
            s.ha,
            s.geo.poly,
            s.geo.left,
            s.goff,
            s.gdir,
            s.gval,
            s.gm,
        ]
        arrays = [np.ascontiguousarray(x) for x in arrays]
        self.lib.full_transport(
            len(out),
            s.W,
            len(s.geo.chi),
            s.geo.ns,
            s.mode,
            int(first),
            0,
            *[x.ctypes.data for x in arrays],
            s.delta,
            height,
            self.g,
            out.ctypes.data,
        )
        return out


class SunGeometryBinder:
    """Bind only the aerosol inputs (profile, scale height, descriptor) while retaining the table identity helpers."""

    def __init__(self, aerosol, aerosol_height, descriptor):
        self.aerosol = aerosol
        self.aerosol_height = aerosol_height
        self.descriptor = descriptor

    def __call__(self, scale, folder):
        return shell_segments.SunGeometry(
            scale, folder, aerosol=self.aerosol, aerosol_height=self.aerosol_height, descriptor=self.descriptor
        )

    def identity(self, scale):
        return shell_segments.SunGeometry.identity(scale, self.aerosol_height, self.descriptor)

    stored_identity = staticmethod(shell_segments.SunGeometry.stored_identity)


class EngineContext(Mapping):
    """Per-state bindings; read-only scope indexing is retained for one release."""

    _keys = ('b', 'Solver', 'basis_at', 'pack', 'C', 'AER_PROFILE', 'AER_HEIGHT', 'AER_G', 'R', 'Z')
    R, Z = R, Z
    basis_at = staticmethod(engine_core.basis_at)

    def __init__(self, b, height, g, profile, prepare_ground):
        self.b = b
        self.C = 3 / (16 * np.pi)
        self.AER_HEIGHT, self.AER_G, self.AER_PROFILE = height, g, profile
        self.prepare_ground = prepare_ground

    def Solver(self, g, kernel, wavelengths, mode=0, delta=0.03):
        return engine_core.Solver(self, g, kernel, wavelengths, mode, delta)

    def pack(self, p, d, h, ch, sun, n):
        return engine_core.pack(self, p, d, h, ch, sun, n)

    @staticmethod
    def _deprecated():
        warnings.warn(
            "sosfuse2 engine scope indexing is deprecated: use the context attributes (scope.b, scope.Solver, …)",
            DeprecationWarning,
            stacklevel=3,  # the caller of the mapping method
        )

    def __bool__(self):
        return True  # a scope is always truthy; len() would warn

    def __getitem__(self, key):
        self._deprecated()
        if key not in self._keys:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        self._deprecated()
        return iter(self._keys)

    def __len__(self):
        self._deprecated()
        return len(self._keys)

    def __contains__(self, key):
        self._deprecated()
        return key in self._keys

    def get(self, key, default=None):
        self._deprecated()
        return getattr(self, key) if key in self._keys else default

    def keys(self):
        self._deprecated()
        return collections.abc.KeysView(self)

    def items(self):
        self._deprecated()
        return collections.abc.ItemsView(self)

    def values(self):
        self._deprecated()
        return collections.abc.ValuesView(self)

    def _readonly(self, *args, **kwargs):
        raise TypeError('E_ENGINE_SCOPE_READONLY: use context attributes')

    __setitem__ = __delitem__ = pop = popitem = update = clear = setdefault = _readonly


class Geometry:
    build = engine_core.build

    def __init__(self, ctx, state_id, h, ch, cache, cfg, sun, mode):
        self.ctx = ctx
        nmu, nphi, path_n = cfg["nmu"], cfg["nphi"], cfg["path_n"]
        self.name = state_id
        self.h, self.chi = h, ch
        self.ns = len(h) * len(ch)
        self.folder = cache
        self.cache = cell_folder(cache, h, ch, dict(nmu=nmu, nphi=nphi, path_n=path_n))
        self.cfg = dict(nmu=nmu, nphi=nphi, path_n=path_n)
        self.compact = False
        self.poly, self.left = zenith_interpolation.chi_poly(ch)
        self.sun = sun
        self.query_n = 24 if mode == 'full' else 6
        self.ram = collections.OrderedDict()
        self.ram_bytes = 0
        self.ram_limit = 64 * 1024**2
        self.meta = dict(
            cells=self.ns,
            nr=len(h),
            nchi=len(ch),
            incoming_rays=0,
            points=0,
            geometry_seconds=0.0,
            first_seconds=0.0,
            fold_seconds=0.0,
            compression_seconds=0.0,
            io_seconds=0.0,
            largest_block_bytes=0,
            largest_basis_bytes=0,
            solid_angle_max_relative=0.0,
            cache_bytes=0,
        )

    def pack(self, p, d, n):
        return self.ctx.pack(p, d, self.h, self.chi, self.sun, n)

    def cell(self, i, s):
        if i in self.ram:
            z = self.ram.pop(i)
            self.ram[i] = z
            return z
        path = self.cache / f'cell_{i:05d}.npz'
        if path.exists():
            with np.load(path) as f:
                z = {k: f[k] for k in f.files}
        else:
            z = self.build(i, s)
            if shutil.disk_usage(self.folder).free < 1_000_000_000:
                raise RuntimeError('E_DISK_LOW: <1 GB free; checkpoint retained')
            tmp = path.with_suffix('.tmp')
            with tmp.open('wb') as f:
                np.savez_compressed(f, **z)
            os.replace(tmp, path)
        size = sum(v.nbytes for v in z.values())
        while self.ram and self.ram_bytes + size > self.ram_limit:
            _, v = self.ram.popitem(last=False)
            self.ram_bytes -= sum(q.nbytes for q in v.values())
        if size <= self.ram_limit:
            self.ram[i] = z
            self.ram_bytes += size
        return z


def create(
    state,
    cache,
    mode,
    threads,
    wavelength=550.0,
    *,
    observer_km=None,
    kernel_factory=None,
    grid=None,
    spec=None,
    sun_geometry=None,
    prepare_ground=True,
    aerosol=None,
):
    """Create a solver, its private context, and the selected settings.

    Keyword-only hooks provide the state inputs used by the imported solver modules:
      observer_km     observer height in km (module OBS when None); read by the column, sky and settings records.
      kernel_factory  callable(threads, H, g) -> transport kernel; the native Kernel above when None.
      grid            callable(mode, h, chi, nmu, nphi, path_n) -> same tuple, applied after the mode's grid is chosen and
                      before Geometry captures it (production refines h/chi and takes the counts from its settings).
      spec            the spectral table dict (assets/spectra.json when None). create() completes it in place (NO2 defaults),
                      so pass a fresh dict per call.
      sun_geometry    callable(old, cache, H) -> sun geometry; old.SunGeometry(1, cache) when None.
      prepare_ground  False defers ground preparation for solvers created through this context. Set
                      context.prepare_ground to True to prepare subsequent solvers, or call solver.prepare_ground().
      aerosol         None (exponential profile) or a namespace with profile(h), column (aod normalisation), folder(cache)
                      (sun-cache folder per profile) and vector (two-layer parameters for the native kernel): tropo_profile.engine_hooks."""
    params = state['parameters']
    H = params['scale_height_km']
    gg = params['g']
    obs = OBS if observer_km is None else observer_km
    spectral = json.loads(SPECTRA.read_text()) if spec is None else spec
    profile = ExponentialProfile(H, clamp=True) if aerosol is None else aerosol.profile
    descriptor = (
        dict(kind='exp', height=float(H))
        if aerosol is None
        else dict(kind='two_layer', vector=np.asarray(aerosol.vector).tolist())
    )
    probe = types.SimpleNamespace(
        gl=shell_segments.gl, frame=shell_segments.frame, boundaries=shell_segments.boundaries
    )
    old = types.SimpleNamespace(
        R=R,
        Z=Z,
        OBS=obs,
        SPEC=spectral,
        AER_PROFILE=profile,
        AER_HEIGHT=H,
        probe=probe,
        grids=shell_segments.grids,
        profiles=partial(shell_segments.profiles, profile),
        geometry_segments=partial(shell_segments.geometry_segments, profile),
        total_columns=partial(shell_segments.total_columns, profile),
        coefficients=partial(extinction.coefficients, spectral),
        SunGeometry=SunGeometryBinder(profile, H, descriptor),
        aerosol_descriptor=descriptor,
    )
    from sosfuse2.atmosphere import density, components, minor_absorbers

    optical = density.build(state)
    amounts = components.optics_state(state, optical)
    if not components.reference_active() and 'no2' not in spectral:
        spectral.update(minor_absorbers.representative(None, spectral['wavelength_nm'], o4=amounts['o4']))
    column = H * (-np.expm1(-Z / H)) if aerosol is None else aerosol.column
    old.NORMAL = dict(
        aer=params['aod550'] / column,
        ssa=params['ssa'],
        ang=params['angstrom'],
        g=gg,
        albedo=params['albedo'],
        ozone_DU=params['ozone_du'],
        **amounts,
        no2_H_f_km=optical['meta']['no2']['H_f_km'],
    )
    sh = types.SimpleNamespace(
        old=old, NORMAL=old.NORMAL, OBS=obs, incoming=incoming_quadrature.incoming, spatial_lookup=incoming_quadrature.spatial_lookup
    )
    legacy = types.SimpleNamespace(ring_edges=incoming_rings.ring_edges, incoming=incoming_rings.incoming)
    b = types.SimpleNamespace(
        sh=sh,
        R=R,
        legacy=legacy,
        peak=peak,
        DIRS=[
            ('sun', 2, 0),
            ('sun', 10, 0),
            ('sun', 30, 0),
            ('anti', 10, 180),
            ('anti', 30, 180),
            ('side', 10, 90),
            ('zenith', 89, 0),
        ],
        HS=(2, -3, -6, -9, -12),
    )
    ctx = EngineContext(
        b, H, gg, ExponentialProfile(H, clamp=False) if aerosol is None else aerosol.profile, prepare_ground
    )
    if mode == 'full':
        h, ch = old.grids(1)
        nmu, nphi, path_n = 12, 64, 12
    elif mode == 'reduced':
        fullh, fullch = old.grids(1)
        h = fullh[[0, 2, 4, 6, 8, 10, 12, 14, 17, 19]]
        degrees = np.rad2deg(fullch)
        ch = np.deg2rad(np.unique(np.r_[degrees[(degrees < 85) | (degrees > 110)], np.arange(85, 111, 1)]))
        nmu, nphi, path_n = 2, 4, 6
    else:
        h = np.array([0.0, 0.1, 8.0, 80.0])
        ch = np.deg2rad([0.0, 60.0, 85.0, 90.0, 93.0, 96.0, 105.0, 110.0, 140.0, 180.0])
        nmu, nphi, path_n = 1, 2, 4
    if grid is not None:
        h, ch, nmu, nphi, path_n = grid(mode, h, ch, nmu, nphi, path_n)
    sun_cache = cache if aerosol is None or aerosol.folder is None else aerosol.folder(cache)

    kernel = (Kernel if kernel_factory is None else kernel_factory)(threads, H, gg)
    if (
        aerosol is not None and aerosol.vector is not None and isinstance(kernel, Kernel)
    ):  # only the native kernel takes the vector; source_solver.Kernel reads the profile in setup()
        kernel.aerosol_vector = np.ascontiguousarray(aerosol.vector, 'f8')
    cache.mkdir(parents=True, exist_ok=True)
    sun = old.SunGeometry(1, sun_cache) if sun_geometry is None else sun_geometry(old, sun_cache, H)
    geo = Geometry(ctx, state["id"], h, ch, cache, dict(nmu=nmu, nphi=nphi, path_n=path_n), sun, mode)
    solver = ctx.Solver(geo, kernel, [wavelength], 1, 0.03)
    solver.active[1::2] = 0
    return (
        solver,
        ctx,
        dict(
            nr=len(h),
            chi_step_deg=0.5 if mode == 'full' else 1 if mode == 'reduced' else None,
            nchi=len(ch),
            nmu=nmu,
            nphi=nphi,
            path_n=path_n,
            L=32,
            interpolation='log',
            observer_km=obs,
        ),
    )
