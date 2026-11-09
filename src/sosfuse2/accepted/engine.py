"""Per-call solver scopes; definition bodies live in editable solver_*.py sources."""

import ast, collections, ctypes, hashlib, json, math, os, resource, shutil, subprocess, sys, tempfile, time, types
from pathlib import Path
import numpy as np
from .. import paths
from ..tropo_profile import ExponentialProfile

ROOT = paths.PKG  # package sources and assets
HERE = Path(__file__).resolve().parent
R, Z, OBS = 6371.0, 80.0, 0.1
THREAD_KEYS = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'BLIS_NUM_THREADS',
)
SOURCE_FILES = [
    'solver_columns.py',
    'solver_optics.py',
    'solver_sky.py',
    'solver_legacy.py',
    'solver_poly.py',
    'solver_core.py',
    'solver_render.py',
    'assets/spectra.json',
]
SPECTRA = ROOT / 'assets/spectra.json'


def definitions(relative, names, scope):
    path = ROOT / relative
    text = path.read_text()
    tree = ast.parse(text)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise RuntimeError(f'E_ADAPTER_DRIFT: {relative}: missing definition')
    scope['SOURCE_PATH'] = path  # exec scopes have no __file__; identity uses this explicit source input
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return types.SimpleNamespace(**scope)


def cell_folder(cache, h, chi, cfg):
    """blocks/<identity>: a cell is reused only by the code, layout and grid that wrote it. Before B2 a point row was 14
    floats and the native transport now reads 18 (5 components), so an unkeyed blocks/ folder would hand old rows to it.
    Optics stay the caller's (one folder per state), as before."""
    from sosfuse2 import components, density

    files = fingerprint()
    files.update(
        {
            paths.rel(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / 'density.py', ROOT / 'components.py')
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
    """Package-relative keys ('sosfuse2/solver_columns.py') of everything the accepted CPU engine reads or compiles."""
    files = [ROOT / p for p in SOURCE_FILES] + [
        HERE / 'engine.py',
        ROOT / 'host/pack_state.py',
        ROOT / 'native/sh_cpu.cpp',
        ROOT / 'assets/rgb_weights.npy',
    ]
    return {paths.rel(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def compile_kernel(threads=2, force=False):
    source = paths.NATIVE / 'sh_cpu.cpp'
    digest = hashlib.sha256(source.read_bytes() + str(threads).encode()).hexdigest()
    # build products are per source digest and per host (-march=native), never inside the package
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
            'E_COMPILER_MISSING: integrated full_transport has no NumPy fallback; install g++ or clang++ on rig'
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
    # the native transport reads it only when it is handed aerosol_height < 0 (native/sh_cpu.cpp). None = exponential profile.
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
    """One solver in a private scope. The keyword-only hooks (1.2.0) are the former text patches of run_gpu_state_v3 and
    tropo_profile made explicit; every default reproduces the call without the hook, so the module's own callers (research
    CLIs, selftests) see no change:
      observer_km     observer height in km (module OBS when None); read by the column, sky and settings records.
      kernel_factory  callable(threads, H, g) -> transport kernel; the native Kernel above when None.
      grid            callable(mode, h, chi, nmu, nphi, path_n) -> same tuple, applied after the mode's grid is chosen and
                      before Geometry captures it (production refines h/chi and takes the counts from its settings).
      spec            the spectral table dict (assets/spectra.json when None). create() completes it in place (NO2 defaults),
                      so pass a fresh dict per call.
      sun_geometry    callable(old, cache, H) -> sun geometry; old.SunGeometry(1, cache) when None.
      prepare_ground  False defers the ground preparation: Solver.prepare_ground is replaced by a no-op on the solver class of
                      this scope and the original is kept as scope['HOST_ORIGINAL_GROUND']; the caller restores it after
                      every Solver it builds in this scope (run_gpu_state_v3.create_configured).
      aerosol         None (exponential profile) or a namespace with profile(h), column (aod normalisation), folder(cache)
                      (sun-cache folder per profile) and vector (two-layer parameters for the native kernel): tropo_profile.engine_hooks."""
    params = state['parameters']
    H = params['scale_height_km']
    gg = params['g']
    obs = OBS if observer_km is None else observer_km
    common = dict(np=np, math=math, time=time, json=json, Path=Path, R=R, Z=Z, ROOT=ROOT)
    probe = definitions('solver_columns.py', ['gl', 'frame', 'boundaries'], dict(common))
    oldscope = dict(common, probe=probe, OBS=obs, SPEC=json.loads(SPECTRA.read_text()) if spec is None else spec)
    from sosfuse2 import density, components, minor_absorbers

    optical = density.build(state)
    amounts = components.optics_state(state, optical)
    if not components.reference_active() and 'no2' not in oldscope['SPEC']:
        oldscope['SPEC'].update(
            minor_absorbers.representative(None, oldscope['SPEC']['wavelength_nm'], o4=amounts['o4'])
        )
    definitions('solver_optics.py', ['coefficients'], oldscope)
    old = definitions(
        'solver_columns.py', ['profiles', 'geometry_segments', 'total_columns', 'SunGeometry', 'grids'], oldscope
    )

    oldscope['AER_PROFILE'] = ExponentialProfile(H, clamp=True) if aerosol is None else aerosol.profile
    oldscope['AER_HEIGHT'] = H
    old.AER_HEIGHT = H
    old.aerosol_descriptor = (
        dict(kind='exp', height=float(H))
        if aerosol is None
        else dict(kind='two_layer', vector=np.asarray(aerosol.vector).tolist())
    )
    # Functions share oldscope, so state-dependent profiles are used by every optical column.
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
    shscope = dict(common, old=old, ctypes=ctypes, NORMAL=old.NORMAL, OBS=obs)
    sh = definitions('solver_sky.py', ['incoming', 'spatial_lookup'], shscope)
    legacy = definitions('solver_legacy.py', ['ring_edges', 'incoming'], dict(common, sh=sh))
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
    polys = definitions('solver_poly.py', ['chi_poly'], dict(common))
    scope = dict(common, b=b, C=3 / (16 * np.pi), AER_HEIGHT=H, AER_G=gg)
    if aerosol is not None:
        scope['AER_PROFILE'] = aerosol.profile
    scope.setdefault(
        'AER_PROFILE', ExponentialProfile(H, clamp=False)
    )  # the solver's own default has no clamp (kept as is)
    definitions('solver_core.py', ['basis_at', 'pack', 'Solver'], scope)
    AcceptedSolver = scope['Solver']
    if not prepare_ground:
        scope['HOST_ORIGINAL_GROUND'] = AcceptedSolver.prepare_ground
        AcceptedSolver.prepare_ground = lambda self: None
    definitions('solver_core.py', ['build'], scope)
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

    class Geometry:
        build = scope['build']

        def __init__(self):
            self.name = state['id']
            self.h, self.chi = h, ch
            self.ns = len(h) * len(ch)
            self.folder = cache
            self.cache = cell_folder(cache, h, ch, dict(nmu=nmu, nphi=nphi, path_n=path_n))
            self.cfg = dict(nmu=nmu, nphi=nphi, path_n=path_n)
            self.compact = False
            self.poly, self.left = polys.chi_poly(ch)
            self.sun = old.SunGeometry(1, sun_cache) if sun_geometry is None else sun_geometry(old, sun_cache, H)
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
            return scope['pack'](p, d, self.h, self.chi, self.sun, n)

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
                if shutil.disk_usage(cache).free < 1_000_000_000:
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

    kernel = (Kernel if kernel_factory is None else kernel_factory)(threads, H, gg)
    if (
        aerosol is not None and aerosol.vector is not None and isinstance(kernel, Kernel)
    ):  # only the native kernel takes the vector; fused.Kernel reads the profile in setup()
        kernel.aerosol_vector = np.ascontiguousarray(aerosol.vector, 'f8')
    geo = Geometry()
    solver = AcceptedSolver(geo, kernel, [wavelength], 1, 0.03)
    solver.active[1::2] = 0
    return (
        solver,
        scope,
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
