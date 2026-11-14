# sosfuse2

Radiance tables of the clear twilight sky for a spherical-shell atmosphere. The solver iterates successive orders of
scattering to convergence on a height and angle grid, accelerated with Anderson mixing, and includes polarised Rayleigh
scattering, a tropospheric aerosol layer, a stratospheric sulfate layer (on in the bundled v2 states), absorption by
ozone, NO₂ and O₄, and a Lambertian ground. One solve gives the sky at every solar elevation of a table, by default +6°
to −18°. The package ships a NumPy reference implementation, an optional CUDA kernel compiled at run time through CuPy
and NVRTC, numeric gates that compare against fixed references within tolerances, and a provenance record that signs
every table with the hashes of the code and data that produced it.

This is research code, published in the form we use it. Versions 1.0.0 to 1.2.3 were published together, and 1.3.3 is
the current release. `docs/CHANGELOG.md` lists the releases since 1.2.0, and *Tested configurations* records the
configurations the releases were verified on. The change that matters most for default output is in 1.2.3: the bundled
`states_v2.json` states now include the stratospheric sulfate layer, and the CLI has no `--stratosphere off` option. The
changelog explains how to reproduce the earlier output.

![Engine and MYSTIC radiance spectra at solar zenith angles 93, 96 and 99 degrees, and their
ratio](docs/validation/mystic_vs_engine.png)

Sun-side sky radiance at 20° elevation for solar elevations −3°, −6° and −9°, from the engine (lines) and from the Monte
Carlo code MYSTIC (markers) on the same optical input, with their ratio and the MYSTIC standard error. The dip near 600 nm
is the ozone Chappuis band. `docs/VALIDATION.md` gives the details.

## Why a spherical-shell solver

With the Sun 6° to 9° below the horizon, the sky overhead is lit only through the upper atmosphere: the Earth's shadow
stands about 35 km above the observer at 6° and about 79 km at 9°. A plane-parallel atmosphere has no such shadow, so the
model uses a spherical shell. Monte Carlo codes handle this geometry, but their radiances carry sampling noise. The MYSTIC
runs for the comparison below, one direction at three solar elevations and 30 wavelengths, took 6.6 CPU-hours to reach a
standard error of 0.5–2.4 % per value. sosfuse2 solves for a source field instead of sampling photons, so a table of
100 × 180 directions and 45 bands is free of sampling noise and covers all solar elevations at once. `docs/MODEL.md`
describes the physics, the grids and the iteration. `docs/PERFORMANCE.md` records where the GPU time goes, how far the
kernels sit from the hardware limits, what was tried, and the question about the main kernel that is still open.

## Validation

On the same optical input, the engine and MYSTIC (libRadtran 2.0.6) agree to within 2.2 % per wavelength at solar zenith
angles 93° and 96°, where the MYSTIC standard error is below 1 %. At 99° the ratios spread over 0.958–1.063, consistent
with the 2.0–2.4 % MYSTIC error; at 99° the largest difference is 2.53 MYSTIC standard errors, and
at 93° the 587.17 nm cell reaches 3.03. The colour differences are |Δu′| ≤ 0.0012
and |Δv′| ≤ 0.00014, at most 1.7 standard errors. The MYSTIC build reproduces the 120 cells of the spherical Rayleigh
benchmark of Korkin et al. (2020) to 0.27 %. This comparison covers one state, one direction and three solar elevations.
It used a table baked with the research build, with the 1.0.0 physics and the legacy spectral inputs rather than the
1.0.0 defaults, and because both codes read the same input it tests the transport rather than the input physics. For
1.1.0, paired MYSTIC runs on the engine's input measure the change caused by NO₂ and O₄ at 12 pixels of two states and
solar zenith angles 93°–99°. Of 108 values of u′, v′ and Y, 66 reproduce MYSTIC's change to 10 % or to 1e-4, none fails,
and 42 are left unresolved by the Monte Carlo noise. `docs/VALIDATION.md` gives the set-up, the tables, the size of the
1.1.0 changes, the sensitivity to the fixed choices listed below, and a qualitative check of the stratospheric aerosol
input against measured zenith polarisation (Ugolnikov et al. 2004).

In those two MYSTIC comparisons the stratospheric layer is centred at 19 km (τ550 0.0037 in the first state, 0.0022 in
the second), and the polarisation check reads the input of the first state. The default layer of the bundled states is
centred at 21 km with τ550 0.002–0.006. A third comparison uses 1.2.3 tables of eight bundled states with the three US
Standard Atmosphere options listed under *Limitations*, at six pixels and solar zenith angles 93° and 96°. At 93° the
luminance agrees with MYSTIC to within 0.8 % in all 48 combinations of state and pixel, with MYSTIC 0.18 % to 0.35 %
brighter in the mean at elevations up to 20.25°, and a grid of 0.25° in solar zenith angle instead of 0.5° removes part of
this difference. At 96° the 95 % interval of the luminance difference excludes 0 in 2 of
48 combinations, as many as chance would give. `docs/VALIDATION.md` also reports two checks of deeper twilight, down to
a solar elevation of −18°: second-order scattering against a direct quadrature, and a Rayleigh atmosphere against
SASKTRAN2. In the IPRT spherical test case E1, at solar zenith angles 93° and 96°, the default quadrature does not meet
the error limits we had set, which we attribute to the layered profile of the case. The numeric gates in
`docs/CHECKS.md` show that a change leaves the tables within tolerance of fixed references, which guards against
regressions across refactors and other hardware; they do not measure physical accuracy.

## Limitations

- The default density row, exp(−h / 8 km), holds more air above 20 km than the US Standard Atmosphere 1976 at the same
  column (1.2 times at 20 km, 1.7 times at 30 km), and deep twilight comes out too bright. Against measured zenith and
  all-sky twilight brightness (Koomen et al. 1952; Patat et al. 2006), a table of a bundled state is 1.4 to 2.1 times
  the measurements in Maryland and at Paranal between solar elevations of −6° and −12°, and 2.9 to 3.2 times those at
  Sacramento Peak. Two variants of one state both reproduce the purple light of Lee and Hernández-Andrés (2003), and
  both use `ozone_profile: usstd76` and `rayleigh_column: usstd`. The variant with the exponential density profile and a
  stratospheric τ550 of 0.006 gives 1.6 to 2.1 times the Maryland and Paranal measurements, and the variant with
  `density_profile: usstd76` and τ550 0.0037 gives 0.65 to 1.04 times. The default is kept; for deep twilight, set
  `density_profile: usstd76`, `ozone_profile: usstd76` and `rayleigh_column: usstd` in the state's `density` block, as
  `states_v1_usstd.json` does.
- NO₂ uses a fixed AFGL profile shape and 220 K cross sections, with a default column of 0.207 DU (`parameters.no2_du`).
  Real columns vary with location, season and time of day, and the model has no photochemistry. In MYSTIC runs at ten
  twilight pixels, replacing the fixed 220 K (NO₂) and 233 K (ozone) cross sections with per-layer temperatures changed
  the colour by at most 1.8e-4 in u′v′ and Y by at most 0.04 % (95 % bounds).
- O₄ uses fixed 293/287 K cross sections over approximately 297–654 nm, within the engine's 360–840 nm grid, and the
  square of the active air density. O₂ A- and B-band CIA, monomer O₂ lines and water vapour are excluded; the excluded CIA
  bands reach vertical optical depths of about 2.2e-3 (A) and 1.7e-4 (B) in the reference profile. The O₄ cross sections
  have no temperature dependence in the model, and we have not tested its effect.
- Rayleigh scattering follows Bodhaine et al. (1999), with vacuum wavelength conversion and the existing 450 nm air
  anchor. The phase depolarisation stays fixed at 0.03; a single-scattering estimate with Bodhaine's wavelength-dependent
  value (0.0305 at 360 nm to 0.0276 at 830 nm) gives at most 3.9e-5 in u′v′ and 0.083 % in Y. NO₂ and O₄ contribute
  extinction only.
- The default states use a fixed Gaussian sulfate layer centred at 21 km with σ = 3 km and optical depths of
  0.002–0.006 at 550 nm. These background choices do not track the observed stratospheric aerosol burden.
- The tables hold total radiance only. The solver follows Rayleigh polarisation internally and treats aerosol scattering
  as depolarising, but writes no Stokes parameters.
- The numeric tolerances bound the discretisation and iteration error against fixed references of the same model. They
  say nothing about the physical error.
- We have not compared the colours with measured twilight colours (Lee and Hernández-Andrés 2003; Lee 2026).

## Install

```sh
pip install .                 # from a checkout; or `pip install <sdist>` outside the source tree
pip install ".[test]"         # + pytest
pip install ".[gpu,test]"     # + CuPy (CUDA 12) and the NVIDIA runtime/NVRTC/cuBLAS wheels
```

The package needs Python ≥ 3.12, NumPy ≥ 2.0 and Pillow, plus two programs on `PATH`. A C++ compiler, `g++` or
`clang++`, builds the CPU spherical-harmonics helper `native/sh_cpu.cpp` once into the work root; without one the run
stops with `E_COMPILER_MISSING`, and the kernel syntax tests call `clang++` specifically. `zstd` packs state outputs as
`.tar.zst`; without it the run stops with `E_ZSTD_MISSING`.

Downloaded inputs, the geometry cache and the CPU helper build go to `SOSFUSE2_WORK_ROOT`, by default
`$XDG_CACHE_HOME/sosfuse2`. A solve (`run_gpu_state_v3`, `pipeline`) writes its tables, intermediate files and packs under
the `--work-root` it is given, in `out/<id>`, `tmp/<id>` and `pack/`. The package directory is read-only data.

On a CUDA machine, NVRTC compiles the kernels on first use. `SOSFUSE2_COMPILE_CACHE=<dir>` keeps them between runs, and
`python -B -m sosfuse2.jobs.precompile --jobs 8` fills that cache before a queue starts.

## Quick start

A first check takes three commands: install, prepare the O₄ absorption data once, and solve one reduced-resolution state
with the NumPy backend. The solve takes about a minute on a laptop, 36 s with two threads in the measured run.

```sh
pip install ".[test]"
python -m sosfuse2.asset_tools.prepare_minor_absorbers --download
python -B -m sosfuse2.jobs.run_state --id clear_34 --reduced --backend numpy --threads 2 --work-root ./work
```

The table and its `meta.json` land under `./work/out/clear_34/`. `meta.json` carries `contract: "v4"`, the state, the
backend, the tolerance settings and the `code_hashes` of a fixed list of production modules, promoted solver definitions
and data assets, plus a `signature` over them. `docs/PROVENANCE_V3.md` defines the record and explains how readers that
also hold older `v3` tables compare them by logical key.

The O₄ step downloads the HITRAN O₂–O₂ collision-induced absorption file into the work root named by
`SOSFUSE2_WORK_ROOT`, verifies the SHA-256 of the source and prints the SHA-256 of the generated table.
`--cia O2-O2_2024.cia` builds from a local copy instead, and `SOSFUSE2_OFFLINE=1` refuses the download. The O₄ data are
not bundled (`THIRD_PARTY.md`), and a solve does not download them either: a missing or stale cache stops the run with
`E_O4_ASSET_MISSING` or `E_O4_ASSET_STALE`.

One other download can happen. A state that selects a `usstd76` density or ozone profile fetches the IPRT input on first
use; the default states do not. `SOSFUSE2_OFFLINE=1` refuses that download as well (`THIRD_PARTY.md` §9).

The state ids come from `sosfuse2/assets/states_v2.json`, the default `--states` file. It holds 64 parameter sets over
`aod550`, `angstrom`, `ssa`, `scale_height_km`, `g`, `ozone_du`, `albedo` and `stratosphere_tau550`; `states_v1.json`
holds the 40 first-generation states. Solar depression is the runner's own axis, set by `solar_elevation_deg` in its
settings, and is not part of a state. All 64 states in `states_v2.json` mark the stratospheric layer `enabled`, so it
is on when `--stratosphere` is omitted. The first-generation `states_v1.json` retains `unsupported`; its layer stays
off unless `--stratosphere on` is given. A full-resolution table needs the CUDA backend (`--backend cupy`).
The production queue that bakes many states with overlapping preparation is `python -B -m sosfuse2.jobs.pipeline`.

### Production queues and gates

The following matters for production queues and gates rather than for a first solve.

- Setting `parameters.no2_du=0` or `parameters.o4=false` turns off NO₂ or O₄ for research runs. `o4` must be a JSON
  boolean. When O₄ is on and the state uses an external air profile, that profile file must declare
  `air_surface_number_density_m3`.
- Old tables without a physics field remain readable as legacy products. To reproduce their numbers, use the package
  revision recorded in their provenance; `spectral_mode=legacy` selects the old spectral grid, not the old physics.
  Reference fixtures come in three generations, the legacy set, the 1.1.0 absorber-validation set and the private
  baseline set of the first public split, and one run cannot mix them. Reusing a saved source field for the same state
  also requires equal resolved `optics_state` values; an omitted amount and its explicit default produce the same
  signature.
- On CUDA, a state checks the minimum memory peak across its ground and source bundles before its first KSPLIT
  allocation. If it cannot fit alone, production runs use KSPLIT=0 for the entire state, whereas memory held by other
  service states only causes a wait. `SOSFUSE2_KSPLIT_FALLBACK=0`, the default for gates and benchmarks, turns the
  fallback into an `E_KSPLIT_MEMORY` error, and `1` permits it. A fallback is recorded once as `KSPLIT_FALLBACK` in the
  phase log and the pipeline summary, and the table metadata records `ksplit_effective`. On a 10 GB card the
  full-resolution gate state falls back, with a minimum peak of 7.06 GB against a budget of 6.74 GB.
- `plan_queue` stops with `E_PLAN_CPU_BUDGET` when GPUs x workers per GPU x column workers exceed `pipeline.cpu_budget()`.
  That bounds what the plan asks for; `pipeline.py` can still widen a state's column pool, so it is not a CPU quota.
  The plan also lists the `ground_aerosol01` host cache that follows `ground`, with an unmeasured cost.
- `tables_from_sources` holds `<out>/.worker.lock`, the lock that `run_gpu_state_v3` takes for the same output, while it
  builds a state. It skips a state that a worker owns, and a worker that starts meanwhile stops with `E_OUTPUT_BUSY`.
- Pipeline logs go to a separate `logs/<run>/` directory per invocation, referenced from `pipeline_summary.json`, so cold
  and warm runs keep both traces. Benchmark stage timers go to `timings-<run>.json`, with the file name recorded in
  `benchmark.json`.

## Numerical tolerances

The gates judge correctness by tolerances against fixed references rather than by byte identity:

| gate | compares | tolerance |
|---|---|---|
| G-P | CUDA kernel vs NumPy golden (fixed applies) | rtol 1e-3 |
| G-T | anchor radiances (`gate/anchors.json`) | ratchet, recorded |
| G-A / H / PS | queue parity, history, persistent service | per case |
| repeat (`gate --stage`) | same build, cold vs warm, repeated runs | rtol 1e-5, max difference recorded |
| G-F | full-grid fixed table (needs the research checkout) | rtol 1e-3 |

Accumulation order, lane mapping and record layout may change between versions, and such changes are judged against the
fixed references and their tolerances. `SOSFUSE2_DETERMINISTIC=1` selects deterministic accumulation for studies that
need bit-stable repeats. The baseline log and the reference manifest live in `src/sosfuse2/gate/baselines/`.

## Tests and checks

`docs/CHECKS.md` describes three tiers and exactly what runs where.

1. Any machine without a GPU, which is also the public CI: `python -B -m pytest -q -p no:cacheprovider --pyargs sosfuse2`
   from the installed package. The expected outcome is `0 failed` and 3 `xfailed`, or 2 where `clang++` is
   absent and one of the three skips. Tests that need the research records
   skip unless `SOSFUSE2_SKY_SOS_ROOT` points at that checkout, and tests that need `clang++` skip when it is absent. The
   tests use a synthetic O₄ table in an isolated temporary root and download no HITRAN data; reproduction from the
   original data is optional (`SOSFUSE2_HITRAN_CIA`, `SOSFUSE2_BIRA_DIR`) and skips when neither is supplied.
2. A CUDA machine: `python -B -m sosfuse2.checks.gate --all --out <dir>` runs G-P, G-T, G-A, H and PS, and
   `gate --stage --out <dir>` runs the repeat checks.
3. A CUDA machine with the research checkout: `gate --gf --out <dir>`.

Three test cases fail in the research baseline the package was split from and are marked `xfail(strict=True)`. Their ids,
causes and the rule for retiring them are in `docs/KNOWN_FAILURES.md`. Because the marker is strict, a case that starts
passing turns the run red until its entry is removed, so the failures stay visible.

## Tested configurations

1.2.1 and 1.2.2 change documentation only, so the 1.2.0 rows apply to them unchanged. The changed default v2 states in
1.2.3 are covered by the separate installed-sdist checks below.

1.3.3 changes documentation only. Its code is that of 1.3.2, and no test or gate was run again for it.

Configurations verified for 1.3.2. The kernel, the solver and the gate are those of 1.3.0, so the GPU row of 1.3.0
applies to 1.3.2 and the gates were not run again:

| platform | Python | backend | what was run |
|---|---|---|---|
| macOS arm64 | 3.14 | NumPy | installed from the release tree in a fresh virtual environment outside it: 565 passed, 65 skipped, 3 xfailed, 0 failed, with the default `PERSISTENT_TEST_DEADLINE` and a short `TMPDIR` |

Configurations verified for 1.3.1. The kernel, the solver and the gate are those of 1.3.0, so the GPU row of 1.3.0
applies to 1.3.1 and the gates were not run again:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, no `clang++` | 3.14 | NumPy | installed from the release tree in a fresh virtual environment outside it: 535 passed, 87 skipped, 2 xfailed, 0 failed, with `PERSISTENT_TEST_DEADLINE=1500` for the persistent pipeline cases; 19 cases skip for the missing compiler, among them D15, which leaves 2 of the 3 known failures; case by case against 1.3.0 on the same host, the only changed outcome is the joint-walk case, from xfailed to passed; cached IPRT inputs and synthetic O₄ test data, offline |
| macOS arm64 | 3.14 | NumPy | tests run from the source tree with an empty work root and `SOSFUSE2_OFFLINE=1`: 504 passed, 117 skipped, 3 xfailed, 0 failed, with `PERSISTENT_TEST_DEADLINE=900` and a short `TMPDIR`; the tests that need the IPRT inputs skip with `E_IPRT_OFFLINE` |

Configurations verified for 1.3.0:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, NVIDIA L4, CUDA 12 | 3.12 | CuPy | wheel in a fresh virtual environment, device profile `eda` (the one the reference table was made with): `gate --all` exit 0, 78 nodes passed; G-F against the fixed reference from the RTX 3080: max relative difference 2.7e-6 (rtol 1e-3), cold/warm radiance and source max relative difference 0.0 |
| Linux x86_64, no `clang++` | 3.14 | NumPy | installed from the release tree in a fresh virtual environment outside it: 529 passed, 87 skipped, 3 xfailed, 0 failed, with `PERSISTENT_TEST_DEADLINE=1500` for the persistent pipeline cases; 19 cases skip for the missing compiler, among them D15, which leaves 3 of the 4 known failures; cached IPRT inputs and synthetic O₄ test data, offline |

Configurations verified for 1.2.3:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, RTX 3080, CUDA 12 | 3.12 | CuPy | installed sdist in a fresh virtual environment (identical source tree): `gate --all` exit 0, 78 nodes passed; `clear_34 --reduced` without `--stratosphere`: layer enabled; G-F against the fixed reference: max relative difference 0.0 (rtol 1e-3), cold/warm radiance and source max relative difference 0.0 |
| macOS arm64 | 3.14 | NumPy | installed sdist in a fresh virtual environment outside the source tree: 525 passed, 65 skipped, 5 xfailed, 0 failed; cached IPRT inputs and synthetic O₄ test data, offline |

Configurations verified for 1.2.0:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, RTX 3080, CUDA 12 | 3.12 | CuPy | installed sdist: `gate --all` exit 0 with all 78 gate nodes passing (G-R, G-P, G-T, G-A, H, PS); the G-F product state baked through the production queue, table byte-identical to the fixed G-F table of 1.1.0 (max relative difference 0.0), same O₄ asset |
| macOS arm64 | 3.14 | NumPy | installed sdist (a fresh virtual environment outside the source tree): 524 passed, 65 skipped, 5 xfailed, 0 failed; comparison with 1.1.1 in separate interpreters: the prepared arrays of five states, with and without the host cache, at rtol 1e-6 and a reduced solve (radiance and source) at rtol 1e-5, max relative difference 0.0 |

Configurations verified for 1.1.1, a documentation-only release:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, RTX 3080, CUDA 12 | 3.12 | CuPy | installed sdist: the G-F product state baked through the production queue, table bit-identical to the 1.1.0 fixed G-F table (max relative difference 0.0), same O₄ asset |
| macOS arm64 | 3.14 | NumPy | installed sdist (a fresh virtual environment outside the source tree): 524 passed, 65 skipped, 5 xfailed, 0 failed; the README quick start as written (O₄ download 3 s, reduced solve 36 s) |

Configurations verified for 1.1.0:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, RTX 3080, CUDA 12 | 3.12 | CuPy | release tree in place (`PYTHONPATH=src`): `gate --all` and `gate --gf` pass at the 1.1.0 baselines (the full-resolution product state runs at KSPLIT 0 on this 10 GB card) |
| macOS arm64 | 3.14 | NumPy | installed package tests (a fresh virtual environment outside the source tree): 462 passed, 0 failed, 5 xfailed; the in-tree `--validate-mac` check and the NumPy gate goldens |
| Linux x86_64, AVX-512 | 3.14 | NumPy | targeted tests only: the log-blend unit tests and the FX-N +1 ULP test, NumPy AVX-512 dispatch on and off |

The package runs on Unix only, because the solver uses `resource`, and Python 3.11 is untested. The NumPy reference runs
in FP32 flush-to-zero mode, which is available on Darwin arm64, Linux arm64 and Linux x86_64; other platforms, macOS
x86_64 for example, stop with `E_FTZ_UNSUPPORTED`. The persistent-queue tests need a larger `PERSISTENT_TEST_DEADLINE`
than the 60 s default: their setup queue took 462 s on a 12-core host, and CI uses 2400 s (`docs/CHECKS.md`).

## Package layout

```
src/sosfuse2/
  fused.py, kernel.cu, kernel_config.py   CUDA kernel, launch configuration, NumPy reference (reference.py)
  ksplit.py                               split transport (KSPLIT) host side of fused.Kernel
  geometry*.py, ray_columns.py, ...       spherical-shell geometry, columns, caches
  stratosphere.py, tropo_*.py, mie.py     aerosol layers and Mie tables
  spectral.py, air_wavelength.py          spectra, colour weights
  pipeline*.py, run_gpu_state_v3.py       state runner, production queue, provenance (v4)
  gate.py, gate/                          numeric gates, fixtures, fixed references
  accepted/                               CPU reference engine: per-state context factory (engine.py)
  solver_*.py                             CPU solver modules; solver_render.py holds the preview definitions
  host/, native/sh_cpu.cpp                host helpers, CPU spherical-harmonics kernel
  assets/, configs/, fixtures/            data (see assets/README.md), kernel flag sets, test records
  tests/, testing.py                      the packaged tests (pytest --pyargs sosfuse2) and their support module
docs/                                     MODEL.md, VALIDATION.md (+ validation/ data and figure script), REFERENCES.md,
                                          PERFORMANCE.md, CHECKS.md, KNOWN_FAILURES.md, PROVENANCE_V3.md
```

The CPU solver functions live in the importable `src/sosfuse2/solver_*.py` modules. `accepted/engine.py` creates an
`EngineContext` for each state and binds only the inputs each function reads. `create()` returns `(solver, context,
settings)`; use context attributes such as `context.b` and `context.Solver`. Read-only scope indexing is deprecated
and retained for one release. `solver_render.py` is still read through its own loader in `host/pack_state.py`.
Optical coefficients are kept apart from the column and geometry producers, so coefficient edits do not invalidate
geometry caches.
`run_gpu_state_v3.create_configured` and `tropo_profile.engine_hooks` pass the production grid, spectrum, sun geometry,
deferred ground preparation and two-layer aerosol profile to `create()` through keyword-only hooks; no source text is
rewritten.

## Related codes

libRadtran (Mayer and Kylling 2005; Emde et al. 2016) includes the Monte Carlo solver MYSTIC (Mayer 2009), which has a
spherical 1D atmosphere (Emde and Mayer 2007) and polarisation (Emde et al. 2010). SCIATRAN (Rozanov et al. 2014) is a
radiative transfer package for atmosphere and ocean. SASKTRAN (Bourassa et al. 2008) is a spherical-geometry code with a
successive-orders mode (Zawada et al. 2015), and vector successive-orders solvers are described by Min and Duan (2004)
and Lenoble et al. (2007). sosfuse2 does less than these: one spherical configuration, a source field iterated in place
of photon sampling, and radiance tables as the only product. Full references are in `docs/REFERENCES.md`.

## Maintenance

The package is maintained as part of ongoing research. Issues and pull requests are read, but there is no
support commitment or release schedule. Changes to the physics (for example an added absorber) change the provenance
hashes, so tables from different versions are not interchangeable.

## Licence

Code is MIT (`LICENSE`). Data generated by this project is CC0 1.0 (`src/sosfuse2/assets/README.md`). Data derived from
third parties keeps its original notices, listed in `THIRD_PARTY.md`, which is also installed with the wheel. The CIE
colour tables are CIE standard data, which CIE publishes under CC BY-SA 4.0; the copies here came through Mitsuba 3 and
colour-science, both BSD-3-Clause. NO₂ is distributed under CC BY 4.0 with BIRA-IASB attribution. O₄ data are prepared
locally and are not bundled. The fine solar and ozone spectra are resampled from TSIS-1 HSRS v2 (LASP) and from IUP Bremen
cross sections under their publishers' citation terms, and the LOWTRAN7 validation values are a U.S. Government work. The
wheel's licence expression lists all of them. Please cite the project (`CITATION.cff`) and the data sources listed in
`THIRD_PARTY.md`.
