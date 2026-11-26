# sosfuse2

Radiance tables of the clear twilight sky for a spherical-shell atmosphere. Iterating successive orders of scattering to
convergence on a height and angle grid with Anderson mixing, the solver models polarised Rayleigh scattering, a
tropospheric aerosol layer, a stratospheric sulfate layer enabled in the bundled v2 states, absorption by ozone, NO₂
and O₄, and a Lambertian ground. One solve covers every solar elevation of a table, by default +6° to −18°. The package
ships a NumPy reference implementation, an optional CUDA kernel compiled at run time through CuPy and NVRTC, numeric
checks against fixed references within tolerances, and a provenance record that signs the tables written by
`run_state` and `tables_from_sources` with hashes of its
producing code and data.

## Start here

1. The input is one atmosphere state from `src/sosfuse2/assets/states_v2.json` with grid settings.
2. `create_configured` (`src/sosfuse2/jobs/run_state.py`) builds the solver for that state.
3. `Solver.solve` (`src/sosfuse2/solve/source_iteration.py`) iterates the source field to convergence.
4. `make_table` (`src/sosfuse2/outputs/radiance_table.py`) turns the converged field into a radiance table.
5. `python -m sosfuse2 walkthrough` runs a small example through these three calls. `docs/ARCHITECTURE.md` explains the
   layers, and `AGENTS.md` lists which files to open first for each task.

We publish this research code in the form we use it, releasing versions 1.0.0 to 1.2.3 together and versions from 1.3.0
on one by one. `docs/CHANGELOG.md` lists releases since 1.2.0, gives the old-to-new module table for upgrades from
1.3.x after 1.4.0 moved modules into layer folders, and explains how to reproduce earlier output.
*Tested configurations* records where and how we verified each release. From 1.6.0, full-grid runs can extend above the
default 80 km boundary. The main change to default output came in 1.5.0: a state without a `density` block now uses
`usstd76`/`usstd76`/`usstd` instead of `exp8`/`triangle`/`keep` (*Limitations*). In 1.2.3 the bundled `states_v2.json`
states began including the stratospheric sulfate layer, and the CLI has no `--stratosphere off` option.

![Engine and MYSTIC radiance and ratios for eight bundled states at solar zenith angles 93 and 96
degrees](docs/validation/mystic_states.png)

Engine radiance from 1.9.0 using `usstd76`/`usstd76`/`usstd` (the 1.5.0 default density configuration) and MYSTIC Monte
Carlo radiance on the same optical input for eight bundled states at solar zenith angles 93° and 96°: (a) L/E₀ of
`clear_87` at the sun-side pixel 20.25° / 0.5°, (b) the MYSTIC / engine ratio for the eight states and their mean, and
(c) the luminance ratio at six pixels with 95 % intervals over 16 seeds. Across all 96 cases, the ratio minus 1 has
mean +0.10 % and root mean square 0.63 %. At 93° the mean is +0.13 %, the root mean square is 0.26 %, and 15 of 48
intervals exclude 1; at 96° the root mean square of 0.85 % is comparable to the Monte Carlo error. `docs/VALIDATION.md`
gives details, and scripts in `docs/validation/mystic` repeat the comparison.

## Why a spherical-shell solver

With the Sun 6° to 9° below the horizon, direct sunlight reaches the sky overhead only above the Earth's shadow, about
35 km above the observer at 6° and about 79 km at 9°. Because a plane-parallel atmosphere lacks this shadow, the model
uses a spherical shell. Monte Carlo codes handle spherical geometry, but their radiances carry sampling noise. The
46,080 MYSTIC validation runs (eight states, six pixels, two solar zenith angles, 30 wavelengths, 16 seeds of 250,000
photons each) took 100.0 CPU-hours of `uvspec` time. Even then, the median standard error of the luminance difference
across the eight states is 0.11–0.18 % at SZA 93° and 0.30–1.35 % at SZA 96°, depending on the pixel. Because sosfuse2
solves for a source field instead of sampling photons, a table of 100 × 180 directions and 45 bands has no sampling
noise and covers all solar elevations at once. `docs/MODEL.md` describes the physics, grids and iteration.
`docs/PERFORMANCE.md` records where GPU time goes, how far the kernels sit from hardware limits, what we tried, and the
open question about the main kernel.

## Validation

Using the same optical input at 30 single wavelengths, we compared 1.9.0 tables of eight bundled states against MYSTIC
(libRadtran 2.0.6) at six pixels and solar zenith angles 93° and 96°. The MYSTIC runs come from the 1.2.3 comparison,
whose input files are the same when written from the 1.9.0 export, and at these pixels the 1.9.0 radiance equals the
1.2.3 radiance within 2.0e-6 relative. These tables use the three US Standard Atmosphere
options from *Limitations* (`density_profile: usstd76`, `ozone_profile: usstd76`, `rayleigh_column: usstd`) and the
bundled stratospheric layer centred at 21 km with τ550 0.002–0.006. Because both codes share the input, the comparison
tests transport rather than input physics. At 93°, luminance agrees with MYSTIC within 0.8 % (−0.30 % to +0.77 %) across
all 48 state–pixel combinations, with MYSTIC 0.18 % to 0.35 % brighter in the mean up to 20.25° elevation. Refining the
solar zenith angle grid from 0.5° to 0.25° removes part of this difference. At 96°, the 95 % luminance-difference
interval excludes 0 in 2 of 48 combinations, as expected by chance. With the MYSTIC standard errors at 96°, a difference
of the size seen at 93° would go unresolved. The MYSTIC build reproduces the 120 cells of the spherical Rayleigh
benchmark of Korkin et al. (2020) to 0.27 %.

Two further MYSTIC comparisons use states with a stratospheric layer centred at 19 km, with τ550 0.0037 in the first
state and 0.0022 in the second. The first covers one state, one direction and three solar elevations. For 1.7.0 we
repeated it using the installed package in `point` mode with NO₂ and O₄ in both inputs. With the same MYSTIC runs, the
1.9.0 engine values differ from the 1.7.0 ones by at most 5.6e-7 relative. The codes agree within 1.7 %
per wavelength at 93° and 96°, where the MYSTIC standard error is below 1 %. At 99°, ratios span 0.966–1.068, consistent
with the 2.0–2.5 % MYSTIC error, and the largest difference is 2.66 standard errors. Colour differences are
|Δu′| ≤ 0.0013 and |Δv′| ≤ 0.0010, at most 1.7 standard errors. For 1.1.0, paired MYSTIC runs on the engine's input
measure the NO₂ and O₄ effect across 12 pixels of both states at solar zenith angles 93°–99°. Of 108 values of u′, v′
and Y, 66 reproduce MYSTIC's change to 10 % or 1e-4, none fails, and Monte Carlo noise leaves 42 unresolved.

`docs/VALIDATION.md` documents the set-up, tables, 1.1.0 change sizes, sensitivity to the fixed choices under
*Limitations*, and a qualitative check of the first state's stratospheric aerosol input against measured zenith
polarisation (Ugolnikov et al. 2004). That report also covers two deeper-twilight checks down to −18° solar elevation:
second-order scattering against direct quadrature, and a Rayleigh atmosphere against SASKTRAN2. Against Paranal zenith
twilight from the V-band fit of Patat et al. (2006) over −5° to −15°, 1.2.3 tables with the three options and a uniform
night-sky floor of V = 21.61 mag arcsec⁻² give 0.72 to 0.89 times the fit for `clear_89` and 0.65 to 0.83 times for
`clear_92` over −5° to −12°, with neither bundled state fitted to the site. Down to −12°, shifting the floor by
±0.20 mag moves the ratio by less than 5 %. From −13° to −15°, the floor is 31 % to 77 % of the fit and its 0.20 mag
uncertainty moves the ratio by 7 % to 17 %, so the floor assumption sets the result. In IPRT spherical test case E1 at
solar zenith angles 93° and 96°, the default quadrature misses our error limits, which we attribute to the case's
layered profile. The numeric checks in `docs/CHECKS.md` show that changes keep tables within tolerance of fixed
references across refactors and other hardware; they do not measure physical accuracy.

## Limitations

- The default density configuration, `usstd76`/`usstd76`/`usstd` (air profile, ozone profile, Rayleigh column), still
  differs from measured twilight brightness. At solar elevations −6° / −9° / −12°, model-to-measurement ratios for
  `clear_89` are 1.00 / 1.05 / 0.87 in Maryland, 1.67 / 1.45 / 1.23 at Sacramento Peak, and 0.88 / 0.75 / 0.59 at
  Paranal (`docs/VALIDATION.md`, Table 13, last column). The earlier default through 1.4.0, exp8/triangle/keep, holds
  more air above 20 km than the US Standard Atmosphere 1976 at the same column (1.2 times at 20 km, 1.7 times at 30 km).
  Against measured zenith and all-sky twilight brightness (Koomen et al. 1952; Patat et al. 2006), that configuration
  gives 1.4 to 2.1 times the Maryland and Paranal measurements and 2.9 to 3.2 times those at Sacramento Peak. Two
  variants fitted to the purple-light timing of Lee and Hernández-Andrés (2003) reproduce its start, peak and end solar
  elevations, approximated by u′ extrema along one line of sight: `density_profile: exp8`, `ozone_profile: usstd76`,
  `rayleigh_column: usstd` with stratospheric τ550 0.006 gives 1.6 to 2.1 times the Maryland and Paranal measurements;
  `density_profile: usstd76`, `ozone_profile: usstd76`, `rayleigh_column: usstd` with τ550 0.0037 gives 0.65 to 1.04
  times. Specify all three options in a state's `density` block when choosing a configuration, because partial blocks
  or partial CLI choices stop with `E_DENSITY_PARTIAL`.
- NO₂ uses a fixed AFGL profile shape, 220 K cross sections, and a default column of 0.207 DU (`parameters.no2_du`).
  Real columns vary by location, season and time of day, and the model omits photochemistry. In MYSTIC runs at ten
  twilight pixels, replacing the fixed 220 K (NO₂) and 233 K (ozone) cross sections with per-layer temperatures shifted
  colour by at most 1.8e-4 in u′v′ and Y by at most 0.04 % (95 % bounds).
- O₄ uses fixed, temperature-independent 293/287 K cross sections over approximately 297–654 nm (within the engine's
  360–840 nm grid) and the square of the active air density, and we have not tested what the missing temperature
  dependence of O₄ does. We exclude O₂ A- and B-band CIA—which reach vertical optical depths of about 2.2e-3 (A) and
  1.7e-4 (B) in the reference profile—monomer O₂ lines and water vapour.
- Rayleigh scattering follows Bodhaine et al. (1999) with vacuum wavelength conversion and the existing 450 nm air
  anchor. Phase depolarisation stays fixed at 0.03. A single-scattering estimate using Bodhaine's wavelength-dependent
  value (0.0305 at 360 nm to 0.0276 at 830 nm) shifts u′v′ by at most 3.9e-5 and Y by at most 0.083 %. NO₂ and O₄
  contribute extinction only.
- Default states use a fixed Gaussian sulfate layer centred at 21 km with σ = 3 km and optical depths of 0.002–0.006 at
  550 nm, background choices that do not track the observed stratospheric aerosol burden.
- Tables store total radiance only: the solver tracks Rayleigh polarisation internally and treats aerosol scattering as
  depolarising, without writing Stokes parameters.
- Numeric tolerances bound discretisation and iteration error against fixed references of the same model; they do not
  measure physical error.
- We have not compared the colours with measured twilight colours (Lee and Hernández-Andrés 2003; Lee 2026).
- A stored source does not record the content of an optics table outside the package (`stratosphere.optics_table` or
  `troposphere_phase.optics_table`), only its path. `outputs/radiance_table` and `jobs/tables_from_sources` therefore
  accept a stored source after that file has changed under the same path, and the signature of `tables_from_sources`
  does not change either. `run_state` and `--extra-elevations` hash the file and stop. Give a changed table a new path,
  or solve the source again.
- The MLX transport is experimental and covers one configuration (see *Experimental MLX transport*); its tables are
  checked against the NumPy reference, not separately validated against measurements or other codes.
- On CUDA, a warm launch that cannot use the angular direction table (for example when the table does not fit in free
  memory, or with `SOSFUSE2_DETERMINISTIC=1`) contracts the coefficients directly or uses the fused kernel. A solve that
  falls back to the fused kernel can take many times longer. A cell above the pre-launch work cap of direct contraction
  stops with `E_LAUNCH_BUDGET` instead of falling back.

## Install

sosfuse2 is not on PyPI. Install it in a virtual environment from a checkout of the tagged commit:

```sh
pip install .                 # from a checkout of the tagged commit; or `pip install <sdist>` outside the source tree
pip install ".[test]"         # + pytest
pip install ".[gpu,test]"     # + CuPy (CUDA 12) and the NVIDIA runtime/NVRTC/cuBLAS wheels
```

The package requires Python ≥ 3.12, NumPy ≥ 2.0, Pillow, and two programs on `PATH`. Either `g++` or `clang++` builds
the CPU spherical-harmonics helper `backends/native/sh_cpu.cpp` once into the work root, while kernel syntax tests call
`clang++` specifically. `zstd` packs state outputs as `.tar.zst`. Missing a compiler or `zstd` stops the run with
`E_COMPILER_MISSING` or `E_ZSTD_MISSING`, respectively.

Downloaded inputs, the CPU helper build, and the geometry cache outside a solve go to `SOSFUSE2_WORK_ROOT` (default
`$XDG_CACHE_HOME/sosfuse2`). A solve (`run_state` or `pipeline`) writes tables, intermediates, and packs under
`--work-root` in `out/<id>`, `tmp/<id>`, and `pack/`, and stores its geometry cache in `geometry_cache/` unless
`--geometry-cache` sets another folder. The package directory is read-only data.

On CUDA machines, NVRTC compiles kernels on first use. Without `SOSFUSE2_COMPILE_CACHE`, `jobs/run_state.py` stores
compiled kernels in `nvrtc_cache_shared` in the parent of `--work-root` (`backends/compile_cache.py` `default_root`),
while a persistent pipeline service and the reference tests each use a run-specific folder. Setting
`SOSFUSE2_COMPILE_CACHE=<dir>` redirects all of them to that directory, and
`python -B -m sosfuse2.jobs.precompile --jobs 8` fills that cache before a queue starts. The angular and direct
warm-transport kernels can still compile on their first use.

### Experimental MLX transport (Apple silicon)

With `SOSFUSE2_MLX=1` and the `numpy` backend, the transport (first scattering and the warm iterations; the
ground term is zero for the black surface it covers) runs on the GPU of an Apple silicon Mac through [MLX](https://github.com/ml-explore/mlx). Install it in
the same virtual environment with `pip install mlx`; with the variable set and no `mlx` package, a run stops with
`E_MLX_UNAVAILABLE`. Without the variable nothing changes.

The MLX code covers one configuration: no stratospheric layer, no extended upper domain, a black surface (albedo 0)
and a Henyey-Greenstein troposphere without a Mie table. Any other configuration runs on the NumPy reference path,
call by call, at NumPy speed. Three limits follow from the current entry points:

- On macOS, `run_state` runs only with the `numpy` backend on the reduced grid (which `--backend auto` and an omitted
  `--reduced` also resolve to), so the command line reaches the reduced grid only.
- None of the bundled states in `assets/states_v2.json` is in the covered configuration (every one has the
  stratospheric layer enabled and a nonzero albedo). Give your own states file with `--states`.
- On the full grid, only the transport passes (first scattering and warm iterations) through
  `solve/ray_geometry.py::TransientGeometry` have been run with MLX, from Python; a whole full-grid table has not. The
  full-grid path of `run_state` uses the resident host geometry, which does not call MLX.

MLX runs and NumPy runs of the same state currently carry the same output signature; keep them in separate
`--work-root` folders.

## Quick start

Install the package, prepare the O₄ and density data once, and solve one reduced-resolution state on the NumPy backend.
The solve takes about a minute on a laptop, and 57 s with two threads in the measured run.

```sh
pip install ".[test]"
python -m sosfuse2.asset_tools.prepare_minor_absorbers --download
python -m sosfuse2 generate_density_profile
python -m sosfuse2 run_state --backend numpy
```

No argument of `run_state` is required. The last line is the short form of
`python -B -m sosfuse2.jobs.run_state --id clear_34 --reduced --backend numpy --work-root ./work`: the state is
`clear_34`, the work root is `./work`, and the NumPy backend always solves the reduced grid. With no argument at all,
`--backend auto` selects `cupy` at full resolution when `nvidia-smi` names a visible device and `numpy` on a machine
without CUDA, and the run prints the choice; `docs/ENVIRONMENT.md` lists the cases in which it stops instead of
choosing. The measured run added `--threads 2`.

During a solve, `--geometry-only`, or `--extra-elevations`, `--threads N` sets any unset or blank preparation-pool
variable (`SOSFUSE2_COLUMN_WORKERS`, `SOSFUSE2_SUN_WORKERS`, `SOSFUSE2_GROUND_WORKERS`, `SOSFUSE2_GEOMETRY_WORKERS`) to
N while leaving nonblank values untouched. Without `--threads`, those variables stay unchanged and unset or blank values
use all usable CPUs.

The output table and `meta.json` land in `./work/out/clear_34/`. `meta.json` records `contract: "v5"`, the state,
backend, tolerance settings, `code_hashes` for a fixed list of production modules, solver sources, and data assets, and
a `signature` over them. `docs/PROVENANCE_V3.md` defines this record and how readers holding older `v3` tables compare
them by logical key.

The O₄ step downloads the HITRAN O₂–O₂ collision-induced absorption file into `SOSFUSE2_WORK_ROOT`, verifies the source
SHA-256, and prints the generated table's SHA-256. Pass `--cia O2-O2_2024.cia` to build from a local copy, or set
`SOSFUSE2_OFFLINE=1` to refuse the download. O₄ data are neither bundled (`THIRD_PARTY.md`) nor downloaded by a solve,
so a missing or stale cache stops the run with `E_O4_ASSET_MISSING` or `E_O4_ASSET_STALE`.

The default density configuration fetches two IPRT inputs on first use unless pre-fetched by the density preparation
command. `SOSFUSE2_OFFLINE=1` refuses missing inputs with `E_IPRT_OFFLINE` (`THIRD_PARTY.md` §9). Set all three density
options explicitly to `exp8`/`triangle`/`keep` to avoid the download.

State ids come from the default `--states` file, `sosfuse2/assets/states_v2.json`, holding 64 parameter sets over
`aod550`, `angstrom`, `ssa`, `scale_height_km`, `g`, `ozone_du`, `albedo`, and `stratosphere_tau550`, while
`states_v1.json` holds the 40 first-generation states. All 64 states in `states_v2.json` mark the stratospheric layer
`enabled` so it stays on when `--stratosphere` is omitted, whereas `states_v1.json` marks it `unsupported` so it stays
off unless `--stratosphere on` is passed. The runner sets solar elevation separately via `solar_elevation_deg`, outside
state definitions. Full-resolution tables require CUDA (`--backend cupy`), and `python -B -m sosfuse2.jobs.pipeline`
runs the production queue with overlapping preparation. For a 200 km top with finer height and `χ` grids, Legendre
degree 24, and a tighter tolerance, *Finer and taller grids together* in `docs/MODEL.md` lists the settings file,
command, and stopping errors.

### Production queues and checks

Production queues and reference checks use these settings:

- For research runs, `parameters.no2_du=0` disables NO₂ and `parameters.o4=false` disables O₄, with `o4` required as a
  JSON boolean. With O₄ enabled and an external air profile, that file must declare `air_surface_number_density_m3`.
- Legacy tables lacking a physics field remain readable. To reproduce their numbers, use the package revision recorded
  in their provenance. `spectral_mode=legacy` selects the old spectral grid, not the old physics. Each run uses one of
  three reference fixture sets: legacy, 1.1.0 absorber-validation, or the first public release baseline. Reusing a saved
  source field for the same state requires matching resolved `optics_state` values, where an omitted amount and its
  explicit default share the same signature.
- On CUDA, the solver checks the minimum memory peak across a state's ground and source bundles before its first KSPLIT
  allocation. When that peak exceeds the lone-state memory budget, the default falls back to `KSPLIT=0` for the whole
  state, whereas a service state that fits alone may wait for memory held by other states. `SOSFUSE2_KSPLIT_FALLBACK=0`
  (default in reference checks and benchmarks) raises `E_KSPLIT_MEMORY` instead, while `1` permits fallback. Any
  fallback logs `WARNING KSPLIT fallback` to stderr, emits a `KSPLIT_FALLBACK` phase event, and records
  `ksplit_effective` in table metadata. See [Device memory diagnostics](docs/ENVIRONMENT.md#device-memory-diagnostics)
  for the reason, itemised estimates, and persistent admission records. Setting `SOSFUSE2_VRAM_TRACE=1` also records
  CuPy memory-pool accounting (`vram_trace`) in `meta.json`.
- Both runner and pipeline enforce a host-dependent per-process resident set size (RSS) limit, with `SOSFUSE2_RSS_GB`
  setting both the default and ceiling for `--rss-gb`. [Environment variables](docs/ENVIRONMENT.md#host-rss-limits)
  describes the host rule and how the pipeline handles concurrent processes.
- `plan_queue` stops with `E_PLAN_CPU_BUDGET` when GPUs x workers per GPU x column workers exceed
  `pipeline.cpu_budget()`. Because `jobs/pipeline.py` can still widen a state's column pool, this bounds only the
  planned request rather than acting as a CPU quota. The plan also lists the `ground_aerosol01` host cache after
  `ground`, with an unmeasured run-time cost.
- While building a state, `tables_from_sources` holds the same `<out>/.worker.lock` used by `run_state`, skipping
  worker-owned states while a worker starting concurrently stops with `E_OUTPUT_BUSY`.
- Each pipeline invocation writes logs to its own `logs/<run>/` directory referenced from `pipeline_summary.json`,
  preserving both cold and warm traces. Benchmark stage timers go to `timings-<run>.json`, named in `benchmark.json`.
- Each CUDA entry point caps the CuPy default memory pool via `backends/device_profile.pool_limit_bytes(free)` at the
  smaller of free device memory minus 1 GiB or `SOSFUSE2_POOL_LIMIT_GIB` in GiB. Values truncating below one byte are
  rejected, and cards with under 4 GiB free stop with `E_VRAM_FREE: need at least 4 GiB free`. This cap replaces the
  1.3.4 limit of min(20 GiB, free − 1 GiB), which rejected states needing over 20 GiB on every card with
  `E_PERSISTENT_VRAM_ADMISSION`. Because concurrent pipeline processes on one GPU each size their pool from startup free
  memory, set `SOSFUSE2_POOL_LIMIT_GIB` to partition the card. During admission, the persistent service requests field
  reserve factor `SOSFUSE2_FIELD_RESERVE` (default 30) and lowers it to fit available memory when a lone state exceeds
  capacity there but fits at `SOSFUSE2_FIELD_RESERVE_MIN`. A smaller reserve admits larger or more concurrent states,
  but because admission is a reservation, setting it too low turns waits into solve-time out-of-memory errors. Neither
  reserve variable enters any identity or alters numerics.

## Numerical tolerances

G-P, the fixed CUDA comparison, G-T, G-H and G-F compare results against fixed references at the stated tolerance. G-R records hash and numerical
differences without blocking. G-A and the repeat check have no fixed reference: G-A compares two solves of one run, and
the repeat check compares runs of one build with each other.

| check | compares | tolerance |
|---|---|---|
| G-R | CUDA outputs vs the baseline of the matching toolchain and configuration, or the canonical baseline | none; differences recorded, not blocking |
| G-P | CUDA kernel vs NumPy golden (fixed number of apply steps) | rtol 1e-3 |
| CUDA | the fixture without a NumPy golden (nonzero albedo) vs its fixed CUDA baseline: radiance after the fixed apply steps and the solved table | rtol 1e-3 |
| G-T | anchor radiances (`checks/gate_data/anchors.json`) | ratchet, recorded |
| G-A | approximate fixture vs the exact fixture of the same run, which must pass its own check: radiance after the fixed apply steps and, when both were solved, the table | rtol 1e-3 (plus `--atol`, default 0); the approximate solve must converge in no more evaluations than the exact one |
| G-H | reduced selftest vs its recorded values | per case |
| G-PS | three persistent-service tests on the CUDA backend (parity, no state mixing, interleaving) | pass or fail |
| repeat (`--stage`) | same build, fresh runs in separate folders (two by default); also runs G-H and G-PS | rtol 1e-5, max difference recorded |
| G-F | full-grid fixed table, solved cold and then warm (needs the research checkout) | rtol 1e-3 |

Accumulation order, lane mapping, record layout and the transport path may change between versions and are judged
against these fixed references and tolerances. Set `SOSFUSE2_DETERMINISTIC=1` when studies require bit-stable repeats of a run on the same CUDA GPU:
it fixes the accumulation order and turns off the angular direction table, the tabulated first-light ground phase and
the GPU arithmetic for Anderson history staged in host memory, so the transport contracts the coefficients directly or
uses the fused kernel. The experimental MLX transport does not read this variable and makes no such promise. `src/sosfuse2/checks/gate_data/baselines/` holds the baseline log and reference manifest.

## Tests and checks

`docs/CHECKS.md` defines three test tiers and where each runs.

1. Any machine without a GPU (including public CI): run `python -B -m pytest -q -p no:cacheprovider --pyargs sosfuse2`
   from the installed package, expecting `0 failed`. Tests needing upstream inputs skip unless `SOSFUSE2_SKY_SOS_ROOT`
   points to that checkout, and tests needing `clang++` skip without it. The suite uses a synthetic O₄ table in an
   isolated temporary root without downloading HITRAN data, while optional reproduction from original data
   (`SOSFUSE2_HITRAN_CIA`, `SOSFUSE2_BIRA_DIR`) skips when neither is set. Pass a module name to run one file, such as
   `python -B -m pytest -q -p no:cacheprovider --pyargs sosfuse2.tests.test_names`.
2. Any CUDA machine: `python -B -m sosfuse2.checks.gate --all --out <dir>` runs G-R, G-P, the fixed CUDA comparison, G-T, G-A, G-H and G-PS, while
   `python -B -m sosfuse2.checks.gate --stage --out <dir>` runs the repeat check.
3. A CUDA machine with the research checkout: run
   `python -B -m sosfuse2.checks.gate --gf --reference <table.npz> --out <dir>` (`docs/CHECKS.md`).

In the tables below, `gate --all`, `gate --stage` and `gate --gf` stand for these commands, and H and PS for G-H and
G-PS.

`docs/KNOWN_FAILURES.md` lists environment-dependent outcomes.

## Tested configurations

1.2.1 and 1.2.2 change documentation only, so the 1.2.0 rows apply unchanged. The installed-sdist checks in the 1.2.3
table cover its changed default v2 states.

1.3.3 and 1.3.4 also change documentation only and share the 1.3.2 code, with no tests or reference comparisons repeated
beyond the CI run.

1.9.1 changes documentation only. Apart from the version number, its code is that of 1.9.0, so the 1.9.0 rows apply
to it. For 1.9.1 we solved the
engine side of the two MYSTIC figures and the `clear_34` command again with 1.9.0 (*Versions behind the comparisons* in
`docs/VALIDATION.md`).

Configurations verified for 1.9.0:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | wheel in a fresh virtual environment, device profile `split`: `gate --all` exit 0 (G-R radiance of the five fixtures within 3.1e-5 and tables within 6.4e-6 of the recorded values, the fixed CUDA comparison, G-P and G-A, the five G-T pairs and G-PS passed, G-H max relative difference 1.3e-6); G-F against the fixed reference table: max relative difference 8.1e-6 (rtol 1e-3), cold/warm radiance and source max relative difference 0.0; the upper-air CUDA check against NumPy: max relative difference 1.9e-5. With `SOSFUSE2_DETERMINISTIC=1`, `gate --stage --runs 2` gave equal output hashes for the five fixtures. The CUDA test modules (projection batches, kernel occupancy, persistent pipeline) passed. On the same 10 GB GPU, the reproduction command of `docs/PERFORMANCE.md` for `clear_34` completed from an empty work root and warm |
| macOS arm64 | 3.14 | NumPy | release tree in place (`PYTHONPATH=src`, `SOSFUSE2_SKY_SOS_ROOT` unset), every test module in file order with `PERSISTENT_TEST_DEADLINE=300`: 1293 passed, 91 skipped, 0 failed |
| macOS arm64, Apple silicon GPU (16 GB) | 3.12 | NumPy + MLX 0.32.4 (`SOSFUSE2_MLX=1`) | `run_state --backend numpy --reduced` on the check fixture state FX-N (`checks/gate_data/fixtures.json`) with the stratospheric layer disabled (its reduced-solve AOD 0.02, black surface, 45 wavelengths in three chunks): every transport call on MLX (first 3, ground 21, warm 21, none on NumPy), Anderson evaluations per chunk 10/6/5 as with NumPy, table against the NumPy run of the same tree: max difference 3.2e-6 of each solar elevation's maximum, 2.6e-5 relative over all elements; on an otherwise idle machine, two runs each with two threads: about 39–41 s against 79–80 s with NumPy (transport passes 0.1–0.2 s against 41 s; the observer queries, on NumPy in both, about 20 s). Fixed applies of the check fixture FX-N against the NumPy golden: radiance max relative difference 9.3e-5 (rtol 1e-3). `tests/test_mlx_*.py`: 17 passed |

Configurations verified for 1.8.0:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | wheel in a fresh virtual environment, device profile `split`: `gate --all` exit 0 with the same results as for 1.7.0 (G-R radiance and table of the five fixtures equal to the recorded values, the fixed CUDA comparison, G-P and G-A, the five G-T pairs and G-PS passed, G-H max relative difference 1.4e-6); G-F against the fixed reference table: max relative difference 2.6e-6 (rtol 1e-3), cold/warm radiance and source max relative difference 0.0. On the same 10 GB GPU, `python -m sosfuse2 run_state` with no argument chose `cupy` and gave a table equal in every array to that of the command typed in full |
| macOS arm64 | 3.14 | NumPy | release tree installed in a fresh virtual environment (`SOSFUSE2_SKY_SOS_ROOT` unset), every test module in file order: 1246 passed, 76 skipped, 0 failed |

Configurations verified for 1.6.1:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | wheel in a fresh virtual environment, device profile `split`: `gate --all` exit 0 with the same results as for 1.6.0 (G-R radiance and table of the five fixtures equal to the recorded values, the CUDA, G-P and G-A checks, the five G-T pairs and PS passed, G-H max relative difference 1.4e-6); G-F against the fixed reference table: max relative difference 2.6e-6 (rtol 1e-3), cold/warm radiance and source max relative difference 0.0. On the same 10 GB GPU, the reproduction command of `docs/PERFORMANCE.md` for `clear_34` completed twice from an empty work root and twice warm (*Release 1.6.1 on GPU A* in that file) |
| macOS arm64 | 3.14 | NumPy | wheel in a fresh virtual environment, all test modules except the persistent-service and r17 port modules, in file order and in two random orders (pytest-xdist, 8 workers): 1077 passed, 82 skipped, 0 failed in each of the three runs |
| macOS arm64 | 3.14 | NumPy | release tree in place (`PYTHONPATH=src`, `SOSFUSE2_SKY_SOS_ROOT` unset), every test module with pytest-xdist (8 workers) and `PERSISTENT_TEST_DEADLINE=900`: 1098 passed, 74 skipped, 0 failed and no expected failures |

Configurations verified for 1.6.0:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | wheel in a fresh virtual environment, device profile `split`: `gate --all` exit 0. In G-R the radiance and the table of all five fixtures equal the recorded values (max relative difference 0.0); only the bytes of the recorded source solve differ, which G-R reports without blocking. The CUDA, G-P and G-A checks, the five G-T pairs and PS passed, and G-H gave the same value as for 1.5.0 (max relative difference 1.4e-6). G-F against the fixed reference table: max relative difference 2.6e-6 (rtol 1e-3), cold/warm radiance and source max relative difference 0.0 |
| macOS arm64 | 3.14 | NumPy | wheel in a fresh virtual environment, all test modules except the persistent-service and r17 port modules, in file order and in two random orders (pytest-xdist, 8 workers): 1067 passed, 82 skipped, 3 xfailed, 0 failed in each of the three runs |
| macOS arm64 | 3.14 | NumPy | release tree in place (`PYTHONPATH=src`, `SOSFUSE2_SKY_SOS_ROOT` unset), every test module with pytest-xdist (8 workers) and `PERSISTENT_TEST_DEADLINE=300`: 1083 passed, 74 skipped, 3 xfailed and 5 errors; the five errors are the tests of the persistent-service module that share one baseline run, which reached the 300 s deadline under that load. Run alone with the default deadline, that module gave 6 passed |

Configurations verified for 1.5.2, sharing the numerical code of 1.5.0:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | wheel in a fresh virtual environment, device profile `split`: `gate --all` exit 0, the same checks as for 1.5.0 passed with the same G-H value (max relative difference 1.4e-6); G-F against the fixed reference table: max relative difference 2.7e-6 (rtol 1e-3), cold/warm radiance and source max relative difference 0.0 |
| Linux x86_64 | 3.12 | NumPy | sdist in a fresh virtual environment outside the source tree, as the four shards of the continuous-integration job (pytest-xdist, 4 workers each): 930 passed, 3 xfailed and 0 failed in total, with 78 skipped cases, among them four files that skip at module level and are reported by every shard; an earlier serial run of the sdist gave 930 passed, 78 skipped, 3 xfailed |
| macOS arm64 | 3.14 | NumPy | earlier sdist in a fresh virtual environment outside the source tree, run as in continuous integration (empty work root, the session fetches the IPRT inputs, no prepared O₄ asset). Environment flags are covered by `test_entry_points`: 930 passed, 78 skipped, 3 xfailed, 0 failed |

1.5.1 changes documentation only and shares the 1.5.0 code, so the 1.5.0 rows apply. Beyond the CI run, its one new
check covers the scripts in `docs/validation/mystic` (*Reproducing the figures and the records* in
`docs/VALIDATION.md`).

Configurations verified for 1.5.0:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | wheel in a fresh virtual environment, device profile `split` selected with its 1.3.x name: `gate --all` exit 0, 78 nodes passed; G-F against the fixed reference from a different GPU: max relative difference 2.7e-6 (rtol 1e-3), cold/warm radiance and source max relative difference 0.0 |
| macOS arm64 | 3.14 | NumPy | release tree in place (`PYTHONPATH=src`), offline, with the IPRT inputs and the prepared O₄ asset in the work root: 939 passed, 68 skipped, 3 xfailed, 0 failed; without the IPRT inputs: 751 passed, 256 skipped (188 of them `E_IPRT_OFFLINE`), 3 xfailed, 0 failed; the README quick start as written in a fresh virtual environment with an empty work root (O₄ and density preparation, reduced solve 57 s) |

Configurations verified for 1.4.0:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | wheel in a fresh virtual environment, device profile `split` selected with its 1.3.x name: `gate --all` exit 0, 78 nodes passed; G-F against the fixed reference from a different GPU: max relative difference 2.7e-6 (rtol 1e-3), cold/warm radiance and source max relative difference 0.0; the state log of G-F shows the parallel geometry adapter in use |
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | wheel installed into a target directory placed first on `PYTHONPATH`: `checks.solver_diagnostics --backend cupy --solve --grid full --geometry-mode endpoints` exit 0, with the device memory reported; `clear_86` on the grid of *Finer grids with the 1.4.0 grid rules* in `docs/VALIDATION.md` (`h_refine` `b2a6`, `chi_refine_deg` `[88, 100]`, `L` 24) with the default CUDA profile: at most 1.9e-7 of the largest value of each solar elevation and colour channel from the research-build table |
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | wheel installed in the same way, device profile `split`: `gate --stage` exit 0, and the two staged runs of every fixture agree in all 131 arrays compared (max difference 0.0) |
| Linux x86_64, no `clang++` | 3.14 | NumPy | installed from the release tree in a fresh virtual environment outside it: 869 passed, 99 skipped, 2 xfailed, 0 failed, with `PERSISTENT_TEST_DEADLINE=1500` for the persistent pipeline cases; 24 cases skip for the missing `clang` and `clang++`, including the spherical-harmonic mismatch test, which leaves 2 of the 3 known failures; cached IPRT inputs and the prepared O₄ asset, offline |
| macOS arm64 | 3.14, 3.12 | NumPy | sdist (3.14 and 3.12) and wheel (3.14) in fresh virtual environments outside the source tree: the 14 test modules on identity, grid rules, the pool limit, tables from sources, the command entry point, layers, names, imports, documentation pointers and the walkthrough, 301 passed, 8 skipped, 0 failed in each; the skips are the 7 documentation pointer cases, whose files are not in the installed package, and the slow walkthrough case |

Configurations verified for 1.3.2. Because its kernel, solver, and reference checks match 1.3.0, the 1.3.0 GPU row
applies without repeating those checks:

| platform | Python | backend | what was run |
|---|---|---|---|
| macOS arm64 | 3.14 | NumPy | installed from the release tree in a fresh virtual environment outside it: 565 passed, 65 skipped, 3 xfailed, 0 failed, with the default `PERSISTENT_TEST_DEADLINE` and a short `TMPDIR` |

Configurations verified for 1.3.1. Because its kernel, solver, and reference checks match 1.3.0, the 1.3.0 GPU row
applies without repeating those checks:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, no `clang++` | 3.14 | NumPy | installed from the release tree in a fresh virtual environment outside it: 535 passed, 87 skipped, 2 xfailed, 0 failed, with `PERSISTENT_TEST_DEADLINE=1500` for the persistent pipeline cases; 19 cases skip for the missing compiler, including the spherical-harmonic mismatch test, which leaves 2 of the 3 known failures; case by case against 1.3.0 on the same host, the only changed outcome is the joint-walk case, from xfailed to passed; cached IPRT inputs and synthetic O₄ test data, offline |
| macOS arm64 | 3.14 | NumPy | tests run from the source tree with an empty work root and `SOSFUSE2_OFFLINE=1`: 504 passed, 117 skipped, 3 xfailed, 0 failed, with `PERSISTENT_TEST_DEADLINE=900` and a short `TMPDIR`; the tests that need the IPRT inputs skip with `E_IPRT_OFFLINE` |

Configurations verified for 1.3.0:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | wheel in a fresh virtual environment, the reference table’s device profile (named `split` after 1.3.x): `gate --all` exit 0, 78 nodes passed; G-F against the fixed reference from a different GPU: max relative difference 2.7e-6 (rtol 1e-3), cold/warm radiance and source max relative difference 0.0 |
| Linux x86_64, no `clang++` | 3.14 | NumPy | installed from the release tree in a fresh virtual environment outside it: 529 passed, 87 skipped, 3 xfailed, 0 failed, with `PERSISTENT_TEST_DEADLINE=1500` for the persistent pipeline cases; 19 cases skip for the missing compiler, including the spherical-harmonic mismatch test, which leaves 3 of the 4 known failures; cached IPRT inputs and synthetic O₄ test data, offline |

Configurations verified for 1.2.3:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | installed sdist in a fresh virtual environment (identical source tree): `gate --all` exit 0, 78 nodes passed; `clear_34 --reduced` without `--stratosphere`: layer enabled; G-F against the fixed reference: max relative difference 0.0 (rtol 1e-3), cold/warm radiance and source max relative difference 0.0 |
| macOS arm64 | 3.14 | NumPy | installed sdist in a fresh virtual environment outside the source tree: 525 passed, 65 skipped, 5 xfailed, 0 failed; cached IPRT inputs and synthetic O₄ test data, offline |

Configurations verified for 1.2.0:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | installed sdist: `gate --all` exit 0 with all 78 test nodes passing (G-R, G-P, G-T, G-A, H, PS); the full-grid G-F reference state baked through the production queue, table equal to the fixed G-F table of 1.1.0 (max relative difference 0.0), same O₄ asset |
| macOS arm64 | 3.14 | NumPy | installed sdist (a fresh virtual environment outside the source tree): 524 passed, 65 skipped, 5 xfailed, 0 failed; comparison with 1.1.1 in separate interpreters: the prepared arrays of five states, with and without the host cache, at rtol 1e-6 and a reduced solve (radiance and source) at rtol 1e-5, max relative difference 0.0 |

Configurations verified for 1.1.1, a documentation-only release:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | installed sdist: the full-grid G-F reference state baked through the production queue, table equal to the 1.1.0 fixed G-F table (max relative difference 0.0), same O₄ asset |
| macOS arm64 | 3.14 | NumPy | installed sdist (a fresh virtual environment outside the source tree): 524 passed, 65 skipped, 5 xfailed, 0 failed; the README quick start as written (O₄ download 3 s, reduced solve 36 s) |

Configurations verified for 1.1.0:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | release tree in place (`PYTHONPATH=src`): `gate --all` and `gate --gf` pass at the 1.1.0 baselines (the full-resolution product state runs at KSPLIT 0 on this GPU) |
| macOS arm64 | 3.14 | NumPy | installed package tests (a fresh virtual environment outside the source tree): 462 passed, 0 failed, 5 xfailed; the in-tree `--validate-mac` check and the NumPy reference arrays |
| Linux x86_64, AVX-512 | 3.14 | NumPy | targeted tests only: the log-blend unit tests and the +1 ULP test for the stratosphere-off fixture, NumPy AVX-512 dispatch on and off |

sosfuse2 runs only on Unix because the solver uses `resource`, and Python 3.11 is untested. The NumPy reference uses
FP32 flush-to-zero mode, supported on Darwin arm64, Linux arm64, and Linux x86_64; other platforms such as macOS x86_64
stop with `E_FTZ_UNSUPPORTED`. Persistent-queue tests and CI default to a `PERSISTENT_TEST_DEADLINE` of 2400 s.

## Package layout

```
src/sosfuse2/
  foundation/     paths and work root, content identity, host limits, timers
  atmosphere/     a state and a wavelength to densities and optical coefficients
  rays/           spherical ray segments, incoming-direction quadrature, columns
  precompute/     reusable CPU arrays and their caches
  backends/       one transport evaluation: numpy_reference.py, kernel.cu, native/sh_cpu.cpp, configs/
  solve/          per-state solver objects and the source iteration (source_solver.py, source_iteration.py, engine.py, ksplit.py)
  outputs/        radiance tables, previews, packs
  jobs/           state runner, production queue, planner, cache store, walkthrough
  checks/         numeric checks (gate.py, gate_data/), diagnostics, fixtures/, testkit.py
  asset_tools/    generators of the data assets, O₄ preparation
  assets/         data (see assets/README.md)
  tests/          the packaged tests (pytest --pyargs sosfuse2)
docs/             MODEL.md, VALIDATION.md (+ validation/ data and figure scripts), REFERENCES.md, PERFORMANCE.md,
                  CHECKS.md, KNOWN_FAILURES.md, PROVENANCE_V3.md, ARCHITECTURE.md, ENVIRONMENT.md, WALKTHROUGH.md
```

`docs/ARCHITECTURE.md` documents folder roles, import direction, and cross-cutting topics. Each folder names its
modules' main duty, though a few mix two.

CPU solver functions live in importable modules: `rays/shell_segments.py`, `rays/incoming_quadrature.py`,
`rays/incoming_rings.py`, `rays/zenith_interpolation.py`, `atmosphere/extinction.py`, `solve/engine_core.py`, and
`outputs/preview_functions.py`. `outputs/pack_state.py` still loads `outputs/preview_functions.py` through its own
loader. `solve/engine.py` builds a per-state `EngineContext` binding only the inputs each function reads, and `create()`
returns `(solver, context, settings)`. Use context attributes such as `context.b` and `context.Solver`. Read-only
indexing by the former scope-dict keys is deprecated and retained for one release. Separating optical coefficients from
column and geometry producers keeps coefficient edits from invalidating geometry caches.
`jobs/run_state.py::create_configured` and `atmosphere/tropo_profile.py::engine_hooks` supply the production grid,
spectrum, sun geometry, deferred ground preparation, and two-layer aerosol profile to `create()` via keyword-only hooks
without rewriting source text.

## Related codes

libRadtran (Mayer and Kylling 2005; Emde et al. 2016) includes the MYSTIC Monte Carlo solver (Mayer 2009), supporting a
spherical 1D atmosphere (Emde and Mayer 2007) and polarisation (Emde et al. 2010). SCIATRAN (Rozanov et al. 2014) models
atmosphere and ocean radiative transfer. SASKTRAN (Bourassa et al. 2008) handles spherical geometry with a
successive-orders mode (Zawada et al. 2015), while Min and Duan (2004) and Lenoble et al. (2007) describe vector
successive-orders solvers. Eradiate (Leroy et al. 2026) is a Mitsuba 3 Monte Carlo ray tracer with plane-parallel and
spherical-shell atmospheres and polarisation. twilight (Mahdi 2026) computes deep-twilight spectral sky radiance via
backward Monte Carlo in a spherical refracting atmosphere and models human visual detection at dawn and dusk. sosfuse2
is narrower: it handles one spherical configuration, iterates a source field rather than sampling photons, and outputs
only radiance tables. See `docs/REFERENCES.md` for full references.

## Maintenance

We maintain sosfuse2 as part of ongoing research and read issues and pull requests, without a support commitment or
release schedule. Physics changes such as an added absorber alter provenance hashes, so tables from different versions
are not interchangeable.

## Licence

Code is MIT (`LICENSE`) and project-generated data are CC0 1.0 (`src/sosfuse2/assets/README.md`). Data derived from
third parties keep their original notices in `THIRD_PARTY.md`, also installed with the wheel. The CIE colour tables are
CIE standard data under CC BY-SA 4.0, and the bundled copies come via Mitsuba 3 and colour-science (both BSD-3-Clause).
NO₂ is distributed under CC BY 4.0 with BIRA-IASB attribution, while O₄ data are prepared locally and not bundled. The
fine solar and ozone spectra are resampled from TSIS-1 HSRS v2 (LASP) and IUP Bremen cross sections under their
publishers' citation terms, and the LOWTRAN7 validation values are a U.S. Government work. All licences appear in the
wheel's licence expression. Please cite the project (`CITATION.cff`) and the data sources in `THIRD_PARTY.md`.