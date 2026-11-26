# Environment variables

**No environment variable is needed to run the package, and `jobs/run_state.py` needs no argument.** Without arguments
it solves the state `clear_34` under `./work`; `--backend auto`, the default, is `cupy` when `nvidia-smi` names a
visible device, and `numpy` on a Mac, with a blank `CUDA_VISIBLE_DEVICES`, or on a machine with neither `nvidia-smi` nor
CuPy; `numpy` solves the reduced grid. The run prints the choice. Three cases stop with `E_BACKEND_AUTO` instead of choosing: `nvidia-smi` without
an answering device, CuPy installed without `nvidia-smi` on `PATH`, and a device named by `nvidia-smi` without CuPy
installed (the package without its `gpu` extra). A typed `--backend` is not replaced by this choice; `--geometry-only`
prepares the geometry on NumPy whichever backend is named, and on macOS it also needs `--reduced`.
`jobs/pipeline.py` has the same defaults for `--states` (the packaged `states_v2.json`), `--work-root` and `--backend`,
and still requires `--ids` or `--plan`; the commands that `jobs/plan_queue.py` writes name `--backend cupy`. The
settings that a person changes from run to run are command-line options of both:

| Option | Sets | Meaning |
|---|---|---|
| `--id NAME` | state of `run_state` | default `clear_34` |
| `--work-root DIR` | folders of the run | where `out/`, `tmp/`, `pack/` and the caches of the run go; default `./work` |
| `--backend auto\|cupy\|numpy` | backend | default `auto` |
| `--threads N` | thread variables and the preparation pools | CPU threads of the run |
| `--rss-gb X` | host memory limit of the process | see [Host RSS limits](#host-rss-limits) |
| `--config NAME` | `SOSFUSE2_KERNEL_FLAGS`, `SOSFUSE2_BLOCK_SIZE` | kernel config `backends/configs/NAME.json`, for instance `C3` |
| `--block-size 128\|256` | the block size of the runner's kernel and compile cache (passed directly, not exported); specialised kernels can use fixed block sizes | CUDA block size without a named config |
| `--ground-sh 0\|16\|24\|32` | `SOSFUSE2_GROUND_SH_L` | ground convolution order |
| `--device-profile NAME` | `SOSFUSE2_DEVICE_PROFILE` | profile of `backends/configs/devices.json` |
| `--compile-cache DIR` | `SOSFUSE2_COMPILE_CACHE` | CUDA runtime compilation cache |
| `--attach-cache DIR\|off` | `STRATO_ATTACH_CACHE` | stratosphere attach cache |
| `--column-cache DIR` | `SOSFUSE2_COLUMN_CACHE` | column-array cache |

An option writes its variable before the package reads it, so the option takes precedence over a value in the
environment and the child processes of the run inherit it. `--block-size` is the exception: it writes no variable and
goes to the kernel directly, and `jobs/pipeline.py` passes it to each state worker as the same option. When it differs from a
nonblank `SOSFUSE2_BLOCK_SIZE`, or from the block size of `--config NAME`, the run stops with `E_KERNEL_BLOCK_SIZE`.
A blank `--compile-cache` or `--attach-cache` selects the default location, like an empty variable. The variables remain
for scripts and queues that export them.

The rest of this page is the reference: every environment variable the package reads, the layer that reads it
(`docs/ARCHITECTURE.md` section 2), its default, and the precedence when multiple settings apply; if no precedence is
stated, the variable is the only way to set the value. The tables are ordered by who sets the variable:

1. [Variables a person sets](#variables-a-person-sets): work root, offline mode, the cache store and a few records.
2. [Tuning](#tuning): performance and memory choices; the defaults suit a first run.
3. [Diagnostics](#diagnostics): opt-in measurement hooks.
4. [Variables the package sets for its own processes](#variables-the-package-sets-for-its-own-processes): not for users.
5. [Variables of the tests only](#variables-of-the-tests-only).

`tests/test_docs_pointers.py` searches the code for literal `SOSFUSE2_` names and fails if any is absent from this page,
so a new variable needs a table row in the same change. The test runs from a source checkout and skips in an installed
package.

Values are strings. Each row describes how its reader handles an empty value and other values; the readers do not all
interpret switches in the same way.

The *Use* column classifies each variable as **user** for settings a person sets, **tuning** for performance and memory
choices, **diagnostic** for opt-in measurement hooks, **internal** for variables that one part of the package sets for
another, or **optional** for research extras.

## Work root, prepared data and state files

Downloaded inputs, the CPU helper build, and the geometry cache outside a solve go to `SOSFUSE2_WORK_ROOT` (default
`$XDG_CACHE_HOME/sosfuse2`). A solve (`run_state` or `pipeline`) writes tables, intermediates, and packs under
`--work-root` in `out/<id>`, `tmp/<id>`, and `pack/`, and stores its geometry cache in `geometry_cache/` unless
`--geometry-cache` sets another folder. The package directory is read-only data.

The O₄ step downloads the HITRAN O₂–O₂ collision-induced absorption file into `SOSFUSE2_WORK_ROOT`, verifies the source
SHA-256, and prints the generated table's SHA-256. Pass `--cia O2-O2_2024.cia` to build from a local copy, or set
`SOSFUSE2_OFFLINE=1` to refuse the download. O₄ data are neither bundled (`THIRD_PARTY.md`) nor downloaded by a solve,
so a missing or stale cache stops the run with `E_O4_ASSET_MISSING` or `E_O4_ASSET_STALE`.

The default density configuration fetches two IPRT inputs on first use unless pre-fetched by the density preparation
command. `SOSFUSE2_OFFLINE=1` refuses missing inputs with `E_IPRT_OFFLINE` (`THIRD_PARTY.md` §9). Set all three density
options explicitly to `exp8`/`triangle`/`keep` to avoid the download.

State ids come from the default `--states` file, `sosfuse2/assets/states_v2.json`, holding 64 parameter sets over `aod550`, `angstrom`, `ssa`, `scale_height_km`, `g`, `ozone_du`, `albedo`, and `stratosphere_tau550`. It names as its parent `states_v1.json`, which holds the 40 first-generation states, so their ids also resolve from the default file. All 64 states in `states_v2.json` mark the stratospheric layer `enabled` so it stays on when `--stratosphere` is omitted, whereas `states_v1.json` marks it `unsupported` so it stays off unless `--stratosphere on` is passed. For a 200 km top with finer height and `χ` grids, Legendre degree 24, and a tighter tolerance, *Finer and taller grids together* in `docs/MODEL.md` lists the settings file, command, and stopping errors.

## Variables a person sets

| Variable | Layer | Default | Effect and precedence | Use |
|---|---|---|---|---|
| `SOSFUSE2_WORK_ROOT` | foundation (`paths.work_root`) | `$XDG_CACHE_HOME/sosfuse2`, else `~/.cache/sosfuse2` | A nonempty path, after trimming, is the writable root for inputs and caches; empty uses the default and an unusable path raises `E_WORK_ROOT`. The runner's `--work-root` is the root of its `out/`, `tmp/` and `pack/` folders and, unless `--geometry-cache` names another folder, of its `geometry_cache/` folder, without replacing this variable; the walkthrough sets the variable to its own work root. | user |
| `SOSFUSE2_PACKAGE_COMMIT` | foundation (`content_identity.package_info`) | unset | Any nonempty string is recorded as the commit beside the signed fields; empty falls through to `direct_url.json`, then this package's git checkout. The commit does not enter the signature. | user |
| `SOSFUSE2_OFFLINE` | asset_tools (`iprt`, `prepare_minor_absorbers`) | unset (downloads allowed) | An empty value or `0` permits downloads, while any other string disables them with `E_IPRT_OFFLINE` or `E_O4_OFFLINE`. In `iprt.fetch`, an explicit `offline` argument overrides the variable, and an already verified local file is returned before any download. | user |
| `SOSFUSE2_DETERMINISTIC` | backends (`flags.deterministic`) | `0` | Exactly `1` selects deterministic accumulation and `0` disables it; empty and every other value raise `E_DETERMINISTIC`. With `1`, the CUDA transport also skips the angular direction table and the tabulated first-light ground phase, and the Anderson step does not use the GPU arithmetic for history staged in host memory. The experimental MLX transport does not read it. The choice enters compile options and the compile cache key. | user |
| `SKY_CACHE_REMOTE` | jobs (`cache_store`) | unset | A nonempty string supplies the rclone remote; empty means no remote, so an operation without an explicit remote raises `E_CACHE_REMOTE_MISSING`. | user |
| `SKY_CACHE_RCLONE_CONFIG` | jobs (`cache_store`) | unset: rclone default | A nonempty string names the rclone configuration file; empty uses rclone's default. A supplied path that is not a file raises `E_CACHE_CONFIG_MISSING`. | user |
| `SKY_CACHE_MIRROR` | jobs (`cache_store`) | `<work root>/sky-cache` | A nonempty string is the local mirror path; empty uses the default. The path is used as supplied. | user |
| `SKY_CACHE_RCLONE` | jobs (`cache_store`) | `rclone` on `PATH`, else `~/.local/bin/rclone` | A nonempty string names the executable; empty searches the defaults. A selected path that is not an executable file raises `E_CACHE_RCLONE_MISSING`. | user |
| `GATE_SOURCE_COMMIT` | checks (`gate.commit`) | the package commit | Any nonempty string is recorded as the source commit; empty uses the package commit. | user |
| `SOSFUSE2_SKY_SOS_ROOT` | foundation (`paths.sky_sos_root`) | unset: no research extras | A directory path after trimming supplies the research checkout for the G-F check, `checks/validate.py` and some tests. Empty means no extras; a non-directory raises `E_SKY_SOS_ROOT`. | optional |
| `SOSFUSE2_FULL_SOURCE` | checks (`validate.full_source_path`) | unset | Path to the converged source NPZ required for `--validate-mac`. The `--full-source` option takes precedence. | optional |
| `SOSFUSE2_TEST_SCRATCH` | checks (`testkit.scratch_parent`) | `sosfuse2-tests-<uid>` in the system temporary folder | A nonempty path after trimming is the parent of test scratch directories; empty uses the default. An unusable path fails when created. | optional |

## Tuning

| Variable | Layer | Default | Effect and precedence | Use |
|---|---|---|---|---|
| `SOSFUSE2_SUN_METHOD` | atmosphere (`density.sun_method`) | `gl` | After trimming, empty selects `gl`; `gl` keeps quadrature and `exact` uses closed-form profile columns. Other values raise `E_SUN_METHOD`. The choice enters dependent cache keys. | tuning |
| `SOSFUSE2_COLUMN_METHOD` | rays (`column_table.method`) | `table-v1` | After trimming, `gl24`, `table-v1` or `exact` selects the end-column method. Empty and all other values raise `E_COLUMN_METHOD`. | tuning |
| `SOSFUSE2_COLUMN_TABLE_CACHE` | rays (`column_table.tables`) | falls back to `SOSFUSE2_COLUMN_CACHE`; with neither the tables are built in memory | Any nonempty string is the profile-table folder; empty falls back to `SOSFUSE2_COLUMN_CACHE`. The path is used as supplied; an unavailable cache falls back to an in-memory build. | tuning |
| `SOSFUSE2_COLUMN_WORKERS` | precompute (`column_parallel.workers`) | all usable CPUs; N for a `jobs/run_state.py --threads N` run when the variable is unset or blank after trimming | A positive integer is the column pool size; `1` is serial. Empty after trimming uses the default; zero, negative or non-integer values raise `E_COLUMN_WORKERS`. `jobs/pipeline.py` only exports a per-state value when the name is unset or blank after trimming. | tuning |
| `SOSFUSE2_SUN_WORKERS` | precompute (`sun_parallel.workers`) | all usable CPUs; N for a `jobs/run_state.py --threads N` run when the variable is unset or blank after trimming | A positive integer is the solar-table pool size; `1` is serial. Empty after trimming uses the default; zero, negative or non-integer values raise `E_SUN_WORKERS`. Pipeline precedence is the same as for `SOSFUSE2_COLUMN_WORKERS`. | tuning |
| `SOSFUSE2_GROUND_WORKERS` | precompute (`ground_parallel.workers`) | all usable CPUs; N for a `jobs/run_state.py --threads N` run when the variable is unset or blank after trimming | A positive integer is the ground-cell pool size; `1` is serial. Empty after trimming uses the default; zero, negative or non-integer values raise `E_GROUND_WORKERS`. Pipeline precedence is the same as for `SOSFUSE2_COLUMN_WORKERS`. | tuning |
| `SOSFUSE2_GEOMETRY_WORKERS` | solve (`ray_geometry_parallel.workers`) | all usable CPUs; N for a `jobs/run_state.py --threads N` run when the variable is unset or blank after trimming | A positive integer sets the cell pool size. `1` selects the serial writer. Empty after trimming uses the default. Zero, negative or non-integer values raise `E_GEOMETRY_WORKERS`. Pipeline precedence is the same as for `SOSFUSE2_COLUMN_WORKERS`. See [Architecture](ARCHITECTURE.md#5-topics-that-cross-layers) for the serial fallback events. | tuning |
| `SOSFUSE2_HOST_CACHE` | precompute (`host_cache.enabled`) | `1` | Exactly `0` bypasses the corrected solar-table and ground-preparation caches; empty and every other string enable them. The runner still uses the disk-cached original solar table at its production scale, and other caches, including profile tables and geometry, remain independent. | tuning |
| `SOSFUSE2_COLUMN_CACHE` | precompute (`column_cache`) | unset (disabled for direct callers, enabled by `jobs/pipeline.py` for full grids) | Direct callers use any nonempty string as the column-array folder, while an empty value disables the cache. `jobs/pipeline.py` treats an explicitly empty value as the current directory and `off` as disabled; in both `jobs/pipeline.py` and `jobs/run_state.py`, `--column-cache` overrides the environment variable. | tuning |
| `SOSFUSE2_KERNEL_FLAGS` | backends (`flags`, `compile_cache`) | empty | Takes a comma-separated list of flag names; spaces and empty entries are removed. An empty value selects no flags, retired names emit a warning and are ignored, unsupported names raise `E_KERNEL_FLAG`, and physical selectors in the runner raise `E_KERNEL_FLAGS`. The runner records the resolved flags in the signed settings. The `--config NAME` option of `jobs/run_state.py` and `jobs/pipeline.py` sets this variable and `SOSFUSE2_BLOCK_SIZE` from `backends/configs/NAME.json` and takes precedence. | tuning |
| `SOSFUSE2_BLOCK_SIZE` | backends (`compile_cache`, `jobs/run_state.py`) | `128` | The runner accepts `128` or `256`; empty after trimming uses `128`, other values raise `E_KERNEL_BLOCK_SIZE`, as does a conflicting `--block-size`. Direct compile/precompile paths parse the value as an integer, so an explicitly empty value can instead fail. `--config NAME` sets it from the named kernel config; a `--block-size` that differs from that config raises `E_KERNEL_BLOCK_SIZE`. | tuning |
| `SOSFUSE2_FMAD` | solve (`source_solver.Kernel.stage1_flags`) | unset: the `stage1_defaults` of the device profile | `1` enables and `0` disables it; empty after trimming uses the profile. Other values raise `E_STAGE1_FLAG`. `checks/gate.py` sets both explicitly. | tuning |
| `SOSFUSE2_FASTMATH` | solve (`source_solver.Kernel.stage1_flags`) | unset: the `stage1_defaults` of the device profile | `1` enables and `0` disables it; empty after trimming uses the profile. Other values raise `E_STAGE1_FLAG`. `checks/gate.py` sets both explicitly. `1` also selects the approximate exponential and logarithm of the warm kernel; `0` keeps the standard ones. Neither value turns off the angular direction table, which has its own eligibility checks. | tuning |
| `SOSFUSE2_COMPILE_CACHE` | backends (`compile_cache.configure`) | unset (caller's root, or `compile_cache.default_root` beside the work root for `jobs/run_state.py`) | A nonempty path after trimming replaces the caller's CUDA runtime compilation (NVRTC) cache root, while an empty value keeps the caller's root. `jobs/precompile.py` uses this variable as its `--cache` default and requires either a nonempty value or an explicit option. The `--compile-cache` option of `jobs/run_state.py` and `jobs/pipeline.py` takes precedence. | tuning |
| `SOSFUSE2_DEVICE_PROFILE` | backends (`device_profile.select`) | by CUDA device name; `cuda` when no profile lists the device, `mac` without CUDA | A profile name in `backends/configs/devices.json` forces selection; empty after trimming uses the default. An unknown name raises `E_DEVICE_PROFILE`. The `--device-profile` option of `jobs/run_state.py` and `jobs/pipeline.py` takes precedence. | tuning |
| `SOSFUSE2_MLX` | backends (`mlx_backend.enabled`) | unset (off) | `1` (after trimming spaces) sends the host transport of the `numpy` backend (`TransientGeometry.evaluate`) to the experimental MLX implementation on Apple silicon for the configurations it covers: no stratosphere, no extended upper domain, a black surface without ground moments and a Henyey-Greenstein troposphere without a Mie table. Other configurations use the NumPy path; every other value leaves MLX off. With `1` and no `mlx` package, a run stops with `E_MLX_UNAVAILABLE`. | user |
| `SOSFUSE2_MLX_TILE_SAMPLES` | backends (`mlx_backend.tile_samples`) | `1048576` (2**20) | A positive integer caps the path samples one MLX device tile holds; a single cell with more samples forms its own tile. Empty after trimming uses the default; zero or negative values raise `ValueError`. | tuning |
| `SOSFUSE2_LAUNCH_BUDGET_SECONDS` | backends (`device_profile.select`) | the selected profile | A finite nonnegative number overrides the launch time limit. Empty, `0`, `none` or `off` disables it; invalid, negative or nonfinite values raise `E_DEVICE_PROFILE`. Disabling it leaves the pre-launch work cap of the warm transport active, computed for 0.8 × 30 s; one cell above that cap raises `E_LAUNCH_BUDGET`. | tuning |
| `SOSFUSE2_LAUNCH_TARGET_SECONDS` | backends (`device_profile.select`) | the selected profile | A finite nonnegative number overrides the target. Empty, `0`, `none` or `off` disables it; invalid, negative or nonfinite values, or a target above the launch time limit, raise `E_DEVICE_PROFILE`. | tuning |
| `SOSFUSE2_LAUNCH_PAUSE_MS` | backends (`device_profile.select`) | the selected profile | A finite nonnegative number overrides the pause. Empty, nonnumeric, negative and nonfinite values raise `E_DEVICE_PROFILE`. | tuning |
| `SOSFUSE2_LAUNCH_CELLS` | backends (`device_profile.select`) | unset (follows the selected profile) | When empty after trimming, the launch mode follows the profile, normally using adaptive splitting when a target is set and a single launch otherwise; a profile that requires split launches stops with `E_LAUNCH_BUDGET_CONFIG` when a grid of several cells has neither a target nor a fixed cell count. Setting `0` requests a single launch where allowed, and a positive integer sets a fixed upper bound that a predicted time budget may reduce further. Other values raise `E_LAUNCH_CELLS`, and `backends/flags_env.py` accepts only an empty value or `0` for single-launch configs. | tuning |
| `SOSFUSE2_RSS_GB` | foundation (`host_limits.rss_override_gb`) | unset (see [Host RSS limits](#host-rss-limits)) | A positive finite number in GB replaces both the per-process default and the ceiling for `--rss-gb`. Empty after trimming uses the host rule. Zero, negative or nonnumeric values and limits that are nonfinite in bytes raise `E_RSS_ENV`. | tuning |
| `SOSFUSE2_POOL_LIMIT_GIB` | backends (`device_profile.pool_limit_bytes`) | unset: the memory available to the pool (driver-free memory plus all bytes the pool holds) minus 1 GiB | A positive finite number caps the CuPy pool in GiB, no higher than the default. Empty after trimming uses the default; invalid, negative or nonfinite values, and values truncating to less than one byte, raise `E_DEVICE_PROFILE`. | tuning |
| `SOSFUSE2_FIELD_RESERVE` | backends (`device_profile.reserved_field_bytes`) | `30` | A positive finite number sets how many source fields the persistent service reserves. Empty after trimming uses `30`; invalid, zero, negative or nonfinite values raise `E_DEVICE_PROFILE`. Smaller values can risk device memory exhaustion during the solve. | tuning |
| `SOSFUSE2_FIELD_RESERVE_MIN` | backends and jobs (persistent admission) | unset | A positive finite number at or below `SOSFUSE2_FIELD_RESERVE` allows the service to reduce the reserve for a lone state. Empty after trimming uses the fixed reserve. Invalid bounds raise `E_DEVICE_PROFILE`. See [Device memory diagnostics](#device-memory-diagnostics). | tuning |
| `SOSFUSE2_ANDERSON_HISTORY` | solve (`Solver.solve`) | `device` | `host` keeps up to four source and residual vectors per wavelength in host memory; an eligible CUDA solve stages the active wavelength's history for FP32 products and mixing on the device, other solves mix on the host (eligibility depends on the Legendre order, the wavelengths per chunk, the stratospheric layer, `SOSFUSE2_DETERMINISTIC` and free device memory), and both solve the small Gram system in FP64 on the host. All other strings except `device`, including empty, raise `E_ANDERSON_HISTORY`. The choice is recorded in `timings[i].anderson_history` for each chunk and leaves the admission estimate unchanged. | tuning |
| `SOSFUSE2_GROUND_SH_L` | solve (`ground_sh.configured_order`) | `0` | After trimming, `0`, `16`, `24` or `32` sets the ground convolution order; `0` is off. Empty and other values raise `E_GROUND_SH_L`. Recorded in signed settings as `kernel.ground_sh`. The `--ground-sh` option of `jobs/run_state.py` and `jobs/pipeline.py` takes precedence. | tuning |
| `SOSFUSE2_KSPLIT` | solve (`source_solver.ksplit_setting`) | `3` for uninstrumented CuPy transport with STRATO, W=15, L≤16 and WARM_RAYSPLIT off. Otherwise `0`, including W=1. | After trimming, empty uses the default; `0` through `3` select a mode. Other values or an unsupported explicit mode raise `E_KSPLIT_FLAGS`; instrumented transport forces mode `0`. | tuning |
| `SOSFUSE2_KSPLIT_FALLBACK` | solve (`ksplit`) | `1`. `checks/gate.py` and benchmark mode of `checks/solver_diagnostics.py` set `0` when unset. | Exactly `1` permits mode `0` when the minimum peak exceeds the solo budget. `0` raises `E_KSPLIT_MEMORY`. Empty and other values raise `E_KSPLIT_FALLBACK` during nonzero split preflight. See [Device memory diagnostics](#device-memory-diagnostics). | tuning |
| `SOSFUSE2_KSPLIT_AERO_MB` | solve (`ksplit`) | `1024` MB for `SOSFUSE2_KSPLIT=1`, otherwise sized by the memory rule | A number converted to a byte count caps the aerosol buffer for one chunk in MB, clamping zero or negative values to at least one point. An empty value after trimming uses the applicable default, while nonnumeric, nonfinite or overflowing values fail during numeric conversion. | tuning |
| `STRATO_ATTACH_CACHE` | atmosphere (`stratosphere`) | `<geometry cache>/attach_cache` under `jobs/run_state.py`, unset (in-process memo only) for other callers | A nonempty path after trimming sets the on-disk attach cache. The runner treats `off` case-insensitively as disabled and an empty value as its default, whereas direct atmosphere callers treat an empty value as no disk cache and any nonempty value, including `off`, as a path. `checks/gate.py` sets this variable for its child processes. `jobs/plan_queue.py` reads it as the attach folder when neither `--attach-cache` nor `--cache-root` is given, treating an empty value and `off` in any letter case as unset, and writes it into the commands of its plan when the attach folder is not `<geometry cache>/attach_cache`. The `--attach-cache DIR|off` option of `jobs/run_state.py` and `jobs/pipeline.py` takes precedence. | tuning |

`SOSFUSE2_FP32_ACC` and `SOSFUSE2_FP32_GEOM` are not variables. The package accepts the two names as retired tokens in
`SOSFUSE2_KERNEL_FLAGS` and warns when it ignores them. The CUDA transport is FP32 throughout, so neither token has an
effect.

## Diagnostics

| Variable | Layer | Default | Effect and precedence | Use |
|---|---|---|---|---|
| `SOSFUSE2_STRATO_HOST_COMPONENT_RADIUS` | atmosphere (`stratosphere`) | `1` | Exactly `0` selects the older radius loop; empty and every other string select the component loop. | diagnostic |
| `SOSFUSE2_STRATO_HOST_NOPREFIX` | atmosphere (`stratosphere`) | unset | Exactly `1` uses the plain node loop; empty and every other string use the engine's prefix segments. | diagnostic |
| `SOSFUSE2_VRAM_TRACE` | backends (`vram_trace.enabled`), state runner and persistent service | off | Exactly `1` enables CuPy memory-pool accounting. See [Device memory diagnostics](#device-memory-diagnostics) for the phase events and metadata. | diagnostic |
| `SOSFUSE2_KSPLIT_VERIFY` | solve (`source_solver.Kernel`) | unset | `1` after trimming requests a PROD/VPROD comparison, which needs mode 2 or 3 where applicable. Empty and all other strings leave it off. | diagnostic |
| `SOSFUSE2_KSPLIT_VERIFY_RTOL` | solve (`ksplit_tolerance.tolerances`) | `1e-5` | Accepts a finite nonnegative number. Empty, nonnumeric, negative and nonfinite values raise `E_KSPLIT_VERIFY`; the default applies only when the name is unset. | diagnostic |
| `SOSFUSE2_KSPLIT_VERIFY_ATOL` | solve (`ksplit_tolerance.tolerances`) | `1e-7` | Accepts a finite nonnegative number. Empty, nonnumeric, negative and nonfinite values raise `E_KSPLIT_VERIFY`; the default applies only when the name is unset. | diagnostic |
| `SOSFUSE2_FL_PROJECT` | solve (`source_solver.Kernel`) | unset | `1` after trimming also runs first light of a single launch as a ray launch followed by a projection; empty and other strings leave this option off. | diagnostic |
| `SOSFUSE2_EXPORT_MOMENT` | solve (`source_iteration.Solver`, `jobs/run_state.py`) | unset | Setting `1` retains and writes spherical-harmonic (SH) moments of the incident radiance, while an empty value or any other string leaves this off. In the persistent queue, `1` raises `E_EXPORT_MOMENT_PERSISTENT`. | diagnostic |
| `SOSFUSE2_SECTION_CLOCKS_DIR` | solve (`source_solver.Kernel`) | unset | A nonempty path after trimming sets the output folder for section counters. An empty value raises `E_SECTION_CLOCK_OUTPUT` when `EXP_SECTION_CLOCKS` is selected. | diagnostic |
| `SOSFUSE2_P2MIN_NVTX` | solve (`ksplit`) | `0` | Exactly `1` prints identification lines for the ray-consumer variant and wraps each of its launches in the NVTX range `p2min_slimray`; empty and other strings leave both off. | diagnostic |
| `SOSFUSE2_VPROD_DUMP` | solve (`ksplit`) | unset | A nonempty file path after trimming saves records of one chunk; empty disables the dump. Other paths are used as supplied. | diagnostic |
| `SOSFUSE2_WF_DUMP` | solve (`ksplit`) | unset | Any nonempty string is the folder for one chunk’s inputs and ray radiances. Empty disables the dump. The path is used as supplied. The `records` array changed from seven to six floats per sample point. The six-float layout has `meta.column_record_version = 2` in the NPZ dump. | diagnostic |

## Variables the package sets for its own processes

| Variable | Layer | Default | Effect and precedence | Use |
|---|---|---|---|---|
| `SOSFUSE2_COLUMN_CACHE_JOB` | precompute (`column_cache.job_root`) | unset: no first-use comparison | A nonempty path after trimming is the job root for cache markers; empty disables first-use comparison. `jobs/run_state.py` sets it to its `--work-root`. | internal |
| `SOSFUSE2_STATE_ID` | precompute (`column_cache`, `column_parallel`), jobs (`run_state.solve_chunk`) | `COLUMN_CACHE` | Any string, including empty, is written as the state label of the PHASE events of the column, host, sun, ground and geometry preparation caches and pools. The solve failure event uses the same label, with `GPU_SERVICE` as its default. `jobs/run_state.py` sets it. | internal |
| `SOSFUSE2_STRATO_ASYNC` | solve (`strato_sh_aux`) | unset: `strato_async` of the device profile | After trimming, `1` or `0` overrides the profile and empty uses it; other values raise `E_STRATO_CONTROL`. The value has no effect on a solve: only the check worker reads it, for the `async_` field of its records, and `checks/gate.py` sets it explicitly. | internal |
| `SOS_PERSISTENT_ROOT` | jobs (`pipeline`, `pipeline_execution`) | unset | The pipeline sets the socket folder for its children. An unset or empty value selects the nonpersistent path. A nonempty value is used as supplied. | internal |
| `SOS_PERSISTENT_KEY` | jobs (`pipeline`, `pipeline_execution`) | unset | The pipeline sets the hexadecimal authentication key for its children. If the runner uses the persistent path, empty decodes to empty bytes and an unset or invalid value fails. | internal |
| `SOS_PERSISTENT_TEST_PREP_FAIL` | jobs (`pipeline_execution`) | unset | A string matching the state name injects a preparation failure. Empty or another nonmatching string does not inject a failure. | internal |
| `SOS_PERSISTENT_TEST_OBSERVER_DELAY` | jobs (`pipeline_execution`) | `0` | A finite nonnegative number sets the delay in seconds; zero does not sleep. Empty or nonnumeric values fail conversion; negative or nonfinite values fail when sleeping. | internal |

## Host RSS limits

The host rule uses a 6 GB ceiling and default on non-Linux hosts and when usable RAM is unknown.
On Linux, usable RAM is the smallest known value among physical RAM and the finite memory limits of visible cgroup
ancestors (v2 `memory.max` and v1 `memory.limit_in_bytes`). The ceiling is 80 % of usable RAM below 24 GB,
75 % from 24 GB up to but excluding 48 GB, and the smaller of usable RAM and 128 GB from 48 GB upward.

On Linux with known usable RAM, the default divides 80 % of usable RAM by the number of concurrent processes,
applies a 6 GB floor, and then applies the ceiling. The runner and solver diagnostics count one process each.
The pipeline counts its state workers and adds one process in persistent mode. The floor can put the sum of the
per-process limits above the shared budget. For example, 4 GB of usable RAM gives a 3.2 GB ceiling and default.

An explicit `--rss-gb` selects a positive limit up to the ceiling, including when `SOSFUSE2_RSS_GB` sets the ceiling.
The limit applies per process. Pipeline and planner worker sizing use the platform and usable RAM independently of
the RSS override. The watchdog checks the process’s peak RSS every 0.5 seconds. At 95% of the limit or
above, it prints `E_RSS_LIMIT` and exits with status 75.

## Device memory diagnostics

In `meta.json`, `i` in `timings[i]` is the chunk index.

The runner writes the KSPLIT decision to `ksplit` in `meta.json` and emits a `ksplit` phase event with the decision.
The pipeline collects that decision in the state's `ksplit` entry in `pipeline_summary.json`. The metadata also
records the effective mode as `ksplit_effective`.

The decision's `reason` is `fits` for a successful split or `state_minimum_exceeds_solo_budget` for a memory fallback.
When the configured mode resolves directly to 0, the reason identifies the setting or configuration, such as
`explicit_zero`, `instrumented`, `numpy`, `fw1`, `sh_order_gt_16`, `no_strato`, or `warm_raysplit`.
The `items` maps hold named byte estimates for the minimum `peak` and budget `reserve`. Both maps are empty
when mode 0 is selected before memory preflight. A fitted plan for mode 2 or 3 also has `items.planned_peak`
when the preflight records its selected plan.

A memory fallback emits `KSPLIT_FALLBACK` and warns on stderr. The warning gives the two largest positive
byte counts available across `items.peak` and `items.reserve`, along with the modes, peak, solo budget and reason.
A standalone run prints the warning when the fallback occurs. The pipeline prints it when collecting the logs at the end of the run and includes the event in the state's `ksplit_fallback` summary entry.

In persistent mode, `vram_wait` records a wait for a resident slot or available memory, `vram_rejected` records an
admission failure, and the per-state `gpu_acquired` event records the initial reservation. Each event includes
`items` for payload arrays, source fields and the KSPLIT floor. Wait and rejection report estimates at the requested
reserve. On admission, `items.source_fields` reflects the effective reserve and the chunk's
`timings[i].vram_admission` records `reserved_bytes` and `items`.

With `SOSFUSE2_FIELD_RESERVE_MIN` set, `gpu_acquired` and the chunk's `timings[i].field_reserve` record the requested,
minimum and effective factors, the byte counts and the reason (`requested_fits` or `reduced_to_fit`). Only a lone
state can reduce its field reserve. A lone state whose payload and minimum reserve exceed available capacity fails
with `E_PERSISTENT_VRAM_ADMISSION`.

With `SOSFUSE2_VRAM_TRACE=1` on CuPy, `vram_alloc` and `vram_free` record pool blocks of at least 1 MiB.
`vram_oom` records the pool composition around a CuPy pool out-of-memory error. The top-level `vram_trace` in
`meta.json` reports `max_resident_bytes` by allocation label and the composition at peak pool usage across
spectral chunks, including smaller tracked blocks.

## Production queues and checks

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
  `ksplit_effective` in table metadata. See [Device memory diagnostics](#device-memory-diagnostics)
  for the reason, itemised estimates, and persistent admission records. Setting `SOSFUSE2_VRAM_TRACE=1` also records
  CuPy memory-pool accounting (`vram_trace`) in `meta.json`.
- Both runner and pipeline enforce a host-dependent per-process resident set size (RSS) limit, with `SOSFUSE2_RSS_GB`
  setting both the default and ceiling for `--rss-gb`. [Host RSS limits](#host-rss-limits)
  describes the host rule and how the pipeline handles concurrent processes.
- `plan_queue` stops with `E_PLAN_CPU_BUDGET` when GPUs x workers per GPU x column workers exceed
  `pipeline.cpu_budget()`. Because `jobs/pipeline.py` can still widen a state's column pool, this bounds only the
  planned request rather than acting as a CPU quota. The plan also lists the `ground_aerosol01` host cache after
  `ground`, with an unmeasured run-time cost.
- While building a state, `tables_from_sources` holds the same `<out>/.worker.lock` used by `run_state`, skipping
  worker-owned states while a worker starting concurrently stops with `E_OUTPUT_BUSY`.
- Each pipeline invocation writes logs to its own `logs/<run>/` directory referenced from `pipeline_summary.json`,
  preserving both cold and warm traces. Benchmark stage timers go to `timings-<run>.json`, named in `benchmark.json`.
- CUDA state solves (`run_state` and the persistent service) and `checks/solver_diagnostics.py` cap the CuPy default
  memory pool via `backends/device_profile.pool_limit_bytes(free)` at the
  smaller of the memory available to the pool minus 1 GiB or `SOSFUSE2_POOL_LIMIT_GIB` in GiB, where the available
  memory is the driver's free memory plus all bytes the pool already holds. Values truncating below one byte are
  rejected, and a card with under 4 GiB of driver-free and reusable pool memory stops with
  `E_VRAM_FREE: need at least 4 GiB free`. This cap replaces the
  1.3.4 limit of min(20 GiB, free − 1 GiB), which rejected states needing over 20 GiB on every card with
  `E_PERSISTENT_VRAM_ADMISSION`. Because concurrent pipeline processes on one GPU each size their pool from startup free
  memory, set `SOSFUSE2_POOL_LIMIT_GIB` to partition the card. During admission, the persistent service requests field
  reserve factor `SOSFUSE2_FIELD_RESERVE` (default 30) and lowers it to fit available memory when a lone state exceeds
  capacity there but fits at `SOSFUSE2_FIELD_RESERVE_MIN`. A smaller reserve admits larger or more concurrent states,
  but because admission is a reservation, setting it too low turns waits into solve-time out-of-memory errors. Neither
  reserve variable enters any identity or alters numerics.
- On CUDA, a warm launch that cannot use the angular direction table (for example when the table does not fit in free
  memory, or with `SOSFUSE2_DETERMINISTIC=1`) contracts the coefficients directly or uses the fused kernel. A solve that
  falls back to the fused kernel can take many times longer. A cell above the pre-launch work cap of direct contraction
  stops with `E_LAUNCH_BUDGET` instead of falling back.

## Kernel compilation cache

On CUDA machines, NVRTC compiles kernels on first use. Without `SOSFUSE2_COMPILE_CACHE`, `jobs/run_state.py` stores
compiled kernels in `nvrtc_cache_shared` in the parent of `--work-root` (`backends/compile_cache.py` `default_root`),
while a persistent pipeline service and the reference tests each use a run-specific folder. Setting
`SOSFUSE2_COMPILE_CACHE=<dir>` redirects all of them to that directory, and
`python -B -m sosfuse2.jobs.precompile --jobs 8` fills that cache before a queue starts. The angular and direct
warm-transport kernels can still compile on their first use.

## Experimental MLX transport (Apple silicon)

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

## Variables of the host that the package reads or sets

The package reads `XDG_CACHE_HOME` to locate the default work root, sets `OMP_NUM_THREADS` and the other thread-count
variables in `jobs/pipeline.py::THREAD_ENV` from `--threads`, records `CUDA_VISIBLE_DEVICES` and sets it from `--gpu`,
and sets `PYTHONDONTWRITEBYTECODE` for child processes. For CuPy, `compile_cache.configure` sets `CUPY_CACHE_DIR` to a
keyed folder under `SOSFUSE2_COMPILE_CACHE` or the caller's root; the runner's default root is the sibling
`<work-root parent>/nvrtc_cache_shared`. The runner sets `CUDA_CACHE_PATH` separately to
`<work-root>/tmp/<state name>/cuda_driver_cache`. An explicit `--threads N` of `jobs/run_state.py` also sets each of the
four preparation-pool variables that is unset or blank after trimming to N for that run; nonblank values take
precedence, and the previous values are restored when the run ends.

## Variables of the tests only

The air wavelength and cache store tests accept a file path for each external input. An unset or blank
variable, or a path that does not name a file, skips only the cases that need that input.

| Variable | Test | Effect |
|---|---|---|
| `SOSFUSE2_TEST_OZONE_SPECTRUM` | `tests/test_air_wavelength.py` | Serdyuchenko ozone spectrum used for relabelling. |
| `SOSFUSE2_TEST_HSRS` | `tests/test_air_wavelength.py` | Full-resolution TSIS-1 HSRS input CSV. |
| `SOSFUSE2_TEST_SPECTRA_BUILDER` | `tests/test_air_wavelength.py` | Python script that builds the fine spectral grid. |
| `SOSFUSE2_TEST_HSRS_P1NM` | `tests/test_air_wavelength.py` | Alternate HSRS product CSV used to check product selection. |
| `SOSFUSE2_TEST_ALTERNATE_OZONE` | `tests/test_air_wavelength.py` | Alternate ozone spectrum used to check source rejection. |
| `SOSFUSE2_TEST_SPECTRA_CHECK` | `tests/test_air_wavelength.py` | CPU comparison script with its supporting files. |
| `SOSFUSE2_TEST_BAKE_DRIVER` | `tests/test_cache_store.py` | Python driver that prepares a run and collects its cache. |
| `SOSFUSE2_TEST_BAKE_PLANNER` | `tests/test_cache_store.py` | Python script that plans cold and warm runs with the bake driver. |
| `SOSFUSE2_CPU_REPLAY_SLOW` | `tests/test_cpu_replay.py` | Exactly `1` runs the albedo 0.3 replay; unset, empty and all other strings skip it. |
| `SOSFUSE2_CPU_REPLAY_RTOL` | `tests/test_cpu_replay.py` | A number in `(0, gate.RTOL]` sets the tolerance. Unset or empty after trimming uses `gate.RTOL`; nonnumeric, nonfinite, zero, negative or looser values fail with `E_TEST_CONFIG`. |
| `SOSFUSE2_WALKTHROUGH_SLOW` | `tests/test_walkthrough.py` | Exactly `1` runs the comparison against `jobs/run_state.py`; unset, empty and all other strings skip it. |
| `SOSFUSE2_OPTION_SMOKE_SLOW` | `tests/test_option_combinations.py` | Exactly `1` runs the solve grid of the 200 km settings example through `run_state` on a CUDA machine, on a test state for one wavelength, two solar elevations and 2×2 directions, and checks its records; unset, empty and all other strings skip it. The NumPy and CUDA comparison on a small 200 km fixture, which shares the example's upper air, stratospheric layer, L, tolerance, history and ground order but not its grid or albedo, runs on every CUDA machine. |
| `SOSFUSE2_APRIME_RUNS` | `tests/test_ozone_temperature.py` | Exactly `1` runs the reduced NumPy solves; unset, empty and all other strings skip them. |
| `SOSFUSE2_KEEP_TEST_OUTPUT` | `tests/test_column_cache.py`, `tests/test_pipeline_persistent.py` | Exactly `1` keeps test work folders; unset, empty and all other strings remove them. |
| `SOSFUSE2_HITRAN_CIA` | `tests/test_b2_s2.py` | A nonempty path supplies original O₄ data. An unset or empty value omits the O₄ comparison. Both input variables empty skip the test. |
| `SOSFUSE2_BIRA_DIR` | `tests/test_b2_s2.py` | A nonempty path supplies original NO₂ data. An unset or empty value omits the NO₂ comparison. Both input variables empty skip the test. |
| `SOSFUSE2_STRATO_BASELINE` | `tests/test_strato_delta_adversarial.py` | A nonempty path after trimming names the fixed measurement folder; unset or empty skips that comparison. The supplied folder must contain its expected files. |
| `SOSFUSE2_STRATO_COLOR_ARTIFACT` | `tests/test_strato_adversarial.py` | An unset value uses the default colour output folder; a supplied string is used as a path, and empty denotes the current directory. The test skips if its summary is missing. |
| `PERSISTENT_TEST_BACKEND` | `tests/test_pipeline_persistent.py`, `tests/test_column_cache.py` | Unset selects `numpy`; `numpy` or `cupy` is passed as the queue backend. Empty and other strings are passed through and fail backend validation. |
| `PERSISTENT_TEST_DEADLINE` | `tests/test_pipeline_persistent.py`, `tests/test_column_cache.py` | A string converted by `float` sets the timeout in seconds; unset defaults to 2400 in both. Empty or nonnumeric strings fail conversion; there is no finite or positive check, and the persistent test clamps its remaining wait to at least 0.01 seconds. |
| `PERSISTENT_TEST_OBSERVER_DELAY` | `tests/test_pipeline_persistent.py` | The string is passed unchanged to the queue of the interleaving case as `SOS_PERSISTENT_TEST_OBSERVER_DELAY`, and unset passes `0.02`. The queue converts it with `float`, so an empty or nonnumeric string fails conversion. |
| `SOSFUSE2_TEST_SIMPSON_FIXTURE` | `tests/test_simpson_prefix.py` | Optional ray fixture path relative to the research checkout. The synthetic cases run without it. |
| `SADDLE_DATA_ROOT` | `tests/test_plan_queue.py`, `tests/test_column_parallel.py` | The tests search this input folder before `SOSFUSE2_SKY_SOS_ROOT`; the cases that need a packed full-grid geometry cache and run logs skip when neither folder holds them. |
