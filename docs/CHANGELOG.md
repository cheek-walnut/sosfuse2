# Changelog

Release changes from 1.2.0 on, newest first. Versions 1.0.0 to 1.2.3 were published together, and earlier entries are
not reconstructed here. The 1.1.0 absorber changes (NO₂ and O₄) are described in `docs/VALIDATION.md`.

## 1.4.0

1.4.0 moves the modules into ten folders by layer (`docs/ARCHITECTURE.md`) and adds a command entry point,
`python -m sosfuse2 <command>`, and a walkthrough. No compatibility modules remain at the old names, so code that
imports `sosfuse2.<old name>` or runs `python -m sosfuse2.<old name>` has to use the new names. Output folders written
by 1.3.x are not resumed and caches are rebuilt (**Compatibility** below).

- **Module names.** The table at the end of this section lists every module that moved; names are relative to
  `sosfuse2`. The folders `accepted/` and `host/` are gone.
- **Data and kernel files.** `kernel.cu` and `native/` are in `backends/`, the three JSON files of `configs/` in
  `backends/configs/`, `fixtures/` in `checks/fixtures/`, and `gate/` in `checks/gate_data/`. `paths.CONFIGS`,
  `paths.GATE`, `paths.NATIVE` and `paths.FIXTURES` point at the new places, and `paths.KERNEL_CU` is new.
  `paths.ACCEPTED` points at `solve/`, which holds `engine.py` together with the other solve modules. `assets/` and
  `paths.ASSETS` are unchanged.
- **Renamed functions.** Three functions were renamed so that no function shares its name with a folder or a new module:
  `checks/gate.py::outputs` is `pinned_hashes` (the recorded key `outputs` is unchanged),
  `backends/numpy_reference.py::extinction` is `extinction_profile`, and `checks/solver_diagnostics.py::run_state` is
  `probe_state`. The constant `host_limits.EDA_MARKERS` is `foundation/host_limits.py::PROTECTION_MARKERS`, with the
  same value.
- **Device profile `split`.** The profile for a card with a short launch budget (split launches,
  `backends/configs/devices.json`) is named `split`. `SOSFUSE2_DEVICE_PROFILE=eda` and `plan_queue --cost-profile eda`,
  the 1.3.x name, still select it. The old name is accepted by the environment variable and the command option only, and
  the key in `backends/configs/devices.json` and in `plan_queue.COST_PROFILES` is `split`. The keys `slots`, `source`,
  `state_rss_gb` and `strato_launch_pause_ms` were removed from every profile, and `memory_source` and
  `requires_split_source` of this profile are now `memory_note` and `requires_split_note`. Run metadata and gate reports
  record `split`. Its CUDA baselines moved to the folders of the recomputed toolkeys (`7de21e0b57f543b1`,
  `0fbb3921b0dbe0eb`, `e8ee3bcaaf06747b`); the array files and their sha256 are unchanged, and
  `baselines/BASELINE_LOG.md` lists the old and new keys.
- **Moved definitions.** `CIA_SHA` and `RULES_VERSION` are defined in `atmosphere/minor_absorbers.py`, and the O₄
  generator imports them from there. The memory monitor `monitor` is in `foundation/host_limits.py`. Four upward imports
  disappear with these two moves. `package_hashes`, `package_info`, `ACCEPTED_SOURCES`, `HOST_MODULES` and `DATA_ASSETS`
  are defined in `foundation/content_identity.py` and stay importable from `jobs/run_state.py`; `outputs/pack_state.py`
  no longer defines `HERE`.
- **Bytecode side effect.** Importing `atmosphere/minor_absorbers.py` no longer imports the O₄ generator, so it no
  longer sets `sys.dont_write_bytecode = True`.
- **Parallel geometry.** The import hook in `sosfuse2/__init__.py` that swapped in the parallel `PackedCache` is gone.
  `solve/ray_geometry.py` imports `solve/ray_geometry_parallel.py` as its last statement, so an error in the adapter
  now surfaces at import time.
- **Command entry point.** `python -m sosfuse2` lists the commands, and `python -m sosfuse2 <module name>` runs that
  module with the same arguments and exit code. An unknown command exits with 2 before any calculation code is imported.
  The commands are `run_state`, `radiance_table`, `tables_from_sources`, `pipeline`, `plan_queue`, `cache_store`,
  `precompile`, `prepare_minor_absorbers`, `gate`, `solver_diagnostics`, `kernel_diagnostics`, `stratosphere_checks`,
  `available_cpus`, `generate_density_profile`, `generate_minor_absorbers`, `generate_strato_optics`,
  `generate_tropo_optics`, `strato_states`, `flags_env`, `pipeline_execution` (internal) and `walkthrough`. Seven of
  them run modules that had another name in 1.3.x: `run_gpu_state_v3` is `run_state`, `table_fast` is `radiance_table`,
  `run_fuse` is `solver_diagnostics`, `gate_flags` is `kernel_diagnostics`, `validate_strato` is `stratosphere_checks`,
  `host.host_cores` is `available_cpus` and `configs.flags_env` is `flags_env`. The direct form
  `python -B -m sosfuse2.<layer>.<module>` works as before.
- **Walkthrough.** `python -m sosfuse2 walkthrough` solves a small reduced state with the NumPy backend and shows the
  three main calls with the input and output of each; `--write` regenerates `docs/WALKTHROUGH.md`.
- **Documentation.** `AGENTS.md` lists the files to open first for each kind of task, `docs/ARCHITECTURE.md` describes
  the layers, and `docs/ENVIRONMENT.md` lists the environment variables. New tests keep these in step with the code:
  `tests/test_layers.py` (import direction), `tests/test_names.py` (names), `tests/test_entry_points.py` (commands),
  `tests/test_walkthrough.py` and `tests/test_docs_pointers.py`.
- **Validation figures.** The README figure is now the eight-state comparison with MYSTIC at SZA 93° and 96°
  (`docs/validation/mystic_states.png`). `docs/VALIDATION.md` adds Figure 2, Figure 3 and Table 14, the Paranal ratio
  from −5° to −15° with and without a night-sky floor, adds a column with all three US Standard Atmosphere options to
  Table 13, and corrects the SZA 93° range of the eight-state comparison to −0.30 % to +0.77 %. Tables 14 to 17 of 1.3.4
  are now Tables 15 to 18. The figure scripts `plot_mystic_states.py`, `plot_paranal_twilight.py` and `figure_style.py`
  and the CSVs `mystic_states.csv` and `paranal_twilight.csv` are in `docs/validation/`.
- **Provenance keys.** The keys of `code_hashes` carry the folder, so signatures and cache keys differ from 1.3.x
  (`docs/PROVENANCE_V3.md`); caches are rebuilt on first use. The entries of `paths.PRODUCTION_MODULES`,
  `solve/engine.py::SOURCE_FILES` and `precompute/endpoint_contract.py::UPSTREAM` carry the folder as well.
- **Content identity.** `foundation/content_identity.py` (rule identifier `sosfuse2-identity-2`) hashes Python sources
  without comments and docstrings, C and CUDA sources without comments, JSON files without their note keys, and numeric
  assets without their per-asset description fields. Changes of spacing inside a line or of indentation still change the
  identity, while trailing whitespace and blank lines do not, and prose in settings and fixture JSON moved to `notes`
  keys. State identity omits the description and label keys `description`, `note`, `notes`, `note_*`, `*_note`, `group`,
  `role`, `holdout`, `out_of_axis_range` and `q2d_override` while keeping the `settings`, `parameters`, `stratosphere`
  and `density` blocks whole. Numeric assets record a computation identity apart from the sha256 of the file bytes; a
  spectral block is checked against that computation identity, so whitespace, indentation and description changes pass
  while changed numbers are rejected. The density profile and the external source files keep their byte pins, and the
  earlier hashes are registered so that older records still match.
- **Minor absorber hashes.** `atmosphere/minor_absorbers.py::load` returns the computation identity of the asset under
  `sha256` and the SHA-256 of the file bytes under the new key `raw_sha256`. Up to 1.3.4 `sha256` was the byte hash, so
  code that compares it with a file checksum has to read `raw_sha256`. `canonical_hashes` in the same module maps the
  byte hash of the present asset, or a registered earlier one, to the identity, so records that carry byte hashes under
  `minor_absorber_sha256` or under the `asset_sha256` of a minor absorber entry still match.
- **Provenance format v5.** The runner, the table tools, the column, geometry, host, sun-table, attach, query and block
  caches and the gate code identity use the `sosfuse2-identity-2` rules, so every cache key changes once and is rebuilt
  on first use. Tables are written with provenance format v5, setting `contract: "v5"` in `meta.json`
  (`jobs/run_state.py` `CONTRACT_VERSION`).
- **CUDA memory pool and admission.** Up to 1.3.4 the runner, the solver diagnostics and the persistent service capped
  the CuPy default memory pool at min(20 GiB, free − 1 GiB), and the service used the same cap as its admission
  capacity, so a state needing more than 20 GiB failed admission on any card with `E_PERSISTENT_VRAM_ADMISSION`. The cap
  is now `backends/device_profile.pool_limit_bytes(free)`, which is the free device memory minus 1 GiB, or
  `SOSFUSE2_POOL_LIMIT_GIB` in GiB when that is smaller; a value that truncates to less than one byte is rejected, and
  the pool shortage error reads `E_VRAM_FREE: need at least 4 GiB free`. The admission reserve is
  `device_profile.reserved_field_bytes`, set by `SOSFUSE2_FIELD_RESERVE` fields per state with default 30. A smaller
  reserve admits larger states or more states at once, though because admission is a reservation, too small a value
  turns a wait into an out-of-memory error during the solve. Several pipeline processes on one GPU each size their pool
  from the free memory they see at start, so set `SOSFUSE2_POOL_LIMIT_GIB` to split a card between them. Neither
  variable is part of any identity, and numerics are unchanged.
- **Grid rules in the settings.** `h_refine` accepts `b<m>a<k>`, which splits each height interval at or below 13 km
  into m parts and each interval above into k parts, and `b<m>a<k1>.<k2>.<k3>.<k4>.<k5>`, which gives the five intervals
  above 13 km their own counts, with m and k from 1 to 12. The names `none`, `mid` and `below13mid_above13thirds` are
  aliases of `b1a1`, `b2a2` and `b2a3` and give the same heights as before. `chi_step_deg` also takes 0.125, and the
  optional `chi_refine_deg` `[lo, hi]` adds the midpoint of every node pair inside that range of the 85–110° solar
  zenith grid. Both ends must be nodes of that grid. `nr` and `nchi` are filled from the rules when omitted and must
  match them when given, while the reduced grid stays fixed. Settings without `chi_refine_deg` give the same grids as
  before.
- **G-F reference.** `gate --gf` always requires `--reference <table.npz>` and stops with `E_GATE_BASELINE_MISSING`
  without an existing file (`checks/gate.py` `gf`). `checks/gate_data/baselines/manifest.json` records only the file
  name under `gf.reference` (`gf_L19s3_tau0p01_aod0p4_g0p65_e2620f6f90c39dcb.npz`) together with its radiance SHA-256.
  G-F still requires the bake template of the research records through `SOSFUSE2_SKY_SOS_ROOT` and stops with
  `E_GATE_CONFIG` without it.
- **Table tools, checks and assets.** `jobs/tables_from_sources` records `tolerance` 0.001 and `convergence_scope`
  `<=0.1%` where it previously recorded 0.005 and `<=0.5%`, which did not match the solve settings.
  `outputs/radiance_table` and `jobs/tables_from_sources` default to `states_v2.json`. `checks/stratosphere_checks`
  reports `peak_rss_bytes` in bytes on Linux, where it previously wrote the kibibyte count at 1024 times too small, and
  no longer writes `risk_class`. `assets/states_v1_usstd.json` no longer has its top-level `derived_from` and `decision`
  keys, unused keys left `backends/configs/devices.json`, four unreferenced baseline `.npz` files were removed, CPU
  replay regression tests were added, and comments and strings were cleaned up across the modules.
- **Compatibility.** v3 and v4 tables are still read, whereas readers of 1.3.4 and earlier stop on v5 tables with
  `E_PROVENANCE_CONTRACT`, so a consumer that reads new tables needs 1.4.0 first. Because 1.4.0 moved the modules,
  renamed the device profile and changed the identity rules of `foundation/content_identity.py`, the code hashes of
  every table change. Output folders and saved sources written by 1.3.4 or any earlier version are therefore not resumed
  or extended: resuming an older output folder stops with `E_OUTPUT_IDENTITY` and reports the stored format version, and
  computing extra elevations from older saved sources stops with `E_EXTRA_SOURCE`, so a solve that should continue needs
  a new output folder. `outputs/radiance_table` and `jobs/tables_from_sources` keep reading older sources. Staged gate
  runs made with earlier code stop at `gate --bless --staged` with `E_GATE_CONFIG` and have to be staged again. Runs of
  `rig` and `mac` profiles are affected in the same way as `split` runs.
- **Known issues.** Reading the metadata of a table whose O₄ asset digest is not a registered earlier byte hash, as in
  tables written by 1.4.0, looks up the local O₄ asset and so resolves and creates the work root, `SOSFUSE2_WORK_ROOT`
  or else `$XDG_CACHE_HOME/sosfuse2` or `~/.cache/sosfuse2`, which 1.3.4 did not need for this read. Where that
  directory cannot be created the read stops with `E_WORK_ROOT`, so set `SOSFUSE2_WORK_ROOT` to a writable directory.
  Building a host cache entry in a folder that already holds entries of other keys prints
  `RuntimeWarning: HOST_CACHE_STALE n=… bytes=…`. `python -m sosfuse2 walkthrough` shows the warning in an empty work
  root, where those entries were written by the same run, and the results are unaffected. `docs/KNOWN_FAILURES.md`
  describes an optional reproduction test that fails, and tier 3 of `docs/CHECKS.md` names a research test that fails.

| Old module | New module |
|---|---|
| `accepted.engine` | `solve.engine` |
| `air_wavelength` | `atmosphere.air_wavelength` |
| `arithmetic` | `backends.arithmetic` |
| `cache_store` | `jobs.cache_store` |
| `column_cache` | `precompute.column_cache` |
| `column_parallel` | `precompute.column_parallel` |
| `column_table` | `rays.column_table` |
| `compile_cache` | `backends.compile_cache` |
| `components` | `atmosphere.components` |
| `configs.flags_env` | `backends.flags_env` |
| `density` | `atmosphere.density` |
| `device_profile` | `backends.device_profile` |
| `fused` | `solve.source_solver` |
| `gate` | `checks.gate` |
| `gate_flags` | `checks.kernel_diagnostics` |
| `generate_density_profile` | `asset_tools.generate_density_profile` |
| `generate_minor_absorbers` | `asset_tools.generate_minor_absorbers` |
| `generate_strato_optics` | `asset_tools.generate_strato_optics` |
| `generate_tropo_optics` | `asset_tools.generate_tropo_optics` |
| `geometry` | `solve.ray_geometry` |
| `geometry_contract` | `precompute.endpoint_contract` |
| `geometry_cpu` | `solve.ray_geometry_cpu` |
| `geometry_parallel` | `solve.ray_geometry_parallel` |
| `ground_parallel` | `precompute.ground_parallel` |
| `ground_sh` | `solve.ground_sh` |
| `host.host_cores` | `foundation.available_cpus` |
| `host.pack_state` | `outputs.pack_state` |
| `host_cache` | `precompute.host_cache` |
| `host_limits` | `foundation.host_limits` |
| `iprt` | `asset_tools.iprt` |
| `kernel_config` | `backends.flags` |
| `ksplit` | `solve.ksplit` |
| `mie` | `asset_tools.mie` |
| `minor_absorbers` | `atmosphere.minor_absorbers` |
| `numerical_checks` | `checks.numerical_checks` |
| `ozone_temperature` | `atmosphere.ozone_temperature` |
| `paths` | `foundation.paths` |
| `pipeline` | `jobs.pipeline` |
| `pipeline_execution` | `jobs.pipeline_execution` |
| `plan_queue` | `jobs.plan_queue` |
| `precompile` | `jobs.precompile` |
| `prepare_minor_absorbers` | `asset_tools.prepare_minor_absorbers` |
| `provenance` | `checks.validation_evidence` |
| `ray_columns` | `rays.ray_columns` |
| `rayleigh` | `atmosphere.rayleigh` |
| `reference` | `backends.numpy_reference` |
| `run_fuse` | `checks.solver_diagnostics` |
| `run_gpu_state_v3` | `jobs.run_state` |
| `sh_layout` | `backends.sh_layout` |
| `solver_columns` | `rays.shell_segments` |
| `solver_core` | `solve.engine_core` |
| `solver_legacy` | `rays.incoming_rings` |
| `solver_optics` | `atmosphere.extinction` |
| `solver_poly` | `rays.zenith_interpolation` |
| `solver_render` | `outputs.preview_functions` |
| `solver_sky` | `rays.incoming_quadrature` |
| `spectral` | `atmosphere.spectral` |
| `strato_parallel` | `solve.strato_sh_aux` |
| `strato_states` | `asset_tools.strato_states` |
| `stratosphere` | `atmosphere.stratosphere` |
| `sun_parallel` | `precompute.sun_parallel` |
| `table_fast` | `outputs.radiance_table` |
| `table_q` | `outputs.quantized_table` |
| `tables_from_sources` | `jobs.tables_from_sources` |
| `testing` | `checks.testkit` |
| `timers` | `foundation.timers` |
| `tropo_mie` | `atmosphere.tropo_mie` |
| `tropo_profile` | `atmosphere.tropo_profile` |
| `validate` | `checks.validate` |
| `validate_strato` | `checks.stratosphere_checks` |
| `verify_tol` | `solve.ksplit_tolerance` |

## 1.3.4

1.3.4 changes documentation only; the code, the assets and the tables are those of 1.3.2.

- `docs/VALIDATION.md` adds two sections: *Grids in solar zenith angle and in height*, which changes the grid in solar
  zenith angle from 0.5° to 0.25° and the number of height nodes from 44 to 39 and 20, and recomputes the comparison with
  MYSTIC for the new grid in solar zenith angle; and *Error of the band approximation*, which compares the 45-band tables
  with `point` tables of 4, 2, 1 and 0.5 nm spacing in one bundled state.
- *Convergence of the path quadrature and of the Legendre order* adds the comparison of order 24 with order 16 for the
  eight bundled states of the MYSTIC comparison. The statement that the cause of the difference from MYSTIC at SZA 93°
  was not found is replaced by what the new grid measurements explain, and *Open checks* is updated, with a new item on
  the height grid in deep twilight.
- README *Validation* adds that the grid of 0.25° in solar zenith angle removes part of the difference from MYSTIC.

## 1.3.3

1.3.3 changes documentation only; the code, the assets and the tables are those of 1.3.2.

- `docs/VALIDATION.md` adds seven sections: eight bundled states against MYSTIC at six pixels and solar zenith angles
  93° and 96°; the convergence of the path quadrature and of the Legendre order; the IPRT spherical test case E1, whose
  limits the default quadrature does not meet on the layered profile of the case; second-order scattering against a
  direct quadrature at solar elevations −6° to −18°; a Rayleigh atmosphere against SASKTRAN2 at −9° to −18°; measured
  twilight brightness (Koomen et al. 1952; Patat et al. 2006); and the timing of the purple light (Lee and
  Hernández-Andrés 2003). The introduction and *Open checks* follow.
- README *Validation* summarises the comparison of the bundled states and no longer says that their stratospheric layer
  is not compared with MYSTIC.
- README *Limitations* corrects the statement on measured twilight brightness. The range of 1.4 to 2.1 times holds for
  the Maryland and Paranal measurements, and Sacramento Peak gives 2.9 to 3.2 times. The two states that reproduce the
  purple light differ in the density profile and in the stratospheric optical depth and share the ozone profile and the
  Rayleigh column. The clause on the ratio growing as the Sun goes down is removed.
- `docs/REFERENCES.md` adds Ramon et al. (2019).

## 1.3.2

1.3.2 fixes the preview images and the metadata of the legacy batch path and leaves the kernel, the solver and
`table.npz` as they are in 1.3.1. `run_gpu_state_v3`, `host/pack_state` and `tables_from_sources` are hashed production
sources, so provenance hashes differ from 1.3.1, and resuming an output saved by 1.3.1 stops with `E_OUTPUT_IDENTITY`
(use a fresh work root).

- The preview images of tables off the 15-band grid weight each band by its width from `bin_edges_nm`. They scaled
  every band by 15/len(wl), so on the 45-band product grid, which splits ten of its 11.75 nm bands into 5.875 nm halves,
  the narrow bands counted 1.78 times and the wide ones 0.89 times their share and G came out about 30 % high. Against
  a direct CIE 1931 integral of a product table, the preview chromaticity moves from 0.044–0.047 to 0.004–0.005 in xy;
  the change between the old and new previews is ΔE76 13–16 (median). Tables without a spectral block keep the old
  weights.
- `tables_from_sources` and `table_fast` record the computed stratosphere for sources on the legacy wavelength grid
  (no spectral block, the legacy wavelengths), which they rebuild with the legacy solver. They wrote
  `unsupported_parameters: ["stratosphere_tau550"]` with the requested `stratosphere.status` and no
  `parameters_computed` or `stratosphere_computed`.
- The `--geometry-only` prefetch attaches the stratospheric geometry through the same private overlay as a worker, so a
  prefetch and a worker of one key no longer write the same `<key>.tmp.npz` in a shared `STRATO_ATTACH_CACHE`. The
  prefetch writes no `attach_cache` PHASE record; the worker's record then reads as a hit.
- `test_pipeline_persistent` waits 2400 s by default (`PERSISTENT_TEST_DEADLINE`, was 60 s), the value the public CI
  already set.
- README *Limitations* states that the default exp(−h / 8 km) density row is 1.4 to 2.1 times too bright against
  measured twilight between −6° and −12° and how to select the US Standard Atmosphere rows. The default is unchanged.

## 1.3.1

1.3.1 changes the planner and the batch writer and leaves the kernel and the solver as they are in 1.3.0. `plan_queue`
and `tables_from_sources` are among the hashed production sources, so provenance hashes differ from 1.3.0, and resuming
an output saved by 1.3.0 stops with `E_OUTPUT_IDENTITY` (use a fresh work root).

- `plan_queue` counts every GPU against the CPU budget: GPUs x workers per GPU x column workers above
  `pipeline.cpu_budget()` now stops with `E_PLAN_CPU_BUDGET`, where only workers x column workers was compared. A plan
  that asked for more than the host has, for example two GPUs with four workers and two column workers on eight cores,
  is now refused.
- The plan lists the `ground_aerosol01` host cache with the key the producer builds and an unmeasured cost. Its extra
  unmeasured item can change which state represents a group and the order and GPU assignment of the states.
- `tables_from_sources` takes `<out>/.worker.lock` for each state and looks again for a finished table and pack once it
  holds the lock. A batch and a worker can no longer write the same output; the one that comes second skips, or stops
  with `E_OUTPUT_BUSY`.
- Tests: the joint-walk case is pinned to `SOSFUSE2_COLUMN_METHOD=gl24` and is no longer a known failure (the default
  table method differs from the 24-point walk by up to 7.5e-4 on rays that start on a shell and run horizontally, and is
  within 3.3e-5 of the exact method there), so the expected outcome is 3 `xfailed`. `test_tropo_profile` keeps
  only `--host`: `--static` and `--paths` are removed (the kernel text and the three-wavelength grid they relied on are
  gone), and it stops with `E_NO_CHECK` when no check is requested. The host-built check of the two-layer aerosol branch
  of `point()` that `--static` carried has no replacement. The persistent-service test skips with
  `E_SOCKET_PATH_TOO_LONG` when `TMPDIR` is too long for its socket.
- Tests without the IPRT inputs: a test session used to refuse the download in every case, so a fresh installation saw
  the tests that need the `usstd76` density asset fail with `E_IPRT_OFFLINE`. The session now fetches the two files
  first unless `SOSFUSE2_OFFLINE` is set, and when they cannot be had those tests skip with that reason.

## 1.3.0

1.3.0 reorganises the CPU reference engine and the kernel source and keeps the physics of 1.2.3. After the upgrade the
sun-table, column, geometry and host caches are rebuilt once on first use, and provenance hashes differ from 1.2.3: the
hashed source files changed and the sun-table identity is now version 3. The code generated from `kernel.cu` differs
from 1.2.3 in one place. Under `SOSFUSE2_FASTMATH` the two `expf` calls of the two-layer aerosol profile now use the
intrinsic, as the rest of the kernel already did, so tables of two-layer aerosol states built with that flag change in
the last bits.

The CPU solver files (`solver_*.py`) are now ordinary imported modules, so an edit to one takes effect only after the
process restarts. `accepted.engine.create()` still returns `(solver, context, settings)`, and the second value is an
`EngineContext` where it used to be a scope dict: read its attributes, such as `context.b` and `context.Solver`.
Indexing it by one of its ten former scope-dict keys still works in 1.3.0 and raises a `DeprecationWarning`. `engine.definitions`
and the `HOST_ORIGINAL_GROUND` key are gone, code that reached into the solver functions through `__globals__`,
`__code__` or `__closure__` no longer works, and `column_table.aerosol` requires an explicit profile descriptor
(`E_COLUMN_AER_DESCRIPTOR`).

`pipeline --threads 0` now stops with `E_PIPELINE_CPU_BUDGET`, and a non-positive `table_fast` batch stops with
`E_TABLE_BATCH`. When the central region of a preview is black, its exposure is taken from the lit pixels of the whole
image.

## 1.2.3

1.2.3 enables the background stratospheric layer in all 64 bundled `states_v2.json` states. The physics code, sampled
optical depths and 21 km layer centre are unchanged; `states_v1.json` retains its previous default. Runs using the
default v2 states now include the layer when `--stratosphere` is omitted, so default tables of 1.2.3 differ from those
of 1.2.2.

The changed `states_v2.json` asset hash changes every new table's provenance signature, even for runs that use another
state file, because the bundled asset is always hashed. Existing tables remain readable; signatures and saved-source
reuse must still satisfy the provenance checks described in `docs/PROVENANCE_V3.md`. To reproduce the earlier output,
copy the state file, set each desired state's `stratosphere.status` to `disabled`, and pass that copy via `--states`;
the CLI has no `--stratosphere off` option.

## 1.2.2

1.2.2 added the L4 stall measurement of the release kernel to `docs/PERFORMANCE.md` and changed nothing else.

## 1.2.1

1.2.1 added `docs/PERFORMANCE.md`.

## 1.2.0

1.2.0 replaced the source-text adapters of the engine loader with keyword-only hooks, moved the packaged tests into
`sosfuse2/tests`, reformatted the Python sources to one statement per line and rewrote the documents. Up to 1.2.2 the
physics is that of 1.1.0: the tables carry new `code_hashes` since 1.2.0, and a comparison of 1.2.0 with 1.1.1 found no
difference in the prepared arrays or the final tables. That comparison does not extend to 1.2.3. It is recorded under
*Tested configurations* in the README.
