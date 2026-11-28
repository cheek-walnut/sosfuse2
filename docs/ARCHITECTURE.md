# Architecture

This document describes how the package is organised and where to start reading. Paths are relative to `src/sosfuse2/`
unless they start with `docs/`. `python -m sosfuse2 walkthrough` runs the smallest complete example. The
[task table](../AGENTS.md#where-to-start) lists the files to open and tests to run for each change. `tests/test_docs_pointers.py` checks that the files
and functions named in `AGENTS.md` and in this document exist.

## 1. One input state and one output table

A *state* is a set of atmosphere parameters: an entry in `assets/states_v2.json` (or `assets/states_v1.json`) specifying
the aerosol, ozone, ground and stratosphere values of one sky. *Settings* (`solve/configuration.py::settings_for`) fix
the height and angle grids, wavelengths and solar elevations, while the runner sets solar elevation separately via
`solar_elevation_deg`, outside state definitions. The state, settings and code that ran are signed together into
`meta.json` next to the table (`docs/PROVENANCE_V3.md`).

The output is one table per state, `table.npz`, whose `radiance` array has the axes (solar elevation, view elevation,
relative azimuth, wavelength), as `docs/MODEL.md` describes. Between the state and the table, the solver iterates a
source field on a grid of heights and angles to convergence once and then reads every table row from that field.

For a first pass, start with the three functions in section 4, then read from `foundation` through the layer you need.
Apart from the function-level upward imports listed in section 6, each layer depends only on earlier layers in the
dependency order.

### Angle notation

Table rows use solar elevation in degrees, negative when the Sun is below the horizon. Grids and comparisons measured
from the zenith use solar zenith angle (SZA), where SZA = 90° − solar elevation. A sunward view looks towards the
Sun's azimuth (0° relative azimuth), and an antisolar view looks towards the opposite azimuth (180°). View elevation
measures the line of sight above the horizon.

## 2. The ten layers and the direction of imports

A module may import only from its own layer and from the layers listed in the `May import from` column. The test
`tests/test_layers.py` reads every import, including those inside functions, and fails on a new upward import.
`assets/` (data) and `tests/` are not layers.

| Layer | Role | Modules | May import from |
|---|---|---|---|
| `foundation` | where resources are, what identifies a file's content, machine limits, time | `paths`, `dependencies`, `content_identity`, `host_limits`, `available_cpus`, `timers`, `top`, `version` | nothing |
| `atmosphere` | turn a state and a wavelength into densities and optical coefficients | `components`, `density`, `upper_air`, `spectral`, `ozone_temperature`, `minor_absorbers`, `rayleigh`, `air_wavelength`, `tropo_profile`, `tropo_mie`, `stratosphere`, `extinction` | foundation |
| `rays` | spherical ray segments, quadrature points for incoming directions and column amounts | `shell_segments`, `incoming_quadrature`, `incoming_rings`, `zenith_interpolation`, `ray_columns`, `column_table` | foundation, atmosphere |
| `precompute` | reusable CPU-prepared arrays and their caches | `host_cache`, `column_cache`, `column_parallel`, `sun_parallel`, `ground_parallel`, `endpoint_contract` | foundation through rays |
| `backends` | transport and observation queries, arithmetic, spherical-harmonic (SH) layout, compile rules | `numpy_reference`, `arithmetic`, `sh_layout`, `flags`, `flags_env`, `device_profile`, `compile_cache`, `vram_budget`, `vram_trace`, `mlx_backend`, `mlx_ops`, `mlx_transport`, `mlx_warm`, `mlx_warm_kernels`, `mlx_query`, `mlx_query_kernel`, `kernel.cu`, `native/`, `configs/` | foundation through precompute |
| `solve` | build the per-state solver objects and iterate the source field | `configuration`, `build`, `engine`, `engine_core`, `ray_geometry`, `ray_geometry_cpu`, `ray_geometry_parallel`, `source_solver`, `source_iteration`, `ksplit`, `ksplit_tolerance`, `ground_sh`, `strato_sh_aux` | foundation through backends |
| `outputs` | turn the converged field into a table, previews and a pack | `radiance_table`, `quantized_table`, `pack_state`, `preview_functions` | foundation through solve |
| `jobs` | run one state or many, with caches, queues and records | `run_state`, `run_storage`, `pipeline`, `pipeline_execution`, `plan_queue`, `tables_from_sources`, `precompile`, `cache_store`, `walkthrough` | foundation through outputs |
| `checks` | check calculations, records and test inputs | `gate`, `validate`, `validation_evidence`, `numerical_checks`, `solver_diagnostics`, `kernel_diagnostics`, `stratosphere_checks`, `testkit`, `gate_data/`, `fixtures/` | foundation through jobs |
| `asset_tools` | build data assets from source data | `generate_density_profile`, `generate_upper_air`, `generate_minor_absorbers`, `generate_strato_optics`, `generate_tropo_optics`, `mie`, `iprt`, `prepare_minor_absorbers`, `strato_states` | all of them |

Each folder groups modules by their main duty, though some modules combine two roles and some functions are long. For
example, `solve/source_solver.py` defines `Kernel` for CUDA launch planning, and `solve/source_iteration.py` defines
`Solver` for the iteration.

Each runner mode calls `solve/configuration.py::state_settings` to select the effective state and resolve its settings,
using input assets only. `jobs/run_state.py` handles device admission, signing, leases and monitoring, and passes host
arrays, coordinates and measured records to `jobs/run_storage.py` for publication under the output lock. Derived tables
hold a shared lock on the source job through verification.

## 3. The walkthrough

```sh
python -m sosfuse2 walkthrough            # run the example and print the steps
python -m sosfuse2 walkthrough --write    # also write docs/WALKTHROUGH.md (in the current directory)
```

`jobs/walkthrough.py` runs a small reduced state with the NumPy backend in an empty temporary work root, without network
access, through the three real calls of section 4. At each call it prints the input, array shapes, number of evaluations,
and the shape and finite values of the table.

The intermediate steps (density, sun path, optical coefficients, cells, segments, columns, first scattering, transport)
form a fixed table in the generated `docs/WALKTHROUGH.md`, giving the file and function of each step. The document
separates values observed during the run from what that table explains and lists the production paths the example does
not exercise. A failed run leaves an existing document unchanged.

## 4. The call spine

```
state + settings
  -> solve/build.py::create_configured     build the Solver for one state
  -> solve/source_iteration.py::Solver.solve     iterate the source field to convergence
  -> outputs/radiance_table.py::make_table    read the table rows from the converged field
```

`create_configured` binds the state to the density and optical coefficients in `atmosphere/density.py` and
`atmosphere/extinction.py`, the ray and cell geometry in `solve/ray_geometry.py`, and `solve/source_solver.py::Kernel`.
In each iteration, `Solver.solve` calls `Solver.apply` for ground reflection and one scattering update; the first call
runs `Solver.initialize` for first scattering. Transport inside those calls runs in `backends/numpy_reference.py::transport`
on the NumPy path and in `backends/kernel.cu`, launched through `Kernel`, on the CUDA path.

On the NumPy path, `solve/ray_geometry.py::TransientGeometry.evaluate` can dispatch transport to
`backends/mlx_backend.py::evaluate`, while observer and table rays use `backends/mlx_backend.py::query`; see
[MLX coverage](ENVIRONMENT.md#experimental-mlx-transport-apple-silicon) for the selection rules. `jobs/run_state.py::run`
coordinates the solve and output, and `jobs/pipeline.py` runs multiple states.

CPU solver functions live in `rays/shell_segments.py`, `rays/incoming_quadrature.py`, `rays/incoming_rings.py`,
`rays/zenith_interpolation.py`, `atmosphere/extinction.py`, and `solve/engine_core.py`. `solve/engine.py` builds a
per-state `EngineContext` binding only the inputs each function reads, and `create()` returns `(solver, context, settings)`.
Read context attributes such as `context.b` and `context.Solver`; read-only indexing by the former scope-dict keys is
deprecated.

Separating optical coefficients from column and geometry producers keeps coefficient edits from invalidating geometry
caches. `solve/build.py::create_configured` and `atmosphere/tropo_profile.py::engine_hooks` supply the production grid,
spectrum, sun geometry, deferred ground preparation, and two-layer aerosol profile to `create()` via keyword-only hooks
without rewriting source text.

## 5. Topics that cross layers

| Topic | Where it lives |
|---|---|
| Ray geometry | segments and grids in `rays/shell_segments.py`, columns in `rays/ray_columns.py`, endpoint arrays in `solve/ray_geometry.py`, the CPU build in `solve/ray_geometry_cpu.py`, the process pool in `solve/ray_geometry_parallel.py`, and shared build rules in `precompute/endpoint_contract.py` |
| Stratosphere | optical layer in `atmosphere/stratosphere.py`, single-precision sidecar in `solve/strato_sh_aux.py`, checks in `checks/stratosphere_checks.py`, and assets in `asset_tools/generate_strato_optics.py` and `asset_tools/strato_states.py` |
| Caches | prepared arrays in `precompute/host_cache.py`, column arrays in `precompute/column_cache.py`, profile tables in `rays/column_table.py`, compiled kernels in `backends/compile_cache.py`, and the shared bake store in `jobs/cache_store.py` |
| Code and data identity | registered inputs in `foundation/dependencies.py::FILES`, output hashes in `foundation/content_identity.py::package_hashes`, baseline identity in `checks/gate.py::code_identity` |
| Split transport (KSPLIT) | host side in `solve/ksplit.py`, tolerance in `solve/ksplit_tolerance.py`, record layout in `atmosphere/components.py` and `backends/kernel.cu` |
| Device memory | named estimates in `backends/vram_budget.py`, CuPy allocation diagnostics in `backends/vram_trace.py`, and persistent admission in `jobs/pipeline_execution.py` |
| Machine limits and parallel sizes | `foundation/host_limits.py`, `foundation/available_cpus.py`, `backends/device_profile.py`, `jobs/pipeline.py::cpu_budget` |
| Additional diagnostics | `checks/solver_diagnostics.py::probe_state`, `checks/kernel_diagnostics.py`, `checks/numerical_checks.py` |

Section 8 explains how to register the files that enter cache keys; `backends/compile_cache.py::cache_key` and
`jobs/cache_store.py` define their own keys.

The process-local memo in `rays/column_table.py::tables` holds up to nine profiles, each with a ground and a tangent
table, covering the four legacy profiles, four corrected profiles and one stratospheric profile that a cell can use.

Four connections are hard to find by name:

- `solve/build.py::create_configured` passes a kernel factory to `solve/engine.py::create`. The factory constructs
  `solve/source_solver.py::Kernel` with the selected backend, block size and state settings, and its `host` method uses
  that instance's array module for transfers. Code that runs on either backend passes `device=xp is not np` to
  `foundation/timers.py::stage`; each timed CUDA interval retains its event provider until the report is read.
- Both the packer and the state runner call `outputs/pack_state.py::render_functions`. The packer imports
  `outputs/preview_functions.py` directly and binds each renderer's luminance weights. Set product PNG compression in
  `outputs/pack_state.py::PREVIEW_PNG_LEVEL`.
- `solve/ray_geometry.py::ResidentGeometry.packed` constructs a `PackedCache` with the current worker setting, while
  `PackedCache(folder, workers=n)` selects a count explicitly. Its `build` method passes the serial writer and saved
  setting to `solve/ray_geometry_parallel.py::build_cells`, where the parent writes cells in order while spawned
  processes compute them from the geometry supplied to their initializer. When a build falls back to the serial writer
  because its callback is unsupported or because it has fewer than two cells despite requesting multiple workers, it
  emits a `PHASE` event with `phase=geometry_parallel`, `result=fallback`, and `reason` set to
  `callback is not ResidentGeometry.cached_cell` or `fewer than two cells`, respectively. With a supported callback and
  one worker, the build goes straight to the serial writer.
- `solve/source_solver.py::load_engine` loads the CPU engine module as a private copy.

## 6. Known upward imports

Twelve module pairs import from a higher layer inside a function across twenty-one places, with all module-level imports
following the layer order. `tests/test_layers.py` lists each pair with its function and reason and fails when the code
and the list differ in either direction, without comparing the number of occurrences.

| Importing module | Imported module | Function | Reason |
|---|---|---|---|
| `atmosphere/density.py` | `rays/column_table.py` | `column_identity`, `exact_profile_columns` | exact column integrals and their identity |
| `atmosphere/density.py` | `rays/ray_columns.py` | `attach` | sun-column correction |
| `atmosphere/density.py` | `precompute/host_cache.py` | `attach` | prepared-array cache |
| `atmosphere/density.py` | `asset_tools/generate_density_profile.py` | `asset_path` | build a missing asset |
| `atmosphere/stratosphere.py` | `precompute/ground_parallel.py` | `_ground_geometry` | ground preparation |
| `atmosphere/stratosphere.py` | `backends/sh_layout.py` | `attach`, `source_size`, `eigen_by_sh` | SH layout sizes |
| `atmosphere/tropo_mie.py` | `backends/sh_layout.py` | `eigen_by_sh` | SH layout sizes |
| `rays/column_table.py` | `precompute/host_cache.py` | `tables` | store the profile tables |
| `solve/source_iteration.py` | `outputs/radiance_table.py` | `Solver.boxes_fixed` | fixed-ray observation |
| `solve/source_iteration.py` | `jobs/pipeline_execution.py` | `Solver.initialize` | prepare GPU bundles |
| `outputs/radiance_table.py` | `jobs/run_state.py` | `load_source`, `check_source_tol`, `restored_settings`, `restored_table`, `main` | restore the settings of a saved source |
| `checks/testkit.py` | `asset_tools/generate_minor_absorbers.py` | `synthetic_o4` | grid constants of the O₄ generator |

To add one, write the function-level import and add the row with its reason to `tests/test_layers.py` and to the table
in section 6.

## 7. Commands and settings

Running `python -m sosfuse2` prints the command table from `__main__.py`, which maps each command name to a module and
replaces its own process with that module (`os.execv`) without importing calculation code first, so unknown commands
fail immediately. For example, `python -m sosfuse2 run_state ...` runs `python -B -m sosfuse2.jobs.run_state ...` with
the same arguments and exit code; the direct `python -B -m sosfuse2.<layer>.<module>` form also works. The table
includes `walkthrough` and marks `pipeline_execution`, the child process of `pipeline`, as internal.

A run takes inputs from three sources: command-line options defined by the target module (`--help`), signed state and
settings files, and environment variables for devices, caches and worker counts (`docs/ENVIRONMENT.md` lists their
layer, default and precedence). `solve/configuration.py::kernel_settings` copies the kernel variables into the signed
settings so that the queue, prefetch and runner of one state read the same values.

## 8. Moving or adding a file

Register each production file once in `foundation/dependencies.py::FILES`. A Python module with an empty row
(`'solve/example.py': {}`) enters package provenance and the gate code identity. If its contents also affect a cache,
add its scope to the row: ordered scopes take the next position, and the path-sorted shared sun table takes
`shared_sun=True`. Each row's fields determine which caches depend on that file; for example, `engine_source` includes
the file in both the engine fingerprint and its cell key. After adding or moving a file, run this check from the
repository root:

```sh
python -m pytest -q src/sosfuse2/tests/test_ozone_temperature.py::test_03c_v4_fixed_provenance_lists src/sosfuse2/tests/test_identity_caches.py
```

When moving a file, rename the path wherever it appears in the fixed lists in
`tests/test_identity_caches.py::test_registered_source_order`, keeping its position for each consumer and its column
slots. Update imports and child-process module names passed to `foundation/paths.py::module_command` in
`jobs/pipeline.py`, `jobs/plan_queue.py` and `checks/gate.py`. Locate package resources through `foundation/paths.py`
and keep `backends/kernel.cu` beside its renderer in `backends/sh_layout.py`.

Preserve recorded names, including environment variables, phase events, provenance formats, result fields and the O₄
asset's recorded generator module. Verify moves with `tests/test_layers.py`, `tests/test_names.py`,
`tests/test_entry_points.py`, `tests/test_structure_imports.py` and `tests/test_docs_pointers.py`.
