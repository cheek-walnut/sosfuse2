# Architecture

This document describes how the package is organised and where to start reading. Paths are relative to `src/sosfuse2/`
unless they start with `docs/`. `python -m sosfuse2 walkthrough` runs the smallest complete example, and `AGENTS.md`
lists, for each kind of task, the two to four files to open first. `tests/test_docs_pointers.py` checks that the files
and functions named in `AGENTS.md` and in this document exist.

## 1. One input state and one output table

A *state* is one set of atmosphere parameters: an entry of `assets/states_v2.json` (or `assets/states_v1.json`) with
the aerosol, ozone, ground and stratosphere values of one sky. *Settings* (`jobs/run_state.py::settings_for`) fix the
height and angle grids, the wavelengths and the solar elevations. The state, the settings and the code that ran are
signed together into the `meta.json` next to the table (`docs/PROVENANCE_V3.md`).

The output is one table per state, `table.npz`. Its array `radiance` has the axes (solar elevation, view elevation,
relative azimuth, wavelength), as `docs/MODEL.md` describes. Everything between the state and the table is the solution
of a source field on a grid of heights and angles: iterate it to convergence once, then read every table row from it.

For a first pass, start with the three functions in section 4, then read from `foundation` through the layer you need;
each layer depends only on earlier layers in the dependency order, apart from the function-level upward imports
listed in section 6.

## 2. The ten layers and the direction of imports

A module may import only from its own layer and from the layers listed in the `May import from` column. The test
`tests/test_layers.py` reads every import, including those inside functions, and fails on a new upward import.
`assets/` (data) and `tests/` are not layers.

| Layer | Role | Modules | May import from |
|---|---|---|---|
| `foundation` | where resources are, what identifies a file's content, machine limits, time | `paths`, `content_identity`, `host_limits`, `available_cpus`, `timers` | nothing |
| `atmosphere` | turn a state and a wavelength into densities and optical coefficients | `components`, `density`, `spectral`, `ozone_temperature`, `minor_absorbers`, `rayleigh`, `air_wavelength`, `tropo_profile`, `tropo_mie`, `stratosphere`, `extinction` | foundation |
| `rays` | spherical ray segments, quadrature points for incoming directions and column amounts | `shell_segments`, `incoming_quadrature`, `incoming_rings`, `zenith_interpolation`, `ray_columns`, `column_table` | foundation, atmosphere |
| `precompute` | reusable CPU-prepared arrays and their caches | `host_cache`, `column_cache`, `column_parallel`, `sun_parallel`, `ground_parallel`, `endpoint_contract` | foundation through rays |
| `backends` | one transport evaluation (NumPy, CUDA, native), arithmetic, spherical-harmonic (SH) layout, compile rules | `numpy_reference`, `arithmetic`, `sh_layout`, `flags`, `flags_env`, `device_profile`, `compile_cache`, `kernel.cu`, `native/`, `configs/` | foundation through precompute |
| `solve` | build the per-state solver objects and iterate the source field | `engine`, `engine_core`, `ray_geometry`, `ray_geometry_cpu`, `ray_geometry_parallel`, `source_solver`, `ksplit`, `ksplit_tolerance`, `ground_sh`, `strato_sh_aux` | foundation through backends |
| `outputs` | turn the converged field into a table, previews and a pack | `radiance_table`, `quantized_table`, `pack_state`, `preview_functions` | foundation through solve |
| `jobs` | run one state or many, with caches, queues and records | `run_state`, `pipeline`, `pipeline_execution`, `plan_queue`, `tables_from_sources`, `precompile`, `cache_store`, `walkthrough` | foundation through outputs |
| `checks` | check calculations, records and test inputs | `gate`, `validate`, `validation_evidence`, `numerical_checks`, `solver_diagnostics`, `kernel_diagnostics`, `stratosphere_checks`, `testkit`, `gate_data/`, `fixtures/` | foundation through jobs |
| `asset_tools` | build data assets from source data | `generate_density_profile`, `generate_minor_absorbers`, `generate_strato_optics`, `generate_tropo_optics`, `mie`, `iprt`, `prepare_minor_absorbers`, `strato_states` | all of them |

Each folder groups modules by their main duty, though some modules combine two roles and some functions are long. For
example, `solve/source_solver.py` defines both `Kernel` for CUDA launch planning and `Solver` for the iteration, while
`jobs/run_state.py` combines the settings rules with the runner.

## 3. The walkthrough

```sh
python -m sosfuse2 walkthrough            # run the example and print the steps
python -m sosfuse2 walkthrough --write    # also write docs/WALKTHROUGH.md (in the current directory)
```

`jobs/walkthrough.py` runs a small reduced state with the NumPy backend, in an empty temporary work root and without
network access, through the three real calls of section 4. At each call it prints the input, the array shapes, the number
of evaluations and the shape and finite values of the table. The steps between the calls (density, sun path, optical
coefficients, cells, segments, columns, first scattering, transport) are a fixed table in the generated
`docs/WALKTHROUGH.md`, with the file and function of each step; the document separates what the run observed from what
that table explains. It also lists the production paths the example does not run. A failed run leaves an existing
document as it was.

## 4. The call spine

```
state + settings
  -> jobs/run_state.py::create_configured     build the Solver for one state
  -> solve/source_solver.py::Solver.solve     iterate the source field to convergence
  -> outputs/radiance_table.py::make_table    read the table rows from the converged field
```

`create_configured` binds the state to the density and optical coefficients in `atmosphere/density.py` and
`atmosphere/extinction.py`, the ray and cell geometry in `solve/ray_geometry.py`, and `solve/source_solver.py::Kernel`.
In each iteration, `Solver.solve` calls `Solver.apply` for ground reflection and one scattering update; the first call
runs `Solver.initialize` for first scattering. The transport inside those
calls runs in `backends/numpy_reference.py::transport` on the NumPy path and in `backends/kernel.cu`, launched through
`Kernel`, on the CUDA path. `jobs/run_state.py::run` wraps the three calls with the cache leases, the phase records, the
signature and the output files; `jobs/pipeline.py` runs many such states.

## 5. Topics that cross layers

| Topic | Where it lives |
|---|---|
| Ray geometry | segments and grids in `rays/shell_segments.py`, columns in `rays/ray_columns.py`, endpoint arrays in `solve/ray_geometry.py`, the CPU build in `solve/ray_geometry_cpu.py`, the process pool in `solve/ray_geometry_parallel.py`, and shared build rules in `precompute/endpoint_contract.py` |
| Stratosphere | optical layer in `atmosphere/stratosphere.py`, single-precision sidecar in `solve/strato_sh_aux.py`, checks in `checks/stratosphere_checks.py`, and assets in `asset_tools/generate_strato_optics.py` and `asset_tools/strato_states.py` |
| Caches | prepared arrays in `precompute/host_cache.py`, column arrays in `precompute/column_cache.py`, compiled kernels in `backends/compile_cache.py`, and the shared bake store in `jobs/cache_store.py` |
| Code and data identity | `foundation/content_identity.py::package_hashes`, `foundation/paths.py::PRODUCTION_MODULES`, `checks/gate.py::code_identity` |
| Split transport (KSPLIT) | host side in `solve/ksplit.py`, tolerance in `solve/ksplit_tolerance.py`, and device code in `backends/kernel.cu` |
| Machine limits and parallel sizes | `foundation/host_limits.py`, `foundation/available_cpus.py`, `backends/device_profile.py`, `jobs/pipeline.py::cpu_budget` |
| Diagnostics outside the gate | `checks/solver_diagnostics.py::probe_state`, `checks/kernel_diagnostics.py`, `checks/numerical_checks.py` |

Section 8 lists the files whose paths enter the cache keys, and `backends/compile_cache.py::cache_key` and
`jobs/cache_store.py` describe their own keys.

Four connections are hard to find by name:

- The `Kernel` class, which handles launch planning and device buffers, lives alongside `Solver` in
  `solve/source_solver.py` rather than in `backends/`.
- The preview functions are read as source text. `outputs/pack_state.py::render_functions` parses
  `outputs/preview_functions.py` with `ast` and runs nine named functions in a prepared scope.
- Importing `solve/ray_geometry_parallel.py` replaces `PackedCache` with its parallel version. `solve/ray_geometry.py`
  imports `solve/ray_geometry_parallel.py` as its last statement, a deliberate cycle inside one layer, and
  `tests/test_structure_imports.py` imports them in both orders with import warnings turned into errors.
- `solve/source_solver.py::load_engine` loads the CPU engine module as a private copy.

## 6. Known upward imports

Twelve module pairs import from a higher layer inside a function across twenty places, with all module-level imports
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
| `solve/source_solver.py` | `outputs/radiance_table.py` | `Solver.boxes_fixed` | fixed-ray observation |
| `solve/source_solver.py` | `jobs/pipeline_execution.py` | `Solver.initialize` | prepare GPU bundles |
| `outputs/radiance_table.py` | `jobs/run_state.py` | `load_source`, `restored_settings`, `restored_table`, `main` | restore the settings of a saved source |
| `checks/testkit.py` | `asset_tools/generate_minor_absorbers.py` | `synthetic_o4` | grid constants of the O₄ generator |

To add one, write the function-level import and add the row with its reason to `tests/test_layers.py` and to the table
in section 6.

## 7. Commands and settings

Running `python -m sosfuse2` prints the command table from `__main__.py`, which maps each command name to a module and
replaces its own process with that module (`os.execv`) without importing calculation code first, so unknown commands
fail immediately. For example, `python -m sosfuse2 run_state ...` runs `python -B -m sosfuse2.jobs.run_state ...` with
the same arguments and exit code, and the direct `python -B -m sosfuse2.<layer>.<module>` form also works. The table
includes `walkthrough` and marks `pipeline_execution`, the child process of `pipeline`, as internal.

A run takes inputs from three places: command-line options defined by the target module (`--help`), the signed state and
settings files, and environment variables for devices, caches and worker counts, which `docs/ENVIRONMENT.md` lists with
their layer, default and precedence.
`jobs/run_state.py::kernel_settings` copies the kernel variables into the signed settings, so the queue, the prefetch and
the runner of one state read the same values.

## 8. Moving or adding a file

1. Add a production module to `foundation/paths.py::PRODUCTION_MODULES`. Because this tuple lists files explicitly
   rather than by glob, a missing listed file stops the run with `E_PACKAGE_FILE_MISSING`.
   `tests/test_ozone_temperature.py::test_03c_v4_fixed_provenance_lists` keeps the list equal to the tree.
2. Fix every list that names files by relative path. Most are edited by hand, each for its own cache or signature input;
   `foundation/content_identity.py::package_hashes` reads `PRODUCTION_MODULES`, and `checks/gate.py::code_identity`
   reads `package_hashes()`. The lists are `package_hashes` (with `ACCEPTED_SOURCES`, `HOST_MODULES` and `DATA_ASSETS`),
   `solve/engine.py::SOURCE_FILES`, `solve/engine.py::fingerprint`, `solve/engine.py::cell_folder`,
   `precompute/endpoint_contract.py::UPSTREAM`, `precompute/column_cache.py::cache_key`,
   `precompute/host_cache.py::sun_geometry`, `precompute/host_cache.py::context`,
   `precompute/host_cache.py::aerosol_identity`, `precompute/column_parallel.py::source_hashes` (a check that every
   worker runs the same sources rather than a cache key), `rays/shell_segments.py::SunGeometry.identity`,
   `rays/column_table.py::source_hash`, `atmosphere/stratosphere.py::_geometry_key`, the query key in
   `jobs/pipeline_execution.py` and `checks/gate.py::code_identity`. If a listed path no longer exists, hashing it
   raises an error.
3. Locate data through the constants in `foundation/paths.py` (`ASSETS`, `NATIVE`, `CONFIGS`, `GATE`, `FIXTURES`,
   `KERNEL_CU`) rather than relative to the calling module. `backends/kernel.cu` stays in the folder of `backends/compile_cache.py`
   and `backends/sh_layout.py`, which hash and render it.
4. Child processes start as `python -B -m sosfuse2.<layer>.<module>` through `foundation/paths.py::module_command`; the
   module names in `jobs/pipeline.py`, `jobs/plan_queue.py` and `checks/gate.py` change with a rename.
5. Recorded names stay fixed across file moves, including environment variable names, phase event names, provenance
   format identifiers, result field names, and the `module` string that `asset_tools/generate_minor_absorbers.py`
   writes into the O₄ asset as part of that asset's physical identity.
6. Run the structure tests `tests/test_layers.py`, `tests/test_names.py`, `tests/test_entry_points.py`,
   `tests/test_structure_imports.py` and `tests/test_docs_pointers.py`, together with the identity tests
   `tests/test_identity.py`, `tests/test_identity_caches.py`, `tests/test_identity_runner.py` and
   `tests/test_identity_gate.py`.
