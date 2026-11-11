# Changelog

Release changes from 1.2.0 on, newest first. Versions 1.0.0 to 1.2.3 were published together, and earlier entries are
not reconstructed here. The 1.1.0 absorber changes (NO₂ and O₄) are described in `docs/VALIDATION.md`.

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
