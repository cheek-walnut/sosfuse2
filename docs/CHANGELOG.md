# Changelog

Release changes from 1.2.0 on, newest first. Versions 1.0.0 to 1.2.3 were published together, and earlier entries are
not reconstructed here. The 1.1.0 absorber changes (NO₂ and O₄) are described in `docs/VALIDATION.md`.

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
