# Provenance records: v3 → v4 → v5

The `meta.json` of a table written by `jobs/run_state.py` or `jobs/tables_from_sources.py` carries `contract`,
`signature` and `code_hashes`. A table written by the `outputs/radiance_table` command on its own carries `contract`
without the other two, and merged tables are described under *Reader rule*. Legacy upstream tables
use v3, tables baked from the `sosfuse2` package up to 1.3.4 are v4, and tables baked from 1.4.0 on are v5
(`jobs/run_state.py` `CONTRACT_VERSION`). This note is for readers that hold these provenance formats.

## Reader rule

A reader accepts signed tables in provenance formats v3, v4 and v5 (`atmosphere/spectral.py`, the list next to
`E_PROVENANCE_CONTRACT`).

What the reader checks is narrower than the record: `spectral.check_meta_contract` requires a known `contract` on a
signed table and agreement of the physics, optics-state and spectral-block metadata. It does not recompute the
signature, does not compare `code_hashes` with the installed package, and does not read the arrays (`read_table` takes
no metadata). The comparisons of `code_hashes` are made where a job continues or reuses earlier work
(`E_OUTPUT_IDENTITY`, `E_EXTRA_SOURCE`, *Signature inputs*), and the rules below say how the keys of each format read.

- When `contract == 'v3'`, `code_hashes` has the repository-relative keys in *Keys*. Such tables are read in place;
  they are not re-baked or re-signed to v4 or v5.
- When `contract == 'v4'`, `code_hashes` uses package-relative keys, so a comparison goes by logical key rather
  than by file location.
- When `contract == 'v5'` (`jobs/run_state.py` `CONTRACT_VERSION`), `code_hashes` uses package-relative keys that carry
  the folder and the content identities of `foundation/content_identity.py` under rule identifier
  `sosfuse2-identity-2`.
- Any other value on a signed table, that is a `meta.json` with `signature` written by `jobs/run_state.py`, is an error
  (`atmosphere/spectral.py`, `spectral.check_meta_contract`, `E_PROVENANCE_CONTRACT`). Readers of 1.3.4 and earlier
  refuse v5 tables with `E_PROVENANCE_CONTRACT`.
- A derived table without a signature is not a signed provenance record. Upstream merge tools write their own descriptive text into `contract`. The v3, v4 or v5 key rules belong to the
  signed source tables of such a table rather than to the table itself, and the reader does not open those source
  tables. Such a table may be paired with a signed table; the product twilight/day pair is exactly that.

## Keys

The v4 and v5 keys below are those written by 1.4.0 (v5), which carry the layer folder. v4 tables of 1.3.4 and earlier
have the same keys without it, for example `sosfuse2/paths.py`. In the v3 column, `<source>` stands for the
repository-relative source directory in the upstream checkout.

| v3 (`code_hashes` key) | v4 and v5 key | note |
|---|---|---|
| `<source>/<module>.py` (every `*.py`, tests included: 141 keys) | `sosfuse2/<layer>/<module>.py` for the fixed list `paths.PRODUCTION_MODULES` | tests and `checks/testkit.py` are not provenance inputs in v4 |
| — (not hashed) | `sosfuse2/solve/__init__.py` and `sosfuse2/solve/engine.py` (`content_identity.ACCEPTED_SOURCES`, 2) | the engine module and package initializer; the CPU solver modules (`rays/`, `atmosphere/extinction.py`, `solve/engine_core.py`, `outputs/preview_functions.py`) are in `PRODUCTION_MODULES` |
| — | `sosfuse2/foundation/__init__.py`, `sosfuse2/foundation/available_cpus.py`, `sosfuse2/outputs/pack_state.py` (`content_identity.HOST_MODULES`), `sosfuse2/backends/native/sh_cpu.cpp` | host helpers and the CPU kernel source |
| `<source>/kernel.cu` | `sosfuse2/backends/kernel.cu` | |
| `<source>/assets/<data>` (when the block used it) | `sosfuse2/assets/<data>` for `DATA_ASSETS`, always | v3 hashed the fine spectral assets only when the spectral block used them |
| `<source>/assets/usstd76_profile.json` | `sosfuse2/assets/usstd76_profile.json` — a logical key (`density.ASSET_KEY`) | the file is generated into the work root on first use (IPRT inputs, pinned sha256); hashed when the state's density configuration reads the asset, which includes the default `usstd76`/`usstd76`/`usstd` |
| `<source>/assets/strato_*.json`, `tropo_*.json` (state dependent) | `sosfuse2/assets/...` (state dependent) | `stratosphere.path_label`, `tropo_mie.path_label` |
| `assets/spectra_fine.json` (spectral block `asset` label) | unchanged `assets/spectra_fine.json` | the label is part of the block signature; kept for block compatibility. The same file is also `sosfuse2/assets/spectra_fine.json` in the fixed list |
| upstream geometry helper | removed | not an engine input |

The keys are serialized in sorted order, and every entry of the fixed lists must exist; a missing file raises
`E_PACKAGE_FILE_MISSING` rather than leaving a silent gap. A new production module must be added to
`paths.PRODUCTION_MODULES`, and the test `test_03c` keeps that list equal to the tree.

## Superseded asset digests

The `usstd76` ozone profile asset (`density.ASSET_KEY`) is regenerated from the IPRT inputs and pinned by
`density.ASSET_SHA256`. When the generator's arithmetic implementation changes without changing formulas, inputs or
normalization, the earlier digest is kept in `density.ASSET_SHA256_SUPERSEDED`, and readers treat a block or table whose
`ozone_profile.asset_sha256` is one of those digests as the same logical asset (`ozone_temperature.same_profile_identity`,
used by `spectral.check_asset`, `resolve` and `representative`). A digest outside that set is another asset and those
checks refuse it (`E_OZONE_T_ASSET_SHA`); reading the metadata of a table alone does not run them. New tables always record the current digest, and re-resolving a block that carries a superseded
digest records the current one.

| digest | generation | why superseded |
|---|---|---|
| `a7ccf7b1…` | float64/numpy generator | last-ulp values differed between arm64 and x86_64 (6 values, ≤1.9e-15), so a pinned digest could not hold across platforms; the decimal generator (`05dfeb6e…`) gave the same bytes on both platforms measured below, values within 4.7e-15 of this one |

The current `usstd76_profile.json` asset is regenerated from the IPRT inputs with `decimal` arithmetic. Its libmpdec
implementation is the same C algorithm in every CPython build, independent of libm and SIMD width; its `power()` is
"almost always" correctly rounded, which does not matter for reproducibility.
The digest `05dfeb6e…` was measured identical on arm64 with CPython 3.14 and on
x86_64 with CPython 3.12; other interpreters are not covered by that measurement. The earlier asset
`a7ccf7b1…` was produced with float64/numpy and differs per CPU architecture in the last ulp, 6 values at ≤ 1.9e-15
relative between arm64 and x86_64. The expressions, physical constants, inputs and normalization are unchanged; what
changed is the arithmetic implementation, the derived asset's values (≤ 4.7e-15 relative for air, ≤ 1.2e-15 for ozone) and
its hash. These differences lie nine orders of magnitude below the 1e-6 tolerance used for
prepared quantities. v3 tables record `ozone_profile.asset_sha256 = a7ccf7b1…` in their resolved spectral block and v4
tables record `05dfeb6e…`; readers compare asset hashes within the same `contract` version only.

## Signature inputs (v4 and v5)

The signed dictionary has the keys `contract`, `physics`, `optics_state`, `state`, `backend`, `reduced`, `settings`,
`polarization_model`, `tolerance`, `code_hashes` and `grid`, plus `derived` when a derived source is used. Its signature is
`hashlib.sha256(json.dumps(signed, sort_keys=True, allow_nan=False).encode()).hexdigest()`.

This is the dictionary of `jobs/run_state.py` (`signed_contract`). It identifies the inputs of a job: neither the table
arrays nor the bytes of a stored source enter it, and a table made from a stored source records that file separately as
`source_sha256`. `jobs/tables_from_sources.py` signs a dictionary of its own (`signature_of`): `physics`,
`optics_state`, `state`, `backend`, `reduced`, the wavelengths, solar elevations and view shape, `tolerance`,
`code_hashes`, and `settings` when given.

`package_version` and `package_commit` are outside the signed dictionary. The version literal in production source
contributes through `code_hashes`. `package_version` is the installed metadata
version when that metadata describes the loaded package, `sosfuse2.__version__` when it belongs to another installed copy,
and `null` in-tree without an install.

`package_commit` is resolved in this order: `SOSFUSE2_PACKAGE_COMMIT`; the
installer's `direct_url.json` (`vcs_info.commit_id` of a pip or uv VCS install), used only when the installed
distribution's `sosfuse2/__init__.py` is this very package; the git HEAD of the checkout whose `src/sosfuse2` is this
package (a consumer checkout that merely holds an installed copy is not recorded); else `null`. `gate.commit()`
records the same value, or `GATE_SOURCE_COMMIT`.

The code identity for baseline updates (`gate.code_identity`) is the same `package_hashes()` manifest minus `sosfuse2/checks/gate.py`, plus
`checks/gate_data/fixtures.json`, `backends/configs/{C3,C3_nostrato,devices}.json` and the ozone asset under its logical key, in one manifest
for both versions. A derived table (`--extra-elevations`) reuses a stored source only when the source job is complete, its
`contract, physics, id, settings, backend, polarization_model, mode, code_hashes` match, it recorded its density
profile, and its requested and computed optical state equal those of the derived job; the source file then passes the
checks of `radiance_table.load_source` (`E_SOURCE_DENSITY` and `E_SOURCE_TOL` below among them). After the tables are
made, the density record of the derived job must equal that of the source job, and its rows at −3° and −6° must differ
from the source job's table by at most 1e-6 of the row maximum (`E_EXTRA_ANCHOR`). `package_version` and
`package_commit` are recorded in its meta but are not a reuse condition. For its whole run the derived job holds the
source job's `<out>/.worker.lock` shared: it stops with `E_EXTRA_SOURCE` while a worker is writing that source, several
derived jobs may read one source at once, and a rerun of the source started meanwhile stops with `E_OUTPUT_BUSY`. Only
a source folder that the user cannot write and that holds no lock file is read without the lock.

## v3 to v4: signatures, caches and regression evidence

Every v4 signature differs from v3 for the same job, because the keys differ and more inputs are hashed.
`ray_geometry.identity` and the column and host caches also use package-relative keys, so the package does not reuse existing
packed geometry and column caches; they are rebuilt once.

As regression evidence, a production state, `L19s3_tau0p01_aod0p15_g0p65`, was baked by the installed package
on an NVIDIA GPU and compared with its v3 table: 0 of 20,250,000 values lie
outside rtol 1e-5. The two tables happened to be byte-identical, which is an observation and not a requirement.

## v4 to v5: identity rules

In 1.4.0, `foundation/content_identity.py` defines content identity under the rule identifier `sosfuse2-identity-2`.
Python sources are hashed without comments and docstrings, C and CUDA sources without comments, JSON files without their
note keys, and numeric assets without their per-asset description fields. Changes of spacing inside a line or of
indentation still change the identity, while trailing whitespace and blank lines do not. A comment inside an f-string
replacement field stays in the identity, a comment line directly after a line continuation (`\`) is replaced by the blank line
that the continuation requires, and a block comment in the middle of a C line is replaced by one space.

State identity leaves out `description`, `note`, `notes`, `note_*`, `*_note`, `group`, `role`, `holdout`,
`out_of_axis_range` and `q2d_override` and keeps `settings`, `parameters`, `stratosphere` and `density` whole. Every
other top-level key of the state file counts as well, `REPORT`, `sampling` and `supported_physics` among them. Numeric
assets record a computation identity apart from the sha256 of the file bytes: a spectral block is checked against the
computation identity, so whitespace, indentation and description changes pass and changed numbers are refused. The
density profile in `atmosphere/density.py` and the external source files in `asset_tools/iprt.py` and
`asset_tools/generate_minor_absorbers.py` keep their byte pins, and the earlier hashes are registered so older records
still match. `SOSFUSE2_POOL_LIMIT_GIB` and `SOSFUSE2_FIELD_RESERVE` are not part of any identity.

In 1.4.0 the keys of `code_hashes` also carry the folder from the module move. Tables are written in provenance format
v5 with `contract: "v5"` (`jobs/run_state.py` `CONTRACT_VERSION`). Readers still read v3 and v4 tables, whereas readers
of 1.3.4 and earlier refuse v5 tables (`E_PROVENANCE_CONTRACT`). Every cache key changes once and is rebuilt on first
use. Output folders and saved sources of earlier versions are not resumed or extended (`E_OUTPUT_IDENTITY`,
`E_EXTRA_SOURCE`), while `outputs/radiance_table` and `jobs/tables_from_sources` keep reading older sources whose
saved state records the density configuration; from 1.5.0 older sources without that record stop with `E_SOURCE_DENSITY`.
For `E_OUTPUT_IDENTITY` on an output from earlier code, use a separate output root or inspect and move the old output at the reported path.

From 1.6.0, `source_contract.anderson_tol` must match both the saved and requested solve tolerances.
Regenerate sources that lack this field (`E_SOURCE_TOL`).

Before sharing run records, verify them locally and redact installation paths in `timings[].solve_worker.file`,
`source_done`’s `worker.file` and `solve_failed`’s `engine_file` in the shared copy.

## Folder layout of 1.4.0

The modules moved into layer folders (`docs/ARCHITECTURE.md`), so the keys of `code_hashes` carry the folder, as in
`sosfuse2/jobs/run_state.py`. The keys and hashes change with the move, and with them the signature, the geometry,
column and host cache keys and the code identity for baseline updates. The caches rebuild on first use. The O₄ asset keeps the
generator module name it records, `sosfuse2.generate_minor_absorbers`, which is part of its physical identity.

## Tag map

The research release v0.1.1 corresponds to public `sosfuse2` v1.0.0.

When the seven solver modules moved into `src/sosfuse2/`, the production manifest included all seven and the
`contract` field stayed at v4. Geometry, host and column caches acquired new key versions. The optics module
contributed to table and baseline hashes only. That release could read old v4 tables but required fresh sources.
