# Provenance records: v3 → v4

The `meta.json` of every table carries `contract`, `signature` and `code_hashes`. Legacy tables in the research
records are v3, and tables baked from the `sosfuse2` package are v4
(`run_gpu_state_v3.CONTRACT_VERSION`). This note is for readers that hold both.

## Reader rule

- `contract == 'v3'`: the reader keeps the existing reading and verification semantics, with the repository-relative keys
  listed below. Such tables are verified in place; they are neither rejected nor accepted without checking, and they are
  not re-baked or re-signed to v4.
- `contract == 'v4'`: package-relative keys. Readers compare hashes by logical key rather than by file location.
- Any other value on a signed table, that is a `meta.json` with `signature` written by `run_gpu_state_v3`, is an error
  (`spectral.check_meta_contract`, E_PROVENANCE_CONTRACT).
- A derived table without a signature is not a signed provenance record. Merge tools, such as the table merge tool in the
  research records, write their own descriptive text into `contract`. Readers apply the v3/v4 key rules to the
  signed source tables of such a table rather than to the table itself, and may pair it with a signed table; the product
  twilight/day pair is exactly that.

## Keys

| v3 (`code_hashes` key) | v4 key | note |
|---|---|---|
| `scripts/sosfuse2/<module>.py` (every `*.py`, tests included: 141 keys) | `sosfuse2/<module>.py` for the fixed list `paths.PRODUCTION_MODULES` | tests and `testing.py` are not provenance inputs in v4 |
| — (not hashed) | `sosfuse2/accepted/<file>.py` (`run_gpu_state_v3.ACCEPTED_SOURCES`, 2) | the engine module and package initializer; the `solver_*.py` modules are in `PRODUCTION_MODULES` |
| — | `sosfuse2/host/{__init__,host_cores,pack_state}.py`, `sosfuse2/native/sh_cpu.cpp` | host helpers and the CPU kernel source |
| `scripts/sosfuse2/kernel.cu` | `sosfuse2/kernel.cu` | |
| `scripts/sosfuse2/assets/<data>` (when the block used it) | `sosfuse2/assets/<data>` for `DATA_ASSETS`, always | v3 hashed the fine spectral assets only when the spectral block used them |
| `scripts/sosfuse2/assets/usstd76_profile.json` | `sosfuse2/assets/usstd76_profile.json` — a logical key (`density.ASSET_KEY`) | the file is generated into the work root on first use (IPRT inputs, pinned sha256); hashed only when the state uses `ozone_profile usstd76` |
| `scripts/sosfuse2/assets/strato_*.json`, `tropo_*.json` (state dependent) | `sosfuse2/assets/...` (state dependent) | `stratosphere.path_label`, `tropo_mie.path_label` |
| `assets/spectra_fine.json` (spectral block `asset` label) | unchanged `assets/spectra_fine.json` | the label is part of the block signature; kept for block compatibility. The same file is also `sosfuse2/assets/spectra_fine.json` in the fixed list |
| `rig/docker/geometry_cpu.py` | removed | not an engine input |

The keys are serialized in sorted order, and every entry of the fixed lists must exist; a missing file raises
`E_PACKAGE_FILE_MISSING` rather than leaving a silent gap. A new production module must be added to
`paths.PRODUCTION_MODULES`, and the test `test_03c` keeps that list equal to the tree.

## Superseded asset digests

The `usstd76` ozone profile asset (`density.ASSET_KEY`) is regenerated from the IPRT inputs and pinned by
`density.ASSET_SHA256`. When the generator's arithmetic implementation changes without changing formulas, inputs or
normalization, the earlier digest is kept in `density.ASSET_SHA256_SUPERSEDED`, and readers treat a block or table whose
`ozone_profile.asset_sha256` is one of those digests as the same logical asset (`ozone_temperature.same_profile_identity`,
used by `spectral.check_asset`, `resolve` and `representative`). A digest outside that set is another asset and is refused
(`E_OZONE_T_ASSET_SHA`). New tables always record the current digest, and re-resolving a block that carries a superseded
digest records the current one.

| digest | generation | why superseded |
|---|---|---|
| `a7ccf7b1…` | float64/numpy generator | last-ulp values differed between arm64 and x86_64 (6 values, ≤1.9e-15), so a pinned digest could not hold across platforms; the decimal generator (`05dfeb6e…`) gives the same bytes everywhere, values within 4.7e-15 of this one |

The current `usstd76_profile.json` asset is regenerated from the IPRT inputs with `decimal` arithmetic. Its libmpdec
implementation is the same C algorithm in every CPython build, independent of libm and SIMD width; its `power()` is
"almost always" correctly rounded, which does not matter for reproducibility.
The digest `05dfeb6e…` was measured identical on arm64 with CPython 3.14 and on
x86_64 with CPython 3.12; other interpreters are not covered by that measurement. The earlier asset
`a7ccf7b1…` was produced with float64/numpy and differs per CPU architecture in the last ulp, 6 values at ≤ 1.9e-15
relative between arm64 and x86_64. The expressions, physical constants, inputs and normalization are unchanged; what
changed is the arithmetic implementation, the derived asset's values (≤ 4.7e-15 relative for air, ≤ 1.2e-15 for ozone) and
its hash. These differences lie nine orders of magnitude below the 1e-6 tolerance that the research records apply to
prepared quantities. v3 tables record `ozone_profile.asset_sha256 = a7ccf7b1…` in their resolved spectral block and v4
tables record `05dfeb6e…`; readers compare asset hashes within the same `contract` version only.

## Signature inputs (v4)

The signed dictionary has the keys `contract`, `physics`, `optics_state`, `state`, `backend`, `reduced`, `settings`,
`polarization_model`, `tolerance`, `code_hashes` and `grid`, plus `derived` when a derived source is used. Its signature is
`hashlib.sha256(json.dumps(signed, sort_keys=True, allow_nan=False).encode()).hexdigest()`.

`package_version` and `package_commit` are not signature inputs. They are recorded beside the signed fields in `meta.json`,
so that the same bytes sign the same whether the run is in-tree or installed. `package_version` is the installed metadata
version, `null` in-tree without an install. `package_commit` is resolved in this order: `SOSFUSE2_PACKAGE_COMMIT`; the
installer's `direct_url.json` (`vcs_info.commit_id` of a pip or uv VCS install), used only when the installed
distribution's `sosfuse2/__init__.py` is this very package; the git HEAD of the checkout whose `src/sosfuse2` is this
package, where a consumer checkout that merely holds an installed copy is not recorded; else `null`. `gate.commit()`
records the same value, or `GATE_SOURCE_COMMIT`.

The gate bless identity (`gate.code_identity`) is the same `package_hashes()` manifest minus `sosfuse2/gate.py`, plus
`gate/fixtures.json`, `configs/{C3,C3_nostrato,devices}.json` and the ozone asset under its logical key, in one manifest
for both versions. A derived table (`--extra-elevations`) reuses a stored source only when
`contract, id, settings, backend, polarization_model, mode, code_hashes` match; `package_version` and `package_commit`
are recorded in its meta but are not a reuse condition.

## v3 to v4: signatures, caches and regression evidence

Every v4 signature differs from v3 for the same job, because the keys differ and more inputs are hashed.
`geometry.identity` and the column and host caches also use package-relative keys, so the package does not reuse existing
packed geometry and column caches; they are rebuilt once.

As regression evidence, a production state, `L19s3_tau0p01_aod0p15_g0p65`, was baked by the installed package
on an RTX 3080 and compared with its v3 table: 0 of 20,250,000 values lie
outside rtol 1e-5. The two tables happened to be byte-identical, which is an observation and not a requirement.

## Tag map

The research release v0.1.1 corresponds to public `sosfuse2` v1.0.0. The tag points and older commit SHAs
remain in the research records.

Moving the solver modules into `src/sosfuse2/` keeps the `contract` field at v4. All seven promoted `solver_*.py` files
enter the fixed production manifest. The geometry, host and column key versions change once for the new source layout;
`solver_optics.py` is excluded from those cache keys but included in product and gate identity. Old v4 tables remain
readable, whereas old sources cannot be reused by new code.
