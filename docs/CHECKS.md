# Checks: what runs where

For the required offline CPU check (`tests/test_offline_cpu.py`) and the command for your change, see
[Your first change](../CONTRIBUTING.md).

G-P compares against the NumPy golden at `rtol=1e-3`. G-T checks the recorded anchors. G-F compares the cold table
against the fixed radiance reference at `rtol=1e-3, atol=0` and compares cold and warm radiance and source arrays
at `rtol=1e-5, atol=0` in both accumulation modes. The checks fall into three tiers according to what the machine has.

## Tolerances of the numeric checks

G-P, the fixed CUDA comparison, G-T, G-H and G-F compare results against fixed references at the stated tolerance. G-R records hash and numerical
differences without blocking. G-A and the repeat check have no fixed reference: G-A compares two solves of one run, and
the repeat check compares runs of one build with each other.

| check | compares | tolerance |
|---|---|---|
| G-R | CUDA outputs vs the baseline of the matching toolchain and configuration, or the canonical baseline | none; differences recorded, not blocking |
| G-P | CUDA kernel vs NumPy golden (fixed number of apply steps) | radiance rtol 1e-3 (plus `--atol`, default 0); final source relative L2 at most 1e-3 per wavelength |
| CUDA | the fixture without a NumPy golden (nonzero albedo) vs its fixed CUDA baseline: radiance after the fixed apply steps and, when it was solved, the table | rtol 1e-3 (plus `--atol`, default 0) |
| G-T | anchor radiances (`checks/gate_data/anchors.json`) | ratchet, recorded |
| G-A | approximate fixture vs the exact fixture of the same run, which must pass its own check: radiance after the fixed apply steps and, when both were solved, the table | rtol 1e-3 (plus `--atol`, default 0); the approximate solve must converge in no more evaluations than the exact one |
| G-H | reduced selftest vs its recorded values | per case |
| G-PS | three persistent-service tests on the CUDA backend (parity, no state mixing, interleaving) | pass or fail |
| repeat (`--stage`) | same build, fresh runs in separate folders (two by default); also runs G-H and G-PS | rtol 1e-5, max difference recorded |
| G-F | full-grid fixed table, solved cold and then warm (needs the research checkout) | cold radiance vs the fixed radiance rtol 1e-3; cold vs warm radiance and source rtol 1e-5 (atol 0 for both) |

Accumulation order, lane mapping, record layout and the transport path may change between versions and are judged
against these fixed references and tolerances. `src/sosfuse2/checks/gate_data/baselines/` holds the baseline log and
reference manifest.

For studies requiring bit-stable repeats on the same CUDA GPU, set `SOSFUSE2_DETERMINISTIC=1`. This fixes the
accumulation order and disables the angular direction table, the tabulated first-light ground phase, and GPU arithmetic
for Anderson history staged in host memory, forcing transport to contract coefficients directly or use the fused kernel.
The experimental MLX transport ignores this variable and makes no determinism guarantee.

## 1. Any machine, no GPU, no upstream research checkout (public CI)

CI first runs the required offline CPU check described in the contributor guide above and fetches IPRT inputs for the
broader suite only after that check passes. The broader suite permits skips for optional inputs and hardware.

For changes to state setup or storage, `tests/test_run_boundaries.py` verifies that settings resolve independently of
the output folder and that storage writes and reuses arrays correctly. `tests/test_identity_runner.py::test_interleaved_runner_kernels`
solves two NumPy states with different settings, constructs a kernel with a stand-in device backend between them, and
compares the second state's arrays and evaluation count against a fresh process. The [task table](../AGENTS.md) lists
commands for each area.

```sh
pip install ".[test]"            # from a checkout of the tagged commit; the package is not on PyPI
python -B -m pytest -q -p no:cacheprovider --pyargs sosfuse2
```

While `SOSFUSE2_SKY_SOS_ROOT` is unset, tests that read inputs from the upstream research checkout skip at the function
or fixture level (setting `SOSFUSE2_SKY_SOS_ROOT` to a checkout missing a record fails rather than skips). CUDA-only
modules (`test_C_registers`, the `gate` CUDA paths) and 9 cases depending on the unpackaged `strato_colors` research
module also skip. The suite substitutes a synthetic O₄ table in an isolated temporary root without downloading HITRAN
data, so optional reproduction tests from original data (`SOSFUSE2_HITRAN_CIA`, `SOSFUSE2_BIRA_DIR`) skip when neither
variable is set.

Pass a module name to run a single file, such as `python -B -m pytest -q -p no:cacheprovider --pyargs sosfuse2.tests.test_names`.
The structure tests (`tests/test_layers.py`, `tests/test_names.py`, `tests/test_entry_points.py`,
`tests/test_structure_imports.py`) run in this tier, whereas `tests/test_docs_pointers.py`—which reads `AGENTS.md` and
`docs/` absent from the installed package—skips outside a source checkout and reports why.

The test session fetches the two IPRT inputs into the work root, `SOSFUSE2_WORK_ROOT` or else
`$XDG_CACHE_HOME/sosfuse2`, when they are not there yet, and the density asset is regenerated with platform-independent
arithmetic (`density.ASSET_SHA256`). The download makes up to three attempts if the server is unreachable (`E_IPRT_DOWNLOAD`). With
`SOSFUSE2_OFFLINE=1`, or when the download fails, the session runs without the IPRT inputs. The tests that depend on the
`usstd76` density asset then skip when they encounter `E_IPRT_OFFLINE` and report the command that fetches the files,
`python -B -c "from sosfuse2.asset_tools import iprt; iprt.inputs()"`. A test that fails for another reason is still a failure.

The check in a fresh virtual environment outside the source tree also runs one reduced NumPy solve after the one-time
O₄ and density preparation from the README
(`python -m sosfuse2.asset_tools.prepare_minor_absorbers --download` and `python -m sosfuse2 generate_density_profile`).
Without the O₄ step the solve stops with `E_O4_ASSET_MISSING`; without the density step it fetches the IPRT inputs on
first use, or stops with `E_IPRT_OFFLINE` under `SOSFUSE2_OFFLINE=1`:

```sh
python -B -m sosfuse2.jobs.run_state --id clear_34 --reduced --backend numpy --threads 2 --work-root <dir>
```

This example uses the bundled `states_v2.json`, with the background stratospheric layer enabled even when
`--stratosphere` is omitted. `test_strato_runner.py::test_bundled_v2_stratosphere_enabled_by_default` checks all 64
bundled states. To reproduce the earlier off state, pass an explicit state file with the layer disabled via `--states`;
there is no `--stratosphere off` option.

On macOS, the persistent queue (`pipeline --persistent`, `test_r17_port`, `test_pipeline_persistent`) listens on a Unix
socket under the run's temporary root, where macOS limits the path (terminator included) to 104 bytes. Keep `TMPDIR`
short; the default `/var/folders/...` is fine (`SOSFUSE2_TEST_SCRATCH` sets a separate folder for test outputs). Longer
paths stop `pipeline --persistent` with `E_PERSISTENT_SERVICE_START`, and tests that open the socket check path length
first and skip with `E_SOCKET_PATH_TOO_LONG`. The five cases of `test_pipeline_persistent` share a 4-state
non-persistent baseline queue and a 2-state persistent queue run once for the class, with some cases starting additional
queues and a separate test solving one state. Each queue waits up to `PERSISTENT_TEST_DEADLINE` seconds (2400 by
default) on NumPy unless `PERSISTENT_TEST_BACKEND` specifies another backend.

Public CI sets `PERSISTENT_TEST_DEADLINE=2400` (whereas the CUDA G-PS check of tier 2 runs three of those cases with
900 s) and splits the suite into four shards per platform and Python version. Because most files keep a single core busy,
each shard runs test files side by side, one file per worker (`pytest-xdist` with `--dist loadfile`, one worker per
CPU). `.github/ci/ci_shard.py` assigns whole files to shards using the measured file times in
`.github/ci/shard_weights.json`, giving unlisted new files the median time so they still run once. The `shard-coverage`
job collects every shard and fails unless together they include each collected test exactly once.

The tests of the long-running files carry the marker `heavy`; `HEAVY_FILES` in `tests/conftest.py` lists those files, each
of which took 20 s or longer in a serial run. `python -B -m pytest --pyargs sosfuse2 -m "not heavy"` runs the other files
and finishes in a few minutes. The public CI runs that selection once in the `quick` job and starts the shards only when
it passes; the shards run every test, with or without the marker.

A C++ compiler is needed only by checks and tests: `g++` or `clang++`
builds the CPU spherical-harmonics helper `backends/native/sh_cpu.cpp` once into the work root for the full CPU
reference comparison of `checks/solver_diagnostics.py --validate-mac`, which stops with `E_COMPILER_MISSING` without one.

The kernel syntax and preprocess tests need a C++ compiler, `clang++` or `clang` on `PATH`: `test_ground_fused`,
`test_sh_l`, `test_kernel_config`, `test_section_clocks_adversarial`, `test_profile_step`, the frozen kernel variant check in
`test_strato_adversarial`, `test_ground_sh`, `test_density_order`, `test_kernel_static`, `test_identity`, `test_regressions_110`,
`test_r17_port`, and the spherical-harmonic mismatch and static report checks in `test_strato_delta_adversarial`.
These tests skip when the required compiler is absent. Most report `E_TOOL_MISSING` through
`sosfuse2.checks.testkit.require_tool`. The three cases of `test_regressions_110` and the one of `test_r17_port` have
their own skip conditions on `clang++`. The kernel itself is compiled by NVRTC at run time and does not need clang.

The expected tier-1 outcome is `0 failed`. See `docs/KNOWN_FAILURES.md` for environment-dependent outcomes.
The +1 ULP tests of the trace interpolation, of on-grid
samples and of the last bit of atan2 are ordinary tests on every platform.

## 2. CUDA machine (CuPy, CUDA 12), no upstream research checkout

```sh
pip install ".[gpu,test]"        # from a checkout of the tagged commit; not on PyPI
python -m sosfuse2.asset_tools.prepare_minor_absorbers --download   # once: the fixtures use the O₄ asset (README, Quick start)
python -B -m sosfuse2.checks.gate --all --out <dir>          # G-R per GPU toolchain and kernel configuration, G-P vs NumPy, G-T anchors, G-A approximate vs exact solve of the same run, G-H reduced selftest vs its record, G-PS persistent service
python -B -m sosfuse2.checks.gate --stage --out <dir>        # repeat runs, rtol 1e-5
```

The installed `sosfuse2.tests.test_upper_air` module compares extended-domain radiances for one wavelength and
for 15 wavelengths with KSPLIT mode 3 against the NumPy result at `rtol=1e-3`. Run it with
`python -B -m pytest -q -rs --pyargs sosfuse2.tests.test_upper_air` and check that both CUDA cases ran.

The installed `sosfuse2.tests.test_option_combinations` module compares at `rtol=1e-3` the NumPy and CUDA tables of a
small 200 km fixture that shares the options of the settings example in `docs/MODEL.md` but not its grid or albedo.
With `SOSFUSE2_OPTION_SMOKE_SLOW=1` it also runs `run_state` on CUDA with the solve grid of the settings example (71
heights, 109 `χ` nodes, Legendre degree 24, tolerance 2e-4) on a test state for one wavelength, two solar elevations and
2×2 directions, and checks its records. Run it with
`SOSFUSE2_OPTION_SMOKE_SLOW=1 python -B -m pytest -q -rs --pyargs sosfuse2.tests.test_option_combinations` and check
that both CUDA cases ran.

The `gpu` extra provides the CUDA libraries loaded at run time by CuPy: NVRTC, the runtime, cuBLAS, cuSOLVER, cuSPARSE
and nvJitLink. Install the extra rather than CuPy alone; the Anderson step forms its products on the GPU or host
depending on history placement, configuration, deterministic mode and free memory, and solves the small regularized Gram
system in FP64 on the host.

The fixtures (`checks/gate_data/fixtures.json`) and baselines (`checks/gate_data/baselines/`) ship with the package.
`--bless` moves no fixed reference, instead recording the SHA, difference, GPU toolchain and configuration key, and
verdict. Only `--bless`, `--bless-numpy` and `--record-baselines` modify `checks/gate_data/baselines/` in the package,
which requires a writable checkout and stops with `E_GATE_CONFIG` on a read-only install.

Every top-level run writes its report to `<out>/gate.json`, while other execution records (such as `manifest.json` and
`BASELINE_LOG.md` from a passing G-F run) go under `<out>/records/` rather than the package. Because
`gate.code_identity()` hashes the generated `usstd76` density asset, the first run on a host downloads the IPRT inputs
against the pinned sha256 unless they are already cached in the work root.

## 3. CUDA machine with the upstream research checkout

```sh
SOSFUSE2_SKY_SOS_ROOT=/path/to/research-checkout python -B -m sosfuse2.checks.gate --gf --reference <table.npz> --out <dir>
```

The G-F check compares against a full-grid fixed table by reusing the upstream bake template via `SOSFUSE2_SKY_SOS_ROOT`,
a research-side dependency; it stops with `E_GATE_CONFIG` without that dependency and with `E_GATE_TEMPLATE` if loading
the template fails (for example, from importing a renamed module). The fixed table itself is excluded from the package.
`checks/gate_data/baselines/manifest.json` records only the file name under `gf.reference`
(`gf_L19s3_tau0p01_aod0p4_g0p65_e2620f6f90c39dcb.npz`) alongside its radiance SHA-256, and `BASELINE_LOG.md` reduces
historical paths to bare file or directory names. `gate --gf` requires `--reference <table.npz>` and stops with
`E_GATE_BASELINE_MISSING` when the file is absent (`checks/gate.py` `gf`); the SHA check `E_GATE_MANIFEST_SHA` still
applies.

Research tests (`test_cache_store`, `test_v14_runner`, section clock checks and others) run in this tier as well.
`gate --gf` and cases loading the bake template (`test_cache_store` and the G-F cases of `test_gate`) require a research
checkout whose scripts import the 1.4.0 module names. For `test_cache_store`, set `SOSFUSE2_TEST_BAKE_DRIVER` and
`SOSFUSE2_TEST_BAKE_PLANNER` to the bake driver and planner scripts (cases skip if either path is unset or missing; see
the [test environment variables](ENVIRONMENT.md#variables-of-the-tests-only)). Upstream cases of `test_ozone_temperature`
need the asset files listed in the checkout’s manifest, and `test_v14_runner`'s provenance check accepts the current
format (in 1.4.0 it expected v3 and failed). The runtime artifact check in `test_strato_delta_adversarial` reads the
retained output of an earlier harness run, skipping when absent and otherwise expecting a source of shape
`(40, 10 + 2 * sh_count(L), 1)` where `L` comes from the retained run’s `meta['settings']['L']`.

None of these research cases run in tier 1; this tier verifies deployment in the upstream research repository. The
optional `SADDLE_DATA_ROOT` names a folder holding the packed full-grid geometry cache and run logs read by
`test_plan_queue` and `test_column_parallel`, without which (and without the same data under `SOSFUSE2_SKY_SOS_ROOT`)
those tests skip. Cases testing the plan's CPU budget and host-cache records in `test_plan_queue` and batch writer locks
in `test_regressions_110` require no such data.

### Local test inputs

In each data root, put the `plan_queue` object in `.sosfuse2-test-inputs.json`. Its `geometry_cache`,
`run_logs` and `cache_root` are directory paths relative to that root, and `cost_profile` is a profile name.
For the optional second case, set `secondary_states` to a path relative to the research checkout and `secondary_logs` to one relative to the data root.
In the checkout's copy, set `section_clocks.evidence_dir`, `section_clocks.checker` and `v14.harness_dir` to relative paths.

A checkout used by the spectral tests also needs a `spectral` object.

```json
{"spectral": {"ozone_raw": "data/ozone.dat", "solar_raw": "data/solar.nc",
              "builder": "scripts/build.py", "bake_checker": "scripts/check.py"}}
```

If either raw file is absent, the test uses the stored self-check. If both are present, the builder is required.
A configured checkout also requires the bake checker. For `--runs`, add `legacy_commit` (40 or 64 lowercase hex digits),
`legacy_archive` (a list of paths to archive) and `legacy_runner` (a path relative to the unpacked tree).

## Contributing hooks

The pre-commit hooks `gitleaks` and `check-machine-paths` are the maintainer's local tools (`language: system`). An
external contributor only needs `pre-commit run public-identifiers`, a built-in pygrep over every text file.
