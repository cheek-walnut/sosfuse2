# Checks: what runs where

G-P compares against the NumPy golden at `rtol=1e-3`. G-T checks the recorded anchors. G-F compares the cold table
against the fixed radiance reference at `rtol=1e-3, atol=0` and compares cold and warm radiance and source arrays
at `rtol=1e-5, atol=0` in both accumulation modes. The checks fall into three tiers according to what the machine has.

## 1. Any machine, no GPU, no upstream research checkout (public CI)

```sh
pip install ".[test]"            # from a checkout of the tagged commit; the package is not on PyPI
python -B -m pytest -q -p no:cacheprovider --pyargs sosfuse2
```

Tests that read inputs from the upstream research checkout skip at function or fixture level while `SOSFUSE2_SKY_SOS_ROOT` is unset.
CUDA-only modules (`test_C_registers`, the `gate` CUDA paths) skip as well, and 9 cases that depend on `strato_colors`, a
research module that is not packaged, skip too. Setting `SOSFUSE2_SKY_SOS_ROOT` to a checkout that lacks a record is a failure rather than a
skip.
The structure tests (`tests/test_layers.py`, `tests/test_names.py`, `tests/test_entry_points.py`,
`tests/test_structure_imports.py`) run in this tier. `tests/test_docs_pointers.py` reads `AGENTS.md` and `docs/`, which
the installed package does not contain, so it skips outside a source checkout and says so.

The test session fetches the two IPRT inputs into the work root, `SOSFUSE2_WORK_ROOT` or else
`$XDG_CACHE_HOME/sosfuse2`, when they are not there yet, and the density asset is regenerated with platform-independent
arithmetic (`density.ASSET_SHA256`). The download is retried 3 times if the server is unreachable (`E_IPRT_DOWNLOAD`). With
`SOSFUSE2_OFFLINE=1`, or when the download fails, the session runs without the IPRT inputs. The tests that depend on the
`usstd76` density asset, about 190 cases including every test that solves a state with the default density configuration,
then skip, reporting `E_IPRT_OFFLINE` and the command that fetches the files,
`python -B -c "from sosfuse2.asset_tools import iprt; iprt.inputs()"`. A test that fails for another reason is still a failure.

The check in a fresh virtual environment outside the source tree also runs one reduced NumPy solve, which takes a few
minutes, after the one-time O₄ and density preparation from the README
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
socket under the run's temporary root, and macOS limits that path to 104 bytes. Keep `TMPDIR` and `SOSFUSE2_TEST_SCRATCH`
short; the default `/var/folders/...` is fine, whereas a 108-byte `TMPDIR` failed with `E_PERSISTENT_SERVICE_START` in a
fresh virtual environment outside the source tree. The five `test_pipeline_persistent` cases wait
`PERSISTENT_TEST_DEADLINE` seconds, 2400 by default, for a 4-state NumPy reduced queue to finish.
The public CI also sets `PERSISTENT_TEST_DEADLINE=2400`, and the CUDA G-A check of tier 2 runs the same cases with 900 s.
The public CI splits this run into four shards for each platform and Python version and runs the test files of a shard
side by side, one file per worker (`pytest-xdist` with `--dist loadfile`, one worker per CPU), because most files keep a
single core busy. `.github/ci/ci_shard.py` assigns whole files to shards by the measured file times in
`.github/ci/shard_weights.json`; a file that is not listed there gets the median time, so a new file still runs once. The
`shard-coverage` job collects every shard and fails unless together they hold each collected test exactly once.

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
python -B -m sosfuse2.checks.gate --all --out <dir>          # G-R per GPU toolchain and kernel configuration, G-P vs NumPy, G-T anchors, G-A queue parity, H historical selftest, PS persistent service
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

The `gpu` extra brings the CUDA libraries that CuPy loads at run time: NVRTC, the runtime, cuBLAS, cuSOLVER, cuSPARSE and
nvJitLink. A venv without cuSOLVER and nvJitLink fails in the Anderson step with `Failure finding "libnvJitLink.so.12"`
(`E_GATE_SOLVE`). The fixtures (`checks/gate_data/fixtures.json`) and baselines (`checks/gate_data/baselines/`) ship with the package. `--bless`
moves no fixed reference; it records the SHA, difference, GPU toolchain and configuration key, and verdict. Only `--bless`, `--bless-numpy` and
`--record-baselines` modify the package's own `checks/gate_data/baselines/`, and they need a writable checkout, stopping with
`E_GATE_CONFIG` on a read-only install. Every other run writes its execution records, `manifest.json`, `BASELINE_LOG.md`
and `anchors.json`, under `<out>/records/`. `gate.code_identity()` hashes the generated `usstd76` density asset and
therefore needs the IPRT inputs, so the first run on a host downloads them over the network against the pinned sha256
unless they are already in the work root.

## 3. CUDA machine with the upstream research checkout

```sh
SOSFUSE2_SKY_SOS_ROOT=/path/to/research-checkout python -B -m sosfuse2.checks.gate --gf --reference <table.npz> --out <dir>
```

The G-F check against the full-grid fixed table reuses the upstream bake template through
`SOSFUSE2_SKY_SOS_ROOT`, a research-side dependency. It stops with `E_GATE_CONFIG` without that dependency and with
`E_GATE_TEMPLATE` when the template fails to load (for example, because of an import of a renamed module). The fixed table itself is not part of the package.
`checks/gate_data/baselines/manifest.json` records only the file name under `gf.reference`
(`gf_L19s3_tau0p01_aod0p4_g0p65_e2620f6f90c39dcb.npz`) together with its radiance SHA-256. Historical paths in
`BASELINE_LOG.md` are reduced to file or directory names, and a new record writes only the bare file name. `gate --gf`
always requires `--reference <table.npz>` and stops with `E_GATE_BASELINE_MISSING` without an existing file
(`checks/gate.py` `gf`); the SHA check `E_GATE_MANIFEST_SHA` still applies. The research tests (`test_cache_store`,
`test_v14_runner`, the section clock checks and others) run in this tier as well. `gate --gf` and the cases that
load the bake template (`test_cache_store` and the G-F cases of `test_gate`) need a research checkout whose scripts
import the 1.4.0 module names. For `test_cache_store`, set `SOSFUSE2_TEST_BAKE_DRIVER` to the bake driver script and
`SOSFUSE2_TEST_BAKE_PLANNER` to the planner script. Cases that use either script skip when its path is unset or missing.
See the [test environment variables](ENVIRONMENT.md#variables-of-the-tests-only).
The upstream cases of `test_ozone_temperature` need the asset files listed in the checkout’s manifest. The provenance check of `test_v14_runner` accepts the current
format; in 1.4.0 it expected format v3 and failed. The runtime artifact check in `test_strato_delta_adversarial` reads the retained output of an earlier harness run, skips
when that output is absent and expects a source of shape `(40, 10 + 2 * sh_count(L), 1)`. Here `L` comes from the retained run’s `meta['settings']['L']`. None of these cases runs in tier 1. This
tier checks deployment in the upstream research repository. The optional `SADDLE_DATA_ROOT` names a folder
with the packed full-grid geometry cache and run logs that
`test_plan_queue` and `test_column_parallel` read. Without it, and without the same data under `SOSFUSE2_SKY_SOS_ROOT`,
those tests skip. The cases for the plan's CPU budget and host-cache records in `test_plan_queue` and for the batch
writer's locks in `test_regressions_110` need no such data.

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
