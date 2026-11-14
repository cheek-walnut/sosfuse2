# Checks: what runs where

Every check compares results against a tolerance: the G-P NumPy golden at rtol 1e-3, the G-T anchors, and final tables at
rtol 1e-5. Byte identity is not a criterion. The checks fall into three tiers according to what the machine has.

## 1. Any machine, no GPU, no sky-sos checkout (public CI)

```sh
pip install ".[test]"            # from a checkout of the tagged commit; the package is not on PyPI
python -B -m pytest -q -p no:cacheprovider --pyargs sosfuse2
```

Tests that read sky-sos research records skip at function or fixture level while `SOSFUSE2_SKY_SOS_ROOT` is unset.
CUDA-only modules (`test_C_registers`, the `gate` CUDA paths) skip as well, and `strato_colors`, a research module that is
not packaged, skips 9 cases. Setting `SOSFUSE2_SKY_SOS_ROOT` to a checkout that lacks a record is a failure rather than a
skip.
The structure tests (`tests/test_layers.py`, `tests/test_names.py`, `tests/test_entry_points.py`,
`tests/test_structure_imports.py`) run in this tier. `tests/test_docs_pointers.py` reads `AGENTS.md` and `docs/`, which
the installed package does not contain, so it skips outside a source checkout and says so.

The test session fetches the two IPRT inputs into the work root, `SOSFUSE2_WORK_ROOT` or else
`$XDG_CACHE_HOME/sosfuse2`, when they are not there yet, and the density asset is regenerated with platform-independent
arithmetic (`density.ASSET_SHA256`). An unreachable server is retried 3 times (`E_IPRT_DOWNLOAD`). With
`SOSFUSE2_OFFLINE=1`, or when the download fails, the session runs without them: the tests that depend on the `usstd76`
density asset, which include every test that solves a state with the default density configuration, about 190 cases,
skip with `E_IPRT_OFFLINE` and the command that fetches the files,
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
`PERSISTENT_TEST_DEADLINE` seconds, 2400 by default, for a 4-state NumPy reduced queue to finish. The default was
60 s up to 1.3.1, which a busy laptop and a 12-core host both exceeded in fresh virtual environments outside the source
tree with `TimeoutExpired` errors; on the latter host the setup queue alone took 462 s and the five cases 838 s in total.
The public CI also sets `PERSISTENT_TEST_DEADLINE=2400`, and the CUDA G-A gate of tier 2 runs the same cases with 900 s.

The kernel syntax and preprocess tests need a C++ compiler, `clang++` or `clang` on `PATH`: `test_ground_fused`,
`test_sh_l`, `test_kernel_config`, `test_section_clocks_adversarial`, `test_profile_step`, `test_strato_adversarial`
A20, `test_ground_sh`, `test_density_order`, `test_kernel_static`, `test_identity`, `test_regressions_110`,
`test_r17_port`, and D15 of `test_strato_delta_adversarial`, which compiles through `test_strato_order.run`. Without one
they skip, 24 cases on a compiler-less host: 20 with `E_TOOL_MISSING` through `sosfuse2.checks.testkit.require_tool`,
and the three cases of `test_regressions_110` and the one of `test_r17_port` through their own skip conditions on
`clang++`. The kernel itself is compiled by NVRTC at run time and does not need clang.

Three cases are known failures from the research baseline and are marked `xfail(strict=True)`; their ids, causes and the
retirement rule are in `docs/KNOWN_FAILURES.md`. The expected tier-1 outcome is `0 failed` and 3 `xfailed` (2 without
`clang++`, where one of the three skips), and an
`XPASS` of a strict entry means that the entry must be retired. The +1 ULP tests of the trace interpolation, on-grid
samples and the atan2 last bit, are ordinary tests on every platform.

## 2. CUDA machine (CuPy, CUDA 12), no sky-sos checkout

```sh
pip install ".[gpu,test]"        # from a checkout of the tagged commit; not on PyPI
python -B -m sosfuse2.checks.gate --all --out <dir>          # G-R per toolkey, G-P vs NumPy golden, G-T ratchet, G-A, H, PS
python -B -m sosfuse2.checks.gate --stage --out <dir>        # repeat runs, rtol 1e-5 (bytes only in SOSFUSE2_DETERMINISTIC=1)
```

The `gpu` extra brings the CUDA libraries that CuPy loads at run time: NVRTC, the runtime, cuBLAS, cuSOLVER, cuSPARSE and
nvJitLink. A venv without cuSOLVER and nvJitLink fails in the Anderson step with `Failure finding "libnvJitLink.so.12"`
(`E_GATE_SOLVE`). The fixtures (`checks/gate_data/fixtures.json`) and baselines (`checks/gate_data/baselines/`) ship with the package. `--bless`
moves no fixed reference; it records SHA, difference, toolkey and verdict. Only `--bless`, `--bless-numpy` and
`--record-baselines` modify the package's own `checks/gate_data/baselines/`, and they need a writable checkout, stopping with
`E_GATE_CONFIG` on a read-only install. Every other run writes its execution records, `manifest.json`, `BASELINE_LOG.md`
and `anchors.json`, under `<out>/records/`. `gate.code_identity()` hashes the generated `usstd76` density asset and
therefore needs the IPRT inputs, so the first run on a host downloads them over the network against the pinned sha256
unless they are already in the work root.

## 3. CUDA machine with the sky-sos research checkout

```sh
SOSFUSE2_SKY_SOS_ROOT=/path/to/sky-sos python -B -m sosfuse2.checks.gate --gf --reference <table.npz> --out <dir>
```

G-F, the full-grid fixed table, reuses the bake template of the research records through `SOSFUSE2_SKY_SOS_ROOT`, a
research-side dependency, and stops with `E_GATE_CONFIG` without it and with `E_GATE_TEMPLATE` when the template
fails to load (for example an import of a renamed module). The fixed table itself is not part of the package.
`checks/gate_data/baselines/manifest.json` records only the file name under `gf.reference`
(`gf_L19s3_tau0p01_aod0p4_g0p65_e2620f6f90c39dcb.npz`) together with its radiance SHA-256. Historical paths in
`BASELINE_LOG.md` are reduced to file or directory names, and a new record writes only the bare file name. `gate --gf`
always requires `--reference <table.npz>` and stops with `E_GATE_BASELINE_MISSING` without an existing file
(`checks/gate.py` `gf`); the SHA check `E_GATE_MANIFEST_SHA` still applies. The research tests (`test_cache_store`,
`test_v14_runner`, the kernel_ab evidence packs and others) run in this tier as well. `gate --gf` and the cases that
load the bake template, `test_cache_store` and the G-F cases of `test_gate`, need a research checkout whose scripts
import the 1.4.0 module names, and the research-record cases of `test_ozone_temperature` need one that holds the files
its manifest names under `scripts/sosfuse2/assets/`. The provenance check of `test_v14_runner` accepts the current
format; in 1.4.0 it expected format v3 and failed. D04 of `test_strato_delta_adversarial` reads the retained output of an earlier harness run, skips
when that output is absent and expects a source of shape `(40, 1132, 1)` in it. None of these cases runs in tier 1. This
tier lies outside the public CI; it is the deployment check of the consumer, sky-sos. `SADDLE_DATA_ROOT`, which is
optional, names a research input root holding the packed full-grid geometry cache and the rig logs that
`test_plan_queue` and `test_column_parallel` read. Without it, and without the same data under `SOSFUSE2_SKY_SOS_ROOT`,
those tests skip. The cases for the plan's CPU budget and host-cache records in `test_plan_queue` and for the batch
writer's locks in `test_regressions_110` need no such data.

## Contributing hooks

The pre-commit hooks `gitleaks` and `check-machine-paths` are the maintainer's local tools (`language: system`). An
external contributor only needs `pre-commit run public-identifiers`, a built-in pygrep over every text file.
