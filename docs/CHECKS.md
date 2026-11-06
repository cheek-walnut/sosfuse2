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

The IPRT inputs are downloaded once into the work root, `SOSFUSE2_WORK_ROOT` or else `$XDG_CACHE_HOME/sosfuse2`, and the
density asset is regenerated with platform-independent arithmetic (`density.ASSET_SHA256`). `SOSFUSE2_OFFLINE=1` refuses
the download and prints the URLs to fetch by hand. An unreachable server is retried 3 times before the run stops with
`E_IPRT_DOWNLOAD`, naming the file, the URL and the cause. With `SOSFUSE2_OFFLINE=1` and an empty work root, the tests
that depend on the `usstd76` density asset, about 23 cases, fail with `E_IPRT_OFFLINE` rather than skip. Download the two
files once with `python -B -c "from sosfuse2 import iprt; iprt.inputs()"` or leave `SOSFUSE2_OFFLINE` unset.

The check in a fresh virtual environment outside the source tree also runs one reduced NumPy solve, which takes a few
minutes, after the one-time O₄ preparation from the README
(`python -m sosfuse2.prepare_minor_absorbers --download`; without it the solve stops with
`E_O4_ASSET_MISSING`):

```sh
python -B -m sosfuse2.run_gpu_state_v3 --id clear_34 --reduced --backend numpy --threads 2 --work-root <dir>
```

On macOS, the persistent queue (`pipeline --persistent`, `test_r17_port`, `test_pipeline_persistent`) listens on a Unix
socket under the run's temporary root, and macOS limits that path to 104 bytes. Keep `TMPDIR` and `SOSFUSE2_TEST_SCRATCH`
short; the default `/var/folders/...` is fine, whereas a 108-byte `TMPDIR` failed with `E_PERSISTENT_SERVICE_START` in a
fresh virtual environment outside the source tree. The five `test_pipeline_persistent` cases wait
`PERSISTENT_TEST_DEADLINE` seconds, 60 by default, for a 4-state NumPy reduced queue to start. A busy laptop and a
12-core host both exceeded 60 s in fresh virtual environments outside the source tree with `TimeoutExpired` errors,
and on the latter host the setup queue alone took 462 s and the five cases 838 s in total, so 300 s is not enough either.
The public CI sets `PERSISTENT_TEST_DEADLINE=2400`, and the
CUDA G-A gate of tier 2 runs the same cases with 900 s.

The kernel syntax and preprocess tests need a C++ compiler, `clang++` or `clang` on `PATH`: `test_ground_fused`,
`test_sh_l`, `test_kernel_config`, `test_section_clocks_adversarial`, `test_profile_step`, `test_strato_adversarial` A20
and `test_strato_order` via D15. Without one they skip with `E_TOOL_MISSING` through `sosfuse2.testing.require_tool`, 16
cases on a compiler-less host. The kernel itself is compiled by NVRTC at run time and does not need clang.

Five cases are known failures from the research baseline and are marked `xfail(strict=True)`; their ids, causes and the
retirement rule are in `docs/KNOWN_FAILURES.md`. The expected tier-1 outcome is `0 failed` and 5 `xfailed`, and an
`XPASS` of a strict entry means that the entry must be retired. The +1 ULP tests of the trace interpolation, on-grid
samples and the atan2 last bit, are ordinary tests on every platform.

## 2. CUDA machine (CuPy, CUDA 12), no sky-sos checkout

```sh
pip install ".[gpu,test]"        # from a checkout of the tagged commit; not on PyPI
python -B -m sosfuse2.gate --all --out <dir>          # G-R per toolkey, G-P vs NumPy golden, G-T ratchet, G-A, H, PS
python -B -m sosfuse2.gate --stage --out <dir>        # repeat runs, rtol 1e-5 (bytes only in SOSFUSE2_DETERMINISTIC=1)
```

The `gpu` extra brings the CUDA libraries that CuPy loads at run time: NVRTC, the runtime, cuBLAS, cuSOLVER, cuSPARSE and
nvJitLink. A venv without cuSOLVER and nvJitLink fails in the Anderson step with `Failure finding "libnvJitLink.so.12"`
(`E_GATE_SOLVE`). The fixtures (`gate/fixtures.json`) and baselines (`gate/baselines/`) ship with the package. `--bless`
moves no fixed reference; it records SHA, difference, toolkey and verdict. Only `--bless`, `--bless-numpy` and
`--record-baselines` modify the package's own `gate/baselines/`, and they need a writable checkout, stopping with
`E_GATE_CONFIG` on a read-only install. Every other run writes its execution records, `manifest.json`, `BASELINE_LOG.md`
and `anchors.json`, under `<out>/records/`. `gate.code_identity()` hashes the generated `usstd76` density asset and
therefore needs the IPRT inputs, so the first run on a host downloads them over the network against the pinned sha256
unless they are already in the work root.

## 3. CUDA machine with the sky-sos research checkout

```sh
SOSFUSE2_SKY_SOS_ROOT=/path/to/sky-sos python -B -m sosfuse2.gate --gf --out <dir>
```

G-F, the full-grid fixed table, reuses the bake template from the research records, a research-side dependency, and
fails with `E_GATE_CONFIG` without the root. The fixed table itself is not part of the package.
`gate/baselines/manifest.json` records it under `gf.reference` as a home-relative path (`~/...`,
expanded against your home directory) together with its radiance SHA-256. Historical paths in `BASELINE_LOG.md` are reduced
to file or directory names, and a new record writes `~/…` for a file under the home directory and the bare file name
otherwise. Pass the table
with `--reference <table.npz>`; the SHA check `E_GATE_MANIFEST_SHA` still applies. The research tests (`test_cache_store`,
`test_v14_runner`, the kernel_ab evidence packs and others) run in this tier as well. This tier lies outside the public CI;
it is the deployment check of the consumer, sky-sos. `SADDLE_DATA_ROOT`, which is optional, names a research input root
holding the packed full-grid geometry cache and the rig logs that `test_plan_queue` and `test_column_parallel` read.
Without it, and without the same data under `SOSFUSE2_SKY_SOS_ROOT`, those tests skip.

## Contributing hooks

The pre-commit hooks `gitleaks` and `check-machine-paths` are the maintainer's local tools (`language: system`). An
external contributor only needs `pre-commit run public-identifiers`, a built-in pygrep over every text file.
