# Checks: what runs where

Results are checked against tolerances (G-P NumPy golden rtol 1e-3, G-T anchors, final tables rtol 1e-5); byte identity is
never a check. Three tiers, by what the machine has.

## 1. Any machine, no GPU, no sky-sos checkout (public CI)

```sh
pip install ".[test]"            # from a checkout of the tagged commit; the package is not on PyPI (2026-11-03 release review)
python -B -m pytest -q -p no:cacheprovider --pyargs sosfuse2
```

- Tests that read sky-sos research records (`SOSFUSE2_SKY_SOS_ROOT` unset) skip at function/fixture level; CUDA-only
  modules (`test_C_registers`, the `gate` CUDA paths) skip; `strato_colors` (research module, not packaged) skips 9 cases.
- The IPRT inputs are downloaded once into the work root (`SOSFUSE2_WORK_ROOT`, else `$XDG_CACHE_HOME/sosfuse2`) and the
  density asset is regenerated with platform-independent arithmetic (`density.ASSET_SHA256`). `SOSFUSE2_OFFLINE=1` refuses the
  download with the URLs to fetch by hand. An unreachable server is retried 3 times and then stops with `E_IPRT_DOWNLOAD`
  (file, URL and cause in the message). With `SOSFUSE2_OFFLINE=1` and an empty work root, the tests that depend on the
  `usstd76` density asset (about 23 cases) fail with `E_IPRT_OFFLINE`; they do not skip. Download the two files once
  (`python -B -c "from sosfuse2 import iprt; iprt.inputs()"`) or leave `SOSFUSE2_OFFLINE` unset.
- One reduced NumPy solve (what the clean-room check runs, ~minutes):
  `python -B -m sosfuse2.run_gpu_state_v3 --id clear_34 --reduced --backend numpy --threads 2 --work-root <dir>`
- Setting `SOSFUSE2_SKY_SOS_ROOT` to a checkout that lacks a record is a failure, not a skip.
- macOS: the persistent queue (`pipeline --persistent`, `test_r17_port`, `test_pipeline_persistent`) listens on a Unix socket
  under the run's temp root; macOS limits that path to 104 bytes. Keep `TMPDIR`/`SOSFUSE2_TEST_SCRATCH` short (the default
  `/var/folders/...` is fine; a 108-byte TMPDIR failed with `E_PERSISTENT_SERVICE_START` in the 2026-11-03 clean room).
  The five `test_pipeline_persistent` cases wait `PERSISTENT_TEST_DEADLINE` seconds (default 60) for a 4-state NumPy reduced
  queue to start; a busy Mac and a 12-core Linux host both exceeded 60 s in the 2026-11-03 clean rooms (`TimeoutExpired`
  errors). On a 12-core Linux host (2026-11-04) the setup queue alone took 462 s and the five cases 838 s in total, so
  300 s is not enough either; the public CI sets `PERSISTENT_TEST_DEADLINE=2400`. The CUDA G-A gate (tier 2) runs the
  same cases with 900 s.
- A C++ compiler (`clang++`/`clang` on PATH) is needed by the kernel syntax/preprocess tests (`test_ground_fused`, `test_sh_l`,
  `test_kernel_config`, `test_section_clocks_adversarial`, `test_profile_step`, `test_strato_adversarial` A20, `test_strato_order`
  via D15). Without it they skip with `E_TOOL_MISSING` (`sosfuse2.testing.require_tool`, 16 cases on a compiler-less host);
  the kernel itself is compiled by NVRTC at run time and does not need clang.
- Five cases are known failures from the research baseline and are `xfail(strict=True)`: ids, causes and the retirement rule are
  in `docs/KNOWN_FAILURES.md`. Expected tier-1 outcome: `0 failed`, 5 `xfailed`; an `XPASS` of a strict entry means an entry
  must be retired. The platform-dependent entry `GATE_V2_ON_GRID` adds a sixth `xfailed` where NumPy dispatches AVX-512.

## 2. CUDA machine (CuPy, CUDA 12), no sky-sos checkout

```sh
pip install ".[gpu,test]"        # from a checkout of the tagged commit; not on PyPI
python -B -m sosfuse2.gate --all --out <dir>          # G-R per toolkey, G-P vs NumPy golden, G-T ratchet, G-A, H, PS
python -B -m sosfuse2.gate --stage --out <dir>        # repeat runs, rtol 1e-5 (bytes only in SOSFUSE2_DETERMINISTIC=1)
```

The `gpu` extra brings the CUDA libraries CuPy loads at run time (NVRTC, runtime, cuBLAS, cuSOLVER, cuSPARSE, nvJitLink); a venv
without cuSOLVER/nvJitLink fails in the Anderson step with `Failure finding "libnvJitLink.so.12"` (`E_GATE_SOLVE`).
The fixtures (`gate/fixtures.json`) and baselines (`gate/baselines/`) ship with the package. `--bless` moves no fixed
reference; it records SHA, difference, toolkey and verdict. Only `--bless`, `--bless-numpy` and `--record-baselines` modify the
package's own `gate/baselines/` (they need a writable checkout and stop with `E_GATE_CONFIG` on a read-only install); every other
run writes its execution records (`manifest.json`, `BASELINE_LOG.md`, `anchors.json`) under `<out>/records/`.
`gate.code_identity()` needs the IPRT inputs (it hashes the generated `usstd76` density asset), so the first run on a host
downloads them (network, pinned sha256) unless they are already in the work root.

## 3. CUDA machine with the sky-sos research checkout

```sh
SOSFUSE2_SKY_SOS_ROOT=/path/to/sky-sos python -B -m sosfuse2.gate --gf --out <dir>
```

G-F (full-grid fixed table) reuses the sky-sos bake template `rig/bake/bake_run.py`, a research-side dependency; without
the root it fails with `E_GATE_CONFIG`. The fixed table itself is not part of the package: `gate/baselines/manifest.json`
records it under `gf.reference` as a home-relative path on the measurement host (`~/...`, expanded against your home
directory; `BASELINE_LOG.md` uses the same form; the absolute form was removed for publication, and a new record writes `~/…`
for a file under the home directory and the bare file name otherwise) together with its radiance SHA-256; pass the table with `--reference <table.npz>` (the SHA
check `E_GATE_MANIFEST_SHA` still applies). The research tests (`test_cache_store`, `test_v14_runner`, the kernel_ab evidence
packs, …) run in this tier as well. This tier is outside the public CI; it is the consumer's (sky-sos) deployment check.
`SADDLE_DATA_ROOT` (optional) names a research input root holding the packed full-grid geometry cache and the rig logs that
`test_plan_queue` and `test_column_parallel` read; without it (and without the same data under `SOSFUSE2_SKY_SOS_ROOT`) they skip.

## Contributing hooks

The pre-commit hooks `gitleaks` and `check-machine-paths` are the maintainer's local tools (`language: system`); an external
contributor only needs `pre-commit run public-identifiers` (a built-in pygrep over every text file).
