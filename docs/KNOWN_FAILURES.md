# Known failures (baseline set, first public release)

Three test cases fail in the research baseline and still fail in this release. They are marked
`xfail(strict=True)` through `sosfuse2.checks.testkit.known_failure(id, reason)`, so public CI stays green without hiding them:
each case still runs, its failure is the expected outcome, and a pass turns the run red until the entry below and the
decorator are removed together. Running a test module directly with `unittest.main()` (for example
`python -m sosfuse2.tests.test_strato_delta_adversarial --out …`) reports the original failure. We
left the numeric and engine code unchanged when splitting out the public package, so that its tables stay numerically
identical to the research baseline; fixing these three cases is deferred to a later release.

With clang++ on `PATH` and `SOSFUSE2_SKY_SOS_ROOT` unset, reproduce an entry by passing the name after `::`
in the second column to `-k`, for example `test_D11_initialize_1_15_lanes_and_source_widths`:

```sh
python -B -m pytest -q -p no:cacheprovider --runxfail --pyargs sosfuse2 -k "<test name from the table>"
```

| id | case (pytest node id under `sosfuse2`) | observed failure | cause as far as known | follow-up |
|---|---|---|---|---|
| `D15` | `test_strato_delta_adversarial.py::Delta::test_D15_SH_gate_accepts_absolute_rejects_material_mismatch` | clang++ error building `kernel.cu` as host C++: `K_FIRST`/`K_PHASE`/`K_BOUNDARY` and `__shfl_xor_sync` undefined | `test_strato_order.run` compiles the kernel with `-DFW=15 -DSTRATO` only, whereas the specialised kernel needs the three `K_*` defines that `backends/flags.py::resolve_config` supplies to NVRTC; the host stub also lacks `__shfl_xor_sync`. Without clang++ on `PATH` the case skips with `E_TOOL_MISSING` instead | pass the defines from `resolve_config` in `test_strato_order.run` and add `__shfl_xor_sync` to the host stub |
| `D16` | `test_strato_delta_adversarial.py::Delta::test_D16_D18_D22_adversarial_static_out_and_coverage` | `KeyError: 'strato'` when reading `kernel_static.json` rows | the report schema changed after the harness was written, so the `test_kernel_static` report rows no longer carry a `strato` key | check coverage using `W` and `STRATO` in `flags`, and replace the old read-count and text-injection assertions with syntax and failure checks |

Running the test module directly on the upstream baseline also reports `316 != 1132` for D11, the `K_FIRST`
clang error for D15 and `KeyError: 'strato'` for D16.

## Environment-dependent cases (not xfail)

Some cases depend on the host. The kernel syntax tests skip without `clang++` or `clang` on `PATH`. The persistent-queue
tests fail with `TimeoutExpired` when `PERSISTENT_TEST_DEADLINE` is too short for the host, and on macOS with
`E_PERSISTENT_SERVICE_START` when `TMPDIR` makes the socket path longer than 104 bytes. These outcomes depend on the host
rather than on the package; tier 1 of `docs/CHECKS.md` lists the conditions and the counts.

## Retiring an entry

1. Fix the cause, or document the deliberate change of behaviour, in a commit that touches only that case.
2. Remove the `@testkit.known_failure(...)` decorator and the table row in the same commit. With the decorator still in
   place, a passing case fails the run because of `strict=True`, which is the intended reminder.
