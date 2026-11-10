# Known failures (baseline set, first public release)

Four test cases fail in the research baseline and still fail here. They are marked
`xfail(strict=True)` through `sosfuse2.testing.known_failure(id, reason)`, so public CI stays green without hiding them:
each case still runs, its failure is the expected outcome, and a pass turns the run red until the entry below and the
decorator are removed together. The research CLI harnesses (for example
`python -m sosfuse2.tests.test_strato_delta_adversarial --out …`, plain `unittest.main()`) ignore the marker and report the failure as before. We
left the numeric and engine code unchanged when splitting out the public package, so that its tables stay numerically
identical to the research baseline; fixing these four cases is deferred to a later release.

To reproduce one entry, with clang++ on `PATH` and `SOSFUSE2_SKY_SOS_ROOT` unset, give `-k` the test name from the second
column, the part after `::`, for example `test_pack_reuses_edges_and_keeps_legacy_bytes` or
`test_D11_initialize_1_15_lanes_and_source_widths`:

```sh
python -B -m pytest -q -p no:cacheprovider --runxfail --pyargs sosfuse2 -k "<test name from the table>"
```

| id | case (pytest node id under `sosfuse2`) | observed failure | cause as far as known | follow-up |
|---|---|---|---|---|
| `LEGACY_JOINT_WALK` | `test_ray_columns.py::JointColumnsTests::test_pack_reuses_edges_and_keeps_legacy_bytes` | `packed['end_columns']` against the joint walk at rtol 1e-6 (the earlier `total_columns` assertion passes): 48 of 245 elements differ, max absolute difference 0.196 (e.g. 238.987 vs 239.018) | uninvestigated | find which of the two column sets is wrong; the marker stays until then |
| `D11` | `test_strato_delta_adversarial.py::Delta::test_D11_initialize_1_15_lanes_and_source_widths` | `st.source_size(s)` is 316, the harness asserts 1132 | the kernel source layout shrank, and the harness literal was never updated | derive the expected width from `sh_layout` instead of a literal |
| `D15` | `test_strato_delta_adversarial.py::Delta::test_D15_SH_gate_accepts_absolute_rejects_material_mismatch` | clang++ error building `kernel.cu` as host C++: `K_FIRST`/`K_PHASE`/`K_BOUNDARY` and `__shfl_xor_sync` undefined | `test_strato_order.run` compiles the kernel with `-DFW=15 -DSTRATO` only, whereas the specialised kernel needs the three `K_*` defines that `kernel_config.resolve_config` supplies to NVRTC; the host stub also lacks `__shfl_xor_sync`. Without clang++ on `PATH` the case skips with `E_TOOL_MISSING` instead | pass the defines from `resolve_config` in `test_strato_order.run` and add `__shfl_xor_sync` to the host stub |
| `D16` | `test_strato_delta_adversarial.py::Delta::test_D16_D18_D22_adversarial_static_out_and_coverage` | `KeyError: 'strato'` reading `kernel_static.json` rows | the `test_kernel_static` report rows no longer carry a `strato` key, because the report schema changed, and the harness predates that | read the current report schema, or the `cases` key it now emits |

These cases fail in the baseline too. The CLI run documented in the research records
(`scripts/sosfuse2/test_strato_delta_adversarial.py --out …`) reports D11 as `316 != 1132`, D15 with the same `K_FIRST`
clang error and D16 with `KeyError: 'strato'`, and the legacy joint-walk failure reproduces on macOS arm64.

## Environment-dependent cases (skip, not xfail)

Some cases skip rather than fail depending on the host: the kernel syntax tests need `clang++` or `clang` on `PATH`, the
persistent-queue tests need a `PERSISTENT_TEST_DEADLINE` suited to the host, and macOS needs a short `TMPDIR` for the Unix
socket. These are not failures of the package; tier 1 of `docs/CHECKS.md` lists the conditions and the counts.

## Retiring an entry

1. Fix the cause, or document the deliberate change of behaviour, in a change that touches only that case.
2. Remove the `@testing.known_failure(...)` decorator and the table row in the same commit. With the decorator still in
   place, a passing case fails the run because of `strict=True`, which is the intended reminder.
