# Known failures (baseline set, first public release)

Five test cases fail in the research baseline that this package was split from and still fail here. They are marked
`xfail(strict=True)` through `sosfuse2.testing.known_failure(id, reason)`, so public CI stays green without hiding them:
each case still runs, its failure is the expected outcome, and a pass turns the run red until the entry below and the
decorator are removed together. The research CLI harnesses (`python -m sosfuse2.tests.test_strato_adversarial`,
`…test_strato_delta_adversarial --out …`, plain `unittest.main()`) ignore the marker and report the failure as before. We
left the numeric and engine code unchanged when splitting out the public package, so that its tables stay numerically
identical to the research baseline; fixing these five cases is deferred to a later release.

To reproduce one entry, with clang++ on `PATH` and `SOSFUSE2_SKY_SOS_ROOT` unset, give `-k` the test name from the second
column, the part after `::`, for example `test_pack_reuses_edges_and_keeps_legacy_bytes` or
`test_D11_initialize_1_15_lanes_and_source_widths`:

```sh
python -B -m pytest -q -p no:cacheprovider --runxfail --pyargs sosfuse2 -k "<test name from the table>"
```

| id | case (pytest node id under `sosfuse2`) | observed failure | cause as far as known | follow-up |
|---|---|---|---|---|
| `LEGACY_JOINT_WALK` | `test_ray_columns.py::JointColumnsTests::test_pack_reuses_edges_and_keeps_legacy_bytes` | `assert_equivalent` rtol 1e-6: 10 of 147 elements differ, max relative 1.28e-6, max absolute 2.6e-9 (one element 5.49e-11 vs 0) | the joint walk (`ray_columns.integrate(..., legacy=True, edges=…)`) and the three-pass `total_columns` integration differ on grazing rays (μ ≈ ±1e-9, h = 0.001 km). The test name says "bytes", but the test already compared with a tolerance; loosening it would change what the test checks, so we did not | investigate the grazing-ray branch, then either fix the walk or justify the tolerance with a measurement |
| `A18` | `test_strato_adversarial.py::TransportContract::test_A18_bounded_endpoint_sidecar_and_profile_identity` | `E_SPECTRAL_UNRESOLVED: profile_mean block without a state q identity` | the assertion predates the profile-mean ozone temperature default and has not been investigated; the sibling cases `test_ozone_temperature` 04 and 04b cover the current rule | decide whether the sidecar-identity rule changed deliberately, then update the assertion or the kernel path |
| `D11` | `test_strato_delta_adversarial.py::Delta::test_D11_initialize_1_15_lanes_and_source_widths` | `st.source_size(s)` is 316, the harness asserts 1132 | the source layout shrank in the kclean series of the kernel, and the harness literal was never updated | derive the expected width from `sh_layout` instead of a literal |
| `D15` | `test_strato_delta_adversarial.py::Delta::test_D15_SH_gate_accepts_absolute_rejects_material_mismatch` | clang++ error building `kernel.cu` as host C++: `K_FIRST`/`K_PHASE`/`K_BOUNDARY` undefined | `test_strato_order.run` compiles the kernel with `-DFW=15 -DSTRATO` only, whereas the specialised kernel needs the three `K_*` defines that `kernel_config.resolve_config` supplies to NVRTC. Without clang++ on `PATH` the case skips with `E_TOOL_MISSING` instead | pass the defines from `resolve_config` in `test_strato_order.run` |
| `D16` | `test_strato_delta_adversarial.py::Delta::test_D16_D18_D22_adversarial_static_out_and_coverage` | `KeyError: 'strato'` reading `kernel_static.json` rows | the `test_kernel_static` report rows no longer carry a `strato` key, because the report schema moved on in the kclean series, and the harness predates that | read the current report schema, or the `cases` key it now emits |

These cases fail in the baseline too. The research tree the package was split from, run through its own CLI
(`scripts/sosfuse2/test_strato_delta_adversarial.py --out …`), reports D11 as `316 != 1132`, D15 with the same `K_FIRST`
clang error and D16 with `KeyError: 'strato'`, and the legacy joint-walk and A18 failures reproduce on macOS arm64 and
Linux x86_64.

## Environment-dependent cases (skip, not xfail)

Some cases skip rather than fail depending on the host: the kernel syntax tests need `clang++` or `clang` on `PATH`, the
persistent-queue tests need a `PERSISTENT_TEST_DEADLINE` suited to the host, and macOS needs a short `TMPDIR` for the Unix
socket. These are not failures of the package; tier 1 of `docs/CHECKS.md` lists the conditions and the counts.

## Retiring an entry

1. Fix the cause, or document the deliberate change of behaviour, in a change that touches only that case.
2. Remove the `@testing.known_failure(...)` decorator and the table row in the same commit. With the decorator still in
   place, a passing case fails the run because of `strict=True`, which is the intended reminder.
