# Known failures (baseline set, first public release)

Five test cases fail in the research baseline that this package was split from and still fail here. They are marked
`xfail(strict=True)` through `sosfuse2.testing.known_failure(id, reason)`, so public CI stays green without hiding them:
each case still runs, its failure is the expected outcome, and a pass turns the run red until the entry below and the decorator
are removed together. The research CLI harnesses (`python -m sosfuse2.test_strato_adversarial`, `…test_strato_delta_adversarial
--out …`, plain `unittest.main()`) ignore the marker and report the failure as before. No numeric or engine code was changed for
these entries (sosfuse2 public split, user decision 2026-11-03); the fixes are follow-up work.

Reproduce one entry (clang++ on PATH, no `SOSFUSE2_SKY_SOS_ROOT`); `-k` takes the test name from the second column
(the part after `::`, e.g. `test_pack_reuses_edges_and_keeps_legacy_bytes` or `test_D11_initialize_1_15_lanes_and_source_widths`):

```sh
python -B -m pytest -q -p no:cacheprovider --runxfail --pyargs sosfuse2 -k "<test name from the table>"
```

| id | case (pytest node id under `sosfuse2`) | observed failure | cause as far as known | follow-up |
|---|---|---|---|---|
| `LEGACY_JOINT_WALK` | `test_ray_columns.py::JointColumnsTests::test_pack_reuses_edges_and_keeps_legacy_bytes` | `assert_equivalent` rtol 1e-6: 10 of 147 elements differ, max relative 1.28e-6, max absolute 2.6e-9 (one element 5.49e-11 vs 0) | joint walk (`ray_columns.integrate(..., legacy=True, edges=…)`) vs the three-pass `total_columns` integration differ on grazing rays (μ ≈ ±1e-9, h = 0.001 km). The test name says "bytes" but it already compared with a tolerance; loosening the tolerance would change what the test checks, so it was not done | investigate the grazing-ray branch; then either fix the walk or justify the tolerance with a measurement |
| `A18` | `test_strato_adversarial.py::TransportContract::test_A18_bounded_endpoint_sidecar_and_profile_identity` | `E_SPECTRAL_UNRESOLVED: profile_mean block without a state q identity` | the assertion predates the P2 A′ default (sky-sos research records, `docs/public_split/S1_REPORT.md` §7); uninvestigated — the sibling `test_ozone_temperature` 04/04b cover the current contract | decide whether the sidecar-identity rule changed deliberately; update the assertion or the kernel path |
| `D11` | `test_strato_delta_adversarial.py::Delta::test_D11_initialize_1_15_lanes_and_source_widths` | `st.source_size(s)` is 316, the harness asserts 1132 | kernel/harness mismatch since the kclean series (source layout shrank); the harness was never updated | re-derive the expected width from `sh_layout` instead of a literal |
| `D15` | `test_strato_delta_adversarial.py::Delta::test_D15_SH_gate_accepts_absolute_rejects_material_mismatch` | clang++ error building `kernel.cu` as host C++: `K_FIRST`/`K_PHASE`/`K_BOUNDARY` undefined | `test_strato_order.run` compiles the kernel with `-DFW=15 -DSTRATO` only; the specialised kernel needs the three `K_*` defines (`kernel_config.resolve_config` supplies them to NVRTC). Without clang++ on PATH the case skips (`E_TOOL_MISSING`) instead | pass the defines from `resolve_config` in `test_strato_order.run` |
| `D16` | `test_strato_delta_adversarial.py::Delta::test_D16_D18_D22_adversarial_static_out_and_coverage` | `KeyError: 'strato'` reading `kernel_static.json` rows | `test_kernel_static` report rows no longer carry a `strato` key (report schema moved on in the kclean series); the harness predates it | read the current report schema (or the `cases` key it now emits) |

### Platform-dependent entry

| id | case | observed failure | cause | follow-up |
|---|---|---|---|---|
| `GATE_V2_ON_GRID` | `test_gate.py::test_zero_weight_rule_removes_one_ulp_sensitivity` | with NumPy AVX-512 dispatch (`X86_V4`): S_max −9° 3.31e-3, −12° 6.63e-3 against the 2e-4 bound; passes with AVX2 dispatch (3.3e-5, 6.4e-5) | the Gate v2 rule ignores a corner only when its weight is exactly 0 (`reference.py`, `kernel.cu` `bad_trace`). About 16 % of the FX-N samples lie exactly on a chi node (ac == 0); +1 ULP on atan2 gives them a corner weight of about 1e-6, and if that corner's trace is below `TRACE_MIN` the whole sample switches to the linear fallback. About 2,840 samples flip this way with either dispatch; only the effect on the −9/−12° radiance differs. The defect is in the rule, on every platform | 1.1.0: make the log/linear choice continuous in the corner weight, in the reference and the kernel together |

This entry uses `known_failure(..., when=testing.numpy_avx512())`: `xfail(strict=False)` where NumPy dispatches AVX-512, a
normal test (must pass) elsewhere. Strict mode is not used because AVX-512 runners differ (`X86_V4` only, or `AVX512_ICL`)
and the outcome there was measured on one CPU family. Expected tier-1 outcome is therefore 5 `xfailed` on AVX2 and Apple
runners and 6 on AVX-512 runners. The CI log prints the dispatch before the tests ("Runner CPU and NumPy SIMD dispatch").

Evidence that these fail in the baseline too: the split-origin research tree (`scripts/sosfuse2/test_strato_delta_adversarial.py
--out …`, its own CLI) reports D11 `316 != 1132`, D15 the same `K_FIRST` clang error and D16 `KeyError: 'strato'`; the legacy
joint-walk and A18 failures reproduce on macOS arm64 and Linux x86_64 (sky-sos research records, `docs/public_split/S2_REPORT.md` §10, §14).

## Environment-dependent cases (skip, not xfail)

These are not failures of the package; `docs/CHECKS.md` tier 1 lists the conditions.

- No C++ compiler: the kernel syntax / preprocess tests call `clang++` or `clang` by name and skip with `E_TOOL_MISSING`
  (`sosfuse2.testing.require_tool`): `test_ground_fused` 4, `test_sh_l` 3, `test_kernel_config` 1, `test_section_clocks_adversarial`
  2, `test_profile_step::test_c_profile_at_masks_and_bad_n` and `test_c_native_profile_value_matches_interpolate32` 3, `test_strato_adversarial`
  A20, and D15 above through `test_strato_order.run`.
- Persistent queue start-up: `test_pipeline_persistent` (5 cases) waits `PERSISTENT_TEST_DEADLINE` seconds (default 60) for a
  4-state NumPy reduced queue; a 2026-11-03 Mac and a 12-core Linux host both exceeded it. On the Linux host the setup
  queue alone took 462 s, so 300 s is not enough either; the public CI uses 2400 s. The CUDA G-A gate runs the same cases
  with 900 s.
- macOS Unix-socket path: keep `TMPDIR`/`SOSFUSE2_TEST_SCRATCH` short (104-byte limit), see `docs/CHECKS.md`.

## Retiring an entry

1. Fix the cause (or document the deliberate change of behaviour) in a change that touches only that case.
2. Remove the `@testing.known_failure(...)` decorator and the table row in the same commit; with the decorator still in
   place a passing case fails the run (`strict=True`), which is the intended reminder.
