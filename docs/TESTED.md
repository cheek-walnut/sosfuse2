# Tested configurations

This file records where and how we verified releases. The notes below cover documentation-only releases, and the tables run from newest to oldest. In the tables below, `gate --all`,
`gate --stage` and `gate --gf` stand for the commands of *Tests and checks* in the README (tiers 2 and 3 of
`docs/CHECKS.md`), and H and PS for G-H and G-PS. Persistent-queue tests and CI default to a `PERSISTENT_TEST_DEADLINE` of 2400 s.

1.9.2 changes documentation only. Apart from the version number, its code is that of 1.9.0, so the 1.9.0 rows apply
to it.

1.9.1 changes documentation only. Apart from the version number, its code is that of 1.9.0, so the 1.9.0 rows apply
to it. For 1.9.1 we solved the
engine side of the two MYSTIC figures and the `clear_34` command again with 1.9.0 (*Versions behind the comparisons* in
`docs/VALIDATION.md`).

1.3.3 and 1.3.4 change documentation only and share the 1.3.2 code, with no tests or reference comparisons repeated
beyond the CI run.

1.2.1 and 1.2.2 change documentation only, so the 1.2.0 rows apply unchanged. The installed-sdist checks in the 1.2.3
table cover its changed default v2 states.

Configurations verified for 1.9.0:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | wheel in a fresh virtual environment, device profile `split`: `gate --all` exit 0 (G-R radiance of the five fixtures within 3.1e-5 and tables within 6.4e-6 of the recorded values, the fixed CUDA comparison, G-P and G-A, the five G-T pairs and G-PS passed, G-H max relative difference 1.3e-6); G-F against the fixed reference table: max relative difference 8.1e-6 (rtol 1e-3), cold/warm radiance and source max relative difference 0.0; the upper-air CUDA check against NumPy: max relative difference 1.9e-5. With `SOSFUSE2_DETERMINISTIC=1`, `gate --stage --runs 2` gave equal output hashes for the five fixtures. The CUDA test modules (projection batches, kernel occupancy, persistent pipeline) passed. On the same 10 GB GPU, the reproduction command of `docs/PERFORMANCE.md` for `clear_34` completed from an empty work root and warm |
| macOS arm64 | 3.14 | NumPy | release tree in place (`PYTHONPATH=src`, `SOSFUSE2_SKY_SOS_ROOT` unset), every test module in file order with `PERSISTENT_TEST_DEADLINE=300`: 1293 passed, 91 skipped, 0 failed |
| macOS arm64, Apple silicon GPU (16 GB) | 3.12 | NumPy + MLX 0.32.4 (`SOSFUSE2_MLX=1`) | `run_state --backend numpy --reduced` on the check fixture state FX-N (`checks/gate_data/fixtures.json`) with the stratospheric layer disabled (its reduced-solve AOD 0.02, black surface, 45 wavelengths in three chunks): every transport call on MLX (first 3, ground 21, warm 21, none on NumPy), Anderson evaluations per chunk 10/6/5 as with NumPy, table against the NumPy run of the same tree: max difference 3.2e-6 of each solar elevation's maximum, 2.6e-5 relative over all elements; on an otherwise idle machine, two runs each with two threads: about 39–41 s against 79–80 s with NumPy (transport passes 0.1–0.2 s against 41 s; the observer queries, on NumPy in both, about 20 s). Fixed applies of the check fixture FX-N against the NumPy golden: radiance max relative difference 9.3e-5 (rtol 1e-3). `tests/test_mlx_*.py`: 17 passed |

Configurations verified for 1.8.0:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | wheel in a fresh virtual environment, device profile `split`: `gate --all` exit 0 with the same results as for 1.7.0 (G-R radiance and table of the five fixtures equal to the recorded values, the fixed CUDA comparison, G-P and G-A, the five G-T pairs and G-PS passed, G-H max relative difference 1.4e-6); G-F against the fixed reference table: max relative difference 2.6e-6 (rtol 1e-3), cold/warm radiance and source max relative difference 0.0. On the same 10 GB GPU, `python -m sosfuse2 run_state` with no argument chose `cupy` and gave a table equal in every array to that of the command typed in full |
| macOS arm64 | 3.14 | NumPy | release tree installed in a fresh virtual environment (`SOSFUSE2_SKY_SOS_ROOT` unset), every test module in file order: 1246 passed, 76 skipped, 0 failed |

Configurations verified for 1.6.1:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | wheel in a fresh virtual environment, device profile `split`: `gate --all` exit 0 with the same results as for 1.6.0 (G-R radiance and table of the five fixtures equal to the recorded values, the CUDA, G-P and G-A checks, the five G-T pairs and PS passed, G-H max relative difference 1.4e-6); G-F against the fixed reference table: max relative difference 2.6e-6 (rtol 1e-3), cold/warm radiance and source max relative difference 0.0. On the same 10 GB GPU, the reproduction command of `docs/PERFORMANCE.md` for `clear_34` completed twice from an empty work root and twice warm (*Release 1.6.1 on GPU A* in that file) |
| macOS arm64 | 3.14 | NumPy | wheel in a fresh virtual environment, all test modules except the persistent-service module and `test_r17_port`, in file order and in two random orders (pytest-xdist, 8 workers): 1077 passed, 82 skipped, 0 failed in each of the three runs |
| macOS arm64 | 3.14 | NumPy | release tree in place (`PYTHONPATH=src`, `SOSFUSE2_SKY_SOS_ROOT` unset), every test module with pytest-xdist (8 workers) and `PERSISTENT_TEST_DEADLINE=900`: 1098 passed, 74 skipped, 0 failed and no expected failures |

Configurations verified for 1.6.0:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | wheel in a fresh virtual environment, device profile `split`: `gate --all` exit 0. In G-R the radiance and the table of all five fixtures equal the recorded values (max relative difference 0.0); only the bytes of the recorded source solve differ, which G-R reports without blocking. The CUDA, G-P and G-A checks, the five G-T pairs and PS passed, and G-H gave the same value as for 1.5.0 (max relative difference 1.4e-6). G-F against the fixed reference table: max relative difference 2.6e-6 (rtol 1e-3), cold/warm radiance and source max relative difference 0.0 |
| macOS arm64 | 3.14 | NumPy | wheel in a fresh virtual environment, all test modules except the persistent-service module and `test_r17_port`, in file order and in two random orders (pytest-xdist, 8 workers): 1067 passed, 82 skipped, 3 xfailed, 0 failed in each of the three runs |
| macOS arm64 | 3.14 | NumPy | release tree in place (`PYTHONPATH=src`, `SOSFUSE2_SKY_SOS_ROOT` unset), every test module with pytest-xdist (8 workers) and `PERSISTENT_TEST_DEADLINE=300`: 1083 passed, 74 skipped, 3 xfailed and 5 errors; the five errors are the tests of the persistent-service module that share one baseline run, which reached the 300 s deadline under that load. Run alone with the default deadline, that module gave 6 passed |

Configurations verified for 1.5.2, sharing the numerical code of 1.5.0:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | wheel in a fresh virtual environment, device profile `split`: `gate --all` exit 0, the same checks as for 1.5.0 passed with the same G-H value (max relative difference 1.4e-6); G-F against the fixed reference table: max relative difference 2.7e-6 (rtol 1e-3), cold/warm radiance and source max relative difference 0.0 |
| Linux x86_64 | 3.12 | NumPy | sdist in a fresh virtual environment outside the source tree, as the four shards of the continuous-integration job (pytest-xdist, 4 workers each): 930 passed, 3 xfailed and 0 failed in total, with 78 skipped cases, among them four files that skip at module level and are reported by every shard; an earlier serial run of the sdist gave 930 passed, 78 skipped, 3 xfailed |
| macOS arm64 | 3.14 | NumPy | earlier sdist in a fresh virtual environment outside the source tree, run as in continuous integration (empty work root, the session fetches the IPRT inputs, no prepared O₄ asset). Environment flags are covered by `test_entry_points`: 930 passed, 78 skipped, 3 xfailed, 0 failed |

1.5.1 changes documentation only and shares the 1.5.0 code, so the 1.5.0 rows apply. Beyond the CI run, its one new
check covers the scripts in `docs/validation/mystic` (*Reproducing the figures and the records* in
`docs/VALIDATION.md`).

Configurations verified for 1.5.0:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | wheel in a fresh virtual environment, device profile `split` selected with its 1.3.x name: `gate --all` exit 0, 78 nodes passed; G-F against the fixed reference from a different GPU: max relative difference 2.7e-6 (rtol 1e-3), cold/warm radiance and source max relative difference 0.0 |
| macOS arm64 | 3.14 | NumPy | release tree in place (`PYTHONPATH=src`), offline, with the IPRT inputs and the prepared O₄ asset in the work root: 939 passed, 68 skipped, 3 xfailed, 0 failed; without the IPRT inputs: 751 passed, 256 skipped (188 of them `E_IPRT_OFFLINE`), 3 xfailed, 0 failed; the quick start of that release's README as written in a fresh virtual environment with an empty work root (O₄ and density preparation, reduced solve 57 s) |

Configurations verified for 1.4.0:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | wheel in a fresh virtual environment, device profile `split` selected with its 1.3.x name: `gate --all` exit 0, 78 nodes passed; G-F against the fixed reference from a different GPU: max relative difference 2.7e-6 (rtol 1e-3), cold/warm radiance and source max relative difference 0.0; the state log of G-F shows the parallel geometry adapter in use |
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | wheel installed into a target directory placed first on `PYTHONPATH`: `checks.solver_diagnostics --backend cupy --solve --grid full --geometry-mode endpoints` exit 0, with the device memory reported; `clear_86` on the grid of *Finer grids with the 1.4.0 grid rules* in `docs/VALIDATION.md` (`h_refine` `b2a6`, `chi_refine_deg` `[88, 100]`, `L` 24) with the default CUDA profile: at most 1.9e-7 of the largest value of each solar elevation and colour channel from the research-build table |
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | wheel installed in the same way, device profile `split`: `gate --stage` exit 0, and the two staged runs of every fixture agree in all 131 arrays compared (max difference 0.0) |
| Linux x86_64, no `clang++` | 3.14 | NumPy | installed from the release tree in a fresh virtual environment outside it: 869 passed, 99 skipped, 2 xfailed, 0 failed, with `PERSISTENT_TEST_DEADLINE=1500` for the persistent pipeline cases; 24 cases skip for the missing `clang` and `clang++`, including the spherical-harmonic mismatch test, which leaves 2 of the 3 known failures; cached IPRT inputs and the prepared O₄ asset, offline |
| macOS arm64 | 3.14, 3.12 | NumPy | sdist (3.14 and 3.12) and wheel (3.14) in fresh virtual environments outside the source tree: the 14 test modules on identity, grid rules, the pool limit, tables from sources, the command entry point, layers, names, imports, documentation pointers and the walkthrough, 301 passed, 8 skipped, 0 failed in each; the skips are the 7 documentation pointer cases, whose files are not in the installed package, and the slow walkthrough case |

Configurations verified for 1.3.2. Because its kernel, solver, and reference checks match 1.3.0, the 1.3.0 GPU row
applies without repeating those checks:

| platform | Python | backend | what was run |
|---|---|---|---|
| macOS arm64 | 3.14 | NumPy | installed from the release tree in a fresh virtual environment outside it: 565 passed, 65 skipped, 3 xfailed, 0 failed, with the default `PERSISTENT_TEST_DEADLINE` and a short `TMPDIR` |

Configurations verified for 1.3.1. Because its kernel, solver, and reference checks match 1.3.0, the 1.3.0 GPU row
applies without repeating those checks:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, no `clang++` | 3.14 | NumPy | installed from the release tree in a fresh virtual environment outside it: 535 passed, 87 skipped, 2 xfailed, 0 failed, with `PERSISTENT_TEST_DEADLINE=1500` for the persistent pipeline cases; 19 cases skip for the missing compiler, including the spherical-harmonic mismatch test, which leaves 2 of the 3 known failures; case by case against 1.3.0 on the same host, the only changed outcome is the joint-walk case, from xfailed to passed; cached IPRT inputs and synthetic O₄ test data, offline |
| macOS arm64 | 3.14 | NumPy | tests run from the source tree with an empty work root and `SOSFUSE2_OFFLINE=1`: 504 passed, 117 skipped, 3 xfailed, 0 failed, with `PERSISTENT_TEST_DEADLINE=900` and a short `TMPDIR`; the tests that need the IPRT inputs skip with `E_IPRT_OFFLINE` |

Configurations verified for 1.3.0:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | wheel in a fresh virtual environment, the reference table’s device profile (named `split` after 1.3.x): `gate --all` exit 0, 78 nodes passed; G-F against the fixed reference from a different GPU: max relative difference 2.7e-6 (rtol 1e-3), cold/warm radiance and source max relative difference 0.0 |
| Linux x86_64, no `clang++` | 3.14 | NumPy | installed from the release tree in a fresh virtual environment outside it: 529 passed, 87 skipped, 3 xfailed, 0 failed, with `PERSISTENT_TEST_DEADLINE=1500` for the persistent pipeline cases; 19 cases skip for the missing compiler, including the spherical-harmonic mismatch test, which leaves 3 of the 4 known failures; cached IPRT inputs and synthetic O₄ test data, offline |

Configurations verified for 1.2.3:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | installed sdist in a fresh virtual environment (identical source tree): `gate --all` exit 0, 78 nodes passed; `clear_34 --reduced` without `--stratosphere`: layer enabled; G-F against the fixed reference: max relative difference 0.0 (rtol 1e-3), cold/warm radiance and source max relative difference 0.0 |
| macOS arm64 | 3.14 | NumPy | installed sdist in a fresh virtual environment outside the source tree: 525 passed, 65 skipped, 5 xfailed, 0 failed; cached IPRT inputs and synthetic O₄ test data, offline |

Configurations verified for 1.2.0:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | installed sdist: `gate --all` exit 0 with all 78 test nodes passing (G-R, G-P, G-T, G-A, H, PS); the full-grid G-F reference state baked through the production queue, table equal to the fixed G-F table of 1.1.0 (max relative difference 0.0), same O₄ asset |
| macOS arm64 | 3.14 | NumPy | installed sdist (a fresh virtual environment outside the source tree): 524 passed, 65 skipped, 5 xfailed, 0 failed; comparison with 1.1.1 in separate interpreters: the prepared arrays of five states, with and without the host cache, at rtol 1e-6 and a reduced solve (radiance and source) at rtol 1e-5, max relative difference 0.0 |

Configurations verified for 1.1.1, a documentation-only release:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | installed sdist: the full-grid G-F reference state baked through the production queue, table equal to the 1.1.0 fixed G-F table (max relative difference 0.0), same O₄ asset |
| macOS arm64 | 3.14 | NumPy | installed sdist (a fresh virtual environment outside the source tree): 524 passed, 65 skipped, 5 xfailed, 0 failed; the quick start of that release's README as written (O₄ download 3 s, reduced solve 36 s) |

Configurations verified for 1.1.0:

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, NVIDIA GPU, CUDA 12 | 3.12 | CuPy | release tree in place (`PYTHONPATH=src`): `gate --all` and `gate --gf` pass at the 1.1.0 baselines (the full-resolution product state runs at KSPLIT 0 on this GPU) |
| macOS arm64 | 3.14 | NumPy | installed package tests (a fresh virtual environment outside the source tree): 462 passed, 0 failed, 5 xfailed; the in-tree `--validate-mac` check and the NumPy reference arrays |
| Linux x86_64, AVX-512 | 3.14 | NumPy | targeted tests only: the log-blend unit tests and the +1 ULP test for the stratosphere-off fixture, NumPy AVX-512 dispatch on and off |
