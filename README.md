# sosfuse2

Radiance tables of the clear twilight sky for a spherical-shell atmosphere. Multiple scattering is iterated to convergence
on a height and angle grid (successive orders of scattering, accelerated with Anderson mixing), with polarised Rayleigh
scattering, a tropospheric aerosol layer, an optional stratospheric sulfate layer, ozone absorption and a Lambert ground.
One solve gives the sky at every solar elevation of a table, by default +6° to −18°. The package has a NumPy reference
implementation, an optional CUDA kernel (compiled at run time through CuPy/NVRTC), numeric gates with tolerances against
fixed references, and a provenance record that signs every table with the hashes of the code and data that made it.

This is research code, published in the form it is used in. Version 1.0.0 is the first public release. The configurations
it was verified on are listed under *Status*, and nothing beyond them is claimed.

![Engine and MYSTIC radiance spectra at solar zenith angles 93, 96 and 99 degrees, and their ratio](docs/validation/mystic_vs_engine.png)

Sky radiance on the sun side at 20° elevation for solar elevations −3°, −6° and −9°: the engine (lines) and the Monte
Carlo code MYSTIC (markers) on the same optical input, and their ratio with the MYSTIC standard error. The dip near
600 nm is the ozone Chappuis band. Details in `docs/VALIDATION.md`.

## Why

With the Sun 6° to 9° below the horizon the sky overhead is lit only through the upper atmosphere: the Earth's shadow
stands about 35 km above the observer at 6° and about 79 km at 9°. A plane-parallel atmosphere has no such shadow, so the
model is a spherical shell. Monte Carlo codes handle this geometry but their radiances carry sampling noise: the MYSTIC
runs for the comparison below, one direction at three solar elevations and 30 wavelengths, took 6.6 CPU-hours for a
standard error of 0.5–2.4 % per value. sosfuse2 solves for a source field instead of
sampling photons, so a table of 100 × 180 directions and 45 bands has no sampling noise and covers all solar elevations
at once. `docs/MODEL.md` describes the physics, the grids and the iteration.

## Validation

On the same optical input, the engine and MYSTIC (libRadtran 2.0.6) agree to within 2.2 % per wavelength at solar
zenith angles 93° and 96° (MYSTIC standard error below 1 %). At 99° the ratios spread over 0.958–1.063, consistent with
the 2.0–2.4 % MYSTIC error (largest deviation 2.7 standard errors). The colour differences are |Δu′| ≤ 0.0012 and
|Δv′| ≤ 0.00014, at most 1.7 standard errors. The MYSTIC build reproduces the
120 cells of the spherical Rayleigh benchmark of Korkin et al. (2020) to 0.27 %. The comparison covers one state, one
direction and three solar elevations, and it used a table baked with the research build and the legacy spectral inputs,
not the 1.0.0 defaults. Because the input is shared, it tests the transport, not the input physics. `docs/VALIDATION.md`
gives the set-up, the tables and a qualitative check of the stratospheric aerosol input against measured zenith
polarisation (Ugolnikov et al. 2004).

## Limitations

- Ozone is the only absorber. NO₂, O₄, water vapour and O₂ are not in the model. In MYSTIC runs on the model's input,
  adding the NO₂ of the AFGL US standard profile (0.207 DU) lowered the blue radiance by 7 to 10 % and raised v′ by
  0.0035 to 0.0092 at the eight twilight points tested.
- The Rayleigh optical depth follows a pure λ⁻⁴ law. Replacing it by the Bodhaine et al. (1999) optical depth changed
  the radiance by −1.1 ± 0.2 % at one twilight pixel and the colour by less than its standard error.
- The tables hold total radiance only. Rayleigh polarisation is followed inside the solver, aerosol scattering is
  treated as depolarising, and no Stokes parameters are written.
- The numeric tolerances bound the discretisation and iteration error against fixed references of the same model. They
  are not a bound on the physical error.
- The colours have not been compared with measured twilight colours (Lee and Hernández-Andrés 2003; Lee 2026).

## Install

```sh
pip install .                 # from a checkout; or `pip install <sdist>` outside the source tree
pip install ".[test]"         # + pytest
pip install ".[gpu,test]"     # + CuPy (CUDA 12) and the NVIDIA runtime/NVRTC/cuBLAS wheels
```

Requires Python ≥ 3.12, NumPy ≥ 2.0 and Pillow, plus two executables on PATH: a C++ compiler (`g++` or `clang++`; the CPU
spherical-harmonics helper `native/sh_cpu.cpp` is built once into the work root, `E_COMPILER_MISSING` otherwise, and the
kernel syntax tests call `clang++`) and `zstd` (state outputs are packed as `.tar.zst`, `E_ZSTD_MISSING` otherwise).

Writable state (downloaded inputs, geometry and compile caches, table outputs) goes to `SOSFUSE2_WORK_ROOT`, by default
`$XDG_CACHE_HOME/sosfuse2`. The package directory is read-only data.

On a CUDA machine the kernels are compiled (NVRTC) on first use; `SOSFUSE2_COMPILE_CACHE=<dir>` keeps them between runs, and
`python -B -m sosfuse2.precompile --jobs 8` fills that cache before a queue starts.

## Quick start

One reduced-resolution state, NumPy backend, a few minutes on a laptop:

```sh
python -B -m sosfuse2.run_gpu_state_v3 --id clear_34 --reduced --backend numpy --threads 2 --work-root ./work
```

The state ids come from `sosfuse2/assets/states_v2.json` (the default `--states`; 64 parameter sets over `aod550`,
`angstrom`, `ssa`, `scale_height_km`, `g`, `ozone_du`, `albedo` and `stratosphere_tau550`; the solar depressions are the
runner's axis (`solar_elevation_deg` in its settings), not part of a state; `states_v1.json` holds the 40 first-generation
states). Every state in these two files marks its stratospheric layer `unsupported`, so the layer is off unless
`--stratosphere on` is given.
A full-resolution table needs the CUDA backend (`--backend cupy`); the
production queue that bakes many states with overlapping preparation is `python -B -m sosfuse2.pipeline`.

Every table is written with `meta.json` carrying `contract: "v4"`, the state, the backend, the tolerance settings and
`code_hashes` of the fixed list of production modules, frozen solver sources and data assets, plus a `signature` over
them. `docs/PROVENANCE_V3.md` defines the record and how readers that also hold older `v3` tables compare by logical
key.

## Numerical tolerances

Correctness is judged by tolerances against fixed references, never by byte identity:

| gate | compares | tolerance |
|---|---|---|
| G-P | CUDA kernel vs NumPy golden (fixed applies) | rtol 1e-3 |
| G-T | anchor radiances (`gate/anchors.json`) | ratchet, recorded |
| G-A / H / PS | queue parity, history, persistent service | per case |
| repeat (`gate --stage`) | same build, cold vs warm, repeated runs | rtol 1e-5, max difference recorded |
| G-F | full-grid fixed table (needs the research checkout) | rtol 1e-3 |

Accumulation order, lane mapping and record layout may change between versions; changes are judged
against the fixed references and their tolerances. `SOSFUSE2_DETERMINISTIC=1` selects deterministic accumulation for studies that need bit-stable repeats.
The baseline log and reference manifest live in `src/sosfuse2/gate/baselines/`.

## Tests and checks

`docs/CHECKS.md` describes three tiers and exactly what runs where:

1. Any machine, no GPU (this is the public CI): `python -B -m pytest -q -p no:cacheprovider --pyargs sosfuse2` from
   the installed package. Expected: `0 failed`, 5 `xfailed` (6 where NumPy dispatches AVX-512). Tests that need the research records skip unless
   `SOSFUSE2_SKY_SOS_ROOT` points at that checkout; tests that need `clang++` skip when it is absent.
2. CUDA machine: `python -B -m sosfuse2.gate --all` (G-P, G-T, G-A, H, PS) and `gate --stage`.
3. CUDA machine with the research checkout: `gate --gf`.

Five test cases fail in the research baseline the package was split from and are marked `xfail(strict=True)`; their
ids, causes and the rule for retiring them are in `docs/KNOWN_FAILURES.md`. They are not hidden: a case that starts
passing turns the run red until its entry is removed. One more entry, `GATE_V2_ON_GRID`, is a defect in the log/linear
interpolation rule that shows in the tests only with NumPy AVX-512 dispatch; it is fixed in 1.1.0 (same document).

## Status

Verified configurations for 1.0.0 (nothing else is claimed):

| platform | Python | backend | what was run |
|---|---|---|---|
| Linux x86_64, RTX 3080, CUDA 12 | 3.12 | CuPy | installed package: `gate --all` pass; one production state re-baked, final table within rtol 1e-5 of the research build |
| Linux x86_64 (clean room, no research checkout) | 3.12 | NumPy | installed package tests, reduced NumPy state |
| macOS arm64 | 3.14 | NumPy | installed package tests (clean room), in-tree tests, reduced NumPy state |

Known limits: Unix only (the solver uses `resource`); the NumPy reference runs in FP32 flush-to-zero mode, available on Darwin
arm64, Linux arm64 and Linux x86_64 only (other platforms, e.g. macOS x86_64, stop with `E_FTZ_UNSUPPORTED`); Python 3.11 untested; the persistent-queue tests need
a larger `PERSISTENT_TEST_DEADLINE` than the 60 s default (their setup queue took 462 s on a 12-core Linux host; CI uses
2400 s, see `docs/CHECKS.md`); a state that selects a `usstd76` density or ozone
profile downloads the IPRT input on first use; the default states do not (`SOSFUSE2_OFFLINE=1` to refuse, see
`THIRD_PARTY.md` §9).

## Layout

```
src/sosfuse2/
  fused.py, kernel.cu, kernel_config.py   CUDA kernel, launch configuration, NumPy reference (reference.py)
  geometry*.py, ray_columns.py, ...       spherical-shell geometry, columns, caches
  stratosphere.py, tropo_*.py, mie.py     aerosol layers and Mie tables
  spectral.py, air_wavelength.py          spectra, colour weights
  pipeline*.py, run_gpu_state_v3.py       state runner, production queue, provenance (v4)
  gate.py, gate/                          numeric gates, fixtures, fixed references
  accepted/                               frozen 2026-09 solver sources (7 files, read-only) and engine.py, the loader of the public split
  host/, native/sh_cpu.cpp                host helpers, CPU spherical-harmonics kernel
  assets/, configs/, fixtures/            data (see assets/README.md), kernel flag sets, test records
  test_*.py, testing.py                   the packaged tests
docs/                                     MODEL.md, VALIDATION.md (+ validation/ data and figure script), REFERENCES.md,
                                          CHECKS.md, KNOWN_FAILURES.md, PROVENANCE_V3.md
```

The seven solver sources in `src/sosfuse2/accepted/` (`a2_driver`, `b2_driver`, `sos_full`, `sos_probe`, `sos_render`,
`sos_sky`, `sos_sky2`) are kept byte-identical because they are the approved reference implementation that the numeric gates were
established against. `accepted/engine.py` is the loader written for the public split (text patches with stated match counts)
and is ordinary code. Formatting and lint changes are kept away from the frozen files.

## Related codes

libRadtran (Mayer and Kylling 2005; Emde et al. 2016) includes the Monte Carlo solver MYSTIC (Mayer 2009), which has a
spherical 1D atmosphere (Emde and Mayer 2007) and polarisation (Emde et al. 2010). SCIATRAN (Rozanov et al. 2014) is a
radiative transfer package for atmosphere and ocean. SASKTRAN (Bourassa et al. 2008) is a spherical-geometry code with a
successive-orders mode (Zawada et al. 2015), and vector successive-orders solvers are described by Min and Duan (2004)
and Lenoble et al. (2007). sosfuse2 does less than these: one spherical configuration, a source field iterated in place
of photon sampling, and radiance tables as the only product. Full references are in `docs/REFERENCES.md`.

## Maintenance

The package is maintained by one person as part of ongoing research. Issues and pull requests are read, but there is no
support commitment or release schedule. Changes to the physics (for example an added absorber) change the provenance
hashes, so tables from different versions are not interchangeable.

## Licence

Code: MIT (`LICENSE`). Data generated by this project: CC0 1.0 (`src/sosfuse2/assets/README.md`). Third-party-derived
data keep their original notices (`THIRD_PARTY.md`, also installed with the wheel): the CIE colour tables are CIE standard
data (CIE publishes them under CC BY-SA 4.0; the copies here came through Mitsuba 3 and colour-science, BSD-3-Clause), the
fine solar and ozone spectra are resampled from TSIS-1 HSRS v2 (LASP) and IUP Bremen cross sections under their publishers'
citation terms, and the LOWTRAN7 validation values are a U.S. Government work. The wheel's licence expression lists all of
them. Please cite the project (`CITATION.cff`) and the data sources listed in `THIRD_PARTY.md`.
