# MYSTIC comparison

These scripts compare MYSTIC radiance with radiance from the sosfuse2 engine for the bundled states and viewing directions.
The full configuration uses 8 states, 6 pixels, solar zenith angles 93° and 96°, 30 wavelengths and 16 seeds.
The default engine configuration is A0.

## Requirements

Use Python with NumPy and sosfuse2 1.9.0 to 1.9.3 installed (their engines give the same tables; the
recorded tables come from 1.9.0).
The scripts are in the repository and the source distribution under `docs/validation/mystic`; the wheel does not
contain them.
Building libRadtran 2.0.6 requires Debian or Ubuntu on x86_64, a compiler, network access and package installation
privileges.
The build script installs `netcdf-bin`, which provides `ncgen` for input generation.
Engine tables require a CUDA GPU, CuPy and the `zstd` command.
Kernel options follow the sosfuse2 device profile of the GPU. The recorded tables used the profile now named `cuda`. The commands below select it explicitly
to reproduce the same kernel options across GPUs. Releases before 1.6.0, 1.5.0 and 1.5.1 among them, have another name for that profile and stop
with `E_DEVICE_PROFILE` at `cuda`: with such a release, replace `cuda` in the two `bake_engine.sh` commands below by the
name that `backends/configs/devices.json` of the installed release gives the same profile.
`bake_engine.sh` clears `SOSFUSE2_FMAD` and `SOSFUSE2_FASTMATH` inherited from the shell, so those two options come
from the profile as the CSV header states.
Set `PYTHON` to the Python executable when it is not `python`.

Prepare the O₄ asset before exporting optics.
On first use, sosfuse2 generates the density profile from its pinned input files.
Both preparation steps need network access unless the required inputs are already available locally.
For offline use, prepare the profile and O₄ asset in the selected work root beforehand and set `SOSFUSE2_OFFLINE=1`.

```sh
export PYTHON=python
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to a directory outside the script folder}"
mkdir -p "$RUN_ROOT"
export SOSFUSE2_WORK_ROOT="$RUN_ROOT/sosfuse2-assets"
"$PYTHON" -m sosfuse2.asset_tools.prepare_minor_absorbers --download
# Alternatively: "$PYTHON" -m sosfuse2.asset_tools.prepare_minor_absorbers --cia FILE
"$PYTHON" -m sosfuse2.asset_tools.generate_density_profile
```

## Full run

Run these commands from a directory containing the scripts and use a separate working directory.
The build directory must be empty.

```sh
H="$PWD"
WORK="$RUN_ROOT/work"
mkdir -p "$WORK" "$RUN_ROOT/build"
bash "$H/build_libradtran.sh" "$RUN_ROOT/build"
LRT="$RUN_ROOT/build/libRadtran-2.0.6"
"$PYTHON" -B "$H/export_optics.py" --work "$WORK"
"$PYTHON" -B "$H/inputs.py" --work "$WORK" --lrt-dir "$LRT"
"$PYTHON" -B "$H/run_mystic.py" --work "$WORK" --lrt-dir "$LRT" --jobs "$(nproc)"
SOSFUSE2_DEVICE_PROFILE=cuda PYTHON="$PYTHON" bash "$H/bake_engine.sh" "$WORK"
SOSFUSE2_DETERMINISTIC=1 SOSFUSE2_GROUND_SH_L=24 "$PYTHON" -B "$H/extract_engine.py" --work "$WORK"
"$PYTHON" -B "$H/compare.py" --work "$WORK"
"$PYTHON" -B "$H/make_csv.py" --work "$WORK" --out "$WORK/mystic_states.csv"
"$PYTHON" -B "$H/summarize.py" --work "$WORK"
```

The full configuration contains 46,080 MYSTIC runs.
The recorded total `uvspec` CPU time was 359,883 seconds; elapsed time depends on the available parallelism.
A nonpositive output is retried with the reserved alternate seed, and the manifest records retry reasons.
The comparison refuses an incomplete MYSTIC manifest unless `--allow-partial` is supplied.
A comparison over an incomplete manifest or an unmerged shard, or one that lacks some of the requested engine settings,
is marked `PARTIAL`; a complete MYSTIC record with no engine setting joined yet is `WAITING_ENGINE`.
The published-format CSV needs the complete MYSTIC manifest and the complete A0 comparison.

For distributed execution, prepare identical inputs once, copy that prepared working directory into a separate
directory per shard, and run one scheduler per working directory.
For N shards, use `--shard 1/N` through `--shard N/N` on N separate copies.
All shards must use identical configuration, optical input, generated input files, executable, data files and script
contents.
Merge them after all shards have finished:

```sh
"$PYTHON" -B "$H/merge_shards.py" --work "$WORK" --shards "$SHARD1" "$SHARD2" --output "$MERGED"
```

List the working directory of every shard after `--shards`; the example has two.

Use `$MERGED` as the working directory for baking, engine extraction, comparison and CSV generation.
Passing `--arms A0,A1,A2` enables the additional engine settings for baking, extraction and comparison;
`--arms-file` names another file of engine settings in the format of `arms.json`. Extraction accepts only the bundled
definitions of `path_n` and `L` (A0 12 and 16, A1 24 and 16, A2 12 and 24) and stops with `E_ARM_DEFINITION` on any
other pair after the bake. Preparation and extraction also require Anderson tolerance 0.001, the 80 km top and ground
convolution order 24 (which the bake script sets), and extraction checks the wavelength, view and solar axes against
the optical export.

## Small run

This example uses two states, one pixel, one solar zenith angle, two seeds and 1,000 photons per run.
The output is a small sample that ends with the comparison; it is not a replacement for the full CSV.
`make_csv.py` and `summarize.py` accept only the full configuration.

```sh
SMALL="$RUN_ROOT/small-work"
CFG="$H/config_small.json"
"$PYTHON" -B "$H/export_optics.py" --work "$SMALL" --config "$CFG"
"$PYTHON" -B "$H/inputs.py" --work "$SMALL" --config "$CFG" --lrt-dir "$LRT"
"$PYTHON" -B "$H/run_mystic.py" --work "$SMALL" --config "$CFG" --lrt-dir "$LRT" --jobs "$(nproc)"
SOSFUSE2_DEVICE_PROFILE=cuda PYTHON="$PYTHON" bash "$H/bake_engine.sh" "$SMALL" --config "$CFG" \
  --states clear_43,clear_80
SOSFUSE2_DETERMINISTIC=1 SOSFUSE2_GROUND_SH_L=24 "$PYTHON" -B "$H/extract_engine.py" --work "$SMALL" --config \
  "$CFG" --states clear_43,clear_80
"$PYTHON" -B "$H/compare.py" --work "$SMALL" --config "$CFG"
```

Use `inputs.py --no-disort` to generate inputs without libRadtran.
With that option, `disort.json` explicitly records that the check was skipped, and MYSTIC execution rejects the inputs.
`"$PYTHON" -m pytest "$H/test_harness.py"` (needs pytest) checks the scheduling, input text, statistics and state
selection offline.

## Outputs and interpretation

Paths are relative to the working directory.

- `states.json`, its parent `states_v1.json` and `settings.json` record the explicit input states and solver settings.
- `out/optics.json` contains shared optical inputs and their identities; `out/tau_ledger.json` checks integrated
  optical depths.
- `inputs/` contains generated molecular and aerosol files; `out/layer_ledger.json`, `out/phase_checks.json`,
  `out/disort.json` and `out/input_sha.json` record input checks.
- `runs/` holds per-attempt inputs, logs and results; `out/full.csv`, `out/manifest.json` and `out/plan_full.json`
  describe the MYSTIC runs.
- `engine/` contains settings, configuration records, tables and metadata; `out/engine_pixels.A0.json` contains extracted spectra
  and hashes.
- `out/compare.json`, `out/compare_long.csv`, `out/engine.csv`, `out/engine_ratios.csv` and `out/manifest_compare.json`
  contain estimates, comparisons and identities.
- `out/summary.json` summarizes the comparisons; the CSV created by `make_csv.py` contains the plotted values.

If the inputs or geometry differ from the intended configuration, stop and resolve the mismatch before comparing the
outputs.
For engine table comparisons, use the relative tolerance `1e-3` at the selected radiance values.
For a new MYSTIC run, interpret differences using its Monte Carlo standard errors.
A new estimate is not required to fall inside the recorded 95% interval.
The published CSV is a record; individual seed values are not distributed, so its means and intervals cannot be
recomputed from that CSV alone.
