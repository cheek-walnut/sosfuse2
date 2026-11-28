# Your first change

Run the example from the checkout root in a virtual environment with Python 3.12 or newer.

```sh
python -m pip install -e ".[test]"
python -m sosfuse2 walkthrough
```

1. In **Input (observed)**, find the state and its single 550 nm wavelength. In **Three boundaries
   (observed)**, follow the 40 cells into the source field `(40, 163, 1)`. Check that it converged,
   then find the radiance table `(2, 4, 6, 1)` and the sample views at both solar elevations.
2. Find the feature you want to change in the [task table](AGENTS.md#where-to-start). Open the file listed under
   **Open first** and run that row's **Run after editing** command once from the checkout root. To run offline,
   set `export SOSFUSE2_OFFLINE=1` first. The walkthrough's production-runner comparison also requires `zstd` on your
   path.
3. Make your change and run the same command again.
4. Run the required offline CPU check below. Documentation, settings, output and CPU changes can
   all be developed on a supported machine without a GPU.

| Checks | When and who | What to run |
|---|---|---|
| Required offline CPU | Contributors, by default for ordinary code changes | `SOSFUSE2_OFFLINE=1 python -m pytest -q --pyargs sosfuse2.tests.test_offline_cpu` |
| Changed area | Contributors, before and after editing settings, previews, caches, optical coefficients or another feature | Use the command in the [task table](AGENTS.md#where-to-start). |
| GPU and full grid | Maintainers, before merging changes affecting CPU/GPU agreement, accelerated paths, runtime or memory | Run the relevant [GPU and full-grid checks](docs/CHECKS.md) on the target hardware. |

The required check solves an analytic exp8/triangle/keep atmosphere in an empty work root and compares
convergence, table axes and shape against fixed expectations. Its eight radiance samples must match within a relative
tolerance of `1e-3` and zero absolute tolerance. A separate test verifies that the default density reports
`E_IPRT_OFFLINE` when its inputs are missing. These are software regression checks; physical validation of the
default atmosphere uses [separate evidence](docs/VALIDATION.md).

The required command must collect tests, and every selected check must pass; skips, expected failures and unexpected
passes all fail the command. Keep reference outputs fixed when refactoring.

## Worked example

For the preview PNG save option, start with **Observation table and previews** in the [task table](AGENTS.md#where-to-start).
Suppose the product should use PNG compression level 3. In
`src/sosfuse2/tests/test_preview_png.py::test_table_preview_compression`, add
`assert levels == [3, 3]` immediately after `expected = render('product')`.
This checks the option Pillow received for the −3° and −6° images.
Run the test from the repository root with the test extra installed.

```sh
SOSFUSE2_OFFLINE=1 python -m pytest -q src/sosfuse2/tests/test_preview_png.py
```

With the current default, the assertion reports `[6, 6]`. Change
`PREVIEW_PNG_LEVEL` from 6 to 3 in `src/sosfuse2/outputs/pack_state.py` and
rerun the same command. Its `render_functions()` passes the product level to
`preview_functions.save_png(..., compress_level=...)` for both the packer and
the state runner. To expose a new Pillow option, add a keyword argument to
`save_png()` in `src/sosfuse2/outputs/preview_functions.py` and forward it there.

The test renders the same three-band table at the shared product default of 6,
overrides it to 1 for the packer, then overrides it to 9 for the runner. While
PNG bytes and compressed sizes may change, both images must keep their decoded
pixels, dimensions, 8-bit RGB mode and sRGB rendering intent of 0. The table
must keep every array name, dtype, shape and value. Preserve the product
filenames and metadata record keys.
