"""Decoded preview pixels, PNG metadata and source tables for both product entry points."""

import json

import numpy as np
from PIL import Image

from sosfuse2.jobs import run_state
from sosfuse2.outputs import pack_state, preview_functions


def test_reference_preview(tmp_path):
    pano = np.arange(4 * 8 * 3, dtype=np.float32).reshape(4, 8, 3)
    pano = (pano % [7, 11, 5] + 1).astype(np.float32)
    funcs = pack_state.render_functions()
    projected = funcs['project'](pano, funcs['projection'](6, 4, 100, 0))
    _, exposure = funcs['central'](projected)
    path = tmp_path / 'preview.png'
    funcs['save_png'](path, funcs['tone'](projected, exposure))
    # Decoded RGB values from the original product renderer on this panorama.
    expected = np.array([
        [[122, 84, 87], [114, 107, 83], [127, 77, 61], [82, 117, 106], [94, 140, 83], [74, 153, 52]],
        [[112, 70, 92], [95, 93, 114], [85, 113, 118], [114, 130, 100], [102, 144, 77], [79, 141, 78]],
        [[103, 108, 88], [117, 123, 105], [107, 135, 104], [71, 146, 79], [58, 142, 80], [79, 120, 102]],
        [[11, 11, 8], [7, 8, 5], [4, 6, 3], [2, 8, 2], [2, 13, 3], [5, 16, 9]],
    ], dtype=np.uint8)
    with Image.open(path) as image:
        assert (image.size, image.mode, image.info) == ((6, 4), 'RGB', {'srgb': 0})
        np.testing.assert_array_equal(np.asarray(image), expected)


def test_imported_preview_exposure():
    pack_state.render_functions()['Y'][:] = [1.0, 0.0, 0.0]
    rgb = np.full((8, 8, 3), [1.0, 2.0, 3.0])
    key, exposure = preview_functions.central(rgb)
    assert np.isclose(key, 0.2126 + 2 * 0.7152 + 3 * 0.0722)
    assert np.isclose(preview_functions.tone(rgb, exposure)[0, 0] @ preview_functions.Y, 0.18)


def test_renderer_luminance_is_local():
    rgb = np.full((8, 8, 3), [1.0, 2.0, 3.0])
    first = pack_state.render_functions()
    first['Y'][:] = [1.0, 0.0, 0.0]
    second = pack_state.render_functions()
    # Luminance and first encoded pixel from the original renderer for each set of weights.
    for funcs, luminance, pixel in ((first, 1.0, [118, 162, 194]), (second, 1.8596, [88, 122, 147])):
        key, exposure = funcs['central'](rgb)
        assert np.isclose(key, luminance)
        np.testing.assert_array_equal(funcs['encode'](funcs['tone'](rgb, exposure))[0, 0], pixel)


def test_save_png_accepts_compression_and_seed(tmp_path):
    rgb = np.full((8, 8, 3), 0.18)
    path = tmp_path / 'preview.png'
    preview_functions.save_png(path, rgb, 23, compress_level=1)
    with Image.open(path) as image:
        np.testing.assert_array_equal(np.asarray(image), preview_functions.encode(rgb, 23))


def test_table_preview_compression(tmp_path, monkeypatch):
    radiance = np.arange(2 * 4 * 6 * 3, dtype=np.float32).reshape(2, 4, 6, 3) % 11
    table = dict(
        radiance=radiance,
        solar_elevation_deg=np.array([-3, -6], dtype=np.float32),
        wavelength_nm=np.array([450, 530, 620], dtype=np.float32),
        view_elevation_deg=np.linspace(80, 10, 4, dtype=np.float32),
        relative_azimuth_deg=np.linspace(15, 165, 6, dtype=np.float32),
    )
    meta = {'contract': 'synthetic preview table'}
    levels = []
    save = Image.Image.save

    def record_save(image, path, **options):
        levels.append(options['compress_level'])
        return save(image, path, **options)

    monkeypatch.setattr(Image.Image, 'save', record_save)

    def render(case):
        out = tmp_path / case
        out.mkdir()
        np.savez(out / 'table.npz', **table)
        (out / 'meta.json').write_text(json.dumps(meta))
        pack_state.previews(out)
        pixels = []
        for name in ('preview_m03.png', 'preview_m06.png'):
            with Image.open(out / name) as image:
                assert (image.size, image.mode, image.info) == ((960, 540), 'RGB', {'srgb': 0})
                pixels.append(np.array(image))
        with np.load(out / 'table.npz') as stored:
            assert set(stored.files) == set(table)
            for key, value in table.items():
                np.testing.assert_array_equal(stored[key], value, strict=True)
        assert json.loads((out / 'meta.json').read_text()) == meta
        return pixels

    product_level = pack_state.PREVIEW_PNG_LEVEL
    expected = render('product')
    monkeypatch.setattr(pack_state, 'PREVIEW_PNG_LEVEL', 1)
    np.testing.assert_array_equal(render('level1'), expected)
    # Install the production runner's override for this three-band table.
    monkeypatch.setattr(pack_state, 'previews', pack_state.previews)
    run_state.enable_spectral_previews(pack_state)
    monkeypatch.setattr(pack_state, 'PREVIEW_PNG_LEVEL', 9)
    np.testing.assert_array_equal(render('runner'), expected)
    assert levels == [product_level, product_level, 1, 1, 9, 9]
