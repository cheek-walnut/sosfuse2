"""Preview weights (host.pack_state, used by both run_gpu_state_v3 and the batch path) for non-15-band tables: each band counts in proportion to its width.

CPU only, no table. ① equal widths reproduce the 1.3.1 15/len(wl) scale (to rounding); ② on the BM45 product edges
(35 bands of 11.75 nm, 10 of 5.875 nm) smooth spectra give the 15-band RGB within 2 %, while the 1.3.1 count scale was
off by about 30 % in G (flat 1.306, lambda^-4 1.375); ③ bin edges come from meta settings, tables
without a spectral block fall back to equal widths, edges that do not bracket the wavelengths are E_PREVIEW_EDGES.
"""

import numpy as np
import pytest

from sosfuse2 import paths, spectral
from sosfuse2.host import pack_state as V3

BASE = np.load(paths.ASSETS / 'rgb_weights.npy')
CENTRES15 = 360 + (np.arange(15) + 0.5) * 470 / 15
EDGES = np.array(spectral.PRODUCT_EDGES)
WL = (EDGES[:-1] + EDGES[1:]) / 2
SPECTRA = {
    'flat': lambda x: np.ones_like(x),
    'rayleigh': lambda x: (x / 550.0) ** -4,
    'red': lambda x: np.exp((x - 830) / 120),
}


def test_equal_widths_match_count_scale():
    wl = 360 + (np.arange(45) + 0.5) * 470 / 45
    old = np.column_stack([np.interp(wl, CENTRES15, BASE[:, k]) for k in range(3)]) * 15 / 45
    np.testing.assert_allclose(V3.preview_weights(wl, np.full(45, 470 / 45), BASE), old, rtol=1e-14, atol=0)


@pytest.mark.parametrize('name', sorted(SPECTRA))
def test_product_edges_follow_band_width(name):
    spectrum = SPECTRA[name]
    reference = spectrum(CENTRES15) @ BASE
    width = spectrum(WL) @ V3.preview_weights(WL, np.diff(EDGES), BASE)
    count = spectrum(WL) @ V3.preview_weights(WL, np.full(len(WL), 470 / len(WL)), BASE)
    np.testing.assert_allclose(width / reference, 1, atol=0.02)
    assert np.abs(count / reference - 1).max() > 0.1  # the 1.3.1 scale this replaces


def test_band_widths_from_meta():
    meta = {'settings': {'spectral': {'bin_edges_nm': EDGES.tolist()}}}
    np.testing.assert_array_equal(V3.preview_band_widths(meta, WL.astype('f4')), np.diff(EDGES))
    np.testing.assert_array_equal(V3.preview_band_widths({'settings': {'spectral': None}}, WL), np.full(45, 470 / 45))
    with pytest.raises(RuntimeError, match='E_PREVIEW_EDGES'):
        V3.preview_band_widths({'settings': {'spectral': {'bin_edges_nm': EDGES[:-1].tolist()}}}, WL)
