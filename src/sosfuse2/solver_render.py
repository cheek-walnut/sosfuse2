"""Solver definitions loaded into per-call scopes by host.pack_state.render_functions.

The loader executes only the top-level def/class nodes, in a fresh scope per state (filled by host.pack_state). The imports and
placeholders below never run there; they name the injected globals for static analysis (ruff, pyright)."""

import math
from typing import Any

import numpy as np
from PIL import Image, PngImagePlugin

Y: Any = None  # Rec. 709 luminance weights (host.pack_state)


def film(x):
    """Hable rational shoulder, asymptotic maximum .98; zero stays zero."""
    a, b, c, d, e, f = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    return 0.98 / (1 - e / f) * ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f) - e / f)


def tone(rgb, k, headroom=1.0):
    # One scalar for all three channels: no white balance, hue rotation, or saturation boost.
    x = np.maximum(rgb * (k / headroom), 0)
    maximum = x.max(-1)
    scale = np.divide(film(maximum), maximum, out=np.zeros_like(maximum), where=maximum > 0)
    return np.maximum(x * scale[..., None] * headroom, 0)


def encode(rgb, seed=17):
    v = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - 0.055)
    # Shared-channel, zero-mean dither immediately before 8-bit quantization.
    rng = np.random.default_rng(seed)
    dither = rng.uniform(-0.50, 0.50, (*rgb.shape[:-1], 1)).astype(np.float32)
    return np.uint8(np.clip(np.rint(v * 255 + dither), 0, 255))


def save_png(path, rgb, seed=17):
    path.parent.mkdir(parents=True, exist_ok=True)
    info = PngImagePlugin.PngInfo()
    info.add(b'sRGB', b'\x00')
    Image.fromarray(encode(rgb, seed)).save(path, compress_level=9, pnginfo=info)


def projection(w, h, hfov, azimuth):
    """Gnomonic pinhole; horizontal field of view at the image centre; horizon .8H.

    Uses the centre/right/up construction in lut_mockup.desk_dirs, with the
    specified field of view and analytically chosen camera pitch.
    """
    focal = w / (2 * math.tan(math.radians(hfov / 2)))
    pitch = math.atan(0.30 * h / focal)
    a = math.radians(azimuth)
    centre = np.array([math.cos(pitch) * math.sin(a), math.cos(pitch) * math.cos(a), math.sin(pitch)])
    right = np.array([math.cos(a), -math.sin(a), 0.0])
    up = np.array([-math.sin(pitch) * math.sin(a), -math.sin(pitch) * math.cos(a), math.cos(pitch)])
    xx, yy = np.meshgrid(
        (np.arange(w, dtype=np.float32) + 0.5 - w / 2) / focal, (h / 2 - np.arange(h, dtype=np.float32) - 0.5) / focal
    )
    p = centre.astype(np.float32) + xx[..., None] * right.astype(np.float32) + yy[..., None] * up.astype(np.float32)
    az = np.degrees(np.arctan2(p[..., 0], p[..., 1]))
    alt = np.degrees(np.arctan2(p[..., 2], np.hypot(p[..., 0], p[..., 1])))
    return az, alt


def cubic_weights(t):
    return (
        -0.5 * t + t * t - 0.5 * t**3,
        1 - 2.5 * t * t + 1.5 * t**3,
        0.5 * t + 2 * t * t - 1.5 * t**3,
        -0.5 * t * t + 0.5 * t**3,
    )


def sample(pano, az, alt):
    """Catmull-Rom bicubic, periodic azimuth, clamped altitude; bound overshoots.

    Supports the original 100x360 SOS and app_pano's 180x720 grids.
    """
    nr, nc, channels = pano.shape
    out = np.empty((*az.shape, channels), np.float32)
    for start in range(0, az.shape[0], 64):
        sl = slice(start, start + 64)
        rr = np.clip((90 - np.maximum(alt[sl], 0)) * nr / 90 - 0.5, 0, nr - 1)
        cc = (az[sl] + 180) * nc / 360 - 0.5
        ri, ci = np.floor(rr).astype(np.int32), np.floor(cc).astype(np.int32)
        wr, wc = cubic_weights(rr - ri), cubic_weights(cc - ci)
        val = np.zeros((*rr.shape, channels), np.float32)
        lower = np.full_like(val, np.inf)
        upper = np.full_like(val, -np.inf)
        for ir in range(4):
            r = np.clip(ri + ir - 1, 0, nr - 1)
            for ic in range(4):
                q = pano[r, (ci + ic - 1) % nc]
                val += q * (wr[ir] * wc[ic])[..., None]
                if ir in (1, 2) and ic in (1, 2):
                    np.minimum(lower, q, out=lower)
                    np.maximum(upper, q, out=upper)
        out[sl] = np.clip(val, lower, upper)
    return out


def project(pano, coords):
    az, alt = coords
    rgb = sample(pano, az, alt)
    # Identical horizon colour on each side of alt=0; quickly fading plain ground.
    below = np.minimum(alt, 0)
    fade = 0.001 + 0.999 * np.exp(below / 1.0)
    rgb *= fade[..., None]
    return rgb


def central(rgb):
    h, w = rgb.shape[:2]
    region = rgb[round(0.25 * h) : round(0.55 * h), round(0.40 * w) : round(0.60 * w)]
    lum = region @ Y
    key = float(np.median(lum))
    if key <= 0:
        positive = rgb[(rgb @ Y) > 0]
        if not len(positive):
            raise ValueError('E_PREVIEW_LUMINANCE: positive luminance required')
        region = positive
        key = float(np.median(region @ Y))
    representative = np.median(region.reshape(-1, 3), axis=0)
    representative *= key / max(float(representative @ Y), 1e-30)
    lo, hi = 0.0, 10.0 / key
    for _ in range(60):
        mid = (lo + hi) / 2
        if float(tone(representative, mid) @ Y) < 0.18:
            lo = mid
        else:
            hi = mid
    return key, (lo + hi) / 2
