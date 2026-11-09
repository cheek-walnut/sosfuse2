"""SH transport layout; L32 renders the unchanged historical translation unit."""

from pathlib import Path


def count(L=32):
    if type(L) is not int or L not in (16, 24, 32):
        raise ValueError('E_SH_L: expected 16,24,32')
    return (L + 1) * (L + 2) // 2


def padded(L=32):
    return ((count(L) + 31) // 32) * 32


def workspace(L=32):
    # GROUND_FUSED borrows the per-warp Y row for 162 floats.
    return max(count(L), 162)


def source_size(s):
    return 10 + count(getattr(getattr(s, 'kernel', None), 'L', 32)) * (
        2 if getattr(s, 'strato', None) is not None else 1
    )


def kernel_source(L=32, combine=False):
    """kernel.cu for SH order L, optionally with the combined SH plane. Both choices are preprocessor symbols of kernel.cu
    (SH_L, SH_COMBINE); this only prepends their definitions, so the L32 unit without the combined plane is kernel.cu."""
    count(L)  # E_SH_L
    code = Path(__file__).with_name('kernel.cu').read_text()
    head = (f'#define SH_L {L}\n' if L != 32 else '') + ('#define SH_COMBINE\n' if combine else '')
    return head + code


def combined_coefficients(src, c, L=32, xp=None, skip_thin=False):
    """FP32, separate multiply/add operations; never mutate the solver source."""
    import numpy as np

    xp = np if xp is None else xp
    K = count(L)
    if src.ndim != 3 or src.shape[1] != 10 + 2 * K:
        raise ValueError('E_SH_COMBINE_SHAPE')
    ha = xp.asarray(c['ha'], dtype='f4')
    hs = xp.asarray(c['strato_density'], dtype='f4')
    if skip_thin and len(ha) <= 64:  # kernel.cu SKIP_ACTIVE masks cover 64 heights
        cutoff = xp.float32(1e-7) * xp.asarray(c['hr'], dtype='f4')
        ha = xp.where(ha > cutoff, ha, xp.float32(0))
        hs = xp.where(hs > cutoff, hs, xp.float32(0))
    nc = len(c['chi'])
    a = src[:, 10 : 10 + K] * xp.repeat(ha, nc)[:, None, None]
    b = src[:, 10 + K :] * xp.repeat(hs, nc)[:, None, None]
    return a + b


def combined_aux(src, c, L=32, xp=None, skip_thin=False):
    import numpy as np

    xp = np if xp is None else xp
    values = combined_coefficients(src, c, L, xp, skip_thin)
    cells, K, waves = values.shape
    out = xp.zeros((cells, (waves + 1) // 2, padded(L), 2), 'f4')
    out[:, :, :K, 0] = xp.transpose(values[:, :, ::2], (0, 2, 1))
    if waves > 1:
        out[:, : waves // 2, :K, 1] = xp.transpose(values[:, :, 1::2], (0, 2, 1))
    return out
