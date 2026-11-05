"""FP32 stratosphere sidecar and legacy gate metadata selector."""
import os
import numpy as np
from .sh_layout import count, padded

def strato_sh_aux(src, xp=np, pair=False, L=32):
    """Append STRATO SH to the private SH sidecar (FP32).

    plane 0: src[:,10:571,:]; plane 1: src[:,571:1132,:].
    The complete plane 0 retains its original layout, not a per-cell interleave.
    Kernel source-cell count nh*nc must equal src.shape[0] (wrapper checks this).
    """
    K=count(L); KPAD=padded(L)
    if src.ndim != 3 or src.shape[1] != 10+2*K or src.dtype != np.dtype('f4'):
        raise ValueError('E_STRATO_SH_SOURCE: FP32 [cell,1132,W] required')
    cells, _, waves = src.shape
    if waves not in (1, 15):
        raise ValueError('E_STRATO_SH_WAVES: W must be 1 or 15')
    if pair:
        out = xp.zeros((2*cells, (waves+1)//2, KPAD, 2), dtype='f4')
        for plane, start in enumerate((10, 10+K)):
            target = out[plane*cells:(plane+1)*cells]
            target[:, :, :K, 0] = xp.transpose(src[:, start:start+K, 0::2], (0, 2, 1))
            if waves > 1:
                target[:, :waves//2, :K, 1] = xp.transpose(src[:, start:start+K, 1::2], (0, 2, 1))
    else:
        out = xp.zeros((2*cells, waves, KPAD), dtype='f4')
        for plane, start in enumerate((10, 10+K)):
            out[plane*cells:(plane+1)*cells, :, :K] = xp.transpose(src[:, start:start+K], (0, 2, 1))
    if not out.flags.c_contiguous:
        raise RuntimeError('E_STRATO_SH_LAYOUT')
    return out


def _async_enabled(kernel):
    """Legacy gate metadata only; the removed round executor no longer consumes this.

    SOSFUSE2_STRATO_ASYNC '1'/'0' wins; unset = device profile strato_async (rig only)."""
    text = os.environ.get('SOSFUSE2_STRATO_ASYNC', '').strip()
    if text not in ('', '0', '1'):
        raise ValueError('E_STRATO_CONTROL: invalid SOSFUSE2_STRATO_ASYNC (0 or 1)')
    if text:
        return text == '1'
    profile = kernel.device()[1] if hasattr(kernel, 'device') else {}
    return bool(profile.get('strato_async'))
