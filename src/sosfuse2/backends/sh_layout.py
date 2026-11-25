"""SH transport layout; L32 renders the unchanged historical translation unit."""

from dataclasses import dataclass
from pathlib import Path
from sosfuse2.backends import vram_trace


def count(L=32):
    if type(L) is not int or L not in (16, 24, 32):
        raise ValueError('E_SH_L: expected 16,24,32')
    return (L + 1) * (L + 2) // 2


def padded(L=32):
    return ((count(L) + 31) // 32) * 32


def workspace(L=32):
    # GROUND_FUSED borrows the per-warp Y row for 162 floats.
    return max(count(L), 162)


@dataclass(frozen=True)
class FastTransport:
    """Compile-time role and layout. Device capacity and input values are selected per call."""

    L: int
    W: int
    role: str
    hg: bool = False
    stage_plane: bool = True

    BATCH = 12
    PAGE_CELLS = 128
    HG_BINS = 64
    HG_PAGE_CELLS = 64
    THREADS = 128
    PROJ_MOMENTS = THREADS // 2  # two wavelength groups per projected moment
    PAIR_LANES = 8  # one lane per pair in the wide producer

    @property
    def visitor_tile(self):
        return self.THREADS // self.PAIR_LANES // (1 if self.stage_plane else 2)

    @classmethod
    def select(cls, L, cfg, first, phase, boundary=False, ray_limit=False, split=False):
        role = 'none'
        if cfg.role == 'transport' and not cfg.strato and not split:
            if not first and not cfg.split_warm:
                role = 'warm'
            elif first and phase == 0 and ray_limit and not boundary:
                role = 'first'
        return cls(L, cfg.W, role, hg=role == 'first' and 'TROPO_MIE' not in cfg.flags)

    @property
    def stride(self):
        # Pairs of adjacent wavelengths, including the final zero pad.
        return 2 * ((self.W + 1) // 2)

    @property
    def bins(self):
        # Half-period of the highest harmonic sampled at 10/3 intervals per order.
        return (10 * self.L + 2) // 3

    @property
    def sh_nodes(self):
        return self.bins + 3  # cubic basis support at the end knots

    @property
    def hg_nodes(self):
        return self.HG_BINS + 5  # six-point angular stencil

    @property
    def direct_shared(self):
        return count(self.L) * self.stride * 4 if self.role == 'direct' and self.stage_plane else 0

    def defines(self):
        if self.role == 'none':
            return ()
        options = [f'-DFAST_WAVE_STRIDE={self.stride}', f'-DFAST_BATCH={self.BATCH}']
        if self.role == 'first':
            options.append('-DFAST_FIRST')
            if self.hg:
                options += [f'-DHG_ANGLE_BINS={self.HG_BINS}', f'-DFAST_THREADS={self.THREADS}']
        else:
            options.append('-DFAST_WARM')
            if self.role in ('direct', 'angular'):
                options += ['-DFAST_VPROD_ON', f'-DFAST_THREADS={self.THREADS}']
            if self.role == 'direct':
                options += [f'-DVP_STAGE_PLANE={int(self.stage_plane)}', f'-DVP_VISITOR_TILE={self.visitor_tile}',
                            f'-DVP_PAIR_LANES={self.PAIR_LANES}']
            if self.role == 'angular':
                options += [f'-DSH_ANGLE_BINS={self.bins}', f'-DSH_PAGE_CELLS={self.PAGE_CELLS}']
        return tuple(options)

    def shared_bytes(self, warps):
        """Match Transport::bind_shared, including padding before the aligned point arrays."""
        if self.role == 'warm':
            c32 = 4 * self.W if self.L == 32 else 0
            floats = workspace(self.L) + 4 * self.W + 33 + 33 + c32 + 15
            return self.W * 8 + 12 * 4 + warps * (floats * 4 + self.BATCH * 144) + 16 + 512
        if self.role == 'first':
            # FirstPoint, midpoint profiles and panel widths, phase, ray constants, frame.
            per_warp = self.BATCH * 176 + (self.BATCH + 1) * 24 + (32 + 15 + 12) * 4
            return self.W * 8 + 16 + warps * per_warp
        return 0


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
    if skip_thin and len(ha) <= 128:  # Same height range as the transport's two-word live-layer mask and sos_vprod.
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
    with vram_trace.item('sh.combine_scratch'):
        values = combined_coefficients(src, c, L, xp, skip_thin)
    cells, K, waves = values.shape
    out = xp.zeros((cells, (waves + 1) // 2, padded(L), 2), 'f4')
    out[:, :, :K, 0] = xp.transpose(values[:, :, ::2], (0, 2, 1))
    if waves > 1:
        out[:, : waves // 2, :K, 1] = xp.transpose(values[:, :, 1::2], (0, 2, 1))
    return out
