"""Single interpretation of physical, wavelength and execution-role selectors.

Legacy optimization names are accepted only as warned no-ops. Both the host and
kernel select the former C3 domain from STRATO; its four FW15-only choices are
inactive at FW1. No stratosphere-off expansion is implied.
"""
from dataclasses import dataclass
import warnings
import os


def deterministic():
    """Single parser for the optional deterministic accumulation contract."""
    value = os.environ.get("SOSFUSE2_DETERMINISTIC", "0")
    if value not in ("0", "1"):
        raise ValueError("E_DETERMINISTIC: SOSFUSE2_DETERMINISTIC must be 0 or 1")
    return value == "1"


KERNEL_FLAGS = ('STRATO', 'TROPO_MIE', 'AEROSOL_2L', 'EXP_SECTION_CLOCKS', 'WARM_RAYSPLIT')
RETIRED_FLAGS = frozenset(('ACC_F32SUM', 'ACC_GLOBAL', 'BS192', 'CARVE_MIN', 'CELL_SPLIT2', 'CELL_SPLIT4', 'DIAG_REGPT', 'DIAG_SH2', 'FL_FASTDIV', 'FL_FMAD', 'FL_GHEAVY', 'FL_GHEAVY_ON', 'FL_GPOINT', 'FL_GSPLIT', 'FL_GSPLIT_ON', 'FL_LB2', 'FL_RAYSPLIT', 'FMAD_WARM', 'GCACHE', 'GC_BS128', 'GC_BS512', 'GC_CARVE', 'GC_DNOFILL', 'GC_DSH0', 'GC_FU2', 'GC_LB1', 'GC_LB3', 'GC_LB4', 'GC_LB5', 'GC_LB6', 'GC_MINB', 'GC_ON', 'GROUND_FUSED', 'GROUND_PHASE_BATCH', 'GROUND_PREFETCH', 'LANE_ATTEN', 'LANE_GROUND', 'LANE_SOURCE_INIT', 'LM_LUT', 'LPT_ORDER', 'PF_L1', 'PF_L2', 'PHASE1_TRACE3', 'PHASE2_NOBARRIER', 'POINT_BATCH2', 'POINT_NOINLINE', 'PROFILE', 'PTREC_READ', 'PTREC_WRITE', 'PT_BATCH', 'PT_NB12', 'PT_NB4', 'PT_NB6', 'RAY_CONST_SHARED', 'RAY_SORT', 'REG_LEAN', 'SH_AUX', 'SH_AUX_FP16', 'SH_AUX_PAIR', 'SH_COMBINE', 'SKIP_THIN', 'SOSFUSE2_FP32_ACC', 'SOSFUSE2_FP32_GEOM', 'SPECIALIZE', 'STRATO_LANE_FIRSTVAL', 'STRATO_LANE_GROUND', 'STRATO_MIE_LUT', 'STRATO_MIE_LUT_FINE', 'STRATO_MIE_RECIP', 'STRATO_ROUND_CHUNKS', 'STRATO_SH_AUX', 'TRANSPORT_RS2', 'TRANSPORT_WAVE_LANES', 'TROPO_PREFETCH', 'WARM_HALF_ON', 'WARM_LANE_DATA', 'WARM_LB2', 'WARM_PLANAR_ON', 'WARM_SH_HALF', 'WARM_SH_PLANAR', 'WAVE_TILE3', 'WAVE_TILE5', 'WAVE_TILE5_APPLY', 'WAVE_TILED'))
STAGE1_FLAGS = ('FMAD', 'FASTMATH')
KSPLIT_VARIANTS = dict(record=('SINV', 'RECORD'), prod=('SINV', 'PROD'), slim=('SINV', 'SLIM'),
                       slimray=('SINV', 'SLIM', 'WARM_RAYSPLIT', 'SLIM_RAYS'),
                       slimdump=('SINV', 'SLIM', 'SLIM_RAYDUMP'))


SUPPORTED_CONFIGS = {
    'transport': {'wavelengths': (1, 15), 'ksplit': (0, 1, 2, 3)},
    'instrumentation': {'wavelengths': (1, 15), 'ksplit': (0,)},
    'lut': {'wavelengths': (1, 15), 'ksplit': (0,)},
}


@dataclass(frozen=True)
class KernelConfig:
    role: str
    flags: tuple[str, ...]
    W: int

    @property
    def strato(self): return 'STRATO' in self.flags
    @property
    def c3_fw15(self): return self.strato and self.W == 15
    @property
    def split_first(self): return self.role == 'transport'
    @property
    def split_warm(self): return 'WARM_RAYSPLIT' in self.flags
    @property
    def instrumented(self): return self.role == 'instrumentation'
    @property
    def ksplit_supported(self): return self.c3_fw15 and self.split_first and not self.split_warm
    @property
    def ground_sh_supported(self): return self.strato and self.split_first and 'TROPO_MIE' not in self.flags
    @property
    def defines(self): return self.flags


def resolve_config(flags=(), *, role=None, W=15):
    flags = set(flags)
    retired = flags & RETIRED_FLAGS
    if retired:
        warnings.warn('W_KERNEL_FLAG_RETIRED: ignored '+','.join(sorted(retired)), UserWarning, stacklevel=2)
    flags -= retired
    bad = flags - set(KERNEL_FLAGS)
    if bad:
        raise ValueError(f'E_KERNEL_FLAG: unsupported flags {sorted(bad)}')
    role = role or ('instrumentation' if 'EXP_SECTION_CLOCKS' in flags else 'transport')
    if role not in SUPPORTED_CONFIGS:
        raise ValueError(f'E_KERNEL_ROLE: {role}')
    if W not in SUPPORTED_CONFIGS[role]['wavelengths']:
        raise ValueError('E_WAVELENGTHS: kernel specializes to 1 or 15 wavelengths')
    if role == 'lut':
        flags = {'LUT_ONLY', 'STRATO'}
    elif role == 'instrumentation':
        flags.add('EXP_SECTION_CLOCKS')
        flags.discard('WARM_RAYSPLIT')
    return KernelConfig(role, tuple(sorted(flags)), W)
