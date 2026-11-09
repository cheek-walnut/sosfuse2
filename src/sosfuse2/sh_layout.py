"""SH transport layout; L32 renders the unchanged historical translation unit."""

from pathlib import Path
import re


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


def _replace(code, old, new, n):
    """str.replace of exactly n occurrences. A kernel.cu edit that moves or duplicates a rewrite target must stop here
    (E_SH_LAYOUT_DRIFT) instead of compiling text the rewrite silently missed; update n only together with that edit."""
    found = code.count(old)
    if found != n:
        raise RuntimeError(f'E_SH_LAYOUT_DRIFT: {old!r} occurs {found} times, expected {n}')
    return code.replace(old, new)


def _replace_word(code, old, new, n):
    """re.sub of the whole-word integer literal old, exactly n occurrences (as _replace)."""
    code, found = re.subn(r'\b' + str(old) + r'\b', str(new), code)
    if found != n:
        raise RuntimeError(f'E_SH_LAYOUT_DRIFT: literal {old} occurs {found} times, expected {n}')
    return code


def kernel_source(L=32, combine=False, code=None):
    """kernel.cu for SH order L, optionally with the combined SH plane (combine_source). code: the kernel text to
    rewrite instead of kernel.cu (tests)."""
    K = count(L)
    if code is None:
        code = Path(__file__).with_name('kernel.cu').read_text()
    if L == 32:
        return combine_source(code, L) if combine else code
    # Warp width and single-scattering Mie expansion remain 32 and L32.
    # Only the transported SH expansion is truncated.
    code = '#define SH_L ' + str(L) + '\n' + code
    code = _replace(code, '#define K 561', '#define K ((SH_L+1)*(SH_L+2)/2)', 1)
    code = _replace(code, '#define KPAD 576', '#define KPAD (((K+31)/32)*32)', 1)
    for old, new, n in ((1132, 10 + 2 * K, 7), (1131, 9 + 2 * K, 1), (571, 10 + K, 5), (570, 9 + K, 1), (565, 4 + K, 5)):
        code = _replace_word(code, old, new, n)
    # l>=2 loops are the physical Mie phase; preserve their full expansion.
    code = _replace(code, 'a<=32', 'a<=SH_L', 1)
    # Keep cosine tables at 33 entries for the warp reduction.
    code = _replace(code, 'm=lane;m<=32', 'm=lane;m<=SH_L', 2)
    code = _replace(code, 'if(m<32)', 'if(m<SH_L)', 2)
    code = _replace(code, 'l=m+2;l<=32', 'l=m+2;l<=SH_L', 2)
    code = _replace(code, 'l=0;l<=32', 'l=0;l<=SH_L', 1)
    # No m=32 lane tail exists for L<32. Remove its loads, including y[560].
    begin = code.index('       if(lane==0){\n#ifndef SKIP_ACTIVE', code.index('// Degree-major loads'))
    end = code.index('\n       tags[corner]=cc;', begin)
    code = (
        code[:begin]
        + '       if(lane==0){\n        for(int w=0;w<TW;w++)s.c32[corner][w]=0.f;\n#ifdef STRATO\n        for(int w=0;w<TW;w++)s.st_c32[corner][w]=0.f;\n#endif\n       }'
        + code[end:]
    )
    # The smaller basis must not shrink aliased first-light ground scratch.
    for old, new, n in (('nwarps*K', 'nwarps*YPAD', 1), ('Y+warp*K', 'Y+warp*YPAD', 2), ('Y[wp*K+sk]', 'Y[wp*YPAD+sk]', 1)):
        code = _replace(code, old, new, n)
    code = '#define YPAD ' + str(workspace(L)) + '\n' + code
    return combine_source(code, L) if combine else code


def _without_if_block(code, start, replacement=''):
    depth = 0
    pos = start
    for line in code[start:].splitlines(keepends=True):
        directive = line.lstrip()
        if re.match(r'#\s*if(?:def|ndef)?\b', directive):
            depth += 1
        elif re.match(r'#\s*endif\b', directive):
            depth -= 1
            if depth == 0:
                return code[:start] + replacement + code[pos + len(line) :]
        pos += len(line)
    raise RuntimeError('E_SH_COMBINE_SOURCE: unmatched preprocessor block')


def combine_source(code, L=32):
    """Warm C3 path: the first SH plane already contains both density factors. L: the SH order kernel_source already
    applied to code (L<32 has removed the m=32 tail, and with it one marker and one sk_a test)."""
    tail = L == 32
    marker = '#ifdef STRATO\n#if (defined(STRATO) && !defined(LUT_ONLY))\n'
    begin = code.index('// Degree-major loads')
    found = code[begin:].count(marker)
    if found != (2 if tail else 1):
        raise RuntimeError(f'E_SH_LAYOUT_DRIFT: strato contraction marker occurs {found} times after the SH loads')
    pos = code.index(marker, begin)
    code = _without_if_block(code, pos)  # remove the second contraction, not extinction/Mie
    if tail:  # L32's separate m=32 tail
        pos = code.index(marker, begin)
        code = _without_if_block(
            code, pos, '#ifdef STRATO\n        for(int w=0;w<TW;w++)s.st_c32[corner][w]=0.f;\n#endif\n'
        )
    code = _replace(code, 'RS2_REDUCE(scache,s.st_c32,own_saero)', 'own_saero=0.f;', 1)
    code = _replace(code, 'if(sk_a)', 'if(sk_a || sk_s)', 3 if tail else 2)
    code = _replace(code, 'if(!first)aero*=ha[ir];', '', 3)
    return code


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
