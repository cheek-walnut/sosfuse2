// Fixed component order: air, tropo_aerosol, o3, no2, o4 (atmosphere/components.py).
#define NCOMP 5
// All current accumulations are deterministic; add nondeterministic atomics only under #if
// !SOSFUSE2_DETERMINISTIC.
#ifdef SLIM
#define SLIM_ON 1
#else
#define SLIM_ON 0
#endif
// Split transport (SOSFUSE2_KSPLIT, solve/ksplit.py): PROD (producer) keeps geometry, tag cache, SH
// contraction and m-reduction, writes own_aero per corner, and compiles out the
// tensor/log/attenuation/Kahan/boundary/projection and moment/flux epilogue (its outputs are not
// meaningful); SLIM (consumer) reads own_aero instead (see SINV below).
#ifdef PROD
#define PROD_ON 1
#else
#define PROD_ON 0
#endif
// RECORD (light column record pass, SINV mode 1): no SH contraction (as SLIM), no
// tensor/boundary/projection/epilogue (as PROD); only geometry, the Simpson prefix and the column
// write. Same column values as the full kernel's mode 1.
#ifdef RECORD
#define REC_ON 1
#else
#define REC_ON 0
#endif
#define NOSH_ON (SLIM_ON || REC_ON)
// The SLIM consumer reads the PROD aerosol of the combined SH plane and has no stratospheric
// contraction of its own (source_solver.Kernel.raw_kernel compiles every warm STRATO FW15 unit with
// SH_COMBINE).
#if SLIM_ON && defined(STRATO) && !defined(SH_COMBINE)
#error E_SLIM_REQUIRES_SH_COMBINE
#endif
// SLIM_RAYS only publishes ray_rads; sos_slim_proj projects them (source_solver.Kernel.slim_rays in
// solve/ksplit.py builds it with ray limits).
#if defined(SLIM_RAYS) && !defined(LAUNCH_RAY_LIMIT)
#error E_SLIM_RAYS_REQUIRES_RAY_LIMIT
#endif
#define NOOUT_ON (PROD_ON || REC_ON)
// NVRTC, CUDA 12. FW is specialized to 1 or 15. Fast math and FMA contraction only through the
// stage-1 options (source_solver.Kernel: --fmad, --prec-div/--prec-sqrt and SOSFUSE2_FASTMATH
// below); the default build has neither. One destination cell (and optional spectral tile)/block,
// one ray/warp. No floating-point atomics: the only atomics are integer counters (ray queue, cell
// and split completion) that hand out work; every sum keeps a fixed order. NVRTC: math builtins
// available without headers
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef long long int64_t;
typedef int int32_t;
typedef unsigned char uint8_t;
// Stage-1 FASTMATH (source_solver.Kernel stage1): intrinsic exp/sin/cos for the whole unit. Defined
// before the first device function so the math mode does not depend on where a function sits in
// this file.
#ifdef SOSFUSE2_FASTMATH
#define expf __expf
#define sinf __sinf
#define cosf __cosf
#endif
#ifndef FW
#define FW 15
#endif
// Aerosol record row stride: FW padded to a multiple of 4 floats (16 for FW 15) so VPROD stores one
// float4 per thread fills 32 B sectors. The pad lane is written as 0 by VPROD,
// never written by PROD and never read.
#define AERO_W (((FW + 3) / 4) * 4)
// SLIM consumer on the RS2 path (the only one that reads SLIM_AERO): every corner ends in
// __syncwarp, so the compiler cannot hoist the next corner's global reads and a step waits for 4+
// serial L2 round trips.
// SLIM_PRELOAD issues a step's PROD aerosol, cell source rows and column record before the corner
// loop; the arithmetic and its order are unchanged.
#if defined(SINV) && defined(SLIM) && defined(STRATO) && FW == 15
#define SLIM_PRELOAD
#endif
// Phase-0 first light: point() and the Simpson
// nodes take the segment ends and quadrature nodes as DS computed once (segment) or held per lane
// (nodes, nq<=32) instead of converting the same doubles every step; the DS values equal ds() of
// them, so results are unchanged. Warm kernels do the same: the ray's p/d DS cache, segment-end DS
// and lane-held node DS feed the batched point()/point_m2 and the Simpson prefix.
#if !K_FIRST || K_PHASE == 0
#define PATH_DS
#endif
// Transported SH order: 16, 24 or 32 (sh_layout.kernel_source defines it below 32). The warp width,
// the m=32 cosine table entry and the single-scattering Mie expansion stay at 32 for every order.
#ifndef SH_L
#define SH_L 32
#endif
#if SH_L != 16 && SH_L != 24 && SH_L != 32
#error E_SH_L: SH_L must be 16, 24 or 32
#endif
#define K ((SH_L + 1) * (SH_L + 2) / 2)
// Rows per cell of src (source_solver.Kernel source layout): 10 + K, and 10 + 2K with the
// stratospheric SH plane; 4 + K moments.
#define SRC_ROWS (10 + K)
#define SRC_ROWS_ST (10 + 2 * K)
#define NMOM (4 + K)
// Per-warp Y row: first-light GROUND_FUSED borrows it for 162 floats (sh_layout.workspace).
#define YPAD (K > 162 ? K : 162)
#define PI 3.14159265358979323846
// Smallest normal float (FLT_MIN). Traces below it are excluded from log-space moments; NaN fails
// >= as well. The trace itself compares the same under FTZ, but subnormal terms inside it do not:
// CuPy always builds with -ftz=true, so numpy_reference.py runs transport under flush_to_zero() to
// match.
#define TRACE_MIN 1.17549435e-38f
// Transport defaults are isolated from the historical LUT translation unit.
// Specialized transport always uses paired SH sidecars, lane attenuation/ground,
// wavelength-owned transport and the existing accumulation order. The former C3
// domain is STRATO; FW1 omits its four FW15-only optimizations. No nostrato expansion.
#if !K_FIRST
#define WARM_LANE_ON
#endif
// First-light/ray-split moments, and FW15 warm moments, live in global rows.
#if (K_FIRST || defined(WARM_RAYSPLIT)) && K_PHASE == 0
#define ACC_IN_GLOBAL
#endif
#if (FW == 15 || defined(FAST_WARM)) && !K_FIRST && K_PHASE == 0
#define ACC_IN_GLOBAL
#endif
// Batched point geometry: lane=step, the original point arithmetic and warp copies.
#if defined(STRATO) && !K_FIRST
#define PT_ACTIVE
#endif
// FastTransport supplies the active role, strides and batch size. The defaults
// are used only to parse inactive template members in other translation units.
#ifndef FAST_WAVE_STRIDE
#define FAST_WAVE_STRIDE (2 * ((FW + 1) / 2))
#endif
#ifndef FAST_BATCH
#define FAST_BATCH 1
#endif
#ifdef FAST_WARM
#define PT_ACTIVE
#endif
#if K_FIRST && K_PHASE == 0
#define SOS_LAUNCH_BOUNDS __launch_bounds__(256, 2)
#endif
#ifndef SKIP_TAU
#define SKIP_TAU 1e-7f
#endif
#if defined(PT_ACTIVE) && !(defined(WARM_RAYSPLIT) && K_PHASE == 0)
#define RS_ACTIVE
#endif
// Longest cells first; FW15 warm also uses the two-part cell reduction.
#if !K_FIRST && !(defined(WARM_RAYSPLIT) && K_PHASE == 0)
#define LPT_ACTIVE
#endif
#if defined(LPT_ACTIVE) && FW == 15 && (defined(STRATO) || !defined(EXP_SECTION_CLOCKS))
#define SPLIT_ACTIVE
#endif
#define SPLIT_S 2
// First-light phases 0/2 share the existing tropo+strato ground loop.
#if defined(STRATO) && K_FIRST && (K_PHASE == 0 || K_PHASE == 2)
#define GF_ON
#endif
// Instrumentation uses unsplit transport; warm split is an internal consumer role.
#if defined(WARM_RAYSPLIT) && defined(EXP_SECTION_CLOCKS)
#error E_FL_RAYSPLIT_REQUIRES_ACC_GLOBAL_NO_CLOCKS
#endif
#if !defined(EXP_SECTION_CLOCKS) && K_FIRST && K_PHASE == 0
#define FL_SPLIT_ON
#elif defined(WARM_RAYSPLIT) && !K_FIRST && K_PHASE == 0
#define FL_SPLIT_ON
#endif
#ifndef SOS_LAUNCH_BOUNDS
#define SOS_LAUNCH_BOUNDS
#endif
#ifndef GFG
#define GFG 8
#endif
#ifdef EXP_SECTION_CLOCKS
#define SECTION_PARAM , unsigned long long *section_clocks
#define SECTION_BEGIN()                                                                            \
    if (lane == 0) {                                                                               \
        section_begin = clock64();                                                                 \
    }
#define SECTION_BEGIN_IF(cond)                                                                     \
    if (lane == 0 && (cond)) {                                                                     \
        section_begin = clock64();                                                                 \
    }
#define SECTION_END(k)                                                                             \
    if (lane == 0) {                                                                               \
        section_cycles[k] += clock64() - section_begin;                                            \
    }
#define SECTION_END_IF(k, cond)                                                                    \
    if (lane == 0 && (cond)) {                                                                     \
        section_cycles[k] += clock64() - section_begin;                                            \
    }
#define SECTION_SYNC(op)                                                                           \
    do {                                                                                           \
        unsigned long long barrier_begin = 0;                                                      \
        if (lane == 0)                                                                             \
            barrier_begin = clock64();                                                             \
        op;                                                                                        \
        if (lane == 0)                                                                             \
            section_cycles[6] += clock64() - barrier_begin;                                        \
    } while (0)
#else
#define SECTION_PARAM
#define SECTION_BEGIN()
#define SECTION_BEGIN_IF(cond)
#define SECTION_END(k)
#define SECTION_END_IF(k, cond)
#define SECTION_SYNC(op) op
#endif
#define KPAD (((K + 31) / 32) * 32)
#ifndef LUT_ONLY
struct LMTable {
    unsigned char l[K], m[K];
    constexpr LMTable() : l(), m() {
        int k = 0;
        for (int a = 0; a <= SH_L; a++)
            for (int b = 0; b <= a; b++) {
                l[k] = (unsigned char)a;
                m[k] = (unsigned char)b;
                k++;
            }
    }
};
__constant__ LMTable LM = LMTable();
#endif
// Opt-in spectral tiles: local buffers use TW; global arrays keep FW.
#define TW FW
#define GW(w) w
// PATH_DS kernels (phase-0 first light, warm) reuse exact two-word ray coordinates.
#ifdef PATH_DS
#define RAY_CONST_F 27
#else
#define RAY_CONST_F 15
#endif
// ---- Double-single arithmetic ----

// Compile-time capabilities; inactive fields remain valid arguments without changing
// the layout of the shared point records.
struct ProfilePolicy {
#ifdef STRATO
    static constexpr bool strato = true;
#else
    static constexpr bool strato = false;
#endif
#ifdef AEROSOL_2L
    static constexpr bool two_layer = true;
#else
    static constexpr bool two_layer = false;
#endif
#ifdef TROPO_MIE
    static constexpr bool tropo_mie = true;
#else
    static constexpr bool tropo_mie = false;
#endif
};

struct DS {
    float h, l;
};
__device__ DS ds(double x) {
    float h = (float)x;
    return {h, (float)(x - (double)h)};
}
__device__ DS ds(DS x) {
    return x;
}
__device__ DS dn(DS a) {
    return {-a.h, -a.l};
}
__device__ DS da(DS a, DS b) {
    float s = a.h + b.h, v = s - a.h;
    float e = ((a.h - (s - v)) + (b.h - v)) + (a.l + b.l), h = s + e;
    return {h, e - (h - s)};
}
__device__ DS dx(DS a, DS b) {
    return da(a, dn(b));
}
__device__ DS dm(DS a, DS b) {
    float p = a.h * b.h, e = fmaf(a.h, b.h, -p);
    e = (e + a.h * b.l) + a.l * b.h;
    float h = p + e;
    return {h, e - (h - p)};
}
__device__ DS dr(DS a) {
    float h = sqrtf(fmaxf(a.h, 0.f));
    DS e = dx(a, dm({h, 0}, {h, 0}));
    float l = (e.h + e.l) / fmaxf(2 * h, 1e-30f);
    return da({h, 0}, {l, 0});
}
__device__ DS ddiv(DS a, DS b) {
    float h = a.h / b.h;
    DS e = dx(a, dm({h, 0}, b));
    float lo = (e.h + e.l) / b.h;
    return da({h, 0}, {lo, 0});
}
__device__ float dq(DS a, DS b) {
    DS x = ddiv(a, b);
    return x.h + x.l;
}
__device__ float dv(DS a) {
    return a.h + a.l;
}
// FTZ: a solar-boundary DS value as an FTZ FP32 build
// computes it, on every build: components and sum below FLT_MIN count as 0; NaN passes
// (fabsf(NaN)<TRACE_MIN is false).
__device__ float ftz_dv(DS a) {
    float h = fabsf(a.h) < TRACE_MIN ? 0.f : a.h, l = fabsf(a.l) < TRACE_MIN ? 0.f : a.l, v = h + l;
    return fabsf(v) < TRACE_MIN ? 0.f : v;
}
__device__ float clip(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}
__device__ int search(const float *a, int n, float x) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int m = (lo + hi) / 2;
        if (a[m] <= x)
            lo = m + 1;
        else
            hi = m;
    }
    return max(0, min(n - 2, lo - 1));
}
// The host supplies the boundary and the corresponding uniform PROFILE grid.
#ifndef PROFILE_N
#define PROFILE_N 161
#endif
#ifndef PROFILE_INV
#define PROFILE_INV 2.f
#endif
#ifndef PROFILE_LAST
#define PROFILE_LAST 159
#endif
#ifndef TOP_KM
#define TOP_KM 80.f
#endif
#ifndef AIR_END_KM
#define AIR_END_KM TOP_KM
#endif
#ifndef LOWER_END_KM
#define LOWER_END_KM TOP_KM
#endif
#if PROFILE_LAST != PROFILE_N - 2
#error                                                                                             \
    "PROFILE_N, PROFILE_INV, PROFILE_LAST and TOP_KM must be defined together (PROFILE_LAST = PROFILE_N - 2)"
#endif
// Component support is closed at its upper endpoint. The default domain keeps
// its historical clipped-profile behavior; extended domains have finite support.
__device__ __forceinline__ bool support_extended() {
    return TOP_KM > LOWER_END_KM;
}
__device__ __forceinline__ float support_end(bool air) {
    return air ? AIR_END_KM : LOWER_END_KM;
}
__device__ __forceinline__ bool outside_support(float h, bool air) {
    return support_extended() && h > support_end(air);
}
__device__ __forceinline__ float support_value(float h, float value, bool air) {
    return outside_support(h, air) ? 0.f : value;
}
// Round a boundary sample toward the segment interior, within one FP32 ULP.
__device__ __forceinline__ float support_height(float h, float inside_h) {
    if (!support_extended())
        return h;
    for (int k = 0; k < 2; ++k) {
        const float end = support_end(k == 1);
        if (h >= nextafterf(end, end - 1.f) && h <= nextafterf(end, end + 1.f))
            h = inside_h > end ? nextafterf(end, end + 1.f) : end;
    }
    return h;
}
__device__ __forceinline__ float profile_coordinate(float h) {
    float x = clip(h, 0.f, TOP_KM) * PROFILE_INV;
    return x;
}
__device__ float profile_value(float h, const float *table, int k) {
    if (outside_support(h, k == 0))
        return 0.f;
    float x = profile_coordinate(h);
    int i = min((int)x, PROFILE_LAST);
    float f = x - (float)i;
    return table[i * 3 + k] + f * (table[(i + 1) * 3 + k] - table[i * 3 + k]);
}
// Two-layer tropospheric aerosol (atmosphere/tropo_profile.py, opt-in by state; without
// -DAEROSOL_2L nothing here exists and s.prof[1] stays expf(-s.h/height)). P = [bl_top-w/2, 1/w,
// a_bl, k_bl, a_ft, k_ft, bl_top] follows sun_h[0..nsh) in the same device allocation
// (source_solver.Kernel.constants_device), so the kernel signature and launch argument lists do not
// change. Smoothstep blend of the boundary layer into the free troposphere over the width w.
__device__ __forceinline__ float aerosol_two_layer(float h, const float *P) {
    float x = clip((h - P[0]) * P[1], 0.f, 1.f), s = x * x * (3.f - 2.f * x);
    return (P[2] * expf(-h * P[3])) * (1.f - s) +
           (P[4] * expf(-(fmaxf(h, P[0]) - P[6]) * P[5])) * s;
}
__device__ int ix(int l, int m) {
    return l * (l + 1) / 2 + m;
}
__device__ float hg(float u, float g) {
    float den = fmaxf(1e-20f, (1.f + g * g) - (2.f * g) * u);
    return (1.f - g * g) / ((float)(4 * PI) * (den * sqrtf(den)));
}
#define STRATO_LUT_BINS 16384
// GPU nodes use exactly the old recurrence, even in a LUT-enabled module.
__device__ float strato_phase_node(float u, int w, const float *chi) {
    float p0 = 1.f, p1 = u, v = p0 * chi[w];
    v += p1 * chi[FW + w];
    for (int l = 2; l <= 32; l++) {
        float pn = ((float)(2 * l - 1) * u * p1 - (float)(l - 1) * p0) / (float)l;
        v += pn * chi[l * FW + w];
        p0 = p1;
        p1 = pn;
    }
    return v / (float)(4 * PI);
}
#ifdef STRATO
extern "C" __global__ void strato_lut_build(float *chi) {
    int k = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (k < (STRATO_LUT_BINS + 1) * FW) {
        int i = k / FW, w = k % FW;
        float u = -1.f + (float)i * (2.f / (float)STRATO_LUT_BINS);
        chi[33 * FW + k] = strato_phase_node(u, w, chi);
    }
}
#endif
__device__ float mie_phase(float u, int w, const float *chi) {
    float x = (clip(u, -1.f, 1.f) + 1.f) * (float)(STRATO_LUT_BINS / 2);
    int i = min((int)x, STRATO_LUT_BINS - 1);
    float f = x - (float)i;
    float a = chi[(33 + i) * FW + w], b = chi[(34 + i) * FW + w];
    return a + f * (b - a);
}
__device__ void kah(float &t, float &c, float v) {
    float y = v - c, z = t + y;
    c = (z - t) - y;
    t = z;
}
// LUT_ONLY (source_solver.Kernel.strato_lut_kernel, jobs/precompile.py) compiles strato_lut_build
// above and nothing below. tests/test_strato_precision.py compiles the STRATO block up to kah() on
// its own: keep this boundary below kah.
#ifdef LUT_ONLY
#if defined(GROUND_SH_L)
#error E_GROUND_SH_PATH: the LUT_ONLY unit has no ground path
#endif
#if defined(TROPO_MIE)
#error E_TROPO_MIE_UNSUPPORTED_GROUND_PATH: the LUT_ONLY unit has no ground path
#endif
#else
struct PointData {
    float T[16], cm[33], sw[4], prof[NCOMP], h, ds, cs;
    float mu, exd, eyd, sc[NCOMP];
    int cell[4], ir, lit;
#ifdef STRATO
    float st_prof, st_sc;
#endif
};
struct Sample : PointData {
#ifdef STRATO
    float st_c32[4][TW];
#endif
    float val[TW][4], logs[TW];
    uint32_t positive;
    float c32[4][TW], att[TW];
};
static_assert(sizeof(PointData) == 300
#ifdef STRATO
                                       + 8
#endif
              ,
              "E_POINTDATA_LAYOUT");
static_assert(sizeof(Sample) == 304 + 40 * TW
#ifdef STRATO
                                    + 8 + 16 * TW
#endif
              ,
              "E_SAMPLE_LAYOUT");
// Copy only fields actually written by point(). Inactive first-light fields
// are deliberately not read; persistent c32/cache/val/log/att never move.
template <class Record, class Point>
__device__ __forceinline__ void copy_point(Record &s, const Point &q, bool first) {
    for (int k = 0; k < 16; k++)
        s.T[k] = q.T[k];
    for (int k = 0; k < 33; k++)
        s.cm[k] = q.cm[k];
    for (int k = 0; k < 4; k++) {
        s.sw[k] = q.sw[k];
        s.cell[k] = q.cell[k];
    }
    for (int k = 0; k < NCOMP; k++)
        s.prof[k] = q.prof[k];
    s.h = q.h;
    s.ds = q.ds;
    s.cs = q.cs;
    s.ir = q.ir;
    if (first) {
        s.mu = q.mu;
        s.exd = q.exd;
        s.eyd = q.eyd;
        s.lit = q.lit;
        for (int k = 0; k < NCOMP; k++)
            s.sc[k] = q.sc[k];
    }
    if constexpr (ProfilePolicy::strato) {
        s.st_prof = q.st_prof;
        if (first)
            s.st_sc = q.st_sc;
    }
}
__device__ void sh_basis(double mu, int lane, float *y, const double *aa, const double *bb,
                         const double *dd) {
    mu = fmin(1., fmax(-1., mu));
    double st = sqrt(fmax(0., 1 - mu * mu));
    for (int m = lane; m <= SH_L; m += 32) {
        double diag = 1 / sqrt(4 * PI);
        for (int j = 1; j <= m; j++)
            diag = (diag * dd[j]) * st;
        double mult = m ? sqrt(2.) : 1., z0 = diag;
        y[ix(m, m)] = (float)(mult * z0);
        if (m < SH_L) {
            double z1 = (sqrt(2. * m + 3) * mu) * z0;
            y[ix(m + 1, m)] = (float)(mult * z1);
            for (int l = m + 2; l <= SH_L; l++) {
                int k = ix(l, m);
                double z2 = (aa[k] * mu) * z1 - bb[k] * z0;
                y[k] = (float)(mult * z2);
                z0 = z1;
                z1 = z2;
            }
        }
    }
}
// sos_vprod_y evaluates sh_basis's recurrence with double-single FMA products.
// aa2/bb2/dd2 hold the high and low words of aa/bb/dd. Keeping both words avoids
// the precision loss of a single-word recurrence near |mu|=1.
__device__ void sh_basis_ds(double mu, int lane, float *y, const float2 *aa2, const float2 *bb2,
                            const float2 *dd2) {
    mu = fmin(1., fmax(-1., mu));
    const DS u = ds(mu), st = ds(sqrt(fmax(0., 1 - mu * mu)));
    for (int m = lane; m <= SH_L; m += 32) {
        DS diag = ds(1 / sqrt(4 * PI));
        for (int j = 1; j <= m; j++)
            diag = dm(dm(diag, DS{dd2[j].x, dd2[j].y}), st);
        const DS mult = m ? ds(sqrt(2.)) : DS{1.f, 0.f};
        DS z0 = diag;
        y[ix(m, m)] = dv(dm(mult, z0));
        if (m < SH_L) {
            DS z1 = dm(dm(ds(sqrt(2. * m + 3)), u), z0);
            y[ix(m + 1, m)] = dv(dm(mult, z1));
            for (int l = m + 2; l <= SH_L; l++) {
                int k = ix(l, m);
                DS z2 = dx(dm(dm(DS{aa2[k].x, aa2[k].y}, u), z1), dm(DS{bb2[k].x, bb2[k].y}, z0));
                y[k] = dv(dm(mult, z2));
                z0 = z1;
                z1 = z2;
            }
        }
    }
}
// The profile is evaluated by one function, profile_at, for
// the quadrature nodes (point()) and the Simpson optical-depth prefix (segment ends, panel
// midpoints); numpy_reference.py profile_at, extinction_profile, SimpsonPrefix use the same
// operations. Path coordinates, panel widths and midpoints are DS, like point().
// ---- Profile evaluation and optical-depth integration ----

struct ProfileParams {
    float height;
    const float *aer2;
    const float *table;
    int mask;
    float air_surface;
    float st_center = 0.f, st_sigma = 1.f, st_norm = 1.f;
};
struct ExtinctionParams {
    ProfileParams profile;
    const float *ext;
    const float *st_ext = nullptr;
};
// Path coordinate of a GL node (u^2 map toward the lower end), DS.
__device__ __forceinline__ DS path_t(DS ad, DS bd, DS length, DS u, DS v, bool lower) {
    return lower ? da(ad, dm(length, dm(u, u))) : dx(bd, dm(length, dm(v, v)));
}
template <class Scalar>
__device__ __forceinline__ DS node_t(Scalar a, Scalar b, bool lower, Scalar gu) {
    DS ad = ds(a), bd = ds(b), u = ds(gu);
    return path_t(ad, bd, dx(bd, ad), u, lower ? u : dx({1, 0}, u), lower);
}
// DS position q = p + t d, q*q and the radius |q|.
template <class Coord>
__device__ __forceinline__ DS path_q(const Coord *p, const Coord *d, DS t, DS *q, DS *q2) {
    for (int k = 0; k < 3; k++) {
        q[k] = da(ds(p[k]), dm(t, ds(d[k])));
        q2[k] = dm(q[k], q[k]);
    }
    return dr(da(da(q2[0], q2[1]), q2[2]));
}
__device__ __forceinline__ float path_h(DS rr) {
    return clip(dv(dx(rr, {6371, 0})), 0, TOP_KM);
}
// Fixed separable components: air, tropospheric aerosol, O3, NO2, O4; stratosphere is separate.
__device__ __forceinline__ void profile_at(float h, const ProfileParams &profile, float *prof,
                                           float &st_prof) {
    const bool ext_lower = outside_support(h, false);
    const bool ext_air = outside_support(h, true);
    float x = profile_coordinate(h);
    int idx = min((int)x, PROFILE_LAST);
    float frac = x - (float)idx;
    if (ext_air) {
        prof[0] = 0.f;
    } else if (profile.mask & 1) {
        float t0 = profile.table[idx * 3];
        prof[0] = t0 + frac * (profile.table[(idx + 1) * 3] - t0);
    } else {
        prof[0] = expf(-h / 8.f);
    }
    if (ext_lower) {
        prof[1] = 0.f;
        prof[2] = 0.f;
        prof[3] = 0.f;
        prof[4] = 0.f;
        st_prof = 0.f;
        return;
    }
    if constexpr (ProfilePolicy::two_layer)
        prof[1] = aerosol_two_layer(h, profile.aer2);
    else
        prof[1] = expf(-h / profile.height);
    if (profile.mask & 2) {
        float t1 = profile.table[idx * 3 + 1];
        prof[2] = t1 + frac * (profile.table[(idx + 1) * 3 + 1] - t1);
    } else {
        prof[2] = fmaxf(0.f, 1.f - fabsf(h - 25.f) / 15.f);
    }
    {
        float t2 = profile.table[idx * 3 + 2];
        prof[3] = t2 + frac * (profile.table[(idx + 1) * 3 + 2] - t2);
    }
    float q = prof[0] / profile.air_surface;
    prof[4] = q * q;
    if constexpr (ProfilePolicy::strato) {
        float st_t = (h - profile.st_center) / profile.st_sigma;
        st_prof = expf(-.5f * (st_t * st_t)) / profile.st_norm;
    } else {
        st_prof = 0.f;
    }
}
// Extinction of wavelength column gw from profile components (sum_k prof_k ext_k, then the
// stratospheric layer).
__device__ __forceinline__ float extinction_at(const float *prof, float st_prof,
                                               const ExtinctionParams &optics, int gw) {
    float b = 0.f;
#pragma unroll
    for (int k = 0; k < NCOMP; k++)
        b += prof[k] * optics.ext[k * FW + gw];
    if constexpr (ProfilePolicy::strato)
        b += st_prof * optics.st_ext[gw];
    return b;
}
template <class Coord>
__device__ __forceinline__ float beta_at(const Coord *p, const Coord *d, DS t,
                                         const ExtinctionParams &optics, int gw) {
    DS q[3], q2[3];
    float pr[NCOMP], pst;
    profile_at(path_h(path_q(p, d, t, q, q2)), optics.profile, pr, pst);
    return extinction_at(pr, pst, optics, gw);
}
// At a support boundary, use the one-sided value belonging to this segment.
template <class Coord>
__device__ __forceinline__ float beta_at_side(const Coord *p, const Coord *d, DS t, DS inside,
                                              const ExtinctionParams &optics, int gw) {
    if (!support_extended())
        return beta_at(p, d, t, optics, gw);
    DS q[3], q2[3];
    const float h = path_h(path_q(p, d, t, q, q2));
    const float ih = path_h(path_q(p, d, inside, q, q2));
    const float side_h = support_height(h, ih);
    float pr[NCOMP], pst;
    profile_at(side_h, optics.profile, pr, pst);
    return extinction_at(pr, pst, optics, gw);
}
// Simpson on the panel [t0,t1] with end extinctions b0,b1: (t1-t0)/6 (b0 + 4 beta(mid) + b1), mid =
// (t0+t1)/2 (exact DS halving).
template <class Coord>
__device__ __forceinline__ float simpson_piece(const Coord *p, const Coord *d, DS t0, DS t1,
                                               float b0, float b1, const ExtinctionParams &optics,
                                               int gw) {
    DS tm = da(t0, t1);
    tm.h *= .5f;
    tm.l *= .5f;
    const float bm = beta_at(p, d, tm, optics, gw);
    return (dv(dx(t1, t0)) / 6.f) * ((b0 + b1) + 4.f * bm);
}
// ---- Quadrature point geometry ----

#define POINT_INLINE __noinline__
// Coord/Scalar = double (every caller but phase-0 first light) or DS: first light passes the ray's
// DS p/d and the segment and node DS values, so the per-step ds() conversions (FP64 adds, the
// remaining first-light FP64 work) are not redone. pf(d[k]) is (float)d[k] = ds(d[k]).h either way;
// the values are identical.
__device__ __forceinline__ float pf(double x) {
    return (float)x;
}
__device__ __forceinline__ float pf(DS x) {
    return x.h;
}
template <class Coord, class Scalar, class Record>
__device__ POINT_INLINE void
point(Record &s, const Coord *p, const Coord *d, const float *B, const float *dl, float st,
      Scalar a, Scalar b, bool lower, Scalar gu, Scalar gw, const float *hgrid, int nh,
      const float *chi, int nc, ProfileParams profile, const float *sun_h, int nsh,
      const float *sun_col, int nsx, bool first, const float *st_sun_col = nullptr) {
    DS ad = ds(a), bd = ds(b), length = dx(bd, ad), u = ds(gu), wu = ds(gw),
       v = lower ? u : dx({1, 0}, u);
    DS t = path_t(ad, bd, length, u, v, lower);
    s.ds = dv(dm(dm(length, v), wu)) * 2.f;
    DS q[3], q2[3];
    DS rr = path_q(p, d, t, q, q2), xy = dr(da(q2[0], q2[1]));
    s.h = path_h(rr);
    float rho = dv(xy);
    float cx = rho > 1e-12f ? dq(q[0], xy) : 1.f, sx = rho > 1e-12f ? dq(q[1], xy) : 0.f;
    s.cs = clip(dq(q[2], rr), -1, 1);
    float angle = atan2f(rho, dv(q[2]));
    int ir = search(hgrid, nh, s.h), j = search(chi, nc, angle);
    s.ir = ir;
    float ah = clip((s.h - hgrid[ir]) / (hgrid[ir + 1] - hgrid[ir]), 0, 1),
          ac = clip((angle - chi[j]) / (chi[j + 1] - chi[j]), 0, 1);
    angle = chi[j] + ac * (chi[j + 1] - chi[j]);
    float sn = sinf(angle), cc = cosf(angle);
    float Q[9] = {-cc * cx, sx, sn * cx, -cc * sx, -cx, sn * sx, sn, 0, cc}, A[9] = {}, dot[3] = {},
          kk[9];
    for (int k = 0; k < 3; k++)
        for (int col = 0; col < 3; col++) {
            dot[col] += pf(d[k]) * Q[3 * k + col];
            for (int row = 0; row < 3; row++)
                A[3 * row + col] += B[3 * k + row] * Q[3 * k + col];
        }
    for (int row = 0; row < 3; row++)
        for (int col = 0; col < 3; col++)
            kk[3 * row + col] = A[3 * row + col] - dl[row] * dot[col];
    const int ii[4] = {0, 1, 2, 0}, jj[4] = {0, 1, 2, 2};
    for (int u2 = 0; u2 < 4; u2++) {
        int i = ii[u2], z = jj[u2];
        s.T[4 * u2] = kk[3 * i] * kk[3 * z];
        s.T[4 * u2 + 1] = kk[3 * i + 1] * kk[3 * z + 1];
        s.T[4 * u2 + 2] = kk[3 * i + 2] * kk[3 * z + 2];
        s.T[4 * u2 + 3] = kk[3 * i] * kk[3 * z + 2] + kk[3 * i + 2] * kk[3 * z];
    }
    float cp = st > 1e-12f ? clip((pf(d[0]) * cx + pf(d[1]) * sx) / st, -1, 1) : 1.f;
    s.cm[0] = 1;
    s.cm[1] = cp;
    for (int m = 2; m <= 32; m++)
        s.cm[m] = (2.f * cp) * s.cm[m - 1] - s.cm[m - 2];
    s.sw[0] = (1.f - ah) * (1.f - ac);
    s.sw[1] = (1.f - ah) * ac;
    s.sw[2] = ah * (1.f - ac);
    s.sw[3] = ah * ac;
    s.cell[0] = ir * nc + j;
    s.cell[1] = s.cell[0] + 1;
    s.cell[2] = s.cell[0] + nc;
    s.cell[3] = s.cell[2] + 1;
    {
        float pst;
        profile_at(s.h, profile, s.prof, pst);
        if constexpr (ProfilePolicy::strato)
            s.st_prof = pst;
    }
    if (first) {
        s.mu = dot[2];
        s.exd = dot[0];
        s.eyd = dot[1];
        DS height = dx(rr, {6371, 0});
        if (ftz_dv(height) <= 0.f)
            height = {0, 0};
        DS rad = da({6371, 0}, height);
        DS mg = dn(dr(ddiv(dm(height, da({2.f * 6371.f, 0}, height)), dm(rad, rad))));
        DS delta = dx(ddiv(q[2], rr), mg);
        if (ftz_dv(delta) == 0.f)
            delta = {0, 0};
        s.lit = dv(delta) >= 0;
        if (!s.lit)
            delta = {0, 0};
        float x = clip(dv(dr(ddiv(delta, dx({1, 0}, mg)))), 0, 1);
        int ih = search(sun_h, nsh, s.h);
        float ar = (s.h - sun_h[ih]) / (sun_h[ih + 1] - sun_h[ih]);
        float f = x * (nsx - 1);
        int ic = max(0, min(nsx - 2, (int)f));
        float br = f - (float)ic;
        for (int k = 0; k < NCOMP; k++) {
            float col = ((1.f - ar) * (1.f - br)) * sun_col[(ih * nsx + ic) * NCOMP + k];
            col += (ar * (1.f - br)) * sun_col[((ih + 1) * nsx + ic) * NCOMP + k];
            col += ((1.f - ar) * br) * sun_col[(ih * nsx + ic + 1) * NCOMP + k];
            col += (ar * br) * sun_col[((ih + 1) * nsx + ic + 1) * NCOMP + k];
            s.sc[k] = col;
        }
        if constexpr (ProfilePolicy::strato) {
            float col = ((1.f - ar) * (1.f - br)) * st_sun_col[ih * nsx + ic];
            col += (ar * (1.f - br)) * st_sun_col[(ih + 1) * nsx + ic];
            col += ((1.f - ar) * br) * st_sun_col[ih * nsx + ic + 1];
            col += (ar * br) * st_sun_col[(ih + 1) * nsx + ic + 1];
            s.st_sc = col;
        }
    }
}

// Compact, float4-aligned sample for batched warm transport.
struct __align__(16) WarmPoint {
    float T[16], sw[4];
    int cell[4];
    float prof[NCOMP];
    int ir;
    float h, ds, cp;
};
static_assert(sizeof(WarmPoint) == 144, "E_WARMPOINT_LAYOUT");

template <class Coord, class Scalar>
__device__ POINT_INLINE DS point_warm(WarmPoint &s, const Coord *p, const Coord *d, const float *B,
                                      const float *dl, float st, Scalar a, Scalar b, bool lower,
                                      Scalar gu, Scalar gw, const float *hgrid, int nh,
                                      const float *chi, int nc, ProfileParams profile) {
    DS ad = ds(a), bd = ds(b), length = dx(bd, ad), u = ds(gu), wu = ds(gw),
       v = lower ? u : dx({1, 0}, u);
    DS t = path_t(ad, bd, length, u, v, lower);
    s.ds = dv(dm(dm(length, v), wu)) * 2.f;
    DS q[3], q2[3];
    DS rr = path_q(p, d, t, q, q2), xy = dr(da(q2[0], q2[1]));
    const float sh = path_h(rr);
    s.h = sh;
    float rho = dv(xy);
    float cx = rho > 1e-12f ? dq(q[0], xy) : 1.f, sx = rho > 1e-12f ? dq(q[1], xy) : 0.f;
    float angle = atan2f(rho, dv(q[2]));
    int ir = search(hgrid, nh, sh), j = search(chi, nc, angle);
    s.ir = ir;
    float ah = clip((sh - hgrid[ir]) / (hgrid[ir + 1] - hgrid[ir]), 0, 1),
          ac = clip((angle - chi[j]) / (chi[j + 1] - chi[j]), 0, 1);
    angle = chi[j] + ac * (chi[j + 1] - chi[j]);
    float sn = sinf(angle), cc = cosf(angle);
    float Q[9] = {-cc * cx, sx, sn * cx, -cc * sx, -cx, sn * sx, sn, 0, cc}, A[9] = {}, dot[3] = {},
          kk[9];
#pragma unroll
    for (int k = 0; k < 3; k++)
#pragma unroll
        for (int col = 0; col < 3; col++) {
            dot[col] += pf(d[k]) * Q[3 * k + col];
#pragma unroll
            for (int row = 0; row < 3; row++)
                A[3 * row + col] += B[3 * k + row] * Q[3 * k + col];
        }
#pragma unroll
    for (int row = 0; row < 3; row++)
#pragma unroll
        for (int col = 0; col < 3; col++)
            kk[3 * row + col] = A[3 * row + col] - dl[row] * dot[col];
    const int ii[4] = {0, 1, 2, 0}, jj[4] = {0, 1, 2, 2};
#pragma unroll
    for (int u2 = 0; u2 < 4; u2++) {
        int i = ii[u2], z = jj[u2];
        s.T[4 * u2] = kk[3 * i] * kk[3 * z];
        s.T[4 * u2 + 1] = kk[3 * i + 1] * kk[3 * z + 1];
        s.T[4 * u2 + 2] = kk[3 * i + 2] * kk[3 * z + 2];
        s.T[4 * u2 + 3] = kk[3 * i] * kk[3 * z + 2] + kk[3 * i + 2] * kk[3 * z];
    }
    s.cp = st > 1e-12f ? clip((pf(d[0]) * cx + pf(d[1]) * sx) / st, -1, 1) : 1.f;
    s.sw[0] = (1.f - ah) * (1.f - ac);
    s.sw[1] = (1.f - ah) * ac;
    s.sw[2] = ah * (1.f - ac);
    s.sw[3] = ah * ac;
    s.cell[0] = ir * nc + j;
    s.cell[1] = s.cell[0] + 1;
    s.cell[2] = s.cell[0] + nc;
    s.cell[3] = s.cell[2] + 1;
    float pst_unused;
    profile_at(sh, profile, s.prof, pst_unused);
    return t;
}

// Compact batched sample for the chunked-VPROD fast slim consumer (128 B, 16-byte aligned).
struct __align__(16) SlimPoint {
    float T[16], sw[4];
    int cell[4];
    float prof[NCOMP];
    int ir;
    float h, ds;
};
static_assert(sizeof(SlimPoint) == 128, "E_SLIMPOINT_LAYOUT");

template <class Coord, class Scalar>
__device__ POINT_INLINE DS point_slim(SlimPoint &s, const Coord *p, const Coord *d, const float *B,
                                      const float *dl, Scalar a, Scalar b, bool lower, Scalar gu,
                                      Scalar gw, const float *hgrid, int nh, const float *chi,
                                      int nc, ProfileParams profile, float st, float *phi_knots, int lane) {
    DS ad = ds(a), bd = ds(b), length = dx(bd, ad), u = ds(gu), wu = ds(gw),
       v = lower ? u : dx({1, 0}, u);
    DS t = path_t(ad, bd, length, u, v, lower);
    s.ds = dv(dm(dm(length, v), wu)) * 2.f;
    DS q[3], q2[3];
    DS rr = path_q(p, d, t, q, q2), xy = dr(da(q2[0], q2[1]));
    const float sh = path_h(rr);
    s.h = sh;
    float rho = dv(xy);
    float cx = rho > 1e-12f ? dq(q[0], xy) : 1.f, sx = rho > 1e-12f ? dq(q[1], xy) : 0.f;
#ifdef SH_ANGLE_BINS
    const float cp = st > 1e-12f ? clip((pf(d[0]) * cx + pf(d[1]) * sx) / st, -1.f, 1.f) : 1.f;
    phi_knots[lane] = acosf(cp) * (SH_ANGLE_BINS / (float)PI);
#endif
    float angle = atan2f(rho, dv(q[2]));
    int ir = search(hgrid, nh, sh), j = search(chi, nc, angle);
    s.ir = ir;
    float ah = clip((sh - hgrid[ir]) / (hgrid[ir + 1] - hgrid[ir]), 0, 1),
          ac = clip((angle - chi[j]) / (chi[j + 1] - chi[j]), 0, 1);
    angle = chi[j] + ac * (chi[j + 1] - chi[j]);
    float sn = sinf(angle), cc = cosf(angle);
    float Q[9] = {-cc * cx, sx, sn * cx, -cc * sx, -cx, sn * sx, sn, 0, cc}, A[9] = {}, dot[3] = {},
          kk[9];
#pragma unroll
    for (int k = 0; k < 3; k++)
#pragma unroll
        for (int col = 0; col < 3; col++) {
            dot[col] += pf(d[k]) * Q[3 * k + col];
#pragma unroll
            for (int row = 0; row < 3; row++)
                A[3 * row + col] += B[3 * k + row] * Q[3 * k + col];
        }
#pragma unroll
    for (int row = 0; row < 3; row++)
#pragma unroll
        for (int col = 0; col < 3; col++)
            kk[3 * row + col] = A[3 * row + col] - dl[row] * dot[col];
    const int ii[4] = {0, 1, 2, 0}, jj[4] = {0, 1, 2, 2};
#pragma unroll
    for (int u2 = 0; u2 < 4; u2++) {
        int i = ii[u2], z = jj[u2];
        s.T[4 * u2] = kk[3 * i] * kk[3 * z];
        s.T[4 * u2 + 1] = kk[3 * i + 1] * kk[3 * z + 1];
        s.T[4 * u2 + 2] = kk[3 * i + 2] * kk[3 * z + 2];
        s.T[4 * u2 + 3] = kk[3 * i] * kk[3 * z + 2] + kk[3 * i + 2] * kk[3 * z];
    }
    s.sw[0] = (1.f - ah) * (1.f - ac);
    s.sw[1] = (1.f - ah) * ac;
    s.sw[2] = ah * (1.f - ac);
    s.sw[3] = ah * ac;
    s.cell[0] = ir * nc + j;
    s.cell[1] = s.cell[0] + 1;
    s.cell[2] = s.cell[0] + nc;
    s.cell[3] = s.cell[2] + 1;
    float pst_unused;
    profile_at(sh, profile, s.prof, pst_unused);
    return t;
}

// Compact batched sample for !STRATO first-light phase 0 (176 B, 16-byte aligned, no dead cm[33]).
struct __align__(16) FirstPoint {
    float T[16], sw[4];
    int cell[4];
    float prof[NCOMP], sc[NCOMP];
    float h, ds, mu, exd, eyd;
    int ir, lit, pad[2];
};
static_assert(sizeof(FirstPoint) == 176, "E_FIRSTPOINT_LAYOUT");

template <class Coord, class Scalar>
__device__ POINT_INLINE DS point_first(FirstPoint &s, const Coord *p, const Coord *d,
                                       const float *B, const float *dl, Scalar a, Scalar b,
                                       bool lower, Scalar gu, Scalar gw, const float *hgrid, int nh,
                                       const float *chi, int nc, ProfileParams profile,
                                       const float *sun_h, int nsh, const float *sun_col, int nsx) {
    DS ad = ds(a), bd = ds(b), length = dx(bd, ad), u = ds(gu), wu = ds(gw),
       v = lower ? u : dx({1, 0}, u);
    DS t = path_t(ad, bd, length, u, v, lower);
    s.ds = dv(dm(dm(length, v), wu)) * 2.f;
    DS q[3], q2[3];
    DS rr = path_q(p, d, t, q, q2), xy = dr(da(q2[0], q2[1]));
    const float sh = path_h(rr);
    s.h = sh;
    float rho = dv(xy);
    float cx = rho > 1e-12f ? dq(q[0], xy) : 1.f, sx = rho > 1e-12f ? dq(q[1], xy) : 0.f;
    float angle = atan2f(rho, dv(q[2]));
    int ir = search(hgrid, nh, sh), j = search(chi, nc, angle);
    s.ir = ir;
    float ah = clip((sh - hgrid[ir]) / (hgrid[ir + 1] - hgrid[ir]), 0, 1),
          ac = clip((angle - chi[j]) / (chi[j + 1] - chi[j]), 0, 1);
    angle = chi[j] + ac * (chi[j + 1] - chi[j]);
    float sn = sinf(angle), cc = cosf(angle);
    float Q[9] = {-cc * cx, sx, sn * cx, -cc * sx, -cx, sn * sx, sn, 0, cc}, A[9] = {}, dot[3] = {},
          kk[9];
#pragma unroll
    for (int k = 0; k < 3; k++)
#pragma unroll
        for (int col = 0; col < 3; col++) {
            dot[col] += pf(d[k]) * Q[3 * k + col];
#pragma unroll
            for (int row = 0; row < 3; row++)
                A[3 * row + col] += B[3 * k + row] * Q[3 * k + col];
        }
#pragma unroll
    for (int row = 0; row < 3; row++)
#pragma unroll
        for (int col = 0; col < 3; col++)
            kk[3 * row + col] = A[3 * row + col] - dl[row] * dot[col];
    const int ii[4] = {0, 1, 2, 0}, jj[4] = {0, 1, 2, 2};
#pragma unroll
    for (int u2 = 0; u2 < 4; u2++) {
        int i = ii[u2], z = jj[u2];
        s.T[4 * u2] = kk[3 * i] * kk[3 * z];
        s.T[4 * u2 + 1] = kk[3 * i + 1] * kk[3 * z + 1];
        s.T[4 * u2 + 2] = kk[3 * i + 2] * kk[3 * z + 2];
        s.T[4 * u2 + 3] = kk[3 * i] * kk[3 * z + 2] + kk[3 * i + 2] * kk[3 * z];
    }
    s.sw[0] = (1.f - ah) * (1.f - ac);
    s.sw[1] = (1.f - ah) * ac;
    s.sw[2] = ah * (1.f - ac);
    s.sw[3] = ah * ac;
    s.cell[0] = ir * nc + j;
    s.cell[1] = s.cell[0] + 1;
    s.cell[2] = s.cell[0] + nc;
    s.cell[3] = s.cell[2] + 1;
    float pst_unused;
    profile_at(sh, profile, s.prof, pst_unused);
    s.mu = dot[2];
    s.exd = dot[0];
    s.eyd = dot[1];
    DS height = dx(rr, {6371, 0});
    if (ftz_dv(height) <= 0.f)
        height = {0, 0};
    DS rad = da({6371, 0}, height);
    DS mg = dn(dr(ddiv(dm(height, da({2.f * 6371.f, 0}, height)), dm(rad, rad))));
    DS delta = dx(ddiv(q[2], rr), mg);
    if (ftz_dv(delta) == 0.f)
        delta = {0, 0};
    s.lit = dv(delta) >= 0;
    if (!s.lit)
        delta = {0, 0};
    float x = clip(dv(dr(ddiv(delta, dx({1, 0}, mg)))), 0, 1);
    int ih = search(sun_h, nsh, sh);
    float ar = (sh - sun_h[ih]) / (sun_h[ih + 1] - sun_h[ih]);
    float f = x * (nsx - 1);
    int ic = max(0, min(nsx - 2, (int)f));
    float br = f - (float)ic;
    const float w00 = (1.f - ar) * (1.f - br), w10 = ar * (1.f - br), w01 = (1.f - ar) * br,
                w11 = ar * br;
    const int i00 = (ih * nsx + ic) * NCOMP, i10 = ((ih + 1) * nsx + ic) * NCOMP,
              i01 = (ih * nsx + ic + 1) * NCOMP, i11 = ((ih + 1) * nsx + ic + 1) * NCOMP;
#pragma unroll
    for (int k = 0; k < NCOMP; k++) {
        float col = w00 * sun_col[i00 + k];
        col += w10 * sun_col[i10 + k];
        col += w01 * sun_col[i01 + k];
        col += w11 * sun_col[i11 + k];
        s.sc[k] = col;
    }
    return t;
}
// Tabulate the original mirrored ground quadrature for a smooth HG phase.
// Evaluate the ghost knots directly, including the reflection at the poles.
#if defined(FAST_FIRST) && defined(HG_ANGLE_BINS) && !defined(TROPO_MIE)
constexpr int HG_NODES = HG_ANGLE_BINS + 5;
extern "C" __global__ void hg_direction_table(const int *owners, const int64_t *off,
                                             const float *dirs, const float *val, float g,
                                             float *table) {
    const int c = blockIdx.x, ti = blockIdx.y;
    constexpr int rows = FAST_THREADS / FAST_WAVE_STRIDE;
    const int pi = blockIdx.z * rows + threadIdx.x / FAST_WAVE_STRIDE, w = threadIdx.x % FAST_WAVE_STRIDE;
    if (pi >= HG_NODES || w >= FW)
        return;
    float st, ct, sp, cp;
    sincosf((ti - 2) * ((float)PI / HG_ANGLE_BINS), &st, &ct);
    sincosf((pi - 2) * ((float)PI / HG_ANGLE_BINS), &sp, &cp);
    const float ex = st * cp, ey = fabsf(st * sp);
    const int cell = owners[c];
    float sum = 0.f;
    for (int64_t j = off[cell]; j < off[cell + 1]; ++j) {
        const float d = ct * dirs[3 * j + 2] + ex * dirs[3 * j];
        const float e = ey * dirs[3 * j + 1];
        const float phase = .5f * (hg(clip(d + e, -1.f, 1.f), g) + hg(clip(d - e, -1.f, 1.f), g));
        sum = fmaf(phase, val[j * FW + w], sum);
    }
    table[((c * HG_NODES + ti) * HG_NODES + pi) * FAST_WAVE_STRIDE + w] = sum;
}

struct HgStencil {
    int knot;
    float w[6];
    __device__ explicit HgStencil(float angle) {
        knot = min(HG_ANGLE_BINS - 1, max(0, (int)angle));
        const float t = clip(angle - knot, 0.f, 1.f);
        // Lagrange weights at the six neighboring knots -2 .. 3.
        const float inv[6] = {-1.f / 120.f, 1.f / 24.f, -1.f / 12.f,
                              1.f / 12.f, -1.f / 24.f, 1.f / 120.f};
#pragma unroll
        for (int j = 0; j < 6; ++j) {
            w[j] = inv[j];
#pragma unroll
            for (int k = 0; k < 6; ++k)
                if (k != j)
                    w[j] *= t + (2 - k);
        }
    }
};

__device__ __forceinline__ float ground_hg_at(const float *const *cells, const FirstPoint &pt, int w) {
    const HgStencil theta(atan2f(hypotf(pt.exd, pt.eyd), pt.mu) * (HG_ANGLE_BINS / (float)PI));
    const HgStencil phi(atan2f(fabsf(pt.eyd), pt.exd) * (HG_ANGLE_BINS / (float)PI));
    float sum = 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const float *table = cells[pt.cell[c]];
        if (table) {
            const float *row = table + (theta.knot * HG_NODES + phi.knot) * FAST_WAVE_STRIDE + w;
            float value = 0.f;
#pragma unroll
            for (int j = 0; j < 6; ++j) {
                float azimuth = 0.f;
#pragma unroll
                for (int i = 0; i < 6; ++i)
                    azimuth = fmaf(phi.w[i], row[(j * HG_NODES + i) * FAST_WAVE_STRIDE], azimuth);
                value = fmaf(theta.w[j], azimuth, value);
            }
            sum = fmaf(pt.sw[c], value, sum);
        }
    }
    return sum;
}
#endif

// and cell as 128-bit shared loads.
struct __align__(16) PointM2 {
    float T[16], sw[4];
    int cell[4], ir;
    float h, ds;
};
static_assert(sizeof(PointM2) == 112, "E_POINTM2_LAYOUT");
// Coord/Scalar as point(): PATH_DS passes the ray's DS p/d, the segment-end DS and the lane's node
// DS.
template <class Coord, class Scalar>
__device__ POINT_INLINE void point_m2(PointM2 &s, const Coord *p, const Coord *d, const float *B,
                                      const float *dl, Scalar a, Scalar b, bool lower, Scalar gu,
                                      Scalar gw, const float *hgrid, int nh, const float *chi,
                                      int nc) {
    DS ad = ds(a), bd = ds(b), length = dx(bd, ad), u = ds(gu), wu = ds(gw),
       v = lower ? u : dx({1, 0}, u);
    DS t = path_t(ad, bd, length, u, v, lower);
    s.ds = dv(dm(dm(length, v), wu)) * 2.f;
    DS q[3], q2[3];
    DS rr = path_q(p, d, t, q, q2), xy = dr(da(q2[0], q2[1]));
    const float sh = path_h(rr);
    s.h = sh;
    float rho = dv(xy);
    float cx = rho > 1e-12f ? dq(q[0], xy) : 1.f, sx = rho > 1e-12f ? dq(q[1], xy) : 0.f;
    float angle = atan2f(rho, dv(q[2]));
    int ir = search(hgrid, nh, sh), j = search(chi, nc, angle);
    s.ir = ir;
    float ah = clip((sh - hgrid[ir]) / (hgrid[ir + 1] - hgrid[ir]), 0, 1),
          ac = clip((angle - chi[j]) / (chi[j + 1] - chi[j]), 0, 1);
    angle = chi[j] + ac * (chi[j + 1] - chi[j]);
    float sn = sinf(angle), cc = cosf(angle);
    float Q[9] = {-cc * cx, sx, sn * cx, -cc * sx, -cx, sn * sx, sn, 0, cc}, A[9] = {}, dot[3] = {},
          kk[9];
    for (int k = 0; k < 3; k++)
        for (int col = 0; col < 3; col++) {
            dot[col] += pf(d[k]) * Q[3 * k + col];
            for (int row = 0; row < 3; row++)
                A[3 * row + col] += B[3 * k + row] * Q[3 * k + col];
        }
    for (int row = 0; row < 3; row++)
        for (int col = 0; col < 3; col++)
            kk[3 * row + col] = A[3 * row + col] - dl[row] * dot[col];
    const int ii[4] = {0, 1, 2, 0}, jj[4] = {0, 1, 2, 2};
    for (int u2 = 0; u2 < 4; u2++) {
        int i = ii[u2], z = jj[u2];
        s.T[4 * u2] = kk[3 * i] * kk[3 * z];
        s.T[4 * u2 + 1] = kk[3 * i + 1] * kk[3 * z + 1];
        s.T[4 * u2 + 2] = kk[3 * i + 2] * kk[3 * z + 2];
        s.T[4 * u2 + 3] = kk[3 * i] * kk[3 * z + 2] + kk[3 * i + 2] * kk[3 * z];
    }
    s.sw[0] = (1.f - ah) * (1.f - ac);
    s.sw[1] = (1.f - ah) * ac;
    s.sw[2] = ah * (1.f - ac);
    s.sw[3] = ah * ac;
    s.cell[0] = ir * nc + j;
    s.cell[1] = s.cell[0] + 1;
    s.cell[2] = s.cell[0] + nc;
    s.cell[3] = s.cell[2] + 1;
}

struct TransportPolicy : ProfilePolicy {
#ifdef SLIM_RAYS
    static constexpr bool slim_rays = true;
#else
    static constexpr bool slim_rays = false;
#endif
#ifdef SH_COMBINE
    static constexpr bool combined_sh = true;
#else
    static constexpr bool combined_sh = false;
#endif
#ifdef SINV
    static constexpr bool column_records = true;
#else
    static constexpr bool column_records = false;
#endif
#ifdef EXP_SECTION_CLOCKS
    static constexpr bool clocks = true;
#else
    static constexpr bool clocks = false;
#endif
#ifdef GROUND_SH_L
    static constexpr bool ground_sh = true;
#else
    static constexpr bool ground_sh = false;
#endif
#ifdef WARM_RAYSPLIT
    static constexpr bool warm_raysplit = true;
#else
    static constexpr bool warm_raysplit = false;
#endif
#ifdef LAUNCH_RAY_LIMIT
    static constexpr bool ray_limit = true;
#else
    static constexpr bool ray_limit = false;
#endif
#ifdef SLIM_RAYDUMP
    static constexpr bool ray_dump = true;
#else
    static constexpr bool ray_dump = false;
#endif
#ifdef FAST_WARM
    static constexpr bool fast_warm = true;
#else
    static constexpr bool fast_warm = false;
#endif
#ifdef FAST_FIRST
    static constexpr bool fast_first = true;
#else
    static constexpr bool fast_first = false;
#endif
    static constexpr bool warm = !K_FIRST;
    static constexpr bool path_ds = warm || K_PHASE == 0;
    static constexpr bool batched_points = (strato || fast_warm) && warm;
    static constexpr bool thin_layers = (strato || fast_warm) && FW == 15 && warm;
    static constexpr bool reduce_scatter = (strato || fast_warm) && FW == 15;
    static constexpr bool ground_fused = strato && K_FIRST && (K_PHASE == 0 || K_PHASE == 2);
    static constexpr bool global_accum =
        (K_FIRST || warm_raysplit || (warm && (fast_warm || FW == 15))) && K_PHASE == 0;
    static constexpr bool split_rays =
        (!clocks && K_FIRST && K_PHASE == 0) || (warm_raysplit && warm && K_PHASE == 0);
    static constexpr bool sort_rays = batched_points && !(warm_raysplit && K_PHASE == 0);
    static constexpr bool queue_cells = warm && !(warm_raysplit && K_PHASE == 0);
    static constexpr bool split_cells = queue_cells && FW == 15 && (strato || !clocks);
    static constexpr bool no_sh = NOSH_ON;
    static constexpr bool no_output = NOOUT_ON;
    static constexpr bool producer = PROD_ON;
    static constexpr bool record = REC_ON;
    static constexpr bool consumer = SLIM_ON;
    static constexpr bool preload = column_records && consumer && (strato && FW == 15);
    static constexpr int batch_size = fast_warm ? FAST_BATCH : 16;
    static constexpr int sample_stride = slim_rays ? 2 : 1;
    static constexpr int source_rows = strato ? SRC_ROWS_ST : SRC_ROWS;
};

// Each lane owns one cached GL node when the quadrature fits in a warp.
#ifdef PATH_DS
using PathScalar = DS;
__device__ __forceinline__ PathScalar path_scalar(double x) {
    return ds(x);
}
#else
using PathScalar = double;
__device__ __forceinline__ PathScalar path_scalar(double x) {
    return x;
}
#endif
struct NodeWeights {
    PathScalar u, w;
};
__device__ __forceinline__ NodeWeights quadrature_node(int node, int nq, bool active,
                                                       const double *gu, const double *gw,
                                                       DS cached_u, DS cached_w) {
#ifdef PATH_DS
    // All lanes must call this before diverging on active.
    if (nq <= 32) {
        return {{__shfl_sync(0xffffffffu, cached_u.h, node),
                 __shfl_sync(0xffffffffu, cached_u.l, node)},
                {__shfl_sync(0xffffffffu, cached_w.h, node),
                 __shfl_sync(0xffffffffu, cached_w.l, node)}};
    }
    if (active)
        return {ds(gu[node]), ds(gw[node])};
#else
    if (active)
        return {gu[node], gw[node]};
#endif
    return {};
}

// Dispatch depends on the record type, without conditional argument lists.
template <class Coord, class Scalar, class Record>
__device__ __forceinline__ void
evaluate_point(Record &s, const Coord *p, const Coord *d, const float *B, const float *dl, float st,
               Scalar a, Scalar b, bool lower, Scalar u, Scalar w, const float *h, int nh,
               const float *chi, int nc, const ProfileParams &profile, const float *sun_h, int nsh,
               const float *sun_col, int nsx, bool first, const float *st_sun_col = nullptr) {
    if constexpr (TransportPolicy::slim_rays && TransportPolicy::batched_points)
        point_m2(s, p, d, B, dl, a, b, lower, u, w, h, nh, chi, nc);
    else
        point(s, p, d, B, dl, st, a, b, lower, u, w, h, nh, chi, nc, profile, sun_h, nsh, sun_col,
              nsx, first, st_sun_col);
}

__device__ __forceinline__ float2 warm_sh_pair(const float *ptr, int64_t cell, int w, int k) {
    const int64_t i = (cell * ((FW + 1) / 2) + w / 2) * KPAD + k;
    return ((const float2 *)ptr)[i];
}

// Fill one corner's Fourier coefficients. Each lane owns m=lane; lane 0 also
// owns m=32. Layer masks are warp uniform and never change the summation order.
template <class Record>
__device__ __forceinline__ void
contract_corner(Record &s, float (&cache)[4][TW], float (&scache)[4][TW], int (&tags)[4],
                const float *y, const float *src_sh, int lane, int corner, int cc, int cells,
                bool live_a, bool live_s) {
    const bool aerosol_live = live_a || (TransportPolicy::combined_sh && live_s);
#pragma unroll
    for (int w = 0; w < TW; ++w)
        cache[corner][w] = 0.f;
    if constexpr (TransportPolicy::strato)
        for (int w = 0; w < TW; ++w)
            scache[corner][w] = 0.f;
    for (int l = 0; l <= SH_L; ++l) {
        if (lane <= l) {
            int k = ix(l, lane);
            float pref = y[k];
            if (aerosol_live) {
#pragma unroll
                for (int w = 0; w < TW; w += 2) {
                    float2 pair = warm_sh_pair(src_sh, cc, w, k);
                    cache[corner][w] += pref * pair.x;
                    if (w + 1 < TW)
                        cache[corner][w + 1] += pref * pair.y;
                }
            }
            if constexpr (TransportPolicy::strato && !TransportPolicy::combined_sh) {
                if (live_s) {
#pragma unroll
                    for (int w = 0; w < TW; w += 2) {
                        float2 pair = warm_sh_pair(src_sh, cells + cc, w, k);
                        scache[corner][w] += pref * pair.x;
                        if (w + 1 < TW)
                            scache[corner][w + 1] += pref * pair.y;
                    }
                }
            }
        }
    }
    if (lane == 0) {
        if constexpr (SH_L == 32) {
            if (aerosol_live) {
#pragma unroll
                for (int w = 0; w < TW; w += 2) {
                    float2 pair = warm_sh_pair(src_sh, cc, w, 560);
                    float v = 0.f;
                    v += y[560] * pair.x;
                    s.c32[corner][w] = v;
                    if (w + 1 < TW) {
                        float v1 = 0.f;
                        v1 += y[560] * pair.y;
                        s.c32[corner][w + 1] = v1;
                    }
                }
            }
            if constexpr (TransportPolicy::strato) {
                if constexpr (TransportPolicy::combined_sh) {
                    for (int w = 0; w < TW; ++w)
                        s.st_c32[corner][w] = 0.f;
                } else if (live_s) {
#pragma unroll
                    for (int w = 0; w < TW; w += 2) {
                        float2 pair = warm_sh_pair(src_sh, cells + cc, w, 560);
                        float v = 0.f;
                        v += y[560] * pair.x;
                        s.st_c32[corner][w] = v;
                        if (w + 1 < TW) {
                            float v1 = 0.f;
                            v1 += y[560] * pair.y;
                            s.st_c32[corner][w + 1] = v1;
                        }
                    }
                }
            }
        } else {
            for (int w = 0; w < TW; ++w)
                s.c32[corner][w] = 0.f;
            if constexpr (TransportPolicy::strato)
                for (int w = 0; w < TW; ++w)
                    s.st_c32[corner][w] = 0.f;
        }
    }
    tags[corner] = cc;
}

// Wavelength-contiguous SH contraction: half-warp 0 owns m0=2p, half-warp 1 owns m1=2p+1,
// lanes 0..14 / 16..30 own wavelengths 0..14 (exact 128-byte coalesced L1 line for k and k+1).
__device__ __forceinline__ void
contract_corner_paired(float (&cache)[4][TW], float *c32_row, int (&tags)[4],
                    const float *y, const float *src_sh, int lane, int corner, int cc) {
    const int hw = lane >> 4;
    const int w = lane & 15;
    const bool w_ok = w < TW;
    const float *base = src_sh + (int64_t)cc * (K * FAST_WAVE_STRIDE);
#pragma unroll 1
    for (int m0 = 0; m0 <= SH_L; m0 += 2) {
        const int m1 = m0 + 1;
        const int my_m = hw ? m1 : m0;
        float acc = 0.f;
        for (int l = m0; l <= SH_L; ++l) {
            if (my_m <= l && w_ok) {
                const int k = ix(l, my_m);
                acc += y[k] * base[(int64_t)k * FAST_WAVE_STRIDE + w];
            }
        }
#pragma unroll
        for (int wi = 0; wi < TW; ++wi) {
            const float v0 = __shfl_sync(0xffffffffu, acc, wi);
            const float v1 = __shfl_sync(0xffffffffu, acc, 16 + wi);
            if constexpr (SH_L == 32) {
                if (m0 == 32) {
                    if (lane == 0)
                        c32_row[corner * TW + wi] = v0;
                } else {
                    if (lane == m0)
                        cache[corner][wi] = v0;
                    if (lane == m1)
                        cache[corner][wi] = v1;
                }
            } else {
                if (lane == m0)
                    cache[corner][wi] = v0;
                if (m1 <= SH_L && lane == m1)
                    cache[corner][wi] = v1;
            }
        }
    }
    tags[corner] = cc;
}

// Every lane participates in the same shuffle tree; one lane owns each wavelength.
__device__ __forceinline__ float reduce_scatter_corner_ptr(const float (&cache)[4][TW],
                                                           const float *c32_corner, const float *cm,
                                                           int corner, int lane) {
    const float cml = lane <= SH_L ? cm[lane] : 0.f;
    float r[8];
    const bool u16 = (lane & 16) != 0;
    _Pragma("unroll") for (int V = 0; V < 8; V++) {
        float xa = 0.f, xb = 0.f;
        if (V < TW) {
            xa = cml * cache[corner][V];
            if constexpr (SH_L == 32) {
                if (lane == 0)
                    xa += cm[32] * c32_corner[V];
            }
        }
        if (V + 8 < TW) {
            xb = cml * cache[corner][V + 8];
            if constexpr (SH_L == 32) {
                if (lane == 0)
                    xb += cm[32] * c32_corner[V + 8];
            }
        }
        const float send = u16 ? xa : xb, keep = u16 ? xb : xa;
        r[V] = keep + __shfl_xor_sync(0xffffffff, send, 16);
    }
    _Pragma("unroll") for (int st = 4; st; st >>= 1) {
        const bool up = (lane & (2 * st)) != 0;
        _Pragma("unroll") for (int V = 0; V < st; V++) {
            const float send = up ? r[V] : r[V + st], keep = up ? r[V + st] : r[V];
            r[V] = keep + __shfl_xor_sync(0xffffffff, send, 2 * st);
        }
    }
    const float fin = r[0] + __shfl_xor_sync(0xffffffff, r[0], 1);
    return __shfl_sync(0xffffffff, fin, (2 * lane) & 31);
}

__device__ __forceinline__ float reduce_scatter_corner(const float (&cache)[4][TW],
                                                       const float (&c32)[4][TW], const float *cm,
                                                       int corner, int lane) {
    return reduce_scatter_corner_ptr(cache, c32[corner], cm, corner, lane);
}

// Fixed bank indexes keep Fourier cache rows eligible for register promotion.
// All lanes call; only lane 0 copies the shared m=32 term.
template <int Dst, int Src, class Record>
__device__ __forceinline__ void reuse_corner(Record &s, float (&cache)[4][TW],
                                             float (&scache)[4][TW], int (&tags)[4], int lane) {
#pragma unroll
    for (int w = 0; w < TW; ++w)
        cache[Dst][w] = cache[Src][w];
    if (lane == 0) {
#pragma unroll
        for (int w = 0; w < TW; ++w)
            s.c32[Dst][w] = s.c32[Src][w];
    }
    if constexpr (TransportPolicy::strato) {
#pragma unroll
        for (int w = 0; w < TW; ++w)
            scache[Dst][w] = scache[Src][w];
        if (lane == 0) {
#pragma unroll
            for (int w = 0; w < TW; ++w)
                s.st_c32[Dst][w] = s.st_c32[Src][w];
        }
    }
    tags[Dst] = s.cell[Dst];
}

// Tropospheric aerosol Mie phase (atmosphere/tropo_mie.py, opt-in by state; without -DTROPO_MIE
// nothing below exists and every hg(u,aerosol_g) site is unchanged). Per-wavelength table per
// steradian behind the asca pointer: asca[0..FW) as before, table node i of wavelength w at
// asca[FW+i*FW+w], u=-1+i*2/TROPO_LUT_BINS (strato FINE node formula).
#define TROPO_LUT_BINS 16384
#define TROPO_LUT (asca + FW)
#if defined(GF_ON) && (!defined(STRATO) || STRATO_LUT_BINS != TROPO_LUT_BINS)
#error E_TROPO_MIE_GF_REQUIRES_STRATO_MIE_LUT_FINE
#endif
__device__ __forceinline__ float tropo_phase(float u, int w, const float *lut) {
    float x = (clip(u, -1.f, 1.f) + 1.f) * (float)(TROPO_LUT_BINS / 2);
    int i = min((int)x, TROPO_LUT_BINS - 1);
    float f = x - (float)i;
    float a = lut[i * FW + w], b = lut[(i + 1) * FW + w];
    return a + f * (b - a);
}
// Paired ground phase of direction j for wavelength w: the hg site's m0/m1 expressions, Mie table
// instead of hg.
__device__ __forceinline__ float tropo_pair(const Sample &s, const float *gdir, int64_t j, int w,
                                            const float *lut) {
    float m0 = s.mu * gdir[3 * j + 2] + s.exd * gdir[3 * j], m1 = fabsf(s.eyd) * gdir[3 * j + 1];
    return (tropo_phase(clip(m0 + m1, -1, 1), w, lut) + tropo_phase(clip(m0 - m1, -1, 1), w, lut)) /
           2.f;
}

#ifdef GROUND_SH_L
#if !defined(GF_ON) || defined(TROPO_MIE)
#error E_GROUND_SH_PATH
#endif
#define GSH_SIDE (GROUND_SH_GRID + 3)
// The SH convolution is synthesized once per cell on a polar/azimuth grid.
// Ghost rows/columns are synthesized too, including the polar reflection.
// Catmull-Rom interpolation retains the signed SH sum (no intensity clipping).
__device__ __forceinline__ void ground_cubic(float t, float *w) {
    const float t2 = t * t, t3 = t2 * t;
    w[0] = -.5f * t + t2 - .5f * t3;
    w[1] = 1.f - 2.5f * t2 + 1.5f * t3;
    w[2] = .5f * t + 2.f * t2 - 1.5f * t3;
    w[3] = -.5f * t2 + .5f * t3;
}
__device__ __forceinline__ void ground_sh_sums(const Sample &s, int lane, const int64_t *goff,
                                               const float *unused, const float *coeff, float &accw,
                                               float &st_acc) {
    bool any = false;
    for (int c = 0; c < 4; c++)
        any |= goff[s.cell[c]] != goff[s.cell[c] + 1];
    if (!any)
        return;
    if (lane < TW) {
        const float theta = atan2f(hypotf(s.exd, s.eyd), s.mu) * (GROUND_SH_GRID / (float)PI);
        const float phi = atan2f(fabsf(s.eyd), s.exd) * (GROUND_SH_GRID / (float)PI);
        const int it = min((int)theta, GROUND_SH_GRID - 1), ip = min((int)phi, GROUND_SH_GRID - 1);
        float wt[4], wp[4];
        ground_cubic(theta - it, wt);
        ground_cubic(phi - ip, wp);
        for (int corner = 0; corner < 4; corner++) {
            const int cc = s.cell[corner];
            if (goff[cc] == goff[cc + 1])
                continue;
            float a = 0.f, b = 0.f;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float ar = 0.f, br = 0.f;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int64_t row = ((int64_t)cc * GSH_SIDE + it + i) * GSH_SIDE + ip + j;
                    const float2 v = ((const float2 *)coeff)[row * FW + GW(lane)];
                    ar += wp[j] * v.x;
                    br += wp[j] * v.y;
                }
                a += wt[i] * ar;
                b += wt[i] * br;
            }
            accw += s.sw[corner] * a;
            st_acc += s.sw[corner] * b;
        }
    }
}
#endif

// Fused tropo+strato ground sums of one sample (GROUND_FUSED); scratch gph >= 32 floats + 66 int2.
__device__ __forceinline__ void gf_sums(const Sample &s, int lane, float *gph, const int64_t *goff,
                                        const float *gdir, const float *gval, const float *st_chi,
                                        float aerosol_g, float &accw, float &st_acc,
                                        const float *tr_lut) {
    const int w = lane & 15;
    const float pmu = s.mu, pexd = s.exd, paey = fabsf(s.eyd);
    int2 *gl = (int2 *)(((unsigned long long)(gph + 32) + 7ull) &
                        ~7ull); // [k]: + node, [33+k]: - node; {li, f bits}
    for (int corner = 0; corner < 4; corner++) {
        int cc = s.cell[corner];
        const float swc = s.sw[corner];
        const int64_t je = goff[cc + 1];
        for (int64_t j0 = goff[cc]; j0 < je; j0 += 32) {
            int64_t j = j0 + lane;
            if (j < je) {
                float m0 = pmu * gdir[3 * j + 2] + pexd * gdir[3 * j], m1 = paey * gdir[3 * j + 1];
                float up = clip(m0 + m1, -1, 1), um = clip(m0 - m1, -1, 1);
                gph[lane] = (hg(up, aerosol_g) + hg(um, aerosol_g)) / 2.f;
                float xp = (clip(up, -1.f, 1.f) + 1.f) * (float)(STRATO_LUT_BINS / 2),
                      xm = (clip(um, -1.f, 1.f) + 1.f) * (float)(STRATO_LUT_BINS / 2);
                int lp = min((int)xp, STRATO_LUT_BINS - 1), lm = min((int)xm, STRATO_LUT_BINS - 1);
                gl[lane] = make_int2(lp, __float_as_int(xp - (float)lp));
                gl[33 + lane] = make_int2(lm, __float_as_int(xm - (float)lm));
            }
            __syncwarp();
            const int count = (int)((je - j0) < 32 ? (je - j0) : 32);
            const int2 *gs = gl + (lane < 16 ? 0 : 33);
            const float *gq = gval + j0 * FW + GW(w);
            int k = 0;
            for (; k + GFG <= count; k += GFG) {
                float a[GFG], b[GFG], f[GFG], g[GFG];
                float ta[GFG], tb[GFG];
                _Pragma("unroll") for (int q = 0; q < GFG; q++) {
                    const int2 e = gs[k + q];
                    f[q] = __int_as_float(e.y);
                    a[q] = 0.f;
                    b[q] = 0.f;
                    g[q] = 0.f;
                    if (w < TW) {
                        a[q] = st_chi[(33 + e.x) * FW + GW(w)];
                        b[q] = st_chi[(34 + e.x) * FW + GW(w)];
                    }
                    if constexpr (TransportPolicy::tropo_mie) {
                        ta[q] = 0.f;
                        tb[q] = 0.f;
                        if (w < TW) {
                            ta[q] = tr_lut[e.x * FW + GW(w)];
                            tb[q] = tr_lut[(e.x + 1) * FW + GW(w)];
                        }
                    }
                    if (lane < TW)
                        g[q] = gq[(k + q) * FW];
                }
                _Pragma("unroll") for (int q = 0; q < GFG; q++) {
                    float paired = 0.f;
                    if (w < TW)
                        paired = a[q] + f[q] * (b[q] - a[q]);
                    float minus = __shfl_down_sync(0xffffffff, paired, 16);
                    if constexpr (TransportPolicy::tropo_mie) {
                        // Same node index and fraction as the strato LUT
                        // (TROPO_LUT_BINS==STRATO_LUT_BINS): = tropo_phase(up/um).
                        float tpaired = 0.f;
                        if (w < TW)
                            tpaired = ta[q] + f[q] * (tb[q] - ta[q]);
                        float tminus = __shfl_down_sync(0xffffffff, tpaired, 16);
                        if (lane < TW) {
                            accw += (swc * ((tpaired + tminus) / 2.f)) * g[q];
                            float ph = (paired + minus) / 2.f;
                            st_acc += (swc * ph) * g[q];
                        }
                    } else {
                        if (lane < TW) {
                            accw += (swc * gph[k + q]) * g[q];
                            float ph = (paired + minus) / 2.f;
                            st_acc += (swc * ph) * g[q];
                        }
                    }
                }
            }
            for (; k < count; k++) {
                const int2 e = gs[k];
                const float fk = __int_as_float(e.y);
                float paired = 0.f, gk = 0.f;
                if (w < TW) {
                    const float ak = st_chi[(33 + e.x) * FW + GW(w)],
                                bk = st_chi[(34 + e.x) * FW + GW(w)];
                    paired = ak + fk * (bk - ak);
                }
                if (lane < TW)
                    gk = gq[k * FW];
                float minus = __shfl_down_sync(0xffffffff, paired, 16);
                if constexpr (TransportPolicy::tropo_mie) {
                    float tpaired = 0.f;
                    if (w < TW) {
                        const float tak = tr_lut[e.x * FW + GW(w)],
                                    tbk = tr_lut[(e.x + 1) * FW + GW(w)];
                        tpaired = tak + fk * (tbk - tak);
                    }
                    float tminus = __shfl_down_sync(0xffffffff, tpaired, 16);
                    if (lane < TW) {
                        accw += (swc * ((tpaired + tminus) / 2.f)) * gk;
                        float ph = (paired + minus) / 2.f;
                        st_acc += (swc * ph) * gk;
                    }
                } else {
                    if (lane < TW) {
                        accw += (swc * gph[k]) * gk;
                        float ph = (paired + minus) / 2.f;
                        st_acc += (swc * ph) * gk;
                    }
                }
            }
            __syncwarp();
        }
    }
}

template <bool Condition, class Then, class Else> struct SelectType {
    using type = Then;
};
template <class Then, class Else> struct SelectType<false, Then, Else> {
    using type = Else;
};

template <class Record> struct FullSampleView {
    const Record &sample;
    __device__ __forceinline__ explicit FullSampleView(const Record &s) : sample(s) {}
    __device__ __forceinline__ int ir() const {
        return sample.ir;
    }
    __device__ __forceinline__ float height() const {
        return sample.h;
    }
    __device__ __forceinline__ int cell(int c) const {
        return sample.cell[c];
    }
    __device__ __forceinline__ float weight(int c) const {
        return sample.sw[c];
    }
    __device__ __forceinline__ float rotate(int component, const float *rm) const {
        float rot = 0.f;
        for (int k = 0; k < 4; ++k)
            rot += sample.T[4 * component + k] * rm[k];
        return rot;
    }
};
template <class Compact> struct CompactSampleViewBase {
    const Compact &sample;
    const float4 t0, t1, t2, t3, weights;
    const int4 cells;
    const int radial;
    __device__ __forceinline__ explicit CompactSampleViewBase(const Compact &q)
        : sample(q), t0(*(const float4 *)q.T), t1(*(const float4 *)(q.T + 4)),
          t2(*(const float4 *)(q.T + 8)), t3(*(const float4 *)(q.T + 12)),
          weights(*(const float4 *)q.sw), cells(*(const int4 *)q.cell), radial(q.ir) {}
    __device__ __forceinline__ int ir() const {
        return radial;
    }
    __device__ __forceinline__ float height() const {
        return sample.h;
    }
    __device__ __forceinline__ int cell(int c) const {
        return c == 0 ? cells.x : c == 1 ? cells.y : c == 2 ? cells.z : cells.w;
    }
    __device__ __forceinline__ float weight(int c) const {
        return c == 0 ? weights.x : c == 1 ? weights.y : c == 2 ? weights.z : weights.w;
    }
    __device__ __forceinline__ float rotate(int component, const float *rm) const {
        // component is unrolled; four scalar multiply/adds keep their original order.
        float rot = 0.f;
        rot += (component == 0   ? t0.x
                : component == 1 ? t1.x
                : component == 2 ? t2.x
                                 : t3.x) *
               rm[0];
        rot += (component == 0   ? t0.y
                : component == 1 ? t1.y
                : component == 2 ? t2.y
                                 : t3.y) *
               rm[1];
        rot += (component == 0   ? t0.z
                : component == 1 ? t1.z
                : component == 2 ? t2.z
                                 : t3.z) *
               rm[2];
        rot += (component == 0   ? t0.w
                : component == 1 ? t1.w
                : component == 2 ? t2.w
                                 : t3.w) *
               rm[3];
        return rot;
    }
};
using CompactSampleView = CompactSampleViewBase<PointM2>;
using WarmSampleView = CompactSampleViewBase<WarmPoint>;
struct SampleLane {
    int wave, sample;
    bool active, odd;
    __device__ __forceinline__ SampleLane(int lane, int step, int batch_end)
        : wave(TransportPolicy::slim_rays ? lane & 15 : lane), sample(step), active(wave < TW),
          odd(step + 1 < batch_end) {
        if constexpr (TransportPolicy::slim_rays) {
            const int half = lane >> 4;
            sample = half && odd ? step + 1 : step;
            active = active && (half == 0 || odd);
        }
    }
};

// ---- Fused transport ----

// One KSPLIT quadrature point. The record pass assigns one unit-basis wave to each slot.
struct ColumnRecord {
    static constexpr int header_words = 16;
    static constexpr int strato_slot = NCOMP;
    static constexpr int width = strato_slot + 1;

    static __device__ __forceinline__ ColumnRecord *rows(float *query) {
        return reinterpret_cast<ColumnRecord *>(query + header_words);
    }
    __device__ __forceinline__ void write_basis(int wave, float mid) {
        if (wave < width)
            values[wave] = mid;
    }
    __device__ __forceinline__ float component(int k) const { return values[k]; }
    __device__ __forceinline__ float stratosphere() const { return values[strato_slot]; }
    __device__ __forceinline__ void preload_from(const ColumnRecord &row) {
#pragma unroll
        for (int k = 0; k < width; ++k)
            values[k] = row.values[k];
    }

  private:
    float values[width];
};

// The launch ABI is normalized once here. Optional inputs are never read by an
// inactive compile-time role; no device allocation or record layout changes.
struct TransportInputs {
    int cells = 0;
    int nh = 0;
    int nc = 0;
    int nq = 0;
    int nsh = 0;
    int nsx = 0;
    int first_arg = 0;
    int phase_arg = 0;
    int boundary_arg = 0;
    const int64_t *cell_off = nullptr;
    const double *starts = nullptr;
    const double *targets = nullptr;
    const double *dirs = nullptr;
    const float *weights = nullptr;
    const uint32_t *edge_off = nullptr;
    const double *edges = nullptr;
    const uint16_t *lower = nullptr;
    const int32_t *end_j = nullptr;
    const float *end_f = nullptr;
    const float *end_col = nullptr;
    const uint8_t *hits = nullptr;
    const double *gu = nullptr;
    const double *gw = nullptr;
    const double *aa = nullptr;
    const double *bb = nullptr;
    const double *dd = nullptr;
    const float *h = nullptr;
    const float *chi = nullptr;
    const float *hr = nullptr;
    const float *ha = nullptr;
    const float *ext = nullptr;
    const float *asca = nullptr;
    const float *sun_h = nullptr;
    const float *sun_col = nullptr;
    const float *gm = nullptr;
    const int64_t *goff = nullptr;
    const float *gdir = nullptr;
    const float *gval = nullptr;
    float height = 0;
    float aerosol_g = 0;
    float mix = 0;
    double albedo = 0;
    const float *src = nullptr;
    const double *boundary = nullptr;
    double *moments = nullptr;
    double *flux = nullptr;
    float *query = nullptr;
    float *slim_aero = nullptr;
    int2 *vp_records = nullptr;
    float *slim_dump = nullptr;
    const float *src_sh = nullptr;
    const float *st_ext = nullptr;
    const float *st_sca = nullptr;
    const float *st_chi = nullptr;
    const float *st_density = nullptr;
    const float *st_sun_col = nullptr;
    const float *st_end_col = nullptr;
    float st_center = 0;
    float st_sigma = 0;
    float st_norm = 0;
    const float *profile_table = nullptr;
    int profile_mask = 0;
    float air_surface = 0;
    const int *cell_order = nullptr;
    double *split_scratch = nullptr;
    double *split_flux = nullptr;
    int *split_count = nullptr;
#ifdef EXP_SECTION_CLOCKS
    unsigned long long *section_clocks = nullptr;
#endif
    float *ray_rads = nullptr;
    int *ray_sync = nullptr;
};

template <class Record> struct Transport : TransportInputs {
    static constexpr int first = K_FIRST, phase = K_PHASE, with_boundary = K_BOUNDARY;
    int tid, lane, warp, nwarps;
    ProfileParams profile;
    ExtinctionParams optics;
    int sinv_mode = 0;
    ColumnRecord *column_rows = nullptr;
    int64_t slim_base = 0;
    unsigned long long live_a = ~0ull, live_a_hi = ~0ull, live_s = ~0ull, live_s_hi = ~0ull;
    __device__ __forceinline__ static bool layer_live(unsigned long long lo, unsigned long long hi,
                                                      int ir) {
        return ir < 64 ? (((lo >> ir) & 1ull) != 0)
                       : (ir < 128 ? (((hi >> (ir - 64)) & 1ull) != 0) : true);
    }
#ifdef EXP_SECTION_CLOCKS
    unsigned long long section_cycles[8] = {}, section_begin = 0, total_begin = 0;
#endif
    Record *samples = nullptr;
    float *ray_constants, *ground_phase = nullptr;
    float *cell_B = nullptr, *warm_cm = nullptr, *warm_c32 = nullptr;
    FirstPoint *fpts = nullptr;
    float *fmp = nullptr, *fB = nullptr;
    using PtBatch = typename SelectType<
        TransportPolicy::slim_rays, PointM2,
        typename SelectType<TransportPolicy::fast_warm, WarmPoint, PointData>::type>::type;
    PtBatch *pts = nullptr;
    unsigned char *rs_key = nullptr, *rs_perm = nullptr;
    __device__ __forceinline__ ColumnRecord &column_record(uint32_t seg, int64_t ray,
                                                           int step) const {
        return column_rows[(int64_t)(seg - ray) * nq + step];
    }
    __device__ __forceinline__ float *aerosol_record(uint32_t seg, int64_t ray, int step,
                                                     int corner) const {
        return slim_aero + ((((int64_t)(seg - ray) * nq + step - slim_base) * 4 + corner) * AERO_W);
    }

    // The compact consumer reads its batch directly. Other batched roles copy
    // only PointData into the persistent sample; the caller publishes the copy.
    __device__ __forceinline__ CompactSampleView sample_view(PointM2 *batch, Record *, int step0,
                                                             int sample) {
        return CompactSampleView(batch[sample - step0]);
    }
    __device__ __forceinline__ WarmSampleView sample_view(WarmPoint *batch, Record *, int step0,
                                                          int sample) {
        return WarmSampleView(batch[sample - step0]);
    }
    __device__ __forceinline__ FullSampleView<Record> sample_view(PointData *batch, Record *s,
                                                                  int step0, int sample) {
        if constexpr (TransportPolicy::batched_points) {
            const int *source = (const int *)&batch[sample - step0];
            int *dest = (int *)static_cast<PointData *>(s);
            for (int k = lane; k < (int)(sizeof(PointData) / 4); k += 32)
                dest[k] = source[k];
        }
        return FullSampleView<Record>(*s);
    }

    // All lanes participate in ground lookup/shuffles. Lane w publishes the
    // tropo/strato ground sums for wavelength w in the shared sample.
    struct GroundLookup {
        float a, b, f, value;
    };
    __device__ __forceinline__ GroundLookup ground_lookup(int64_t j, float mu, float exd, float aey,
                                                          float d0, float d1, float d2) const {
        GroundLookup result{};
        const int w = lane & 15;
        if (w < TW) {
            float m0 = mu * d2 + exd * d0, m1 = aey * d1;
            float u = lane < 16 ? clip(m0 + m1, -1, 1) : clip(m0 - m1, -1, 1);
            float x = (clip(u, -1.f, 1.f) + 1.f) * (float)(STRATO_LUT_BINS / 2);
            int i = min((int)x, STRATO_LUT_BINS - 1);
            result.a = st_chi[(33 + i) * FW + GW(w)];
            result.b = st_chi[(34 + i) * FW + GW(w)];
            result.f = x - (float)i;
        }
        if (lane < TW)
            result.value = gval[j * FW + GW(lane)];
        return result;
    }
    __device__ __forceinline__ void tropo_ground(Record &s) {

        float accw = 0.f;
        for (int corner = 0; corner < 4; corner++) {
            int cc = s.cell[corner];
            for (int64_t j0 = goff[cc]; j0 < goff[cc + 1]; j0 += 32) {
                int64_t j = j0 + lane;
                if (j < goff[cc + 1]) {
                    float m0 = s.mu * gdir[3 * j + 2] + s.exd * gdir[3 * j],
                          m1 = fabsf(s.eyd) * gdir[3 * j + 1];
                    ground_phase[lane] = (hg(clip(m0 + m1, -1, 1), aerosol_g) +
                                          hg(clip(m0 - m1, -1, 1), aerosol_g)) /
                                         2.f;
                }
                __syncwarp();
                int count = (int)((goff[cc + 1] - j0) < 32 ? (goff[cc + 1] - j0) : 32);
                if constexpr (TransportPolicy::tropo_mie) {
                    // Mie: the per-direction phase depends on the wavelength, so lane w
                    // evaluates it itself (ground_phase unused).
                    if (lane < TW) {
                        const float swc = s.sw[corner];
                        const float *gq = gval + j0 * FW + GW(lane);
                        int k = 0;
                        for (; k + 8 <= count; k += 8) {
                            float g[8], ph[8];
                            _Pragma("unroll") for (int q = 0; q < 8; q++) {
                                g[q] = gq[(k + q) * FW];
                                ph[q] = tropo_pair(s, gdir, j0 + k + q, GW(lane), TROPO_LUT);
                            }
                            _Pragma("unroll") for (int q = 0; q < 8; q++) accw +=
                                (swc * ph[q]) * g[q];
                        }
                        for (; k < count; k++)
                            accw += (swc * tropo_pair(s, gdir, j0 + k, GW(lane), TROPO_LUT)) *
                                    gq[k * FW];
                    }
                } else {
                    if (lane < TW) {
                        const float swc = s.sw[corner];
                        const float *gq = gval + j0 * FW + GW(lane);
                        int k = 0;
                        for (; k + 8 <= count; k += 8) {
                            float g[8], ph[8];
                            _Pragma("unroll") for (int q = 0; q < 8; q++) {
                                g[q] = gq[(k + q) * FW];
                                ph[q] = ground_phase[k + q];
                            }
                            _Pragma("unroll") for (int q = 0; q < 8; q++) accw +=
                                (swc * ph[q]) * g[q];
                        }
                        for (; k < count; k++)
                            accw += (swc * ground_phase[k]) * gq[k * FW];
                    }
                }
                __syncwarp();
            }
        }
        if (lane < TW)
            s.c32[0][lane] = accw;
        __syncwarp();
    }
    __device__ __forceinline__ void strato_ground(Record &s) {

        int w = lane & 15;
        float st_acc = 0.f;
        const float pmu = s.mu, pexd = s.exd, paey = fabsf(s.eyd);
        for (int corner = 0; corner < 4; corner++) {
            int cc = s.cell[corner];
            const float swc = s.sw[corner];
            int64_t j = goff[cc], je = goff[cc + 1];
            GroundLookup current{};
            float d0 = 0.f, d1 = 0.f, d2 = 0.f;
            if (j < je) {
                current = ground_lookup(j, pmu, pexd, paey, gdir[3 * j], gdir[3 * j + 1],
                                        gdir[3 * j + 2]);
            }
            if (j + 1 < je) {
                d0 = gdir[3 * (j + 1)];
                d1 = gdir[3 * (j + 1) + 1];
                d2 = gdir[3 * (j + 1) + 2];
            }
            for (; j < je; j++) {
                const float a = current.a, b = current.b, f = current.f, gvj = current.value;
                if (j + 1 < je) {
                    current = ground_lookup(j + 1, pmu, pexd, paey, d0, d1, d2);
                }
                if (j + 2 < je) {
                    d0 = gdir[3 * (j + 2)];
                    d1 = gdir[3 * (j + 2) + 1];
                    d2 = gdir[3 * (j + 2) + 2];
                }
                float paired = 0.f;
                if (w < TW)
                    paired = a + f * (b - a);
                float minus = __shfl_down_sync(0xffffffff, paired, 16);
                if (lane < TW) {
                    float ph = (paired + minus) / 2.f;
                    st_acc += (swc * ph) * gvj;
                }
            }
        }
        if (lane < TW)
            s.st_c32[0][lane] = st_acc;
        __syncwarp();
    }
    __device__ __forceinline__ void fused_ground(Record &s, float *y) {
        float accw = 0.f, st_acc = 0.f;
        if constexpr (TransportPolicy::ground_sh) {
            ground_sh_sums(s, lane, goff, gdir, gval, accw, st_acc);
        } else {
            if constexpr (TransportPolicy::split_rays || K_PHASE == 2) {
                // FL_RAYSPLIT projects after the ray; phase 2 has no projection. Both
                // can borrow the warp's Y row.
                gf_sums(s, lane, y, goff, gdir, gval, st_chi, aerosol_g, accw, st_acc, asca + FW);
            } else {
                gf_sums(s, lane, ground_phase, goff, gdir, gval, st_chi, aerosol_g, accw, st_acc,
                        asca + FW);
            }
        }
        if (lane < TW) {
            s.c32[0][lane] = accw;
            s.st_c32[0][lane] = st_acc;
        }
        __syncwarp();
    }
    __device__ __forceinline__ void first_light_ground(Record &s, float *y) {
        if (first) {
            if constexpr (TransportPolicy::ground_fused)
                fused_ground(s, y);
            else {
                tropo_ground(s);
                if constexpr (TransportPolicy::strato)
                    strato_ground(s);
            }
        }
    }
    // Lane 0 adds boundary light after the transport sum; phase 2 publishes
    // the completed warp radiance without a cross-warp dependency.
    __device__ __forceinline__ void boundary_and_query(int64_t ray, const float *pu, float *rad) {
        if (lane == 0 && !NOOUT_ON && with_boundary) {
            int j = end_j[ray];
            float f = end_f[ray];
            for (int w = 0; w < TW; w++) {
                float e = (1.f - f) * (float)boundary[j * FW + GW(w)];
                e += f * (float)boundary[(j + 1) * FW + GW(w)];
                float od = 0;
                for (int k = 0; k < NCOMP; k++)
                    od += end_col[NCOMP * ray + k] * ext[k * FW + GW(w)];
                if constexpr (TransportPolicy::strato) {
                    od += st_end_col[ray] * st_ext[GW(w)];
                }
                e *= expf(-od) * (float)hits[ray];
                for (int k = 0; k < (phase == 1 ? 3 : 4); k++)
                    rad[4 * w + k] += (e * pu[k]) / 2.f;
            }
        }
        if (phase == 2) {
            SECTION_SYNC(__syncwarp());
            for (int k = lane; k < 4 * TW; k += 32)
                query[ray * 4 * FW + k] = rad[k];
        }
    }
    // Per-thread ray state. Shared pointers refer to this warp's fixed slices;
    // Fourier cache and compensated sums live until the ray is complete.
    using PathCoord = typename SelectType<TransportPolicy::path_ds, DS, const double>::type;
    struct RayState {
        const double *p, *target, *d;
        const float *B;
        float *y, *rad, *rc, *cd, *dl, *pu, *se, *direct;
        PathCoord *beta_p, *beta_d;
        float global_st, pu_r[4];
        float lane_rad[4] = {}, lane_rc[4] = {}, lane_val[4], lane_log, lane_wg, lane_am;
        float ray_ext[NCOMP] = {}, ray_st_ext = 0.f;
        float cache[4][TW] = {}, scache[4][TW];
        int tags[4] = {-1, -1, -1, -1};
        float tau_lane = 0.f, tc_lane = 0.f;
        DS sp_t = {0.f, 0.f};
        float sp_b = 0.f, sp_acc = 0.f;
        float sunphase, strato_sunphase = 0.f;
        DS gu_l = {0.f, 0.f}, gw_l = {0.f, 0.f};
        float lane_mp[NCOMP] = {}, lane_dt = 0.f;
    };
    struct SampleScratch {
        float paired[4] = {};
        float aero[4] = {}, rm[4][4] = {};
        ColumnRecord column{};
        float air[2] = {}, strato[2] = {};
    };

    // A complete warp traces one ray. Scratch belongs to this warp until the
    // caller's warp/block barrier; shared radiance is published before return.
    __device__ __forceinline__ void trace_ray(int cell, int64_t ray, const double *p,
                                              const double *target, const float *B, float *y,
                                              float *rad, float *rc, float *cd) {
        RayState r;
        r.p = p;
        r.target = target;
        r.B = B;
        r.y = y;
        r.rad = rad;
        r.rc = rc;
        r.cd = cd;

        r.d = dirs + 3 * ray;
        // global_st, dl[3], pu[4], se[3], direct[4]; PATH_DS adds p/d as six DS values.
        r.dl = ray_constants + 1;
        r.pu = ray_constants + 4;
        r.se = ray_constants + 8;
        r.direct = ray_constants + 11;
        if constexpr (TransportPolicy::path_ds) {
            r.beta_p = (DS *)(ray_constants + 15);
            r.beta_d = r.beta_p + 3;
        } else {
            r.beta_p = r.p;
            r.beta_d = r.d;
        }

        if (lane == 0) {
            ray_constants[0] = (float)sqrt(fmax(0., 1. - r.d[2] * r.d[2]));
            if constexpr (TransportPolicy::path_ds) {
                // Same conversion as path_q, once per ray; the existing syncwarp publishes both
                // words.
                for (int k = 0; k < 3; k++) {
                    r.beta_p[k] = ds(r.p[k]);
                    r.beta_d[k] = ds(r.d[k]);
                }
            }

            for (int j = 0; j < 3; j++) {
                double x = 0;
                for (int k = 0; k < 3; k++)
                    x += r.target[3 * k + j] * r.d[k];
                r.dl[j] = (float)x;
            }

            r.pu[0] = 1.f - r.dl[0] * r.dl[0];
            r.pu[1] = 1.f - r.dl[1] * r.dl[1];
            r.pu[2] = 1.f - r.dl[2] * r.dl[2];
            r.pu[3] = -r.dl[0] * r.dl[2];
            float sun_dot = (r.dl[0] * r.B[6] + r.dl[1] * r.B[7]) + r.dl[2] * r.B[8];
            for (int k = 0; k < 3; k++)
                r.se[k] = r.B[6 + k] - r.dl[k] * sun_dot;
            r.direct[0] = (r.pu[0] - r.se[0] * r.se[0]) / 2;
            r.direct[1] = (r.pu[1] - r.se[1] * r.se[1]) / 2;
            r.direct[2] = (r.pu[2] - r.se[2] * r.se[2]) / 2;
            r.direct[3] = (r.pu[3] - r.se[0] * r.se[2]) / 2;
        }
        SECTION_SYNC(__syncwarp());
        r.global_st = ray_constants[0];
        // pu is fixed for the ray; the warm path reuses register copies.
        r.pu_r[0] = r.pu[0];
        r.pu_r[1] = r.pu[1];
        r.pu_r[2] = r.pu[2];
        r.pu_r[3] = r.pu[3];
        SECTION_BEGIN()
        if constexpr ((!TransportPolicy::split_rays || !K_FIRST) && !TransportPolicy::slim_rays &&
                      !(TransportPolicy::ground_fused && K_PHASE == 2)) {
            sh_basis(
                r.d[2], lane, r.y, aa, bb,
                dd); // SLIM_RAYS: y only feeds the projection (sos_slim_proj reads the chunk y)
        }
        SECTION_END(2)
        if (lane == 0) {
#ifndef FL_SPLIT_ON

            double st = sqrt(fmax(0., 1. - r.d[2] * r.d[2]));
            float cp = st > 1e-12 ? (float)(r.d[0] / st) : 1.f;
            cp = clip(cp, -1, 1);

            r.cd[0] = 1;
            r.cd[1] = cp;
            for (int m = 2; m <= 32; m++)
                r.cd[m] = (2.f * cp) * r.cd[m - 1] - r.cd[m - 2];
#endif
            for (int k = 0; k < 4 * TW; k++) {
                r.rad[k] = 0;
                if constexpr (!TransportPolicy::fast_warm)
                    r.rc[k] = 0;
            }
        }
        SECTION_SYNC(__syncwarp());

        // Per-lane extinction rows of the column-record attenuation are constant over the ray:
        // read them once.

        if constexpr (TransportPolicy::preload) {
            const int rw = TransportPolicy::slim_rays ? lane & 15 : lane;
            if (rw < TW) {
                for (int k = 0; k < NCOMP; ++k)
                    r.ray_ext[k] = ext[k * FW + GW(rw)];
                r.ray_st_ext = st_ext[GW(rw)];
            }
        }

        // Simpson prefix state of lane w<TW (numpy_reference.py SimpsonPrefix): panel start (DS
        // t, extinction) and the segment sum.

        if constexpr (first)
            r.sunphase = hg((float)r.d[2], aerosol_g);

        if constexpr (TransportPolicy::strato)
            if (first && lane < TW)
                r.strato_sunphase = mie_phase((float)r.d[2], GW(lane), st_chi);

        if constexpr (TransportPolicy::path_ds) {
            if (lane < nq && nq <= 32) {
                r.gu_l = ds(gu[lane]);
                r.gw_l = ds(gw[lane]);
            }
        }
        integrate_segments(r, ray);
        if constexpr (TransportPolicy::warm) {
            if (lane < TW) {
#pragma unroll
                for (int k = 0; k < 4; k++)
                    r.rad[4 * lane + k] = r.lane_rad[k];
            }
            __syncwarp();
        }
        boundary_and_query(ray, r.pu, r.rad);

        if constexpr (TransportPolicy::column_records && TransportPolicy::ray_dump) {
            // KSPLIT=3 VERIFY: this ray's final radiance (slim_dump argument; header int64 [5] =
            // chunk first ray), as SLIM_RAYS publishes it.
            if (phase != 2) {
                __syncwarp();
                float *dump_ = slim_dump + (ray - ((const int64_t *)query)[5]) * 4 * TW;
                for (int k = lane; k < 4 * TW; k += 32)
                    dump_[k] = r.rad[k];
            }
        }
    }
    // First-light direct/ground source, or reset the warm interpolation state.
    __device__ __forceinline__ void initialize_wave(RayState &r, int w) {
        if constexpr (TransportPolicy::warm) {
            r.lane_log = r.lane_wg = r.lane_am = 0;
            for (int k = 0; k < 4; k++)
                r.lane_val[k] = 0;
        } else {
            Record &s = samples[warp];
            s.logs[w] = 0;
            for (int k = 0; k < 4; k++)
                s.val[w][k] = 0;
        }
        if constexpr (first) {
            Record &s = samples[warp];
            float od = 0;
            for (int k = 0; k < NCOMP; k++)
                od += s.sc[k] * ext[k * FW + GW(w)];
            if constexpr (TransportPolicy::strato) {
                od += s.st_sc * st_ext[GW(w)];
            }
            float sun = s.lit ? expf(-od) : 0.f, ga = s.c32[0][w];
            float phase_value = r.sunphase;
            if constexpr (TransportPolicy::tropo_mie)
                phase_value = tropo_phase((float)r.d[2], GW(w), asca + FW);
            float beam = (s.prof[1] * asca[GW(w)]) * (sun * phase_value + ga),
                  ray = (sun * s.prof[0]) * ext[GW(w)];
            for (int k = 0; k < 4; k++)
                s.val[w][k] = ray * (mix * (float)(3 / (8 * PI)) * r.direct[k] +
                                     (1.f - mix) * (float)(1 / (8 * PI)) * r.pu[k]) +
                              (beam * r.pu[k]) / 2.f;
            if constexpr (TransportPolicy::strato) {
                float sb = (s.st_prof * st_sca[GW(w)]) * (sun * r.strato_sunphase + s.st_c32[0][w]);
                for (int k = 0; k < 4; k++)
                    s.val[w][k] += (sb * r.pu[k]) / 2.f;
            }
        }
    }
    __device__ __forceinline__ void initialize_source(RayState &r, const SampleLane &ownership) {
        if constexpr (TransportPolicy::warm || TransportPolicy::strato) {
            if (ownership.active)
                initialize_wave(r, ownership.wave);
        } else if (lane == 0) {
            for (int w = 0; w < TW; ++w)
                initialize_wave(r, w);
        }
    }
    template <int Dst, int Src>
    __device__ __forceinline__ void reuse_fast_corner(float (&cache)[4][TW], int (&tags)[4],
                                                      int target_cc) const {
        if (tags[Src] == target_cc) {
#pragma unroll
            for (int w = 0; w < TW; ++w)
                cache[Dst][w] = cache[Src][w];
            if constexpr (SH_L == 32) {
                if (lane == 0) {
#pragma unroll
                    for (int w = 0; w < TW; ++w)
                        warm_c32[Dst * TW + w] = warm_c32[Src * TW + w];
                }
            }
            tags[Dst] = target_cc;
        }
    }
    // Reuse/contract Fourier corners, rotate tensors, then form log-space moments.
    // Every lane participates in the reductions; only wavelength owners accumulate.
    template <class View>
    __device__ __forceinline__ void shade_corners(RayState &r, SampleScratch &scratch, int64_t ray,
                                                  uint32_t seg, int step,
                                                  const SampleLane &ownership, const View &view) {
        if constexpr (TransportPolicy::fast_warm) {
            const int c0 = view.cell(0);
            if (r.tags[0] >= 0 && r.tags[0] != c0) {
                const int di = c0 / nc - r.tags[0] / nc, dj = c0 % nc - r.tags[0] % nc;
                if (di == 0 && dj == 1) {
                    reuse_fast_corner<0, 1>(r.cache, r.tags, view.cell(0));
                    reuse_fast_corner<2, 3>(r.cache, r.tags, view.cell(2));
                } else if (di == 0 && dj == -1) {
                    reuse_fast_corner<1, 0>(r.cache, r.tags, view.cell(1));
                    reuse_fast_corner<3, 2>(r.cache, r.tags, view.cell(3));
                } else if (di == 1 && dj == 0) {
                    reuse_fast_corner<0, 2>(r.cache, r.tags, view.cell(0));
                    reuse_fast_corner<1, 3>(r.cache, r.tags, view.cell(1));
                } else if (di == -1 && dj == 0) {
                    reuse_fast_corner<2, 0>(r.cache, r.tags, view.cell(2));
                    reuse_fast_corner<3, 1>(r.cache, r.tags, view.cell(3));
                } else if (di == 1 && dj == 1) {
                    reuse_fast_corner<0, 3>(r.cache, r.tags, view.cell(0));
                } else if (di == 1 && dj == -1) {
                    reuse_fast_corner<1, 2>(r.cache, r.tags, view.cell(1));
                } else if (di == -1 && dj == 1) {
                    reuse_fast_corner<2, 1>(r.cache, r.tags, view.cell(2));
                } else if (di == -1 && dj == -1) {
                    reuse_fast_corner<3, 0>(r.cache, r.tags, view.cell(3));
                }
            }
            const bool any_live =
                !TransportPolicy::thin_layers ||
                layer_live(live_a, live_a_hi, view.ir()) ||
                layer_live(live_a, live_a_hi, view.ir() + 1);
            if (any_live) {
                if (lane == 0) {
                    const float cp = view.sample.cp;
                    warm_cm[0] = 1.f;
                    warm_cm[1] = cp;
                    for (int m = 2; m <= SH_L; ++m)
                        warm_cm[m] = (2.f * cp) * warm_cm[m - 1] - warm_cm[m - 2];
                }
                __syncwarp();
            }
#pragma unroll
            for (int corner = 0; corner < 4; corner++) {
                const int cc = view.cell(corner), ir = view.ir() + corner / 2;
                const bool sk_a = !TransportPolicy::thin_layers || layer_live(live_a, live_a_hi, ir);
                float own_aero = 0.f;
                if (sk_a) {
                    if (r.tags[corner] != cc)
                        contract_corner_paired(r.cache, warm_c32, r.tags, r.y, src_sh, lane, corner, cc);
                    own_aero = reduce_scatter_corner_ptr(
                        r.cache, warm_c32 ? warm_c32 + corner * TW : nullptr, warm_cm, corner, lane);
                }
                if (ownership.active) {
                    const int w = ownership.wave;
                    const float corner_sw = view.weight(corner);
                    float rm[4];
#pragma unroll
                    for (int k = 0; k < 4; ++k)
                        rm[k] = src[((int64_t)cc * TransportPolicy::source_rows + k) * FW + GW(w)];
                    float density = support_value(view.height(), hr[ir], true);
                    float aero = sk_a ? support_value(view.height(), own_aero * ha[ir], false) : 0.f;
                    float tr = (rm[0] + rm[1]) + rm[2], tensors[3];
#pragma unroll
                    for (int component = 0; component < 4; component++) {
                        if (component == 3 && phase == 1)
                            continue;
                        float rot = view.rotate(component, rm);
                        const float puc = r.pu_r[component];
                        float tensor = density * (mix * rot + ((1.f - mix) / 3.f * tr) * puc);
                        tensor += (aero * puc) / 2.f;
                        r.lane_val[component] += corner_sw * tensor;
                        if (component < 3)
                            tensors[component] = tensor;
                    }
                    const float trace = (tensors[0] + tensors[1]) + tensors[2];
                    const bool good = trace >= TRACE_MIN;
                    r.lane_log += good ? corner_sw * logf(trace) : 0.f;
                    r.lane_wg += good ? corner_sw : 0.f;
                    r.lane_am += good ? corner_sw * trace : 0.f;
                }
                __syncwarp();
            }
        } else {
            Record &s = samples[warp];
        if (!first && !NOSH_ON && r.tags[0] >= 0) {
            int di = s.cell[0] / nc - r.tags[0] / nc, dj = s.cell[0] % nc - r.tags[0] % nc;
            if (di == 0 && dj == 1) {
                reuse_corner<0, 1>(s, r.cache, r.scache, r.tags, lane);
                reuse_corner<2, 3>(s, r.cache, r.scache, r.tags, lane);
            } else if (di == 0 && dj == -1) {
                reuse_corner<1, 0>(s, r.cache, r.scache, r.tags, lane);
                reuse_corner<3, 2>(s, r.cache, r.scache, r.tags, lane);
            } else if (di == 1 && dj == 0) {
                reuse_corner<0, 2>(s, r.cache, r.scache, r.tags, lane);
                reuse_corner<1, 3>(s, r.cache, r.scache, r.tags, lane);
            } else if (di == -1 && dj == 0) {
                reuse_corner<2, 0>(s, r.cache, r.scache, r.tags, lane);
                reuse_corner<3, 1>(s, r.cache, r.scache, r.tags, lane);
            } else if (di == 1 && dj == 1) {
                reuse_corner<0, 3>(s, r.cache, r.scache, r.tags, lane);
            } else if (di == 1 && dj == -1) {
                reuse_corner<1, 2>(s, r.cache, r.scache, r.tags, lane);
            } else if (di == -1 && dj == 1) {
                reuse_corner<2, 1>(s, r.cache, r.scache, r.tags, lane);
            } else if (di == -1 && dj == -1) {
                reuse_corner<3, 0>(s, r.cache, r.scache, r.tags, lane);
            }
        }
        if (TransportPolicy::preload && ownership.active) {
            if (!first) {
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    scratch.air[i] = hr[view.ir() + i];
                    scratch.strato[i] = st_density[view.ir() + i];
                }
#pragma unroll
                for (int corner = 0; corner < 4; corner++) {
                    scratch.aero[corner] =
                        aerosol_record(seg, ray, ownership.sample, corner)[ownership.wave];
                    const int64_t pc = view.cell(corner);
#pragma unroll
                    for (int k = 0; k < 4; k++)
                        scratch.rm[corner][k] =
                            src[(pc * SRC_ROWS_ST + k) * FW + GW(ownership.wave)];
                }
            }
            if (sinv_mode == 2)
                scratch.column.preload_from(column_record(seg, ray, ownership.sample));
        }
#pragma unroll
        for (int corner = 0; corner < 4; corner++) {
            int cc = view.cell(corner), ir = view.ir() + corner / 2;
            const bool sk_a = !TransportPolicy::thin_layers || layer_live(live_a, live_a_hi, ir);
            const bool sk_s = !TransportPolicy::thin_layers || layer_live(live_s, live_s_hi, ir);
            if (!first && !NOSH_ON && r.tags[corner] != cc)
                contract_corner(s, r.cache, r.scache, r.tags, r.y, src_sh, lane, corner, cc,
                                nh * nc, sk_a, sk_s);
            // Do not access cache[corner][lane]: retain compile-time wave indexes
            // so the Fourier cache can stay register promoted. Every lane executes
            // every shuffle; the selected owner keeps one reduced wavelength.
            float own_aero = 0.f;
            float own_saero = 0.f;
            if constexpr (TransportPolicy::reduce_scatter) {
                if (!first) {
                    if ((sk_a || (TransportPolicy::combined_sh && sk_s)) && !NOSH_ON)
                        own_aero = reduce_scatter_corner(r.cache, s.c32, s.cm, corner, lane);
                    if (sk_s) {
                        if constexpr (TransportPolicy::strato && !TransportPolicy::combined_sh)
                            own_saero =
                                reduce_scatter_corner(r.scache, s.st_c32, s.cm, corner, lane);
                    }
                }
                if constexpr (TransportPolicy::column_records) {
                    if (!first) {
                        if (SLIM_ON) {
                            if constexpr (TransportPolicy::preload)
                                own_aero = ownership.active ? scratch.aero[corner] : 0.f;
                            else
                                own_aero =
                                    lane < TW ? aerosol_record(seg, ray, step, corner)[lane] : 0.f;
                        } else if (PROD_ON && lane < TW)
                            aerosol_record(seg, ray, step, corner)[lane] = own_aero;
                    }
                }
            } else {
                if (!first) {
#pragma unroll
                    for (int w = 0; w < TW; w++) {
                        float aero = s.cm[lane] * r.cache[corner][w];
                        if (lane == 0)
                            aero += s.cm[32] * s.c32[corner][w];
                        for (int stride = 16; stride; stride >>= 1)
                            aero += __shfl_down_sync(0xffffffff, aero, stride);
                        aero = __shfl_sync(0xffffffff, aero, 0);
                        if (lane == w)
                            own_aero = aero;
                        if constexpr (TransportPolicy::strato) {
                            float saero = s.cm[lane] * r.scache[corner][w];
                            if (lane == 0)
                                saero += s.cm[32] * s.st_c32[corner][w];
                            for (int stride = 16; stride; stride >>= 1)
                                saero += __shfl_down_sync(0xffffffff, saero, stride);
                            saero = __shfl_sync(0xffffffff, saero, 0);
                            if (lane == w)
                                own_saero = saero;
                        }
                    }
                }
            }
            bool bad_trace = false;
            if (ownership.active && !NOOUT_ON) {
                int w = ownership.wave;
                float aero = own_aero;
                const float corner_sw = view.weight(corner);
                float saero = own_saero;
                float rm[4];
                for (int k = 0; k < 4; ++k) {
                    if (first)
                        rm[k] = gm[((int64_t)cc * 4 + k) * FW + GW(w)];
                    else if constexpr (TransportPolicy::preload)
                        rm[k] = scratch.rm[corner][k];
                    else
                        rm[k] = src[((int64_t)cc * TransportPolicy::source_rows + k) * FW + GW(w)];
                }
                float density = first
                                    ? s.prof[0]
                                    : (TransportPolicy::preload ? scratch.air[corner / 2] : hr[ir]);
                if constexpr (!TransportPolicy::combined_sh)
                    if (!first)
                        aero *= ha[ir];
                if constexpr (TransportPolicy::strato)
                    if (!first)
                        saero *=
                            TransportPolicy::preload ? scratch.strato[corner / 2] : st_density[ir];
                if (!first) {
                    density = support_value(view.height(), density, true);
                    aero = support_value(view.height(), aero, false);
                    if constexpr (TransportPolicy::strato) {
                        saero = support_value(view.height(), saero, false);
                    }
                }
                float tr = (rm[0] + rm[1]) + rm[2], tensors[3];
#pragma unroll
                for (int component = 0; component < 4; component++) {
                    if (component == 3 && phase == 1)
                        continue;
                    float rot = view.rotate(component, rm);
                    const float puc = TransportPolicy::warm ? r.pu_r[component] : r.pu[component];
                    float tensor = density * (mix * rot + ((1.f - mix) / 3.f * tr) * puc);
                    tensor += (aero * puc) / 2.f;
                    if constexpr (TransportPolicy::strato) {
                        if (!first)
                            tensor += (saero * puc) / 2.f;
                    }
                    if constexpr (TransportPolicy::warm) {
                        r.lane_val[component] += corner_sw * tensor;
                    } else {
                        s.val[w][component] += s.sw[corner] * tensor;
                    }
                    if (component < 3)
                        tensors[component] = tensor;
                }
                if (!first) {
                    float trace = (tensors[0] + tensors[1]) + tensors[2];
                    if constexpr (TransportPolicy::warm) {
                        // Log-space blend: traces below TRACE_MIN (or NaN) stay out of
                        // the log moments but stay in lane_val.
                        const float cw = corner_sw;
                        const bool good = trace >= TRACE_MIN;
                        r.lane_log += good ? cw * logf(trace) : 0.f;
                        r.lane_wg += good ? cw : 0.f;
                        r.lane_am += good ? cw * trace : 0.f;
                    } else {
                        // K_FIRST=1 only; the !first branch never runs.
                        bad_trace = !(trace >= TRACE_MIN) && s.sw[corner] != 0.f;
                        s.logs[w] += s.sw[corner] * logf(fmaxf(trace, TRACE_MIN));
                    }
                }
            }
            if constexpr (!TransportPolicy::warm) {
                // K_FIRST=1 only; the !first branch never runs.
                uint32_t bad = __ballot_sync(0xffffffff, bad_trace);
                if (lane == 0 && !first)
                    s.positive &= ~bad;
            }
            // Order shared consumers before lane-0 c32 copies / next corner, and
            // complete all val/log writes before the next point/attenuation phase.
            __syncwarp();
        }
        }
    }
    // Advance Simpson to this node or read recorded columns, then accumulate in
    // the original even/odd order. No lane may leave before the paired shuffle.
    template <class View>
    __device__ __forceinline__ void attenuate_sample(RayState &r, SampleScratch &scratch,
                                                     int64_t ray, uint32_t seg, int step, int step0,
                                                     PathScalar seg_a, PathScalar seg_b, bool low,
                                                     const NodeWeights &node,
                                                     const SampleLane &ownership, const View &view) {
        const float sample_ds = view.sample.ds;
        if constexpr (TransportPolicy::fast_warm) {
            const int bi = step - step0;
            float mp[NCOMP];
#pragma unroll
            for (int k = 0; k < NCOMP; ++k)
                mp[k] = __shfl_sync(0xffffffffu, r.lane_mp[k], bi);
            const float dt6 = __shfl_sync(0xffffffffu, r.lane_dt, bi);
            if (ownership.active) {
                const int w = ownership.wave;
                const bool ok = r.lane_wg > 0.f && r.lane_am >= TRACE_MIN;
                const float lane_scale =
                    ok ? expf(r.lane_log - r.lane_wg * logf(r.lane_am / r.lane_wg)) : 1.f;
                const float bk = extinction_at(view.sample.prof, 0.f, optics, GW(w));
                const float bm = extinction_at(mp, 0.f, optics, GW(w));
                r.sp_acc += dt6 * ((r.sp_b + bk) + 4.f * bm);
                r.sp_b = bk;
                const float mid = (r.tau_lane - r.tc_lane) + r.sp_acc;
                const float lane_att = sample_ds * expf(-mid);
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    if (k == 3 && phase == 1)
                        continue;
                    kah(r.lane_rad[k], r.lane_rc[k], (r.lane_val[k] * lane_scale) * lane_att);
                }
            }
        } else {
        Record &s = samples[warp];
        if (ownership.active && !PROD_ON) {
            int w = ownership.wave;
            {
                float lane_scale = 1.f;
                if constexpr (TransportPolicy::warm) {
                    const bool ok = r.lane_wg > 0.f && r.lane_am >= TRACE_MIN;
                    lane_scale =
                        ok ? expf(r.lane_log - r.lane_wg * logf(r.lane_am / r.lane_wg)) : 1.f;
                } else {
                    // K_FIRST=1 only; the !first branch never runs.
                    if (!first) {
                        float tr = (s.val[w][0] + s.val[w][1]) + s.val[w][2];
                        float scale = (s.positive & (1u << w)) && tr >= TRACE_MIN
                                          ? expf(s.logs[w]) / fmaxf(tr, TRACE_MIN)
                                          : 1.f;
                        s.logs[w] = scale;
                    } else
                        s.logs[w] = 1.f;
                }
                // Simpson prefix to this node: panel [previous node or
                // a, this node], node extinction from point()'s profile, new
                // midpoint evaluation; replaces the GL-weight prefix
                // mid=(tau-tc)+dt/2, tau+=dt.
                if (sinv_mode != 2) {
                    float pst = 0.f;
                    if constexpr (TransportPolicy::strato)
                        pst = s.st_prof;
                    DS tk;
                    if constexpr (TransportPolicy::path_ds)
                        tk = node_t(seg_a, seg_b, low, node.u);
                    else
                        tk = node_t(seg_a, seg_b, low, path_scalar(gu[step]));
                    const float bk = extinction_at(s.prof, pst, optics, GW(w));
                    r.sp_acc +=
                        simpson_piece(r.beta_p, r.beta_d, r.sp_t, tk, r.sp_b, bk, optics, GW(w));
                    r.sp_t = tk;
                    r.sp_b = bk;
                }
                float mid = (r.tau_lane - r.tc_lane) + r.sp_acc;
                if constexpr (TransportPolicy::warm) {
                    float lane_att = sample_ds * expf(-mid);
                    if constexpr (TransportPolicy::column_records) {
                        if (sinv_mode == 1) {
                            column_record(seg, ray, step).write_basis(w, mid);
                            // KSPLIT=2 (VPROD): vp_records points to the per-point (c0, cp
                            // bits) int2 records (device memory, or mapped pinned host
                            // memory when the device cannot hold them), bundle-global like
                            // the column rows; 0 (KSPLIT=1) writes nothing. c0 = s.cell[0], cp =
                            // s.cm[1] (point()). RECORD variant only (compiled out of
                            // PROD/SLIM and the unsplit kernel).
                            if (REC_ON && w == 0) {
                                if (vp_records)
                                    vp_records[(int64_t)(seg - ray) * nq + step] =
                                        make_int2(s.cell[0], __float_as_int(s.cm[1]));
                            }
                        }
                        if (sinv_mode == 2) {
                            const ColumnRecord &record = TransportPolicy::preload
                                                             ? scratch.column
                                                             : column_record(seg, ray, step);
                            float tq = 0.f;
                            for (int k = 0; k < NCOMP; ++k)
                                tq += record.component(k) *
                                      (TransportPolicy::preload ? r.ray_ext[k] : ext[k * FW + GW(w)]);
                            tq += record.stratosphere() *
                                  (TransportPolicy::preload ? r.ray_st_ext : st_ext[GW(w)]);
                            lane_att = sample_ds * expf(-tq);
                        }
                    }
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        if (REC_ON)
                            break;
                        if (k == 3 && phase == 1)
                            continue;
                        if constexpr (TransportPolicy::slim_rays) {
                            scratch.paired[k] = (r.lane_val[k] * lane_scale) * lane_att;
                        } else {
                            kah(r.lane_rad[k], r.lane_rc[k],
                                (r.lane_val[k] * lane_scale) * lane_att);
                        }
                    }
                } else {
                    s.att[w] = s.ds * expf(-mid);
                }
            }
        }
        if constexpr (TransportPolicy::slim_rays) {
// The lower half owns the Kahan state and adds the even sample, then the odd
// one, per component (the one-step order). Every lane takes part in the shuffle; an absent odd
// sample adds nothing (no kah(0)).
#pragma unroll
            for (int k = 0; k < 4; k++) {
                if (k == 3 && phase == 1)
                    continue;
                const float v_odd = __shfl_down_sync(0xffffffffu, scratch.paired[k], 16);
                if (lane < TW) {
                    kah(r.lane_rad[k], r.lane_rc[k], scratch.paired[k]);
                    if (ownership.odd)
                        kah(r.lane_rad[k], r.lane_rc[k], v_odd);
                }
            }
        }
        }
    }
    template <class View>
    __device__ __forceinline__ void consume_sample(RayState &r, int64_t ray, uint32_t seg, int step,
                                                   int step0, PathScalar seg_a, PathScalar seg_b,
                                                   bool low, const NodeWeights &node,
                                                   const SampleLane &ownership, const View &view) {
        SampleScratch scratch;
        if constexpr (TransportPolicy::fast_warm) {
            initialize_source(r, ownership);
            shade_corners(r, scratch, ray, seg, step, ownership, view);
            attenuate_sample(r, scratch, ray, seg, step, step0, seg_a, seg_b, low, node, ownership,
                             view);
        } else {
        Record &s = samples[warp];
        SECTION_SYNC(__syncwarp());
        if (lane == 0) {
            s.positive = (1u << TW) - 1u;
        }
        SECTION_BEGIN_IF(first)
        first_light_ground(s, r.y);
        SECTION_END_IF(1, first)
        if constexpr (TransportPolicy::warm || TransportPolicy::strato) {
            // Includes lane-0 STRATO/scalar ground writes before other lanes read.
            SECTION_SYNC(__syncwarp());
        }
        SECTION_BEGIN()
        initialize_source(r, ownership);
        SECTION_END(3)
        SECTION_SYNC(__syncwarp());
        SECTION_BEGIN()
        shade_corners(r, scratch, ray, seg, step, ownership, view);
        SECTION_END(3)
        SECTION_SYNC(__syncwarp());
        SECTION_BEGIN()
        attenuate_sample(r, scratch, ray, seg, step, step0, seg_a, seg_b, low, node, ownership,
                         view);
        SECTION_END(4)
        if constexpr (!TransportPolicy::warm) {
            SECTION_SYNC(__syncwarp());
            if (lane < (phase == 1 ? 3 : 4)) {
#pragma unroll
                for (int w = 0; w < TW; w++) {
                    float value = s.val[w][lane] * s.logs[w];
                    kah(r.rad[4 * w + lane], r.rc[4 * w + lane], value * s.att[w]);
                }
            }
            SECTION_SYNC(__syncwarp());
        }
        }
    }
    // Each segment restarts the one-sided Simpson panel; ray optical depth and
    // Fourier caches persist across segments. Batch tails keep the same GL order.
    __device__ __forceinline__ void integrate_segments(RayState &r, int64_t ray) {
        for (uint32_t seg = edge_off[ray]; seg + 1 < edge_off[ray + 1]; seg++) {
            double a = edges[seg], b = edges[seg + 1];
            if (b - a <= 1e-8)
                continue;
            bool low = (lower[ray] >> (seg - edge_off[ray])) & 1;
            const PathScalar seg_a = path_scalar(a), seg_b = path_scalar(b);
            // Simpson prefix: beta(a) is evaluated anew for every segment (never carried over a
            // skipped b-a<=1e-8 segment).
            r.sp_t = ds(seg_a);
            const DS seg_inside = ds((a + b) * .5);
            r.sp_acc = 0.f;
            float seg_be = 0.f;
            if constexpr (TransportPolicy::fast_warm) {
                float end_p[NCOMP] = {};
                if (lane < 2) {
                    DS q[3], q2[3];
                    const DS te = lane == 0 ? ds(seg_a) : ds(seg_b);
                    float sh = path_h(path_q(r.beta_p, r.beta_d, te, q, q2));
                    if (support_extended()) {
                        const float ih = path_h(path_q(r.beta_p, r.beta_d, seg_inside, q, q2));
                        sh = support_height(sh, ih);
                    }
                    float pst_unused;
                    profile_at(sh, profile, end_p, pst_unused);
                }
                float pa[NCOMP], pb[NCOMP];
#pragma unroll
                for (int k = 0; k < NCOMP; ++k) {
                    pa[k] = __shfl_sync(0xffffffffu, end_p[k], 0);
                    pb[k] = __shfl_sync(0xffffffffu, end_p[k], 1);
                }
                if (lane < TW) {
                    r.sp_b = extinction_at(pa, 0.f, optics, GW(lane));
                    seg_be = extinction_at(pb, 0.f, optics, GW(lane));
                }
            } else if (lane < TW && !PROD_ON && sinv_mode != 2) {
                r.sp_b = beta_at_side(r.beta_p, r.beta_d, r.sp_t, seg_inside, optics, GW(lane));
            }
            constexpr int capacity =
                TransportPolicy::batched_points ? TransportPolicy::batch_size : 1;
            for (int step0 = 0; step0 < nq; step0 += capacity) {
                const int nb = min(capacity, nq - step0);
                if constexpr (TransportPolicy::fast_warm) {
                    __syncwarp();  // every lane has finished reading the previous batch of pts
                    const NodeWeights node =
                        quadrature_node(step0 + lane, nq, lane < nb, gu, gw, r.gu_l, r.gw_l);
                    DS tk = {0.f, 0.f};
                    if (lane < nb)
                        tk = point_warm(pts[lane], r.beta_p, r.beta_d, r.B, r.dl, r.global_st,
                                        seg_a, seg_b, low, node.u, node.w, h, nh, chi, nc, profile);
                    const bool is_last = (step0 + nb == nq);
                    DS prev_t;
                    prev_t.h = __shfl_up_sync(0xffffffffu, tk.h, 1);
                    prev_t.l = __shfl_up_sync(0xffffffffu, tk.l, 1);
                    if (lane == 0)
                        prev_t = r.sp_t;
                    DS curr_t = tk;
                    if (is_last && lane == nb)
                        curr_t = ds(seg_b);
                    const int n_panels = nb + (is_last ? 1 : 0);
                    if (lane < n_panels) {
                        DS tm = da(prev_t, curr_t);
                        tm.h *= .5f;
                        tm.l *= .5f;
                        DS q[3], q2[3];
                        float pst_unused;
                        profile_at(path_h(path_q(r.beta_p, r.beta_d, tm, q, q2)), profile,
                                   r.lane_mp, pst_unused);
                        r.lane_dt = dv(dx(curr_t, prev_t)) / 6.f;
                    }
                    r.sp_t.h = __shfl_sync(0xffffffffu, tk.h, nb - 1);
                    r.sp_t.l = __shfl_sync(0xffffffffu, tk.l, nb - 1);
                    __syncwarp();
                    for (int step = step0; step < step0 + nb; step++) {
                        const SampleLane ownership(lane, step, step0 + nb);
                        auto view = sample_view(pts, nullptr, step0, ownership.sample);
                        consume_sample(r, ray, seg, step, step0, seg_a, seg_b, low, node, ownership,
                                       view);
                    }
                    if (is_last) {
                        float mp_end[NCOMP];
#pragma unroll
                        for (int k = 0; k < NCOMP; ++k)
                            mp_end[k] = __shfl_sync(0xffffffffu, r.lane_mp[k], nb);
                        const float dt6_end = __shfl_sync(0xffffffffu, r.lane_dt, nb);
                        if (lane < TW) {
                            const float bm_end = extinction_at(mp_end, 0.f, optics, GW(lane));
                            r.sp_acc += dt6_end * ((r.sp_b + seg_be) + 4.f * bm_end);
                            kah(r.tau_lane, r.tc_lane, r.sp_acc);
                        }
                    }
                } else {
                if constexpr (TransportPolicy::batched_points) {
                    // All lanes shuffle before the active lanes evaluate their records.
                    const NodeWeights node =
                        quadrature_node(step0 + lane, nq, lane < nb, gu, gw, r.gu_l, r.gw_l);
                    if (lane < nb)
                        evaluate_point(pts[lane], r.beta_p, r.beta_d, r.B, r.dl, r.global_st, seg_a,
                                       seg_b, low, node.u, node.w, h, nh, chi, nc, profile, sun_h,
                                       nsh, sun_col, nsx, first, st_sun_col);
                    __syncwarp();
                }
                for (int step = step0; step < step0 + nb; step += TransportPolicy::sample_stride) {
                    SECTION_BEGIN()
                    const SampleLane ownership(lane, step, step0 + nb);
                    auto view = sample_view(pts, samples + warp, step0, ownership.sample);
                    const NodeWeights node = quadrature_node(
                        step, nq, TransportPolicy::path_ds || lane == 0, gu, gw, r.gu_l, r.gw_l);
                    if constexpr (!TransportPolicy::batched_points) {
                        if (lane == 0)
                            evaluate_point(samples[warp], r.beta_p, r.beta_d, r.B, r.dl,
                                           r.global_st, seg_a, seg_b, low, node.u, node.w, h, nh,
                                           chi, nc, profile, sun_h, nsh, sun_col, nsx, first,
                                           st_sun_col);
                    }
                    SECTION_END(0)
                    consume_sample(r, ray, seg, step, step0, seg_a, seg_b, low, node, ownership,
                                   view);
                }
                }
            }
            // Simpson end piece to b (beta(b) anew), then the segment sum into tau (Kahan), as
            // numpy_reference.py SimpsonPrefix.finish.
            if constexpr (!TransportPolicy::fast_warm) {
                if (lane < TW && !PROD_ON && sinv_mode != 2) {
                    const DS te = ds(seg_b);
                    const float be =
                        beta_at_side(r.beta_p, r.beta_d, te, seg_inside, optics, GW(lane));
                    r.sp_acc +=
                        simpson_piece(r.beta_p, r.beta_d, r.sp_t, te, r.sp_b, be, optics, GW(lane));
                    kah(r.tau_lane, r.tc_lane, r.sp_acc);
                }
            }
        }
    }

    // Shared offsets are the host workspace contract. The views only name those
    // slices; no storage, row stride, alignment or block ownership changes.
    struct SharedViews {
        int nm;
        double *acc, *ground;
        float *Y, *rads, *err, *cosdest;
    };
    __device__ __forceinline__ bool select_cell(int &cell, int &part) const {
        if constexpr (TransportPolicy::split_cells) {
            if (cell >= cells * SPLIT_S)
                return false;
            const int unit = cell_order[cell];
            cell = unit / SPLIT_S;
            part = unit % SPLIT_S;
        } else {
            if constexpr (!TransportPolicy::slim_rays)
                if (cell >= cells)
                    return false;
            if constexpr (TransportPolicy::queue_cells)
                cell = cell_order[cell];
        }
        return true;
    }
    __device__ __forceinline__ void initialize_layers() {
        // Live-layer masks for up to 128 heights (two 64-bit words; 200 km grids have nh=71). The host
        // combined sidecar (sh_layout.combined_coefficients) and sos_vprod zero the same thin layers.
        if constexpr (TransportPolicy::thin_layers) {
            if (nh <= 128) {
                bool la[4] = {}, ls[4] = {};
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const int idx = lane + 32 * b;
                    if (idx < nh) {
                        const float r = hr[idx];
                        la[b] = ha[idx] > SKIP_TAU * r;
                        if constexpr (ProfilePolicy::strato)
                            ls[b] = st_density[idx] > SKIP_TAU * r;
                    }
                }
                live_a = (unsigned long long)__ballot_sync(0xffffffffu, la[0]) |
                         ((unsigned long long)__ballot_sync(0xffffffffu, la[1]) << 32);
                live_a_hi = (unsigned long long)__ballot_sync(0xffffffffu, la[2]) |
                            ((unsigned long long)__ballot_sync(0xffffffffu, la[3]) << 32);
                if constexpr (ProfilePolicy::strato) {
                    live_s = (unsigned long long)__ballot_sync(0xffffffffu, ls[0]) |
                             ((unsigned long long)__ballot_sync(0xffffffffu, ls[1]) << 32);
                    live_s_hi = (unsigned long long)__ballot_sync(0xffffffffu, ls[2]) |
                                ((unsigned long long)__ballot_sync(0xffffffffu, ls[3]) << 32);
                }
            }
        }
    }
    __device__ __forceinline__ SharedViews bind_shared(int cell, int part, double *shared) {
        SharedViews memory;
        memory.nm = phase == 0 ? NMOM * TW : 0;
        if constexpr (TransportPolicy::global_accum) {
            if constexpr (TransportPolicy::split_cells)
                memory.acc = part == 0
                                 ? moments + (int64_t)cell * memory.nm
                                 : split_scratch + ((int64_t)(part - 1) * cells + cell) * memory.nm;
            else
                memory.acc = moments + (int64_t)cell * memory.nm;
            memory.ground = shared;
        } else {
            memory.acc = shared;
            memory.ground = memory.acc + memory.nm;
        }
        memory.Y = (float *)(memory.ground + TW);
        memory.rads = memory.Y + nwarps * YPAD;
        if constexpr (TransportPolicy::fast_warm) {
            memory.err = nullptr;
            memory.cosdest = memory.rads + nwarps * 4 * TW;
            cell_B = memory.cosdest + nwarps * 33;
            float *cursor = cell_B + 12;
            warm_cm = cursor + warp * 33;
            cursor += nwarps * 33;
            if constexpr (SH_L == 32) {
                warm_c32 = cursor + warp * 4 * TW;
                cursor += nwarps * 4 * TW;
            } else {
                warm_c32 = nullptr;
            }
            ray_constants = cursor + warp * RAY_CONST_F;
            cursor += nwarps * RAY_CONST_F;
            PtBatch *pts_base = (PtBatch *)((((unsigned long long)cursor) + 15ull) & ~15ull);
            pts = pts_base + warp * TransportPolicy::batch_size;
            if constexpr (TransportPolicy::sort_rays) {
                rs_key = (unsigned char *)(pts_base + nwarps * TransportPolicy::batch_size);
                rs_perm = rs_key + 256;
            }
            return memory;
        }
        if constexpr (TransportPolicy::fast_first) {
            memory.err = nullptr;
            memory.cosdest = nullptr;
            float *cursor = (float *)(memory.ground + TW);
            FirstPoint *fpts_base = (FirstPoint *)((((unsigned long long)cursor) + 15ull) & ~15ull);
            fpts = fpts_base + warp * FAST_BATCH;
            float *after_fpts = (float *)(fpts_base + nwarps * FAST_BATCH);
            fmp = after_fpts + warp * ((FAST_BATCH + 1) * (NCOMP + 1));
            after_fpts += nwarps * ((FAST_BATCH + 1) * (NCOMP + 1));
            ground_phase = after_fpts + warp * 32;
            after_fpts += nwarps * 32;
            ray_constants = after_fpts + warp * RAY_CONST_F;
            after_fpts += nwarps * RAY_CONST_F;
            fB = after_fpts + warp * 12;
            return memory;
        }
        memory.err = memory.rads + nwarps * 4 * TW;
        memory.cosdest = memory.err + nwarps * 4 * TW;
        samples = (Record *)(memory.cosdest + nwarps * 33);
        ray_constants = (float *)(samples + nwarps) + warp * RAY_CONST_F;
        constexpr int ground_stride =
            TransportPolicy::ground_fused && !TransportPolicy::split_rays && phase != 2 ? 162 : 32;
        ground_phase = (float *)(samples + nwarps) + nwarps * RAY_CONST_F + warp * ground_stride;
        if constexpr (TransportPolicy::batched_points) {
            pts = (PtBatch *)((float *)(samples + nwarps) + nwarps * RAY_CONST_F + nwarps * 32) +
                  warp * TransportPolicy::batch_size;
            if constexpr (TransportPolicy::slim_rays)
                pts =
                    (PtBatch *)((((unsigned long long)(pts - warp * TransportPolicy::batch_size)) +
                                 15) &
                                ~15ull) +
                    warp * TransportPolicy::batch_size;
            if constexpr (TransportPolicy::sort_rays) {
                rs_key = (unsigned char *)(pts - warp * TransportPolicy::batch_size +
                                           nwarps * TransportPolicy::batch_size);
                rs_perm = rs_key + 256;
            }
        }
        return memory;
    }
    __device__ __forceinline__ void initialize_outputs(int cell, const SharedViews &memory) {
        if constexpr (TransportPolicy::split_rays) {
            // Empty cells have no completing ray; their own block publishes zeros.
            if constexpr (!TransportPolicy::slim_rays) {
                if (cell_off[cell] == cell_off[cell + 1]) {
                    double zero = 0.;
                    for (int k = tid; k < memory.nm; k += blockDim.x)
                        memory.acc[k] = zero;
                    for (int w = tid; w < TW; w += blockDim.x)
                        flux[cell * FW + GW(w)] = zero * (albedo / PI);
                }
            }
        } else {
            if constexpr (TransportPolicy::global_accum) {
                for (int k = tid; k < memory.nm + TW; k += blockDim.x) {
                    if (k < memory.nm)
                        memory.acc[k] = 0.;
                    else
                        memory.ground[k - memory.nm] = 0.;
                }
            } else {
                for (int k = tid; k < memory.nm + TW; k += blockDim.x)
                    memory.acc[k] = 0.;
            }
            SECTION_SYNC(__syncthreads());
        }
    }
    // Stable descending segment-count order, with the original slot as tie break.
    // The whole block builds and publishes each window before any warp reads it.
    __device__ __forceinline__ void schedule_window(int64_t window, int count) {
        for (int t = tid; t < count; t += blockDim.x) {
            int64_t ray = window + t;
            rs_key[t] =
                (unsigned char)min((int64_t)255, (int64_t)(edge_off[ray + 1] - edge_off[ray]));
        }
        __syncthreads();
        for (int t = tid; t < count; t += blockDim.x) {
            const int key = rs_key[t];
            int rank = 0;
            for (int j = 0; j < count; ++j)
                rank += (rs_key[j] > key) || (rs_key[j] == key && j < t);
            rs_perm[rank] = (unsigned char)t;
        }
        __syncthreads();
    }
    __device__ __forceinline__ int64_t ordered_ray(int64_t slot, int64_t window) const {
        if constexpr (TransportPolicy::sort_rays)
            return window + (int64_t)rs_perm[slot - window];
        else
            return slot;
    }
    // The last published ray owns projection. Fences precede reads of other
    // warps' radiance; moments retain cell-ray order and FP64 accumulation.
    __device__ __forceinline__ void project_completed_cell(int cell, int64_t ray_lo, int64_t cb,
                                                           int64_t ce, const SharedViews &memory,
                                                           const double *target, float *y,
                                                           float *cd) {

        // Last ray of this cell: project rays cb..ce-1 in order (== original
        // round-then-warp order).
        __threadfence();
        double *mrow = moments + (int64_t)cell * memory.nm, gsum = 0.;
#pragma unroll 1
        for (int64_t r = cb; r < ce; r++) {
            const double *drow = dirs + 3 * r;
            sh_basis(drow[2], lane, y, aa, bb, dd);
            if (lane == 0) {
                double st = sqrt(fmax(0., 1. - drow[2] * drow[2]));
                float cp = st > 1e-12 ? (float)(drow[0] / st) : 1.f;
                cp = clip(cp, -1, 1);
                cd[0] = 1;
                cd[1] = cp;
                for (int m = 2; m <= 32; m++)
                    cd[m] = (2.f * cp) * cd[m - 1] - cd[m - 2];
            }
            __syncwarp();
            const float *rr = ray_rads + (r - ray_lo) * 4 * TW;
            float wt = weights[r];
#pragma unroll 1
            for (int k = lane; k < memory.nm; k += 32) {
                int moment = k / TW, w = k % TW;
                float v;
                if (moment < 4)
                    v = __ldcg(rr + 4 * w + moment) * wt;
                else {
                    int sk = moment - 4;
                    int m = LM.m[sk];
                    float tr =
                        (__ldcg(rr + 4 * w) + __ldcg(rr + 4 * w + 1)) + __ldcg(rr + 4 * w + 2);
                    v = (tr * wt) * (y[sk] * cd[m]);
                }
                double a = r == cb ? 0. : mrow[k];
                a += (double)v;
                mrow[k] = a;
            }
            if (lane < TW) {
                int w = lane;
                double dlz = 0;
                for (int k = 0; k < 3; k++)
                    dlz += target[3 * k + 2] * drow[k];
                float tr = (__ldcg(rr + 4 * w) + __ldcg(rr + 4 * w + 1)) + __ldcg(rr + 4 * w + 2),
                      wg = weights[r] * fmaxf(0.f, (float)dlz);
                gsum += (double)(tr * wg);
            }
            __syncwarp();
        }
        if (lane < TW)
            flux[cell * FW + GW(lane)] = gsum * (albedo / PI);
    }
    __device__ __forceinline__ void publish_ray(int cell, int64_t ray, int64_t ray_lo,
                                                const SharedViews &memory, const double *target,
                                                float *y, float *rad, float *cd) {
        __syncwarp();
        for (int k = lane; k < 4 * TW; k += 32)
            ray_rads[(ray - ray_lo) * 4 * TW + k] = rad[k];
        if constexpr (!TransportPolicy::ray_limit) {
            __threadfence();
            __syncwarp();
            const int64_t cb = cell_off[cell], ce = cell_off[cell + 1];
            int done = 0;
            if (lane == 0)
                done = atomicAdd(ray_sync + 1 + cell, 1);
            done = __shfl_sync(0xffffffff, done, 0);
            if (done == (int)(ce - cb) - 1)
                project_completed_cell(cell, ray_lo, cb, ce, memory, target, y, cd);
        }
    }
    __device__ __forceinline__ int owning_cell(int64_t ray) const {
        int lo = 0, hi = cells;
        while (lo < hi) {
            int mid = (lo + hi + 1) >> 1;
            if (cell_off[mid] <= ray)
                lo = mid;
            else
                hi = mid - 1;
        }
        return lo;
    }
    __device__ __forceinline__ void trace_fast_first_ray(int cell, int64_t ray, int64_t ray_lo) {
        const double *p = starts + 3 * cell, *target = targets + 9 * cell, *d = dirs + 3 * ray;
        float *dl = ray_constants + 1;
        float *pu = ray_constants + 4;
        float *se = ray_constants + 8;
        float *direct = ray_constants + 11;
        DS *beta_p = (DS *)(ray_constants + 15);
        DS *beta_d = beta_p + 3;
        if (lane == 0) {
            for (int k = 0; k < 3; ++k) {
                beta_p[k] = ds(p[k]);
                beta_d[k] = ds(d[k]);
            }
            for (int k = 0; k < 9; ++k)
                fB[k] = (float)target[k];
            for (int j = 0; j < 3; ++j) {
                double x = 0.;
                for (int k = 0; k < 3; ++k)
                    x += target[3 * k + j] * d[k];
                dl[j] = (float)x;
            }
            pu[0] = 1.f - dl[0] * dl[0];
            pu[1] = 1.f - dl[1] * dl[1];
            pu[2] = 1.f - dl[2] * dl[2];
            pu[3] = -dl[0] * dl[2];
            float sun_dot = (dl[0] * fB[6] + dl[1] * fB[7]) + dl[2] * fB[8];
            for (int k = 0; k < 3; ++k)
                se[k] = fB[6 + k] - dl[k] * sun_dot;
            direct[0] = (pu[0] - se[0] * se[0]) / 2.f;
            direct[1] = (pu[1] - se[1] * se[1]) / 2.f;
            direct[2] = (pu[2] - se[2] * se[2]) / 2.f;
            direct[3] = (pu[3] - se[0] * se[2]) / 2.f;
        }
        __syncwarp();
        const float pu_r[4] = {pu[0], pu[1], pu[2], pu[3]};
        const float dir_r[4] = {direct[0], direct[1], direct[2], direct[3]};
        const float sunphase = hg((float)d[2], aerosol_g);
        float ray_ext[NCOMP] = {};
        float ray_asca = 0.f;
        float phase_value = sunphase;
        if (lane < TW) {
#pragma unroll
            for (int k = 0; k < NCOMP; ++k)
                ray_ext[k] = ext[k * FW + GW(lane)];
            ray_asca = asca[GW(lane)];
            if constexpr (TransportPolicy::tropo_mie)
                phase_value = tropo_phase((float)d[2], GW(lane), asca + FW);
        }
        const DS gu_l = (lane < nq && nq <= 32) ? ds(gu[lane]) : DS{0.f, 0.f};
        const DS gw_l = (lane < nq && nq <= 32) ? ds(gw[lane]) : DS{0.f, 0.f};
        float lane_rad[4] = {}, lane_rc[4] = {};
        float tau_lane = 0.f, tc_lane = 0.f;
        const uint32_t sb = edge_off[ray], se_idx = edge_off[ray + 1];
        for (uint32_t seg = sb; seg + 1 < se_idx; ++seg) {
            const double a = edges[seg], b = edges[seg + 1];
            if (b - a <= 1e-8)
                continue;
            const bool low = (lower[ray] >> (seg - sb)) & 1;
            const DS seg_a = ds(a), seg_b = ds(b), seg_inside = ds((a + b) * 0.5);
            DS sp_t = seg_a;
            float sp_acc = 0.f, sp_b = 0.f, seg_be = 0.f;
            float end_p[NCOMP] = {};
            if (lane < 2) {
                DS qv[3], q2[3];
                const DS te = lane == 0 ? seg_a : seg_b;
                float sh = path_h(path_q(beta_p, beta_d, te, qv, q2));
                if (support_extended()) {
                    const float ih = path_h(path_q(beta_p, beta_d, seg_inside, qv, q2));
                    sh = support_height(sh, ih);
                }
                float pst_unused;
                profile_at(sh, profile, end_p, pst_unused);
            }
            float pa[NCOMP], pb[NCOMP];
#pragma unroll
            for (int k = 0; k < NCOMP; ++k) {
                pa[k] = __shfl_sync(0xffffffffu, end_p[k], 0);
                pb[k] = __shfl_sync(0xffffffffu, end_p[k], 1);
            }
            if (lane < TW) {
                float ba = 0.f, bb = 0.f;
#pragma unroll
                for (int k = 0; k < NCOMP; ++k) {
                    ba += pa[k] * ray_ext[k];
                    bb += pb[k] * ray_ext[k];
                }
                sp_b = ba;
                seg_be = bb;
            }
            for (int step0 = 0; step0 < nq; step0 += FAST_BATCH) {
                const int nb = min(FAST_BATCH, nq - step0);
                __syncwarp();  // every lane has finished reading the previous fpts/fmp batch
                const bool is_last = (step0 + nb == nq);
                const NodeWeights node =
                    quadrature_node(step0 + lane, nq, lane < nb, gu, gw, gu_l, gw_l);
                DS tk = {0.f, 0.f};
                if (lane < nb)
                    tk = point_first(fpts[lane], beta_p, beta_d, fB, dl, seg_a, seg_b, low, node.u,
                                     node.w, h, nh, chi, nc, profile, sun_h, nsh, sun_col, nsx);
                DS prev_t;
                prev_t.h = __shfl_up_sync(0xffffffffu, tk.h, 1);
                prev_t.l = __shfl_up_sync(0xffffffffu, tk.l, 1);
                if (lane == 0)
                    prev_t = sp_t;
                DS curr_t = tk;
                if (is_last && lane == nb)
                    curr_t = seg_b;
                const int n_panels = nb + (is_last ? 1 : 0);
                if (lane < n_panels) {
                    DS tm = da(prev_t, curr_t);
                    tm.h *= .5f;
                    tm.l *= .5f;
                    DS qv[3], q2[3];
                    float mp_tmp[NCOMP], pst_unused;
                    profile_at(path_h(path_q(beta_p, beta_d, tm, qv, q2)), profile, mp_tmp,
                               pst_unused);
                    float *row = fmp + lane * (NCOMP + 1);
#pragma unroll
                    for (int k = 0; k < NCOMP; ++k)
                        row[k] = mp_tmp[k];
                    row[NCOMP] = dv(dx(curr_t, prev_t)) / 6.f;
                }
                sp_t.h = __shfl_sync(0xffffffffu, tk.h, nb - 1);
                sp_t.l = __shfl_sync(0xffffffffu, tk.l, nb - 1);
                __syncwarp();
                for (int bi = 0; bi < nb; ++bi) {
                    const FirstPoint &pt = fpts[bi];
                    float ga = 0.f;
#if defined(FAST_FIRST) && defined(HG_ANGLE_BINS) && !defined(TROPO_MIE)
                    if (goff == nullptr) {
                        if (pt.prof[1] != 0.f && lane < TW)
                            ga = ground_hg_at((const float *const *)gval, pt, lane);
                    } else
#endif
                    if (pt.prof[1] != 0.f) {
                        const float pmu = pt.mu, pexd = pt.exd, paey = fabsf(pt.eyd);
#pragma unroll
                        for (int corner = 0; corner < 4; ++corner) {
                            const int cc = pt.cell[corner];
                            const float swc = pt.sw[corner];
                            const int64_t je = goff[cc + 1];
                            for (int64_t j0 = goff[cc]; j0 < je; j0 += 32) {
                                const int64_t j = j0 + lane;
                                if constexpr (!TransportPolicy::tropo_mie) {
                                    if (j < je) {
                                        float m0 = pmu * gdir[3 * j + 2] + pexd * gdir[3 * j],
                                              m1 = paey * gdir[3 * j + 1];
                                        ground_phase[lane] =
                                            (hg(clip(m0 + m1, -1.f, 1.f), aerosol_g) +
                                             hg(clip(m0 - m1, -1.f, 1.f), aerosol_g)) /
                                            2.f;
                                    }
                                    __syncwarp();
                                }
                                const int count = (int)((je - j0) < 32 ? (je - j0) : 32);
                                if (lane < TW) {
                                    const float *gq = gval + j0 * FW + GW(lane);
                                    if constexpr (TransportPolicy::tropo_mie) {
                                        for (int k = 0; k < count; ++k) {
                                            const int64_t jj = j0 + k;
                                            float m0 =
                                                pmu * gdir[3 * jj + 2] + pexd * gdir[3 * jj],
                                                  m1 = paey * gdir[3 * jj + 1];
                                            float ph =
                                                (tropo_phase(m0 + m1, GW(lane), TROPO_LUT) +
                                                 tropo_phase(m0 - m1, GW(lane), TROPO_LUT)) /
                                                2.f;
                                            ga += (swc * ph) * gq[k * FW];
                                        }
                                    } else {
                                        int k = 0;
                                        for (; k + 8 <= count; k += 8) {
                                            float g[8], ph[8];
#pragma unroll
                                            for (int q = 0; q < 8; ++q) {
                                                g[q] = gq[(k + q) * FW];
                                                ph[q] = ground_phase[k + q];
                                            }
#pragma unroll
                                            for (int q = 0; q < 8; ++q)
                                                ga += (swc * ph[q]) * g[q];
                                        }
                                        for (; k < count; ++k)
                                            ga += (swc * ground_phase[k]) * gq[k * FW];
                                    }
                                }
                                if constexpr (!TransportPolicy::tropo_mie)
                                    __syncwarp();
                            }
                        }
                    }
                    if (lane < TW) {
                        float od = 0.f;
#pragma unroll
                        for (int k = 0; k < NCOMP; ++k)
                            od += pt.sc[k] * ray_ext[k];
                        const float sun = pt.lit ? expf(-od) : 0.f;
                        const float beam = (pt.prof[1] * ray_asca) * (sun * phase_value + ga);
                        const float ray_s = (sun * pt.prof[0]) * ray_ext[0];
                        float val[4];
#pragma unroll
                        for (int k = 0; k < 4; ++k)
                            val[k] = ray_s * (mix * (float)(3 / (8 * PI)) * dir_r[k] +
                                              (1.f - mix) * (float)(1 / (8 * PI)) * pu_r[k]) +
                                     (beam * pu_r[k]) / 2.f;
                        const float density = pt.prof[0];
                        const float4 swv = *(const float4 *)pt.sw;
                        const float sw_arr[4] = {swv.x, swv.y, swv.z, swv.w};
                        const int4 cells4 = *(const int4 *)pt.cell;
                        const int cc_arr[4] = {cells4.x, cells4.y, cells4.z, cells4.w};
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            const int cc = cc_arr[c];
                            const float corner_sw = sw_arr[c];
                            float rm[4];
#pragma unroll
                            for (int k = 0; k < 4; ++k)
                                rm[k] = gm[((int64_t)cc * 4 + k) * FW + GW(lane)];
                            const float tr = (rm[0] + rm[1]) + rm[2];
#pragma unroll
                            for (int comp = 0; comp < 4; ++comp) {
                                const float4 tv = *(const float4 *)(pt.T + 4 * comp);
                                float rot = 0.f;
                                rot += tv.x * rm[0];
                                rot += tv.y * rm[1];
                                rot += tv.z * rm[2];
                                rot += tv.w * rm[3];
                                const float tensor =
                                    density * (mix * rot + ((1.f - mix) / 3.f * tr) * pu_r[comp]);
                                val[comp] += corner_sw * tensor;
                            }
                        }
                        const float *mp_row = fmp + bi * (NCOMP + 1);
                        float bk = 0.f, bm = 0.f;
#pragma unroll
                        for (int k = 0; k < NCOMP; ++k) {
                            bk += pt.prof[k] * ray_ext[k];
                            bm += mp_row[k] * ray_ext[k];
                        }
                        sp_acc += mp_row[NCOMP] * ((sp_b + bk) + 4.f * bm);
                        sp_b = bk;
                        const float att = pt.ds * expf(-((tau_lane - tc_lane) + sp_acc));
#pragma unroll
                        for (int k = 0; k < 4; ++k)
                            kah(lane_rad[k], lane_rc[k], val[k] * att);
                    }
                }
                if (is_last && lane < TW) {
                    const float *mp_end = fmp + nb * (NCOMP + 1);
                    float bm_end = 0.f;
#pragma unroll
                    for (int k = 0; k < NCOMP; ++k)
                        bm_end += mp_end[k] * ray_ext[k];
                    sp_acc += mp_end[NCOMP] * ((sp_b + seg_be) + 4.f * bm_end);
                    kah(tau_lane, tc_lane, sp_acc);
                }
            }
        }
        if (lane < TW) {
            float4 *dst = (float4 *)(ray_rads + (ray - ray_lo) * (4 * TW) + 4 * lane);
            *dst = make_float4(lane_rad[0], lane_rad[1], lane_rad[2], lane_rad[3]);
        }
    }
    // Each warp claims a queue slot. A ray-limit launch stops at the host's
    // boundary and leaves completion/projection to the selected output policy.
    __device__ __forceinline__ void trace_queued_rays(const SharedViews &memory) {
        const int64_t ray_lo = cell_off[0], ray_hi = cell_off[cells];
        int limit = 0;
        if constexpr (TransportPolicy::ray_limit)
            limit = ray_sync[1 + cells];
        for (;;) {
            int q = 0;
            if (lane == 0)
                q = atomicAdd(ray_sync, 1);
            q = __shfl_sync(0xffffffff, q, 0);
            if ((TransportPolicy::ray_limit && q >= limit) || ray_lo + q >= ray_hi)
                break;
            const int64_t ray = ray_lo + q;
            const int cell = owning_cell(ray);
            if constexpr (TransportPolicy::fast_first) {
                trace_fast_first_ray(cell, ray, ray_lo);
            } else {
                const double *p = starts + 3 * cell, *target = targets + 9 * cell;
                float B[9];
                for (int k = 0; k < 9; ++k)
                    B[k] = (float)target[k];
                float *y = memory.Y + warp * YPAD, *rad = memory.rads + warp * 4 * TW;
                float *rc = memory.err + warp * 4 * TW, *cd = memory.cosdest + warp * 33;
                trace_ray(cell, ray, p, target, B, y, rad, rc, cd);
                publish_ray(cell, ray, ray_lo, memory, target, y, rad, cd);
            }
        }
    }
    // Consume one complete round in warp order. Warm moments keep an FP32
    // partial followed by one FP64 add; first light keeps per-term FP64 adds.
    __device__ __forceinline__ void accumulate_round(int64_t round, int64_t end, int64_t window,
                                                     const SharedViews &memory,
                                                     const double *target) {

        SECTION_BEGIN()
        for (int k = tid; k < memory.nm; k += blockDim.x) {
            int moment = k / TW, w = k % TW;
            double acc_k = 0.;
            if constexpr (TransportPolicy::global_accum)
                acc_k = memory.acc[k];
            double &acc_slot = TransportPolicy::global_accum ? acc_k : memory.acc[k];
            float acc_part = 0.f;
            for (int wp = 0; wp < nwarps; wp++)
                if (round + wp < end) {
                    float *rr = memory.rads + wp * 4 * TW;
                    float wt = weights[ordered_ray(round + wp, window)], v;
                    if (moment < 4)
                        v = rr[4 * w + moment] * wt;
                    else {
                        int sk = moment - 4;
                        int m = LM.m[sk];
                        float tr = (rr[4 * w] + rr[4 * w + 1]) + rr[4 * w + 2];
                        v = (tr * wt) * (memory.Y[wp * YPAD + sk] * memory.cosdest[wp * 33 + m]);
                    }
                    if constexpr (TransportPolicy::warm)
                        acc_part += v;
                    else
                        acc_slot += (double)v;
                }
            if constexpr (!K_FIRST) {
                acc_slot += (double)acc_part;
            }
            if constexpr (TransportPolicy::global_accum) {
                memory.acc[k] = acc_k;
            }
        }
        for (int w = tid; w < TW; w += blockDim.x)
            for (int wp = 0; wp < nwarps; wp++)
                if (round + wp < end) {
                    int64_t rridx = ordered_ray(round + wp, window);
                    const double *d = dirs + 3 * rridx;
                    double dlz = 0;
                    for (int k = 0; k < 3; k++)
                        dlz += target[3 * k + 2] * d[k];
                    float *rr = memory.rads + wp * 4 * TW,
                          tr = (rr[4 * w] + rr[4 * w + 1]) + rr[4 * w + 2],
                          wt = weights[rridx] * fmaxf(0.f, (float)dlz);
                    memory.ground[w] += (double)(tr * wt);
                }
        SECTION_END(5)
    }
    __device__ __forceinline__ void trace_cell_rays(int cell, int part, const SharedViews &memory) {
        const double *p = starts + 3 * cell, *target = targets + 9 * cell;
        const float *B_ptr;
        float B_local[9];
        if constexpr (TransportPolicy::fast_warm) {
            if (tid < 9)
                cell_B[tid] = (float)target[tid];
            __syncthreads();
            B_ptr = cell_B;
        } else {
            for (int k = 0; k < 9; ++k)
                B_local[k] = (float)target[k];
            B_ptr = B_local;
        }
        int64_t begin = cell_off[cell], end = cell_off[cell + 1];
        if constexpr (TransportPolicy::split_cells) {
            const int64_t count = end - begin;
            const int64_t b = begin + count * part / SPLIT_S;
            const int64_t e = begin + count * (part + 1) / SPLIT_S;
            begin = b;
            end = e;
        }
        for (int64_t round = begin; round < end; round += nwarps) {
            int64_t window = 0;
            if constexpr (TransportPolicy::sort_rays) {
                const int width = 32 * nwarps;
                window = begin + ((round - begin) / width) * width;
                const int count = (int)min((int64_t)width, end - window);
                if (round == window)
                    schedule_window(window, count);
            }
            int64_t ray = round + warp;
            if constexpr (TransportPolicy::sort_rays)
                ray = ray < end ? ordered_ray(ray, window) : end;
            float *y = memory.Y + warp * YPAD, *rad = memory.rads + warp * 4 * TW;
            float *rc = memory.err ? memory.err + warp * 4 * TW : nullptr;
            float *cd = memory.cosdest + warp * 33;
            if (ray < end)
                trace_ray(cell, ray, p, target, B_ptr, y, rad, rc, cd);
            // Phase 2 has only warp-private consumers; other phases consume all
            // warps' radiance and therefore require both block barriers.
            if (phase == 2)
                SECTION_SYNC(__syncwarp());
            else
                SECTION_SYNC(__syncthreads());
            if (phase != 2 && !NOOUT_ON)
                accumulate_round(round, end, window, memory, target);
            if (phase != 2)
                SECTION_SYNC(__syncthreads());
        }
    }
    __device__ __forceinline__ void finish_cell(int cell, int split_part,
                                                const SharedViews &memory) {
        if constexpr (!TransportPolicy::global_accum) {
            if (phase == 0 && !NOOUT_ON)
                for (int k = tid; k < memory.nm; k += blockDim.x)
                    moments[(int64_t)cell * memory.nm + k] = memory.acc[k];
        }
        if constexpr (!TransportPolicy::split_rays) {

            if constexpr (TransportPolicy::split_cells) {
                if (phase != 2 && !NOOUT_ON) {
                    for (int w = tid; w < TW; w += blockDim.x)
                        split_flux[((int64_t)split_part * cells + cell) * TW + w] =
                            memory.ground[w];
                    __threadfence();
                    __syncthreads();
                    int split_last = 0;
                    if (tid == 0)
                        split_last = atomicAdd(split_count + cell, 1) == SPLIT_S - 1;
                    split_last = __syncthreads_or(split_last);
                    if (split_last) {
                        __threadfence();
                        for (int w = tid; w < TW; w += blockDim.x) {
                            double g = __ldcg(split_flux + (int64_t)cell * TW + w);
                            for (int q = 1; q < SPLIT_S; q++)
                                g += __ldcg(split_flux + ((int64_t)q * cells + cell) * TW + w);
                            flux[cell * FW + GW(w)] = g * (albedo / PI);
                        }
                        if (phase == 0)
                            for (int k = tid; k < memory.nm; k += blockDim.x) {
                                double v = __ldcg(moments + (int64_t)cell * memory.nm + k);
                                for (int q = 1; q < SPLIT_S; q++)
                                    v += __ldcg(split_scratch +
                                                ((int64_t)(q - 1) * cells + cell) * memory.nm + k);
                                moments[(int64_t)cell * memory.nm + k] = v;
                            }
                    }
                }
            } else {
                if (phase != 2 && !NOOUT_ON)
                    for (int w = tid; w < TW; w += blockDim.x)
                        flux[cell * FW + GW(w)] = memory.ground[w] * (albedo / PI);
            }
        }
    }
    __device__ __forceinline__ void run() {
        profile = ProfileParams{height,      sun_h + nsh, profile_table, profile_mask,
                                air_surface, st_center,   st_sigma,      st_norm};
        optics = ExtinctionParams{profile, ext, st_ext};
        if constexpr (TransportPolicy::column_records) {
            if constexpr (TransportPolicy::slim_rays)
                sinv_mode = 2;
            else
                sinv_mode = (!first && phase != 2) ? ((const int *)query)[0] : 0;
            column_rows = ColumnRecord::rows(query);
            slim_base = ((const int64_t *)query)[2];
        }
        int cell = blockIdx.x, split_part = 0;
        if (!select_cell(cell, split_part))
            return;
        tid = threadIdx.x;
        lane = tid & 31;
        warp = tid >> 5;
        nwarps = blockDim.x >> 5;
        initialize_layers();
#ifdef EXP_SECTION_CLOCKS
        if (lane == 0)
            total_begin = clock64();
#endif
        extern __shared__ double shared[];
        const SharedViews memory = bind_shared(cell, split_part, shared);
        initialize_outputs(cell, memory);
        if constexpr (TransportPolicy::split_rays)
            trace_queued_rays(memory);
        else
            trace_cell_rays(cell, split_part, memory);
        finish_cell(cell, split_part, memory);
#ifdef EXP_SECTION_CLOCKS
        if (lane == 0) {
            section_cycles[7] = clock64() - total_begin;
            int64_t slot = (((int64_t)cell * (FW / TW) + (int)blockIdx.y) * nwarps + warp) * 8;
            for (int k = 0; k < 8; k++)
                section_clocks[slot + k] = section_cycles[k];
        }
#endif
    }
};

extern "C" __global__ void SOS_LAUNCH_BOUNDS sos_fused(
    int cells, int nh, int nc, int nq, int nsh, int nsx, int first_arg, int phase_arg,
    int boundary_arg, const int64_t *cell_off, const double *starts, const double *targets,
    const double *dirs, const float *weights, const uint32_t *edge_off, const double *edges,
    const uint16_t *lower, const int32_t *end_j, const float *end_f, const float *end_col,
    const uint8_t *hits, const double *gu, const double *gw, const double *aa, const double *bb,
    const double *dd, const float *h, const float *chi, const float *hr, const float *ha,
    const float *ext, const float *asca, const float *sun_h, const float *sun_col, const float *gm,
    const int64_t *goff, const float *gdir, const float *gval, float height, float aerosol_g,
    float mix, double albedo, const float *src, const double *boundary, double *moments,
    double *flux,
    float *query
#ifdef SINV
    // Explicit allocation pointers are part of the launch ABI.
    ,
    float *slim_aero, int2 *vp_records, float *slim_dump
#endif
    ,
    const float *src_sh
#ifdef STRATO
    ,
    const float *st_ext, const float *st_sca, const float *st_chi, const float *st_density,
    const float *st_sun_col, const float *st_end_col, float st_center, float st_sigma, float st_norm
#endif
    ,
    const float *profile_table, int profile_mask, float air_surface
#ifdef LPT_ACTIVE
    ,
    const int *cell_order
#ifdef SPLIT_ACTIVE
    ,
    double *split_scratch, double *split_flux,
    int *split_count
#endif
#endif
        SECTION_PARAM
#ifdef FL_SPLIT_ON
    ,
    float *ray_rads, int *ray_sync
#endif
) {
    Transport<Sample> transport;
    transport.cells = cells;
    transport.nh = nh;
    transport.nc = nc;
    transport.nq = nq;
    transport.nsh = nsh;
    transport.nsx = nsx;
    transport.first_arg = first_arg;
    transport.phase_arg = phase_arg;
    transport.boundary_arg = boundary_arg;
    transport.cell_off = cell_off;
    transport.starts = starts;
    transport.targets = targets;
    transport.dirs = dirs;
    transport.weights = weights;
    transport.edge_off = edge_off;
    transport.edges = edges;
    transport.lower = lower;
    transport.end_j = end_j;
    transport.end_f = end_f;
    transport.end_col = end_col;
    transport.hits = hits;
    transport.gu = gu;
    transport.gw = gw;
    transport.aa = aa;
    transport.bb = bb;
    transport.dd = dd;
    transport.h = h;
    transport.chi = chi;
    transport.hr = hr;
    transport.ha = ha;
    transport.ext = ext;
    transport.asca = asca;
    transport.sun_h = sun_h;
    transport.sun_col = sun_col;
    transport.gm = gm;
    transport.goff = goff;
    transport.gdir = gdir;
    transport.gval = gval;
    transport.height = height;
    transport.aerosol_g = aerosol_g;
    transport.mix = mix;
    transport.albedo = albedo;
    transport.src = src;
    transport.boundary = boundary;
    transport.moments = moments;
    transport.flux = flux;
    transport.query = query;
#ifdef SINV
    transport.slim_aero = slim_aero;
    transport.vp_records = vp_records;
    transport.slim_dump = slim_dump;
#endif
    transport.src_sh = src_sh;
#ifdef STRATO
    transport.st_ext = st_ext;
    transport.st_sca = st_sca;
    transport.st_chi = st_chi;
    transport.st_density = st_density;
    transport.st_sun_col = st_sun_col;
    transport.st_end_col = st_end_col;
    transport.st_center = st_center;
    transport.st_sigma = st_sigma;
    transport.st_norm = st_norm;
#endif
    transport.profile_table = profile_table;
    transport.profile_mask = profile_mask;
    transport.air_surface = air_surface;
#ifdef LPT_ACTIVE
    transport.cell_order = cell_order;
#ifdef SPLIT_ACTIVE
    transport.split_scratch = split_scratch;
    transport.split_flux = split_flux;
    transport.split_count = split_count;
#endif
#endif
#ifdef EXP_SECTION_CLOCKS
    transport.section_clocks = section_clocks;
#endif
#ifdef FL_SPLIT_ON
    transport.ray_rads = ray_rads;
    transport.ray_sync = ray_sync;
#endif
    transport.run();
}

// ---- Ray projection ----

// Populate the projection tile after its direction weights are published.
// Rows 0..3 are weighted components; rows 4/5 are weighted trace/ground flux.
template <int Threads, int Rays>
__device__ __forceinline__ void
projection_radiance(int tid, int count, int64_t begin, int64_t ray_base, const float *ray_rads,
                    const float *weights, const float (&ground_weight)[Rays],
                    float (&rad)[Rays][6 * TW]) {
    for (int i = tid; i < count * TW; i += Threads) {
        const int r = i / TW, w = i % TW;
        const float *rr = ray_rads + (begin + r - ray_base) * 4 * TW + 4 * w;
        const float weight = weights[begin + r];
        const float r0 = rr[0], r1 = rr[1], r2 = rr[2], r3 = rr[3], trace = (r0 + r1) + r2;
        rad[r][w] = r0 * weight;
        rad[r][TW + w] = r1 * weight;
        rad[r][2 * TW + w] = r2 * weight;
        rad[r][3 * TW + w] = r3 * weight;
        rad[r][4 * TW + w] = trace * weight;
        rad[r][5 * TW + w] = trace * ground_weight[r];
    }
}

#if defined(FL_SPLIT_ON) && defined(LAUNCH_RAY_LIMIT)
// Batch eight rays: every warp builds one basis row, all threads load weighted radiance,
// and each output uses an FP32 batch partial followed by one FP64 addition. No atomics.
#define FLP_THREADS 256
#define FLP_RAYS 8
#define FLP_WAVES ((TW + 1) / 2)
#define FLP_MOMENTS ((NMOM + 127) / 128)
extern "C" __global__ void __launch_bounds__(FLP_THREADS)
    fl_project(int cells, long long ray_lo, const int64_t *cell_off, const double *targets,
               const double *dirs, const float *weights, const double *aa, const double *bb,
               const double *dd, const float *ray_rads, double albedo, double *moments,
               double *flux) {
    const int cell = blockIdx.x;
    if (cell >= cells)
        return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nm = NMOM * TW;
    const int mo = tid % 128, wbase = (tid / 128) * FLP_WAVES;
    __shared__ float y[FLP_RAYS][K], cd[FLP_RAYS][33];
    __shared__ float rad[FLP_RAYS][6 * TW], wg[FLP_RAYS];
    const int64_t cb = cell_off[cell], ce = cell_off[cell + 1];
    const double *target = targets + 9 * cell;
    double acc[FLP_MOMENTS][FLP_WAVES] = {}, gsum = 0.;
#pragma unroll 1
    for (int64_t rb = cb; rb < ce; rb += FLP_RAYS) {
        const int nr = (int)min((int64_t)FLP_RAYS, ce - rb);
        if (warp < nr) {
            const double *d = dirs + 3 * (rb + warp);
            sh_basis(d[2], lane, y[warp], aa, bb, dd);
            if (lane == 0) {
                double st = sqrt(fmax(0., 1. - d[2] * d[2]));
                float cp = st > 1e-12 ? (float)(d[0] / st) : 1.f;
                cp = clip(cp, -1, 1);
                cd[warp][0] = 1;
                cd[warp][1] = cp;
                for (int m = 2; m <= 32; m++)
                    cd[warp][m] = (2.f * cp) * cd[warp][m - 1] - cd[warp][m - 2];
                double dlz = 0;
                for (int k = 0; k < 3; k++)
                    dlz += target[3 * k + 2] * d[k];
                wg[warp] = weights[rb + warp] * fmaxf(0.f, (float)dlz);
            }
        }
        __syncthreads();
        projection_radiance<FLP_THREADS>(tid, nr, rb, ray_lo, ray_rads, weights, wg, rad);
        __syncthreads();
#pragma unroll
        for (int j = 0; j < FLP_MOMENTS; j++) {
            const int moment = mo + j * 128, sk = moment - 4;
            if (moment < NMOM) {
                float part[FLP_WAVES] = {};
                int m = 0;
                if (moment >= 4) {
                    m = LM.m[sk];
                }
                for (int r = 0; r < nr; r++) {
                    const float yc = moment < 4 ? 1.f : y[r][sk] * cd[r][m];
#pragma unroll
                    for (int q = 0; q < FLP_WAVES; q++)
                        if (wbase + q < TW)
                            part[q] += rad[r][(moment < 4 ? moment : 4) * TW + wbase + q] * yc;
                }
#pragma unroll
                for (int q = 0; q < FLP_WAVES; q++)
                    acc[j][q] += (double)part[q];
            }
        }
        if (tid < TW) {
            float part = 0.f;
            for (int r = 0; r < nr; r++)
                part += rad[r][5 * TW + tid];
            gsum += (double)part;
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < FLP_MOMENTS; j++)
        if (mo + j * 128 < NMOM) {
#pragma unroll
            for (int q = 0; q < FLP_WAVES; q++)
                if (wbase + q < TW)
                    moments[(int64_t)cell * nm + (mo + j * 128) * TW + wbase + q] = acc[j][q];
        }
    if (tid < TW)
        flux[cell * FW + GW(tid)] = gsum * (albedo / PI);
}

#endif

// ---- Column-record aerosol producer ----

#if (defined(SINV) && PROD_ON && defined(STRATO) && FW == 15 && !K_FIRST && SH_L <= 16) ||         \
    defined(FAST_VPROD_ON)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define VP_CP_ASYNC
#endif
__device__ __forceinline__ void vp_copy4(float *dst, const float *src) {
#ifdef VP_CP_ASYNC
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned)__cvta_generic_to_shared(dst)),
        "l"(src)
        : "memory");
#else
    *dst = *src;
#endif
}
__device__ __forceinline__ void vp_copy8(float2 *dst, const float2 *src) {
#ifdef VP_CP_ASYNC
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"((unsigned)__cvta_generic_to_shared(dst)),
        "l"(src)
        : "memory");
#else
    *dst = *src;
#endif
}
__device__ __forceinline__ void
vp_copy_wait() { // commit this thread's copies and wait for all of them
#ifdef VP_CP_ASYNC
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
#endif
}
// y rows of visitors tb..tb+nv-1 (nv <= 32) into vy[j*K+k]: lane j reads visitor
// tb+j's ray once, each warp copies whole rows (warp-uniform j, coalesced k).
__device__ __forceinline__ void vp_fill(float *vy, const float *yb, const int *vis_ray, int tb,
                                        int nv, int lane, int wp, int nwp) {
    const int ray = lane < nv ? vis_ray[tb + lane] : 0;
    for (int j = wp; j < nv; j += nwp) {
        const float *src = yb + (int64_t)__shfl_sync(0xffffffffu, ray, j) * K;
        float *dst = vy + j * K;
        for (int k = lane; k < K; k += 32)
            vp_copy4(dst + k, src + k);
    }
}
extern "C" __global__ void __launch_bounds__(128)
    sos_vprod_y(int nray, const double *dirs, const float2 *aa2, const float2 *bb2,
                const float2 *dd2, float *yout) {
    const int lane = threadIdx.x & 31,
              t = (int)blockIdx.x * ((int)blockDim.x >> 5) + ((int)threadIdx.x >> 5);
    if (t >= nray)
        return;
    const double *d = dirs + 3 * (int64_t)t;
    sh_basis_ds(d[2], lane, yout + (int64_t)t * K, aa2, bb2, dd2);
}
#endif

#if defined(SINV) && PROD_ON && defined(STRATO) && FW == 15 && !K_FIRST
#if SH_L <= 16
#define VP_TILE 32
// Stage 2 of one record for quad q (waves 4q..4q+3): cm recurrence from the record's cp, the fixed
// RS2-order butterfly over m, one float4 store into the AERO_W row (q=3, i=3 is the pad wave: its C
// column is zero, so it stores 0).
__device__ __forceinline__ void vp_record(int sl, float cp, bool live,
                                          const float (&a)[SH_L + 1][4], int q, float *aero) {
    float4 *out = (float4 *)(aero + (int64_t)sl * AERO_W +
                             4 * q); // 16 B aligned: AERO_W*4 B rows, quad q at +16q B
    if (!live) {
        *out = make_float4(0.f, 0.f, 0.f, 0.f);
        return;
    }
    float cm[33];
    cm[0] = 1;
    cm[1] = cp;
#pragma unroll
    for (int m = 2; m < 33; m++)
        cm[m] = (2.f * cp) * cm[m - 1] - cm[m - 2];
    const float c32t = 0.f; // SH_L<=16: no m=32 coefficient
    float res[4];           // quad 3, i=3 is the pad wave: its C column is zero, so it stores 0
#pragma unroll
    for (int i = 0; i < 4; i++) {
        float p[16];
#pragma unroll
        for (int j = 0; j < 16; j++) {
            // Ordinary products permit FMA contraction under --fmad=true; the reduction tree is
            // unchanged. VPROD/PROD agreement uses the numerical tolerance requirement, not
            // separate product rounding.
            float lo = (j <= SH_L) ? cm[j] * a[min(j, SH_L)][i] : 0.f;
            if (j == 0)
                lo += c32t;
            const float hi = (j + 16 <= SH_L) ? cm[j + 16] * a[min(j + 16, SH_L)][i] : 0.f;
            p[j] = lo + hi;
        }
#pragma unroll
        for (int j = 0; j < 8; j++)
            p[j] = p[j] + p[j + 8];
#pragma unroll
        for (int j = 0; j < 4; j++)
            p[j] = p[j] + p[j + 4];
        p[0] = p[0] + p[2];
        p[1] = p[1] + p[3];
        res[i] = p[0] + p[1];
    }
    *out = make_float4(res[0], res[1], res[2], res[3]);
}
__device__ __forceinline__ void vp_load_coefficients(float4 *plane, const float *src_sh, int cell,
                                                     int tid) {
    float *dst = (float *)plane;
    const float2 *src = (const float2 *)src_sh + (int64_t)cell * ((FW + 1) / 2) * KPAD;
    for (int i = tid; i < 8 * K; i += blockDim.x) {
        const int pair = i / K, k = i - pair * K;
        vp_copy8((float2 *)(dst + k * 16 + 2 * pair), src + pair * KPAD + k);
    }
}
__device__ __forceinline__ void vp_contract(const float4 *plane, const float *y, int quad,
                                            bool live, float (&a)[SH_L + 1][4]) {
#pragma unroll
    for (int m = 0; m <= SH_L; ++m) {
        a[m][0] = 0.f;
        a[m][1] = 0.f;
        a[m][2] = 0.f;
        a[m][3] = 0.f;
    }
    if (live) {
#pragma unroll
        for (int l = 0; l <= SH_L; ++l) {
#pragma unroll
            for (int m = 0; m <= l; ++m) {
                const int k = ix(l, m);
                const float basis = y[k];
                const float4 c = plane[k * 4 + quad];
                a[m][0] += basis * c.x;
                a[m][1] += basis * c.y;
                a[m][2] += basis * c.z;
                a[m][3] += basis * c.w;
            }
        }
    }
}
// Independent slot/cp loads precede each batch's stores; pad waves stay zero.
__device__ __forceinline__ void vp_publish_records(int begin, int end, bool live,
                                                   const int *rec_slot, const int2 *c0cp,
                                                   const float (&a)[SH_L + 1][4], int quad,
                                                   float *aero) {
    for (int rb = begin; rb < end; rb += 8) {
        const int count = min(8, end - rb);
        int slot[8], cp[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
            slot[i] = i < count ? rec_slot[rb + i] : 0;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            cp[i] = (live && i < count) ? c0cp[slot[i] >> 2].y : 0;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (i < count)
                vp_record(slot[i], __int_as_float(cp[i]), live, a, quad, aero);
    }
}

extern "C" __global__ void __launch_bounds__(128)
    sos_vprod(const int *blk, const int *vis_rbeg, const int *vis_ray, const int *rec_slot,
              const int2 *c0cp, const float *yb, const float *src_sh, const float *hr,
              const float *ha, const float *st_density, int nh, int nc, float *aero) {
    __shared__ float4
        vp_c[K * 4]; // [k][wave quad]: combined SH plane of the block's cell (wave 15 = pad 0)
    __shared__ float
        vp_y[VP_TILE * K]; // y rows of the tile's visitors (row stride K: conflict-free for odd K)
    const int b = blockIdx.x, cell = blk[3 * b], v0 = blk[3 * b + 1], v1 = blk[3 * b + 2];
    const int tid = threadIdx.x, vl = tid >> 2, q = tid & 3, lane = tid & 31, wp = tid >> 5,
              nwp = (int)blockDim.x >> 5;
    bool live = true;
    if constexpr (TransportPolicy::thin_layers) {
        const int ir = cell / nc;
        if (nh <= 128) {  // same height range as Transport::initialize_layers and combined_coefficients
            const float r = hr[ir];
            live = (ha[ir] > SKIP_TAU * r) || (st_density[ir] > SKIP_TAU * r);
        }
    }
    if (live) {
        vp_load_coefficients(vp_c, src_sh, cell, tid);
        if (v0 < v1)
            vp_fill(vp_y, yb, vis_ray, v0, min(VP_TILE, v1 - v0), lane, wp, nwp);
    }
    for (int tb = v0; tb < v1; tb += VP_TILE) {
        const int nv = min(VP_TILE, v1 - tb);
        if (live)
            vp_copy_wait();
        __syncthreads();
        const int v = tb + vl;
        int r0 = 0, r1 = 0;
        float a[SH_L + 1][4];
        if (vl < nv) {
            r0 = vis_rbeg[v];
            r1 = vis_rbeg[v + 1];
            vp_contract(vp_c, vp_y + vl * K, q, live, a);
        }
        __syncthreads(); // vp_y is free: the next tile's rows load while this tile's records are
                         // written
        if (live && tb + VP_TILE < v1)
            vp_fill(vp_y, yb, vis_ray, tb + VP_TILE, min(VP_TILE, v1 - tb - VP_TILE), lane, wp,
                    nwp);
        if (vl < nv)
            vp_publish_records(r0, r1, live, rec_slot, c0cp, a, q, aero);
    }
    if (live && v0 >= v1)
        vp_copy_wait(); // no tile: the C-plane copies still complete before exit
}
#endif
#endif

#ifdef FAST_VPROD_ON
// Build point-to-cell records per chunk and integrate the Simpson prefix in the ray consumer.
extern "C" __global__ void __launch_bounds__(FAST_THREADS)
    vp_c0cp_chunk(int cells, int nh, int nc, int nq, long long r0, int nray, long long base,
                  const int64_t *cell_off, const double *starts, const double *dirs,
                  const uint32_t *edge_off, const double *edges, const uint16_t *lower,
                  const double *gu, const float *h, const float *chi, const float *hr,
                  const float *ha, int2 *c0cp) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int q = (int)blockIdx.x * (FAST_THREADS / 32) + warp;
    if (q >= nray)
        return;
    const int64_t ray = r0 + q;
    unsigned long long live_a = ~0ull, live_a_hi = ~0ull;
    if (FW != 1 && nh <= 128) {
        bool la[4] = {};
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int idx = lane + 32 * b;
            if (idx < nh)
                la[b] = ha[idx] > SKIP_TAU * hr[idx];
        }
        const uint32_t m0 = __ballot_sync(0xffffffffu, la[0]);
        const uint32_t m1 = __ballot_sync(0xffffffffu, la[1]);
        const uint32_t m2 = __ballot_sync(0xffffffffu, la[2]);
        const uint32_t m3 = __ballot_sync(0xffffffffu, la[3]);
        live_a = (unsigned long long)m0 | ((unsigned long long)m1 << 32);
        live_a_hi = (unsigned long long)m2 | ((unsigned long long)m3 << 32);
    }
    int cell = 0;
    if (lane == 0) {
        int lo = 0, hi = cells - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) >> 1;
            if (cell_off[mid] <= ray)
                lo = mid;
            else
                hi = mid - 1;
        }
        cell = lo;
    }
    cell = __shfl_sync(0xffffffffu, cell, 0);
    __shared__ float sh_ray[FAST_THREADS / 32][13];
    if (lane == 0) {
        const double *p = starts + 3 * cell;
        const double *d = dirs + 3 * ray;
        DS *sp = (DS *)sh_ray[warp];
        for (int k = 0; k < 3; ++k) {
            sp[k] = ds(p[k]);
            sp[3 + k] = ds(d[k]);
        }
        sh_ray[warp][12] = (float)sqrt(fmax(0., 1. - d[2] * d[2]));
    }
    __syncwarp();
    const DS *sp = (const DS *)sh_ray[warp];
    const DS bp[3] = {sp[0], sp[1], sp[2]};
    const DS bd[3] = {sp[3], sp[4], sp[5]};
    const float st = sh_ray[warp][12];
    const DS gu_l = (lane < nq && nq <= 32) ? ds(gu[lane]) : DS{0.f, 0.f};
    const uint32_t sb = edge_off[ray], se = edge_off[ray + 1];
    for (uint32_t seg = sb; seg + 1 < se; ++seg) {
        const double a = edges[seg], b = edges[seg + 1];
        const int64_t slot0 = (int64_t)(seg - ray) * nq - base;
        if (b - a <= 1e-8) {
            for (int step = lane; step < nq; step += 32)
                c0cp[slot0 + step] = make_int2(-1, 0);
            continue;
        }
        const bool low = (lower[ray] >> (seg - sb)) & 1;
        const DS ad = ds(a), bd_s = ds(b), length = dx(bd_s, ad);
        for (int step0 = 0; step0 < nq; step0 += 32) {
            const int step = step0 + lane;
            if (step < nq) {
                const DS u = (nq <= 32) ? gu_l : ds(gu[step]);
                const DS v = low ? u : dx({1.f, 0.f}, u);
                const DS t = path_t(ad, bd_s, length, u, v, low);
                DS qv[3], q2[3];
                const DS rr = path_q(bp, bd, t, qv, q2), xy = dr(da(q2[0], q2[1]));
                const float sh = path_h(rr);
                const float rho = dv(xy);
                const float cx = rho > 1e-12f ? dq(qv[0], xy) : 1.f;
                const float sx = rho > 1e-12f ? dq(qv[1], xy) : 0.f;
                const float angle = atan2f(rho, dv(qv[2]));
                const int ir = search(h, nh, sh), j = search(chi, nc, angle);
                const bool any_live =
                    !outside_support(sh, false) &&
                    (FW != 1 && nh <= 128 ? (Transport<Sample>::layer_live(live_a, live_a_hi, ir) ||
                                  Transport<Sample>::layer_live(live_a, live_a_hi, ir + 1))
                               : true);
                if (!any_live) {
                    c0cp[slot0 + step] = make_int2(-1, 0);
                } else {
                    const float cp =
                        st > 1e-12f ? clip((pf(bd[0]) * cx + pf(bd[1]) * sx) / st, -1.f, 1.f) : 1.f;
                    c0cp[slot0 + step] = make_int2(ir * nc + j, __float_as_int(cp));
                }
            }
        }
    }
}

#ifdef VP_STAGE_PLANE
__device__ __forceinline__ void vp_load_plane(float2 *plane, const float2 *source, int tid) {
    constexpr int pairs = FAST_WAVE_STRIDE / 2;
    for (int i = tid; i < K * pairs; i += blockDim.x)
        vp_copy8(plane + i, source + i);
}
__device__ __forceinline__ void vp_contract_pair(const float2 *plane, const float *y, int pair,
                                                bool live, float (&a)[SH_L + 1][2]) {
#pragma unroll
    for (int m = 0; m <= SH_L; ++m) {
        a[m][0] = 0.f;
        a[m][1] = 0.f;
    }
    if (live) {
#pragma unroll
        for (int l = 0; l <= SH_L; ++l) {
#pragma unroll
            for (int m = 0; m <= l; ++m) {
                const int k = ix(l, m);
                const float basis = y[k];
                const float2 c = plane[k * (FAST_WAVE_STRIDE / 2) + pair];
                a[m][0] += basis * c.x;
                a[m][1] += basis * c.y;
            }
        }
    }
}
__device__ __forceinline__ void vp_record_pair(int sl, float cp, bool live,
                                              const float (&a)[SH_L + 1][2], int pair,
                                              float *aero) {
    float2 *out = (float2 *)(aero + (int64_t)sl * AERO_W + 2 * pair);
    if (!live) {
        *out = float2{0.f, 0.f};
        return;
    }
    float s0 = a[0][0], s1 = a[0][1];
    s0 += cp * a[1][0];
    s1 += cp * a[1][1];
    float cm0 = 1.f, cm1 = cp;
    const float cp2 = 2.f * cp;
#pragma unroll
    for (int m = 2; m <= SH_L; ++m) {
        const float cm2 = cp2 * cm1 - cm0;
        s0 += cm2 * a[m][0];
        s1 += cm2 * a[m][1];
        cm0 = cm1;
        cm1 = cm2;
    }
    *out = float2{s0, s1};
}
__device__ __forceinline__ void vp_publish_pairs(int begin, int end, bool live,
                                                const int *rec_slot, const int2 *c0cp,
                                                const float (&a)[SH_L + 1][2], int pair,
                                                float *aero) {
    for (int rb = begin; rb < end; rb += 8) {
        const int count = min(8, end - rb);
        int slot[8], cp[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
            slot[i] = i < count ? rec_slot[rb + i] : 0;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            cp[i] = (live && i < count) ? c0cp[slot[i] >> 2].y : 0;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (i < count)
                vp_record_pair(slot[i], __int_as_float(cp[i]), live, a, pair, aero);
    }
}

extern "C" __global__ void __launch_bounds__(FAST_THREADS)
    sos_vprod_direct(const int *blk, const int *vis_rbeg, const int *vis_ray, const int *rec_slot,
                  const int2 *c0cp, const float *yb, const float *src_sh, const float *hr,
                  const float *ha, int nh, int nc, float *aero) {
    extern __shared__ float2 staged_plane[];
    __shared__ float vp_y[VP_VISITOR_TILE * K];
    const int b = blockIdx.x, cell = blk[3 * b], v0 = blk[3 * b + 1], v1 = blk[3 * b + 2];
    const int tid = threadIdx.x, vl = tid / VP_PAIR_LANES, pair = tid % VP_PAIR_LANES,
              lane = tid & 31, wp = tid >> 5,
              nwp = (int)blockDim.x >> 5;
    constexpr int pairs = FAST_WAVE_STRIDE / 2;
    const float2 *source = (const float2 *)src_sh + (int64_t)cell * K * pairs;
    const float2 *plane = VP_STAGE_PLANE ? staged_plane : source;
    const int ir = cell / nc;
    const bool live = FW == 1 || nh > 128 || ha[ir] > SKIP_TAU * hr[ir];
    if (live) {
        if constexpr (VP_STAGE_PLANE)
            vp_load_plane(staged_plane, source, tid);
        if (v0 < v1)
            vp_fill(vp_y, yb, vis_ray, v0, min(VP_VISITOR_TILE, v1 - v0), lane, wp, nwp);
    }
    for (int tb = v0; tb < v1; tb += VP_VISITOR_TILE) {
        const int nv = min(VP_VISITOR_TILE, v1 - tb);
        if (live)
            vp_copy_wait();
        __syncthreads();
        const int v = tb + vl;
        int r0 = 0, r1 = 0;
        float a[SH_L + 1][2];
        if (vl < nv && pair < pairs) {
            r0 = vis_rbeg[v];
            r1 = vis_rbeg[v + 1];
            vp_contract_pair(plane, vp_y + vl * K, pair, live, a);
        }
        __syncthreads();
        if (live && tb + VP_VISITOR_TILE < v1)
            vp_fill(vp_y, yb, vis_ray, tb + VP_VISITOR_TILE, min(VP_VISITOR_TILE, v1 - tb - VP_VISITOR_TILE), lane,
                    wp, nwp);
        if (vl < nv && pair < pairs)
            vp_publish_pairs(r0, r1, live, rec_slot, c0cp, a, pair, aero);
    }
    if (live && v0 >= v1)
        vp_copy_wait();
}

#endif // VP_STAGE_PLANE

#ifdef SH_ANGLE_BINS
constexpr int SH_NODES = SH_ANGLE_BINS + 3;  // one ghost knot on either end
// Each block synthesizes all azimuths for one cell and one polar knot.
extern "C" __global__ void sh_direction_table(const float *src, const float *polar,
                                             const float *azimuth, float *table) {
    const int cell = blockIdx.x, theta = blockIdx.y, tid = threadIdx.x;
    __shared__ float amplitude[SH_L + 1][FAST_WAVE_STRIDE];
    for (int i = tid; i < (SH_L + 1) * FAST_WAVE_STRIDE; i += blockDim.x) {
        const int m = i / FAST_WAVE_STRIDE, w = i % FAST_WAVE_STRIDE;
        float sum = 0.f;
        for (int l = m; l <= SH_L; ++l) {
            const int k = ix(l, m);
            sum = fmaf(src[((long long)cell * K + k) * FAST_WAVE_STRIDE + w], polar[theta * K + k], sum);
        }
        amplitude[m][w] = sum;
    }
    __syncthreads();
    for (int i = tid; i < SH_NODES * FAST_WAVE_STRIDE; i += blockDim.x) {
        const int phi = i / FAST_WAVE_STRIDE, w = i % FAST_WAVE_STRIDE;
        float sum = 0.f;
#pragma unroll
        for (int m = 0; m <= SH_L; ++m)
            sum = fmaf(azimuth[phi * (SH_L + 1) + m], amplitude[m][w], sum);
        table[((long long)cell * SH_NODES + theta) * SH_NODES * FAST_WAVE_STRIDE + i] = sum;
    }
}

struct ShSpline {
    int knot;
    float w[4];
    __device__ explicit ShSpline(float angle) {
        knot = min(SH_ANGLE_BINS - 1, max(0, (int)angle));
        const float t = clip(angle - knot, 0.f, 1.f), u = 1.f - t;
        w[0] = u * u * u / 6.f;
        w[1] = (4.f + t * t * (3.f * t - 6.f)) / 6.f;
        w[2] = (4.f + u * u * (3.f * u - 6.f)) / 6.f;
        w[3] = t * t * t / 6.f;
    }
};

__device__ __forceinline__ float sh_spline_at(const float *const *pages, int cell, int wave,
                                              const ShSpline &theta, const ShSpline &phi) {
    const float *page = pages[cell / SH_PAGE_CELLS];
    const int polar = (cell % SH_PAGE_CELLS) * SH_NODES + theta.knot;
    const float *row = page + (polar * SH_NODES + phi.knot) * FAST_WAVE_STRIDE + wave;
    float value = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float azimuth = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i)
            azimuth = fmaf(phi.w[i], row[(j * SH_NODES + i) * FAST_WAVE_STRIDE], azimuth);
        value = fmaf(theta.w[j], azimuth, value);
    }
    return value;
}
#endif

#define FS_WARPS (FAST_THREADS / 32)
#define FS_BATCH FAST_BATCH
extern "C" __global__ void __launch_bounds__(FAST_THREADS)
    sos_fast_slim(int cells, int nh, int nc, int nq, long long r0, int q0, int q1,
                  long long slim_base, const int64_t *cell_off, const double *starts,
                  const double *targets, const double *dirs, const uint32_t *edge_off,
                  const double *edges, const uint16_t *lower, const int32_t *end_j,
                  const float *end_f, const float *end_col, const uint8_t *hits, const double *gu,
                  const double *gw, const float *h, const float *chi, const float *hr,
                  const float *ha, const float *ext, const float *sun_h, int nsh, float height,
                  float mix, const float *src, const double *boundary, const float *slim_aero,
                  const float *profile_table, int profile_mask, float air_surface,
                  float *ray_rads) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int q = q0 + (int)blockIdx.x * FS_WARPS + warp;
    if (q >= q1)
        return;
    const int64_t ray = r0 + q;
    const ProfileParams profile{height, sun_h + nsh, profile_table, profile_mask, air_surface,
                                0.f,    1.f,         1.f};
    unsigned long long live_a = ~0ull, live_a_hi = ~0ull;
    if (FW != 1 && nh <= 128) {
        bool la[4] = {};
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int idx = lane + 32 * b;
            if (idx < nh)
                la[b] = ha[idx] > SKIP_TAU * hr[idx];
        }
        const uint32_t m0 = __ballot_sync(0xffffffffu, la[0]);
        const uint32_t m1 = __ballot_sync(0xffffffffu, la[1]);
        const uint32_t m2 = __ballot_sync(0xffffffffu, la[2]);
        const uint32_t m3 = __ballot_sync(0xffffffffu, la[3]);
        live_a = (unsigned long long)m0 | ((unsigned long long)m1 << 32);
        live_a_hi = (unsigned long long)m2 | ((unsigned long long)m3 << 32);
    }
    int cell = 0;
    if (lane == 0) {
        int lo = 0, hi = cells - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) >> 1;
            if (cell_off[mid] <= ray)
                lo = mid;
            else
                hi = mid - 1;
        }
        cell = lo;
    }
    cell = __shfl_sync(0xffffffffu, cell, 0);
#ifdef SH_ANGLE_BINS
    __shared__ float sh_phi[FS_WARPS][FS_BATCH];
    float *phi_knots = sh_phi[warp];
    const double dz = dirs[3 * ray + 2];
    const float st = (float)sqrt(fmax(0., 1. - dz * dz));
    const ShSpline theta((float)(acos(fmin(1., fmax(-1., dz))) * (SH_ANGLE_BINS / PI)));
#else
    float *phi_knots = nullptr;
    const float st = 0.f;
#endif
    __shared__ SlimPoint sh_pts[FS_WARPS][FS_BATCH];
    __shared__ float sh_mp[FS_WARPS][FS_BATCH + 1][NCOMP + 1];
    __shared__ float sh_B[FS_WARPS][9];
    __shared__ float sh_rc[FS_WARPS][20];
    float *B = sh_B[warp];
    float *dl = sh_rc[warp];
    float *pu = sh_rc[warp] + 3;
    DS *beta_p = (DS *)(sh_rc[warp] + 8);
    DS *beta_d = beta_p + 3;
    if (lane == 0) {
        const double *p = starts + 3 * cell;
        const double *target = targets + 9 * cell;
        const double *d = dirs + 3 * ray;
        for (int k = 0; k < 3; ++k) {
            beta_p[k] = ds(p[k]);
            beta_d[k] = ds(d[k]);
        }
        for (int k = 0; k < 9; ++k)
            B[k] = (float)target[k];
        for (int j = 0; j < 3; ++j) {
            double x = 0.;
            for (int k = 0; k < 3; ++k)
                x += target[3 * k + j] * d[k];
            dl[j] = (float)x;
        }
        pu[0] = 1.f - dl[0] * dl[0];
        pu[1] = 1.f - dl[1] * dl[1];
        pu[2] = 1.f - dl[2] * dl[2];
        pu[3] = -dl[0] * dl[2];
    }
    __syncwarp();
    const float pu_r[4] = {pu[0], pu[1], pu[2], pu[3]};
    const int half = lane >> 4, w = lane & 15;
    float ray_ext[NCOMP] = {};
    if (w < TW) {
#pragma unroll
        for (int k = 0; k < NCOMP; ++k)
            ray_ext[k] = ext[k * FW + w];
    }
    const DS gu_l = (lane < nq && nq <= 32) ? ds(gu[lane]) : DS{0.f, 0.f};
    const DS gw_l = (lane < nq && nq <= 32) ? ds(gw[lane]) : DS{0.f, 0.f};
    float lane_rad[4] = {}, lane_rc[4] = {};
    float tau_lane = 0.f, tc_lane = 0.f;
    const uint32_t sb = edge_off[ray], se = edge_off[ray + 1];
    for (uint32_t seg = sb; slim_aero && seg + 1 < se; ++seg) {
        const double a = edges[seg], b = edges[seg + 1];
        if (b - a <= 1e-8)
            continue;
        const bool low = (lower[ray] >> (seg - sb)) & 1;
        const DS seg_a = ds(a), seg_b = ds(b), seg_inside = ds((a + b) * 0.5);
        DS sp_t = seg_a;
        float sp_acc = 0.f, sp_b = 0.f, seg_be = 0.f;
        float end_p[NCOMP] = {};
        if (lane < 2) {
            DS qv[3], q2[3];
            const DS te = lane == 0 ? seg_a : seg_b;
            float sh = path_h(path_q(beta_p, beta_d, te, qv, q2));
            if (support_extended()) {
                const float ih = path_h(path_q(beta_p, beta_d, seg_inside, qv, q2));
                sh = support_height(sh, ih);
            }
            float pst_unused;
            profile_at(sh, profile, end_p, pst_unused);
        }
        float pa[NCOMP], pb[NCOMP];
#pragma unroll
        for (int k = 0; k < NCOMP; ++k) {
            pa[k] = __shfl_sync(0xffffffffu, end_p[k], 0);
            pb[k] = __shfl_sync(0xffffffffu, end_p[k], 1);
        }
        if (lane < TW) {
            float ba = 0.f, bb = 0.f;
#pragma unroll
            for (int k = 0; k < NCOMP; ++k) {
                ba += pa[k] * ray_ext[k];
                bb += pb[k] * ray_ext[k];
            }
            sp_b = ba;
            seg_be = bb;
        }
        const float seg_tau = __shfl_up_sync(0xffffffffu, tau_lane - tc_lane, 16);
        for (int step0 = 0; step0 < nq; step0 += FS_BATCH) {
            const int nb = min(FS_BATCH, nq - step0);
            __syncwarp();  // every lane has finished reading the previous sh_pts/sh_mp batch
            const bool is_last = (step0 + nb == nq);
            const NodeWeights node =
                quadrature_node(step0 + lane, nq, lane < nb, gu, gw, gu_l, gw_l);
            DS tk = {0.f, 0.f};
            if (lane < nb)
                tk = point_slim(sh_pts[warp][lane], beta_p, beta_d, B, dl, seg_a, seg_b, low,
                                node.u, node.w, h, nh, chi, nc, profile, st, phi_knots, lane);
            DS prev_t;
            prev_t.h = __shfl_up_sync(0xffffffffu, tk.h, 1);
            prev_t.l = __shfl_up_sync(0xffffffffu, tk.l, 1);
            if (lane == 0)
                prev_t = sp_t;
            DS curr_t = tk;
            if (is_last && lane == nb)
                curr_t = seg_b;
            const int n_panels = nb + (is_last ? 1 : 0);
            if (lane < n_panels) {
                DS tm = da(prev_t, curr_t);
                tm.h *= .5f;
                tm.l *= .5f;
                DS qv[3], q2[3];
                float mp_tmp[NCOMP], pst_unused;
                profile_at(path_h(path_q(beta_p, beta_d, tm, qv, q2)), profile, mp_tmp, pst_unused);
#pragma unroll
                for (int k = 0; k < NCOMP; ++k)
                    sh_mp[warp][lane][k] = mp_tmp[k];
                sh_mp[warp][lane][NCOMP] = dv(dx(curr_t, prev_t)) / 6.f;
            }
            sp_t.h = __shfl_sync(0xffffffffu, tk.h, nb - 1);
            sp_t.l = __shfl_sync(0xffffffffu, tk.l, nb - 1);
            __syncwarp();
            for (int bi = 0; bi < nb; bi += 2) {
                const bool has_odd = (bi + 1 < nb);
                const int my_bi = (half && has_odd) ? (bi + 1) : bi;
                const bool active = (w < TW) && (half == 0 || has_odd);
                const SlimPoint &pt = sh_pts[warp][my_bi];
                float lane_val[4] = {};
                float lane_scale = 1.f;
                float bk = 0.f, bm = 0.f;
                const float dt6 = sh_mp[warp][my_bi][NCOMP];
                if (active) {
                    const int ir0 = pt.ir;
                    const float ph = pt.h;
                    const bool in_aer_sup = !outside_support(ph, false);
                    const float hr0 = hr[ir0], hr1 = hr[ir0 + 1];
                    const float ha0 = ha[ir0], ha1 = ha[ir0 + 1];
                    const bool live0 =
                        in_aer_sup && (FW != 1 && nh <= 128 ? Transport<Sample>::layer_live(live_a, live_a_hi, ir0) : true);
                    const bool live1 =
                        in_aer_sup &&
                        (FW != 1 && nh <= 128 ? Transport<Sample>::layer_live(live_a, live_a_hi, ir0 + 1) : true);
                    const int4 cells4 = *(const int4 *)pt.cell;
                    const int cc_arr[4] = {cells4.x, cells4.y, cells4.z, cells4.w};
                    const int64_t pt_slot = (int64_t)(seg - ray) * nq + (step0 + my_bi) - slim_base;
#ifdef SH_ANGLE_BINS
                    const ShSpline phi(sh_phi[warp][my_bi]);
#endif
                    float aero_pre[4] = {};
                    float rm_pre[4][4];
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        const bool lk = (c < 2) ? live0 : live1;
                        if (lk)
#ifdef SH_ANGLE_BINS
                            aero_pre[c] = sh_spline_at((const float *const *)slim_aero, cc_arr[c], w, theta, phi);
#else
                            aero_pre[c] = slim_aero[(pt_slot * 4 + c) * AERO_W + w];
#endif
                        const int64_t sbase = (int64_t)cc_arr[c] * SRC_ROWS;
#pragma unroll
                        for (int k = 0; k < 4; ++k)
                            rm_pre[c][k] = src[(sbase + k) * FW + w];
                    }
                    const float4 swv = *(const float4 *)pt.sw;
                    const float sw_arr[4] = {swv.x, swv.y, swv.z, swv.w};
                    float lane_log = 0.f, lane_wg = 0.f, lane_am = 0.f;
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        const float corner_sw = sw_arr[c];
                        const float *rm = rm_pre[c];
                        const float density = support_value(ph, c < 2 ? hr0 : hr1, true);
                        const float aero =
                            (c < 2 ? live0 : live1)
                                ? support_value(ph, aero_pre[c] * (c < 2 ? ha0 : ha1), false)
                                : 0.f;
                        const float tr = (rm[0] + rm[1]) + rm[2];
                        float tensors[3];
#pragma unroll
                        for (int comp = 0; comp < 4; ++comp) {
                            const float4 tv = *(const float4 *)(pt.T + 4 * comp);
                            float rot = 0.f;
                            rot += tv.x * rm[0];
                            rot += tv.y * rm[1];
                            rot += tv.z * rm[2];
                            rot += tv.w * rm[3];
                            const float puc = pu_r[comp];
                            float tensor = density * (mix * rot + ((1.f - mix) / 3.f * tr) * puc);
                            tensor += (aero * puc) / 2.f;
                            lane_val[comp] += corner_sw * tensor;
                            if (comp < 3)
                                tensors[comp] = tensor;
                        }
                        const float trace = (tensors[0] + tensors[1]) + tensors[2];
                        const bool good = trace >= TRACE_MIN;
                        lane_log += good ? corner_sw * logf(trace) : 0.f;
                        lane_wg += good ? corner_sw : 0.f;
                        lane_am += good ? corner_sw * trace : 0.f;
                    }
                    const bool ok = lane_wg > 0.f && lane_am >= TRACE_MIN;
                    lane_scale = ok ? expf(lane_log - lane_wg * logf(lane_am / lane_wg)) : 1.f;
#pragma unroll
                    for (int k = 0; k < NCOMP; ++k) {
                        bk += pt.prof[k] * ray_ext[k];
                        bm += sh_mp[warp][my_bi][k] * ray_ext[k];
                    }
                }
                const float bk_even = __shfl_up_sync(0xffffffffu, bk, 16);
                if (half == 0 && w < TW)
                    sp_acc += dt6 * ((sp_b + bk) + 4.f * bm);
                const float acc_even = __shfl_up_sync(0xffffffffu, sp_acc, 16);
                float my_acc = sp_acc;
                if (half == 1 && active)
                    my_acc = acc_even + dt6 * ((bk_even + bk) + 4.f * bm);
                const float acc_odd = __shfl_down_sync(0xffffffffu, my_acc, 16);
                const float bk_odd = __shfl_down_sync(0xffffffffu, bk, 16);
                if (half == 0 && w < TW) {
                    sp_acc = has_odd ? acc_odd : sp_acc;
                    sp_b = has_odd ? bk_odd : bk;
                }
                float paired[4] = {};
                if (active) {
                    const float lane_att = pt.ds * expf(-(seg_tau + my_acc));
#pragma unroll
                    for (int k = 0; k < 4; ++k)
                        paired[k] = (lane_val[k] * lane_scale) * lane_att;
                }
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float v_odd = __shfl_down_sync(0xffffffffu, paired[k], 16);
                    if (lane < TW) {
                        kah(lane_rad[k], lane_rc[k], paired[k]);
                        if (has_odd)
                            kah(lane_rad[k], lane_rc[k], v_odd);
                    }
                }
            }
            if (is_last && lane < TW) {
                float bm_end = 0.f;
#pragma unroll
                for (int k = 0; k < NCOMP; ++k)
                    bm_end += sh_mp[warp][nb][k] * ray_ext[k];
                sp_acc += sh_mp[warp][nb][NCOMP] * ((sp_b + seg_be) + 4.f * bm_end);
                kah(tau_lane, tc_lane, sp_acc);
            }
        }
    }
    if constexpr (K_BOUNDARY) {
        if (lane < TW) {
            const int j = end_j[ray];
            const float f = end_f[ray];
            float e = (1.f - f) * (float)boundary[j * FW + lane];
            e += f * (float)boundary[(j + 1) * FW + lane];
            float od = 0.f;
#pragma unroll
            for (int k = 0; k < NCOMP; ++k)
                od += end_col[NCOMP * ray + k] * ray_ext[k];
            e *= expf(-od) * (float)hits[ray];
#pragma unroll
            for (int k = 0; k < 4; ++k)
                lane_rad[k] += (e * pu_r[k]) / 2.f;
        }
    }
    if (lane < TW) {
        float4 *dst = (float4 *)(ray_rads + (ray - r0) * (4 * TW) + 4 * lane);
        *dst = make_float4(lane_rad[0], lane_rad[1], lane_rad[2], lane_rad[3]);
    }
}
#endif

#if (defined(SINV) && SLIM_ON && !K_FIRST && K_PHASE == 0 && defined(ACC_IN_GLOBAL) &&              \
     !defined(SLIM_RAYS) && !defined(SLIM_RAYDUMP)) ||                                              \
    defined(FAST_VPROD_ON)
// A 64-moment x wavelength register tile over 32 contiguous rays. Each thread owns half
// the wavelengths of one moment and reuses y*cd across them. The producer's ray sort is
// irrelevant to storage (ray_rads and yb are indexed by original ray); projection uses
// original ray order, FP32 batch partials, and deterministic FP64 batch/part combination.
#ifdef SPLIT_ACTIVE
#define SP_PARTS SPLIT_S
#else
#define SP_PARTS 1
#endif
#ifdef FAST_VPROD_ON
#define SP_THREADS FAST_THREADS
#else
#define SP_THREADS 128
#endif
#define SP_MOMENTS (SP_THREADS / 2)
#define SP_RAYS 32
#define SP_WAVES ((TW + 1) / 2)
extern "C" __global__ void __launch_bounds__(SP_THREADS)
    sos_slim_proj(int cells, int nw, long long ray_base, const int64_t *cell_off,
                  const double *targets, const double *dirs, const float *weights,
                  const uint32_t *edge_off, const float *yb, const float *ray_rads, double albedo,
                  double *moments, double *flux) {
    __shared__ float y[SP_RAYS][SP_MOMENTS + 1], cd[SP_RAYS][33];
    __shared__ float rad[SP_RAYS][6 * TW], wg[SP_RAYS];
    const int cell = blockIdx.x;
    if (cell >= cells)
        return;
    const int tid = threadIdx.x, mi = tid % SP_MOMENTS;
    // The ground-only caller requests just the final projection row.
    const int mb = nw == -1 ? 4 + K : (int)blockIdx.y * SP_MOMENTS;
    const int moment = mb + mi, wbase = (tid / SP_MOMENTS) * SP_WAVES, nm = (4 + K) * TW;
    const bool isM = moment < 4 + K, isF = moment == 4 + K;
    const int sk = moment - 4, m = (isM && moment >= 4) ? LM.m[sk] : 0;
    const double *target = targets + 9 * cell;
    const int64_t cb = cell_off[cell], ce = cell_off[cell + 1];
    double total[SP_WAVES] = {};
    for (int part = 0; part < SP_PARTS; part++) {
        const int64_t begin = cb + (ce - cb) * part / SP_PARTS,
                      end = cb + (ce - cb) * (part + 1) / SP_PARTS;
        double acc[SP_WAVES] = {};
        for (int64_t rb = begin; rb < end; rb += SP_RAYS) {
            const int nr = (int)min((int64_t)SP_RAYS, end - rb);
            if (tid < nr) {
                const double *d = dirs + 3 * (rb + tid);
                double st = sqrt(fmax(0., 1. - d[2] * d[2]));
                float cp = st > 1e-12 ? (float)(d[0] / st) : 1.f;
                cp = clip(cp, -1, 1);
                cd[tid][0] = 1;
                cd[tid][1] = cp;
                for (int mm = 2; mm < 33; mm++)
                    cd[tid][mm] = (2.f * cp) * cd[tid][mm - 1] - cd[tid][mm - 2];
                double dlz = 0;
                for (int k = 0; k < 3; k++)
                    dlz += target[3 * k + 2] * d[k];
                wg[tid] = weights[rb + tid] * fmaxf(0.f, (float)dlz);
            }
            // Consecutive lanes read consecutive coefficients in each original ray row.
            for (int i = tid; i < nr * SP_MOMENTS; i += SP_THREADS) {
                const int r = i / SP_MOMENTS, k = i % SP_MOMENTS, s = mb + k - 4;
                y[r][k] = (s >= 0 && s < K) ? yb[(rb + r - ray_base) * K + s] : 0.f;
            }
            __syncthreads();
            projection_radiance<SP_THREADS>(tid, nr, rb, ray_base, ray_rads, weights, wg, rad);
            __syncthreads();
            float sum[SP_WAVES] = {};
            if (isM || isF)
                for (int r = 0; r < nr; r++) {
                    const float yc = (moment < 4 || isF) ? 1.f : y[r][mi] * cd[r][m];
#pragma unroll
                    for (int q = 0; q < SP_WAVES; q++)
                        if (wbase + q < TW)
                            sum[q] += rad[r][(isF          ? 5
                                              : moment < 4 ? moment
                                                           : 4) *
                                                 TW +
                                             wbase + q] *
                                      yc;
                }
#pragma unroll
            for (int q = 0; q < SP_WAVES; q++)
                acc[q] += (double)sum[q];
            __syncthreads();
        }
#pragma unroll
        for (int q = 0; q < SP_WAVES; q++)
            total[q] += acc[q];
    }
#pragma unroll
    for (int q = 0; q < SP_WAVES; q++)
        if (wbase + q < TW) {
            if (isM)
                moments[(int64_t)cell * nm + moment * TW + wbase + q] = total[q];
            else if (isF)
                flux[cell * FW + GW(wbase + q)] = total[q] * (albedo / PI);
        }
}
#endif
#endif // !LUT_ONLY
