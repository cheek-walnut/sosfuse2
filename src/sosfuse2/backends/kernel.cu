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
// The slim transport's approximate PTX and split logarithm are device-only.
#if defined(SOSFUSE2_FASTMATH) && defined(__CUDA_ARCH__)
#define SLIM_FASTMATH 1
#else
#define SLIM_FASTMATH 0
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
#define PI 3.14159265358979323846f
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
#define RAY_CONST_F 15

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
// The original shell intersections fix the side even if a computed altitude crosses it.
__device__ __forceinline__ float support_height(float h, bool above) {
    if (!support_extended())
        return h;
    return above ? fmaxf(h, nextafterf(LOWER_END_KM, TOP_KM)) : fminf(h, LOWER_END_KM);
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
// Neumaier summation preserves small terms when the new term dominates the partial.
struct ProjectionSum {
    float sum, error;
    __device__ void add(float v) {
        const float next = sum + v;
        error += fabsf(sum) >= fabsf(v) ? (sum - next) + v : (v - next) + sum;
        sum = next;
    }
    __device__ float value() const { return sum + error; }
};
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
__device__ void sh_basis(float mu, float st, int lane, float *y, const float *aa, const float *bb,
                         const float *dd) {
    mu = clip(mu, -1.f, 1.f);
    for (int m = lane; m <= SH_L; m += 32) {
        float diag = 1 / sqrtf(4 * PI);
        for (int j = 1; j <= m; j++)
            diag = (diag * dd[j]) * st;
        float mult = m ? sqrtf(2.f) : 1.f, z0 = diag;
        y[ix(m, m)] = (float)(mult * z0);
        if (m < SH_L) {
            float z1 = (sqrtf(2.f * m + 3) * mu) * z0;
            y[ix(m + 1, m)] = (float)(mult * z1);
            for (int l = m + 2; l <= SH_L; l++) {
                int k = ix(l, m);
                float z2 = (aa[k] * mu) * z1 - bb[k] * z0;
                y[k] = (float)(mult * z2);
                z0 = z1;
                z1 = z2;
            }
        }
    }
}
// The profile is evaluated by one function, profile_at, for
// the quadrature nodes (point()) and the Simpson optical-depth prefix (segment ends, panel
// midpoints); numpy_reference.py profile_at, extinction_profile, SimpsonPrefix use the same
// profiles. Panel widths and midpoints are computed within the segment.
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
// Geometry is in km. The origin row is [x, y, z, height, radius].
struct RayGeometry {
    float pd, shadow_begin, shadow_end;
};
struct PathEdge {
    float t, signed_length;
};
struct PathSegment {
    float anchor, length, h0, r0;
    RayGeometry ray;
    bool lower, above;

    // Signed fraction from the lower end, increasing along the ray in both orientations.
    __device__ float start() const { return lower ? 0.f : -1.f; }
    __device__ float end() const { return lower ? 1.f : 0.f; }
    __device__ float node(float u) const {
        const float v = lower ? u : 1.f - u;
        return lower ? v * v : -(v * v);
    }
    __device__ float distance(float x) const { return fmaf(length, x, anchor); }
    __device__ float width(float a, float b) const { return length * (b - a); }
    __device__ float height(float x) const {
        const float t = distance(x);
        const float diff = t * (2.f * ray.pd + t);
        const float r = sqrtf(fmaf(r0, r0, diff));
        return support_height(clip(h0 + diff / (r + r0), 0.f, TOP_KM), above);
    }
    __device__ bool lit(float x) const {
        // Subtract the segment anchor first to resolve nodes very close to a shadow crossing.
        const float offset = length * x;
        return !(offset > ray.shadow_begin - anchor && offset < ray.shadow_end - anchor);
    }
};
__device__ __forceinline__ PathSegment path_segment(const float *p, RayGeometry ray,
                                                     const PathEdge *edge, bool lower) {
    return {edge[lower ? 0 : 1].t, fabsf(edge->signed_length), p[3], p[4], ray,
            lower, edge->signed_length < 0.f};
}
struct PathPoint {
    float x, h, rho, z, cx, sx;
};
__device__ __forceinline__ PathPoint path_point(const float *p, const float *d,
                                                const PathSegment &path, float u) {
    const float x = path.node(u), t = path.distance(x);
    const float qx = fmaf(t, d[0], p[0]), qy = fmaf(t, d[1], p[1]);
    const float rho = hypotf(qx, qy);
    return {x, path.height(x), rho, fmaf(t, d[2], p[2]),
            rho > 1e-12f ? qx / rho : 1.f, rho > 1e-12f ? qy / rho : 0.f};
}
// Direct SH records and their consumer must address the same four source corners.
struct PathCell {
    PathPoint point;
    float angle;
    int ir, j;
};
__device__ __forceinline__ PathCell path_cell(const float *p, const float *d, const PathSegment &path,
                                              float u, const float *h, int nh, const float *chi, int nc) {
    const PathPoint pt = path_point(p, d, path, u);
    const float angle = atan2f(pt.rho, pt.z);
    return {pt, angle, search(h, nh, pt.h), search(chi, nc, angle)};
}
__device__ __forceinline__ float solar_coordinate(float h, float z) {
    const float tangent_distance = sqrtf(h * (12742.f + h));
    return sqrtf(clip((z + tangent_distance) / ((6371.f + h) + tangent_distance), 0.f, 1.f));
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
__device__ __forceinline__ float beta_at(const PathSegment &path, float x,
                                         const ExtinctionParams &optics, int gw) {
    float pr[NCOMP], pst;
    profile_at(path.height(x), optics.profile, pr, pst);
    return extinction_at(pr, pst, optics, gw);
}
__device__ __forceinline__ float simpson_piece(const PathSegment &path, float x0, float x1,
                                              float b0, float b1, const ExtinctionParams &optics,
                                              int gw) {
    const float bm = beta_at(path, (x0 + x1) * .5f, optics, gw);
    return (path.width(x0, x1) / 6.f) * ((b0 + b1) + 4.f * bm);
}
// ---- Quadrature point geometry ----

#define POINT_INLINE __noinline__
template <class Record>
__device__ POINT_INLINE void
point(Record &s, const float *p, const float *d, const float *B, const float *dl, float st,
      const PathSegment &path, float gu, float gw, const float *hgrid, int nh,
      const float *chi, int nc, ProfileParams profile, const float *sun_h, int nsh,
      const float *sun_col, int nsx, bool first, const float *st_sun_col = nullptr) {
    const PathPoint pt = path_point(p, d, path, gu);
    const float sh = pt.h, rho = pt.rho, cx = pt.cx, sx = pt.sx;
    s.h = sh;
    s.ds = (2.f * path.length) * ((path.lower ? gu : 1.f - gu) * gw);
    s.cs = clip(pt.z / (6371.f + sh), -1.f, 1.f);
    float angle = atan2f(rho, pt.z);
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
            dot[col] += d[k] * Q[3 * k + col];
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
    float cp = st > 1e-12f ? clip((d[0] * cx + d[1] * sx) / st, -1, 1) : 1.f;
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
        s.lit = path.lit(pt.x);
        float x = s.lit ? solar_coordinate(sh, pt.z) : 0.f;
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
#ifdef STRATO
    float st_prof;
    float _pad[3];
#endif
};
static_assert(sizeof(WarmPoint) == 144
#ifdef STRATO
                                       + 16
#endif
              ,
              "E_WARMPOINT_LAYOUT");

__device__ POINT_INLINE float point_warm(WarmPoint &s, const float *p, const float *d, const float *B,
                                      const float *dl, float st, const PathSegment &path,
                                      float gu, float gw, const float *hgrid, int nh,
                                      const float *chi, int nc, ProfileParams profile) {
    const PathPoint pt = path_point(p, d, path, gu);
    const float sh = pt.h, rho = pt.rho, cx = pt.cx, sx = pt.sx;
    s.h = sh;
    s.ds = (2.f * path.length) * ((path.lower ? gu : 1.f - gu) * gw);
    float angle = atan2f(rho, pt.z);
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
            dot[col] += d[k] * Q[3 * k + col];
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
    s.cp = st > 1e-12f ? clip((d[0] * cx + d[1] * sx) / st, -1, 1) : 1.f;
    s.sw[0] = (1.f - ah) * (1.f - ac);
    s.sw[1] = (1.f - ah) * ac;
    s.sw[2] = ah * (1.f - ac);
    s.sw[3] = ah * ac;
    s.cell[0] = ir * nc + j;
    s.cell[1] = s.cell[0] + 1;
    s.cell[2] = s.cell[0] + nc;
    s.cell[3] = s.cell[2] + 1;
    float pst;
    profile_at(sh, profile, s.prof, pst);
#ifdef STRATO
    s.st_prof = pst;
#endif
    return pt.x;
}

// Compact batched sample for the chunked-VPROD fast slim consumer (128 B, 16-byte aligned).
struct __align__(16) SlimPoint {
    float T[16], sw[4];
    int cell[4];
    float prof[NCOMP];
    int ir;
    float h, ds;
#ifdef STRATO
    float st_prof;
    float _pad[3];
#endif
};
static_assert(sizeof(SlimPoint) == 128
#ifdef STRATO
                                       + 16
#endif
              ,
              "E_SLIMPOINT_LAYOUT");

#if SLIM_FASTMATH
struct FastProfileParams {
    float inv_height;
    const float *aer2;
    const float *table;
    int mask;
    float inv_air_surface;
    float st_center, inv_st_sigma, inv_st_norm;
};

__device__ __forceinline__ float fast_exp_neg(float x) {
    float v;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(v) : "f"(x * 1.4426950408889634f));
    return v;
}

__device__ __forceinline__ void profile_at(float h, const FastProfileParams &profile,
                                           float *prof, float &st_prof) {
    const bool ext_lower = outside_support(h, false);
    const bool ext_air = outside_support(h, true);
    const float x = profile_coordinate(h);
    const int idx = min((int)x, PROFILE_LAST);
    const float frac = x - (float)idx;
    if (ext_air) {
        prof[0] = 0.f;
    } else if (profile.mask & 1) {
        const float t0 = profile.table[idx * 3];
        prof[0] = fmaf(frac, profile.table[(idx + 1) * 3] - t0, t0);
    } else {
        prof[0] = fast_exp_neg(-h * 0.125f);
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
        prof[1] = fast_exp_neg(-h * profile.inv_height);
    if (profile.mask & 2) {
        const float t1 = profile.table[idx * 3 + 1];
        prof[2] = fmaf(frac, profile.table[(idx + 1) * 3 + 1] - t1, t1);
    } else {
        prof[2] = fmaxf(0.f, fmaf(-fabsf(h - 25.f), 1.f / 15.f, 1.f));
    }
    {
        const float t2 = profile.table[idx * 3 + 2];
        prof[3] = fmaf(frac, profile.table[(idx + 1) * 3 + 2] - t2, t2);
    }
    const float q = prof[0] * profile.inv_air_surface;
    prof[4] = q * q;
    if constexpr (ProfilePolicy::strato) {
        const float st_t = (h - profile.st_center) * profile.inv_st_sigma;
        st_prof = fast_exp_neg(-0.5f * (st_t * st_t)) * profile.inv_st_norm;
    } else {
        st_prof = 0.f;
    }
}
#endif

__device__ __forceinline__ float point_slim(SlimPoint &s, const float *__restrict__ p, const float *__restrict__ d,
                                      const float *__restrict__ B, const float *__restrict__ dl,
                                      const PathSegment &path, float gu, float gw,
                                      const float *__restrict__ hgrid, int nh,
                                      const float *__restrict__ chi, int nc,
                                      float d0_n, float d1_n, bool has_st, float *__restrict__ phi_knots, int lane,
                                      float &out_h) {
    const PathCell cell = path_cell(p, d, path, gu, hgrid, nh, chi, nc);
    const float sh = cell.point.h, cx = cell.point.cx, sx = cell.point.sx;
    s.h = sh;
    out_h = sh;
    s.ds = (2.f * path.length) * ((path.lower ? gu : 1.f - gu) * gw);
#ifdef SH_ANGLE_BINS
    const float cp = has_st ? clip(fmaf(d0_n, cx, d1_n * sx), -1.f, 1.f) : 1.f;
    phi_knots[lane] = acosf(cp) * (SH_ANGLE_BINS / (float)PI);
#endif
    float angle = cell.angle;
    const int ir = cell.ir, j = cell.j;
    s.ir = ir;
    const float dh = hgrid[ir + 1] - hgrid[ir], dchi = chi[j + 1] - chi[j];
    float ah = clip((sh - hgrid[ir]) / dh, 0.f, 1.f),
          ac = clip((angle - chi[j]) / dchi, 0.f, 1.f);
    angle = fmaf(ac, dchi, chi[j]);
#ifdef __CUDA_ARCH__
    float sn, cc;
    sincosf(angle, &sn, &cc);
#else
    float sn = sinf(angle), cc = cosf(angle);
#endif
    float Q[9] = {-cc * cx, sx, sn * cx, -cc * sx, -cx, sn * sx, sn, 0, cc}, A[9] = {}, dot[3] = {},
          kk[9];
#pragma unroll
    for (int k = 0; k < 3; k++)
#pragma unroll
        for (int col = 0; col < 3; col++) {
            const float qv = Q[3 * k + col];
            dot[col] = fmaf(d[k], qv, dot[col]);
#pragma unroll
            for (int row = 0; row < 3; row++)
                A[3 * row + col] = fmaf(B[3 * k + row], qv, A[3 * row + col]);
        }
#pragma unroll
    for (int row = 0; row < 3; row++)
#pragma unroll
        for (int col = 0; col < 3; col++)
            kk[3 * row + col] = fmaf(-dl[row], dot[col], A[3 * row + col]);
    const int ii[4] = {0, 1, 2, 0}, jj[4] = {0, 1, 2, 2};
#pragma unroll
    for (int u2 = 0; u2 < 4; u2++) {
        int i = ii[u2], z = jj[u2];
        s.T[4 * u2] = kk[3 * i] * kk[3 * z];
        s.T[4 * u2 + 1] = kk[3 * i + 1] * kk[3 * z + 1];
        s.T[4 * u2 + 2] = kk[3 * i + 2] * kk[3 * z + 2];
        s.T[4 * u2 + 3] = fmaf(kk[3 * i], kk[3 * z + 2], kk[3 * i + 2] * kk[3 * z]);
    }
    const float om_ah = 1.f - ah, om_ac = 1.f - ac;
    s.sw[0] = om_ah * om_ac;
    s.sw[1] = om_ah * ac;
    s.sw[2] = ah * om_ac;
    s.sw[3] = ah * ac;
    s.cell[0] = ir * nc + j;
    s.cell[1] = s.cell[0] + 1;
    s.cell[2] = s.cell[0] + nc;
    s.cell[3] = s.cell[2] + 1;
    return cell.point.x;
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

__device__ POINT_INLINE float point_first(FirstPoint &s, const float *p, const float *d,
                                       const float *B, const float *dl, const PathSegment &path,
                                       float gu, float gw, const float *hgrid, int nh,
                                       const float *chi, int nc, ProfileParams profile,
                                       const float *sun_h, int nsh, const float *sun_col, int nsx) {
    const PathPoint pt = path_point(p, d, path, gu);
    const float sh = pt.h, rho = pt.rho, cx = pt.cx, sx = pt.sx;
    s.h = sh;
    s.ds = (2.f * path.length) * ((path.lower ? gu : 1.f - gu) * gw);
    float angle = atan2f(rho, pt.z);
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
            dot[col] += d[k] * Q[3 * k + col];
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
    s.lit = path.lit(pt.x);
    float x = s.lit ? solar_coordinate(sh, pt.z) : 0.f;
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
    return pt.x;
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
__device__ POINT_INLINE void point_m2(PointM2 &s, const float *p, const float *d, const float *B,
                                      const float *dl, const PathSegment &path, float gu, float gw, const float *hgrid, int nh, const float *chi,
                                      int nc) {
    const PathPoint pt = path_point(p, d, path, gu);
    const float sh = pt.h, rho = pt.rho, cx = pt.cx, sx = pt.sx;
    s.h = sh;
    s.ds = (2.f * path.length) * ((path.lower ? gu : 1.f - gu) * gw);
    float angle = atan2f(rho, pt.z);
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
            dot[col] += d[k] * Q[3 * k + col];
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
struct NodeWeights {
    float u, w;
};
__device__ __forceinline__ NodeWeights quadrature_node(int node, int nq, bool active,
                                                       const float *gu, const float *gw,
                                                       float cached_u, float cached_w) {
    if (nq <= 32)
        return {__shfl_sync(0xffffffffu, cached_u, node), __shfl_sync(0xffffffffu, cached_w, node)};
    return active ? NodeWeights{gu[node], gw[node]} : NodeWeights{};
}

// Dispatch depends on the record type, without conditional argument lists.
template <class Record>
__device__ __forceinline__ void
evaluate_point(Record &s, const float *p, const float *d, const float *B, const float *dl, float st,
               const PathSegment &path, float u, float w, const float *h, int nh,
               const float *chi, int nc, const ProfileParams &profile, const float *sun_h, int nsh,
               const float *sun_col, int nsx, bool first, const float *st_sun_col = nullptr) {
    if constexpr (TransportPolicy::slim_rays && TransportPolicy::batched_points)
        point_m2(s, p, d, B, dl, path, u, w, h, nh, chi, nc);
    else
        point(s, p, d, B, dl, st, path, u, w, h, nh, chi, nc, profile, sun_h, nsh, sun_col,
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
    const float *starts = nullptr;
    const float *targets = nullptr;
    const float *dirs = nullptr;
    const float *weights = nullptr;
    const uint32_t *edge_off = nullptr;
    const PathEdge *edges = nullptr;
    const RayGeometry *ray_geometry = nullptr;
    const uint16_t *lower = nullptr;
    const int32_t *end_j = nullptr;
    const float *end_f = nullptr;
    const float *end_col = nullptr;
    const uint8_t *hits = nullptr;
    const float *gu = nullptr;
    const float *gw = nullptr;
    const float *aa = nullptr;
    const float *bb = nullptr;
    const float *dd = nullptr;
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
    float albedo = 0;
    const float *src = nullptr;
    const float *boundary = nullptr;
    float *moments = nullptr;
    float *moment_error = nullptr;
    float *flux = nullptr;
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
    float *split_scratch = nullptr;
    ProjectionSum *split_flux = nullptr;
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
    struct RayState {
        const float *p, *target, *d;
        const float *B;
        float *y, *rad, *rc, *cd, *dl, *pu, *se, *direct;
        float global_st, pu_r[4];
        float lane_rad[4] = {}, lane_rc[4] = {}, lane_val[4], lane_log, lane_wg, lane_am;
        float ray_ext[NCOMP] = {}, ray_st_ext = 0.f;
        float cache[4][TW] = {}, scache[4][TW];
        int tags[4] = {-1, -1, -1, -1};
        float tau_lane = 0.f, tc_lane = 0.f;
        float sp_t = 0.f;
        float sp_b = 0.f, sp_acc = 0.f;
        float sunphase, strato_sunphase = 0.f;
        float gu_l = 0.f, gw_l = 0.f;
        float lane_mp[NCOMP] = {}, lane_mp_st = 0.f, lane_dt = 0.f;
    };
    struct SampleScratch {
        float paired[4] = {};
        float aero[4] = {}, rm[4][4] = {};
        ColumnRecord column{};
        float air[2] = {}, strato[2] = {};
    };

    // A complete warp traces one ray. Scratch belongs to this warp until the
    // caller's warp/block barrier; shared radiance is published before return.
    __device__ __forceinline__ void trace_ray(int cell, int64_t ray, const float *p,
                                              const float *target, const float *B, float *y,
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
        // global_st, dl[3], pu[4], se[3], direct[4].
        r.dl = ray_constants + 1;
        r.pu = ray_constants + 4;
        r.se = ray_constants + 8;
        r.direct = ray_constants + 11;
        if (lane == 0) {
            ray_constants[0] = hypotf(r.d[0], r.d[1]);
            for (int j = 0; j < 3; j++) {
                float x = 0;
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
            sh_basis(r.d[2], r.global_st, lane, r.y, aa, bb, dd);
        }
        SECTION_END(2)
        if (lane == 0) {
#ifndef FL_SPLIT_ON

            const float st = r.global_st;
            float cp = st > 1e-12f ? (float)(r.d[0] / st) : 1.f;
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

        // Simpson prefix state of lane w<TW (numpy_reference.py SimpsonPrefix): panel start
        // (local coordinate, extinction) and the segment sum.

        if constexpr (first)
            r.sunphase = hg(r.d[2], aerosol_g);

        if constexpr (TransportPolicy::strato)
            if (first && lane < TW)
                r.strato_sunphase = mie_phase(r.d[2], GW(lane), st_chi);

        if (lane < nq && nq <= 32) {
            r.gu_l = gu[lane];
            r.gw_l = gw[lane];
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
                phase_value = tropo_phase(r.d[2], GW(w), asca + FW);
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
                layer_live(live_a, live_a_hi, view.ir() + 1) ||
                (ProfilePolicy::strato &&
                 (layer_live(live_s, live_s_hi, view.ir()) ||
                  layer_live(live_s, live_s_hi, view.ir() + 1)));
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
                const bool sk_a = !TransportPolicy::thin_layers || layer_live(live_a, live_a_hi, ir) ||
                                  (ProfilePolicy::strato && layer_live(live_s, live_s_hi, ir));
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
                    float aero = sk_a ? support_value(view.height(),
                                                      TransportPolicy::combined_sh ? own_aero : own_aero * ha[ir],
                                                      false)
                                      : 0.f;
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
                                                     const PathSegment &path,
                                                     const NodeWeights &node,
                                                     const SampleLane &ownership, const View &view) {
        const float sample_ds = view.sample.ds;
        if constexpr (TransportPolicy::fast_warm) {
            const int bi = step - step0;
            float mp[NCOMP];
#pragma unroll
            for (int k = 0; k < NCOMP; ++k)
                mp[k] = __shfl_sync(0xffffffffu, r.lane_mp[k], bi);
            const float mp_st = __shfl_sync(0xffffffffu, r.lane_mp_st, bi);
            const float dt6 = __shfl_sync(0xffffffffu, r.lane_dt, bi);
            if (ownership.active) {
                const int w = ownership.wave;
                const bool ok = r.lane_wg > 0.f && r.lane_am >= TRACE_MIN;
                const float lane_scale =
                    ok ? expf(r.lane_log - r.lane_wg * logf(r.lane_am / r.lane_wg)) : 1.f;
                float pt_st = 0.f;
#ifdef STRATO
                pt_st = view.sample.st_prof;
#endif
                const float bk = extinction_at(view.sample.prof, pt_st, optics, GW(w));
                const float bm = extinction_at(mp, mp_st, optics, GW(w));
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
                    const float tk = path.node(node.u);
                    const float bk = extinction_at(s.prof, pst, optics, GW(w));
                    r.sp_acc +=
                        simpson_piece(path, r.sp_t, tk, r.sp_b, bk, optics, GW(w));
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
                                                   int step0, const PathSegment &path, const NodeWeights &node,
                                                   const SampleLane &ownership, const View &view) {
        SampleScratch scratch;
        if constexpr (TransportPolicy::fast_warm) {
            initialize_source(r, ownership);
            shade_corners(r, scratch, ray, seg, step, ownership, view);
            attenuate_sample(r, scratch, ray, seg, step, step0, path, node, ownership,
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
        attenuate_sample(r, scratch, ray, seg, step, step0, path, node, ownership,
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
            float length = fabsf(edges[seg].signed_length);
            if (length <= 1e-8f)
                continue;
            bool low = (lower[ray] >> (seg - edge_off[ray])) & 1;
            const PathSegment path = path_segment(r.p, ray_geometry[ray], edges + seg, low);
            // Simpson prefix: beta(a) is evaluated anew for every segment (never carried over a
            // skipped b-a<=1e-8 segment).
            r.sp_t = path.start();
            r.sp_acc = 0.f;
            float seg_be = 0.f;
            if constexpr (TransportPolicy::fast_warm) {
                float end_p[NCOMP] = {}, end_st = 0.f;
                if (lane < 2) {
                    const float sh = path.height(lane == 0 ? path.start() : path.end());
                    profile_at(sh, profile, end_p, end_st);
                }
                float pa[NCOMP], pb[NCOMP];
#pragma unroll
                for (int k = 0; k < NCOMP; ++k) {
                    pa[k] = __shfl_sync(0xffffffffu, end_p[k], 0);
                    pb[k] = __shfl_sync(0xffffffffu, end_p[k], 1);
                }
                const float pa_st = __shfl_sync(0xffffffffu, end_st, 0);
                const float pb_st = __shfl_sync(0xffffffffu, end_st, 1);
                if (lane < TW) {
                    r.sp_b = extinction_at(pa, pa_st, optics, GW(lane));
                    seg_be = extinction_at(pb, pb_st, optics, GW(lane));
                }
            } else if (lane < TW && !PROD_ON && sinv_mode != 2) {
                r.sp_b = beta_at(path, r.sp_t, optics, GW(lane));
            }
            constexpr int capacity =
                TransportPolicy::batched_points ? TransportPolicy::batch_size : 1;
            for (int step0 = 0; step0 < nq; step0 += capacity) {
                const int nb = min(capacity, nq - step0);
                if constexpr (TransportPolicy::fast_warm) {
                    __syncwarp();  // every lane has finished reading the previous batch of pts
                    const NodeWeights node =
                        quadrature_node(step0 + lane, nq, lane < nb, gu, gw, r.gu_l, r.gw_l);
                    float tk = 0.f;
                    if (lane < nb)
                        tk = point_warm(pts[lane], r.p, r.d, r.B, r.dl, r.global_st,
                                        path, node.u, node.w, h, nh, chi, nc, profile);
                    const bool is_last = (step0 + nb == nq);
                    float prev_t = __shfl_up_sync(0xffffffffu, tk, 1);
                    if (lane == 0)
                        prev_t = r.sp_t;
                    float curr_t = tk;
                    if (is_last && lane == nb)
                        curr_t = path.end();
                    const int n_panels = nb + (is_last ? 1 : 0);
                    if (lane < n_panels) {
                        const float tm = (prev_t + curr_t) * .5f;
                        profile_at(path.height(tm), profile,
                                   r.lane_mp, r.lane_mp_st);
                        r.lane_dt = path.width(prev_t, curr_t) / 6.f;
                    }
                    r.sp_t = __shfl_sync(0xffffffffu, tk, nb - 1);
                    __syncwarp();
                    for (int step = step0; step < step0 + nb; step++) {
                        const SampleLane ownership(lane, step, step0 + nb);
                        auto view = sample_view(pts, nullptr, step0, ownership.sample);
                        consume_sample(r, ray, seg, step, step0, path, node, ownership,
                                       view);
                    }
                    if (is_last) {
                        float mp_end[NCOMP];
#pragma unroll
                        for (int k = 0; k < NCOMP; ++k)
                            mp_end[k] = __shfl_sync(0xffffffffu, r.lane_mp[k], nb);
                        const float mp_end_st = __shfl_sync(0xffffffffu, r.lane_mp_st, nb);
                        const float dt6_end = __shfl_sync(0xffffffffu, r.lane_dt, nb);
                        if (lane < TW) {
                            const float bm_end = extinction_at(mp_end, mp_end_st, optics, GW(lane));
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
                        evaluate_point(pts[lane], r.p, r.d, r.B, r.dl, r.global_st, path, node.u, node.w, h, nh, chi, nc, profile, sun_h,
                                       nsh, sun_col, nsx, first, st_sun_col);
                    __syncwarp();
                }
                for (int step = step0; step < step0 + nb; step += TransportPolicy::sample_stride) {
                    SECTION_BEGIN()
                    const SampleLane ownership(lane, step, step0 + nb);
                    auto view = sample_view(pts, samples + warp, step0, ownership.sample);
                    const NodeWeights node = quadrature_node(
                        step, nq, true, gu, gw, r.gu_l, r.gw_l);
                    if constexpr (!TransportPolicy::batched_points) {
                        if (lane == 0)
                            evaluate_point(samples[warp], r.p, r.d, r.B, r.dl,
                                           r.global_st, path, node.u, node.w, h, nh,
                                           chi, nc, profile, sun_h, nsh, sun_col, nsx, first,
                                           st_sun_col);
                    }
                    SECTION_END(0)
                    consume_sample(r, ray, seg, step, step0, path, node, ownership,
                                   view);
                }
                }
            }
            // Simpson end piece to b (beta(b) anew), then the segment sum into tau (Kahan), as
            // numpy_reference.py SimpsonPrefix.finish.
            if constexpr (!TransportPolicy::fast_warm) {
                if (lane < TW && !PROD_ON && sinv_mode != 2) {
                    const float te = path.end();
                    const float be =
                        beta_at(path, te, optics, GW(lane));
                    r.sp_acc +=
                        simpson_piece(path, r.sp_t, te, r.sp_b, be, optics, GW(lane));
                    kah(r.tau_lane, r.tc_lane, r.sp_acc);
                }
            }
        }
    }

    // Accumulator and correction planes share the chunk-local indexing of the output.
    struct SharedViews {
        int nm;
        float *acc, *error;
        ProjectionSum *ground;
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
    __device__ __forceinline__ SharedViews bind_shared(int cell, int part, float *shared) {
        SharedViews memory;
        memory.nm = phase == 0 ? NMOM * TW : 0;
        if constexpr (TransportPolicy::global_accum) {
            const int64_t base = (int64_t)cell * memory.nm;
            memory.acc = moments + base;
            memory.error = moment_error + base;
            if constexpr (TransportPolicy::split_cells) {
                if (part > 0) {
                    const int64_t stride = (int64_t)cells * memory.nm;
                    memory.acc = split_scratch + 2 * (part - 1) * stride + base;
                    memory.error = memory.acc + stride;
                }
            }
            memory.ground = (ProjectionSum *)shared;
        } else {
            memory.acc = shared;
            memory.error = shared + memory.nm;
            memory.ground = (ProjectionSum *)(memory.error + memory.nm);
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
                    float zero = 0.f;
                    for (int k = tid; k < memory.nm; k += blockDim.x)
                        memory.acc[k] = zero;
                    for (int w = tid; w < TW; w += blockDim.x)
                        flux[cell * FW + GW(w)] = zero * (albedo / PI);
                }
            }
        } else {
            for (int k = tid; k < memory.nm; k += blockDim.x) {
                memory.acc[k] = 0.f;
                memory.error[k] = 0.f;
            }
            for (int w = tid; w < TW; w += blockDim.x)
                memory.ground[w] = {};
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
    // warps' radiance; moments retain cell-ray order and compensated accumulation.
    __device__ __forceinline__ void project_completed_cell(int cell, int64_t ray_lo, int64_t cb,
                                                           int64_t ce, const SharedViews &memory,
                                                           const float *target, float *y,
                                                           float *cd) {

        // Last ray of this cell: project rays cb..ce-1 in order (== original
        // round-then-warp order).
        __threadfence();
        float *mrow = moments + (int64_t)cell * memory.nm;
        float *error = moment_error + (int64_t)cell * memory.nm;
        ProjectionSum gsum{};
#pragma unroll 1
        for (int64_t r = cb; r < ce; r++) {
            const float *drow = dirs + 3 * r;
            const float st = hypotf(drow[0], drow[1]);
            sh_basis(drow[2], st, lane, y, aa, bb, dd);
            if (lane == 0) {
                float cp = st > 1e-12f ? (float)(drow[0] / st) : 1.f;
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
                ProjectionSum a = r == cb ? ProjectionSum{} : ProjectionSum{mrow[k], error[k]};
                a.add(v);
                error[k] = a.error;
                mrow[k] = r + 1 == ce ? a.value() : a.sum;
            }
            if (lane < TW) {
                int w = lane;
                float dlz = 0;
                for (int k = 0; k < 3; k++)
                    dlz += target[3 * k + 2] * drow[k];
                float tr = (__ldcg(rr + 4 * w) + __ldcg(rr + 4 * w + 1)) + __ldcg(rr + 4 * w + 2),
                      wg = weights[r] * fmaxf(0.f, (float)dlz);
                gsum.add(tr * wg);
            }
            __syncwarp();
        }
        if (lane < TW)
            flux[cell * FW + GW(lane)] = gsum.value() * (albedo / PI);
    }
    __device__ __forceinline__ void publish_ray(int cell, int64_t ray, int64_t ray_lo,
                                                const SharedViews &memory, const float *target,
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
        const float *p = starts + 5 * (int64_t)cell, *target = targets + 9 * cell, *d = dirs + 3 * ray;
        float *dl = ray_constants + 1;
        float *pu = ray_constants + 4;
        float *se = ray_constants + 8;
        float *direct = ray_constants + 11;
        if (lane == 0) {
            for (int k = 0; k < 9; ++k)
                fB[k] = target[k];
            for (int j = 0; j < 3; ++j) {
                float x = 0.f;
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
        const float sunphase = hg(d[2], aerosol_g);
        float ray_ext[NCOMP] = {};
        float ray_asca = 0.f;
        float phase_value = sunphase;
        if (lane < TW) {
#pragma unroll
            for (int k = 0; k < NCOMP; ++k)
                ray_ext[k] = ext[k * FW + GW(lane)];
            ray_asca = asca[GW(lane)];
            if constexpr (TransportPolicy::tropo_mie)
                phase_value = tropo_phase(d[2], GW(lane), asca + FW);
        }
        const float gu_l = (lane < nq && nq <= 32) ? gu[lane] : 0.f;
        const float gw_l = (lane < nq && nq <= 32) ? gw[lane] : 0.f;
        float lane_rad[4] = {}, lane_rc[4] = {};
        float tau_lane = 0.f, tc_lane = 0.f;
        const uint32_t sb = edge_off[ray], se_idx = edge_off[ray + 1];
        for (uint32_t seg = sb; seg + 1 < se_idx; ++seg) {
            const float length = fabsf(edges[seg].signed_length);
            if (length <= 1e-8f)
                continue;
            const bool low = (lower[ray] >> (seg - sb)) & 1;
            const PathSegment path = path_segment(p, ray_geometry[ray], edges + seg, low);
            float sp_t = path.start();
            float sp_acc = 0.f, sp_b = 0.f, seg_be = 0.f;
            float end_p[NCOMP] = {};
            if (lane < 2) {
                const float sh = path.height(lane == 0 ? path.start() : path.end());
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
                float tk = 0.f;
                if (lane < nb)
                    tk = point_first(fpts[lane], p, d, fB, dl, path, node.u,
                                     node.w, h, nh, chi, nc, profile, sun_h, nsh, sun_col, nsx);
                float prev_t = __shfl_up_sync(0xffffffffu, tk, 1);
                if (lane == 0)
                    prev_t = sp_t;
                float curr_t = tk;
                if (is_last && lane == nb)
                    curr_t = path.end();
                const int n_panels = nb + (is_last ? 1 : 0);
                if (lane < n_panels) {
                    const float tm = (prev_t + curr_t) * .5f;
                    float mp_tmp[NCOMP], pst_unused;
                    profile_at(path.height(tm), profile, mp_tmp,
                               pst_unused);
                    float *row = fmp + lane * (NCOMP + 1);
#pragma unroll
                    for (int k = 0; k < NCOMP; ++k)
                        row[k] = mp_tmp[k];
                    row[NCOMP] = path.width(prev_t, curr_t) / 6.f;
                }
                sp_t = __shfl_sync(0xffffffffu, tk, nb - 1);
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
                const float *p = starts + 5 * (int64_t)cell, *target = targets + 9 * cell;
                float B[9];
                for (int k = 0; k < 9; ++k)
                    B[k] = target[k];
                float *y = memory.Y + warp * YPAD, *rad = memory.rads + warp * 4 * TW;
                float *rc = memory.err + warp * 4 * TW, *cd = memory.cosdest + warp * 33;
                trace_ray(cell, ray, p, target, B, y, rad, rc, cd);
                publish_ray(cell, ray, ray_lo, memory, target, y, rad, cd);
            }
        }
    }
    // Consume one complete round in warp order.
    __device__ __forceinline__ void accumulate_round(int64_t round, int64_t end, int64_t window,
                                                     const SharedViews &memory,
                                                     const float *target) {

        SECTION_BEGIN()
        for (int k = tid; k < memory.nm; k += blockDim.x) {
            int moment = k / TW, w = k % TW;
            ProjectionSum acc{memory.acc[k], memory.error[k]};
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
                    acc.add(v);
                }
            memory.acc[k] = acc.sum;
            memory.error[k] = acc.error;
        }
        for (int w = tid; w < TW; w += blockDim.x)
            for (int wp = 0; wp < nwarps; wp++)
                if (round + wp < end) {
                    int64_t rridx = ordered_ray(round + wp, window);
                    const float *d = dirs + 3 * rridx;
                    float dlz = 0;
                    for (int k = 0; k < 3; k++)
                        dlz += target[3 * k + 2] * d[k];
                    float *rr = memory.rads + wp * 4 * TW,
                          tr = (rr[4 * w] + rr[4 * w + 1]) + rr[4 * w + 2],
                          wt = weights[rridx] * fmaxf(0.f, (float)dlz);
                    memory.ground[w].add(tr * wt);
                }
        SECTION_END(5)
    }
    __device__ __forceinline__ void trace_cell_rays(int cell, int part, const SharedViews &memory) {
        const float *p = starts + 5 * (int64_t)cell, *target = targets + 9 * cell;
        const float *B_ptr;
        float B_local[9];
        if constexpr (TransportPolicy::fast_warm) {
            if (tid < 9)
                cell_B[tid] = target[tid];
            __syncthreads();
            B_ptr = cell_B;
        } else {
            for (int k = 0; k < 9; ++k)
                B_local[k] = target[k];
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
        if constexpr (!TransportPolicy::split_cells && !TransportPolicy::split_rays) {
            if (phase == 0 && !NOOUT_ON)
                for (int k = tid; k < memory.nm; k += blockDim.x)
                    moments[(int64_t)cell * memory.nm + k] = memory.acc[k] + memory.error[k];
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
                            ProjectionSum g{};
                            for (int q = 0; q < SPLIT_S; q++) {
                                const auto *v = split_flux + ((int64_t)q * cells + cell) * TW + w;
                                g.add(__ldcg(&v->sum));
                                g.add(__ldcg(&v->error));
                            }
                            flux[cell * FW + GW(w)] = g.value() * (albedo / PI);
                        }
                        if (phase == 0)
                            for (int k = tid; k < memory.nm; k += blockDim.x) {
                                const int64_t idx = (int64_t)cell * memory.nm + k;
                                const int64_t stride = (int64_t)cells * memory.nm;
                                ProjectionSum v{__ldcg(moments + idx), __ldcg(moment_error + idx)};
                                for (int q = 1; q < SPLIT_S; q++) {
                                    const float *row = split_scratch + 2 * (q - 1) * stride + idx;
                                    v.add(__ldcg(row));
                                    v.add(__ldcg(row + stride));
                                }
                                moments[idx] = v.value();
                            }
                    }
                }
            } else {
                if (phase != 2 && !NOOUT_ON)
                    for (int w = tid; w < TW; w += blockDim.x)
                        flux[cell * FW + GW(w)] = memory.ground[w].value() * (albedo / PI);
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
        extern __shared__ float4 shared_vectors[];
        const SharedViews memory = bind_shared(cell, split_part, (float *)shared_vectors);
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
    int boundary_arg, const int64_t *cell_off, const float *starts, const float *targets,
    const float *dirs, const float *weights, const uint32_t *edge_off, const PathEdge *edges,
    const uint16_t *lower, const int32_t *end_j, const float *end_f, const float *end_col,
    const uint8_t *hits, const float *gu, const float *gw, const float *aa, const float *bb,
    const float *dd, const float *h, const float *chi, const float *hr, const float *ha,
    const float *ext, const float *asca, const float *sun_h, const float *sun_col, const float *gm,
    const int64_t *goff, const float *gdir, const float *gval, float height, float aerosol_g,
    float mix, float albedo, const float *src, const float *boundary, float *moments,
    float *flux,
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
    const float *profile_table, int profile_mask, float air_surface, float *moment_error,
    const RayGeometry *ray_geometry
#ifdef LPT_ACTIVE
    ,
    const int *cell_order
#ifdef SPLIT_ACTIVE
    ,
    float *split_scratch, ProjectionSum *split_flux,
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
    transport.ray_geometry = ray_geometry;
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
    transport.moment_error = moment_error;
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
        const float4 rv = *(const float4 *)(ray_rads + (begin + r - ray_base) * 4 * TW + 4 * w);
        const float weight = weights[begin + r];
        const float trace = (rv.x + rv.y) + rv.z;
        rad[r][w] = rv.x * weight;
        rad[r][TW + w] = rv.y * weight;
        rad[r][2 * TW + w] = rv.z * weight;
        rad[r][3 * TW + w] = rv.w * weight;
        rad[r][4 * TW + w] = trace * weight;
        rad[r][5 * TW + w] = trace * ground_weight[r];
    }
}

#if (defined(FL_SPLIT_ON) && defined(LAUNCH_RAY_LIMIT)) || defined(FAST_VPROD_ON)
__device__ __forceinline__ void sh_basis_fast(float mu, float st, int lane, float *__restrict__ y,
                                              const float *__restrict__ aa,
                                              const float *__restrict__ bb,
                                              const float *__restrict__ dd) {
    mu = clip(mu, -1.f, 1.f);
    for (int m = lane; m <= SH_L; m += 32) {
        float diag = 0.28209479177387814f;
        for (int j = 1; j <= m; j++)
            diag = (diag * dd[j]) * st;
        const float mult = m ? 1.4142135623730951f : 1.f;
        float z0 = diag;
        y[ix(m, m)] = mult * z0;
        if (m < SH_L) {
            float z1 = (sqrtf(fmaf(2.f, (float)m, 3.f)) * mu) * z0;
            y[ix(m + 1, m)] = mult * z1;
            for (int l = m + 2; l <= SH_L; l++) {
                const int k = ix(l, m);
                const float z2 = fmaf(aa[k] * mu, z1, -(bb[k] * z0));
                y[k] = mult * z2;
                z0 = z1;
                z1 = z2;
            }
        }
    }
}
// Batch eight rays: every warp builds one basis row, all threads load weighted radiance,
// and each output combines the FP32 batch partials with Neumaier summation. No atomics.
#define FLP_THREADS 256
#define FLP_RAYS 8
#define FLP_WAVES ((TW + 1) / 2)
#define FLP_MOMENTS ((NMOM + 127) / 128)
extern "C" __global__ void __launch_bounds__(FLP_THREADS)
    fl_project(int cells, long long ray_lo, const int64_t *__restrict__ cell_off,
               const float *__restrict__ targets, const float *__restrict__ dirs,
               const float *__restrict__ weights, const float *__restrict__ aa,
               const float *__restrict__ bb, const float *__restrict__ dd,
               const float *__restrict__ ray_rads, float albedo, float *__restrict__ moments,
               float *__restrict__ flux) {
    const int cell = blockIdx.x;
    if (cell >= cells)
        return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nm = NMOM * TW;
    const int mo = tid % 128, wbase = (tid / 128) * FLP_WAVES;
    __shared__ float y[FLP_RAYS][K], cd[FLP_RAYS][33];
    __shared__ float rad[FLP_RAYS][6 * TW], wg[FLP_RAYS];
    const int64_t cb = cell_off[cell], ce = cell_off[cell + 1];
    const float *target = targets + 9 * cell;
    ProjectionSum acc[FLP_MOMENTS][FLP_WAVES] = {}, gsum{};
#pragma unroll 1
    for (int64_t rb = cb; rb < ce; rb += FLP_RAYS) {
        const int nr = (int)min((int64_t)FLP_RAYS, ce - rb);
        if (warp < nr) {
            const float *d = dirs + 3 * (rb + warp);
            const float st = sqrtf(fmaf(d[0], d[0], d[1] * d[1]));
            sh_basis_fast(d[2], st, lane, y[warp], aa, bb, dd);
            if (lane == 0) {
                float cp = st > 1e-12f ? (float)(d[0] / st) : 1.f;
                cp = clip(cp, -1, 1);
                cd[warp][0] = 1;
                cd[warp][1] = cp;
                const float cp2 = 2.f * cp;
                for (int m = 2; m <= SH_L; m++)
                    cd[warp][m] = fmaf(cp2, cd[warp][m - 1], -cd[warp][m - 2]);
                float dlz = 0;
                for (int k = 0; k < 3; k++)
                    dlz = fmaf(target[3 * k + 2], d[k], dlz);
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
                    const float *rrow = rad[r] + (moment < 4 ? moment : 4) * TW + wbase;
#pragma unroll
                    for (int q = 0; q < FLP_WAVES; q++)
                        if (wbase + q < TW)
                            part[q] = fmaf(rrow[q], yc, part[q]);
                }
#pragma unroll
                for (int q = 0; q < FLP_WAVES; q++)
                    acc[j][q].add(part[q]);
            }
        }
        if (tid < TW) {
            float part = 0.f;
            for (int r = 0; r < nr; r++)
                part += rad[r][5 * TW + tid];
            gsum.add(part);
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < FLP_MOMENTS; j++)
        if (mo + j * 128 < NMOM) {
#pragma unroll
            for (int q = 0; q < FLP_WAVES; q++)
                if (wbase + q < TW)
                    moments[(int64_t)cell * nm + (mo + j * 128) * TW + wbase + q] = acc[j][q].value();
        }
    if (tid < TW)
        flux[cell * FW + GW(tid)] = gsum.value() * (albedo / PI);
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
    sos_vprod_y(int nray, const float *dirs, const float *aa, const float *bb,
                const float *dd, float *yout) {
    const int lane = threadIdx.x & 31,
              t = (int)blockIdx.x * ((int)blockDim.x >> 5) + ((int)threadIdx.x >> 5);
    if (t >= nray)
        return;
    const float *d = dirs + 3 * (int64_t)t;
    sh_basis(d[2], hypotf(d[0], d[1]), lane, yout + (int64_t)t * K, aa, bb, dd);
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
                  const int64_t *cell_off, const float *starts, const float *dirs,
                  const uint32_t *edge_off, const PathEdge *edges, const uint16_t *lower, const RayGeometry *ray_geometry,
                  const float *gu, const float *h, const float *chi, const float *hr,
                  const float *ha, int2 *c0cp
#ifdef STRATO
                  , const float *st_density
#endif
    ) {
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
            if (idx < nh) {
                const float r = hr[idx];
                bool live = ha[idx] > SKIP_TAU * r;
#ifdef STRATO
                live = live || (st_density[idx] > SKIP_TAU * r);
#endif
                la[b] = live;
            }
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
    const float *p = starts + 5 * (int64_t)cell, *d = dirs + 3 * ray;
    const float st = hypotf(d[0], d[1]);
    const float gu_l = (lane < nq && nq <= 32) ? gu[lane] : 0.f;
    const uint32_t sb = edge_off[ray], se = edge_off[ray + 1];
    for (uint32_t seg = sb; seg + 1 < se; ++seg) {
        const float length = fabsf(edges[seg].signed_length);
        const int64_t slot0 = (int64_t)(seg - ray) * nq - base;
        if (length <= 1e-8f) {
            for (int step = lane; step < nq; step += 32)
                c0cp[slot0 + step] = make_int2(-1, 0);
            continue;
        }
        const bool low = (lower[ray] >> (seg - sb)) & 1;
        const PathSegment path = path_segment(p, ray_geometry[ray], edges + seg, low);
        for (int step0 = 0; step0 < nq; step0 += 32) {
            const int step = step0 + lane;
            if (step < nq) {
                const float u = (nq <= 32) ? gu_l : gu[step];
                const PathCell sample = path_cell(p, d, path, u, h, nh, chi, nc);
                const float sh = sample.point.h, cx = sample.point.cx, sx = sample.point.sx;
                const int ir = sample.ir, j = sample.j;
                const bool any_live =
                    !outside_support(sh, false) &&
                    (FW != 1 && nh <= 128 ? (Transport<Sample>::layer_live(live_a, live_a_hi, ir) ||
                                  Transport<Sample>::layer_live(live_a, live_a_hi, ir + 1))
                               : true);
                if (!any_live) {
                    c0cp[slot0 + step] = make_int2(-1, 0);
                } else {
                    const float cp =
                        st > 1e-12f ? clip((d[0] * cx + d[1] * sx) / st, -1.f, 1.f) : 1.f;
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
                  const float *ha, int nh, int nc, float *aero
#ifdef STRATO
                  , const float *st_density
#endif
    ) {
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
    bool live = FW == 1 || nh > 128 || ha[ir] > SKIP_TAU * hr[ir];
#ifdef STRATO
    live = live || (nh <= 128 && st_density[ir] > SKIP_TAU * hr[ir]);
#endif
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
// Each block synthesizes all azimuths for one live cell and one polar knot.
extern "C" __global__ void sh_direction_table(const int *owners, const float *src,
                                             const float *polar, const float *azimuth,
                                             float *table) {
    const int c = blockIdx.x, cell = owners[c], theta = blockIdx.y, tid = threadIdx.x;
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
        table[((long long)c * SH_NODES + theta) * SH_NODES * FAST_WAVE_STRIDE + i] = sum;
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

__device__ __forceinline__ float sh_spline_at(const float *const *cells, int cell,
                                              int64_t spline_off, const float *theta_w,
                                              const ShSpline &phi) {
    const float *table = cells[cell];
    if (!table)
        return 0.f;
    const float *row = table + spline_off;
    float value = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float *r = row + j * (SH_NODES * FAST_WAVE_STRIDE);
        float azimuth = phi.w[0] * r[0];
        azimuth = fmaf(phi.w[1], r[FAST_WAVE_STRIDE], azimuth);
        azimuth = fmaf(phi.w[2], r[2 * FAST_WAVE_STRIDE], azimuth);
        azimuth = fmaf(phi.w[3], r[3 * FAST_WAVE_STRIDE], azimuth);
        value = fmaf(theta_w[j], azimuth, value);
    }
    return value;
}

__device__ __forceinline__ void sh_spline_pair(const float *const *__restrict__ cells, int cell0, int cell1,
                                               int64_t spline_off, const float *__restrict__ theta_w,
                                               const ShSpline &phi, float &out0, float &out1) {
    const float *__restrict__ t0 = cells[cell0];
    const float *__restrict__ t1 = cells[cell1];
    float v0 = 0.f, v1 = 0.f;
    if (t0 && t1) {
        const float *__restrict__ row0 = t0 + spline_off;
        const float *__restrict__ row1 = t1 + spline_off;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t joff = (int64_t)j * (SH_NODES * FAST_WAVE_STRIDE);
            const float *__restrict__ r0 = row0 + joff;
            const float *__restrict__ r1 = row1 + joff;
            const float a00 = r0[0], a01 = r0[FAST_WAVE_STRIDE],
                        a02 = r0[2 * FAST_WAVE_STRIDE], a03 = r0[3 * FAST_WAVE_STRIDE];
            const float b00 = r1[0], b01 = r1[FAST_WAVE_STRIDE],
                        b02 = r1[2 * FAST_WAVE_STRIDE], b03 = r1[3 * FAST_WAVE_STRIDE];
            const float twj = theta_w[j];
            float az0 = phi.w[0] * a00;
            az0 = fmaf(phi.w[1], a01, az0);
            az0 = fmaf(phi.w[2], a02, az0);
            az0 = fmaf(phi.w[3], a03, az0);
            v0 = fmaf(twj, az0, v0);
            float az1 = phi.w[0] * b00;
            az1 = fmaf(phi.w[1], b01, az1);
            az1 = fmaf(phi.w[2], b02, az1);
            az1 = fmaf(phi.w[3], b03, az1);
            v1 = fmaf(twj, az1, v1);
        }
    } else {
        if (t0)
            v0 = sh_spline_at(cells, cell0, spline_off, theta_w, phi);
        if (t1)
            v1 = sh_spline_at(cells, cell1, spline_off, theta_w, phi);
    }
    out0 = v0;
    out1 = v1;
}
#endif

#define FS_WARPS (FAST_THREADS / 32)
#define FS_BATCH FAST_BATCH
extern "C" __global__ void __launch_bounds__(FAST_THREADS, 5)
    sos_fast_slim(int cells, int nh, int nc, int nq, long long r0, int q0, int q1,
                  long long slim_base, const int64_t *__restrict__ cell_off,
                  const float *__restrict__ starts, const float *__restrict__ targets,
                  const float *__restrict__ dirs, const uint32_t *__restrict__ edge_off,
                  const PathEdge *__restrict__ edges, const uint16_t *__restrict__ lower,
                  const RayGeometry *__restrict__ ray_geometry, const int32_t *__restrict__ end_j,
                  const float *__restrict__ end_f, const float *__restrict__ end_col,
                  const uint8_t *__restrict__ hits, const float *__restrict__ gu,
                  const float *__restrict__ gw, const float *__restrict__ h,
                  const float *__restrict__ chi, const float *__restrict__ hr,
                  const float *__restrict__ ha, const float *__restrict__ ext,
                  const float *__restrict__ sun_h, int nsh, float height,
                  float mix, const float *__restrict__ src, const float *__restrict__ boundary,
                  const float *__restrict__ slim_aero, const float *__restrict__ profile_table,
                  int profile_mask, float air_surface, float *__restrict__ ray_rads
#ifdef STRATO
                  , const float *__restrict__ st_ext, const float *__restrict__ st_density,
                  const float *__restrict__ st_end_col, float st_center, float st_sigma, float st_norm
#endif
    ) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    __shared__ float sh_ext[NCOMP + 1][FAST_WAVE_STRIDE];
    __shared__ uint32_t sh_live[4];
    __shared__ float sh_h[128], sh_chi[128];
    for (int i = tid; i < NCOMP * FW; i += FAST_THREADS)
        sh_ext[i / FW][i % FW] = ext[i];
#ifdef STRATO
    if (tid < FW)
        sh_ext[NCOMP][tid] = st_ext[tid];
#endif
    if (nh <= 128 && tid < nh)
        sh_h[tid] = h[tid];
    if (nc <= 128 && tid < nc)
        sh_chi[tid] = chi[tid];
    if (warp == 0) {
        uint32_t m[4] = {~0u, ~0u, ~0u, ~0u};
        if (FW != 1 && nh <= 128) {
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int idx = lane + 32 * b;
                bool live = false;
                if (idx < nh) {
                    const float r = hr[idx];
                    live = ha[idx] > SKIP_TAU * r;
#ifdef STRATO
                    live = live || (st_density[idx] > SKIP_TAU * r);
#endif
                }
                m[b] = __ballot_sync(0xffffffffu, live);
            }
        }
        if (lane < 4)
            sh_live[lane] = m[lane];
    }
    __syncthreads();
    const float *__restrict__ h_grid = nh <= 128 ? sh_h : h;
    const float *__restrict__ chi_grid = nc <= 128 ? sh_chi : chi;
    const int q = q0 + (int)blockIdx.x * FS_WARPS + warp;
    if (q >= q1)
        return;
    const int64_t ray = r0 + q;
    const ProfileParams params{height, sun_h + nsh, profile_table, profile_mask, air_surface
#ifdef STRATO
                               , st_center, st_sigma, st_norm
#endif
    };
#if SLIM_FASTMATH
    const FastProfileParams profile{1.f / params.height, params.aer2, params.table, params.mask,
                                    1.f / params.air_surface, params.st_center,
                                    1.f / params.st_sigma, 1.f / params.st_norm};
#else
    const ProfileParams &profile = params;
#endif
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
    const float *p = starts + 5 * (int64_t)cell, *d = dirs + 3 * ray;
#ifdef SH_ANGLE_BINS
    __shared__ float sh_phi[FS_WARPS][FS_BATCH];
    __shared__ int sh_theta_knot[FS_WARPS];
    __shared__ float sh_theta_w[FS_WARPS][4];
    float *phi_knots = sh_phi[warp];
    const float st = hypotf(d[0], d[1]);
    const bool has_st = st > 1e-12f;
    const float inv_st = has_st ? 1.f / st : 0.f;
    const float d0_n = d[0] * inv_st, d1_n = d[1] * inv_st;
    if (lane == 0) {
        const ShSpline th((float)(atan2f(st, d[2]) * (SH_ANGLE_BINS / PI)));
        sh_theta_knot[warp] = th.knot;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            sh_theta_w[warp][j] = th.w[j];
    }
#else
    float *phi_knots = nullptr;
    const bool has_st = false;
    const float d0_n = 0.f, d1_n = 0.f;
#endif
    constexpr int MP_ST = NCOMP;
    constexpr int MP_DT = ProfilePolicy::strato ? (NCOMP + 1) : NCOMP;
    __shared__ SlimPoint sh_pts[FS_WARPS][FS_BATCH];
    __shared__ float sh_mp[FS_WARPS][FS_BATCH + 1][MP_DT + 1];
    __shared__ float sh_B[FS_WARPS][9];
    __shared__ float sh_rc[FS_WARPS][7];
    float *B = sh_B[warp];
    float *dl = sh_rc[warp];
    float *pu = sh_rc[warp] + 3;
    if (lane == 0) {
        const float *target = targets + 9 * cell;
        for (int k = 0; k < 9; ++k)
            B[k] = target[k];
        for (int j = 0; j < 3; ++j) {
            float x = 0.f;
            for (int k = 0; k < 3; ++k)
                x = fmaf(target[3 * k + j], d[k], x);
            dl[j] = (float)x;
        }
        pu[0] = fmaf(-dl[0], dl[0], 1.f);
        pu[1] = fmaf(-dl[1], dl[1], 1.f);
        pu[2] = fmaf(-dl[2], dl[2], 1.f);
        pu[3] = -dl[0] * dl[2];
    }
    __syncwarp();
    const int half = lane >> 4, w = lane & 15;
    const float gu_l = (lane < nq && nq <= 32) ? gu[lane] : 0.f;
    const float gw_l = (lane < nq && nq <= 32) ? gw[lane] : 0.f;
    float lane_rad[4] = {}, lane_rc[4] = {};
    float tau_lane = 0.f, tc_lane = 0.f;
    const uint32_t sb = edge_off[ray], se = edge_off[ray + 1];
    for (uint32_t seg = sb; slim_aero && seg + 1 < se; ++seg) {
        const float length = fabsf(edges[seg].signed_length);
        if (length <= 1e-8f)
            continue;
        const bool low = (lower[ray] >> (seg - sb)) & 1;
        const PathSegment path = path_segment(p, ray_geometry[ray], edges + seg, low);
        float sp_t = path.start();
        float sp_acc = 0.f, sp_b = 0.f, seg_be = 0.f;
        float end_p[NCOMP] = {}, end_st = 0.f;
        if (lane < 2) {
            const float sh = path.height(lane == 0 ? path.start() : path.end());
            profile_at(sh, profile, end_p, end_st);
        }
        float pa[NCOMP], pb[NCOMP];
#pragma unroll
        for (int k = 0; k < NCOMP; ++k) {
            pa[k] = __shfl_sync(0xffffffffu, end_p[k], 0);
            pb[k] = __shfl_sync(0xffffffffu, end_p[k], 1);
        }
        const float pa_st = __shfl_sync(0xffffffffu, end_st, 0);
        const float pb_st = __shfl_sync(0xffffffffu, end_st, 1);
        if (lane < TW) {
            float ba = 0.f, bb = 0.f;
#pragma unroll
            for (int k = 0; k < NCOMP; ++k) {
                const float ek = sh_ext[k][lane];
                ba = fmaf(pa[k], ek, ba);
                bb = fmaf(pb[k], ek, bb);
            }
            if constexpr (ProfilePolicy::strato) {
                const float est = sh_ext[NCOMP][lane];
                ba = fmaf(pa_st, est, ba);
                bb = fmaf(pb_st, est, bb);
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
            float tk = 0.f, node_h = 0.f;
            if (lane < nb)
                tk = point_slim(sh_pts[warp][lane], p, d, B, dl, path,
                                node.u, node.w, h_grid, nh, chi_grid, nc, d0_n, d1_n, has_st, phi_knots, lane, node_h);
            float prev_t = __shfl_up_sync(0xffffffffu, tk, 1);
            if (lane == 0)
                prev_t = sp_t;
            float curr_t = tk;
            if (is_last && lane == nb)
                curr_t = path.end();
            const int n_panels = nb + (is_last ? 1 : 0);
            float mid_h = 0.f;
            if (lane < n_panels) {
                mid_h = path.height((prev_t + curr_t) * .5f);
                sh_mp[warp][lane][MP_DT] = path.width(prev_t, curr_t) * (1.f / 6.f);
            }
            const float mid_h_hi = __shfl_up_sync(0xffffffffu, mid_h, 16);
            const bool do_node = lane < nb;
            const bool do_mid = lane >= 16 && (lane - 16) < n_panels;
            if (do_node || do_mid) {
                float pr_tmp[NCOMP], pst_tmp = 0.f;
                profile_at(do_node ? node_h : mid_h_hi, profile, pr_tmp, pst_tmp);
                if (do_node) {
#pragma unroll
                    for (int k = 0; k < NCOMP; ++k)
                        sh_pts[warp][lane].prof[k] = pr_tmp[k];
#ifdef STRATO
                    sh_pts[warp][lane].st_prof = pst_tmp;
#endif
                } else {
                    const int mi = lane - 16;
#pragma unroll
                    for (int k = 0; k < NCOMP; ++k)
                        sh_mp[warp][mi][k] = pr_tmp[k];
                    if constexpr (ProfilePolicy::strato)
                        sh_mp[warp][mi][MP_ST] = pst_tmp;
                }
            }
            sp_t = __shfl_sync(0xffffffffu, tk, nb - 1);
            __syncwarp();
            for (int bi = 0; bi < nb; bi += 2) {
                const bool has_odd = (bi + 1 < nb);
                const int my_bi = (half && has_odd) ? (bi + 1) : bi;
                const bool active = (w < TW) && (half == 0 || has_odd);
                const SlimPoint &pt = sh_pts[warp][my_bi];
                float lane_val[4] = {};
                // Base 2 for the PTX exponential, natural log for the FP32 expf path.
                float log_scale = 0.f;
                float bk = 0.f, bm = 0.f;
                const float dt6 = sh_mp[warp][my_bi][MP_DT];
                if (active) {
                    const int ir0 = pt.ir;
                    const float ph = pt.h;
                    const bool in_aer_sup = !outside_support(ph, false);
                    const float d0 = support_value(ph, hr[ir0], true);
                    const float d1 = support_value(ph, hr[ir0 + 1], true);
                    float ha0 = 0.f, ha1 = 0.f;
                    if constexpr (!TransportPolicy::combined_sh) {
                        ha0 = ha[ir0];
                        ha1 = ha[ir0 + 1];
                    }
                    const bool live0 =
                        in_aer_sup && (FW == 1 || nh > 128 || ((sh_live[ir0 >> 5] >> (ir0 & 31)) & 1u));
                    const bool live1 =
                        in_aer_sup &&
                        (FW == 1 || nh > 128 || ((sh_live[(ir0 + 1) >> 5] >> ((ir0 + 1) & 31)) & 1u));
                    const int4 cells4 = *(const int4 *)pt.cell;
                    const int cc_arr[4] = {cells4.x, cells4.y, cells4.z, cells4.w};
                    float aero_pre[4] = {};
#ifdef SH_ANGLE_BINS
                    if (live0 || live1) {
                        const int th_knot = sh_theta_knot[warp];
                        const float *th_w = sh_theta_w[warp];
                        const ShSpline phi(sh_phi[warp][my_bi]);
                        const int64_t spline_off =
                            (int64_t)(th_knot * SH_NODES + phi.knot) * FAST_WAVE_STRIDE + w;
                        if (live0)
                            sh_spline_pair((const float *const *)slim_aero, cc_arr[0], cc_arr[1], spline_off, th_w, phi, aero_pre[0], aero_pre[1]);
                        if (live1)
                            sh_spline_pair((const float *const *)slim_aero, cc_arr[2], cc_arr[3], spline_off, th_w, phi, aero_pre[2], aero_pre[3]);
                    }
#else
                    const int64_t pt_slot = (int64_t)(seg - ray) * nq + (step0 + my_bi) - slim_base;
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        if ((c < 2) ? live0 : live1)
                            aero_pre[c] = slim_aero[(pt_slot * 4 + c) * AERO_W + w];
                    }
#endif
                    const float4 swv = *(const float4 *)pt.sw;
                    const float sw_arr[4] = {swv.x, swv.y, swv.z, swv.w};
                    const float pu0 = pu[0], pu1 = pu[1], pu2 = pu[2], pu3 = pu[3];
                    const float pu_a[4] = {pu0, pu1, pu2, pu3};
                    const float mix_iso = (1.f - mix) * (1.f / 3.f);
                    float lane_log = 0.f, lane_wg = 0.f, lane_am = 0.f;
#if SLIM_FASTMATH
                    float lane_exp = 0.f;
#endif
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        const float corner_sw = sw_arr[c];
                        const int64_t sbase = (int64_t)cc_arr[c] * 4;
                        float rm[4];
#pragma unroll
                        for (int k = 0; k < 4; ++k)
                            rm[k] = src[(sbase + k) * FAST_WAVE_STRIDE + w];
                        const float density = c < 2 ? d0 : d1;
                        const float aero_half =
                            ((c < 2) ? live0 : live1)
                                ? 0.5f * (TransportPolicy::combined_sh
                                              ? aero_pre[c]
                                              : aero_pre[c] * (c < 2 ? ha0 : ha1))
                                : 0.f;
                        const float iso = (mix_iso * ((rm[0] + rm[1]) + rm[2]));
                        float tensors[3];
#pragma unroll
                        for (int comp = 0; comp < 4; ++comp) {
                            const float4 tv = *(const float4 *)(pt.T + 4 * comp);
                            float rot = tv.x * rm[0];
                            rot = fmaf(tv.y, rm[1], rot);
                            rot = fmaf(tv.z, rm[2], rot);
                            rot = fmaf(tv.w, rm[3], rot);
                            const float puc = pu_a[comp];
                            const float tensor = fmaf(density, fmaf(mix, rot, iso * puc), aero_half * puc);
                            lane_val[comp] = fmaf(corner_sw, tensor, lane_val[comp]);
                            if (comp < 3)
                                tensors[comp] = tensor;
                        }
                        const float trace = (tensors[0] + tensors[1]) + tensors[2];
                        const bool good = trace >= TRACE_MIN;
                        if (good) {
#if SLIM_FASTMATH
                            const int bits = __float_as_int(trace);
                            const int e = (bits >> 23) - 127;
                            const float m = __int_as_float((bits & 0x007fffff) | 0x3f800000);
                            lane_exp = fmaf(corner_sw, (float)e, lane_exp);
                            lane_log = fmaf(corner_sw, __log2f(m), lane_log);
#else
                            lane_log = fmaf(corner_sw, logf(trace), lane_log);
#endif
                            lane_wg += corner_sw;
                            lane_am = fmaf(corner_sw, trace, lane_am);
                        }
                    }
                    const bool ok = lane_wg > 0.f && lane_am >= TRACE_MIN;
                    if (ok) {
                        const float avg = lane_am / lane_wg;
#if SLIM_FASTMATH
                        const int ab = __float_as_int(avg);
                        const int ae = (ab >> 23) - 127;
                        const float am = __int_as_float((ab & 0x007fffff) | 0x3f800000);
                        log_scale = fmaf(-lane_wg, (float)ae, lane_exp) + fmaf(-lane_wg, __log2f(am), lane_log);
#else
                        log_scale = fmaf(-lane_wg, logf(avg), lane_log);
#endif
                    }
#pragma unroll
                    for (int k = 0; k < NCOMP; ++k) {
                        const float ek = sh_ext[k][w];
                        bk = fmaf(pt.prof[k], ek, bk);
                        bm = fmaf(sh_mp[warp][my_bi][k], ek, bm);
                    }
#ifdef STRATO
                    const float est = sh_ext[NCOMP][w];
                    bk = fmaf(pt.st_prof, est, bk);
                    bm = fmaf(sh_mp[warp][my_bi][MP_ST], est, bm);
#endif
                }
                const float bk_even = __shfl_up_sync(0xffffffffu, bk, 16);
                if (half == 0 && w < TW)
                    sp_acc = fmaf(dt6, fmaf(4.f, bm, sp_b + bk), sp_acc);
                const float acc_even = __shfl_up_sync(0xffffffffu, sp_acc, 16);
                float my_acc = sp_acc;
                if (half == 1 && active)
                    my_acc = fmaf(dt6, fmaf(4.f, bm, bk_even + bk), acc_even);
                const float acc_odd = __shfl_down_sync(0xffffffffu, my_acc, 16);
                const float bk_odd = __shfl_down_sync(0xffffffffu, bk, 16);
                if (half == 0 && w < TW) {
                    sp_acc = has_odd ? acc_odd : sp_acc;
                    sp_b = has_odd ? bk_odd : bk;
                }
                float paired[4] = {};
                if (active) {
#if SLIM_FASTMATH
                    float att_exp;
                    asm("ex2.approx.ftz.f32 %0, %1;"
                        : "=f"(att_exp)
                        : "f"(fmaf(-(seg_tau + my_acc), 1.4426950408889634f, log_scale)));
                    const float lane_att = pt.ds * att_exp;
                    const float lane_scale = 1.f;
#else
                    // Exact path: (component x scale) x (ds x attenuation) with separate exponentials, so a
                    // normal contribution is never lost to an underflowed combined exponent.
                    const float lane_scale = expf(log_scale);
                    const float lane_att = pt.ds * expf(-(seg_tau + my_acc));
#endif
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
                    bm_end = fmaf(sh_mp[warp][nb][k], sh_ext[k][lane], bm_end);
#ifdef STRATO
                bm_end = fmaf(sh_mp[warp][nb][MP_ST], sh_ext[NCOMP][lane], bm_end);
#endif
                sp_acc = fmaf(sh_mp[warp][nb][MP_DT], fmaf(4.f, bm_end, sp_b + seg_be), sp_acc);
                kah(tau_lane, tc_lane, sp_acc);
            }
        }
    }
    if constexpr (K_BOUNDARY) {
        if (lane < TW) {
            const int j = end_j[ray];
            const float f = end_f[ray];
            float e = fmaf(f, (float)boundary[(j + 1) * FW + lane], (1.f - f) * (float)boundary[j * FW + lane]);
            float od = 0.f;
#pragma unroll
            for (int k = 0; k < NCOMP; ++k)
                od = fmaf(end_col[NCOMP * ray + k], sh_ext[k][lane], od);
#ifdef STRATO
            od = fmaf(st_end_col[ray], sh_ext[NCOMP][lane], od);
#endif
            e *= expf(-od) * (float)hits[ray];
#pragma unroll
            for (int k = 0; k < 4; ++k)
                lane_rad[k] = fmaf(0.5f * e, pu[k], lane_rad[k]);
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
// original ray order, FP32 batch partials, and compensated accumulation across batches and parts.
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
                  const float *targets, const float *dirs, const float *weights,
                  const uint32_t *edge_off, const float *yb, const float *ray_rads, float albedo,
                  float *moments, float *flux) {
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
    const float *target = targets + 9 * cell;
    const int64_t cb = cell_off[cell], ce = cell_off[cell + 1];
    ProjectionSum total[SP_WAVES] = {};
    for (int part = 0; part < SP_PARTS; part++) {
        const int64_t begin = cb + (ce - cb) * part / SP_PARTS,
                      end = cb + (ce - cb) * (part + 1) / SP_PARTS;
        ProjectionSum acc[SP_WAVES] = {};
        for (int64_t rb = begin; rb < end; rb += SP_RAYS) {
            const int nr = (int)min((int64_t)SP_RAYS, end - rb);
            if (tid < nr) {
                const float *d = dirs + 3 * (rb + tid);
                float st = hypotf(d[0], d[1]);
                float cp = st > 1e-12f ? (float)(d[0] / st) : 1.f;
                cp = clip(cp, -1, 1);
                cd[tid][0] = 1;
                cd[tid][1] = cp;
                for (int mm = 2; mm < 33; mm++)
                    cd[tid][mm] = (2.f * cp) * cd[tid][mm - 1] - cd[tid][mm - 2];
                float dlz = 0;
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
                acc[q].add(sum[q]);
            __syncthreads();
        }
#pragma unroll
        for (int q = 0; q < SP_WAVES; q++) {
            total[q].add(acc[q].sum);
            total[q].add(acc[q].error);
        }
    }
#pragma unroll
    for (int q = 0; q < SP_WAVES; q++)
        if (wbase + q < TW) {
            if (isM)
                moments[(int64_t)cell * nm + moment * TW + wbase + q] = total[q].value();
            else if (isF)
                flux[cell * FW + GW(wbase + q)] = total[q].value() * (albedo / PI);
        }
}
#endif
#endif // !LUT_ONLY
