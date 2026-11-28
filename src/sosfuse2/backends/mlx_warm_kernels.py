"""Metal source for the normal-domain warm transport and cell projection."""

HEADER = r"""
#pragma clang fp contract(off)

inline float altitude(float t, float pd, float h0, float top) {
    float r0 = 6371.f + h0;
    float delta = t * fma(2.f, pd, t);
    float r = sqrt(max(0.f, fma(r0, r0, delta)));
    return clamp(h0 + delta / (r + r0), 0.f, top);
}

template<typename P> inline int bracket(P a, int n, float x) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int m = (lo + hi) >> 1;
        if (a[m] <= x) lo = m + 1;
        else hi = m;
    }
    return clamp(lo - 1, 0, n - 2);
}

struct Density {
    float4 p;
    float o4;
};
template<bool extended>
inline float path_altitude(float t, float pd, float offset, float shell, float top) {
    if (!extended) return altitude(t, pd, offset, top);
    float r0 = (6371.f + shell) + offset;
    float delta = t * fma(2.f, pd, t);
    float r = sqrt(max(0.f, fma(r0, r0, delta)));
    float relative_h = offset + delta / (r + r0);
    return clamp(shell + relative_h, 0.f, top);
}
template<typename P>
inline float endpoint_support_height(float h, float inside_h, P phys) {
    if (h >= phys[11] && h <= phys[12]) h = inside_h > phys[9] ? phys[12] : phys[9];
    if (h >= phys[13] && h <= phys[14]) h = inside_h > phys[10] ? phys[14] : phys[10];
    return h;
}
template<int mask, bool layer, int nt, bool extended, typename T, typename P, typename A>
inline Density density(float h, T tab, P phys, A aer) {
    bool out_air = extended && (h > phys[10]);
    bool out_low = extended && (h > phys[9]);
    if (out_air && out_low) return {float4(0.f), 0.f};
    float x = h * phys[1];
    int j = min(int(x), nt - 2);
    float f = x - float(j);
    auto a = tab + 3*j;
    float air = out_air ? 0.f : (mask & 1 ? fma(f, a[3]-a[0], a[0]) : fast::exp(-0.125f*h));
    if (out_low) return {float4(air, 0.f, 0.f, 0.f), 0.f};
    float o3 = mask & 2 ? fma(f, a[4]-a[1], a[1]) : max(0.f, 1.f - abs(h-25.f)/15.f);
    float no2 = fma(f, a[5]-a[2], a[2]);
    float aerosol;
    if (layer) {
        float z = clamp((h - aer[0]) * aer[1], 0.f, 1.f);
        float blend = z*z*(3.f - 2.f*z);
        aerosol = (aer[2]*exp(-h*aer[3]))*(1.f-blend)
                 + (aer[4]*exp(-(max(h,aer[0])-aer[6])*aer[5]))*blend;
    } else {
        aerosol = fast::exp(-h*phys[2]);
    }
    float norm = air*phys[3];
    return {float4(air, aerosol, o3, no2), norm*norm};
}
template<typename P> inline float3 load3(P v) { return float3(v[0],v[1],v[2]); }
template<typename P> inline float4 load4(P v) { return float4(v[0],v[1],v[2],v[3]); }
inline void store4(device float *v, float4 x) {
    v[0]=x.x; v[1]=x.y; v[2]=x.z; v[3]=x.w;
}
inline float4 rotation(float3 a, float3 b) {
    return float4(a*b, a.x*b.z + a.z*b.x);
}
inline float ftz(float x) { return NATIVE_FTZ ? x : (abs(x) < 0x1p-126f ? 0.f : x); }
inline float2 ftz(float2 x) { return NATIVE_FTZ ? x : select(x, float2(0.f), abs(x) < 0x1p-126f); }
inline float4 ftz(float4 x) { return NATIVE_FTZ ? x : select(x, float4(0.f), abs(x) < 0x1p-126f); }

// Source rotations and SH sums for corners where a sign controls log interpolation.
inline float scalar_dot3(float3 a, float3 b) {
    return (a.x * b.x + a.y * b.y) + a.z * b.z;
}
inline float3 reference_row(float3 b, float dl, float3 d, float3 q0, float3 q1, float3 q2) {
    float3 frame(scalar_dot3(b,q0), scalar_dot3(b,q1), scalar_dot3(b,q2));
    float3 ray(scalar_dot3(d,q0), scalar_dot3(d,q1), scalar_dot3(d,q2));
    return frame - dl * ray;
}
template<int lmax, int nw, typename S, typename Y>
inline float reference_sh(S src, Y y, float h, float cp) {
    float cm0 = 1.f, cm1 = cp;
    float part[32] = {};
    for (int m = 0; m <= lmax; ++m) {
        float accum = 0.f;
        for (int l = m; l <= lmax; ++l) {
            int k = l * (l + 1) / 2 + m;
            auto t = y + 4*k;
            float b = fma(h,fma(h,fma(h,t[3],t[2]),t[1]),t[0]);
            accum += b * src[k * nw];
        }
        float v = accum * (m ? cm1 : 1.f);
        if (m == 32) part[0] += v;
        else part[m] = v;
        if (m) {
            float next = (2.f * cp) * cm1 - cm0;
            cm0 = cm1;
            cm1 = next;
        }
    }
    for (int step = 16; step; step >>= 1)
        for (int j = 0; j < step; ++j) part[j] += part[j + step];
    return part[0];
}

inline float4 cubic_weights(float u) {
    float um = u - 1.f, up = u + 1.f, v = 2.f - u;
    return float4(u * um * v / 6.f, -up * um * v * 0.5f,
                  up * u * v * 0.5f, up * u * um / 6.f);
}
inline float solar_coordinate(float h, float z) {
    float td = sqrt(max(0.f, h * (12742.f + h)));
    return sqrt(clamp((z + td) / ((6371.f + h) + td), 0.f, 1.f));
}
inline float2 azimuth(float2 xy, float rho) {
    if (rho <= 1e-12f) return float2(1.f, 0.f);
    bool major_x = abs(xy.x) >= abs(xy.y);
    float major = major_x ? xy.x : xy.y, minor = major_x ? xy.y : xy.x;
    float z = minor / major, z2 = z * z, rt = sqrt(1.f + z2);
    float2 v(copysign(1.f - z2 / (rt * (rt + 1.f)), major), copysign(abs(z) / rt, minor));
    return major_x ? v : v.yx;
}
inline float hg_phase(float u, float g, float g2, float num) {
    float den = max(1e-20f, (1.f + g2) - (2.f * g) * u);
    return num / (den * sqrt(den));
}
template<bool extended, typename P>
inline float strato_density(float h, P phys) {
    if (extended && h > phys[9]) return 0.f;
    float t = (h - phys[6]) * phys[7];
    return ftz(fast::exp(-0.5f * (t * t)) * phys[8]);
}
constant constexpr int STRATO_LUT_BINS = 16384;
template<typename P>
inline float strato_mie_phase(float u, int w, P chi, int nw) {
    float p0 = 1.f, p1 = u;
    float v = p0 * chi[w] + p1 * chi[nw + w];
    for (int l = 2; l <= 32; ++l) {
        float pn = ((float(2 * l - 1) * u) * p1 - float(l - 1) * p0) / float(l);
        v += pn * chi[l * nw + w];
        p0 = p1;
        p1 = pn;
    }
    return v / 12.566370614359172f;
}
template<typename P>
inline float strato_mie_lut(float u, int w, P lut, int nw) {
    float x = (clamp(u, -1.f, 1.f) + 1.f) * float(STRATO_LUT_BINS / 2);
    int i = min(int(x), STRATO_LUT_BINS - 1);
    float f = x - float(i);
    float a = lut[i * nw + w];
    float b = lut[(i + 1) * nw + w];
    return fma(f, b - a, a);
}
constant constexpr int HG_BINS = 32;
constant constexpr int HG_NODES = HG_BINS + 5;
inline void hg_stencil(float angle, thread int &knot, thread float w[6]) {
    knot = clamp(int(angle), 0, HG_BINS - 1);
    float t = clamp(angle - float(knot), 0.f, 1.f);
    const float inv[6] = {-1.f / 120.f, 1.f / 24.f, -1.f / 12.f,
                          1.f / 12.f, -1.f / 24.f, 1.f / 120.f};
    for (int j = 0; j < 6; ++j) {
        float acc = inv[j];
        for (int k = 0; k < 6; ++k) if (k != j) acc *= t + float(2 - k);
        w[j] = acc;
    }
}

template<typename R, typename H, typename C>
inline void node_frame(device float *pt, device int *c0cp_out, float h, float t, float3 origin, R ray, H hg, int nh, C cg, int nc, int c0_flags) {
    float3 d = load3(ray+3);
    float3 q = fma(float3(t), d, origin);
    float rho = length(q.xy);
    float2 axis = rho > 1e-12f ? q.xy / rho : float2(1.f, 0.f);
    float angle = fast::atan2(rho, q.z);
    int i = bracket(hg, nh, h), j = bracket(cg, nc, angle);
    float ah = clamp((h-hg[i])/(hg[i+1]-hg[i]), 0.f, 1.f);
    float ac = clamp((angle-cg[j])/(cg[j+1]-cg[j]), 0.f, 1.f);
    // Inside the grid the interpolated angle reconstructs atan2(rho,q.z).
    // Recover its sine and cosine directly from the normalized position.
    float inv_r=rsqrt(dot(q,q));
    float cs=q.z*inv_r, sn=rho*inv_r;
    if (angle<cg[0] || angle>cg[nc-1])
        sn=sincos(clamp(angle,cg[0],cg[nc-1]),cs);
    float cp = ray[9] > 1e-12f ? clamp(dot(d.xy,axis)/ray[9], -1.f, 1.f) : 1.f;
    pt[1] = ah; pt[2] = ac;
    pt[8] = axis.x; pt[9] = axis.y; pt[10] = cs; pt[11] = sn;
    c0cp_out[0] = (i*nc+j) | c0_flags;
    c0cp_out[1] = as_type<int>(cp);
}
"""

# Each subgroup scans Simpson panels for one retained path interval.
PARALLEL_GEOMETRY = r"""
uint tid=thread_position_in_grid.x;
int seg=tid/P, local=tid%P;
int r=0;
if (local==0) {
    int left=0, right=ro_shape[0]-1;
    while (left+1<right) {
        int mid=(left+right)>>1;
        if (ro[mid]<=seg) left=mid; else right=mid;
    }
    r=left;
    owners[seg]=r;
}
r=simd_shuffle(r, ushort((thread_index_in_simdgroup/P)*P));
auto ray=rays+29*r;
float base=EXTENDED ? 0.f : intervals[2*seg], span=intervals[2*seg+1], width=abs(span);
bool lower=span>0.f;
float start=lower ? base : base+span;
float3 origin=EXTENDED ? load3(seg_ext+11*seg) : load3(ray);
float pd=EXTENDED ? seg_ext[11*seg+4] : ray[10];
float h0=EXTENDED ? seg_ext[11*seg+3] : ray[11];
float shell=EXTENDED ? seg_ext[11*seg+7] : 0.f;
float top=phys[0];
// For long quadratures, carry the last inclusive sum to the next SIMD scan.
float4 carry(0.f);
float carry4=0.f;
float carry_st=0.f;
for (int first=0; first<=NQ; first+=P) {
    int k=first+local;
    float u=k<NQ ? quad[2*k] : 1.f;
    float old_u=k==0 ? 0.f : k<=NQ ? quad[2*(k-1)] : 1.f;
    float v=lower ? u : 1.f-u;
    float f=lower ? u*u : u*(2.f-u);
    float old_f=lower ? old_u*old_u : old_u*(2.f-old_u);
    float t=fma(span,v*v,base);
    float h=(EXTENDED && k==NQ)
        ? endpoint_support_height(seg_ext[11*seg+10],seg_ext[11*seg+9],phys)
        : path_altitude<EXTENDED>(t,pd,h0,shell,top);
    Density now=density<MASK,LAYER,NT,EXTENDED>(h,tab,phys,aer);
    float now_st=STRATO ? strato_density<EXTENDED>(h,phys) : 0.f;
    float tm=fma(width, 0.5f*(old_f+f), start);
    float hm=path_altitude<EXTENDED>(tm,pd,h0,shell,top);
    Density mid=density<MASK,LAYER,NT,EXTENDED>(hm,tab,phys,aer);
    float mid_st=STRATO ? strato_density<EXTENDED>(hm,phys) : 0.f;
    float panel=(f-old_f)*width/6.f;
    Density old={simd_shuffle_up(now.p,1),simd_shuffle_up(now.o4,1)};
    float old_st=STRATO ? simd_shuffle_up(now_st,1) : 0.f;
    if (local==0) {
        float last=lower ? old_u : 1.f-old_u;
        float pt=lower || k>0 ? fma(span,last*last,base) : start;
        float ho=(EXTENDED && k==0)
            ? endpoint_support_height(seg_ext[11*seg+8],seg_ext[11*seg+9],phys)
            : path_altitude<EXTENDED>(pt,pd,h0,shell,top);
        old=density<MASK,LAYER,NT,EXTENDED>(ho,tab,phys,aer);
        if (STRATO) old_st=strato_density<EXTENDED>(ho,phys);
    }
    float4 acc=panel*((old.p+now.p)+4.f*mid.p);
    float acc4=panel*((old.o4+now.o4)+4.f*mid.o4);
    float acc_st=STRATO ? panel*((old_st+now_st)+4.f*mid_st) : 0.f;
    for (ushort stride=1;stride<P;stride*=2) {
        float4 v=simd_shuffle_up(acc,stride);
        float z=simd_shuffle_up(acc4,stride);
        float zs=STRATO ? simd_shuffle_up(acc_st,stride) : 0.f;
        if (local>=stride) { acc+=v; acc4+=z; if (STRATO) acc_st+=zs; }
    }
    acc+=carry; acc4+=carry4;
    if (STRATO) acc_st+=carry_st;
    if (k<NQ) {
        int n=seg*NQ+k;
        device float *pt=points+12*n;
        pt[0]=2.f*(width*v)*quad[2*k+1];
        pt[3]=acc4;
        store4(pt+4,acc);
        if (STRATO) st_acc[n]=acc_st;
        int c0_flags=0;
        if (EXTENDED) {
            if (h>phys[10]) c0_flags|=0x20000000;
            if (h>phys[9]) c0_flags|=0x40000000;
        }
        node_frame(pt,c0cp+2*n,h,t,origin,ray,hg,NH,cg,NC,c0_flags);
    } else if (k==NQ) {
        constexpr int NSEG = STRATO ? 6 : 5;
        store4(segment_sum+NSEG*seg,acc);
        segment_sum[NSEG*seg+4]=acc4;
        if (STRATO) segment_sum[NSEG*seg+5]=acc_st;
    }
    if (NQ>=P) {
        ushort last=ushort((thread_index_in_simdgroup/P)*P+P-1);
        carry=simd_shuffle(acc,last);
        carry4=simd_shuffle(acc4,last);
        if (STRATO) carry_st=simd_shuffle(acc_st,last);
    }
}
"""

# Reuse the associated Legendre basis for scattering and projection.
BASIS = r"""
constexpr int K=(L+1)*(L+2)/2;
int m=thread_position_in_grid.x, r=thread_position_in_grid.y;
auto ray=compact+9*r;
if (NEED_EXP && m==0) {
    int left=0, right=co_shape[0]-1;
    while (left+1<right) {
        int mid=(left+right)/2;
        if (co[mid]<=r) left=mid; else right=mid;
    }
    auto f=frames+14*left;
    auto out=expanded+29*r;
    for (int j=0;j<3;++j) { out[j]=f[j]; out[3+j]=ray[j]; out[6+j]=ray[3+j]; }
    out[9]=ray[6]; out[10]=ray[7]; out[11]=f[12]; out[21]=ray[8];
    for (int i=0;i<3;++i)
        for (int j=0;j<3;++j) out[12+3*i+j]=fma(-ray[3+i],ray[j],f[3+3*i+j]);
    if (bd_flag[0]) for (int j=0;j<7;++j) out[22+j]=ends[7*r+j];
}
float mu=ray[2], st=ray[6];
float cp=st>1e-12f ? clamp(ray[0]/st,-1.f,1.f) : 1.f;
float cm=1.f, prev=cp;
if (m>0) {
    cm=cp; prev=1.f;
    for (int i=2; i<=m; ++i) {
        float next=2.f*cp*cm-prev; prev=cm; cm=next;
    }
}
if (!PREMUL) cosine[r*(L+1)+m]=cm;
float z0=0.28209479177387814f;
for (int i=1; i<=m; ++i) z0=(z0*norms[2*K+i])*st;
float factor=m ? 1.4142135623730951f : 1.f;
float z1=0.f;
for (int l=m; l<=L; ++l) {
    int k=l*(l+1)/2+m;
    float z;
    if (l==m) z=z0;
    else if (l==m+1) { z1=(sqrt(float(2*m+3))*mu)*z0; z=z1; }
    else { z=(norms[k]*mu)*z1-norms[K+k]*z0; z0=z1; z1=z; }
    float yv=factor*z;
    basis[r*K+k]=PREMUL ? (yv*cm) : yv;
}
"""

# Per-segment lower bound on optical depth across all wavelengths for pruning opaque segments.
SEG_PREFIX = r"""
int r = int(thread_position_in_grid.x);
int nr = ro_shape[0] - 1;
if (r >= nr) return;
constexpr int NSEG = STRATO ? 6 : 5;
float4 em03 = load4(ext_min);
float em4 = ext_min[4];
float em5 = STRATO ? ext_min[5] : 0.f;
int s0 = ro[r], s1 = ro[r + 1];
float prefix = 0.f;
for (int seg = s0; seg < s1; ++seg) {
    seg_depth[seg] = prefix;
    auto a = segment_sum + NSEG * seg;
    prefix += dot(em03, load4(a)) + em4 * a[4] + (STRATO ? em5 * a[5] : 0.f);
}
"""

# Count live corner records per active source cell in each sample block.
VP_COUNT = r"""
threadgroup atomic_int hist[NLCELL];
uint lid = thread_position_in_threadgroup.x;
uint b = threadgroup_position_in_grid.x;
for (int c = int(lid); c < NLCELL; c += 256)
    atomic_store_explicit(&hist[c], 0, memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);
int ns = meta[0], seg_n = meta[1];
int n0 = int(b) * seg_n;
int n1 = min(ns, n0 + seg_n);
for (int n = n0 + int(lid); n < n1; n += 256) {
    if (seg_depth[n / NQ] >= 92.f) continue;
    int raw = c0cp[2 * n];
    if (EXTENDED && (raw & 0x40000000)) continue;
    int c0 = EXTENDED ? (raw & 0x1FFFFFFF) : raw;
    int ir = c0 / NC, jc = c0 - ir * NC;
    int li0 = live_map[ir];
    if (li0 >= 0) {
        int lc = li0 * NC + jc;
        atomic_fetch_add_explicit(&hist[lc], 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&hist[lc + 1], 1, memory_order_relaxed);
    }
    int li1 = live_map[ir + 1];
    if (li1 >= 0) {
        int lc = li1 * NC + jc;
        atomic_fetch_add_explicit(&hist[lc], 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&hist[lc + 1], 1, memory_order_relaxed);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int c = int(lid); c < NLCELL; c += 256)
    counts[c * NB + int(b)] = atomic_load_explicit(&hist[c], memory_order_relaxed);
"""

# Scatter live corner records (ray, cp_bits) grouped by active source cell in monotonic order.
VP_SCATTER = r"""
threadgroup int run[NLCELL];
uint lid = thread_position_in_threadgroup.x;
uint b = threadgroup_position_in_grid.x;
for (int c = int(lid); c < NLCELL; c += 32)
    run[c] = offs[c * NB + int(b)];
simdgroup_barrier(mem_flags::mem_threadgroup);
int ns = meta[0], seg_n = meta[1];
int n0 = int(b) * seg_n;
int n1 = min(ns, n0 + seg_n);
auto out_rcp = (device uint2 *)sorted_rcp;
auto inv4 = (device int4 *)inv_slot;
for (int n_base = n0; n_base < n1; n_base += 32) {
    int n = n_base + int(lid);
    bool in_range = n < n1;
    int seg = in_range ? (n / NQ) : 0;
    bool live_seg = in_range && (seg_depth[seg] < 92.f);
    int raw = live_seg ? c0cp[2 * n] : -1;
    bool active_node = live_seg && (!EXTENDED || !(raw & 0x40000000));
    int c0 = active_node ? (EXTENDED ? (raw & 0x1FFFFFFF) : raw) : -1;
    uint cp_bits = active_node ? uint(c0cp[2 * n + 1]) : 0u;
    uint r_id = active_node ? uint(owners[seg]) : 0u;
    int ir = active_node ? (c0 / NC) : 0;
    int jc = active_node ? (c0 - ir * NC) : 0;
    int li0 = active_node ? live_map[ir] : -1;
    int li1 = active_node ? live_map[ir + 1] : -1;
    bool any_live = (li0 >= 0) || (li1 >= 0);
    uint2 rcp = uint2(r_id, cp_bits);
    int4 iv = int4(-1);
    uint active = uint(simd_vote::vote_t(simd_ballot(any_live)));
    while (active) {
        int leader = ctz(active);
        int tc = simd_shuffle(c0, ushort(leader));
        bool match = any_live && (c0 == tc);
        uint m_mask = uint(simd_vote::vote_t(simd_ballot(match)));
        int rank = popcount(m_mask & ((1u << lid) - 1u));
        int cnt = popcount(m_mask);
        int t_li0 = simd_shuffle(li0, ushort(leader));
        int t_li1 = simd_shuffle(li1, ushort(leader));
        int t_jc = simd_shuffle(jc, ushort(leader));
        if (t_li0 >= 0) {
            int lc = t_li0 * NC + t_jc;
            int p0 = run[lc];
            int p1 = run[lc + 1];
            if (match) {
                out_rcp[p0 + rank] = rcp;
                out_rcp[p1 + rank] = rcp;
                iv.x = p0 + rank;
                iv.y = p1 + rank;
            }
            if (int(lid) == leader) {
                run[lc] = p0 + cnt;
                run[lc + 1] = p1 + cnt;
            }
        }
        if (t_li1 >= 0) {
            int lc = t_li1 * NC + t_jc;
            int p2 = run[lc];
            int p3 = run[lc + 1];
            if (match) {
                out_rcp[p2 + rank] = rcp;
                out_rcp[p3 + rank] = rcp;
                iv.z = p2 + rank;
                iv.w = p3 + rank;
            }
            if (int(lid) == leader) {
                run[lc] = p2 + cnt;
                run[lc + 1] = p3 + cnt;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        active &= ~m_mask;
    }
    if (any_live) inv4[n] = iv;
}
"""

# Fused block partitioner: computes bstart and blk directly from offs and inc in one threadgroup.
VP_BLOCKS = r"""
threadgroup int sg_sum[8];
threadgroup int tg_tot;
uint lid = thread_position_in_threadgroup.x;
int sg = int(lid >> 5), lane = int(thread_index_in_simdgroup);
int total_recs = inc[NLCELL * NB - 1];
int blk_carry = 0;
for (int base = 0; base < NLCELL; base += 256) {
    int lc = base + int(lid);
    int s0 = lc < NLCELL ? offs[lc * NB] : total_recs;
    int s1 = (lc + 1 < NLCELL) ? offs[(lc + 1) * NB] : total_recs;
    int nblk = lc < NLCELL ? (s1 - s0 + (BLK - 1)) / BLK : 0;
    int excl = simd_prefix_exclusive_sum(nblk);
    if (lane == 31) sg_sum[sg] = excl + nblk;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        int sv = lane < 8 ? sg_sum[lane] : 0;
        int sp = simd_prefix_exclusive_sum(sv);
        if (lane < 8) sg_sum[lane] = sp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int b0 = blk_carry + sg_sum[sg] + excl;
    if (sg == 7 && lane == 31) tg_tot = sg_sum[7] + excl + nblk;
    if (lc < NLCELL) {
        bstart[lc] = b0;
        int li = lc / NC, jc = lc - li * NC;
        int c = live_ir[li] * NC + jc;
        for (int t = 0; t < nblk; ++t) {
            int a = s0 + t * BLK;
            int e = min(s1, a + BLK);
            blk[3 * (b0 + t)] = c;
            blk[3 * (b0 + t) + 1] = a;
            blk[3 * (b0 + t) + 2] = e;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    blk_carry += tg_tot;
}
if (lid == 0) bstart[NLCELL] = blk_carry;
"""

# General VPROD for L > 16 or W > 16: preserves exact ascending l summation order.
VP_VPROD = r"""
constexpr int K = (L + 1) * (L + 2) / 2;
constexpr int NW = 128 / WCHUNK;
threadgroup float plane[K * WCHUNK];

uint lid = thread_position_in_threadgroup.x;
uint tgid = threadgroup_position_in_grid.x;
if (int(tgid) >= bstart[NLCELL]) return;

int c = blk[3 * int(tgid)];
int s0 = blk[3 * int(tgid) + 1];
int s1 = blk[3 * int(tgid) + 2];
if (s0 >= s1) return;

float ha_c = densities[2 * (c / NC) + 1];
int worker = int(lid) / WCHUNK;
int lw = int(lid) % WCHUNK;
int total = s1 - s0;
int p0 = s0 + (total * worker) / NW;
int p1 = s0 + (total * (worker + 1)) / NW;

auto cell_sh = src_sh + size_t(c) * K * W;
for (int w0 = 0; w0 < W; w0 += WCHUNK) {
    int w = w0 + lw;
    for (int idx = int(lid); idx < K * WCHUNK; idx += 128) {
        int k = idx / WCHUNK;
        int wl = idx % WCHUNK;
        int gw = w0 + wl;
        plane[idx] = gw < W ? cell_sh[k * W + gw] : 0.f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float modes[L + 1];
    int prev_ray = -1;
    auto in_rcp = (device const uint2 *)sorted_rcp;
    for (int idx = p0; idx < p1; ++idx) {
        uint2 rcp = in_rcp[idx];
        int r = int(rcp.x);
        if (r != prev_ray) {
            auto yb = basis + r * K;
#pragma unroll
            for (int m = 0; m <= L; ++m) {
                float acc = 0.f;
                int k = (m * (m + 1)) / 2 + m;
                for (int l = m; l <= L; ++l) {
                    if (l > m) k += l;
                    acc = fma(yb[k], plane[k * WCHUNK + lw], acc);
                }
                modes[m] = acc;
            }
            prev_ray = r;
        }
        if (w < WP) {
            if (w < W) {
                float cp = as_type<float>(rcp.y);
                float val = modes[0];
                float prev = 1.f, cm = cp;
#pragma unroll
                for (int m = 1; m <= L; ++m) {
                    val = fma(cm, modes[m], val);
                    float next = 2.f * cp * cm - prev;
                    prev = cm; cm = next;
                }
                aero[size_t(idx) * WP + w] = ftz(val * ha_c);
            } else {
                aero[size_t(idx) * WP + w] = 0.f;
            }
        }
    }
    if (w0 + WCHUNK < W)
        threadgroup_barrier(mem_flags::mem_threadgroup);
}
"""

# Vectorized float4 VPROD (64-record window, (m,l)-ordered plane4 pre-scaled by sqrt(2) for m>0,
# float2 norms_ml, configurable NTG threads/block for 3x TG/core occupancy).
VP_VPROD_F4_W64 = r"""
constexpr int K = (L + 1) * (L + 2) / 2;
constexpr int NSG = NTG / 32;
threadgroup float4 plane4[K * 4];

uint lid = thread_position_in_threadgroup.x;
uint tgid = threadgroup_position_in_grid.x;
if (int(tgid) >= bstart[NLCELL]) return;

int c = blk[3 * int(tgid)];
int s0 = blk[3 * int(tgid) + 1];
int s1 = blk[3 * int(tgid) + 2];
if (s0 >= s1) return;

auto cell_sh4 = (device const float4 *)(src_sh + size_t(c) * K * 16);
for (int i = int(lid); i < K * 4; i += NTG)
    plane4[i] = cell_sh4[i];
threadgroup_barrier(mem_flags::mem_threadgroup);

float ha_c = densities[2 * (c / NC) + 1];
int sg = int(lid >> 5);
int lane = int(thread_index_in_simdgroup);
int v = lane >> 2;
int q = lane & 3;
int total = s1 - s0;
int p0 = s0 + (total * sg) / NSG;
int p1 = s0 + (total * (sg + 1)) / NSG;

auto in_rcp = (device const uint2 *)sorted_rcp;
auto out_a4 = (device float4 *)aero;
int cur = p0;
while (cur < p1) {
    int win = min(64, p1 - cur);
    int i0 = cur + lane;
    int i1 = cur + 32 + lane;
    int r0 = i0 < p1 ? int(in_rcp[i0].x) : -1;
    int r1 = i1 < p1 ? int(in_rcp[i1].x) : -1;
    int prev0 = simd_shuffle_up(r0, 1);
    int last0 = simd_shuffle(r0, ushort(31));
    int prev1_up = simd_shuffle_up(r1, 1);
    int prev1 = lane == 0 ? last0 : prev1_up;
    bool new0 = (i0 < p1) && (lane == 0 || r0 != prev0);
    bool new1 = (i1 < p1) && (r1 != prev1);
    uint mask0 = uint(simd_vote::vote_t(simd_ballot(new0)));
    uint mask1 = uint(simd_vote::vote_t(simd_ballot(new1)));
    ulong mask64 = ulong(mask0) | (ulong(mask1) << 32);
    int n_vis = popcount(mask0) + popcount(mask1);

    int my_start = win, my_end = win;
    ulong m_scan = mask64;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        if (m_scan) {
            int pos = int(ctz(m_scan));
            m_scan &= m_scan - 1ull;
            int next_pos = m_scan ? int(ctz(m_scan)) : win;
            if (v == t) {
                my_start = pos;
                my_end = next_pos;
            }
        }
    }
    int advance = (n_vis > 8) ? int(ctz(m_scan)) : win;
    bool has_v = (v < min(n_vis, 8));
    int r_lo = simd_shuffle(r0, ushort(my_start & 31));
    int r_hi = simd_shuffle(r1, ushort(my_start & 31));
    int cand_r = my_start < 32 ? r_lo : r_hi;

    if (has_v) {
        float mu = compact[9 * cand_r + 2];
        float st = compact[9 * cand_r + 6];
        float4 modes[L + 1];
        float diag = 0.28209479177387814f;
        int idx = 0;
#pragma unroll
        for (int m = 0; m <= L; ++m) {
            if (m > 0) diag = (diag * NORMS_ML[idx].x) * st;
            float z0 = diag;
            float4 acc = z0 * plane4[idx * 4 + q];
            ++idx;
            if (m < L) {
                float4 p1_tg = plane4[idx * 4 + q];
                float z1 = (NORMS_ML[idx].x * mu) * z0;
                acc = fma(float4(z1), p1_tg, acc);
                ++idx;
                for (int l = m + 2; l <= L; ++l) {
                    float4 p_tg = plane4[idx * 4 + q];
                    float2 nm = NORMS_ML[idx];
                    float z2 = (nm.x * mu) * z1 - nm.y * z0;
                    acc = fma(float4(z2), p_tg, acc);
                    z0 = z1; z1 = z2;
                    ++idx;
                }
            }
            modes[m] = acc;
        }
        int off = my_start;
        for (; off + 1 < my_end; off += 2) {
            uint g0 = uint(cur + off);
            uint g1 = g0 + 1u;
            float cp0 = as_type<float>(in_rcp[g0].y);
            float cp1 = as_type<float>(in_rcp[g1].y);
            float cp2_0 = 2.f * cp0;
            float cp2_1 = 2.f * cp1;
            float4 val0 = modes[0];
            float4 val1 = modes[0];
            float prev0 = 1.f, cm0 = cp0;
            float prev1 = 1.f, cm1 = cp1;
#pragma unroll
            for (int m = 1; m <= L; ++m) {
                float4 mm = modes[m];
                val0 = fma(float4(cm0), mm, val0);
                val1 = fma(float4(cm1), mm, val1);
                float next0 = cp2_0 * cm0 - prev0;
                float next1 = cp2_1 * cm1 - prev1;
                prev0 = cm0; cm0 = next0;
                prev1 = cm1; cm1 = next1;
            }
            out_a4[g0 * 4u + uint(q)] = ftz(val0 * ha_c);
            out_a4[g1 * 4u + uint(q)] = ftz(val1 * ha_c);
        }
        if (off < my_end) {
            uint gidx = uint(cur + off);
            float cp = as_type<float>(in_rcp[gidx].y);
            float cp2 = 2.f * cp;
            float4 val = modes[0];
            float prev = 1.f, cm = cp;
#pragma unroll
            for (int m = 1; m <= L; ++m) {
                val = fma(float4(cm), modes[m], val);
                float next = cp2 * cm - prev;
                prev = cm; cm = next;
            }
            out_a4[gidx * 4u + uint(q)] = ftz(val * ha_c);
        }
    }
    cur += advance;
}
"""

# Slim warm consumer: reads pre-contracted aerosol scattering and compact Rayleigh source.
INTEGRATE = r"""
int w=thread_position_in_grid.x, r=thread_position_in_grid.y;
if (w>=W) return;
int nr=offsets_shape[0]-1;
bool retry=false;
if (REFINE && !prior_flags[r*W+w]) {
    for (int j=0;j<5;++j) weighted[(j*nr+r)*W+w]=coarse[(j*nr+r)*W+w];
    return;
}
int cell=0;
if (REFINE) {
    int hi=cell_off_shape[0]-1;
    while (cell+1<hi) {
        int mid=(cell+hi)/2;
        if (cell_off[mid]<=r) cell=mid; else hi=mid;
    }
}
auto ray=rays+29*r;
float3 dl=load3(ray+6);
float3 b0=load3(ray+12), b1=load3(ray+15), b2=load3(ray+18);
float4 pu=float4(1.f-dl*dl, -dl.x*dl.z);
float trace_pu=(pu.x+pu.y)+pu.z;
float4 total(0.f), correction(0.f);
float mix=phys[4], iso=(1.f-mix)/3.f;
float4 ext=float4(extinction[w],extinction[W+w],extinction[2*W+w],extinction[3*W+w]);
float ext4=extinction[4*W+w];
float ext5=STRATO ? extinction[5*W+w] : 0.f;
float prefix=0.f, compensation=0.f;
int theta0 = 0, phi_prev = -1, angular_base = -1;
float4 theta_w(0.f), mer[4], h_a(0.f);
if (ANG) {
    float x = fast::atan2(ray[9], ray[5]) * (float(ANG) / 3.14159265358979323846f);
    theta0 = clamp(int(x), 1, max(1, ANG-2)) - 1;
    theta_w = cubic_weights(x - float(theta0+1));
}
int prev_base=-1;
float4 rm0(0.f), rm1(0.f), rm2(0.f), rm3(0.f), hr_cell(0.f), isotropic(0.f);
bool l0=false, l1=false;
auto pt4=(device const float4 *)points;

int seg0=offsets[r], seg1=offsets[r+1];
for (int seg=seg0; seg<seg1; ++seg) {
    float base_depth = prefix - compensation;
    if (NONNEG_EXT && base_depth >= 92.f) break;
    int n_base = seg * NQ;
    bool dead = false;
    for (int kq=0; kq<NQ; ++kq) {
        int n = n_base + kq;
        float4 p0 = pt4[3 * n];
        float4 p1 = pt4[3 * n + 1];
        float depth = base_depth + (dot(ext, p1) + ext4 * p0.w + (STRATO ? ext5 * st_acc[n] : 0.f));
        if (NONNEG_EXT && depth >= 92.f) { dead = true; break; }
        float att = ftz(p0.x * ftz(fast::exp(-depth)));
        if (att == 0.f) continue;

        int raw_c0 = c0cp[2 * n];
        bool out_air = EXTENDED && ((raw_c0 & 0x20000000) != 0);
        bool out_low = EXTENDED && ((raw_c0 & 0x40000000) != 0);
        if (out_air && out_low) continue;
        int base = EXTENDED ? (raw_c0 & 0x1FFFFFFF) : raw_c0;
        if (base != prev_base) {
            int ir = base / NC;
            constexpr int JR = 4 * W;
            auto fr = src_rayleigh + base * JR + w;
            rm0 = float4(fr[0], fr[JR], fr[NC * JR], fr[(NC + 1) * JR]);
            auto fr1 = fr + W;
            rm1 = float4(fr1[0], fr1[JR], fr1[NC * JR], fr1[(NC + 1) * JR]);
            auto fr2 = fr + 2 * W;
            rm2 = float4(fr2[0], fr2[JR], fr2[NC * JR], fr2[(NC + 1) * JR]);
            auto fr3 = fr + 3 * W;
            rm3 = float4(fr3[0], fr3[JR], fr3[NC * JR], fr3[(NC + 1) * JR]);
            hr_cell = float4(densities[2 * ir], densities[2 * ir], densities[2 * ir + 2], densities[2 * ir + 2]);
            isotropic = ftz(iso * ((rm0 + rm1) + rm2));
            l0 = live_map[ir] >= 0;
            l1 = live_map[ir + 1] >= 0;
            prev_base = base;
        }
        float4 hr = out_air ? float4(0.f) : hr_cell;
        bool nl0 = l0 && !out_low;
        bool nl1 = l1 && !out_low;

        float4 aer(0.f);
        float4 saer(0.f);
        if (ANG && (nl0 || nl1)) {
            float cp = as_type<float>(c0cp[2*n+1]);
            float x = fast::acos(clamp(cp,-1.f,1.f)) * (float(ANG) / 3.14159265358979323846f);
            int ip = clamp(int(x), 1, max(1, ANG-2)) - 1;
            if (base != angular_base || ip != phi_prev) {
                int ir = base / NC;
                h_a = float4(l0 ? densities[2*ir+1] : 0.f,l0 ? densities[2*ir+1] : 0.f,
                             l1 ? densities[2*ir+3] : 0.f,l1 ? densities[2*ir+3] : 0.f);
                for (int j = 0; j < 4; ++j) {
                    float4 acc(0.f);
                    for (int i = 0; i < 4; ++i) {
                        auto a = aero + (((theta0+i)*(ANG+1) + ip+j)*size_t(NS) + base)*W+w;
                        float4 corner = float4(a[0], a[W], a[NC*W], a[(NC+1)*W]);
                        acc = fma(float4(theta_w[i]),corner,acc);
                    }
                    mer[j] = acc;
                }
                angular_base=base; phi_prev=ip;
            }
            float4 v=cubic_weights(x-float(ip+1));
            aer=h_a*(v.x*mer[0]+v.y*mer[1]+v.z*mer[2]+v.w*mer[3]);
        } else if (!ANG) {
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        if (nl0 || nl1) {
            constexpr uint AERO_W = STRATO_SEP ? uint(2 * WP) : uint(WP);
            auto iv = inv_slot + 4*n;
            if (nl0) {
                uint i0 = uint(iv[0]) * AERO_W + uint(w);
                uint i1 = uint(iv[1]) * AERO_W + uint(w);
                a0 = aero[i0];
                a1 = aero[i1];
                if (STRATO_SEP) { s0 = aero[i0 + uint(WP)]; s1 = aero[i1 + uint(WP)]; }
            }
            if (nl1) {
                uint i2 = uint(iv[2]) * AERO_W + uint(w);
                uint i3 = uint(iv[3]) * AERO_W + uint(w);
                a2 = aero[i2];
                a3 = aero[i3];
                if (STRATO_SEP) { s2 = aero[i2 + uint(WP)]; s3 = aero[i3 + uint(WP)]; }
            }
        }
        aer = float4(a0, a1, a2, a3);
        if (STRATO_SEP) saer = float4(s0, s1, s2, s3);

        }
        float ah = p0.y, ac = p0.z;
        float4 weight = float4(1.f - ah, 1.f - ah, ah, ah) * float4(1.f - ac, ac, 1.f - ac, ac);

        float4 p2 = pt4[3 * n + 2];
        float2 axis = p2.xy;
        float cs = p2.z, sn = p2.w;
        float r0 = dot(b0.xy, axis), r1 = dot(b1.xy, axis), r2 = dot(b2.xy, axis);
        float3 k0 = float3(fma(-cs, r0, b0.z * sn), fma(b0.x, axis.y, -b0.y * axis.x), fma(sn, r0, b0.z * cs));
        float3 k1 = float3(fma(-cs, r1, b1.z * sn), fma(b1.x, axis.y, -b1.y * axis.x), fma(sn, r1, b1.z * cs));
        float3 k2 = float3(fma(-cs, r2, b2.z * sn), fma(b2.x, axis.y, -b2.y * axis.x), fma(sn, r2, b2.z * cs));

        float4 t0 = rotation(k0, k0), t1 = rotation(k1, k1), t2 = rotation(k2, k2), t3 = rotation(k0, k2);
        float4 trrot = (t0 + t1) + t2;
        float4 aer_tot = STRATO_SEP ? (aer + saer) : aer;
        float4 isot = hr * isotropic + 0.5f * aer_tot;
        float4 trace = hr * mix * (trrot.x * rm0 + trrot.y * rm1 + trrot.z * rm2 + trrot.w * rm3) + isot * trace_pu;
        float4 val;
        if (any((abs(trace) < 1e-32f) & ((abs(rm0) + abs(rm1) + abs(rm2) + abs(rm3) + abs(aer) + (STRATO_SEP ? abs(saer) : float4(0.f))) > 0.f))) {
            trace = float4(0.f);
            for (int k = 0; k < 4; ++k) {
                float4 t = k == 0 ? t0 : k == 1 ? t1 : k == 2 ? t2 : t3;
                float4 rot = ftz(t.x * rm0);
                rot = ftz(rot + ftz(t.y * rm1));
                rot = ftz(rot + ftz(t.z * rm2));
                rot = ftz(rot + ftz(t.w * rm3));
                float4 tensor = ftz(hr * ftz(ftz(mix * rot) + ftz(isotropic * pu[k])));
                tensor = ftz(tensor + ftz(0.5f * ftz(aer * pu[k])));
                if (STRATO_SEP) tensor = ftz(tensor + ftz(0.5f * ftz(saer * pu[k])));
                float4 linear = ftz(weight * tensor);
                val[k] = ((linear.x + linear.y) + linear.z) + linear.w;
                if (k < 3) trace += tensor;
            }
        } else {
            float4 ww = weight * hr;
            float4 mixed = mix * float4(dot(ww, rm0), dot(ww, rm1), dot(ww, rm2), dot(ww, rm3));
            val = float4(dot(t0, mixed), dot(t1, mixed), dot(t2, mixed), dot(t3, mixed)) + pu * dot(weight, isot);
        }
        float4 strength = hr * (abs(rm0) + abs(rm1) + abs(rm2) + 2.f * abs(rm3));
        float4 magnitude = strength + abs(aer) + (STRATO_SEP ? abs(saer) : float4(0.f));
        bool4 sensitive = (abs(trace) <= (ANG ? 0.001f : 0x1p-16f) * magnitude) & (magnitude > 0.f) & (weight > 0.f);
        bool4 omitted = out_low ? bool4(false) : (bool4(!l0, !l0, !l1, !l1) & (weight > 0.f));
        if (BOUNDS && any(omitted)) {
            // Each unsorted SH contribution is bounded by 2^-23 times
            // its local Rayleigh coefficient scale. Inspect cancellation.
            float total_strength = dot(weight, select(float4(0.f), strength, omitted));
            float signal = max(max(abs(val.x), abs(val.y)), max(abs(val.z), abs(val.w)));
            if (total_strength > 0.f && signal < 0.01f * total_strength) sensitive |= omitted;
        }
        if (any(sensitive)) {
            if (!REFINE) {
                retry = true;
            } else {
                auto f = frame + 14 * cell;
                float base_t = EXTENDED ? 0.f : intervals[2*seg], span_t = intervals[2*seg+1];
                float u = quad[2*kq], v = span_t>0.f ? u : 1.f-u;
                float t = fma(span_t, v*v, base_t);
                float3 d = load3(ray + 3);
                float3 pos = fma(float3(t),d,EXTENDED ? load3(seg_ext + 11 * seg) : load3(ray));
                float rho = length(pos.xy);
                if (rho>1e-12f) {
                    bool xmajor = abs(pos.x) >= abs(pos.y);
                    float major = xmajor ? pos.x : pos.y, minor = xmajor ? pos.y : pos.x;
                    float z = minor / major, z2 = z*z, rt = sqrt(1.f+z2);
                    float vmaj = copysign(1.f-z2/(rt*(rt+1.f)),major);
                    float vmin = copysign(abs(z)/rt,minor);
                    axis = xmajor ? float2(vmaj,vmin) : float2(vmin,vmaj);
                }
                float a = precise::atan2(rho,pos.z);
                float a0 = angles[base % NC], da = angles[base % NC+1]-a0;
                float ang = a0 + clamp((a-a0)/da,0.f,1.f)*da;
                float ss = precise::sin(ang), cc = precise::cos(ang);
                float3 q0(-cc * axis.x, -cc * axis.y, ss);
                float3 q1(axis.y, -axis.x, 0.f);
                float3 q2(ss * axis.x, ss * axis.y, cc);
                float3 x0 = reference_row(load3(f + 3), dl.x, d, q0, q1, q2);
                float3 x1 = reference_row(load3(f + 6), dl.y, d, q0, q1, q2);
                float3 x2 = reference_row(load3(f + 9), dl.z, d, q0, q1, q2);
                float4 rot[3] = {rotation(x0, x0), rotation(x1, x1), rotation(x2, x2)};
                float cp = ray[9]>1e-12f ? clamp((d.x*axis.x+d.y*axis.y)/ray[9],-1.f,1.f) : 1.f;
                constexpr int K = (L + 1) * (L + 2) / 2;
                for (int j = 0; j < 4; ++j) if (sensitive[j]) {
                    int corner = base + (j / 2) * NC + (j % 2);
                    float av = 0.f;
                    float sav = 0.f;
                    if (!out_low) {
                        auto sh = raw_sh + size_t(corner) * K * W + w;
                        float den = densities[2 * (corner / NC) + 1];
                        av = den == 0.f ? 0.f : ftz(den * reference_sh<L, W>(sh, basis_table + int(polar[2*r])*K*4, polar[2*r+1], cp));
                        if (STRATO_SEP) {
                            auto sh2 = raw_sh + size_t(NS + corner) * K * W + w;
                            int nh = NS / NC;
                            float sden = densities[2 * (nh + corner / NC) + 1];
                            sav = sden == 0.f ? 0.f : ftz(sden * reference_sh<L, W>(sh2, basis_table + int(polar[2*r])*K*4, polar[2*r+1], cp));
                        }
                    }
                    float tr = 0.f;
                    for (int k = 0; k < 3; ++k) {
                        float4 a = rot[k];
                        float rotval = ((a.x * rm0[j] + a.y * rm1[j]) + a.z * rm2[j]) + a.w * rm3[j];
                        float rayval = hr[j] * (mix * rotval + isotropic[j] * pu[k]);
                        float tv = rayval + (av * pu[k]) * 0.5f;
                        if (STRATO_SEP) tv += (sav * pu[k]) * 0.5f;
                        tr += tv;
                    }
                    if (ANG || omitted[j]) {
                        float diff = (av - aer[j]) + (STRATO_SEP ? (sav - saer[j]) : 0.f);
                        val += (0.5f * (diff * pu)) * weight[j];
                    }
                    trace[j] = tr;
                }
            }
        }
        bool4 valid = trace >= 0x1p-126f;
        float scale = 1.f;
        if (all(valid)) {
            float4 lg = weight * fast::log2(trace);
            float4 am4 = ftz(weight * trace);
            float am = ((am4.x + am4.y) + am4.z) + am4.w;
            float logs = ((lg.x + lg.y) + lg.z) + lg.w;
            scale = am >= 0x1p-126f ? fast::exp2(logs - fast::log2(am)) : 1.f;
        } else if (any(valid)) {
            float4 tr = select(float4(1.f), trace, valid);
            float4 wg4 = select(float4(0.f), weight, valid);
            float4 lg = wg4 * fast::log2(tr);
            float4 am4 = ftz(wg4 * tr);
            float wg = ((wg4.x + wg4.y) + wg4.z) + wg4.w;
            float am = ((am4.x + am4.y) + am4.z) + am4.w;
            float logs = ((lg.x + lg.y) + lg.z) + lg.w;
            scale = wg > 0.f && am >= 0x1p-126f ? fast::exp2(logs - wg * fast::log2(am / wg)) : 1.f;
        }
        float4 y = ftz(ftz(ftz(val * scale) * att) - correction);
        float4 updated = ftz(total + y);
        correction = ftz(ftz(updated - total) - y);
        total = updated;
    }
    if (dead) break;
    if (seg + 1 < seg1) {
        constexpr int NSEG = STRATO ? 6 : 5;
        auto a = segment_sum + NSEG * seg;
        float piece = dot(ext, load4(a)) + ext4 * a[4] + (STRATO ? ext5 * a[5] : 0.f);
        float y = piece - compensation;
        float t = prefix + y;
        compensation = (t - prefix) - y;
        prefix = t;
    }
}
if (BD && end_index[r]>=0) {
    int i=end_index[r]*W+w;
    float frac=ray[27];
    float e=(1.f-frac)*boundary[i]+frac*boundary[i+W];
    float depth=dot(load4(ray+22),ext)+ray[26]*ext4+(STRATO ? ray[28]*ext5 : 0.f);
    total+=ftz(0.5f*ftz((e*exp(-depth))*pu));
}
if (!REFINE) flags[r * W + w] = int(retry);
float weight=ray[21];
for (int k=0;k<4;++k) weighted[(k*nr+r)*W+w]=total[k]*weight;
weighted[(4*nr+r)*W+w]=((total.x+total.y)+total.z)*weight;
"""

# Coalesced cell projection: one threadgroup per (wave-quad, cell), streaming basis in coalesced
# order while sharing each 8-ray trace block in threadgroup memory. Exact 8-ray block + Kahan sum.
# proj_mode[0]: 0 = moments only (4+K rows), 1 = moments + flux (5+K rows), 2 = flux only (1 row).
PROJECT = r"""
uint lid = thread_position_in_threadgroup.x;
int w0 = int(threadgroup_position_in_grid.x) * 4;
int c = int(threadgroup_position_in_grid.y);
int nr = cells[cells_shape[0] - 1];
int r0 = cells[c], r1 = cells[c + 1];
int mode = proj_mode[0];

if (mode == 2) {
    if (lid != 0) return;
    float4 sum(0.f), correction(0.f);
    for (int start = r0; start < r1; start += 8) {
        int end = min(start + 8, r1);
        float4 block(0.f);
        for (int r = start; r < end; ++r) {
            float dz = max(0.f, compact[9 * r + 5]);
            if (dz > 0.f) {
                auto base_w = weighted + (4 * nr + r) * W + w0;
                block += float4(
                    w0 < W ? base_w[0] : 0.f,
                    w0 + 1 < W ? base_w[1] : 0.f,
                    w0 + 2 < W ? base_w[2] : 0.f,
                    w0 + 3 < W ? base_w[3] : 0.f
                ) * dz;
            }
        }
        float4 y = block - correction;
        float4 next = sum + y;
        correction = (next - sum) - y;
        sum = next;
    }
    auto out = moment + c * W + w0;
    if (w0 < W) out[0] = sum.x;
    if (w0 + 1 < W) out[1] = sum.y;
    if (w0 + 2 < W) out[2] = sum.z;
    if (w0 + 3 < W) out[3] = sum.w;
    return;
}

int total_k = mode == 1 ? (5 + K) : (4 + K);
threadgroup float4 tw[64];
for (int k_base = 0; k_base < total_k; k_base += TG) {
    int k = k_base + int(lid);
    bool active = k < total_k;
    int sh = k - 4;
    int m = 0;
    if (active && !PREMUL && sh >= 0 && sh < K) {
        int l = int((sqrt(float(8 * sh + 1)) - 1.f) * 0.5f);
        m = sh - l * (l + 1) / 2;
    }
    float4 sum(0.f), correction(0.f);
    for (int chunk = r0; chunk < r1; chunk += 64) {
        int chunk_end = min(chunk + 64, r1);
        int chunk_len = chunk_end - chunk;
        for (int j = int(lid); j < chunk_len; j += TG) {
            auto base_w = weighted + (4 * nr + chunk + j) * W + w0;
            tw[j] = float4(
                w0 < W ? base_w[0] : 0.f,
                w0 + 1 < W ? base_w[1] : 0.f,
                w0 + 2 < W ? base_w[2] : 0.f,
                w0 + 3 < W ? base_w[3] : 0.f
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (int start = chunk; start < chunk_end; start += 8) {
                int end = min(start + 8, chunk_end);
                float4 block(0.f);
                if (k < 4) {
                    auto v = weighted + k * nr * W + w0;
                    for (int r = start; r < end; ++r) {
                        auto row = v + r * W;
                        block += float4(
                            w0 < W ? row[0] : 0.f,
                            w0 + 1 < W ? row[1] : 0.f,
                            w0 + 2 < W ? row[2] : 0.f,
                            w0 + 3 < W ? row[3] : 0.f
                        );
                    }
                } else if (k == 4 + K) {
                    for (int r = start; r < end; ++r) {
                        float dz = max(0.f, compact[9 * r + 5]);
                        if (dz > 0.f) block += tw[r - chunk] * dz;
                    }
                } else {
                    for (int r = start; r < end; ++r) {
                        float fac = PREMUL ? basis[r * K + sh] : (basis[r * K + sh] * cosine[r * (L + 1) + m]);
                        block += tw[r - chunk] * fac;
                    }
                }
                float4 y = block - correction;
                float4 next = sum + y;
                correction = (next - sum) - y;
                sum = next;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        auto out = moment + (c * total_k + k) * W + w0;
        if (w0 < W) out[0] = sum.x;
        if (w0 + 1 < W) out[1] = sum.y;
        if (w0 + 2 < W) out[2] = sum.z;
        if (w0 + 3 < W) out[3] = sum.w;
    }
}
"""

# Tabulate the mirrored ground quadrature on an angular grid for smooth HG scattering.
HG_TABLE = r"""
uint tx = thread_position_in_grid.x;
int lane = int(tx & 15u);
int pi = int(tx >> 4);
int ti = int(thread_position_in_grid.y);
int c_idx = int(thread_position_in_grid.z);
if (pi >= HG_NODES) return;
int base_lane = int(thread_index_in_simdgroup & 16u);

float step = 3.14159265358979323846f / float(HG_BINS);
float ta = float(ti - 2) * step;
float pa = float(pi - 2) * step;
float st = precise::sin(ta), ct = precise::cos(ta);
float sp = precise::sin(pa), cp = precise::cos(pa);
float ex = st * cp, ey = abs(st * sp);

float g = phys[5], g2 = g * g;
float hg_num = (1.f - g2) * 0.07957747154594767f;

int cell = owners[c_idx];
int j_start = goff[cell], j_end = goff[cell + 1];

float sum[WG];
for (int wg = 0; wg < WG; ++wg) sum[wg] = 0.f;

for (int j0 = j_start; j0 < j_end; j0 += 16) {
    int j = j0 + lane;
    float ph = 0.f;
    if (j < j_end) {
        auto gd = gdir + 3 * j;
        float d_dot = ct * gd[2] + ex * gd[0];
        float e_dot = ey * gd[1];
        ph = 0.5f * (hg_phase(clamp(d_dot + e_dot, -1.f, 1.f), g, g2, hg_num)
                     + hg_phase(clamp(d_dot - e_dot, -1.f, 1.f), g, g2, hg_num));
    }
    int cnt = min(16, j_end - j0);
    for (int k = 0; k < cnt; ++k) {
        float pk = simd_shuffle(ph, ushort(base_lane + k));
        for (int wg = 0; wg < WG; ++wg) {
            int w = wg * 16 + lane;
            if (w < W) sum[wg] = fma(pk, gval[(j0 + k) * W + w], sum[wg]);
        }
    }
}
size_t out_off = ((size_t(c_idx) * HG_NODES + ti) * HG_NODES + pi) * W;
for (int wg = 0; wg < WG; ++wg) {
    int w = wg * 16 + lane;
    if (w < W) table[out_off + w] = sum[wg];
}
"""

# Build 1D Mie phase LUT [STRATO_LUT_BINS + 1, W] on device from strato_chi [33, W].
STRATO_LUT_BUILD = r"""
uint idx = thread_position_in_grid.x;
int total = (STRATO_LUT_BINS + 1) * W;
if (int(idx) >= total) return;
int i = int(idx) / W;
int w = int(idx) % W;
float u = -1.f + float(i) * (2.f / float(STRATO_LUT_BINS));
lut[idx] = strato_mie_phase(u, w, chi, W);
"""

# Tabulate the mirrored ground quadrature on an angular grid for smooth stratospheric Mie scattering.
MIE_TABLE = r"""
uint tx = thread_position_in_grid.x;
int lane = int(tx & 15u);
int pi = int(tx >> 4);
int ti = int(thread_position_in_grid.y);
int c_idx = int(thread_position_in_grid.z);
if (pi >= HG_NODES) return;
int base_lane = int(thread_index_in_simdgroup & 16u);

float step = 3.14159265358979323846f / float(HG_BINS);
float ta = float(ti - 2) * step;
float pa = float(pi - 2) * step;
float st = precise::sin(ta), ct = precise::cos(ta);
float sp = precise::sin(pa), cp = precise::cos(pa);
float ex = st * cp, ey = abs(st * sp);

int cell = owners[c_idx];
int j_start = goff[cell], j_end = goff[cell + 1];

float sum[WG];
for (int wg = 0; wg < WG; ++wg) sum[wg] = 0.f;

for (int j0 = j_start; j0 < j_end; j0 += 16) {
    int j = j0 + lane;
    float u_p = 0.f, u_m = 0.f;
    if (j < j_end) {
        auto gd = gdir + 3 * j;
        float d_dot = ct * gd[2] + ex * gd[0];
        float e_dot = ey * gd[1];
        u_p = clamp(d_dot + e_dot, -1.f, 1.f);
        u_m = clamp(d_dot - e_dot, -1.f, 1.f);
    }
    int cnt = min(16, j_end - j0);
    for (int k = 0; k < cnt; ++k) {
        float upk = simd_shuffle(u_p, ushort(base_lane + k));
        float umk = simd_shuffle(u_m, ushort(base_lane + k));
        for (int wg = 0; wg < WG; ++wg) {
            int w = wg * 16 + lane;
            if (w < W) {
                float pk = 0.5f * (strato_mie_lut(upk, w, st_lut, W) + strato_mie_lut(umk, w, st_lut, W));
                sum[wg] = fma(pk, gval[(j0 + k) * W + w], sum[wg]);
            }
        }
    }
}
size_t out_off = ((size_t(c_idx) * HG_NODES + ti) * HG_NODES + pi) * W;
for (int wg = 0; wg < WG; ++wg) {
    int w = wg * 16 + lane;
    if (w < W) table[out_off + w] = sum[wg];
}
"""

# Fused Simpson prefix, solar visibility/attenuation, and first-light integration.
# Each 16-lane half-SIMDgroup traces one ray; all 32 lanes stay lockstep across shuffles.
FIRST_INTEGRATE = r"""
uint tid = thread_position_in_grid.x;
int nr = ro_shape[0] - 1;
int raw_r = int(tid >> 4);
int local = int(tid & 15u);
int base_lane = int(thread_index_in_simdgroup & 16u);
int r = min(raw_r, nr - 1);

int left = 0, right = co_shape[0] - 1;
while (left + 1 < right) {
    int mid = (left + right) >> 1;
    if (co[mid] <= r) left = mid; else right = mid;
}
auto f = frames + 14 * left;
auto ray = compact + 9 * r;

float3 p = load3(f);
float3 d = load3(ray);
float3 dl = load3(ray + 3);
float pd = ray[7];
float h0 = f[12];
float cyl0 = f[13];
float pd_xy = fma(p.x, d.x, p.y * d.y);
float dxy2 = dot(d.xy, d.xy);
float top = phys[0];
float mix = phys[4];
float iso = (1.f - mix) / 3.f;
float aer_g = phys[5];
float weight = ray[8];
int sb = shadow[2 * r];
int se = shadow[2 * r + 1];

float4 pu = float4(1.f - dl * dl, -dl.x * dl.z);
float3 u_vec = float3(f[5], f[8], f[11]);
float sun_dot = (dl.x * u_vec.x + dl.y * u_vec.y) + dl.z * u_vec.z;
float3 sv = u_vec - dl * sun_dot;
float4 direct = 0.5f * (pu - float4(sv.x * sv.x, sv.y * sv.y, sv.z * sv.z, sv.x * sv.z));
float4 ray_pol = (mix * 0.11936620731892151f) * direct + ((1.f - mix) * 0.039788735772973836f) * pu;

float3 b0 = float3(0.f), b1 = float3(0.f), b2 = float3(0.f);
if (GROUND) {
    b0 = load3(f + 3) - dl.x * d;
    b1 = load3(f + 6) - dl.y * d;
    b2 = load3(f + 9) - dl.z * d;
}

float g2 = aer_g * aer_g;
float hg_num = (1.f - g2) * 0.07957747154594767f;
float phase = hg_phase(d.z, aer_g, g2, hg_num);

float4 ext03[WG];
float ext4[WG];
float ext5[WG];
float asca_w[WG];
float st_sca_w[WG];
float sphase[WG];
float prefix[WG];
float comp[WG];
float4 total[WG];
float4 correction[WG];
for (int g = 0; g < WG; ++g) {
    int w = g * 16 + local;
    if (w < W) {
        ext03[g] = float4(extinction[w], extinction[W + w], extinction[2 * W + w], extinction[3 * W + w]);
        ext4[g] = extinction[4 * W + w];
        ext5[g] = STRATO ? extinction[5 * W + w] : 0.f;
        asca_w[g] = asca[w];
        st_sca_w[g] = STRATO ? asca[W + w] : 0.f;
        sphase[g] = STRATO ? strato_mie_phase(d.z, w, st_phase, W) : 0.f;
    } else {
        ext03[g] = float4(0.f);
        ext4[g] = 0.f;
        ext5[g] = 0.f;
        asca_w[g] = 0.f;
        st_sca_w[g] = 0.f;
        sphase[g] = 0.f;
    }
    prefix[g] = 0.f;
    comp[g] = 0.f;
    total[g] = float4(0.f);
    correction[g] = float4(0.f);
}

int seg0 = ro[r], seg1 = ro[r + 1];
int nseg = seg1 - seg0;
int max_nseg = simd_max(nseg);

for (int s_idx = 0; s_idx < max_nseg; ++s_idx) {
    bool active_seg = s_idx < nseg;
    int seg = active_seg ? (seg0 + s_idx) : 0;
    float3 a_pos = (EXTENDED && active_seg) ? load3(seg_ext + 11 * seg) : p;
    float s_h0 = (EXTENDED && active_seg) ? seg_ext[11 * seg + 3] : h0;
    float s_pd = (EXTENDED && active_seg) ? seg_ext[11 * seg + 4] : pd;
    float s_cyl0 = (EXTENDED && active_seg) ? seg_ext[11 * seg + 5] : cyl0;
    float s_pd_xy = (EXTENDED && active_seg) ? seg_ext[11 * seg + 6] : pd_xy;
    float shell = (EXTENDED && active_seg) ? seg_ext[11 * seg + 7] : 0.f;
    float base = active_seg ? (EXTENDED ? 0.f : intervals[2 * seg]) : 0.f;
    float span = active_seg ? intervals[2 * seg + 1] : 0.f;
    float width = abs(span);
    bool lower = span > 0.f;
    float start = lower ? base : base + span;

    float4 carry03(0.f);
    float carry4 = 0.f;
    float carry_st = 0.f;
    for (int first_k = 0; first_k <= NQ; first_k += 16) {
        int k = first_k + local;
        float u = k < NQ ? quad[2 * k] : 1.f;
        float old_u = k == 0 ? 0.f : (k <= NQ ? quad[2 * (k - 1)] : 1.f);
        float v = lower ? u : 1.f - u;
        float f_cur = lower ? u * u : u * (2.f - u);
        float old_f = lower ? old_u * old_u : old_u * (2.f - old_u);
        float t = fma(span, v * v, base);
        float h = active_seg ? path_altitude<EXTENDED>(t, s_pd, s_h0, shell, top) : 0.f;
        if (EXTENDED && active_seg && k == NQ)
            h = endpoint_support_height(seg_ext[11 * seg + 10], seg_ext[11 * seg + 9], phys);
        Density now = active_seg ? density<MASK, LAYER, NT, EXTENDED>(h, tab, phys, aer) : Density{float4(0.f), 0.f};
        float now_st = (STRATO && active_seg) ? strato_density<EXTENDED>(h, phys) : 0.f;
        float tm = fma(width, 0.5f * (old_f + f_cur), start);
        float hm = active_seg ? path_altitude<EXTENDED>(tm, s_pd, s_h0, shell, top) : 0.f;
        Density mid = active_seg ? density<MASK, LAYER, NT, EXTENDED>(hm, tab, phys, aer) : Density{float4(0.f), 0.f};
        float mid_st = (STRATO && active_seg) ? strato_density<EXTENDED>(hm, phys) : 0.f;
        float panel = (f_cur - old_f) * width / 6.f;

        Density old = {simd_shuffle_up(now.p, 1), simd_shuffle_up(now.o4, 1)};
        float old_st = STRATO ? simd_shuffle_up(now_st, 1) : 0.f;
        if (local == 0) {
            float last_v = lower ? old_u : 1.f - old_u;
            float pt_t = (lower || k > 0) ? fma(span, last_v * last_v, base) : start;
            float ho = active_seg ? path_altitude<EXTENDED>(pt_t, s_pd, s_h0, shell, top) : 0.f;
            if (EXTENDED && active_seg && k == 0)
                ho = endpoint_support_height(seg_ext[11 * seg + 8], seg_ext[11 * seg + 9], phys);
            old = active_seg ? density<MASK, LAYER, NT, EXTENDED>(ho, tab, phys, aer) : Density{float4(0.f), 0.f};
            if (STRATO) old_st = active_seg ? strato_density<EXTENDED>(ho, phys) : 0.f;
        }
        float4 acc03 = panel * ((old.p + now.p) + 4.f * mid.p);
        float acc4 = panel * ((old.o4 + now.o4) + 4.f * mid.o4);
        float acc_st = STRATO ? panel * ((old_st + now_st) + 4.f * mid_st) : 0.f;
        for (ushort stride = 1; stride < 16; stride *= 2) {
            float4 v03 = simd_shuffle_up(acc03, stride);
            float v4 = simd_shuffle_up(acc4, stride);
            float vst = STRATO ? simd_shuffle_up(acc_st, stride) : 0.f;
            if (local >= stride) { acc03 += v03; acc4 += v4; if (STRATO) acc_st += vst; }
        }
        acc03 += carry03;
        acc4 += carry4;
        if (STRATO) acc_st += carry_st;

        int node = s_idx * NQ + k;
        bool active_node = active_seg && k < NQ;
        bool is_lit = active_node && !(sb <= node && node < se);
        float3 pos = active_node ? fma(float3(t), d, a_pos) : float3(0.f);
        float4 sc03(0.f);
        float sc4 = 0.f;
        float sc_st = 0.f;
        if (is_lit) {
            float td = sqrt(max(0.f, h * (12742.f + h)));
            float numer = pos.z + td;
            if (pos.z < 0.f && td > 0.f) {
                float cyl = fma(t, fma(t, dxy2, 2.f * s_pd_xy), s_cyl0);
                numer = cyl / (td - pos.z);
            }
            float x_sun = sqrt(clamp(numer / ((6371.f + h) + td), 0.f, 1.f));
            int ih = bracket(sun_h, NSH, h);
            float ar = (h - sun_h[ih]) / (sun_h[ih + 1] - sun_h[ih]);
            float fx = x_sun * float(NSX - 1);
            int ic = clamp(int(fx), 0, NSX - 2);
            float br = fx - float(ic);
            float w00 = (1.f - ar) * (1.f - br);
            float w10 = ar * (1.f - br);
            float w01 = (1.f - ar) * br;
            float w11 = ar * br;
            constexpr int NSC = STRATO ? 6 : 5;
            auto s00 = sun_col + (ih * NSX + ic) * NSC;
            auto s10 = sun_col + ((ih + 1) * NSX + ic) * NSC;
            auto s01 = sun_col + (ih * NSX + ic + 1) * NSC;
            auto s11 = sun_col + ((ih + 1) * NSX + ic + 1) * NSC;
            sc03 = w00 * load4(s00) + w10 * load4(s10) + w01 * load4(s01) + w11 * load4(s11);
            sc4 = w00 * s00[4] + w10 * s10[4] + w01 * s01[4] + w11 * s11[4];
            if (STRATO) {
                sc_st = w00 * s00[5] + w10 * s10[5] + w01 * s01[5] + w11 * s11[5];
            }
        }

        int base_cell = 0;
        int has_g = 0;
        float4 sw(0.f);
        float2 axis(1.f, 0.f);
        float sn = 0.f, cs = 1.f;
        if (GROUND && active_node) {
            float rho = length(pos.xy);
            float ang = precise::atan2(rho, pos.z);
            int gi = bracket(hg, NH, h);
            int gj = bracket(cg, NC, ang);
            base_cell = gi * NC + gj;
            has_g = cell_gflag[base_cell];
            if (has_g) {
                float ah = clamp((h - hg[gi]) / (hg[gi + 1] - hg[gi]), 0.f, 1.f);
                float ac = clamp((ang - cg[gj]) / (cg[gj + 1] - cg[gj]), 0.f, 1.f);
                sw = float4(1.f - ah, 1.f - ah, ah, ah) * float4(1.f - ac, ac, 1.f - ac, ac);
                axis = azimuth(pos.xy, rho);
                float r_ang = cg[gj] + ac * (cg[gj + 1] - cg[gj]);
                sn = precise::sin(r_ang);
                cs = precise::cos(r_ang);
            }
        }
        bool has_scat = !EXTENDED || now.p.x != 0.f || now.p.y != 0.f || now_st != 0.f;
        bool need_node = has_scat && (GROUND ? (is_lit || has_g != 0) : is_lit);
        float ds = need_node ? (2.f * (width * v) * quad[2 * k + 1]) : 0.f;

        uint vote = uint(simd_vote::vote_t(simd_ballot(need_node)));
        uint lit_mask = (vote >> base_lane) & 0xFFFFu;
        uint both_lit = (vote | (vote >> 16)) & 0xFFFFu;
        while (both_lit) {
            int j = ctz(both_lit);
            both_lit &= both_lit - 1u;
            ushort src = ushort(base_lane + j);
            float n_ds = simd_shuffle(ds, src);
            float4 n_acc03 = simd_shuffle(acc03, src);
            float n_acc4 = simd_shuffle(acc4, src);
            float n_acc_st = STRATO ? simd_shuffle(acc_st, src) : 0.f;
            float4 n_sc03 = simd_shuffle(sc03, src);
            float n_sc4 = simd_shuffle(sc4, src);
            float n_sc_st = STRATO ? simd_shuffle(sc_st, src) : 0.f;
            float n_air = simd_shuffle(now.p.x, src);
            float n_aer = simd_shuffle(now.p.y, src);
            float n_st = STRATO ? simd_shuffle(now_st, src) : 0.f;
            int n_lit = GROUND ? simd_shuffle(int(is_lit), src) : 1;
            int n_has_g = GROUND ? simd_shuffle(has_g, src) : 0;
            int n_base = GROUND ? simd_shuffle(base_cell, src) : 0;
            float4 n_sw = GROUND ? simd_shuffle(sw, src) : float4(0.f);
            float2 n_axis = GROUND ? simd_shuffle(axis, src) : float2(1.f, 0.f);
            float n_sn = GROUND ? simd_shuffle(sn, src) : 0.f;
            float n_cs = GROUND ? simd_shuffle(cs, src) : 1.f;

            if (lit_mask & (1u << j)) {
                float att_g[WG];
                bool any_att = false;
                for (int g = 0; g < WG; ++g) {
                    int w = g * 16 + local;
                    if (w < W) {
                        float depth = (prefix[g] - comp[g]) + (dot(ext03[g], n_acc03) + ext4[g] * n_acc4 + (STRATO ? ext5[g] * n_acc_st : 0.f));
                        float att = ftz(n_ds * ftz(fast::exp(-depth)));
                        att_g[g] = att;
                        any_att |= (att != 0.f);
                    } else {
                        att_g[g] = 0.f;
                    }
                }
                float ga[WG];
                float sga[WG];
                for (int g = 0; g < WG; ++g) { ga[g] = 0.f; sga[g] = 0.f; }
                float4 t0(0.f), t1(0.f), t2(0.f), t3(0.f);
                if (GROUND && n_has_g) {
                    float3 q0(-n_cs * n_axis.x, -n_cs * n_axis.y, n_sn);
                    float3 q1(n_axis.y, -n_axis.x, 0.f);
                    float3 q2(n_sn * n_axis.x, n_sn * n_axis.y, n_cs);
                    if (n_aer != 0.f || (STRATO && n_st != 0.f)) {
                        float mu_g = scalar_dot3(d, q2);
                        float exd_g = scalar_dot3(d, q0);
                        float aey_g = abs(scalar_dot3(d, q1));
                        if (HG_TABLE) {
                            if (any_att) {
                                int tk, pk;
                                float tw[6], pw[6];
                                float ta = precise::atan2(length(float2(exd_g, aey_g)), mu_g) * (float(HG_BINS) / 3.14159265358979323846f);
                                float pa = (aey_g == 0.f && exd_g == 0.f) ? 0.f : precise::atan2(aey_g, exd_g) * (float(HG_BINS) / 3.14159265358979323846f);
                                hg_stencil(ta, tk, tw);
                                hg_stencil(pa, pk, pw);
                                for (int c = 0; c < 4; ++c) {
                                    int cc = n_base + (c / 2) * NC + (c % 2);
                                    int tidx = g_idx[cc];
                                    if (tidx >= 0) {
                                        size_t cell_off_t = ((size_t(tidx) * HG_NODES + tk) * HG_NODES + pk) * W;
                                        for (int g = 0; g < WG; ++g) {
                                            int w = g * 16 + local;
                                            if (w < W && att_g[g] != 0.f) {
                                                if (n_aer != 0.f) {
                                                    auto row = g_f0 + cell_off_t + w;
                                                    float gv_sum = 0.f;
                                                    for (int jj = 0; jj < 6; ++jj) {
                                                        float az_sum = 0.f;
                                                        for (int ii = 0; ii < 6; ++ii)
                                                            az_sum = fma(pw[ii], row[(jj * HG_NODES + ii) * W], az_sum);
                                                        gv_sum = fma(tw[jj], az_sum, gv_sum);
                                                    }
                                                    ga[g] = fma(n_sw[c], gv_sum, ga[g]);
                                                }
                                                if (STRATO && n_st != 0.f) {
                                                    auto srow = g_f1 + cell_off_t + w;
                                                    float sgv_sum = 0.f;
                                                    for (int jj = 0; jj < 6; ++jj) {
                                                        float az_sum = 0.f;
                                                        for (int ii = 0; ii < 6; ++ii)
                                                            az_sum = fma(pw[ii], srow[(jj * HG_NODES + ii) * W], az_sum);
                                                        sgv_sum = fma(tw[jj], az_sum, sgv_sum);
                                                    }
                                                    sga[g] = fma(n_sw[c], sgv_sum, sga[g]);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            uint att_v = uint(simd_vote::vote_t(simd_ballot(any_att)));
                            if ((att_v >> base_lane) & 0xFFFFu) {
                                for (int c = 0; c < 4; ++c) {
                                    int cc = n_base + (c / 2) * NC + (c % 2);
                                    int j_start = g_idx[cc], j_end = g_idx[cc + 1];
                                    float swc = n_sw[c];
                                    for (int j0 = j_start; j0 < j_end; j0 += 16) {
                                        int jj = j0 + local;
                                        float ph = 0.f, u_p = 0.f, u_m = 0.f;
                                        if (jj < j_end) {
                                            auto gd = g_f0 + 3 * jj;
                                            float m0 = mu_g * gd[2] + exd_g * gd[0];
                                            float m1 = aey_g * gd[1];
                                            u_p = clamp(m0 + m1, -1.f, 1.f);
                                            u_m = clamp(m0 - m1, -1.f, 1.f);
                                            if (n_aer != 0.f)
                                                ph = 0.5f * (hg_phase(u_p, aer_g, g2, hg_num)
                                                             + hg_phase(u_m, aer_g, g2, hg_num));
                                        }
                                        int cnt = min(16, j_end - j0);
                                        for (int kk = 0; kk < cnt; ++kk) {
                                            float pk = n_aer != 0.f ? simd_shuffle(ph, ushort(base_lane + kk)) : 0.f;
                                            float wph = ftz(swc * pk);
                                            float upk = (STRATO && n_st != 0.f) ? simd_shuffle(u_p, ushort(base_lane + kk)) : 0.f;
                                            float umk = (STRATO && n_st != 0.f) ? simd_shuffle(u_m, ushort(base_lane + kk)) : 0.f;
                                            for (int g = 0; g < WG; ++g) {
                                                int w = g * 16 + local;
                                                if (w < W && att_g[g] != 0.f) {
                                                    float gvw = g_f1[(j0 + kk) * W + w];
                                                    if (n_aer != 0.f)
                                                        ga[g] = ftz(ga[g] + ftz(wph * gvw));
                                                    if (STRATO && n_st != 0.f) {
                                                        float sph = 0.5f * (strato_mie_lut(upk, w, st_phase + 33 * W, W)
                                                                            + strato_mie_lut(umk, w, st_phase + 33 * W, W));
                                                        sga[g] = ftz(sga[g] + ftz(ftz(swc * sph) * gvw));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (n_air != 0.f && any_att) {
                        float r0 = dot(b0.xy, n_axis), r1 = dot(b1.xy, n_axis), r2 = dot(b2.xy, n_axis);
                        float3 k0 = float3(fma(-n_cs, r0, b0.z * n_sn), fma(b0.x, n_axis.y, -b0.y * n_axis.x), fma(n_sn, r0, b0.z * n_cs));
                        float3 k1 = float3(fma(-n_cs, r1, b1.z * n_sn), fma(b1.x, n_axis.y, -b1.y * n_axis.x), fma(n_sn, r1, b1.z * n_cs));
                        float3 k2 = float3(fma(-n_cs, r2, b2.z * n_sn), fma(b2.x, n_axis.y, -b2.y * n_axis.x), fma(n_sn, r2, b2.z * n_cs));
                        t0 = rotation(k0, k0); t1 = rotation(k1, k1); t2 = rotation(k2, k2); t3 = rotation(k0, k2);
                    }
                }
                for (int g = 0; g < WG; ++g) {
                    int w = g * 16 + local;
                    if (w < W && att_g[g] != 0.f) {
                        float opt = dot(ext03[g], n_sc03) + ext4[g] * n_sc4 + (STRATO ? ext5[g] * n_sc_st : 0.f);
                        float sun = (!GROUND || n_lit) ? ftz(fast::exp(-opt)) : 0.f;
                        float ray_s = ftz(ftz(sun * n_air) * ext03[g].x);
                        float beam_s = ftz(ftz(n_aer * asca_w[g]) * ftz(ftz(sun * phase) + ga[g]));
                        float4 val = ftz(ftz(ray_s * ray_pol) + ftz(ftz(beam_s * pu) * 0.5f));
                        if (STRATO) {
                            float sbeam_s = ftz(ftz(n_st * st_sca_w[g]) * ftz(ftz(sun * sphase[g]) + sga[g]));
                            val = ftz(val + ftz(ftz(sbeam_s * pu) * 0.5f));
                        }
                        if (GROUND && n_has_g && n_air != 0.f) {
                            float4 rm(0.f);
                            for (int c = 0; c < 4; ++c) {
                                int cc = n_base + (c / 2) * NC + (c % 2);
                                auto gp = gm + (cc * 4) * W + w;
                                rm += n_sw[c] * float4(gp[0], gp[W], gp[2 * W], gp[3 * W]);
                            }
                            float tr = (rm.x + rm.y) + rm.z;
                            float iso_tr = ftz(iso * tr);
                            float4 rot(dot(t0, rm), dot(t1, rm), dot(t2, rm), dot(t3, rm));
                            val = ftz(val + ftz(n_air * ftz(ftz(mix * rot) + ftz(iso_tr * pu))));
                        }
                        float4 y = ftz(ftz(val * att_g[g]) - correction[g]);
                        float4 upd = ftz(total[g] + y);
                        correction[g] = ftz(ftz(upd - total[g]) - y);
                        total[g] = upd;
                    }
                }
            }
        }

        if (first_k + 16 <= NQ) {
            ushort last_lane = ushort(base_lane + 15);
            carry03 = simd_shuffle(acc03, last_lane);
            carry4 = simd_shuffle(acc4, last_lane);
            if (STRATO) carry_st = simd_shuffle(acc_st, last_lane);
        }
        if (NQ >= first_k && NQ < first_k + 16) {
            ushort end_lane = ushort(base_lane + (NQ - first_k));
            float4 ssum03 = simd_shuffle(acc03, end_lane);
            float ssum4 = simd_shuffle(acc4, end_lane);
            float ssum_st = STRATO ? simd_shuffle(acc_st, end_lane) : 0.f;
            if (active_seg) {
                for (int g = 0; g < WG; ++g) {
                    int w = g * 16 + local;
                    if (w < W) {
                        float piece = dot(ext03[g], ssum03) + ext4[g] * ssum4 + (STRATO ? ext5[g] * ssum_st : 0.f);
                        float y = piece - comp[g];
                        float t_sum = prefix[g] + y;
                        comp[g] = (t_sum - prefix[g]) - y;
                        prefix[g] = t_sum;
                    }
                }
            }
        }
    }
}

if (raw_r < nr) {
    for (int g = 0; g < WG; ++g) {
        int w = g * 16 + local;
        if (w < W) {
            float4 tot = total[g];
            if (BD && end_index[r] >= 0) {
                int idx = end_index[r] * W + w;
                auto eptr = ends + 7 * r;
                float frac = eptr[5];
                float e = (1.f - frac) * boundary[idx] + frac * boundary[idx + W];
                float depth = dot(load4(eptr), ext03[g]) + eptr[4] * ext4[g] + (STRATO ? eptr[6] * ext5[g] : 0.f);
                tot += ftz(0.5f * ftz((e * exp(-depth)) * pu));
            }
            for (int k = 0; k < 4; ++k)
                weighted[(k * nr + r) * W + w] = tot[k] * weight;
            weighted[(4 * nr + r) * W + w] = ((tot.x + tot.y) + tot.z) * weight;
        }
    }
}
"""

KERNELS = {
    'panels': ('rays ro intervals quad hg cg tab phys aer seg_ext', 'points c0cp segment_sum owners st_acc', PARALLEL_GEOMETRY),
    'seg_prefix': ('ro segment_sum ext_min', 'seg_depth', SEG_PREFIX),
    'basis': ('compact norms co frames ends bd_flag', 'basis cosine expanded', BASIS),
    'vp_count': ('c0cp live_map seg_depth meta', 'counts', VP_COUNT),
    'vp_scatter': ('c0cp owners live_map seg_depth offs meta', 'sorted_rcp inv_slot', VP_SCATTER),
    'vp_blocks': ('offs inc live_ir', 'bstart blk', VP_BLOCKS),
    'vp_vprod': ('bstart blk sorted_rcp basis src_sh densities', 'aero', VP_VPROD),
    'vp_vprod_f4_w64': ('bstart blk sorted_rcp compact src_sh densities', 'aero', VP_VPROD_F4_W64),
    'integrate': ('rays offsets points c0cp segment_sum src_rayleigh aero inv_slot densities live_map extinction phys boundary end_index frame cell_off angles raw_sh intervals quad polar basis_table st_acc seg_ext coarse prior_flags',
                  'weighted flags', INTEGRATE),
    'first': ('compact frames co ro intervals shadow quad sun_h sun_col tab phys aer extinction asca boundary ends end_index hg cg gm cell_gflag g_idx g_f0 g_f1 st_phase seg_ext',
              'weighted', FIRST_INTEGRATE),
    'project': ('cells basis cosine weighted compact proj_mode', 'moment', PROJECT),
    'hg_table': ('owners goff gdir gval phys', 'table', HG_TABLE),
    'strato_lut': ('chi', 'lut', STRATO_LUT_BUILD),
    'mie_table': ('owners goff gdir gval st_lut', 'table', MIE_TABLE),
}
