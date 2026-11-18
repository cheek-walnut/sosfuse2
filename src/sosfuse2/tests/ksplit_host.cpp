// Each build includes the kernel under one warm transport policy.
#include <algorithm>
#include <cmath>
#include <vector>

#include "cuda_stub.h"
Dim blockIdx{}, threadIdx{}, blockDim{};
extern "C" { double shared[50000]; }
template <> inline float __shfl_sync(unsigned, float value, int source_lane) {
    return warp_read(value, source_lane);
}
#include "kernel.cu"

template <class Point>
static void export_point(const Point &point, float *row) {
    std::copy_n(point.T, 16, row);
    std::copy_n(point.sw, 4, row + 16);
    std::copy_n(point.cell, 4, row + 20);
    row[24] = point.ir;
    row[25] = point.h;
    row[26] = point.ds;
}

extern "C" __attribute__((visibility("default")))
void run_segment(int mode, int ray, int seg, int lower, double a, double b,
                 const double *start, const double *direction, int nq,
                 const double *nodes, const double *weights, int nh, const float *heights,
                 int nc, const float *angles, const float *coeff,
                 float *columns, float *geometry, float *attenuation) {
    using Kernel = Transport<Sample>;
    constexpr int capacity = TransportPolicy::batch_size;
    const float basis[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    DS p[3], d[3];
    float dl[3];
    for (int k = 0; k < 3; ++k) {
        p[k] = ds(start[k]);
        d[k] = ds(direction[k]);
        dl[k] = direction[k];
    }
    const float st = std::hypot(dl[0], dl[1]);
    float table[3 * PROFILE_N];
    for (int i = 0; i < PROFILE_N; ++i) {
        table[3 * i] = 0.82f;
        table[3 * i + 1] = 0.55f;
        table[3 * i + 2] = 0.23f;
    }
    Sample sample{};
    Kernel k{};
    k.samples = &sample;
    k.nq = nq;
    k.nh = nh;
    k.nc = nc;
    k.sinv_mode = mode;
    k.sinv_rec = columns;
    k.profile = {5.5f, nullptr, table, 3, 1.2f, 12.f, 7.f, 1.7f};
    k.ext = coeff;
    k.st_ext = coeff + NCOMP * FW;
    k.optics = {k.profile, k.ext, k.st_ext};

    // With mix=0, unit diagonal moments and air density give a unit source in each selected component.
    std::vector<float> source(k.nh * k.nc * SRC_ROWS_ST * FW, 1.f);
    std::vector<float> aerosol(nq * 4 * AERO_W);
    std::vector<float> density(k.nh, 1.f);
    k.src = source.data();
    k.live_a = k.live_s = 0;
    k.slim_aero = aerosol.data();
    k.slim_base = (seg - ray) * nq;
    k.hr = k.st_density = density.data();

    float y[YPAD]{};
    Kernel::PtBatch batch[capacity]{};
    auto run_lane = [&](int lane, Kernel transport) {
        host_lane = transport.lane = lane;
        Kernel::RayState r{};
        r.y = y;
        r.beta_p = p;
        r.beta_d = d;
        // In a pair the even node contributes to component 0, the odd to component 1.
        // Both can then be read individually after the real shuffle and compensated sum.
        const bool upper = TransportPolicy::slim_rays && lane >= 16;
        const int w = upper ? lane - 16 : lane;
        r.pu_r[0] = upper ? 0.f : 1.f;
        r.pu_r[1] = upper ? 1.f : 0.f;
        r.pu_r[2] = 1.f;
        if (nq <= 32 && lane < nq) {
            r.gu_l = ds(nodes[lane]);
            r.gw_l = ds(weights[lane]);
        }
        if (w < FW && mode == 2) {
            for (int c = 0; c < NCOMP; ++c)
                r.ray_ext[c] = k.ext[c * FW + w];
            r.ray_st_ext = k.st_ext[w];
        } else if (w < FW) {
            r.sp_t = ds(a);
            r.sp_b = beta_at_side(p, d, r.sp_t, ds((a + b) / 2), k.optics, w);
        }
        for (int first = 0; first < nq; first += capacity) {
            const int end = std::min(nq, first + capacity);
            const int j = first + lane;
            const auto batch_node = quadrature_node(j, nq, j < end, nodes, weights, r.gu_l, r.gw_l);
            if (j < end)
                evaluate_point(batch[lane], p, d, basis, dl, st, ds(a), ds(b), lower,
                               batch_node.u, batch_node.w, heights, k.nh, angles, k.nc,
                               k.profile, nullptr, 0, nullptr, 0, false);
            __syncwarp();
            for (int step = first; step < end; step += TransportPolicy::sample_stride) {
                const SampleLane owner(lane, step, end);
                auto view = transport.sample_view(batch, sample, first, owner.sample);
                std::fill_n(r.lane_rad, 4, 0.f);
                std::fill_n(r.lane_rc, 4, 0.f);
                const auto node = quadrature_node(step, nq, true, nodes, weights, r.gu_l, r.gw_l);
                transport.consume_sample(r, ray, seg, step, ds(a), ds(b), lower, node, owner, view);
                if (owner.active && owner.wave == 0)
                    export_point(view.sample, geometry + (step + int(upper)) * 27);
                if (!TransportPolicy::record && lane < FW) {
                    attenuation[step * FW + lane] = r.lane_rad[0];
                    if (TransportPolicy::slim_rays)
                        attenuation[(step + 1) * FW + lane] = r.lane_rad[1];
                }
                __syncwarp();
            }
        }
    };
    std::vector<std::thread> lanes;
    for (int lane = 0; lane < 32; ++lane)
        lanes.emplace_back(run_lane, lane, k);
    for (auto &lane : lanes)
        lane.join();
}
