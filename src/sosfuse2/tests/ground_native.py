"""Host execution of the actual first-light ground stage with a complete warp.

The scalar oracle evaluates each direction independently. The host warp uses
blocking barriers and real lane exchange; this is not a CUDA scheduling test.
"""

import json
import subprocess

from sosfuse2.checks import testkit
from sosfuse2.foundation import paths
from sosfuse2.tests.test_kernel_static import STUB
from sosfuse2.tests.test_tropo_mie import SYNTAX_STUB_EXTRA

WARP = r'''
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

class WarpBarrier {
    std::mutex mutex;
    std::condition_variable changed;
    unsigned arrived = 0, generation = 0;
public:
    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        const unsigned before = generation;
        if (++arrived == 32) {
            arrived = 0;
            ++generation;
            changed.notify_all();
        } else {
            changed.wait(lock, [&] { return generation != before; });
        }
    }
};
static WarpBarrier warp_barrier;
static thread_local int host_lane;
static float shuffle_values[32];
static void warp_sync() { warp_barrier.wait(); }
static float warp_read(float value, int source_lane) {
    shuffle_values[host_lane] = value;
    warp_sync();
    float result = shuffle_values[source_lane & 31];
    warp_sync();
    return result;
}
static float warp_down(float value, int offset) {
    return warp_read(value, host_lane + offset < 32 ? host_lane + offset : host_lane);
}
// Use the kernel's integer typedefs without colliding with system headers.
#define int64_t ground_int64_t
#define uint64_t ground_uint64_t
'''

POISON = r'''
Dim blockIdx{}, threadIdx{}, blockDim{};
extern "C" { double shared[50000]; }
struct Sample;
// Deliberately undefined: an inactive SH reference must fail at link time.
void forbidden_ground_sh_sums(const Sample &, int, const int64_t *, const float *,
                              const float *, float &, float &);
void forbidden_ground_cubic(float, float *);
#define ground_sh_sums forbidden_ground_sh_sums
#define ground_cubic forbidden_ground_cubic
'''

PROGRAM = r'''
#ifdef TROPO_MIE
static constexpr bool reference_mie = true;
#else
static constexpr bool reference_mie = false;
#endif
#if K_PHASE == 2 || (K_PHASE == 0 && !defined(EXP_SECTION_CLOCKS))
static constexpr bool reference_uses_y = true;
#else
static constexpr bool reference_uses_y = false;
#endif
static double reference_clip(double x) {
    return x < -1. ? -1. : x > 1. ? 1. : x;
}
static double reference_hg(double u, double g) {
    const double d = 1. + g * g - 2. * g * u;
    return (1. - g * g) / (4. * 3.14159265358979323846 * d * std::sqrt(d));
}
static double reference_lut(double u, int wave, const float *table) {
    const double x = (reference_clip(u) + 1.) * 8192.;
    const int node = x >= 16384. ? 16383 : (int)x;
    const double fraction = x - node;
    return (1. - fraction) * table[node * FW + wave] + fraction * table[(node + 1) * FW + wave];
}
// Direct, unbatched convolution: no kernel helpers or warp scratch are reused.
static void reference_ground(const Sample &s, const std::vector<int64_t> &offsets,
                             const std::vector<float> &directions, const std::vector<float> &values,
                             const std::vector<float> &strato, const std::vector<float> &tropo,
                             float g, double (&out)[2][FW]) {
    for (int corner = 0; corner < 4; ++corner) {
        const int cell = s.cell[corner];
        for (int64_t j = offsets[cell]; j < offsets[cell + 1]; ++j) {
            const double a = (double)s.mu * directions[3 * j + 2] + (double)s.exd * directions[3 * j];
            const double b = std::fabs((double)s.eyd) * directions[3 * j + 1];
            const double plus = reference_clip(a + b), minus = reference_clip(a - b);
            for (int wave = 0; wave < FW; ++wave) {
                const double value = (double)s.sw[corner] * values[j * FW + wave];
                const double phase = reference_mie
                    ? (reference_lut(plus, wave, tropo.data() + FW) +
                       reference_lut(minus, wave, tropo.data() + FW)) / 2.
                    : (reference_hg(plus, g) + reference_hg(minus, g)) / 2.;
                out[0][wave] += value * phase;
                out[1][wave] += value * (reference_lut(plus, wave, strato.data() + 33 * FW) +
                                        reference_lut(minus, wave, strato.data() + 33 * FW)) / 2.;
            }
        }
    }
}

int main() {
    static_assert(!TransportPolicy::ground_sh, "the disabled path is the subject");
    static_assert(TransportPolicy::ground_fused, "exercise the direct fused ground path");
    const int counts[][4] = {{0, 0, 0, 0}, {1, 7, 8, 9}, {31, 32, 33, 65}, {2, 17, 39, 64}};
    const float views[][3] = {{1.f, 0.f, 0.f}, {0.f, 0.6f, -0.8f}, {-0.6f, 0.64f, 0.48f}};
    std::vector<float> strato((33 + 16385) * FW), tropo((1 + 16385) * FW);
    for (int node = 0; node <= 16384; ++node) {
        const double x = -1. + node / 8192.;
        for (int wave = 0; wave < FW; ++wave) {
            strato[(33 + node) * FW + wave] = (float)((1. + wave * 0.07) * (0.3 + 0.2 * x + x * x));
            tropo[(1 + node) * FW + wave] = (float)((1. + wave * 0.11) * (0.7 - 0.3 * x + 0.2 * x * x));
        }
    }
    double max_relative = 0.;
    int checks = 0;
    for (int fixture = 0; fixture < 4; ++fixture) {
        std::vector<int64_t> offsets(5);
        for (int cell = 0; cell < 4; ++cell)
            offsets[cell + 1] = offsets[cell] + counts[fixture][cell];
        std::vector<float> directions(3 * offsets[4]), values(FW * offsets[4]);
        for (int64_t j = 0; j < offsets[4]; ++j) {
            double x = std::sin(j * 0.71), y = std::cos(j * 0.37), z = std::sin(j * 0.23);
            if (j % 11 == 0) { x = 0.; y = 0.; z = j % 2 ? 1. : -1.; }
            const double length = std::sqrt(x * x + y * y + z * z);
            directions[3 * j] = (float)(x / length);
            directions[3 * j + 1] = (float)(y / length);
            directions[3 * j + 2] = (float)(z / length);
            for (int wave = 0; wave < FW; ++wave)
                values[j * FW + wave] = (wave % 5 == 3) ? 0.f : (float)(0.01 * (1 + j % 19) * (wave + 1));
        }
        for (int view = 0; view < 3; ++view) {
            Sample sample{};
            sample.mu = views[view][0]; sample.exd = views[view][1]; sample.eyd = views[view][2];
            const float weights[4] = {0.125f, 0.25f, 0.375f, 0.25f};
            const int cells[4] = {3, 0, 2, 1};
            for (int corner = 0; corner < 4; ++corner) {
                sample.cell[corner] = cells[corner]; sample.sw[corner] = weights[corner];
                for (int wave = 0; wave < FW; ++wave) {
                    sample.c32[corner][wave] = -17.f;
                    sample.st_c32[corner][wave] = -19.f;
                }
            }
            const float g = view == 0 ? 0.9f : view == 1 ? 0.f : 0.65f;
            double expected[2][FW] = {};
            reference_ground(sample, offsets, directions, values, strato, tropo, g, expected);
            alignas(16) float y[YPAD + 16], ground[YPAD + 16];
            for (int i = 0; i < YPAD + 16; ++i) { y[i] = -23.f; ground[i] = -29.f; }
            std::vector<std::thread> lanes;
            for (int lane = 0; lane < 32; ++lane) {
                lanes.emplace_back([&, lane] {
                    host_lane = lane;
                    Transport<Sample> transport{};
                    transport.lane = lane;
                    transport.goff = offsets.data(); transport.gdir = directions.data();
                    transport.gval = values.data(); transport.st_chi = strato.data();
                    transport.asca = tropo.data(); transport.aerosol_g = g;
                    transport.ground_phase = ground;
                    transport.first_light_ground(sample, y);
                });
            }
            for (auto &lane : lanes) lane.join();
            for (int wave = 0; wave < FW; ++wave) {
                const float actual[2] = {sample.c32[0][wave], sample.st_c32[0][wave]};
                for (int component = 0; component < 2; ++component) {
                    const double ref = expected[component][wave];
                    const double error = std::fabs(actual[component] - ref);
                    const double relative = ref == 0. ? error : error / std::fabs(ref);
                    max_relative = std::fmax(max_relative, relative);
                    if (!std::isfinite(actual[component]) || error > 1e-3 * std::fabs(ref)) {
                        std::fprintf(stderr, "E_GROUND_RESULT fixture=%d view=%d component=%d wave=%d actual=%.9g expected=%.17g relative=%.9g\n",
                                     fixture, view, component, wave, actual[component], ref, relative);
                        return 17;
                    }
                    ++checks;
                }
                for (int corner = 1; corner < 4; ++corner)
                    if (sample.c32[corner][wave] != -17.f || sample.st_c32[corner][wave] != -19.f)
                        return 18;
            }
            for (int i = 0; i < YPAD + 16; ++i) {
                if ((!reference_uses_y || i >= 162) && y[i] != -23.f) return 19;
                if ((reference_uses_y || i >= 162) && ground[i] != -29.f) return 20;
            }
        }
    }
    std::printf("{\"checks\":%d,\"max_relative\":%.17g}\n", checks, max_relative);
    return 0;
}
'''


def host_warp_stub():
    """CUDA host stubs with a 32-lane barrier and shuffle-down exchange."""
    stub = STUB.replace('inline void __syncwarp(){}', 'inline void __syncwarp(){warp_sync();}')
    stub = stub.replace(
        'template<class T>T __shfl_down_sync(unsigned,T v,int){return v;}',
        'template<class T>T __shfl_down_sync(unsigned,T v,int d){return (T)warp_down((float)v,d);}',
    )
    extra = SYNTAX_STUB_EXTRA.replace(
        'inline int __float_as_int(float){return 0;} inline float __int_as_float(int){return 0;}',
        'inline int __float_as_int(float x){int y;std::memcpy(&y,&x,4);return y;} '
        'inline float __int_as_float(int x){float y;std::memcpy(&y,&x,4);return y;}',
    )
    return WARP + stub + extra


def check_ground(tmp_path, waves, phase, clocks, mie, kernel=None):
    """Compile/link and execute one disabled-SH variant; return measured error."""
    compiler = testkit.require_tool('clang++')
    source = tmp_path / 'ground.cpp'
    binary = tmp_path / 'ground'
    source.write_text(host_warp_stub() + POISON + f'\n#include "{kernel or paths.KERNEL_CU}"\n' + PROGRAM)
    flags = [f'-DFW={waves}', '-DSH_L=16', '-DSTRATO', '-DK_FIRST=1', f'-DK_PHASE={phase}', '-DK_BOUNDARY=0']
    if clocks:
        flags.append('-DEXP_SECTION_CLOCKS')
    if mie:
        flags.append('-DTROPO_MIE')
    cmd = [compiler, '-std=c++17', '-O2', '-ffp-contract=off', '-pthread', *flags, str(source), '-o', str(binary)]
    compiled = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=60)
    (tmp_path / 'compile.log').write_text(compiled.stdout + compiled.stderr)
    assert compiled.returncode == 0, 'E_GROUND_LINK: ' + compiled.stderr
    run = subprocess.run([str(binary)], text=True, capture_output=True, check=False, timeout=30)
    (tmp_path / 'run.log').write_text(run.stdout + run.stderr)
    assert run.returncode == 0, f'E_GROUND_RESULT: exit={run.returncode}\n{run.stdout}{run.stderr}'
    measured = json.loads(run.stdout)
    (tmp_path / 'result.json').write_text(json.dumps(measured, indent=2) + '\n')
    return measured
