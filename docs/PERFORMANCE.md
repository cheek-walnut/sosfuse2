# Performance

Over five weeks of GPU kernel work, we measured where solve time goes, how close the kernels get to hardware limits and what
each change achieved. One question remains open. The solver is fast enough for its purpose, as discussed under
*Headroom*, but its main kernel has never been bound by memory bandwidth or by arithmetic throughput, and a month of
profiling kept finding `long_scoreboard` as its largest measured stall reason without accounting for all stalled cycles.
The last section asks how to reduce the work per sample.

The measurements below were made in the research repository that this package was split from, on three cards: a GeForce
RTX 3080 (10 GB, power limit 180–200 W, the development host), an NVIDIA L4 (23 GB, 72 W, the production card) and a
GeForce RTX 4090 (24 GB, 450 W, rented). Only the cross-card scaling law comes from the 2026-10 survey of the
RTX 3060, 3080 and 5090. The dates identify the kernel used for each measurement.
The kernel changed little after 2026-11-03, and the 1.2.0 stall probe in *Stall reasons* measured its `KSPLIT=0` path.
Values marked *estimate* were derived from those measurements.

## How the work is laid out

One state solve is a fixed-point iteration over a source field on a (height, χ) grid of 3,300 cells, with 45
wavelengths handled in groups of 15 (`FW=15`). Such a group is a chunk, called a bundle in some of the records,
and the three chunks each converge separately in 7–8 transport evaluations.
A transport evaluation integrates the source along 12 Gauss–Legendre directions per polar interval for every cell, so it
walks about 2.4×10⁸ ray samples per chunk (`results/wavefront_20261101/REPORT_census.md`). One evaluation on the
production grid takes 1.6–1.7 s on the RTX 3080 and a full state takes 53–57 s of kernel time per solve on either
the 3080 or the L4.

The CUDA implementation has one fused kernel, `sos_fused` in `src/sosfuse2/kernel.cu`, specialised into variants at
NVRTC compile time, and four helpers. With the production configuration `KSPLIT=3`, each chunk of rays goes through
the following kernels during an evaluation.

| kernel | job | threads |
|---|---|---|
| `sos_fused` K_FIRST | first light: direct sun scattered once into every cell | one cell per block, one ray per warp |
| `sos_fused` RECORD | walks the rays once and writes the per-sample column records (28 B per sample point) | same |
| `vp_scatter`, `sos_vprod_y`, `sos_vprod` | aerosol phase-function value of every sample point and corner, written as one `float4` per thread | plan built by CUB sort/scan |
| `sos_fused` slimray (SLIM + SLIM_RAYS) | the consumer: rebuilds attenuation from the column records, reads the four corner source rows and the aerosol record, accumulates the four Rayleigh tensor components and the aerosol Legendre moments | two ray steps per warp (half-warp `lane>>4`), wavelength `lane&15` (15 active lanes and one pad) |
| `sos_slim_proj` | projects the accumulated radiance onto the source basis | 32-ray tiles of 64 moments |

There are no global atomics, and the current kernel accumulates in a fixed order in both modes;
`SOSFUSE2_DETERMINISTIC=1` is the mode in which the package guarantees bit-stable repeats. `sos_fused` carries
`__launch_bounds__(256,2)`. The slimray variant compiles to 127 registers with 128-thread blocks and 18–19 KB of shared
memory. Registers limit it to 4 blocks per SM, or 16 warps and 33 % of maximum occupancy. Shared memory would allow 5.
The kernel is 2,200 lines of CUDA C, with host launch planning in `src/sosfuse2/fused.py`.

`KSPLIT=3` needs the column records of a whole bundle in device memory. Adding NO₂ and O₄ in 1.1.0 grew the records
to 28 B per sample point from 20 B. The product state's minimum peak of 7.06 GB exceeds the 6.7 GB solo budget of a
10 GB card, so the runner falls back to `KSPLIT=0`, in which one `sos_fused` launch does the whole transport step.
The field `ksplit_effective` in `meta.json` records this choice. The L4 and the 4090 run `KSPLIT=3`.
The figures below identify the path measured.

## Where the time goes

RTX 3080, G-F product state, transport evaluation 2 under Nsight Compute application replay (36 metrics, 12 passes,
2026-11-02; `results/speed_bench_20261102/REPORT.md` §5). Under the profiler the chunks were 0.56 of the production
size, so launch counts differ from production. The utilisation figures are per launch.

| kernel | share of the profiled kernel time | SM % | memory % | issue slot % | occupancy achieved / theoretical | registers | FP64 pipe % | limit if pushed to its own peak |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `sos_fused` slimray | 34 % | 61.4 | 35.4 | 63.8 | 32.0 / 33.3 | 127 | 39.7 | 1.57× (issue) |
| `sos_vprod` | 34 % | 53.5 | 58.9 | 52.3 | 24.3 / 25.0 | 128 | 0 | 1.70× (L1 61 %, LSU 55 %) |
| `sos_slim_proj` | 19 % | 5.3 | 5.3 | 10.7 | 8.3 / 33.3 | 119 | 8.6 | grid too small on this card only (3.6 waves on the L4) |
| `sos_fused` phase-1 ground | 7 % | 6.9 | 6.9 | 21.2 | 8.3 / 25.0 | 162 | 19.2 | grid of 8–34 blocks on this card; 0.8 % of the time on the L4 |
| `vp_scatter` | 4 % | 12.1 | 41.1 | 7.5 | 13.3 / 14.6 | 39 | 0 | 2.4× |
| `sos_vprod_y` | 3 % | 80.4 | 35.4 | 47.9 | 90.6 / 100 | 28 | 82.4 | 1.24× (FP64 pipe) |

On the L4 the split of the 112.9 s of kernel time of a cold and a warm solve is `sos_fused` 60.6 %, `sos_vprod`
28.9 %, `sos_vprod_y` 2.7 %, an FP64 GEMM from the Anderson step 2.4 % and `sos_slim_proj` 2.2 %
(`results/cardperf_l4_20261103/REPORT.md` §3).

In roofline terms the slimray consumer runs at 15.6 % of FP32 peak and 180 GB/s of DRAM traffic (24 % of the 3080's
peak). Its arithmetic intensity of 25.8 FLOP/B is in the model's bandwidth-limited region, below the ridge point of
39 FLOP/B, but the kernel runs far below both roofs. `sos_vprod` is on the memory pipe (`mio_throttle` 0.83, L1 61 %).
The FP64 pipe figure of slimray is not explained by its count of FP64 arithmetic instructions (DFMA, DADD, DMUL):
6.9×10⁷ warp-level FP64 arithmetic instructions against 1.9×10¹⁰ SM-cycles of pipe activity is about 280 cycles
per instruction where the hardware needs 16, and `math_pipe_throttle` is 0.11, so the pipe is busy with FP64
instructions we did not count, conversions and comparisons (*estimate*), and is not the bottleneck. Removing the FP64
from the inner geometry (see the table below) changed the slimray time by 0.4 %.

## Stall reasons

The `issue slot %` above is the fraction of cycles in which a scheduler issued an instruction.
In the remaining cycles every warp it owns was stalled. The largest *measured* stall reason has been
`long_scoreboard` at every stage where it was measured, and the probe below shows how much of the stall budget
those measurements left out.

| date, kernel state | occupancy | issue active | dominant stall | other stalls | local memory |
|---|---|---|---|---|---|
| 2026-10-03, first fused kernel | 16.7 % (shared memory, 1 block/SM) | — | `long_scoreboard` 52 % | FP64 pipe 2.5 %, DRAM 0.15 % | spill 0 |
| 2026-10-31, slimray after the register work | 33.0 % (registers, 4 blocks/SM) | 57.2 % | `long_scoreboard` 2.85 warps per issue (about 41 % of stalled warps) | `short_scoreboard` 0.57, `mio_throttle` 0.07 | 6.06 M local load sectors, under 0.1 % of the instructions: not spills |
| 2026-11-01, slimray after two steps per warp | 33.1 % (128 registers, 4 blocks/SM) | 65.8 % | `long_scoreboard` 1.36 warps per issue-active cycle, largest of the four reasons measured | `short_scoreboard` 0.47, `math_pipe_throttle` 0.10, `mio_throttle` 0.012; threads per instruction 24.8 of 32 | spill 0 |
| 2026-11-06, slimray of release 1.2.0 | not measured | not measured | not measured | not measured | not measured: no 127-register, 128-thread launch captured |

The four reasons measured on 2026-11-01 sum to 1.95 of an estimated 5.0 stalled warps per issue, leaving about 3 in
unmeasured reasons such as `wait`, `not_selected`, `branch_resolving` and `no_instruction` (*estimate*).

`long_scoreboard` is the wait for a load through L1TEX, here global loads in slimray. The SASS shows why these waits arise:
every corner of a sample ends in `__ballot_sync` and `__syncwarp`, so the compiler cannot hoist the next corner's loads and a
ray step waits for four or more serial L2 round trips. Moving the current step's own aerosol, source-row and column-record
loads before the corner loop (`SLIM_PRELOAD`) shortened slimray by 13 %.
In a separate L4 knockout on 2026-10-31, folding every step's global-load addresses onto an L1-resident range made the
values wrong and the loads effectively free, but only reduced `long_scoreboard` from 0.476 to 0.239 warps per issue
and launch time from 125.6 to 122.3 ms (−2.6 %).
Hiding the remaining load latency is therefore worth at most about 3 %, and slimray is not latency-bound.
We tried raising occupancy in every form
the register budget allows. A ray step at 15 wavelengths has a 127-register working set.
Forcing 5 blocks per SM through `__launch_bounds__(128,5)` reduced the register count to 96 and gained 0.5 % of cycles.

The release 1.2.0 probe used Nsight Compute 2025.3.1 on the RTX 3080 on 2026-11-06, with `run_gpu_state_v3` running
the full G-F state through the `KSPLIT=0` fused kernel after `gate --gf` refused the memory fallback.
All four runs succeeded, but the launches captured from the small `gate --all` fixtures had 256-thread blocks and
128 or 251 registers, so they did not measure the 127-register, 128-thread slimray variant.
Timing used one pass per run, while PC sampling reports 2 aggregated passes per launch, and the production grids
differed between the timing and sampling runs.

| timing run | grid blocks | threads per block | registers per thread | duration | `long_scoreboard` per issue active | local-load sectors |
|---|---:|---:|---:|---|---:|---:|
| production | 1894 | 128 | 255 | 1.820608 s | 1.973504 | 118332756 |
| production | 648 | 128 | 255 | 0.659964 s | 1.760762 | 45947466 |
| production | 2 | 128 | 251 | 0.006786 s | 1.441187 | 6912 |
| fixtures, four launches | 1 | 256 | 128 | 733.344–816.608 µs | 0.424129–0.528831 | 3447–3987 |
| fixtures, four launches | 2 | 256 | 251 | 230.976–244.928 µs | 0.523260–0.610426 | 567–657 |

| PC sampling run | grid blocks | threads per block | registers per thread | most frequent category | second | third |
|---|---:|---:|---:|---|---|---|
| production | 1874 | 128 | 255 | `wait` 27.2 % | `long_scoreboard` 26.4 % | `selected` 17.4 % |
| production | 672 | 128 | 255 | `long_scoreboard` 29.0 % | `wait` 26.2 % | `selected` 16.8 % |
| production | 508 | 128 | 255 | `long_scoreboard` 28.7 % | `wait` 25.9 % | `selected` 16.8 % |
| fixtures | 1 | 256 | 128 | `short_scoreboard` 32.6 % | `wait` 28.1 % | `selected` 17.2 % |
| fixtures | 2 | 256 | 251 | `wait` 30.3 % | `short_scoreboard` 22.0 % | `selected` 20.1 % |
| fixtures | 1 | 256 | 128 | `short_scoreboard` 32.3 % | `wait` 28.8 % | `selected` 17.2 % |
| fixtures | 2 | 256 | 251 | `wait` 27.9 % | `short_scoreboard` 20.8 % | `selected` 20.5 % |
| fixtures | 1 | 256 | 128 | `short_scoreboard` 32.0 % | `wait` 28.6 % | `selected` 15.9 % |
| fixtures | 2 | 256 | 251 | `short_scoreboard` 25.3 % | `wait` 23.2 % | `selected` 19.1 % |
| fixtures | 1 | 256 | 128 | `short_scoreboard` 30.4 % | `wait` 26.5 % | `selected` 20.1 % |
| fixtures | 2 | 256 | 251 | `wait` 28.4 % | `selected` 22.3 % | `short_scoreboard` 18.6 % |

The first table gives stalled warps per issue-active cycle, while the second gives percentages of the sum of the
reported PC categories, including `selected`, which is not a stall.
The `KSPLIT=0` fused kernel compiles to the hardware limit of 255 registers and generates 1.18×10⁸ local-load sectors
in a 1.8 s launch, evidence of register spilling. This is the path that the 10 GB card actually runs, unlike the
127-register slimray variant with spill 0.
In the `KSPLIT=0` PC samples, `wait` for fixed-latency dependencies and `long_scoreboard` each account for about a
quarter, with `selected` at about 17 % and not a stall, a pattern of a kernel constrained by dependencies and spills.
The largest SASS `long_scoreboard` count in the production captures was 187608 samples at
`FMUL.FTZ R129, R128, R132`, or 18.1 % of that launch's `long_scoreboard` samples and 5.11 % of all its PC samples.
This `FMUL` consumes a load result, and PC sampling attributes the wait to the consuming instruction.
The probe does not establish slimray's current stall distribution: the 10 GB card no longer runs that variant,
slimray remains unmeasured in 1.2.0, and measuring it on the L4 remains to be done.

## Headroom

The five kernels `sos_fused` slimray, `sos_vprod`, `sos_slim_proj`, `vp_scatter` and `sos_vprod_y` account for 88.7 % of
evaluation kernel time. Taking each to its own limiting unit, with the work it does today, shortens the kernel time of
an evaluation by 1.72× (1.50× if `sos_slim_proj` is already at its limit). Wall time of an evaluation moves by 1.36–1.49×
with the host gaps left as they are, and by 1.9–2.2× if they were removed entirely (*estimate*). Across cards the kernel
time of a state follows t ∝ (SM count × clock)^−0.67 over three anchor points (RTX 3060, 3080 and 5090,
`results/sosfuse2_r17/gpu_survey/`), so doubling the SM count at the same clock returns about 1.6×.
The kernel figures above point to issue and latency as possible reasons for incomplete scaling, but the fit has three
anchor points and two parameters, mixes power regimes and `KSPLIT` paths, and includes tails from kernels with small
grids. The exponent is a clue, not evidence that establishes the bottleneck.

One state of 25 solar elevations, 100 × 180 directions and 45 bands takes
76 s of wall time per state on an L4 with two states overlapped, and the Monte Carlo alternative needs 17× to over 1,800×
that for one solar elevation at the same colour noise (`docs/VALIDATION.md`, `results/speed_bench_20261102/REPORT.md`
§3). The remaining 1.5–1.7× would reduce the cost of running the solver.

## What was tried

The series `refactor/kernel-accuracy-perf` of 2026-10-29 to 11-01 (57 commits) took the GPU time of the `gate --gf`
run, a cold and a warm solve of the product state, on the RTX 3080 from 151.0 to 106.9 s (−29 %), that is 53.5 s per
solve, and cut the L4 time per state by 26 %. The table lists the changes and the dead ends, the largest lesson first.

| change | result | kept? |
|---|---|---|
| **Wavefront streaming consumer (`SLIM_STREAM`, 2026-11-01 to 11-03).** Split slimray into a geometry pass that materialises every sample (thread = sample) and a consumer with no lane communication (thread = ray × wavelength lane), the textbook un-fusing. | The micro-benchmark showed 3.3–5.4× per sample against a baseline of 6.49 ns/sample. Integrated into production and measured on three cards, the kernel time per sample went 4090 0.55 → 1.00 ns, L4 2.44 → 2.84 ns, 3080 2.21 → 1.94 ns, and solve wall time rose 23 % (4090), 8.6 % (L4) and 0–10 % (3080). **The baseline had been overstated 1.9–2.9×**: the micro-benchmark divided a state's slimray time by the samples of one bundle instead of three, and discarded the production timer as "asynchronous" when it was event-synchronised. The streaming formulation moves about 508 B per sample through DRAM against 276 B for the fused consumer (*estimate*) and cannot win even with zero launch gaps. | reverted |
| Two ray steps per warp in slimray (`P3′`, half-warps), 2026-11-01 | slimray −29 % (predicted −37 % from the static instruction count); issue active 77.7 → 65.8 %, with much of the remaining stall budget unmeasured | kept |
| Load hoisting before the corner loop (`SLIM_PRELOAD`), 2026-10-31 | slimray 41.5 → 36.0 s (−13 %), registers 107 → 117, spill 0 | kept |
| Knockout of every step global load onto L1-resident addresses (L4, 2026-10-31) | `long_scoreboard` 0.476 → 0.239 warps per issue, launch time −2.6 %. Hiding the remaining load latency is worth about 3 %; the kernel is not latency-bound. | measurement |
| Prefetch of the next ground direction (`GROUND_PREFETCH`) and 16-slot reduce-scatter instead of shuffles (`TRANSPORT_RS2`), 2026-10-03 | instructions −15 %, time −9 to −17 %; the 32-slot variant spilled 64 → 144 B and was dropped | kept |
| Register-resident ray constants and 128-bit record reads (`WARM_LANE`, `speculative D`), 2026-11-01 | LSU instructions in the step loop 92 → 41 (−55 %), slimray −4.3 % for the first −27 %: the LSU count was not the limiter | kept |
| Replacing FP64 by double-single arithmetic in the point geometry (`PATH_DS`, first light and `vprod_y`), 2026-10-31 | first light −17 to −22 %, RECORD −38 %, `sos_vprod_y` 11.6 → 2.6 s; **slimray unchanged (−0.4 %)**: FP64 is not on its critical path even though the FP64 pipe reads 39 % busy | kept |
| Projection kernel rebuilt around 32-ray tiles (`sos_slim_proj`, `fl_project`), 2026-10-31 | `sos_slim_proj` 14.0 → 2.4 s (−83 %), `fl_project` 5.1 → 1.6 s, GPU total −7.5 % | kept |
| Occupancy: global instead of shared FP64 accumulators (`ACC_GLOBAL`), 2026-10-03 | occupancy 16.7 → 24.5 %, IPC 0.58 → 0.81, first light −22 % | kept |
| Occupancy: `__launch_bounds__(128,5)` on slimray (96 registers) | −0.5 % of cycles | rejected |
| Occupancy: 3 blocks per SM for the ground batch (`GROUND_BATCH_SHFL` + bounds) | slower | rejected |
| Plan sorting by coefficient, `vp_scatter` without scattering cells, `sos_vprod_y` in double-single, 2026-10-31 | adopted on bit-identical output; the `sos_vprod` shared-memory limit (30 KB, 3 blocks/SM) remains | kept |
| Host: pinned allocation repeated per launch (about 11 s per state) | removed | fixed |
| Host: launch issue cost of the plan kernels. On the 10 GB card the plan is rebuilt per chunk because of memory, 12,612 small launches per evaluation, and the gaps after them are 73 % of a 21 % host gap | on the 23 GB L4 the plan is stored, 828 launches per evaluation, and the same gap class is 1.3 %. CUDA graph capture and persistent kernels were considered for the 10 GB case and **not tried**, because the production card does not have the problem | not pursued |
| Host: bundle switch in production bakes recomputed the stratospheric attach geometry every bundle (23.6–29.1 s), so the client CPU time of a state (103 s) exceeded its GPU time (55 s) and campaign GPU utilisation on the L4 was 66 % | attach cache on and a process memo: 15.7 → 1.1 s per bundle on the 3080 (2026-11-03); the L4 campaign figure has not been re-measured since | fixed |
| Port to Apple Metal (MLX) | examined, folded | not pursued |

Two measurement traps shaped these numbers. First, a kernel that does more work per cycle lowers the boost clock on a
power-capped card, so a change that shortens one kernel can lengthen the untouched ones by 1–2 s. We judge changes by
the GPU total of a state and record the clock alongside it. Second, under Nsight Compute the resident state of the
persistent pipeline is 10–20 GB and multi-pass kernel replay fails on save and restore, so each stall metric has to be
collected alone in a one-pass run (`-k regex:^sos_fused$ --launch-skip N --launch-count 1`), or with application replay.

## Power and energy

The L4 runs this workload at a median SM clock of 1,185–1,260 MHz against a maximum of 2,040, with 69 % of the samples
throttled at its 72 W cap; a state costs 4.3 kJ with two states overlapped. The RTX 3080 at 180 W sits at 1,440–1,710 MHz
depending on the kernel mix, and the RTX 4090 at 337 W is not capped and runs slimray at 0.55 ns per sample, 4.4× the L4.
A kernel that issues more instructions per cycle therefore buys fewer seconds than its instruction count suggests on the
production card, which is why the table above quotes GPU totals and why we record joules per state as a performance
figure next to seconds.

## Reproducing

A full-resolution state on a CUDA machine:

```sh
python -B -m sosfuse2.run_gpu_state_v3 --id L19s3_tau0p01_aod0p4_g0p65 --backend cupy --work-root ./work
```

The state id is the G-F product state of `gate/fixtures.json`; any id from `assets/states_v2.json` works the same way.
`meta.json` under `work/out/<id>/` records `elapsed_seconds`, `source_seconds`, `query_seconds` and, under
`kernel.launch.by_kind`, the launch count and GPU seconds of every kernel variant, measured with CUDA events. The
tolerance checks of `docs/CHECKS.md` tier 2 run the same kernels on the reduced fixtures. For profiling, run the same
command under `nsys` for the timeline and under `ncu --target-processes all` with one metric group per run.
On cards with at least 23 GB, the state runs with `KSPLIT=3` and the launches of interest are `sos_fused` slimray
with 127 registers and 128-thread blocks. On a 10 GB card it falls back to `KSPLIT=0`, whose `sos_fused` launches
use 255 registers and 128-thread blocks.

## The open question

The consumer kernel walks rays through a spherical shell, with 15 wavelengths per ray in the lanes of a half-warp and
two steps per warp. At each sample it needs the column record of the step, the four corner rows of the cell source and
one aerosol record, interpolates, attenuates and accumulates. It sits at 33 % occupancy because a step needs 127
registers, issues on 65 % of the cycles and has 24.8 active lanes out of 32. It is bound by neither DRAM nor FP32
throughput, nor by load latency: the load knockout gained only 2.6 %. Moving the current step's own loads before the
corner loop bought 13 %, but the separate knockout bounds the benefit of hiding the remaining load latency at about
3 %. We therefore exclude interleaving two independent rays and lane-cooperative loads from the candidates.

The static census (`results/wavefront_20261101/REPORT_census.md`) counts about 660 warp instructions per ray step
for the corner and attenuation work, 570 for `corner_tensor` across four corners and 91 for attenuation. A hand-counted
arithmetic minimum is about 250, from 47 per corner × 4 and 62 for attenuation (*estimate*). The gap includes four
`logf` calls per corner, precise-division slow paths, convergence management through `__shfl_sync`, `__ballot_sync`
and `__syncwarp`, and padding that uses only 15 of 16 lanes.
The streaming formulation removed lane communication and was 3.3–5.4× faster per sample in the micro-benchmark,
whose baseline was overstated as described above, but doubled the bytes and lost in production.

Is there a layout for ray walking × 15 wavelengths that reduces lane communication and transcendental calls, lowering
the instruction count per sample without paying the streaming formulation's byte cost?
An issue with a sketch is welcome. The research records named in this page can be shared on request.
