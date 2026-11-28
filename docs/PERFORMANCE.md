# Performance

## Release 1.10.0

On GPU A, a 10 GB NVIDIA GPU (described below), these times cover the whole state process on the full `state200` production grid with cache
hits.

| State | Stratospheric aerosol | 1.9.2 wall time | 1.10.0 wall time |
|---|---|---:|---:|
| `clear_33` | on | about 310 s | about 58–66 s |
| `clear_02` | off | 130–144 s | about 53 s |

`clear_80`, the largest state, also completed on the 10 GB card. Warm phase 0 makes fewer passes per transport point
and uses a centred quadratic angular B-spline lookup, while ground and phase 2 table launches keep the cubic lookup.
The GPU handles stratospheric first scattering and eligible Anderson history, and phase 2 accelerates table queries on
the GPU using angular tables and two-sample transport for table first scattering.

On the host, preparation of the next wavelength group overlaps the solve, workers receive file offsets for shared
geometry and reuse ray paths across wavelengths, and stratospheric columns can be prepared in parallel. The run-time C99
library for column integrals and its NumPy fallback agreed within 1e-12. [Tested configurations](TESTED.md) gives the
radiance comparisons.

On an Apple silicon GPU with experimental MLX, the small-grid stratospheric fixture FX-C3 went from about 29 s to about
6.7 s end to end, including about 0.5 s for queries in the faster run. The cache state for these timings is unknown.

## Earlier GPU measurements

GPU A, GPU B and GPU C denote three NVIDIA GPUs spanning multiple generations in the kernel measurements. Their power limits were
180–200 W, 72 W and 450 W, respectively. A separate survey of GPU A and two other GPUs supplied the cross-card scaling law.
The 1.2.0 stall probe in *Stall reasons* measured its `KSPLIT=0` path.
Values marked *estimate* were derived from those measurements.

## Release 1.9.0 on GPU A

Two states of the 200 km settings (spherical-harmonic degree 24, Anderson tolerance 2e-4, FMAD and FASTMATH off) were
solved with 1.8.0 and with 1.9.0 on GPU A through the same benchmark harness, with the same power and clock settings and
warm column and geometry caches. Times are the wall time of one state.

| state | stratosphere | 1.8.0 (s) | 1.9.0 (s) | ratio |
|---|---|---:|---:|---:|
| `clear_02` | off | 1,863 | 136.8 | 13.6 |
| `clear_33` | on | 1,239.6 | 277.5 | 4.5 |

In both states the evaluations per wavelength group and the per-wavelength convergence masks were those of the
reference build, and the tables agreed within about 1e-6 of each solar elevation's maximum. In `clear_33` the warm
phase 0 took 810 s of GPU time with 1.8.0 and 23 s with 1.9.0, all of it on the angular direction-table path; the first
scattering (about 126 s) and the Anderson step took most of the remaining source solve in 1.9.0.

## MLX transport on Apple silicon (1.9.0, experimental)

We measured this on an otherwise idle Apple silicon Mac with 16 GB of unified memory in the covered configuration: a
reduced NumPy `run_state` solve (two threads) of the check fixture state FX-N using its state and reduced-solve
settings from `checks/gate_data/fixtures.json` with the stratospheric layer disabled (AOD 0.02, black surface, 45
wavelengths in three chunks, 21 Anderson evaluations), run twice with `SOSFUSE2_MLX=1` and twice without.

| what | MLX | NumPy |
|---|---|---|
| whole run | 39–41 s | 79–80 s |
| source solve | 20–21 s | 61 s |
| of which transport passes | 0.1–0.2 s | 41 s |
| of which observer queries (NumPy in both runs) | about 20 s | about 20 s |

In 1.9.0, the observer queries after each evaluation and the geometry preparation still ran on the CPU, so the whole
run was about twice as fast. The full-grid MLX measurements for that release covered only transport passes.

## Release 1.6.1 on GPU A

Both releases ran the command under *Reproducing* for `clear_34` on GPU A, using the `split` device profile and the same
power and clock settings. Each cold run starts from an empty work root and empty caches, and the warm run repeats the
command on the same root. Times are the wall time of the command.

| release | cold runs (s) | warm runs (s) | split path (`ksplit_effective`) |
|---|---:|---:|---|
| 1.5.2 | 315.8, 315.3 | 219.5, 219.6 | 0, reason `state_minimum_exceeds_solo_budget` |
| 1.6.1 | 241.1, 240.7 | 156.2, 155.5 | 3, reason `fits` (the 24-byte column records of 1.6.0) |

1.6.1 is 1.31 times faster cold and 1.41 times faster warm on this state. The difference combines the transport kernel
changes of 1.6.0 with the split path that the smaller records enable on a 10 GB card. On this card 1.6.0 stopped at the
second chunk with `E_VRAM_FREE`; 1.6.1 releases the chunk's device temporaries before the next chunk.

The stall figures in *Stall reasons* were measured on earlier releases. Nsight Compute 2025.3.1 profiled the same
command with 1.6.1 on GPU A, filtered by kernel name to `vp_scatter`, `sos_vprod_y`, `sos_vprod` and `sos_slim_proj`,
and stopped after the first 40 matching launches, all in the first iteration of the solve. While the profiler replayed
each matching launch, a chunk took about 8 s, so the adaptive launch planner reduced the later `sos_vprod` chunks to one
block. The table leaves out those 30 one-block launches (SM 0.5 %) and gives the range over the launches kept. Stalls
are warps per issue-active cycle (`smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`).

| kernel | launches kept (grid blocks) | SM % | memory % | occupancy achieved / theoretical | registers | largest stall | next stalls |
|---|---|---:|---:|---|---:|---|---|
| `sos_vprod` | 4 (760–12,897) | 57.7–63.1 | 64.2–70.0 | 23.2–24.7 / 25.0 | 128 | `mio_throttle` 1.09–1.19 | `not_selected` 0.73–0.79, `wait` 0.60–0.61, `long_scoreboard` 0.53–0.59 |
| `sos_vprod_y` | 3 (1,314–25,251) | 73.2–84.7 | 31.8–37.1 | 79.5–97.2 / 100 | 28 | `tex_throttle` 8.75–12.34 | `short_scoreboard` 6.08–6.64, `wait` 1.55–1.58 |
| `vp_scatter` | 3 (127–1,015) | 8.5–14.7 | 30.3–49.2 | 3.8–14.3 / 14.6 | 39 | `short_scoreboard` 4.79–20.25 | `long_scoreboard` 4.01–8.92, `wait` 2.06–2.13 |
| `sos_slim_proj` | 1 (60) | 11.7 | 11.7 | 8.3 / 33.3 | 119 | `short_scoreboard` 2.93 | `barrier` 2.01, `wait` 1.45 |
| `sos_fused` slimray | 1 (6,566) | 51.8 | 37.9 | 24.6 / 25.0 | 147 | `wait` 1.25 | `long_scoreboard` 0.90, `not_selected` 0.73 |

Of the 20 stall metrics measured, the 17 that are neither `selected` nor a per-pipe sub-breakdown sum to 4.2–4.3 warps
per issue-active cycle in `sos_vprod`. Adding the issuing warp gives the measured 5.2–5.3 warp cycles per issued
instruction, so these reasons account for its stalled cycles. The pre-1.1.0 probe under *Where the time goes* placed
`sos_vprod` on the memory pipe (`mio_throttle` 0.83, L1 61 %). In 1.6.1 `mio_throttle` is its largest stall, with the
L1 cache at 69–71 % of its peak.

The last two rows come from two further runs of the same command, because the capture above reached its launch limit
before the first `sos_slim_proj` launch, and the slimray variant shares the name `sos_fused` with the first-light and
record variants, so a kernel-name filter alone cannot select it. Each run skipped the first matching launch, profiled
the next one and then ended the command. `sos_slim_proj` was selected by kernel name. The slimray launch was selected
by kernel name and by the NVTX range `p2min_slimray`, which the runner opens around each slimray launch when
`SOSFUSE2_P2MIN_NVTX=1` (`ncu --nvtx --nvtx-include 'p2min_slimray/'`). Both launches come early in the first
wavelength chunk, so each row is a single sample, and its grid is that launch's grid only.

## How the work is laid out

One state solve is a fixed-point iteration over a source field on a (height, χ) grid of 3,300 cells, with 45
wavelengths handled in groups of 15 (`FW=15`). Such a group is a chunk, called a bundle in some of the records,
and the three chunks converge separately; in the `clear_34` run of *Reproducing* with 1.9.0 they took 8, 6 and 6
evaluations.
A transport evaluation integrates the source along 12 Gauss–Legendre directions per polar interval for every cell, so it
walks about 2.4×10⁸ ray samples per chunk. With the pre-1.1.0 research kernel, one evaluation
on the production grid took 1.6–1.7 s in the reported probe. Full-grid solves took 53–57 s of kernel time in those
measurements.

The CUDA implementation centres on one fused kernel, `sos_fused` in `src/sosfuse2/backends/kernel.cu`, specialised into
variants at NVRTC compile time, with helper kernels around it. When the effective mode is `KSPLIT=3`, the solve uses
the following kernels during an evaluation.

| kernel | job | threads |
|---|---|---|
| `sos_fused` K_FIRST | first light: direct sun scattered once into every cell | one cell per block, one ray per warp |
| `sos_fused` RECORD | walks the rays once and writes the per-sample column records (24 B per sample point) | same |
| `vp_scatter`, `sos_vprod_y`, `sos_vprod` | aerosol phase-function value of every sample point and corner, written as one `float4` per thread | plan built by CUB sort/scan |
| `sos_fused` slimray (SLIM + SLIM_RAYS) | rebuilds attenuation from the column records and the path weight computed for each sample, reads the four corner source rows and the aerosol record, and accumulates the four Rayleigh tensor components and the aerosol Legendre moments | two ray steps per warp (half-warp `lane>>4`), wavelength `lane&15` (15 active lanes and one pad) |
| `sos_slim_proj` | projects the accumulated radiance onto the source basis | 32-ray tiles of 64 moments |

Outside the KSPLIT transport, eligible warm passes use the fast transport kernels instead, all with `sos_fast_slim`.
The direct path computes phase values with `sos_vprod_y`, `vp_c0cp_chunk` and `sos_vprod_direct` and projects with
`sos_slim_proj`; the angular path reads a direction table built by `sh_direction_table` and projects phase 0 with
`fl_project` (*Release 1.9.0 on GPU A*). First light can read a ground table built by `hg_direction_table`.

There are no floating-point atomics—the only atomics are integer counters that hand out rays and mark finished
cells—and the kernel accumulates in a fixed order in both modes. Because the transport path is chosen per launch,
`SOSFUSE2_DETERMINISTIC=1` is required to guarantee bit-stable repeats of the same run on the same GPU.

The first-light phase-0 variant of `sos_fused` carries `__launch_bounds__(256,2)`. In 1.0.x, the slimray variant
compiled to 127 registers with 128-thread blocks and 18–19 KB of shared memory, limiting residency to 4 blocks per SM
(16 warps, 33 % of maximum occupancy); with NO₂ and O₄ included, the 1.2.1 probe used 141 registers, limiting it to
3 blocks per SM and 25 % occupancy even though shared memory allowed 5 blocks per SM. CUDA transport stages and feature
policies live in `src/sosfuse2/backends/kernel.cu`, host launch planning in `src/sosfuse2/solve/source_solver.py`, and
the split path in `src/sosfuse2/solve/ksplit.py`.

`KSPLIT=3` needs the column records of a whole chunk in device memory. Each record now holds six floats (24 B),
one for each of the five component columns and one for the stratospheric column. The consumer computes the
quadrature path weight with the sample and carries it in the 112 B `PointM2` layout.

For an earlier memory fallback, see the 1.1.0 release. The figures below describe the kernel versions and paths
used in those runs.

## Where the time goes

On GPU A, Nsight Compute profiled transport evaluation 2 of the full-grid reference state using application replay
(36 metrics, 12 passes). Under the profiler the chunks were 0.56 of the production
size, so launch counts differ from production. The utilisation figures are per launch.

| kernel | share of the profiled kernel time | SM % | memory % | issue slot % | occupancy achieved / theoretical | registers | FP64 pipe % | limit if pushed to its own peak |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `sos_fused` slimray | 34 % | 61.4 | 35.4 | 63.8 | 32.0 / 33.3 | 127 | 39.7 | 1.57× (issue) |
| `sos_vprod` | 34 % | 53.5 | 58.9 | 52.3 | 24.3 / 25.0 | 128 | 0 | 1.70× (L1 61 %, LSU 55 %) |
| `sos_slim_proj` | 19 % | 5.3 | 5.3 | 10.7 | 8.3 / 33.3 | 119 | 8.6 | grid too small on this card only (3.6 waves on GPU B) |
| `sos_fused` phase-1 ground | 7 % | 6.9 | 6.9 | 21.2 | 8.3 / 25.0 | 162 | 19.2 | grid of 8–34 blocks on this card; 0.8 % of the time on GPU B |
| `vp_scatter` | 4 % | 12.1 | 41.1 | 7.5 | 13.3 / 14.6 | 39 | 0 | 2.4× |
| `sos_vprod_y` | 3 % | 80.4 | 35.4 | 47.9 | 90.6 / 100 | 28 | 82.4 | 1.24× (FP64 pipe) |

The table reports measurements of the research kernel before 1.1.0. See *Stall reasons* for the 1.2.1 measurements.

A cold and a warm solve with the pre-1.1.0 research kernel took 112.9 s of kernel time. The split
was `sos_fused` 60.6 %, `sos_vprod` 28.9 %, `sos_vprod_y` 2.7 %, an FP64 GEMM from the Anderson step 2.4 % and
`sos_slim_proj` 2.2 %.

In roofline terms the slimray consumer runs at 15.6 % of FP32 peak and 180 GB/s of DRAM traffic (24 % of GPU A's
peak). Its arithmetic intensity of 25.8 FLOP/B is in the model's bandwidth-limited region, below the ridge point of
39 FLOP/B, but the kernel runs far below both roofs. `sos_vprod` is on the memory pipe (`mio_throttle` 0.83, L1 61 %).
The FP64 pipe figure of slimray is not explained by its count of FP64 arithmetic instructions (DFMA, DADD, DMUL):
6.9×10⁷ warp-level FP64 arithmetic instructions against 1.9×10¹⁰ SM-cycles of pipe activity is about 280 cycles
per instruction where the hardware needs 16, and `math_pipe_throttle` is 0.11, so the pipe is busy with FP64
instructions we did not count, conversions and comparisons (*estimate*), and is not the bottleneck. Removing the FP64
from the inner geometry (see the table below) changed the slimray time by 0.4 %.

## Stall reasons

The `issue slot %` above is the fraction of cycles in which a scheduler issued an instruction.
In the remaining cycles every warp it owns was stalled. The largest *measured* stall reason in the earlier slimray
probes was `long_scoreboard`, and the probes below show how much of the stall budget those measurements left out.

| kernel stage | occupancy | issue active | dominant stall | other stalls | local memory |
|---|---|---|---|---|---|
| first fused kernel | 16.7 % (shared memory, 1 block/SM) | — | `long_scoreboard` 52 % | FP64 pipe 2.5 %, DRAM 0.15 % | spill 0 |
| slimray after the register work | 33.0 % (registers, 4 blocks/SM) | 57.2 % | `long_scoreboard` 2.85 warps per issue (about 41 % of stalled warps) | `short_scoreboard` 0.57, `mio_throttle` 0.07 | 6.06 M local load sectors, under 0.1 % of the instructions: not spills |
| slimray after two steps per warp | 33.1 % (128 registers, 4 blocks/SM) | 65.8 % | `long_scoreboard` 1.36 warps per issue-active cycle, largest of the four reasons measured | `short_scoreboard` 0.47, `math_pipe_throttle` 0.10, `mio_throttle` 0.012; threads per instruction 24.8 of 32 | spill 0 |
| release 1.2.1 on GPU B (KSPLIT=3) | 24.9 % (141 registers, 3 blocks/SM) | 52.2 % | `wait` 1.50 warps per issue-active cycle | `long_scoreboard` 1.19, `not_selected` 0.62, `short_scoreboard` 0.52, `no_instruction` 0.23, `math_pipe_throttle` 0.09; threads per instruction 24.7 of 32 | 18.4 M local-load sectors per 25 ms launch; 141 registers, so not read as spills |

The four reasons measured after two steps per warp sum to 1.95 of an estimated 5.0 stalled warps per issue, leaving about 3
in
unmeasured reasons such as `wait`, `not_selected`, `branch_resolving` and `no_instruction` (*estimate*).

`long_scoreboard` is the wait for a load through L1TEX—here global loads in slimray. The SASS shows why these waits
arise: every corner of a sample ends in `__ballot_sync` and `__syncwarp`, preventing the compiler from hoisting the next
corner's loads so each ray step waits for four or more serial L2 round trips. Moving the current step's own aerosol,
source-row and column-record loads before the corner loop (`SLIM_PRELOAD`) shortened slimray by 13 %. In a separate
GPU B knockout, folding every step's global-load addresses onto an L1-resident range (making values wrong and loads
effectively free) only reduced `long_scoreboard` from 0.476 to 0.239 warps per issue and launch time from 125.6 to
122.3 ms (−2.6 %). Hiding the remaining load latency is therefore worth at most about 3 %, and slimray is not
latency-bound.

Before the absorbers were added, we tried raising occupancy in every way the register budget allowed. A ray step at 15
wavelengths had a 127-register working set; forcing 5 blocks per SM through `__launch_bounds__(128,5)` reduced the
register count to 96 and gained 0.5 % of cycles.

The release 1.2.0 probe used Nsight Compute 2025.3.1 on GPU A, with the state runner (`jobs/run_state.py` since 1.4.0)
running the full-grid reference state through the `KSPLIT=0` fused kernel after `gate --gf` refused the
memory fallback.
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
The 1.2.0 `KSPLIT=0` fused kernel compiled to the hardware limit of 255 registers and generated 1.18×10⁸ local-load
sectors in a 1.8 s launch, evidence of register spilling. The earlier 127-register slimray variant had no spills.
In the `KSPLIT=0` PC samples, `wait` for fixed-latency dependencies and `long_scoreboard` each account for about a
quarter, with `selected` at about 17 % and not a stall, a pattern of a kernel constrained by dependencies and spills.
The largest SASS `long_scoreboard` count in the production captures was 187608 samples at
`FMUL.FTZ R129, R128, R132`, or 18.1 % of that launch's `long_scoreboard` samples and 5.11 % of all its PC samples.
This `FMUL` consumes a load result, and PC sampling attributes the wait to the consuming instruction.

On GPU B, the state runner (`jobs/run_state.py` since 1.4.0) in the installed 1.2.1 release solved the full-grid
reference state with `KSPLIT=3` under Nsight Compute, using separate one-pass stall-metric runs and one PC-sampling run.
The slimray launches use 128-thread blocks, 18.4 KB of dynamic shared memory and grids of 77k–81k blocks; 141 registers
limit residency to 3 blocks per SM while shared memory allows 5, giving 24.9 % occupancy, 52.2 % issue activity and
24.7 threads per instruction out of 32. The nine measured stall reasons—including `lg_throttle` 0.03, `mio_throttle` 0.02
and `barrier` 0—sum to 4.20 stalled warps per issue-active cycle against an estimated total of 4.73 from occupancy and
issue activity (explaining 89 %), with `long_scoreboard` taken from the earlier run. Although that earlier one-pass run
measured 18.4 M local-load sectors per 25 ms launch, 141 registers sits below the 255-register hardware limit, so this is
not read as evidence of spills.

As percentages of the reported PC-sampling category sum (including the non-stall `selected`), the breakdown gives `wait`
26.6 %, `long_scoreboard` 21.0 %, `selected` 17.7 %, `not_selected` 10.6 %, `short_scoreboard` 9.3 %,
`branch_resolving` 7.2 %, `dispatch_stall` 3.1 % and `tex_throttle` 2.0 %. The SASS instruction with the most
`long_scoreboard` samples is an `FFMA` consuming a load result, and the largest measured stall is the fixed-latency
dependency `wait` rather than `long_scoreboard`, consistent with the load knockout gaining only 2.6 %. Following the NO₂
and O₄ additions in 1.1.0, registers rose from 127 to 141 and residency fell from 4 blocks per SM to 3, leaving 3 warps
per scheduler at 25 % occupancy—too few to hide all fixed pipeline latency. Recovering the fourth block through
`__launch_bounds__` remains a candidate for a future experiment.

## Headroom

The five kernels `sos_fused` slimray, `sos_vprod`, `sos_slim_proj`, `vp_scatter` and `sos_vprod_y` account for 88.7 % of
evaluation kernel time in the pre-1.1.0 profile. Taking each to its own limiting unit at that workload shortens the
kernel time of an evaluation by 1.72× (1.50× if `sos_slim_proj` is already at its limit). Wall time of an evaluation
shortens by 1.36–1.49× with the same host gaps and by 1.9–2.2× if they were removed entirely (*estimate*). Across cards the kernel
time of a state follows t ∝ (SM count × clock)^−0.67 in the scaling survey, which implies about 1.6× when
doubling the SM count at the same clock.
The kernel figures above point to issue and latency as possible reasons for incomplete scaling, but the fit has three
anchor points and two parameters, mixes power regimes and `KSPLIT` paths, and includes tails from kernels with small
grids. The exponent is a clue, not evidence that establishes the bottleneck.

With two states overlapped, the pre-1.1.0 research kernel took 76 s of wall time per state of 25 solar elevations,
100 × 180 directions and 45 bands. The Monte Carlo alternative needed 17× to over 1,800× that time
for one solar elevation at the same colour noise (see `docs/VALIDATION.md`).
The estimates then put the remaining room to shorten the solve at 1.5–1.7×.

## What was tried

The *kept?* column records the outcome in the build that was measured. Release 1.9.0 replaced the double-single and FP64
rows below with FP32 device geometry and spherical harmonics and compensated FP32 projection sums.

In the pre-1.1.0 builds, the kernel work took the GPU time of the `gate --gf` run, a cold and a warm solve of the
full-grid reference state, from 151.0 to 106.9 s (−29 %), or 53.5 s per solve. In a separate run it cut time per state by 26 %.

| change | result | kept? |
|---|---|---|
| **Wavefront streaming consumer (`SLIM_STREAM`).** Split slimray into a geometry pass that materialises every sample (thread = sample) and a consumer with no lane communication (thread = ray × wavelength lane), the textbook un-fusing. | The micro-benchmark showed 3.3–5.4× per sample against a baseline of 6.49 ns/sample. Integrated into production and measured on three cards, the kernel time per sample went GPU C 0.55 → 1.00 ns, GPU B 2.44 → 2.84 ns, GPU A 2.21 → 1.94 ns, and solve wall time rose 23 % (GPU C), 8.6 % (GPU B) and 0–10 % (GPU A). **The baseline had been overstated 1.9–2.9×**: the micro-benchmark divided a state's slimray time by the samples of one chunk instead of three, and discarded the production timer as "asynchronous" when it was event-synchronised. The streaming formulation moves about 508 B per sample through DRAM against 276 B for the fused consumer (*estimate*) and cannot win even with zero launch gaps. | reverted |
| Two ray steps per warp in slimray (half-warps) | slimray −29 % (predicted −37 % from the static instruction count); issue active 77.7 → 65.8 %, with much of the remaining stall budget unmeasured | kept |
| Load hoisting before the corner loop (`SLIM_PRELOAD`) | slimray 41.5 → 36.0 s (−13 %), registers 107 → 117, spill 0 | kept |
| Knockout of every step global load onto L1-resident addresses (GPU B) | `long_scoreboard` 0.476 → 0.239 warps per issue, launch time −2.6 %. Hiding the remaining load latency is worth about 3 %; the kernel is not latency-bound. | measurement |
| Prefetch of the next ground direction (`GROUND_PREFETCH`) and 16-slot reduce-scatter instead of shuffles (`TRANSPORT_RS2`) | instructions −15 %, time −9 to −17 %; the 32-slot variant spilled 64 → 144 B and was dropped | kept |
| Register-resident ray constants and 128-bit record reads (`WARM_LANE`) | LSU instructions in the step loop 92 → 41 (−55 %), slimray −4.3 % for the first −27 %: the LSU count was not the limiter | kept |
| Replacing FP64 by double-single arithmetic in the point geometry (`PATH_DS`, first light and `vprod_y`) | first light −17 to −22 %, RECORD −38 %, `sos_vprod_y` 11.6 → 2.6 s; **slimray unchanged (−0.4 %)**: FP64 is not on its critical path even though the FP64 pipe reads 39 % busy | kept; replaced by FP32 in 1.9.0 |
| Projection kernel rebuilt around 32-ray tiles (`sos_slim_proj`, `fl_project`) | `sos_slim_proj` 14.0 → 2.4 s (−83 %), `fl_project` 5.1 → 1.6 s, GPU total −7.5 % | kept |
| Occupancy: global instead of shared FP64 accumulators (`ACC_GLOBAL`) | occupancy 16.7 → 24.5 %, IPC 0.58 → 0.81, first light −22 % | kept; the flag is retired in 1.9.0 |
| Occupancy: `__launch_bounds__(128,5)` on slimray (96 registers) | −0.5 % of cycles | rejected |
| Occupancy: 3 blocks per SM for the ground batch (`GROUND_BATCH_SHFL` + bounds) | slower | rejected |
| Plan sorting by coefficient, `vp_scatter` without scattering cells, `sos_vprod_y` in double-single | adopted with unchanged output; the `sos_vprod` shared-memory limit (30 KB, 3 blocks/SM) remains | kept; the double-single part replaced in 1.9.0 |
| Host: pinned allocation repeated per launch (about 11 s per state) | removed | fixed |
| Host launch issue cost of the plan kernels. A pre-1.1.0 run rebuilt the plan per chunk, with 12,612 small launches per evaluation. The gaps after them were 73 % of a 21 % host gap. | A run that stored the plan used 828 launches per evaluation and had a 1.3 % gap of the same class. CUDA graph capture and persistent kernels were considered for the case that rebuilt the plan. | not pursued |
| Host: a chunk switch in multi-chunk runs recomputed the stratospheric attach geometry every chunk (23.6–29.1 s), so the client CPU time of a state (103 s) exceeded its GPU time (55 s) and GPU utilisation of such runs on GPU B was 66 % | attach cache on and a process memo: 15.7 → 1.1 s per chunk on GPU A; the GPU B utilisation figure has not been re-measured since | fixed |
| Port to Apple silicon (MLX) | an experimental backend in 1.9.0 for a subset of configurations (`docs/ENVIRONMENT.md`) | experimental |

Two measurement traps shaped these numbers. First, a kernel that does more work per cycle lowers the boost clock on a
power-capped card, so a change that shortens one kernel can lengthen the untouched ones by 1–2 s. We judge changes by
the GPU total of a state and record the clock alongside it. Second, under Nsight Compute the resident state of the
persistent pipeline is 10–20 GB and multi-pass kernel replay fails on save and restore, so each stall metric has to be
collected alone in a one-pass run (`-k regex:^sos_fused$ --launch-skip N --launch-count 1`), or with application replay.

## Power and clocks

In a pre-1.1.0 run, the median SM clock ranged from 1,185 to 1,260 MHz against a maximum of 2,040 MHz, with 69 %
of the samples throttled. Another run ranged from 1,440 to 1,710 MHz depending on the kernel mix. The uncapped
slimray run took 0.55 ns per sample.
On a power-capped card, a kernel that issues more instructions per cycle therefore saves fewer seconds than its
instruction count suggests, so the table above quotes GPU totals.

## Reproducing

To reproduce the earlier `clear_34` runs on the default grid, prepare O₄ in the same asset cache following the
[quick start](../README.md#quick-start), then run

```sh
python -B -m sosfuse2.jobs.run_state --id clear_34 --backend cupy --work-root ./work
```

The state id is bundled in `assets/states_v2.json`; other ids from that file work the same way. The device profile is
chosen from the GPU name—add `--device-profile split` to select the profile of the measurements above on another card,
noting that launch-related environment variables still override its settings.

`meta.json` under `work/out/<id>/` records `elapsed_seconds`, `source_seconds`, `query_seconds` and, under
`kernel.launch.by_kind`, the launch count and GPU seconds (measured with CUDA events) for each kind of timed launch or
launch group in the last wavelength chunk. Launches are timed when the device profile sets a launch time limit, as both
CUDA profiles do by default, or when the launch size adapts. The tolerance checks of `docs/CHECKS.md` tier 2 run the
same kernels on the reduced fixtures; for profiling, run the same command under `nsys` for the timeline and under
`ncu --target-processes all` with one metric group per run. Use `ksplit_effective` in `meta.json` to identify the path
used in a run: with `KSPLIT=3`, look for `sos_fused` slimray launches, and inspect the `ksplit` decision for the reason
and named memory estimates. See [Device memory diagnostics](ENVIRONMENT.md#device-memory-diagnostics) for the fallback
warning and CuPy pool trace.

## The open question

The consumer kernel walks rays through a spherical shell, with 15 wavelengths per ray in the lanes of a half-warp and
two steps per warp. At each sample it interpolates, attenuates and accumulates using the column record, the sample's
path weight, four corner rows of the cell source and one aerosol record. In the 1.2.1 probe, a step needed 141 registers.
The kernel reached 25 % occupancy, issued on 52 % of cycles and had 24.7 active lanes out of 32.
The largest measured stall is `wait`, the fixed-latency dependency between consecutive instructions, with
`long_scoreboard` second. It is bound by neither DRAM nor FP32 throughput, nor by load latency: the load knockout gained
only 2.6 %. Moving the current step's own loads before the
corner loop bought 13 %, but the separate knockout bounds the benefit of hiding the remaining load latency at about
3 %. We therefore exclude interleaving two independent rays and lane-cooperative loads from the candidates.
Recovering the fourth block per SM lost in 1.1.0 is the one cheap experiment not yet run.

The static census counts about 660 warp instructions per ray step
for the corner and attenuation work, 570 for `corner_tensor` across four corners and 91 for attenuation. A hand-counted
arithmetic minimum is about 250, from 47 per corner × 4 and 62 for attenuation (*estimate*). The gap includes four
`logf` calls per corner, precise-division slow paths, convergence management through `__shfl_sync`, `__ballot_sync`
and `__syncwarp`, and padding that uses only 15 of 16 lanes.
The streaming formulation removed lane communication and was 3.3–5.4× faster per sample in the micro-benchmark,
whose baseline was overstated as described above, but doubled the bytes and lost in production.

Is there a layout for ray walking × 15 wavelengths that reduces lane communication and transcendental calls, lowering
the instruction count per sample without paying the streaming formulation's byte cost?
An issue with a sketch is welcome.
