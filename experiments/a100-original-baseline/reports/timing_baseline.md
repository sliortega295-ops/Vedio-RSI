# Fresh dense SANA-Video 2B A100 timing ledger

Status: **PASS as a frozen dense performance/reference-video baseline only**.
This is the site-adapted SGLang 2B workload, not the archived 5B/B200 workload
and not a paper-level quality claim. Collection is `completed`; baseline
LPIPS/visual judges were intentionally not run (`quality.status=blocked_quality`).

## Frozen denominator and structure

- Raw benchmark: `artifacts/baseline_dense_raw/benchmark.json`, SHA-256
  `88bb41c7b89141e2bf4a24d95b2b60822d07292c118934abcc4779538ede12c5`.
  The post-collection benchmark retains exact raw `total_s`, all five
  `per_prompt` rows, export/budget records, and `timing_scope`.
- Five sequential real requests: **115.99 + 115.50 + 115.50 + 115.50 +
  115.53 = 578.02 s**. This sums SGLang request metrics after internal
  warmup and before MP4 file save, not job wall. Source-reported denoise is
  546.3913 s and decode 28.1129 s. Each prompt has seed 42 and one distinct
  PID on the same authorized GPU UUID0. The five MP4 SHA-256 values match
  the raw benchmark; each ffprobe check gives 832×480, 81 frames, 16 fps.
  The export contains five videos and 80 selected baseline comparison frames
  (16 per prompt), not a single continuous 405-frame video.
- `run_config.json` records world size/allocated GPUs **1**, available pool
  **8**, 50 steps, BF16 DiT/FP32 VAE contract, SGLang source SHA-256
  `6a4c853e8f443734db72befdca0910b4cbf2ccf4a83792c7be03427a36522332`,
  `torch_fallback` A100 support backend, and compile/linattn-BF16/QKV-merge/
  EasyCache all OFF. The source/trace proof that the apparent `25/50` progress
  display is a second-order scheduler display rather than 25 denoise steps is
  in `reports/structural_step_audit.md`; 100 CFG DiT calls remain a
  source-derived expectation, not a directly observed counter.
- Frozen `BASELINE.json` for clean sequence 0002 is
  `repo/output/orchestrated/sana2ba100-20260913-093521/BASELINE.json`,
  SHA-256 `cdcf7036432bb60cc51df38d3a1f497cafdb7bdf0f4df32cd68f27cffcba7613`.
  It says `source=override_run_dir`, `total_s=578.02`, the exact request
  timing scope, and observed `world_size=1` from `run_config.json`.

## Independently timed host events and sampled resources

All event differences below use the *remote host's monotonic clock*.
UTC/host/UUID/PID are retained for joins. Remote `site_job.py` ran from
09:10:18.440 to 09:26:50.641 UTC, **992.206 s**; the inner SGLang runner's
`run_start→run_end` was **991.354 s**. The dense GPU queue lock/audit window
was **0.604 s**; five process allocation windows totaled **984.590 GPU-s**;
post-prompt export was **3.792 s**. The separate real-GPU import preflight
was **11.029 GPU-s** on UUID0 and started the fixed six-hour cap at
09:09:39.836 UTC (deadline 15:09:39.836 UTC).

| Dense prompt | Child PID | Process GPU-s | SGLang request s | Build/load window s | Generate-call window s | 1 Hz GPU-util weighted s | Sampled CPU user/system s, lower bound |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p01 | 2589180 | 202.673 | 115.99 | 49.890 | 133.971 | 125.73 | 511.87 / 1782.69 |
| p02 | 2591136 | 195.560 | 115.50 | 50.885 | 125.787 | 123.70 | 508.33 / 1882.69 |
| p03 | 2592928 | 195.426 | 115.50 | 51.045 | 125.613 | 124.04 | 509.51 / 1855.64 |
| p04 | 2594486 | 195.516 | 115.50 | 51.578 | 125.660 | 125.85 | 514.11 / 1935.19 |
| p05 | 2596042 | 195.414 | 115.53 | 51.026 | 125.821 | 123.58 | 526.28 / 1865.02 |

The exact independent phase windows summed over the five processes are
Python startup→stage-1 marker **48.751 s**, stage-3→stage-4 build including
model load **254.424 s**, stage-4→`GENERATE_OK` **636.851 s**, and
post-generate→process exit **43.611 s**. The stage-4 window includes the
internal one-step warmup and file save; it is *not* the denominator.
SGLang's own warmup reports 16.85/9.07/9.02/9.10/9.09 s (sum 53.13 s),
but those are source-reported durations, not separately monotonic-instrumented
subwindows. Its source-reported denoise/decode durations are also kept
separate from independent markers. Compile is OFF, so no compile time can be
claimed. The exact model-load-only subwindow within build is unavailable.

During the five child windows, sampled descendant-process CPU was at least
**2570.10 user + 9321.23 system CPU-s**; observed peak process RSS was
13.56 GB in p01, and sampled GPU memory peaked at **23,466 MiB**. These CPU
and `/proc` I/O deltas are lower bounds because short-lived descendants can
fall between samples; no precise CPU billing is inferred.

| GPU index / fixed UUID suffix | Allocated dense GPU-s | 1 Hz samples in runner window | Mean NVML GPU util | Util-weighted s estimate | Peak mem MiB | Compute apps |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 / `cad55364` | 984.590 | 992 | 62.79% | 622.90 | 23,466 | 5 SGLang worker PIDs |
| 1 / `a904c6f6` | 0 | 992 | 0% | 0 | 0 | none |
| 2 / `79ab3e5e` | 0 | 992 | 0% | 0 | 0 | none |
| 3 / `bab43db0` | 0 | 992 | 0% | 0 | 0 | none |
| 4 / `44bddf94` | 0 | 992 | 0% | 0 | 0 | none |
| 5 / `af8dba1f` | 0 | 992 | 0% | 0 | 0 | none |
| 6 / `5edaa614` | 0 | 992 | 0% | 0 | 0 | none |
| 7 / `01fff32a` | 0 | 992 | 0% | 0 | 0 | none |

GPU0 mean power during the runner window was 258.40 W; the seven unallocated
cards averaged roughly 53–58 W. The latter are **structurally unallocated by
the single-video, sequential five-prompt baseline policy**, not failed
Executor scheduling. GPU0's low-util startup/build intervals are chiefly
site-adapter process/model reload overhead: the archived Sana 5B wrapper
bundled samples in one process, while this SGLang adapter restarts for each
prompt. Do not count this as Sol-Agent's inherent rollout scheduling cost.
The trace does not separately identify every avoidable idle cause.

The passive observer started at 08:40:42.970 UTC, before any GPU job. The
baseline snapshot has **2,978 one-second rows**, 8/8 fixed UUIDs in every
row, zero error/incomplete rows and no missing job end event. The 09:27 UTC
audit had zero gaps >1.5 s through baseline; two later <=1.857 s gaps occurred
while no candidate job existed and are separately tracked. At 09:27 UTC the
observer's own mean collection duration was 0.3615 s/sample and CPU
169.07 s over 2779.0 s (~6.08% of one core). This quantifies observer work,
*not* an A/B estimate of perturbation to generation. NVML's 1 Hz utilization
samples are neither exact kernel-active time nor SM occupancy.

Machine-readable per-job/per-phase/UUID data and the raw, hash-checked remote
event/sampling snapshot are `reports/timing_baseline.json` and
`telemetry/baseline_remote_snapshot/`. Future Codex `agent_attempt_*.timed.jsonl`
boundaries will be classified as observed API/tool events or inferred gaps,
never exact private thinking time.
