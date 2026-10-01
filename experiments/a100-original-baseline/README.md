# Original Sol-Agent on A100: audited partial reproduction

Experiment dates: 2026-09-13–14. Publication: 2026-10-01.

The run restores the archived orchestration-lite Harness at
`d2c6407cc9b9133f3fff49fe4b561f14980d3f8b`, exposes eight A100-SXM4-40GB GPUs,
and uses original Master/Kernel/Cache workflow contracts with documented site
adaptations. A video request uses one GPU. The dense SANA-Video 2B contract is
832×480, 81 frames, 50 steps, guidance 6, seed 42 and five prompts. No GPU
experiment is executed by reading or verifying this snapshot.

| Result | Five-request sum | Ratio to frozen dense | Acceptance at latest checkpoint |
| --- | ---: | ---: | --- |
| Dense | 578.02 s | 1.000× | Frozen |
| Kernel R9 | 358.39 s | 1.6128× | Later native structural/method/objective check; no bitwise parity claim |
| Cache R05 | 371.36 s | 1.5565× | Fresh LPIPS + Master sampled-image check under lite contract |
| Cache R08 | 480.96 s | 1.2018× | Fresh LPIPS + Master sampled-image check under lite contract |
| Kernel+Cache | NOT_RUN | Unknown | No integrated delivery |

The ratio uses source-reported request time, excluding model load, internal
one-step warmup and video export. The current site adapter starts a separate
process for every prompt. These request measurements are not end-to-end
rollout latency. The Kernel contract permits reducing FP32 arithmetic to
BF16; “lossless” here is not a numerical proof for all inputs/seeds.

Read [Phase1](reports/final_reproduction.md) first for dense and Kernel search,
then [Phase2 terminal report](reports/phase3/phase2_terminal_partial.md) for
Cache continuation, later verification, per-card accounting and remaining
gates. The Phase1 verification failure and later Phase2 component checks are
different events. The overall result remains **PARTIAL**.

## Included evidence

- [Phase1 six-hour capacity](reports/full_gpu_window_six_hour_cap.json)
- [Phase2 six-hour capacity](artifacts/phase3/phase2_full_window_terminal/full_window_cap.json)
- [Agent CPU lower-bound summary](artifacts/phase3/agent_cpu_summary_master11_to13.json)
- [Frozen baseline](repo/output/orchestrated/sana2ba100-20260913-093521/BASELINE.json)
- [Kernel delivery](repo/output/experiments/sana2ba100-kernel_aw-0002/worktree/DELIVERY.json)
- [Cache delivery](repo/output/experiments/sana2ba100-cache_ca-0002/worktree/DELIVERY.json)
- [Independent reverify failure/recovery](reports/phase3/master_reverify_recovery_timeline.md)
- [Methodology limits](reports/methodology_boundaries.md)
- [Source adaptation snapshot](reproduction/README.md)

Small run/config/quality/receipt files are under `repo/output/` in the same
relative layout as the experiment. Videos, frames and raw session/GPU logs
remain local. Numeric JSON is copied without rewriting its raw blockers.
Historical report links to omitted evidence are marked accordingly.
