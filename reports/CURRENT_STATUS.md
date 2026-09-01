# Sol-Video-Agent SANA-Video 2B reproduction status

Status: `FP8_COMPONENT_VALIDATED_INTEGRATION_PARTIAL` over the prior
`COMPLETE_APPROVED_SCOPE` Kernel + Cache reproduction.

- Dense baseline: `VALIDATED`, 61.7 s on H100 UUID `GPU-847305ce-670b-91ee-e0a9-aa3b7833df23`.
- Kernel: `VALIDATED`, 23 rounds, stopped by the archived three-distinct-miss plateau rule; delivery `e8684f3fa9077d1387de44bbb0521a38ac6b7097`.
- Cache: `VALIDATED` under the approved lightweight protocol, 12 rounds, genuine convergence; balanced R12 selected; delivery `e7cf11c877a91220af2f2ea2cc5e38000c0765f8`.
- PISA: `NOT_APPLICABLE` for SANA linear attention/head dimension 112.
- Integration: composition `6d9ad0e984c6d236da7be3c3bc0e3d513f23173c`; closeout evidence `d33a58948e33dea0dacee361e5b50c004a0d8d65`.
- Final integrated prompt 1: 29.2 s versus 61.7 s dense, 2.1130x.
- Final integrated prompt 2: 33.0 s versus 60.5 s dense, 1.8333x.
- Both outputs: valid 832x480, 81 frames at 16 fps; first/middle/last visual screen passed.
- Final GPU audit: 0 MiB, 0% utilization, no compute applications or residual process, lock free; experiment lease released.

FP8 extension:

- New `fp8 -> quant_qe` executor workflow and SANA-specific W8A8 E4M3 runtime
  are wired and unit-tested; the autonomous executor search itself is `NOT_RUN`.
- 40/40 selected FFN pointwise projections executed with native FP8 weights on H100 SM90.
- The bounded joint `Kernel R20 + FP8 + EasyCache 0.13` candidate produced
  20.99 and 20.98 s steady requests versus the fresh BF16/cache-0.10 integrated
  control at 23.32 s, or 1.111x.
- Its denoise median was 17.5986 s versus 19.8883 s, or 1.130x.
- Both retained runs produced the same valid MP4 SHA-256; first/middle/last visual screen passed.
- These two speedups are joint FP8-plus-cache-retune gains, not pure FP8 gains.
  VBench, LPIPS, a second prompt, BF16/cache-0.13 counterfactual, isolated
  FP8-only end-to-end attribution, and official schema-v2 executor delivery are
  `NOT_RUN`.
- GPU 6 ended at 0 MiB with no compute apps; the FP8 experiment lease is released.

Canonical report:
`artifacts/integration/sana2b-kernel-cache-final/worktree/reports/FINAL-REPORT.md`.

FP8 canonical report: `reports/FP8-EXECUTOR-REPORT.md`.

Taxonomy: the prior Kernel/Cache baseline, trajectories, deliveries, integration,
and two-prompt sanity are `VALIDATED`; FP8 code/runtime activation/component
smoke are `VALIDATED`; FP8 direct integrated evaluation is `PARTIAL`; isolated
FP8 executor delivery, broad perceptual generalization, VBench, LPIPS, full paper
table, B200/B300, second model, and productionization are `NOT_RUN`.
