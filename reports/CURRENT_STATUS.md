# Sol-Video-Agent SANA-Video 2B reproduction status

Status: `COMPLETE_APPROVED_SCOPE`.

- Dense baseline: `VALIDATED`, 61.7 s on H100 UUID `GPU-847305ce-670b-91ee-e0a9-aa3b7833df23`.
- Kernel: `VALIDATED`, 23 rounds, stopped by the archived three-distinct-miss plateau rule; delivery `e8684f3fa9077d1387de44bbb0521a38ac6b7097`.
- Cache: `VALIDATED` under the approved lightweight protocol, 12 rounds, genuine convergence; balanced R12 selected; delivery `e7cf11c877a91220af2f2ea2cc5e38000c0765f8`.
- PISA: `NOT_APPLICABLE` for SANA linear attention/head dimension 112.
- Integration: composition `6d9ad0e984c6d236da7be3c3bc0e3d513f23173c`; closeout evidence `d33a58948e33dea0dacee361e5b50c004a0d8d65`.
- Final integrated prompt 1: 29.2 s versus 61.7 s dense, 2.1130x.
- Final integrated prompt 2: 33.0 s versus 60.5 s dense, 1.8333x.
- Both outputs: valid 832x480, 81 frames at 16 fps; first/middle/last visual screen passed.
- Final GPU audit: 0 MiB, 0% utilization, no compute applications or residual process, lock free; experiment lease released.

Canonical report:
`artifacts/integration/sana2b-kernel-cache-final/worktree/reports/FINAL-REPORT.md`.

Taxonomy: archival/compatibility, baseline, trajectories, deliveries, integration and two-prompt sanity are `VALIDATED`; broad perceptual generalization and the timing-scope caveat are `PARTIAL`; VBench, LPIPS, independent validator, full paper table, B200/B300, second model and productionization are `NOT_RUN`.
