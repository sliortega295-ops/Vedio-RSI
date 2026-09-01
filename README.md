# Sol-Video-Agent SANA-Video 2B reproduction

This branch contains the reproducible SANA-Video 2B H100 baseline, Kernel and
Cache trajectories, their integrated result, and a new quality-gated FP8
executor component/workflow modelled on the archival NVFP4 search role.

The FP8 component converts the two 1x1 FFN projections in all 20 transformer
blocks to real H100 W8A8 E4M3 execution. Its retained integrated candidate uses
EasyCache threshold 0.13 and reproduces 20.99/20.98-second warmed requests,
versus 23.32 seconds for the fresh BF16 integrated control. The result is
bounded to one prompt plus an identical confirmation. Because the retained
candidate also retunes EasyCache from 0.10 to 0.13, its latency gain is a joint
integration result rather than isolated FP8 attribution; the autonomous
FP8-only executor run is `NOT_RUN`.

- [FP8 implementation and H100 report](reports/FP8-EXECUTOR-REPORT.md)
- [Machine-readable direct integration packet](reports/fp8_executor/FP8-DELIVERY.json)
- [Original Kernel + Cache report](reports/FINAL-REPORT.md)
- [Current status](reports/CURRENT_STATUS.md)
