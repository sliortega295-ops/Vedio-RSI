# Vedio-RSI

Research artifacts for agent-driven video inference optimization. The intended
system combines faster rollout, experience transfer across hardware/models,
and retrieval of new kernels and methods. Current measured evidence concerns
rollout only.

## Current research snapshot — 2026-10-01

Start with the [research conclusions and evidence boundaries](reports/RESEARCH-STATUS-20261001.md).
The latest original Sol-Agent A100 reproduction is **PARTIAL**: Kernel R9 and
two Cache points received component checks, but their composed GPU run and
`INTEGRATED-DELIVERY` have not been completed.

| Evidence | Status | Entry point |
| --- | --- | --- |
| Original orchestration-lite Sol-Agent on 8 available A100s | Component checks completed; integration pending | [A100 reproduction](experiments/a100-original-baseline/README.md) |
| A100 GPU/CPU/stage timing | Two separate six-hour windows; limitations retained | [Phase1](experiments/a100-original-baseline/reports/final_reproduction.md), [Phase2](experiments/a100-original-baseline/reports/phase3/phase2_terminal_partial.md) |
| H100 scheduling pilot | Ten-candidate exploratory replay; old OptRoll did not beat FIFO | [Report and result](experiments/h100-rollout-pilot/README.md) |
| Cross-hardware/model memory and new-method retrieval | NOT_RUN | [Remaining evidence](reports/RESEARCH-STATUS-20261001.md#还缺什么) |

The snapshot contains small reports, numerical receipts, analysis scripts and
the A100 source adaptations. [Publication notes](PUBLICATION-NOTE.md) explain
omitted artifacts and historical status statements. [The manifest](PUBLICATION-MANIFEST.json)
records source and published hashes.

Validate the snapshot without network access or a GPU:

```bash
python3 tools/verify_publication.py
python3 experiments/a100-original-baseline/tests/test_phase2_full_window.py
```

## Earlier implementation branches

- [Recovered Harness and initial H100 search](https://github.com/sliortega295-ops/Vedio-RSI/tree/repro/sana-video-2b-full-exploration)
- [Kernel component](https://github.com/sliortega295-ops/Vedio-RSI/tree/repro/sana2b-kernel-executor)
- [Cache component](https://github.com/sliortega295-ops/Vedio-RSI/tree/repro/sana2b-cache-executor)
- [Earlier bounded H100 integration](https://github.com/sliortega295-ops/Vedio-RSI/tree/repro/sana2b-kernel-cache-integrated)
- [FP8 component](https://github.com/sliortega295-ops/Vedio-RSI/tree/repro/sana2b-fp8-executor)
- [Sol-RolloutBench v0 implementation](https://github.com/sliortega295-ops/Vedio-RSI/tree/benchmark/sol-rolloutbench-v0)

Earlier H100 integration results and the later A100 original-agent baseline
have different workloads, hardware and quality contracts. Their results are
not pooled.
