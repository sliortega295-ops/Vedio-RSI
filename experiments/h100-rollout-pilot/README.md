# H100 rollout pilot

The [report](reports/ADAPTIVE-PILOT-REPORT.md) and
[compact result](reports/PILOT-6340C3E.json) are imported byte-identically from
the previously published benchmark branch. The pilot replayed ten candidates
under serial1/fifo2/optroll1/optroll2 with 3/3/2/2 valid repetitions.

Median Time-to-Validated-Frontier was 6987.23/6337.06/7145.46/6362.41 s.
FIFO2 improved on serial1 by 1.1026×; OptRoll2 was about 0.4% slower than FIFO2.
This is a negative result for the old scheduler. Semantic candidate decisions
agreed across all accepted runs, and median-selected frontier agreed across
systems; two raw individual frontiers differed.

The pilot is exploratory. The full 35-episode trace and formal comparison
remain NOT_RUN. Its quality contract and utilization denominator differ from
the A100 original-agent run. Do not pool their measurements. Full ledgers,
video/score artifacts and the analysis source/code receipts remain in project
storage; the included compact result is a summary, not those raw inputs.

Implementation: [benchmark/sol-rolloutbench-v0](https://github.com/sliortega295-ops/Vedio-RSI/tree/benchmark/sol-rolloutbench-v0).
