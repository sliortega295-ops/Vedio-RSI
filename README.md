# Sol-RolloutBench v0

Sol-RolloutBench turns the recovered SANA-Video 2B Sol-Agent trajectory into a
deterministic rollout-infrastructure benchmark. It replays the same 23 Kernel
and 12 Cache candidates under four execution systems and measures how long each
system takes to reach the same validated frontier.

This branch contains the benchmark runtime, its CPU-validated contracts and the
first exploratory H100 pilot. The pilot is deliberately smaller than the formal
benchmark and is not a publication-grade result.

## Frozen benchmark

- Model/workload: SANA-Video 2B, one H100 per generation, 832x480, 81 frames,
  16 fps, 50 denoising steps.
- Search trace: `K01-K23` plus `C01-C12`, exactly 35 public episodes.
- Kernel candidates: exact structural validation and one-shot process-wall
  ranking. Warm generation time is reported separately and cannot select the
  frontier.
- Formal Cache candidates: C02, C03, C04, C06, C07, C09, C10, C11 and C12.
- Cache quality workload: four predeclared prompts times seeds 42 and 12345,
  using a pinned seven-dimension VBench mini-gate. LPIPS covers all 81 matched
  frames and is secondary only.
- Repetitions: predeclare five; summarize the first three, then execute the last
  two for every system only when any candidate's first-three sample latency CV
  exceeds 3%.
- Expected full frontier: K20 and C12. This historical oracle is hidden from
  scheduling and consulted only after fresh decisions are sealed.

The four systems are:

| ID | Meaning |
| --- | --- |
| `serial1` | Historical Sol-Agent style, global FIFO, one GPU |
| `fifo2` | Naive global FIFO, two GPUs |
| `optroll1` | Typed, decision-aware scheduling, one GPU |
| `optroll2` | Typed Kernel/Cache streams, two GPUs |

Persistent model workers are not enabled in v0: every candidate remains a
one-shot process until compatibility and reset proofs exist. OptRoll v0 tests
typed scheduling, exact declared reuse and recovery contracts, not an
unimplemented persistent-worker speedup.

## Current status

- Frozen 35-episode suite: `VALIDATED`.
- CPU benchmark/runtime tests: `VALIDATED` (see
  [reports/ROLLOUTBENCH-V0-IMPLEMENTATION.md](reports/ROLLOUTBENCH-V0-IMPLEMENTATION.md)).
- Read-only H100 readiness: `TECHNICAL_READY`; the pinned CUDA/Python runtime,
  model, VBench/DINO sources, ten quality weights and offline LPIPS load pass.
- Adaptive H100 pilot: `COMPLETE` for 10 representative candidates and a
  user-approved 3/3/2/2 repetition policy; see
  [reports/ADAPTIVE-PILOT-REPORT.md](reports/ADAPTIVE-PILOT-REPORT.md).
- Pilot median TTVF: serial1 6987.23 s, fifo2 6337.06 s, optroll1 7145.46 s and
  optroll2 6362.41 s. The two-GPU systems are about 1.10x faster than serial;
  OptRoll2 does not beat FIFO in this pilot.
- Formal 3+2 H100 comparison and full 35-episode execution: `NOT_RUN`.
- All pilot GPU leases were released after independent replay and two idle
  samples. Point-in-time idleness is never treated as future ownership.

The earlier two-prompt Sol-Video reproduction remains documented in
[reports/FINAL-REPORT.md](reports/FINAL-REPORT.md); its 1.83-2.11x results are
historical inputs, not new RolloutBench results.

## CPU verification

```bash
python3 -m rolloutbench validate-suite \
  benchmarks/sana_video_2b_h100_v0 --repo-root .
python3 -m unittest discover -s tests/rolloutbench -v
python3 -m compileall -q rolloutbench \
  models/sana_video_2b_h100/baseline tests/rolloutbench
```

## Formal workflow

The formal path is deliberately split into read-only readiness, deterministic
planning, preparation, externally authorized execution, per-system aggregation
and four-system comparison:

```text
h100-preflight -> experiment-plan -> prepare-experiment
       -> run-formal -> summarize-system -> compare-systems
```

Create one five-repetition plan before execution. After runs 1-3 complete for
all four systems, call `summarize-system --completed-repetitions 3`. If any
summary returns `NEEDS_TWO_ADDITIONAL_REPETITIONS`, execute the already planned
runs 4-5 for every system and summarize with `--completed-repetitions 5`.
Otherwise compare the four three-repetition results. This keeps one plan hash
and reuses the first three ledgers instead of creating a second plan and
rerunning them.

`run-formal` requires both a fresh external authorization file and the explicit
`--execute-authorized-gpu-plan` acknowledgement. The authorization JSON is a
procedural trust boundary, not a cryptographic signature; see
[docs/launch-authorization.md](docs/launch-authorization.md).

The formal Cache contract intentionally generates each dense video once per
prompt/seed/run, but reruns dense VBench scoring inside every candidate-specific
matched-pair plan. That cost is frozen into all four systems and cannot be
silently optimized away.

`compare-systems` reopens each result's exact plan, preparation receipt and run
ledgers, independently re-aggregates them, and requires byte-canonical object
agreement before it ranks TTVF. A self-consistent hand-written result JSON is
therefore not accepted as benchmark evidence.

The completed exploratory pilot uses a separate fail-closed reader so it cannot
be mistaken for the formal result:

```bash
python3 -m rolloutbench summarize-adaptive-pilot \
  --plan /path/to/pilot.json \
  --preparation /path/to/preparation.json \
  --state-root /path/to/formal-state \
  --completed serial1=3 --completed fifo2=3 \
  --completed optroll1=2 --completed optroll2=2 \
  --ttvf-relative-range-threshold 0.05 \
  --output /path/to/adaptive-pilot.json
```

This command replays every selected ledger and reports semantic decision
agreement, raw and median-selected frontiers, TTVF spread, GPU-hours and the
formal-policy boundary. Replay uses a non-repairing ledger snapshot, binds the
plan to the exact suite files, and separately fingerprints on-disk source and
the live loaded Python callables. Its output is explicitly incompatible with
`compare-systems`.

Large videos, weights, environments, logs and run ledgers stay in persistent
experiment storage and are not committed to Git.
