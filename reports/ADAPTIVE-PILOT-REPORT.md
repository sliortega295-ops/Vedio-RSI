# Sol-RolloutBench adaptive H100 pilot

Date: 2026-09-05 EDT / 2026-09-06 UTC

## Outcome

The authorized H100 pilot is complete under the later, explicitly exploratory
3/3/2/2 repetition policy. Ten valid runs replayed the same ten representative
candidates and sealed 100 candidate decisions. All valid runs passed their
event-count checks, controller-exit check and independent ledger replay before
their GPU leases were released.

The result is useful but negative for the current OptRoll design: two GPUs cut
median Time-to-Validated-Frontier by about 10%, not by 50%. `optroll2` did not
beat naive `fifo2`; their medians differ by only 0.4% in FIFO's favor.

This is an engineering pilot, not a paper-level performance result. It covers a
10-candidate subset, not the full 35-episode trace, and it intentionally does
not satisfy the repository's original formal 3+2 repetition rule.

## Frozen execution scope

- Source commit: `6340c3ed250d271b09f91d396e2ee1efaf756e9d`
- Source tree: `68fda20b364bdfebc25792eb6981d4b64d33a89f`
- Plan ID: `23da91169c2043b0b77bf2439a854a5fae908c7b4895adbbd3b17ff7fb4ea932`
- Pilot candidates: K01, K02, K15, K19, K20, K22, C01, C02, C09 and C12
- Per run: 138 stages, 10 sealed decisions and one provisional frontier
- Valid repetitions: `serial1=3`, `fifo2=3`, `optroll1=2`, `optroll2=2`
- Total valid work: 10 runs, 100 decisions and 13.3249 measured GPU-hours
- Cache quality gate: four prompts, seeds 42 and 12345, pinned VBench-7D mini
  metrics plus all-frame LPIPS

The failed/interfered FIFO attempt archived during recovery is not included in
any number below. The replacement `pilot-fifo2-repeat-03` completed 138/138
stages with zero failed stages.

## Measured result

| System | Valid TTVF samples (s) | Median (s) | Versus serial | GPU h / run | Scheduler utilization |
| --- | --- | ---: | ---: | ---: | ---: |
| `serial1` | 6509.84, 6987.23, 7232.24 | 6987.23 | 1.0000x | 1.3215 | 68.85% |
| `fifo2` | 5749.47, 6360.66, 6337.06 | 6337.06 | 1.1026x | 1.3511 | 39.55% |
| `optroll1` | 7129.92, 7161.01 | 7145.46 | 0.9779x | 1.3260 | 66.81% |
| `optroll2` | 6431.66, 6293.16 | 6362.41 | 1.0982x | 1.3275 | 37.56% |

`optroll2 / fifo2` is 0.9960x: the current typed scheduler is effectively tied
with, and slightly slower than, naive two-GPU FIFO in this small sample.

The two-GPU utilization denominator contains both authorized GPUs for the full
TTVF interval. Values near 38-40% show why a theoretical 2x speedup is not
available here: long dependency chains leave only one runnable generation task
for substantial periods. Adding a second card cannot parallelize those serial
spans. The nearly unchanged GPU-hours per repetition also shows that scheduling
moved the same work around rather than eliminating it.

## Why only two OptRoll repetitions are included

The user-approved pilot gate required both first two OptRoll runs to have:

1. identical semantic candidate decisions;
2. identical raw provisional frontiers; and
3. TTVF relative range `(max - min) / mean <= 5%`.

Both systems passed:

| System | Semantic decisions | Raw frontier | Relative range | Decision |
| --- | --- | --- | ---: | --- |
| `optroll1` | agree | K20+C09 twice | 0.435% | stop after 2 |
| `optroll2` | agree | K20+C09 twice | 2.177% | stop after 2 |

No third OptRoll run, and no fourth or fifth run for any system, was launched
according to the retained runner inventory. The adaptive result itself records
the narrower machine-proven fact that repeat 3 is not included; it does not
infer launch absence from a missing included context.

## Decision agreement and its boundary

After removing run-specific paths, hashes and measurements, all ten valid runs
have the same semantic result for every candidate. K01/K02/K19/K20 passed exact
validation; K15 was rejected at preflight; K22 produced the expected rejected
failure; C01 was excluded on provenance; and C02/C09/C12 passed the quality
gate.

Median candidate ranking selects K20+C09 for all four systems. Raw single-run
frontiers are not perfectly stable:

- `pilot-serial1-repeat-02` selected K19+C09;
- `pilot-fifo2-repeat-03` selected K20+C12;
- the other eight valid runs selected K20+C09.

This distinction matters. We have semantic decision agreement and a shared
median-selected frontier, but not raw frontier agreement across every run.

## Formal benchmark boundary

The repository's original formal rule examines each candidate's first-three
latency CV and requires repetitions 4-5 when any CV exceeds 3%. Both completed
three-run summaries triggered that rule:

- `serial1`: C01, C09, C12, K01, K02 and K19 exceeded 3%;
- `fifo2`: C01, C02, C09, K01, K02 and K20 exceeded 3%.

OptRoll has only two repetitions, so its formal rule cannot be evaluated. The
official `compare-systems` command therefore remains `NOT_RUN`, and this result
must not be presented as a formal four-system comparison or a full
Sol-RolloutBench result.

The pilot also does not establish broad video-quality preservation, a full
35-episode replay, a second model, B200/B300 behavior, FP8 benefit, fault
injection recovery time or paper-wide Sol reproduction.

## Operational observations

The final run completed remotely while the local VPN tunnel temporarily lost
its IPv4 route. Once SSH returned, the recovery wrapper replayed the complete
138-stage ledger, confirmed both GPUs idle twice, released both leases and
exited zero. This is useful recovery evidence, but it was not a controlled fault
injection experiment, so recovery latency remains `NOT_RUN`.

## Recommended next step

Do not spend more H100 time repeating this unchanged v0 scheduler. The pilot has
already answered the engineering question: scheduling alone exposes only about
10% wall-time benefit on this graph.

The next implementation should reduce the serial work itself, starting with a
persistent model worker and validated reuse of dense/reference scoring. After
that change passes reset and equivalence tests, rerun only this same
10-candidate pilot. Continue to the full 35 episodes or formal 3+2 campaign only
if the revised `optroll2` clearly beats `fifo2` while preserving the same
semantic decisions and frontier.

## Evidence

- Public compact result: [PILOT-6340C3E.json](PILOT-6340C3E.json)
- Replay command implementation: `rolloutbench/adaptive_pilot.py`
- Analysis implementation receipt SHA-256:
  `712ac2e3de63ce1815b49c0f2467dc22f2fa5ec4558b168cea745f9e073473fc`
- On-disk analysis-source manifest SHA-256:
  `eda110a17f328a47eb1ffe6a252a4801374cd3c07984cd97e87f9a5820fb78f4`
- Live loaded-code manifest SHA-256:
  `667ac898df3663b516bd4a7f0745f0f06cf70acf21f4ef0a32f1f0d726a61935`
- Full derived result SHA-256:
  `4b88ac260a9be541883714b1dff4bcd486e6fc43acae6e1a7b152029eff6aa12`
- Formal serial summary SHA-256:
  `c1509e890fc149bb1622b223f76b5cfaa22f51e6d01f906f04d16d8b88ea089d`
- Formal FIFO summary SHA-256:
  `3d2113c81e4331f80723c683bd7e66501d84eddc50f1242a7bfb58e493a3358b`

The full v5 result, videos, weights, logs and ledgers remain in persistent
project storage and are not committed to GitHub. The v5 result binds the exact
suite files, rejects repairable partial ledger tails and keeps the on-disk
source receipt distinct from the live code-object receipt.
