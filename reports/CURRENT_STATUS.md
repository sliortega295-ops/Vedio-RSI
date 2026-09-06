# Current status

Status: `EXPLORATORY_H100_PILOT_COMPLETE` (2026-09-05 EDT / 2026-09-06 UTC).

- Historical Sol-Video reproduction: complete for its bounded two-prompt scope;
  see `FINAL-REPORT.md`.
- Sol-RolloutBench v0: exact 23 Kernel + 12 Cache source trajectory frozen and
  valid.
- Runtime: formal dispatcher, typed quality path, recovery, aggregation and
  four-system comparison implemented.
- H100 pilot: 10 representative candidates executed under serial1, fifo2,
  optroll1 and optroll2 for 3/3/2/2 valid repetitions respectively.
- Valid evidence: 10 runs, 1380 completed stages, 100 sealed candidate
  decisions, zero failed stages in the accepted runs and 13.3249 GPU-busy
  hours.
- Median TTVF: serial1 6987.23 s; fifo2 6337.06 s; optroll1 7145.46 s;
  optroll2 6362.41 s.
- Median speedup versus serial1: fifo2 1.1026x; optroll1 0.9779x; optroll2
  1.0982x.
- Candidate semantics agree across all valid runs. All four median-selected
  frontiers are K20+C09, but two raw single-run frontiers differ.
- The 5% OptRoll gate passed (0.435% and 2.177% relative ranges), so repeat 3 is
  not included. Retained runner inventory separately records that it was not
  launched.
- Every accepted run was replay-validated before its GPU leases were released;
  the final target state had two empty GPU samples.

Formal status remains `NOT_RUN`. Serial1 and fifo2 both triggered the frozen
candidate-level 3% CV rule, OptRoll has fewer than three runs, and the full
35-episode trace has not been executed. Therefore the formal
`compare-systems` command cannot be used and no paper-level performance claim is
made.

See [ADAPTIVE-PILOT-REPORT.md](ADAPTIVE-PILOT-REPORT.md) and the machine-readable
[PILOT-6340C3E.json](PILOT-6340C3E.json). The recommended next step is to
implement persistent model workers and validated dense/reference reuse before
spending more GPU time on the unchanged scheduler.
