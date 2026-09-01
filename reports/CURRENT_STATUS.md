# Current status

Status: `CPU_VALIDATED_H100_NOT_RUN` (2026-09-01).

- Historical Sol-Video reproduction: complete for its bounded two-prompt scope;
  see `FINAL-REPORT.md`.
- Sol-RolloutBench v0: exact 23 Kernel + 12 Cache episodes frozen and valid.
- Runtime: formal dispatcher, typed quality path, recovery, aggregation and
  four-system comparison implemented.
- Local validation: 148/148 RolloutBench CPU tests and compileall pass.
- Remote target: `/home/jiangzhikun/yongyan_liu/Experiments/SolRolloutBench/20260901-v0`.
- H100 execution: `NOT_RUN`; no performance comparison exists yet.
- Ownership: `NOT_AUTHORIZED`; point-in-time idle GPUs do not grant use.
- Required next step: publish the clean commit, deploy it, regenerate read-only
  preflight, then wait for explicit ownership authorization before a pilot.

The formal comparison is valid only when all four systems execute the same plan
and exact episode set, finish the same repetition policy, and independently
reach K20/C12. Until those receipts exist, no RolloutBench speedup is claimed.
The plan predeclares five repetitions: summarize runs 1-3 first, and execute
runs 4-5 for every system only if the frozen 3% sample-CV rule fires.
