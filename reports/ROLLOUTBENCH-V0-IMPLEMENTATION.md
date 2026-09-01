# Sol-RolloutBench v0 implementation report

Date: 2026-09-01

## Outcome

The recovered 23-round Kernel plus 12-round Cache trajectory is now frozen as a
35-episode executable benchmark. The repository can prepare and resume the same
candidate graph under `serial1`, `fifo2`, `optroll1` and `optroll2`, validate
typed evidence, aggregate repeated runs and seal one fail-closed four-system
comparison.

No H100 candidate was launched during this implementation pass. Consequently,
this report contains no new performance claim.

## Implemented contracts

- Exact authority closure for all 35 public episodes and their source/config
  artifacts.
- Four-system pilot/full run plans, fixed GPU UUID mapping and isolated
  plan/run/cache namespaces.
- Durable event-sourced unit states, exactly-once claims, crash resume and
  write-once evidence.
- Completion-driven FIFO for the naive two-GPU system and typed Kernel/Cache
  dispatch for OptRoll.
- Worker-to-GPU UUID, lease and `CUDA_VISIBLE_DEVICES` binding, with one global
  physical-GPU lock namespace across runs.
- Procedural external launch authorization, ten-minute preflight freshness,
  per-unit revalidation and release-aware cooperative leases.
- One-shot candidate ranking by complete child-process wall time. Warm
  generation latency remains diagnostic only, so compile-heavy K01/K02 cannot
  win by hiding cold cost.
- Exact K22 expected-failure identity: episode, runtime, config hash, failure
  code/stage, child return code and unique source marker. OOM and import failure
  are explicitly rejected.
- Nine formal Cache candidates, four prompts and two seeds (72 candidate pairs
  per benchmark run), pinned VBench source/assets, seven dimensions and
  all-frame LPIPS evidence.
- Deep replay of VBench plan/execution receipts, input videos, LPIPS receipt and
  final Cache quality decision.
- Exact 35-decision full-run finalization, predeclared 3+2 repetition rule and
  system-level TTVF/GPU-hour/utilization aggregation. One five-run plan can be
  summarized after runs 1-3 and extended with runs 4-5 without rerunning the
  first three.
- Four-system comparison that accepts exactly one sealed result per system from
  the same plan, scope and repetition policy; it requires the same validated
  frontier and replays the exact plan, preparation receipt, event ledgers,
  frontier receipts and aggregate metrics before ranking TTVF. Hand-written
  self-consistent result JSON is rejected.

## Validation completed locally

- `python3 -m rolloutbench validate-suite ...`: 35 episodes, valid.
- `python3 -m unittest discover -s tests/rolloutbench -v`: 149/149 pass.
- `python3 -m compileall -q ...`: pass.
- Targeted adversarial tests cover missing/duplicate systems, cross-plan input,
  partial repetitions, missing episodes, forged/unreplayable results, changed
  frontier or ledger receipts, per-run GPU over-capacity, concurrent conflicting
  writers, authorization revocation, wrong worker GPU, stale preflight, released
  leases, K22 OOM/import substitution and changed VBench/LPIPS evidence.

The repository-root Cache controller test was not rerun in the current local
Python because PyTorch is absent. Its historical 9/9 result remains recorded in
`FINAL-REPORT.md`; it is not counted in the 149 tests above.

## Important design boundaries

- `Time-to-Validated-Frontier` is the sum of isolated repetition intervals from
  durable `run_started` to the sealed provisional frontier. Human gaps are not
  included; cross-boot recovery downtime is included through UTC receipts.
- Candidate ranking uses one-shot process wall time. Quality-stage wall time is
  included in run TTVF but not confused with candidate generation latency.
- Dense video generation is reused once per prompt/seed/run. Dense VBench
  scoring is deliberately rerun for each candidate pair because the frozen v0
  contract keeps both sides in one candidate-specific evidence plan.
- The VBench gate is paper-inspired and deliberately small, not the official
  full VBench suite.
- Persistent model workers, an approximate low-fidelity filter and general
  compile-cache reuse are not enabled. K01-to-K02 has only its explicitly
  declared compile lineage.
- Authorization issuer identity is procedural and not cryptographically
  verified.

## Status

### VALIDATED

- Frozen suite, plans, preparation contracts and CPU-only simulation.
- Event ledger, resume, typed validation, quality evidence chain, decisions,
  aggregation and four-system comparison code paths under CPU fixtures.
- Fail-closed launch, lease and worker-binding logic under CPU fixtures.

### PREPARED

- Pinned model/runtime/VBench/DINO/weight profiles and persistent remote path.
- Clean published branch deployed to the canonical remote repository.
- Fresh read-only preflight passes technical readiness, including the pinned
  CUDA/Python runtime, model, VBench/DINO sources, ten quality weights and
  offline LPIPS construction. It does not grant GPU ownership.

### NOT_RUN

- H100 pilot and full 35-episode runs.
- Fresh formal VBench/LPIPS measurements.
- Actual `serial1`/`fifo2`/`optroll1`/`optroll2` TTVF and GPU-hours.
- Real fault-injection recovery timing and blinded visual review.
- Second model, B200/B300, FP8 in this benchmark, Memory Agent and paper-wide
  quality/performance reproduction.

## Next authorized step

Wait for explicit GPU ownership authorization. The formal plan must predeclare
five repetitions; execute and summarize runs 1-3 first. If any candidate exceeds
3% sample latency CV, execute its already planned runs 4-5 for all four systems;
otherwise stop at three. Then use `compare-systems` to issue the first
performance-bearing comparison.
