## FP8 Delivery Contract

The executor-owned `DELIVERY.json` uses schema version 2, `status="complete"`
or `status="structured_negative"`, `component="fp8"`, the exact `model_id`, and
at most one entry in `frontier_points`. The point must reference a real isolated
FP8 run under the executor worktree and include:

- `run_dir`, `config_id`, `activation.env` with every `SANA_FP8_*` value;
- the committed config manifest and source identity;
- recomputed latency/speedup against the frozen baseline;
- `quality.mode="quality_gated"`, LPIPS when available, and the bounded visual
  verdict;
- `fp8_evidence.component_smoke` as a worktree-relative path to the component
  smoke report;
- parsed install and active-module receipts proving E4M3 executed;
- the output MP4, frames, benchmark, and assessment artifacts.

Fallback-only, integrated Kernel/Cache compositions, or `NOT_RUN` evidence
cannot appear as an executor frontier point. A direct integration acceptance
packet must use a different filename and must never masquerade as this official
executor contract.

When no isolated candidate survives after real measured attempts, use
`status="structured_negative"`, `frontier_points=[]`, and a
`negative_evidence` object containing a non-empty reason, exact positive
`attempt_count`, worktree-relative `trajectory` and `search_state` paths, plus a
non-empty list of worktree-relative `evidence_files`. The trajectory must contain
exactly `attempt_count` parseable records, and the search state must say
`component="fp8"`, `status="structured_negative"`. Hardware or ownership
failures use the `blocked` outcome instead.
