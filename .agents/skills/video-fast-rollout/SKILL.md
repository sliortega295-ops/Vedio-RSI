---
name: video-fast-rollout
description: Batch and execute video-deployment candidate rollouts with FastRoll, available GPUs, immutable source snapshots, verified reference reuse and evidence-bound collection. Use for Sol/video optimization execution, not general research, target-policy RL or claims of proven speedup.
---

# Video fast rollout

You are the optimizer, not the GPU scheduler. Use native Codex research, coding,
subagents and memory. Use FastRoll for execution. Read `docs/FASTROLLOUT.md`.

## Work ownership

The primary agent owns the task graph, numerical/quality contract and integration.
Technique subagents own distinct immutable source worktrees; do not edit a
worktree while its candidates are running. A subagent is not a GPU worker: one
agent should submit many independent prompt/seed tasks in a batch. Do not start
one LLM agent per prompt or per GPU. Do not invent candidates to occupy cards.

## Execution workflow

1. Obtain the workload, frozen baseline and an explicit budget/authorization from
   the resource owner. Idle devices and this Skill are NOT authorization.
2. Create/pin an isolated clean source snapshot. Declare model/evaluator assets.
   Keep old reports, candidate histories and experiment contracts unchanged.
3. Construct all already-known independent prompt/seed tasks at once. Include
   actual causal dependencies. Optional primary measurements stay one-shot;
   quality tasks may use a certified resident adapter. Use `expand` for a matrix.
4. `python -m fastroll submit --workspace <state> --plan <plan.json>`
   submits without starting GPU work. Several agents may submit separate batches
   to the same workspace. Reusing a batch ID with different content is rejected.
5. An authorized operator starts ONE controller for the workspace:
   `python -m fastroll start --workspace <state> --authorization <owner.json>`.
   Status SPAWNED is not READY; inspect the controller log and `status`.
6. Continue CPU research/editing while the controller executes. Poll `status`
   at sensible intervals, not in a busy loop. Do not manually choose GPU indices,
   spawn SSH jobs or wrap an existing GPU-locking launcher inside FastRoll.
7. `collect` replays receipts. Read artifacts before claiming success. Execution
   success is not quality success; missing metrics/images are UNKNOWN, not PASS.
8. Commit only source, configuration and compact sanitized reports. Do not commit
   weights, videos, raw sessions, credentials, authorization grants or live state.

## Parallelism policy

When there is a ready task and an eligible idle GPU, let the controller dispatch
it. Do not pin a whole technique to a GPU or wait for one prompt before submitting
another independent prompt. Same candidate != same GPU. Preserve a real
same-device measurement requirement with explicit UUID affinity only when needed.
Do not remove causal dependencies or use future/golden candidate outcomes.

## Reuse policy

Reference generation/standalone scoring may be reused only with complete source,
assets, sampling, input-artifact and device identities. Final verdicts, candidate
latency measurements and candidate-specific LPIPS values must never be cached.
Residency requires `qualify-reset` A-B-A evidence for that source/recipe/domain/GPU.
A failure invalidates the resident process. Never fabricate a reset certificate.

## Completion and reporting

Report queue wait, controller elapsed time, task occupancy, reserved GPU time,
startup/reset/execute time, retries and cache producers separately. Simulation is
not a performance result. Do not rename this backend Sol's original runtime or
pass its output into frozen RolloutBench v0 as though the contract were unchanged.
Stop on revoked/expired authority, source drift, output corruption or an exhausted
budget. Record what remains queued/failed; do not quietly retry a benchmark until
it looks better. GPU lifecycle/quality validation is still required at each site.
