# Contaminated control-plane preflight, sequence 0001

Status: **EXCLUDED / contaminated_preflight**. This trial is retained only as
setup, failure, recovery and timing evidence. Its agents and any candidate
ideas are not eligible for the A100 zero-search result. The frozen dense
baseline is unaffected and will be reused without rerunning it.

- Master launched at 2026-09-13 09:29:50.534 UTC through the restricted host
  transport, using the unchanged archived Master role prompt plus the initial
  A100 site appendix. At 09:30:13.700 UTC, its `item_5` explicitly queried
  `/home/lyy/.codex/memories/MEMORY.md` for SANA/A100/SolAgent/baseline terms.
  This violates the user's no-history information boundary. The visible raw
  stream contained old experiment context, so a clean result cannot be
  inferred even if no old recipe was visibly used.
- The first kernel `spawn_executor` failed at 09:31:00 UTC because
  `Path.glob("models/sana_video_2b_a100/**")` did not enumerate files. The
  Master repaired the site's general `expand_include` handling of a trailing
  `/**` at 09:31:37 UTC; this is setup/control-plane repair, not an
  optimization candidate. It made **no** `git add`, commit, push, or PR.
- The Master then spawned only the registered kernel and cache Executors at
  09:31:51.980 and 09:31:57.262 UTC and polled them at 09:32:09 and
  09:32:53 UTC. It had not reached a real GPU DELIVERY or `verify_delivery`.
- Remote GPU audit immediately before stopping found **zero** compute apps and
  no candidate `site-job.json`; no candidate GPU result exists in this trial.
  The scoped orchestrator PID 247496 was stopped, then Master PID 247514 at
  09:32:54.471 UTC, kernel Executor PID 251361 and cache Executor PID 252123
  at 09:33:10.502/503 UTC. Their session states are `stopped`; OS entries
  were defunct/zombie, not running. No prior directories were deleted.

Raw evidence is under `repo/output/orchestrated/sana2ba100-20260913-092950`,
`repo/output/experiments/sana2ba100-{kernel_aw,cache_ca}-0001`,
`repo/.symposium/site-goal-sessions`, `telemetry/local_events.jsonl`, and
`telemetry/broker_requests.jsonl`. Master raw JSONL SHA-256:
`afefec446b570f4f8f6986d847b221610d288bc2841f3fbbcc21cb5e501a34b0`;
timestamped JSONL SHA-256:
`189f4a61e84b9a995c664a6fb04237841a75f7a082b59a3987682724a0c349f3`.
The future clean sequence must use distinct goal/worktree IDs and must not
read this trial's agent artifacts or candidate content.
