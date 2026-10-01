# Local Agent control-plane timing snapshot (live, ~13:42 UTC)

The machine-readable [joined result](timing_agent_control_live_1342.json) covers **18 clean seq0002 Codex attempts**: Master 1–5, Kernel 1–7, Cache 1–6, against `telemetry/local_events.jsonl` (714 rows, zero malformed at this snapshot). Each attempt's raw `agent_attempt_XX.timed.jsonl` is preserved in its own current project worktree; the result stores its snapshot SHA256 and last event UTC. Seq0001 contamination is deliberately excluded from these clean-search totals but retained separately as failed preflight, not erased. Active attempts and recovered attempts without `agent_api_end` are **lower-bound envelopes ending at the last stream event or local stop**, not exact lifetime.

| Role, clean attempts | Sum of observed per-attempt envelopes s | Union of completed tool-item intervals within each attempt s | Envelope outside completed tool items s, **unclassified** |
|---|---:|---:|---:|
| Master, 5 | 14,646.019 | 1,261.511 | 13,384.508 |
| Kernel Executor, 7 | 14,276.124 | 2,976.047 | 11,300.077 |
| Cache Executor, 6 | 3,608.822 | 236.055 | 3,372.768 |

These sums are **per-role session-seconds, not additive node wall**: Master and Executors overlap. The large unclassified intervals include Codex model/API work, event buffering, reasoning, and invisible operations; they must **not** be labeled measured thinking or model latency. Tool intervals can include sleeping or subprocess waits and are not CPU-busy time. Category sums below may overlap within a session and should not be added to produce wall time.

| Role | Selected directly visible tool-item wall sums s |
|---|---|
| Master | poll Executor 694.866; other tool 416.460; launch submit 117.553; remote status/sync 67.583; spawn Executor 2.225; resume 1.684 |
| Kernel | remote status/sync 1,288.779; other tool 1,096.644; launch submit 529.546; collect/quality 63.100; file-change item 13.745 |
| Cache | other tool 103.344; launch submit 60.302; quality assessment 36.409; collect 27.343; remote status/sync 10.406 |

The classifier names only matched command strings; e.g. the 2.225-second `spawn_executor` tool category **does not capture all** host materialization/source-copy/transport setup, which is separately bracketed by broker/site transfer receipts. The Master attempt4 auto-resume began13:06:08 and was superseded by sole authoritative attempt5 at13:06:26, an observed ~18-second duplicate-control recovery. Site transport repairs also paused/stopped/restarted scoped Agent groups; those costs remain in the envelopes and must not be attributed to the archived algorithm's intrinsic reasoning. Likewise the site runner's repeated per-prompt model load/warmup is not an archived Harness scheduling decision. No local/remote monotonic clocks have been subtracted across hosts.

At this snapshot Master5 and Kernel7 were still active, so refresh this JSON and table after terminal status. Remote observer GPU/CPU stages, queue/transfer receipts, and source-reported denoise/decode are independent layers; do not use the Codex tool interval as a substitute for any of them.
