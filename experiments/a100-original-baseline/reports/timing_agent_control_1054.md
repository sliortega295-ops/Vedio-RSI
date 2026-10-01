# Clean seq0002 control-plane event timing (snapshot 10:54 UTC)

Machine-readable `reports/timing_agents_1054.json` was derived by `scripts/analyze_agent_timing.py` from the 12 raw `agent_attempt_*.timed.jsonl` streams (Master attempts 1–2, Kernel 1–4, Cache 1–6) and `telemetry/local_events.jsonl`. Each session entry includes its raw-stream SHA256, event counts, completed tool-item UTC/host-monotonic boundaries, and category tags. This is a **local host** stream envelope analysis, not a measurement of agent “thinking time.” Concurrent Master/Executor envelopes overlap and must not be added into total wall time.

| Clean seq0002 role/attempt | Local API envelope, s | Completed tool-event union, s | Outside completed tool events, **unclassified**, s | API end receipt? |
|---|---:|---:|---:|---|
| Master 1 | 3070.661 | 157.862 | 2912.799 | yes; scoped safety restart after prohibited `uv pip` attempt |
| Master 2 | ≥1563.638 | 59.489 | ≥1504.149 | running at snapshot |
| Kernel 1 | 1027.463 | 3.291 | 1024.172 | yes; transport preflight/recovery |
| Kernel 2 | ≥109.167 | 0.571 | ≥108.596 | no API end; scoped local stop/restart |
| Kernel 3 | ≥1225.787 | 39.886 | ≥1185.900 | no API end; scoped local stop/restart; remote profile kept running |
| Kernel 4 | ≥1959.570 | 209.933 | ≥1749.637 | running at snapshot, includes R1 completed and R2 start |
| Cache 1 | 525.098 | 36.011 | 489.088 | yes; failed transport preflight |
| Cache 2 | ≥101.157 | 0.622 | ≥100.535 | no API end; scoped local stop/restart |
| Cache 3 | 891.707 | 74.037 | 817.670 | yes; safety interruption after failed package/browser probes |
| Cache 4 | ≥147.640 | 9.198 | ≥138.443 | no API end; force-resumed by Master |
| Cache 5 | 1681.304 | 115.758 | 1565.546 | yes; R02 finished, exited without DELIVERY |
| Cache 6 | 261.916 | 0.429 | 261.487 | yes; durable LPIPS BLOCKED closeout |

The sums outside tool events can include API wait, model work, planning, stream buffering, hidden I/O, intentional polling gaps, and SIGSTOP time; they are **not** precise LLM reasoning or network latency. Completed tool-item wall includes subprocess waits, not CPU busy time. A live/no-API-end row is a lower-bound snapshot. `launch_submit` event sums (for example Kernel attempt4's 3 calls totaling 124.970 s, Cache attempt5's R02 call 60.032 s) include project-local broker materialization/transfer and do **not** represent GPU compute; failed submit attempts remain visible. Master attempt1 included two `spawn_executor` calls totaling 2.225 s of tool-event wall plus materialization work outside that call; the control chain is not claimed to be byte-for-byte node-local archived transport. `poll_executor` call totals only the observable CLI invocation wall, not the full interval spent waiting for remote jobs. `verify_delivery.py` has not run on a real DELIVERY at this snapshot.

The raw stream is the authoritative per-action ledger. In particular, Cache attempt5's 150/250 reuse trace and R03=0.016 estimate are agent-reported analysis; the independently observed GPU timing/UUIDs are in `reports/timing_cache_r02_1042.md`. Master attempt1's failed PyPI install, Cache attempt3's failed proxy/browser probes, skill-read/restart overhead, contaminated seq0001 exclusion, and broker repair are separate attribution categories in the safety/preflight/methodology reports; none is silently assigned to GPU generation. No local and remote monotonic timestamps are subtracted.
