# Cache search journal

This journal is a primary-integrator reconstruction of the Cache executor's already committed 12-round ledger. The executor completed the search and frontier summary at `804491e704636754bbabdc8d4cc06ad684fb9815`, but missed the final bounded delivery checkpoint. No hypothesis, metric, run, decision, or artifact below was invented or rerun during reconstruction.

Frozen denominator: 61.7 s, 17,770 MiB, `BASELINE-LOCK.json` SHA-256 `3fdafdf00554ae4bafc91bc7729c3ba3e96af4edebac809782e6d6122ed23954`. Screening used one fixed model-card prompt at 832x480, 81 frames, 16 fps, 50 steps, guidance 6, seed 42, with first/middle/last-frame checks.

| Round | Family / hypothesis | Result | Decision |
| --- | --- | --- | --- |
| 1 | EasyCache 0.05 seed | 52.8 s; valid video, but stale warmup/CFG residual provenance | Preserve failure; exclude and retry same parameters after formal-state repair |
| 2 | Corrected EasyCache 0.05 | 46.2 s, 9 hits / 41 computes, 17,770 MiB | Retain conservative point |
| 3 | Faithful TeaCache 0.10 | 28.5 s, 29 / 21, 17,910 MiB; material visual drift | Retain aggressive comparator, not equal-quality claim |
| 4 | First-order TaylorSeer 0.10 | 28.0 s, 29 / 21, 17,910 MiB; strongest matched drift | Reject from selected frontier; retain comparator evidence |
| 5 | TeaCache conservative calibration 0.012 | 44.9 s, 10 / 40, 17,910 MiB | Retain calibration; exclude from shared band (2.81% outside) |
| 6 | TeaCache matched conservative 0.01125 | 46.7 s, 8 / 42, 17,910 MiB | Retain conservative comparator |
| 7 | TaylorSeer matched conservative 0.01125 | 46.9 s, 8 / 42, 17,910 MiB | Retain comparator; slightly more drift |
| 8 | EasyCache aggressive calibration 0.20 | 29.2 s, 28 / 22, 17,770 MiB | Retain calibration; exclude from shared band (2.456% outside) |
| 9 | EasyCache matched aggressive 0.25 | 28.4 s, 29 / 21, 17,770 MiB | Retain aggressive; tentative edge or tie with TeaCache |
| 10 | TeaCache balanced 0.020 | 39.0 s, 17 / 33, 17,910 MiB | Retain balanced comparator |
| 11 | TaylorSeer balanced 0.020 | 38.8 s, 17 / 33, 17,910 MiB | Retain comparator; medium drift |
| 12 | EasyCache balanced 0.10 | 38.8 s, 17 / 33, 17,770 MiB | Retain balanced; genuine convergence |

Every formal round has a pair of durable records under `evidence/cache/round_NN.json` and `evidence/cache/round_NN_trajectory.json`, plus the append-only `TRAJECTORY.jsonl`. They contain the hypothesis, executor prompt/session hashes, parent and candidate commits, manifest hash, exact command, build/test status, run directory, benchmark, video hash, validity screen, decision, and reason. Visual amendments for rounds 7, 10, and 11 are preserved separately.

The matched bands are conservative 46.2-46.9 s, balanced 38.8-39.0 s, and aggressive 28.0-28.5 s. All are within the required 2% relative time band, and every family's speedup increases monotonically across tiers. EasyCache is the bounded conservative winner/tie, clear fixed-prompt balanced winner, and tentative fixed-prompt aggressive winner/tie. This conclusion is intentionally scoped to the approved fixed-prompt screen.

The archived workflow's five-prompt plus independent blind-review completion gate was not applied because the approved reproduction deliberately uses one fixed screening prompt, lightweight validity/visual checks, and a later master-owned small multi-prompt sanity check. LPIPS is therefore not promoted as an acceptance claim in the delivery.

Delivery handoff disclosure: after a final explicit checkpoint requesting `DELIVERY-DRAFT.json`, `CACHE-SEARCH-STATE.json`, `AGENT-STATUS.json`, `SEARCH_JOURNAL.md`, and `SUMMARY.md`, the child still left a clean worktree at `804491e`. The primary integrator interrupted/retired it and reconstructed only these delivery documents from the durable evidence above.
