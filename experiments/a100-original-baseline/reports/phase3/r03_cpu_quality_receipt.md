# R03 exact offline LPIPS backfill (site CPU transport; not a quality verdict)

Result at 2026-09-14 05:30:32 UTC: the archived collector and LPIPS judge
scored 41 aligned pairs from the existing 80 R03/frozen-dense frames. Typed
transport and numerical LPIPS evaluation PASS. Original collector quality
remains `blocked_quality` due `nvidia_gemini:api_key_missing`; R03 visual,
Master independent verification, frontier, and DELIVERY are **not** passed
by this operation. The Archive's `plan_eval` output still has
`baseline_total_s=null`, `speedup=null`, and `tier=null`; the objective
comparison uses the separate frozen baseline and raw five-request timing.

## Exact admission and implementation boundary

- Real seq0002 Cache run:
  `repo/output/experiments/sana2ba100-cache_ca-0002/worktree/runs/20260914-045221-cache_ca_r03_teacache_t0016`.
  Its pre-quality remote raw and local post-`plan_eval` SHA chain is in
  [r03_prequality_provenance.md](r03_prequality_provenance.md).
- Only the trusted CPU worker `repo/orchestration/site_quality_worker.py`
  changed to add this exact R03 run: SHA256
  `2621e434f3b4b410996a7b393cf9d8f53102f03e8252de3330f311401d64328b`.
  The prior remote worker SHA
  `11b45d660e26323f5723aaa26366674a7e204e9df3927d1a0468f3069b349e9d`
  was preserved at remote project
  `artifacts/phase3/trusted_worker_pre_r03/site_quality_worker.py` before
  one-member PTY tar transfer. The 6,886-byte archive SHA256 was
  `5db763f13e9b5bd014f5dbc9142740ba4e5cec2cab9046c52e926a4c75eddf37`,
  member count 1, remote final SHA matched. Transfer ran
  05:26:59.525–05:27:06.337 UTC via
  `scripts/sync_r03_quality_worker.py` SHA256
  `8b43af53021956cd568a806e1cf71650a303fba09dc311d419e48188fc82a296`.
- Host typed transport SHA256 `6e2c48e35a0c164e8166e15a65c541f092788b2c983cd260be6331e653172d08`.
  Twelve CPU-only route tests PASS, including R03 exact/tamper and R01/R02
  repeated-reassessment fail-closed. No archived collector, LPIPS judge,
  `plan_eval`, model, or candidate algorithm was changed for this backfill.
  Host broker old PID 1391772 was stopped only after its exact identity,
  fresh heartbeat, three queues pending=0/unresolved=0 and no child process
  were checked at 05:27:55. New PID 1578907 started 05:28:03.282493 UTC;
  goal broker and active R04 remote GPU worker were not restarted.
- The coordinator's host-only typed `assess_quality` client was pinned to the
  exact Cache seq0002 queue for this one run; no arbitrary shell, other run
  ID, or GPU submit was accepted. Typed request ID
  `57995c95d83a4f488d568b0cc990ea98`, request SHA256
  `27536061f33a1f680b78fe986e8af5792237f14be8fe171706b9ddb75b02681d`,
  broker response SHA256
  `e779ce5ed52f1796f8e3f1477c560182c86fed7420aa88ff96bfff0f3349e8c1`,
  daemon PID 1578907. This is a disclosed site dependency transport, not an
  original Agent search decision or a graph blind reviewer.

## Provenance, snapshots, and timing

- Local preflight verified exact launch, manifest, metadata, benchmark,
  run_config, quality, collection, and assess SHA pins. Remote read-only
  preflight verified pinned archived collector/judge/plan_eval, network guard,
  fixed Python/LPIPS wheel/linear/AlexNet SHAs, original remote benchmark
  `f181fa6493cc...`, all five video SHAs, dense and R03 80-frame digests.
  The full R03 frame digest is
  `d833d5685e177d565937409d90416d90e733d89b68f3b678617c303712a80f52`.
  Remote pre-quality metadata SHA `48ba389c73e0df460a413943867db1885a1d2e6a4bf038b1f9eb6038a78c6c3e`;
  post metadata SHA `f67085769ec3c0c5c57c09ea7a1dc6db04df70184253a7c46354bf2351d6071b`.
  Host metadata was not synced, preserving its separate original provenance.
- Host pre snapshot:
  `artifacts/phase3/pre_quality_snapshots/host/R03/20260914T053014.340272+0000_1578907/`.
  It contains the actual pre bytes for metadata and four quality/timing JSONs:
  benchmark `f5e3e715…`, quality `5699cc1a…`, collection `6970acd8…`,
  assess `c80f8956…`, metadata `47bd63e3…` (full hashes in its manifest).
  Host transport receipt SHA256
  `ab86151bfc61150656c5175941ed37e54dfd321a5b804bdb609a2bbbe5c71e47`.
- Remote attempt:
  `repo/output/experiments/sana2ba100-cache_ca-0002/worktree/runs/20260914-045221-cache_ca_r03_teacache_t0016/phase3_quality_attempts/20260914T053014.679948+0000_2942501/`,
  receipt SHA256
  `26709c701d366e795236b6294187b0dbaaaa4d18168fb6645cdf1da5706dd6c4`.
  Remote pre snapshot had original benchmark SHA
  `f181fa6493ccebf46d22b65748a9284061d29b6c2d14a6ecf536c800f617926c`;
  remote pre quality/collection/assess did not exist. Both pre and post
  bytes are retained. Exactly four small post JSONs were synced to host:
  benchmark `0987b7211fc5cc08fa272c1fc05afad9197507353debbf7cde5dcc8b09dad3c2`,
  quality `ee68ec15cbd53eb1cac5d3d0eaf221b7336caa1f0d51c9b7a47023a7f945e89e`,
  collection `025837f641a7bd7be57483d0e7c781d45f1486599e0838bcfbff24edf552ddbd`,
  assess `2c4d6beb3681d512d665890571d21997dd9925a448b396157a1bde82431d568e`.
  The benchmark still has all five request values totaling 370.52 s,
  unchanged timing scope, five videos, and 80 existing frames. No video or
  frame was transferred by this quality route.
- Host typed transport wall 18.379786689 s (05:30:14.343–05:30:32.723 UTC).
  Remote worker monotonic wall 17.17614155 s
  (05:30:14.679948–05:30:31.856055 UTC); remote preflight 0.700138714 s.
  Original collector wall 14.35246491 s and direct-child CPU lower bounds
  311.035253 user-s / 20.037363 system-s. Original
  `plan_eval --no-gemini --no-refresh-collection` wall 0.18460253 s.
  LPIPS runs inside the collector, so no separate LPIPS-only wall is
  fabricated. These host/remote durations are independently closed
  same-host intervals, not cross-host monotonic subtraction.

## Numerical evidence and unresolved gates

- Original collector used all 80 pre-existing aligned frames to select
  41 LPIPS pairs (not 80 scores), stratified plus worst-case. p01–p05
  selected counts **7/6/6/6/16**; per-prompt means
  **0.106462/0.231996/0.080065/0.142403/0.227695**, maxima
  **0.163234/0.258215/0.081228/0.150340/0.242424**. Original LPIPS
  result `status=ok,n=41`, global mean **0.173539983063**, max
  **0.258215188980**. The receipt contains ordered pair indices and exact
  per-frame scores. These numbers are not substituted for visual judgment.
- Offline worker set `CUDA_VISIBLE_DEVICES=''` and project-local fixed
  `TORCH_HOME`, LPIPS target, no proxy/PIP index, `HF_HUB_OFFLINE=1`, and a
  socket-deny guard. Three subprocesses logged `guard_loaded`, with
  `blocked_network_attempts=[]`; network audit SHA256
  `f9087ae2b23d5c9d2a36501f1728a7a7fdb838ff991ce67b227f133fff3243ca`.
  R04 GPU was concurrently active on this node; no NVML GPU utilization
  is attributed to the CPU quality subprocess merely from overlap.
- `quality.status=blocked_quality` with
  `nvidia_gemini:api_key_missing`; archived `plan_eval` emitted no
  baseline total/speedup/tier and explicitly says retain/backfill but do
  not select final profile yet. The active orchestration-lite Cache
  contract's LPIPS+real Executor visual path and later Master independent
  visual/verification must be separately evidenced. Cache attempt 9 had
  only R02 image inputs; R03 montage-only self-report is unsupported as
  a real Codex pixel review. R04 was autonomously launched before this
  R03 LPIPS backfill; that ordering and spent GPU time remain part of the
  native search trajectory, not erased retroactively.
