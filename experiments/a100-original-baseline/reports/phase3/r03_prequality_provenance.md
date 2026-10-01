# R03 pre-quality provenance (Phase 2 GPU window)

Status at 2026-09-14 05:13 UTC: GPU run PASS for execution/structure only;
LPIPS, Executor visual, Master verify, and frontier remain REVIEW/NOT_RUN. This
receipt does not modify the Phase 1 terminal evidence or any candidate result.

## Native submission and budget

- Candidate: seq0002 Cache R03 `20260914-045221-cache_ca_r03_teacache_t0016`,
  autonomously submitted by Cache Executor attempt 9 via
  `scripts/launch_config.py ... --mode sbatch --confirm-submit` at
  04:52:21.478 UTC. Host typed request `251b3fd268d24509b70365c64dda5e47`
  returned `ok=true` at 04:53:50.474 UTC. Request-to-response wall is
  89.046 s and includes validation/source transfer/remote admission, not
  generation. Host response SHA256
  `92c4235985f378e8591c4eb8df49d28abbb9dce7a7047526c28f922ef4d022f8`.
- Trusted remote worker PID 2912395 started 04:53:50.446 UTC, finished
  05:07:13.650173 UTC, exit 0, duration 803.145334299 s, no budget timeout.
  The independent Phase 2 first-job receipt starts at
  04:53:50.441482 UTC; deadline 10:53:50.441475703 UTC and new-submit
  cutoff 10:32:50.441475703 UTC. It is an accepted site-job start, not a
  claim that CUDA was already active at that instant. The observer's first
  positive GPU0 sample lies in same-host monotonic interval
  `(24.186319405, 25.186261661]` s after budget start, with query latency
  uncertainty. Other GPUs are not inferred used from the eight-GPU profile.

## Remote raw run (before local collector)

- Raw `outputs/benchmark.json` SHA256
  `f181fa6493ccebf46d22b65748a9284061d29b6c2d14a6ecf536c800f617926c`;
  `total_s=370.52`, equal to five request seconds
  `[74.29, 74.04, 74.03, 74.07, 74.09]`. Reported denoising sum
  338.8783 s, decoding sum 28.1299 s. Timing scope remains
  `five_prompt_sum_sglang_request_metrics_excluding_model_load_internal_one_step_warmup_and_video_export`.
  Every prompt reports 832×480, 81 frames at 16 fps, 50 steps, seed 42,
  world size 1 and UUID `GPU-cad55364-e014-c35f-337e-db899afd1c6a`.
  The five separate prompt processes repeat model load/warmup; this is
  the A100 site wrapper, not an archived Harness scheduling claim.
- Remote launch SHA256 `7ecb20676ff681fdfbbceab3a650fa900aa075e5eddabf35e6ae4458d0e7c002`;
  manifest SHA256 `1fdeccc38e5702e15dd37cd52f933ccd09d509ccbb3baa12705ee1dc04577cc4`;
  metadata SHA256 at post-worker read `48ba389c73e0df460a413943867db1885a1d2e6a4bf038b1f9eb6038a78c6c3e`;
  run_config SHA256 `d6c90681de8062cf937b468d6db6d6f9ee221a77f2b4b6914a185ba31cad8e90`.
- Five remote video SHA256s in prompt order:
  `992b717f477e37afd5da43b37ada50d56a19779e0f663a571ffc28e2fc90af1f`,
  `e188e36f177d4a266eb1729e41103cdf218ee22f445f0889b4ddf9db8ad0ee0b`,
  `407bc097656ce746c1ea34d4359692f20c33cef16821f02fe5874647736deeef`,
  `8f3284eba4a78c36f6ef5a323f09217bc518529a651c7681add61116d10ed926`,
  `3c95c371e792ce0cbdf4b77b6ccb8d11f7b388d9778e4feddb0fb31fc8470500`.
  Each is independently `sha256sum`-checked on the remote run; benchmark
  ffprobe fields claim 81 frames/832×480/16 fps. Exactly 80 regular,
  non-symlink `f_001.png`–`f_080.png` files are present. SHA256 of the
  concatenated per-frame SHA256 lines (with newlines) is
  `d833d5685e177d565937409d90416d90e733d89b68f3b678617c303712a80f52`.
- Each prompt log has 50 `TC step=0..49` entries and 19 `run=0` first-branch
  decisions, 95 observed reuse decisions/250 logged steps total.
  `sana_video.py` copies the branch-0 run decision to unlogged CFG branch 1;
  corresponding branch-1 reuse count is source-derived, not directly counted.
  The progress bar's `25/50` is scheduler-order display, not a 25-step run.

## Local native collect and assess (not quality acceptance)

- Cache attempt 9 raw stream is
  `repo/output/experiments/sana2ba100-cache_ca-0002/worktree/goals/cache_ca/agent_attempt_09.timed.jsonl`.
  Item 71 completed the original local collector at 05:08:31.282 UTC,
  with `frames.status=existing,count=80` and no baseline path.
  Item 73 measured first-collected benchmark SHA256
  `705c2957e0e2da689338722d87eb88e773a48edaa54bee213eca136d53812d0f`
  and quality SHA256
  `bec2f0a576bd0acdbfd78dcfc5d98af7d9be273dab9de126a6c5c03a47db66fd`.
  Item 77 contains the complete 3,612-byte quality text with that same SHA;
  item 75 printed only 12,017 of 18,538 benchmark bytes, so the entire
  first-collected benchmark body is **not** recoverable from that tool output.
  The hash and timing fields remain evidenced; do not synthesize a sidecar.
- Item 86 ran `python3 search/plan_eval.py --model sana_video_2b_a100
  --no-gemini --assess runs/20260914-045221-cache_ca_r03_teacache_t0016
  --baseline-frames /home/lyy/Experiments/SolAgent/20260913-a100-original-baseline/repo/runs/20260913-091006-sana_video_2b_a100_dense/outputs/frames
  --out runs/20260914-045221-cache_ca_r03_teacache_t0016/assess_verdict.json`,
  completed 05:10:04.090 UTC with process exit 0. This call refreshed local
  collection and changed small JSON bytes. Post-assess SHA256s at read:
  benchmark `f5e3e7154f492d9d0e57f86a17d1e9d3faa6bbf6a840b5eb854b58df86cbaa49`,
  quality `5699cc1a5ba43df4cfd92cf02b57ec5641623c89a6c0b1ad30980ef032651a7b`,
  collection `6970acd8d69d779ea38fc7dd95277848018a868c676cee9fa04ff8ce279201f6`,
  assess `c80f89563afc5057409de92e6c9c6a46be442ef72e658dfe3ad949442ad2f193`.
  The verdict gives baseline 578.02 s/config 370.52 s/speedup 1.56× but
  `quality_status=blocked_quality`, blockers `lpips:dependencies_missing`
  and `nvidia_gemini:tool_missing`, `tier=null`, no LPIPS value. Exit 0 must
  not be called quality PASS. Current native Agent continues; later hashes
  require a fresh chain rather than assuming these remain immutable.

## Remaining gates

The R01/R02-only trusted CPU quality route rejects R03 by design. A scoped
extension requires exact post-run local and remote pre-hashes, same archived
collector/LPIPS/plan_eval under CUDA-hidden network-denied execution, first
pre/post snapshots, negative tests, and root review. Cache attempt 9 received
R02 image attachments only; its R03 montage creation is **not** R03 pixel
input to Codex. R03 visual judgment requires a later, separately hash-pinned
image transport. Master independent visual, native verify, and integration
are not established by this receipt.
