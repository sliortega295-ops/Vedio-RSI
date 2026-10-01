# Phase 2 R06 execution receipt (quality pending)

Observed 2026-09-14. This is an outer, read-only timing/provenance receipt for the
original Cache Executor's R06, not a frontier or visual-quality verdict.

- Exact run: `repo/output/experiments/sana2ba100-cache_ca-0002/worktree/runs/20260914-063907-cache_ca_r06_easycache_t010_w30`.
  Native `launch_config.py --mode sbatch --confirm-submit` CLI stream item254
  started at 06:39:07.387631Z and completed exit 0 at 06:40:51.205058Z:
  **103.817s local command wall**. It includes bundle preparation, site validation,
  transfer, broker response, and remote admission; no narrower queue/build split
  is directly measured by this item. Broker request ID
  `0c05a9961cd74977b55a3844983cc8d3`; 14,377,930-byte worktree archive,
  3,375-byte run archive. The Agent's premature admission claim (item256),
  correction after a read-only queue error (item258), and actual receipt-based
  correction (item261) are retained in the raw timed stream. No second submit.
- Trusted remote `site_job.py` SHA
  `ef686fb4131e2846286ba474ba4a76e09413ee156f8a1b587f58e6f513b30f1d`
  admitted worker PID 2982383 at 06:40:51.092600Z, monotonic
  1876401316252798; finished exit 0 at 06:55:57.464777Z, monotonic
  1877307688424888. Worker-reported duration 906.3165s; exact receipt-clock
  interval 906.3722s. No budget timeout/error. Phase 2 authorization SHA
  `b2a827a8d2fd65486987155f826fd23bd59a03ed16602bec8604d214954bec5c`.
  Do not subtract local and remote monotonic timestamps.
- Manifest SHA `fce503fabbb219c9cab0df441b6555163eb3e66c115e9f7119febd46d9530b9f`:
  one video at a time on GPU0, five frozen prompts, seed 42, 832×480, 81 frames,
  16 fps, 50 denoising steps, BF16 DiT/FP32 VAE, fixed 2B revision. EasyCache
  threshold 0.10, warmup 30, tail 2, max-consecutive-reuse 1; TeaCache,
  TaylorSeer, QKV/linear-attention fusion, and compile disabled. This is one
  agent-selected operating point, not a new kernel implementation.

## Direct and source-reported timing

| Prompt | Process wall (s) | Warmup-excluded request (s) | Source denoise (s) | Source decode (s) |
| --- | ---: | ---: | ---: | ---: |
| p01 | 181.237 | 96.39 | 90.0554 | 5.6288 |
| p02 | 181.028 | 95.99 | 89.6647 | 5.6268 |
| p03 | 180.414 | 95.98 | 89.6529 | 5.6276 |
| p04 | 178.152 | 95.98 | 89.6620 | 5.6200 |
| p05 | 180.335 | 96.03 | 89.7000 | 5.6299 |
| Sum | 901.165 | **480.37** | 448.7350 | 28.1331 |

Original runner's `benchmark.json.total_s=480.37` equals all five `request_s`
values; timing scope excludes model load, internal one-step warmup, and video
export, identical to frozen dense. Five `generate_call_s` total 531.6s.
Five process restarts are a disclosed A100/SGLang site adapter: source log
timestamps bound repeated module load from `Loading pipeline modules...` to
`Loaded scheduler` at approximately 40, 40, 41, 39, 40 seconds respectively
(whole-second stamps); one-step warmups report 9.19, 9.12, 9.08, 9.11, 9.05s.
Other cold-path/setup time is not assigned to an invented subphase. Progress
bar 25/50 reflects order-2 display updating; R06 logs list 50 logical EC steps
per prompt. The five videos each probe as 832×480, 81 frames, 16/1 fps.

The local collector preserved `per_prompt`, `export` (80 frames, 16/prompt),
`timing_scope`, and `includes_video_write=false`. Its benchmark SHA is
`84c67a70374e3553242f408ee158ec53191e9b7c34d709a010c6ee0a9a29c86d`;
the untouched remote runner benchmark was
`2dfedb8ad1a2fb4ed68729a8fe3c42adcc1cb66309d5c3c69e5d5a98a5787972`.
The early local ordered 80-frame **path-inclusive** SHA-list digest from
`sha256sum outputs/frames/f_*.png | sha256sum` was
`9c79c0c48b90d45b38987bcafc7cc7f8f83259c7aacc6ebfdf598138f5d2fd31`.
The identical command after native local `plan_eval` still returns that exact
digest. The separate quality worker hashes only each frame's 64-hex SHA plus
newline, excluding the path; its content-only digest is
`4c9f96ea9e749688ea335e705731b3b41c7460313902edae800824fdd670717c`.
These are distinct algorithms over the same ordered 80 frames, not evidence
of a frame rewrite. No individual pre-evaluation frame-SHA list was preserved;
the identical pre/post path-inclusive aggregate is the available comparison.
five video SHAs remain in `benchmark.json.export.videos` and match local files.
Original small run files before any later native reverify were byte-snapshotted
with receipt SHA `696fb737b568316208d9032933bcd0794950fbc018b99ff46ecc9c7e7359edda`
under `artifacts/phase3/pre_native_verify_snapshots/20260914T065944.571295Z/`.

## Passive eight-UUID resource window

Read-only analysis used `scripts/analyze_phase2_observer.py` on exact remote
monotonic admission→finish; selected raw observer stdout SHA
`d3f37ad741add3b5230eb52920ec30312a5698f523e1b05051d7b494a01ad6fa`,
910 boundary rows, 906 in-window sample starts. **902.6163s observed**, 3.7559s
unobserved at two >1.5s p01 sampling gaps; zero error/incomplete-UUID rows.
Gap start pairs are located at 06:42:21.238512→22.491815Z (same-host monotonic
1.891271s) and 06:42:24.356392→25.559222Z (1.864581s). UTC is only a join
label, not the duration clock.

| UUID / index | App-present estimate (s) | >100 MiB estimate (s) | NVML util-equivalent estimate (s) | Sampled peak memory |
| --- | ---: | ---: | ---: | ---: |
| `GPU-cad55364-e014-c35f-337e-db899afd1c6a` / 0 | 774.244 | 773.255 | 524.258 | 23,466 MiB |
| Each of the other seven authorized UUIDs / 1–7 | 0 | 0 | 0 | 0 MiB |

GPU0 sampled mean utilization over covered time 58.082%, peak 100%; these are
NVML device samples, **not exact SM busy time**. The full single-GPU assignment
lasted 906.372s even while no app was sampled during load/process transitions;
it is distinct from 774.244s app-present and 524.258 util-equivalent seconds.
GPU0 first >100 MiB sample was +24.5352s after admission; preceding +23.5351s
sample had 3 MiB, so CUDA occupancy onset is interval-bounded, not a point.
The seven other cards' unallocated time is measured, but not automatically a
scheduler failure under per-candidate single-GPU/five-sequential-prompt semantics.

Visible remote worker-process lineage CPU **sampled lower bound**: user
2629.82s + system 9702.10s; peak summed RSS 14,165,561,344 bytes, write I/O
64,913,408 bytes. Short-lived descendants can be missed, and NVML app PIDs
are not joined to this `/proc` namespace. Local Cache/Master Agent CPU before
the separate 06:47:06Z local passive observer is unknown; Codex command/API
wall is a different measure.

## Quality boundary

The original immediate collector produced `quality.status=blocked_quality`,
LPIPS `baseline_frame_missing`. The Executor subsequently ran native local
`plan_eval --baseline-frames`, changing that intermediate reason to LPIPS
`dependencies_missing` (local Torch/LPIPS absent) and Gemini `tool_missing`;
neither state is a quality FAIL or PASS for R06. The authorized exact offline
aligned LPIPS backfill has not run. Cache attempt9's only actual `--image`
attachments were R02, so its later R06 frame-view/PASS self-report is not
evidence that its Codex session saw R06 pixels. Do not promote R06 to a
quality-gated frontier from 480.37s or montage/metadata-only description.
