# Phase 3 R01 CPU quality transport — first evaluation

Status: **LPIPS_COMPLETE / QUALITY_BLOCKED / GPU_NOT_ARMED**. This is a site-only
transport to the unmodified archived collector, LPIPS judge, and `plan_eval` on
the real seq0002 Cache R01 run. It is not a Cache frontier or DELIVERY decision.

## Exact execution and provenance

- Cache worktree request: `PYTHONDONTWRITEBYTECODE=1 python3 -m orchestration.site_quality_client R01`, from the exact seq0002 Cache worktree. Host broker PID 1386974, started 2026-09-14 03:15:05 UTC, source SHA256 `c253839a7321f67ff3e7da1cdb66d4d165af2f16ebaf5a03733ba404d73ff38f`. Typed request ID `5b0b6b8f68ee4eab8293484aeebd2128`, request SHA256 `54acb5bb01484f3441b1bcbe8723f5165a1e9547bb30d9bebb8a3d44e6ac1e19`, response SHA256 `7a50ab0b0860393a934774f3a1c9586789a505bad6c12cc5182ce76e1eca39ae`.
- Trusted remote worker SHA256 `11b45d660e26323f5723aaa26366674a7e204e9df3927d1a0468f3069b349e9d`, network-deny guard SHA256 `f2d1a27030f0456380f410d4bb7e118e628d543730718f74c038ddf292e71fe2`, remote Python executable SHA256 `7c9e72ea1f06138ed5f52f49894c62f29e51e302f0d02f8fb18490e90e994331`. Worker and guard were the only transferred code files (project PTY archive SHA256 `4d81dfc14d00e4c3e240885e2d0b0525c59141700f602c19adcbf1e0230e7a63`, three tar members including the guard directory). Remote read-only R01 preflight passed all fixed collector, judge, package, backbone, baseline/candidate frame, video, config, and benchmark hashes.
- Original collector SHA256 `34d85bf04eb5ee603c6c05964d0137a426c6710e9cd828ae04b54b66c05e158f` is the archived file with only five site timing fields preserved, as separately approved. Original LPIPS judge SHA256 `f68f7c72f31ec8018176e7b9be7047eba97524b73053842d99ba69ed3e1570bc`; original `plan_eval.py` SHA256 `a06d9e8f845cd269cd80f3b085c5c03675129f145801ed77e9fe1e5d48ad4840`. Exact `lpips==0.1.4` wheel SHA256 `fd537af5828b69d2e6ffc0a397bd506dbc28ca183543617690844c08e102ec5e`, bundled linear weights `df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0`, and AlexNet backbone `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02` were fixed before evaluation. No evaluator weight crossed hosts.
- The Phase 2 authorization and first-job budget files did not exist; the remote 8 UUID snapshot immediately before execution had 0 MiB/0% and no compute apps. This typed quality operation did not call a GPU submit route. Phase 1 terminal report, gate receipt, and six-hour window JSON retained SHAs `bea9932b...`, `1d161cc5...`, and `4d981007...` respectively.

## Measured stages and result

Remote worker 2026-09-14 03:15:55.441–03:16:13.035 UTC, monotonic wall 17.594 s, worker process CPU 9.711 s. Host typed transport 03:15:54.861–03:16:14.256 UTC, host monotonic wall 19.395 s; do not subtract host and remote monotonic clocks. Remote preflight 0.731 s; archived collector 14.532 s, direct-child CPU user/system lower bounds 328.275/15.472 s; archived `plan_eval --no-gemini --no-refresh-collection` 0.246 s, direct-child CPU user/system lower bounds 0.045/0.010 s. The original LPIPS judge runs inside the collector and is not separately timed by this wrapper; no invented split is assigned to it.

The collector used all 80 existing aligned frames (16 from each of five prompts), then its original stratified-plus-worst selector scored **41 pairs**, not all 80: p01/p02/p03/p04/p05 = **7/15/6/6/7**. LPIPS mean 0.173654, max 0.264209. The full ordered pair indices, per-prompt mean/max, and raw judge values are in the remote receipt. Three subprocesses emitted `guard_loaded`; `blocked_network_attempts=[]` and the network audit has three lines. The network guard is a site transport adaptation, not a replacement quality metric.

The five request times remain 76.62, 76.24, 71.88, 71.91, and 78.53 s; sum = **375.18 s** = post-collector `total_s` (floating representation 375.17999999999995). Timing scope still excludes model load, internal one-step warmup, and export. Collection reports `frames.status=existing,count=80`; no re-extraction or video/frame sync occurred. LPIPS judge `status=complete,result.status=ok,n=41`. Collector quality remains `blocked_quality` with `nvidia_gemini:api_key_missing`; the original `plan_eval` verdict has `baseline_total_s=null`, `speedup=null`, `tier=null`, so it does not independently establish objective/tier. Neither the LPIPS result nor the site route is a quality-gate PASS. Master visual and original independent verify remain outstanding.

## Immutable first-evaluation chain

Host pre snapshot: `artifacts/phase3/pre_quality_snapshots/host/R01/20260914T031554.857957+0000_1386974/` (all local old bytes), transport receipt SHA256 `37acf98993516869a36863afbe01aaa14c83aca8795de7c551ac71c0b5d0c92f`. Remote first attempt: `repo/output/experiments/sana2ba100-cache_ca-0002/worktree/runs/20260913-095257-cache_ca_r01_easycache_t010/phase3_quality_attempts/20260914T031555.440781+0000_2895069/`, receipt SHA256 `935c31120ce70c3e2079450d6f0dbf039325abf4bc491fee04d170d656963179`, network audit SHA256 `2d9c31512aba87fec4b01b603175ff8177774c018b2eb2b75900d6528ada9bdc`.

| Local real-run file | Host pre SHA256 | Host post SHA256 |
|---|---|---|
| `outputs/benchmark.json` | `9e18f8a4a7f9be54886c7290d16d3e4a2eb86c67dc6b37d1a22420462aafe111` | `a7e74304d275f298b34590944a0887ad24057b155452a4fcb0cedc95dd34b0db` |
| `outputs/quality.json` | `a49d87a87ff9b98bf33be56aa149e2a68ea788953cfff596340061128225a197` | `d4e0df1c06bfed90ae381f59f933e0313653b6ddfd86e17f57404db464ab5370` |
| `outputs/collection.json` | `ea6db12c54ea2ab7d60e20d8ad1105e9e682e69075cdf567e11fb5b2d3b4370a` | `e86c1522245eea680fb18532e66e5d1a72f0ab958cb2d1f7b8ecd2c15c46f81a` |
| `assess_verdict.json` | `d8767370d252a5f52b8b881f237a3429938d0fa338f5fea4cb026e1b364e9a43` | `49eeb0ccf6bb18001ae4320cf99731b1c9e26d39e0f11f813180a41d4f0d8bfc` |

Remote pre benchmark SHA256 `ecf0598168343475536d7929042dcc2ae123b1df708ed66fb8b7d5a918353f1f`, remote post benchmark SHA256 `a7e74304d275f298b34590944a0887ad24057b155452a4fcb0cedc95dd34b0db`. Remote pre quality/collection/verdict did not exist; the pre/post snapshots and receipt preserve that distinction. Remote collector also changed its metadata (`977c5712...`→`a16e91ef...`), while host metadata was intentionally not synced and remains its pre-evaluation original. No assumption of byte-idempotent recollection is made. Current strict prehash allowlist **rejects a second assessment**; before any Master reassess, the exact post hashes and first receipt SHA must be explicitly pinned as the trusted predecessor under coordinator review. Arbitrary new benchmark/quality bytes are never accepted.

R02 remains host-disabled until coordinator review. No Cache DELIVERY or integrated run was produced by this CPU assessment.
