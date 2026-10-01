# Phase 2 GPU pre-arm read-only check — NOT ARMED

Observed 2026-09-14 04:28–04:29 UTC, after Cache8 requested a future newly authorized R03 full run. This receipt does **not** authorize, start, or reserve a GPU window; every time-sensitive check must be repeated immediately before arming.

- Exact target hostname: `job-c6cfcde5-db4b-4527-8eb8-7518319a0277-master-0`; target login in its container reports uid 0 and this experiment's remote project/three allowed run parents are `root:root 755`. Only this previously authorized node was queried. The remote `nvidia-smi` snapshot showed the eight fixed UUIDs in expected order, each 0 MiB and 0% GPU utilization; `--query-compute-apps` produced no rows. This is a momentary idle observation, not future ownership proof.
- Host/remote trusted `repo/orchestration/site_job.py` SHA256 `ef686fb4131e2846286ba474ba4a76e09413ee156f8a1b587f58e6f513b30f1d`; trusted `repo/orchestration/site_observer.py` SHA256 `4c541a43ac76d65416839b009fc07e3eeb9f83e5274c01a6931bc86fa237bf83`. Local `site_remote.py` SHA256 `f34c1c1b78a60848c521450590cf15cdc63e2b57668e19de346001bfe4ee0029` pins exactly those two remote hashes. The authorized model and SGLang overlay were not moved or edited in this check.
- Remote `site_observer.py --phase phase2 audit` returned `{"status":"not_started"}`. Remote trusted `site_job.py budget-status` exited 1 with `RuntimeError: phase2 GPU authorization is absent`; both remote `state/phase2_authorization.json` and `state/phase2_first_gpu_job.json` were absent. The corresponding local files and the real `telemetry/phase2/observer/` directory were absent. This confirms fail-closed NOT_ARMED, not readiness for GPU submit.
- Current host typed site-job broker PID `1391772`, started `2026-09-14T03:19:04.393347Z`, exact coordinator `site_job_broker.py serve` command; current source SHA256 `c253839a7321f67ff3e7da1cdb66d4d165af2f16ebaf5a03733ba404d73ff38f` (later than the older pre-quality-route readiness SHA; it must be reviewed as the current version). The source mtime precedes that daemon start. State/heartbeat agreed on PID/start, with heartbeat observed `04:29:40.514517Z`. Its coordinator/Kernel/Cache queues had pending `.json` 0/0/0; historical `.processing` 17/372/51 all had matching responses, unresolved 0/0/0. The existing Phase 2 CPU window tests passed 11/11 from the correct repo cwd; `git diff --check` passed. An initial invocation from the project parent failed only because `orchestration` was not on Python's import path; it did not access GPU/remote.
- The three exact allowed run parents are coordinator `repo/runs`, Kernel seq0002 worktree `runs`, and Cache seq0002 worktree `runs`. Their child-name union was 27 locally and 19 remotely; `remote_only=[]`. This snapshot includes files/CPU probes as conservative excluded names, not only GPU jobs. Do not reuse it as authorization without a fresh host+remote union at arm time:

```text
20260913-084751-sana_video_2b_a100_dense-cpu-preflight
20260913-091006-sana_video_2b_a100_dense
20260913-094250-cache_ca_r01_easycache_t010
20260913-094634-cache_ca_r01_easycache_t010
20260913-094731-kernel_preflight_profile
20260913-095257-cache_ca_r01_easycache_t010
20260913-095426-kernel_preflight_profile
20260913-102305-kernel_compile_default
20260913-102324-kernel_compile_default
20260913-102721-cache_ca_r02_teacache_t010
20260913-104624-kernel_compile_bf16_agg
20260913-110805-kernel_compile_bf16_max_autotune
20260913-114414-kernel_compile_bf16_autotune_qkv
20260913-121315-kernel_compile_bf16_default_qkv
20260913-123046-kernel_compile_bf16_default_ffn_inverted_linear
20260913-125812-kernel_compile_bf16_max_autotune_ffn_inverted_linear
20260913-133020-kernel_compile_bf16_max_autotune_rope_bf16
20260913-140334-kernel_compile_bf16_max_autotune_ffn_temporal_linear
20260913-143436-kernel_compile_bf16_max_autotune_ffn_all_linear
20260914-022502-sana_video_2b_a100_dense-site-phase2-unarmed-negative
README.md
site-transport-broker-cpu-01
site-transport-broker-cpu-02
site-transport-broker-cpu-03
site-transport-broker-cpu-04
site-transport-codex-sandbox-01
site-transport-cpu-smoke-01
```

Remaining release gates: Cache8 natural endpoint/Agent-owned R03 launch intent; current host broker source and queue review; fresh remote UUID/app/owner and excluded-name union; real remote Phase2 1 Hz observer started with at least two complete eight-UUID samples, fresh PID/heartbeat and measured overhead; then separate host+remote authorization files with identical SHA and short arm expiry, reviewed by root. The six-hour cap starts on the first **accepted remote site-job admission**, whether or not compute later starts; host pre-submit cutoff reserves 21 minutes, and remote worker enforces the hard deadline. Phase 1 budget/reports remain untouched. No GPU launch can occur from this receipt alone.
