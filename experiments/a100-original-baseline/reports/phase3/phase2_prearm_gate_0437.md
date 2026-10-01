# Phase 2 immediate pre-arm gate — read-only PASS, authorization still absent

Checked 2026-09-14 04:35–04:37 UTC, after Cache8 and Master10 had exited. This receipt records a *momentary* safe pre-arm state; it does not claim GPU use or guarantee the node remains idle. The exact proposed authorization is `reports/phase3/phase2_auth_proposal_0434.md` SHA256 `9e8d455a5df2f98b299d21f6fff479182793f9bdc7d3451d410770aff22405d0`, which root reviewed in principle. No phase2 authorization or first-job budget file existed on host or remote at this gate.

- Remote hostname returned `job-c6cfcde5-db4b-4527-8eb8-7518319a0277-master-0`. `nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu` showed exactly the eight preassigned UUIDs in index order 0–7, each 0 MiB and 0% at the read; `--query-compute-apps` returned no rows. No second A100 node was accessed for GPU work.
- Host and exact remote trusted coordinator `repo/orchestration/site_job.py` SHA256 both `ef686fb4131e2846286ba474ba4a76e09413ee156f8a1b587f58e6f513b30f1d`; observer script both `4c541a43ac76d65416839b009fc07e3eeb9f83e5274c01a6931bc86fa237bf83`; host transport `repo/orchestration/site_remote.py` `f34c1c1b78a60848c521450590cf15cdc63e2b57668e19de346001bfe4ee0029` pins both. Host broker source SHA `c253839a7321f67ff3e7da1cdb66d4d165af2f16ebaf5a03733ba404d73ff38f` predates the currently running daemon PID 1391772, started 03:19:04.393347 UTC, exact coordinator-owned `site_job_broker.py serve`; state and heartbeat agreed on PID/start, heartbeat at 04:36:35.710238 UTC. Its only allowed operations are submit/status/sync/assess_quality. Coordinator/Kernel/Cache queues had pending 0/0/0, historical processing 17/372/51 and unresolved 0/0/0; they were not cleared.
- Remote Phase2 observer PID 2899723 was already running before any new GPU work. Fresh 04:36:50.993595 UTC sample was the 348th; every sample contained all eight UUIDs, error 0, incomplete 0, gaps >1.5s 0, last age 1.392s at audit, mean query overhead 0.4248s, observer CPU 22.9495s across 346.9971s (~6.61% of one CPU core). The observer data remain remote at `telemetry/phase2/observer/` and must continue until the new window ends.
- Remote model snapshot directory `/share/project/shuchenweng/yongyan_liu/Experiments/SolRolloutBench/20260912-a100-eight/models/sana-video-2b/db5f398b13ca086d09a50ce156c20527773841b1` and SGLang overlay directory `/share/project/shuchenweng/yongyan_liu/Experiments/SolRolloutBench/20260912-a100-eight/envs/sglang-overlay` both existed at their pinned paths (`root:root`, mode 755). They were inspected only; no model/weight transfer or write occurred. The host broker pins those paths and offline env (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`) for GPU runs. No dependency download is part of arming.
- Phase1 final report, terminal gate, and six-hour-cap SHA256 still respectively `bea9932b8d6986dd4df12988ef9827976dd1d3c48a2c9405da58088f4469f7d4`, `1d161cc53a1c857fa4a4a8a806d3fcc8f097f26ce672491e315fe2e758329369`, `4d981007ae12dc2abab40a1e4f7a7de78e693f64a1b307096ac2278c258d9b4a`. Remote Phase1 `state/first_gpu_job.json` SHA remains `f5502224c63b9bf21b6e93bc33921bc8f66b9281780a83d7b22214fc5684b262`. Neither Phase1 state nor terminal report is a source of Phase2 budget.
- Correct-cwd Phase2 control tests had passed 11/11 and `git diff --check` was clean before this gate. No new control source changes occurred between those tests and this read-only gate.

The three exact permitted run-parent directories (`repo/runs`, seq0002 Kernel worktree `runs`, seq0002 Cache worktree `runs`) were **relisted** on both host and remote at 04:35–04:36 UTC. Host count 27, remote count 19, union 27, remote-only empty. The complete union, conservatively excluding even CPU probes/README, matched the root-reviewed proposal byte-for-byte:

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

The native Cache8 final state is `resumable_no_delivery_cpu_only`, its autonomously prepared R03 TeaCache manifest/config are site-tagged Phase2 but not launched, and Master10 ended PARTIAL with no Cache DELIVERY or integration. Arming would permit only a future Agent-created, hash-locked bundle passing the broker; it is **not** an outer submission or a quality/frontier verdict. At the first accepted remote `site_job start`, the separate Phase2 six-hour receipt begins whether or not compute starts. New submit closes 21 minutes before that receipt's deadline; remote worker checks a hard deadline every second. If the observer, GPU ownership, exclusion names, broker, or trusted SHA changes before actual file write, stop and re-audit rather than use this stale snapshot.
