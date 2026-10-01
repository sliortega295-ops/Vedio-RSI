# Proposed Phase 2 GPU authorization — NOT WRITTEN / NOT ARMED

Prepared 2026-09-14 04:34 UTC for root read-only review. This is a proposal in a report, **not** `state/phase2_authorization.json`. No Phase 2 GPU job has been submitted; the Phase 1 budget and terminal receipts remain unchanged. Immediately before activation, refresh all volatile checks and replace the proposal if any excluded name changes or the short expiry becomes unsuitable.

The current trusted remote and local `orchestration/site_job.py` SHA256 is `ef686fb4131e2846286ba474ba4a76e09413ee156f8a1b587f58e6f513b30f1d`; local `site_remote.py` SHA256 is `f34c1c1b78a60848c521450590cf15cdc63e2b57668e19de346001bfe4ee0029`. Host broker source SHA256 is `c253839a7321f67ff3e7da1cdb66d4d165af2f16ebaf5a03733ba404d73ff38f`; remote/local observer SHA256 is `4c541a43ac76d65416839b009fc07e3eeb9f83e5274c01a6931bc86fa237bf83`. The observer began at 04:31:03 UTC (PID 2899723), ahead of any Phase 2 GPU job. Its first complete eight-UUID sample was 04:31:04 UTC. No authorization file or Phase 2 first-job budget file exists yet. The frozen SANA model and dependency overlay remain read-only; no model transfer, download, or optimization recipe change is part of this authorization.

The three allowed run parents were listed again on both host and exact remote at 04:31:54 UTC: local 27 names, remote 19 names, union 27, remote-only empty. Their complete union is the conservative exclusion list below. Source of truth must be rechecked at arm time. The proposed `arm_expires_unix_ns` is 2026-09-14 05:00:00 UTC (`1789362000000000000`); if this time is reached before the first accepted remote `site_job start`, authorization cannot open the window and must not be stretched silently.

Proposed exact JSON value for **identical host and remote bytes** after approval and a fresh gate:

```json
{
  "schema_version": 1,
  "phase": "phase2",
  "status": "armed",
  "remote_project": "/share/project/shuchenweng/yongyan_liu/Experiments/SolAgent/20260913-a100-original-baseline",
  "remote_host": "job-c6cfcde5-db4b-4527-8eb8-7518319a0277-master-0",
  "max_window_s": 21600,
  "submit_reserve_s": 1260,
  "gpu_uuids": [
    "GPU-cad55364-e014-c35f-337e-db899afd1c6a",
    "GPU-a904c6f6-6410-dc0f-c7c3-efc068004617",
    "GPU-79ab3e5e-e382-1f9c-bede-1900666e9283",
    "GPU-bab43db0-a051-0419-39f7-e33effa9445b",
    "GPU-44bddf94-7411-b3ad-f195-fdbdd12fa73e",
    "GPU-af8dba1f-f74e-6a55-bb9e-6c8961edbd6e",
    "GPU-5edaa614-d5f8-8a3f-11ec-b128f8d029f3",
    "GPU-01fff32a-1dd6-93a7-1aae-60d378382961"
  ],
  "excluded_run_names": [
    "20260913-084751-sana_video_2b_a100_dense-cpu-preflight",
    "20260913-091006-sana_video_2b_a100_dense",
    "20260913-094250-cache_ca_r01_easycache_t010",
    "20260913-094634-cache_ca_r01_easycache_t010",
    "20260913-094731-kernel_preflight_profile",
    "20260913-095257-cache_ca_r01_easycache_t010",
    "20260913-095426-kernel_preflight_profile",
    "20260913-102305-kernel_compile_default",
    "20260913-102324-kernel_compile_default",
    "20260913-102721-cache_ca_r02_teacache_t010",
    "20260913-104624-kernel_compile_bf16_agg",
    "20260913-110805-kernel_compile_bf16_max_autotune",
    "20260913-114414-kernel_compile_bf16_autotune_qkv",
    "20260913-121315-kernel_compile_bf16_default_qkv",
    "20260913-123046-kernel_compile_bf16_default_ffn_inverted_linear",
    "20260913-125812-kernel_compile_bf16_max_autotune_ffn_inverted_linear",
    "20260913-133020-kernel_compile_bf16_max_autotune_rope_bf16",
    "20260913-140334-kernel_compile_bf16_max_autotune_ffn_temporal_linear",
    "20260913-143436-kernel_compile_bf16_max_autotune_ffn_all_linear",
    "20260914-022502-sana_video_2b_a100_dense-site-phase2-unarmed-negative",
    "README.md",
    "site-transport-broker-cpu-01",
    "site-transport-broker-cpu-02",
    "site-transport-broker-cpu-03",
    "site-transport-broker-cpu-04",
    "site-transport-codex-sandbox-01",
    "site-transport-cpu-smoke-01"
  ],
  "arm_expires_unix_ns": 1789362000000000000
}
```

The proposed value follows the site adaptation schema in `orchestration/site_job.py:68–82`; the host broker also checks exact remote budget/auth SHA before GPU submit. Before any write to the actual state path: confirm this proposal with root; check that the Cache Executor has a real prepared Phase 2 launch need; repeat host/remote run-parent union, exact GPU UUID ownership and compute-app idle checks, observer audit (running PID/heartbeat, at least two complete samples, no stale/gap/error), host broker PID/start/heartbeat/source SHA, remote trusted site-job/observer SHAs, and absence of old/new budget contamination. Then write the same serialized JSON to host and remote state paths, compare SHA256, and verify both `budget-status` receipts. The six-hour limit begins on the first accepted remote `site_job start`, even if subsequent model loading fails; new submits close 21 minutes before that deadline, and the remote worker has a separate hard deadline. There is no authority to extend Phase 1.
