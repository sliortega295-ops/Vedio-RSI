# Phase 3 R02 CPU quality transport — first evaluation

Status: **LPIPS_COMPLETE / QUALITY_BLOCKED / GPU_NOT_ARMED**. This is the same
restricted site transport and unmodified archived collector/judge/plan_eval as
the [R01 receipt](r01_cpu_quality_receipt.md); it is not a Cache frontier or
DELIVERY decision. R02 was enabled only after coordinator review of R01.

## Exact run and time

The real seq0002 Cache R02 run is
`repo/output/experiments/sana2ba100-cache_ca-0002/worktree/runs/20260913-102721-cache_ca_r02_teacache_t010`.
Remote read-only preflight before refresh checked 80 aligned baseline/R02
frames, five exported MP4s, five-prompt timing scope and 251.95 s sum, plus
fixed runner/config, archived collector/judge/plan_eval, package/backbone,
and interpreter hashes. R02 frame-content digest was
`e427746e6aaf5d3d512ba4a7cc73789b3161025dc5890d3791fd1beefe45df37`;
dense digest was `728ddacbaa5c9656cf7952e60c835192de70f64a5db8c903f192fd148ed4c964`.
Remote pre benchmark SHA256 was
`e7cf36c4ac5b5b3880a991aa653ab8a8dddf90a2c75cacff5e3314b5933a1579`.
Worker/guard/interpreter hashes are unchanged from R01; host transport SHA256
after exact R02 admission was
`c1d4aeb8b2e3c05cdb7970ff0099fa143246f5a2216a96b5843cac86b15c3a9f`.
The ten route unit/negative tests passed again. Host broker PID 1391772
started 2026-09-14 03:19:04 UTC after three queues had zero pending and zero
unresolved `.processing`; heartbeat matched the PID/start. The Cache worktree
used `PYTHONDONTWRITEBYTECODE=1 python3 -m orchestration.site_quality_client R02`.
Typed request ID `4e0e534350f247c5a6ceefaf5b2e6764`, request SHA256
`d9670e24756d842a8ef153e6c340ea9d5695689e5af8c18cc897475ad8057475`,
response SHA256 `c13c0700bd0f2ad0acc34dc8a94034a188760f08553a54ef0460a3dd012831cc`.

Host transport 2026-09-14 03:19:17.278–03:19:36.379 UTC had monotonic wall
19.101 s. Remote worker 03:19:17.678–03:19:35.478 UTC had remote monotonic wall
17.800 s and worker-process CPU 9.722 s; no cross-host monotonic subtraction.
Remote preflight 0.731 s; archived collector 14.777 s, direct-child CPU
user/system lower bounds 320.868/16.040 s; archived `plan_eval --no-gemini
--no-refresh-collection` 0.223 s, direct-child CPU user/system lower bounds
0.031/0.022 s. Original LPIPS runs inside the collector and lacks a separate
direct timer; its time is not fabricated. Three subprocesses loaded the
network-deny guard, with zero blocked network attempts.

## Original metric and gates

All 80 existing aligned frames remained intact, 16 per prompt; the original
stratified-plus-worst selection scored **42 pairs**, distributed p01–p05 as
**7/15/6/6/8**. Original LPIPS mean 0.307370, max 0.452535. The full ordered
pair indices and per-prompt mean/max are in the raw remote receipt. Request
times 50.53, 50.32, 50.35, 50.36, and 50.39 s sum to **251.95 s**, equal to
post-collector `total_s`; the five-prompt request-only timing scope remains.
Collection reports `frames.status=existing,count=80`.

Original collector quality remains **`blocked_quality`** with
`nvidia_gemini:api_key_missing`. Original `plan_eval` verdict has
`baseline_total_s=null`, `speedup=null`, and `tier=null`; this site-model
profile did not itself carry the frozen denominator. These blockers were not
silenced or reinterpreted as quality PASS. The archived lite path and stricter
graph/reviewer wording have a documented contract tension; Cache Executor and
Master must make their original visual/quality decisions, not the site shim.

## First pre/post chain

Host pre snapshot:
`artifacts/phase3/pre_quality_snapshots/host/R02/20260914T031917.274203+0000_1391772/`,
host transport receipt SHA256
`019d7eed2b8c31e8f9fd550cff1bc01a7a62ffe3abb6467209528ffe389c4d75`.
Remote attempt:
`repo/output/experiments/sana2ba100-cache_ca-0002/worktree/runs/20260913-102721-cache_ca_r02_teacache_t010/phase3_quality_attempts/20260914T031917.678076+0000_2896179/`,
receipt SHA256 `0d3519251e0509b997b7245c93ecc1934ddd4bd7660b558c9021c81d9d19efc0`,
network audit SHA256
`489a8eae14a8d6e12886688ad491d3707d9a846577cf640e68767bbfb26552c6`.

| Local real-run file | Host pre SHA256 | Host post SHA256 |
|---|---|---|
| `outputs/benchmark.json` | `f84052900331a508fca5a2332a9537d99ea03d2aa97cb537b85132324844aac2` | `9a8b5c30db3f5230a08a71606f265b50bdb78f1fe4f0d027ea5dcde1823132dd` |
| `outputs/quality.json` | `9fbdc7f38dcf9ae999e2b3d3446a6e5cf43b83bf03685651ce277b01f6cc3365` | `5c6a6a224a68ed6578081fecba1a21a5b119f0d916657573090939c4ffd44e9a` |
| `outputs/collection.json` | `e2c3d45b957b6261c0cad322c82327fb91972bd80cfa095cc1b328c7ab3b1b71` | `350074a97c8e0cfc3ecd048a136d8a2b6553e983c553bed6e541eb5d62ebfbc6` |
| `assess_verdict.json` | `f73fd60644911fe254680b79e1d373d38e75e0fc14a41fbd54512e241fb8930b` | `299aea891e61a084300c82a3fa391133c46c2648a19108aab010ec3f93091171` |

Remote post benchmark SHA256 `9a8b5c30db3f5230a08a71606f265b50bdb78f1fe4f0d027ea5dcde1823132dd`.
Remote pre quality/collection/verdict did not exist. The remote collector
changed metadata (`2b3a1cf1...`→`b31fbb83...`); host metadata was not synced.
No byte-idempotent repeat is assumed: the current initial-prehash allowlist
fails closed on another R02 assessment until this exact post-hash and receipt
chain is explicitly accepted and pinned. No GPU authorization, Cache DELIVERY,
or integrated run resulted from this operation.
