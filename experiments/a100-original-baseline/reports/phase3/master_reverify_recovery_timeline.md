# Master Cache reverify recovery: native evidence and remaining gate

All UTC dates are 2026-09-14. This is an outer site/control audit of the
original seq0002 Master/Cache flow, not a replacement for the archived
Master's decisions. The Cache `DELIVERY.json` remains SHA256
`0940aca6c4e90502b808f5d3441f49a7b3d0d1ac1c2f10a4170e6e4ab2920e85`
through this recovery; it names R05 and R08. No composed run was submitted
before the Phase2 new-submit cutoff 10:32:50.441475703Z.

| UTC | Native/site event | Meaning |
| --- | --- | --- |
| 10:19:28–53 | Master12 original `verify_delivery.py` exit 1, R05 fresh objective pass, R08 `plan_eval_reverify_no_lpips` | Genuine failure, not a Cache algorithm verdict. R08 remote worker rejected before collector because the site preflight inherited the *pre*-quality `metadata.json` SHA `81d36e…` rather than approved current post-quality SHA `3a609f…`. Host failed transport receipt is `artifacts/phase3/pre_reverify_snapshots/host/20260914-083326-cache_ca_r08_taylorseer_t0180_w30_o1/20260914T101951.403256+0000_2054361/transport_receipt.json`. |
| 10:23:19 | Recoverable R05 second-call guard | First fresh R05 verdict SHA `fd0de26e…` moved within the same real run to `phase3_reverify_preflight/20260914T102319.216294Z_b316cb493f7444f1a3126ab13d5e5e43/`; guard receipt SHA `cd5d4a0156103c80ac8e451ab00f352ba266c1d60822410f3512f7bd9325c7d4`. Fixed path absent before a repeated native verifier call. |
| 10:23:59–10:25:11 | Master12 called native `resume_executor` on Cache12, then both sessions were scoped-stopped via goal broker | The Master blamed Cache's R08 point on the site preflight failure and asked it to remove/fix R08. This feedback is incorrect and preserved, not edited. Cache12 raw stream SHA at stop `f97d915a…`; Master12 SHA at stop `65a35602…`. Goal-broker stop requests `28b2bb3eda3641509c26ce9c90072643` and `d6fb230c54ae4489ba5c23b08a2ca71e`. No new candidate/GPU job. |
| 10:25:54–10:26:03 | Exact site boundary repair | Trusted remote reverify worker old SHA `b57cfd74…` backed up and new SHA `6cc90d29303feee23a1a6aa3e3f99b7b46444c4a4708afbea78df7390fa61ff9` verified from one 4,411-byte code archive SHA `15a2d70…`; host typed broker restarted at PID 2078034 after all three queues had pending/unresolved 0. Host route SHA `23ebca5dc3931ad689032e38e803ac5f984841cb70f43b893e9100b95490077e`; 27 related tests passed after the later image-adapter update. No package/model/data download or GPU use. |
| 10:29:17 | Same Master identity, attempt13 API begin | Exact `--image` input had 20 SHA-pinned R05/R08 overviews/details, five prompts each, manifest SHA `0ffc6adef723ebb054599c4e7128f3b6bd6ba8cdbd16528f7357693d1249b867`; fixed `PLAN_EVAL_PYTHON` shim and trusted worker SHA recorded in `telemetry/local_events.jsonl`. Codex WebSocket reconnect/fallback 10:29:31–59 is API/control overhead. |
| 10:31:58–10:32:39 | Master13 original `verify_delivery.py` exit 0, `objective_ok=true`, empty issues | Both points generated **new** original collector + original `plan_eval --no-gemini` remote receipts, not stale host numerical files. R05 remote receipt SHA `b5c9897532832a64a4a0a690438a5ba39b6df15ff2afdb1156bfb78ef8abf8d3`; R08 remote SHA `63d62cb8503a97238aa82f6450011f473d8357e65ef780f64b12e9a77baf744f`. Each has collect exit 0, plan_eval exit 0, no blocked network attempts, five-prompt LPIPS max respectively `0.07933352887630463` and `0.026091445237398148`. Host transport receipts SHA `ecadb43824b9f70db485e84a08f873d52a820a2fb57626ca321b8d2483e4405d` and `3c532ce915aeb207c0589cb2370a86f016d9ef0c556bab3b0e74b018d0bae792` verify post-to-remote SHA synchronization. Raw collector Gemini blocker remains. |
| After 10:32:50 | GPU submit window closed | The Master stated no late composition submit. A full integrated result still requires a new authorized window, a real composed GPU run, the composed run's own original quality/visual gates, and `INTEGRATED-DELIVERY`; component verification alone cannot supply these. |
| 10:38:55 | Master13 self-reported zero-history boundary encounter | Its workspace-wide filename `find .` listed two seq0001 `sana_video.py` paths, and it states it did not open either. Raw command supports path-only exposure, not old-source/results reading; the incident is preserved in `BOUNDARY-INCIDENTS.jsonl`. It elected to stop source merging/submission/integration in this attempt. Its preceding diff was written to `/tmp/lyy-experiments-sana-kernel-cache.diff`, outside the prescribed project scratch subtree, separately recorded as a control-path deviation. |

Master13's actual attached-image comments distinguish the moving plate, panda,
forest portrait and two static-room prompts and state the sampled-frame limit;
that is materially stronger than Master11's unsupported `80/80` pixel claim
with `transport_images=[]`. It does **not** prove every frame/continuous
motion was viewed. The archived lite and graph-specific Cache visual contracts
remain distinct; no independent graph reviewer was run. No final composed
quality or authenticity PASS is asserted here.

Master13 naturally exited 0 at **10:40:53.059877Z**, native attempt receipt
SHA256 `88e218d11e7b16880cbe2d15cbb4c3a5708744eb5e1b402374cd50fede3e3833`,
API duration `695.738382053 s`, with 20 image inputs. Its final message marks
the component Kernel structural/objective and R05/R08 objective, sampled
visual and authenticity gates PASS, while explicitly stating that integration
was not run. The boundary incident log SHA256 is
`c4de16f8652f663848c3e0289e4ea5261ea61d0571432ab4718f62fdf628ce94`.
