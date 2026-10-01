# R05 exact offline LPIPS route: staged, not executed

At 2026-09-14 06:15 UTC, the site-only typed CPU quality route was extended for **only** the real seq0002 Cache R05 run `20260914-054324-cache_ca_r05_taylorseer_t0180_w10_o1`. This is an exact dependency/quality measurement route, not a candidate selection, GPU submission, visual review, Cache pass, or independent Master verification. The remote trusted quality worker is **still the previous R04 version** until reviewed transfer; R05 judge has **NOT_RUN**.

The remote raw pre-collector `outputs/benchmark.json` SHA256 is `e88373b0bd6d438f3efe453b0b4a605e5fe9ceb944fa41a4665ff5d30ff3d323`. The local native-collected post-state benchmark is `8476c72990b29e4a179a4a925894404d24864caed49b178ca22ccc8e33478411`; `total_s=371.36` is exactly the five request values `74.51+74.19+74.18+74.20+74.28`, under the frozen request-only timing scope. It retains five 832×480/81-frame/16-fps videos and 16 exported frames per prompt. The video SHAs in that benchmark match all five actual local **and remote** MP4s. Both local and remote have exactly `f_001..f_080`; SHA-line frame digest `36b9df309d8b3ab3ca47c3e82b42bf3c2da002af493336ac01702d83b6f1e3d0` matches across hosts. Dense frame digest remains pinned independently by the original worker. The existing local collector says LPIPS dependencies missing; this route has not yet produced a numeric R05 LPIPS result.

| Exact local run predecessor | SHA256 |
| --- | --- |
| `launch.sh` | `392f725844f3606360c56624126f169e98c5a376f0c5e17cd36f911d50ca487e` |
| `manifest.resolved.toml` | `3d1cddb42c2f04fb2c387c81fa00d9dee14b1ff6c3432b74c5ba37f61b585342` |
| `metadata.json` | `598904fecc09e1d4d264f6787e99a8b8ccc778c1679792af289684ff8a5e387e` |
| `outputs/benchmark.json` | `8476c72990b29e4a179a4a925894404d24864caed49b178ca22ccc8e33478411` |
| `outputs/run_config.json` | `0c98e68c6aa33fbb2a6cc1f3d0041108206439efb915af97557aa92ebcfded51` |
| `outputs/quality.json` | `40019c7439645fcf97541fd5938476e865601189800dc28b377a73aa526273b8` |
| `outputs/collection.json` | `a01a58ef32d7eff8fe9412fca71b5ba6d8dae2c0acd7528967c6680ed391a5ce` |
| `assess_verdict.json` | `b2ec773ba16490ef6552a85465fd6c6bb6d777cf9b7b776a5a110251406db868` |

Staged source SHA256: `repo/orchestration/site_quality_worker.py` `525fd4dd8ee58718d4c3d62e80be78224b57e5b40e51a02e7435b7b692863a96`; `repo/orchestration/site_quality_transport.py` `c6723366de4a7c31a114de0798abc9914c3c383670158dfdabb09b4a323ee608`; `repo/tests/test_site_quality_route.py` `414f4344e84b8a7af420acc1fe93ca9dbd4358eca013df1b86ac6665b3a88573`; one-file transfer wrapper `scripts/sync_r05_quality_worker.py` `50397113abeae6fe07f30b9a5707c7e6bd8d9189f23dc1721b2f7f7238f39a58`. Wrapper read-only dry-run at `06:17:32.045508Z` observed remote old worker SHA `3088b1fb...` and exact new-project backup target, without transfer. Local exact R05 validation passes; combined quality/reverify route tests **24/24 PASS**, `git diff --check` **PASS**. Earlier R01–R04 prehashes still fail on a second assessment unless explicitly re-approved with a posthash/receipt chain.

Before any R05 remote CPU judge, recheck every current local and remote pin, the three fixed queues and broker process/children, synchronize only the reviewed trusted worker with old-file backup and SHA verification, and scoped-restart only the host job broker. The worker must run the original archived collector/LPIPS and plan_eval with CUDA hidden, network socket-denied, fixed project-local wheel/linear/AlexNet hashes, and pre/post snapshots; preserve the raw Gemini blocker. A later true R05 visual gate requires separately hash-locked R05 pixel attachments to the original Agent, not the current R02-only Cache9 session.
