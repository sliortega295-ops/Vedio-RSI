# Cache image transport design — static, NOT DEPLOYED

Status: CPU-only design at 2026-09-14 03:39 UTC. No current Agent session has received these images. In particular, Master attempt 9 had `transport_images=[]`; its `montage` and pixel-review claim do not constitute a visible-image input. This is a disclosed site transport adapter for the archived orchestration-lite visual gates, not a change to either Agent's quality criteria.

## Pinned inputs and selection

Use only the frozen dense `runs/20260913-091006-sana_video_2b_a100_dense` and seq0002 Cache's completed real R01 `runs/20260913-095257-cache_ca_r01_easycache_t010` and R02 `runs/20260913-102721-cache_ca_r02_teacache_t010`. Each has exactly `outputs/frames/f_001.png` through `f_080.png`, 16 aligned frames per prompt. The original offline judge's ordered-frame SHA256 digests are dense `728ddacbaa5c9656cf7952e60c835192de70f64a5db8c903f192fd148ed4c964`, R01 `ebebb661eed466d9668fd1a39e3ffc8f855bcb3bd70743f1f7701d21a7a4828c`, and R02 `e427746e6aaf5d3d512ba4a7cc73789b3161025dc5890d3791fd1beefe45df37`; locally recomputed digests match all three. Do not use unrelated run images or an external visual verdict.

For each candidate and prompt, attach an overview with all 16 time-ordered candidate frames next to the same 16 dense frames. The Master-created `/tmp/sana-cache-visual.xwIK52/r01-pN-pair.png` and `r02-pN-pair.png` meet that temporal overview shape (8-by-2 thumbnails per run, candidate above dense) but are only 220×127 thumbnails. They must be copied into project artifacts with before/after SHA checks **after Master attempt 9 ends**, and never mislabelled as images that attempt 9 saw.

Also construct full-original-resolution (832×480 pixels per frame, lossless PNG) candidate/dense detail comparisons. The fixed selection is the first, middle, and last aligned frame in each prompt (local indices 1, 8, 16), plus the prompt's largest-LPIPS selected pair from the already frozen Phase 3 original-judge receipt; duplicate indices are included once. The LPIPS-derived choice is a coverage rule only: do not overlay its numeric value, a speed label, or an outer quality judgment on images or Agent feedback. Frame indices to include:

| Prompt | R01 detail frame indices | R02 detail frame indices |
|---|---|---|
| p01 | 001, 008, 016 | 001, 004, 008, 016 |
| p02 | 017, 024, 032 | 017, 022, 024, 032 |
| p03 | 033, 039, 040, 048 | 033, 040, 042, 048 |
| p04 | 049, 056, 064 | 049, 056, 064 |
| p05 | 065, 072, 080 | 065, 072, 080 |

Before any attachment: independently hash every selected source frame and resulting image; write a manifest binding run ID, candidate/dense role, prompt, frame indices, frame SHA256, image SHA256, dimensions, renderer command/source SHA, and exact target session/attempt. Reject missing/symlinked/wrong-digest frames, unexpected image size or path, and stale attempts. Use only project artifacts or `/tmp/lyy-experiments/SolAgent/<run-id>/` for disposable construction. Preserve original `/tmp` paths as incident evidence, without deleting or silently moving them.

## Delivery boundary

After current Master and Cache invocations end, propose an exact allowlist in the host Codex transport for one new seq0002 Cache attempt and one later, separate seq0002 Master attempt. A CPU/CLI dry-run must prove the resulting `codex exec --image` command includes exactly the pinned project images and that its raw `agent_api_begin` and final receipt list identical paths/SHA. Cache's own visible-image gate precedes any Cache `DELIVERY.json`; the archived Master then independently verifies a delivered candidate's objective evidence and views its own image inputs. The graph-only reviewer wording in `workflow/cache_ca/nodes/codex_executor/cache_scope.md` is not an executed orchestration-lite node. Neither Agent may infer visual PASS from `montage`, `identify`, file hashes, image dimensions, or an outer review alone.

The image prompt must state that overview sheets are downsampled and full-resolution details cover only selected frames; neither can prove every frame, motion continuity, or broad paper-level quality. No candidate algorithm suggestion or outer visual assessment may be added. GPU authorization remains separate and absent.
