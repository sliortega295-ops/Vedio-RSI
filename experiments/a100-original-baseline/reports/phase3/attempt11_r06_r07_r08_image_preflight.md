# Cache attempt11 R06/R07/R08 image transport preflight (not activated)

UTC 2026-09-14 09:26. Cache attempt10 ended with a real exit-0 receipt at 09:19:25.896958Z; no Cache DELIVERY exists. Master attempt11 remains active without image input. This packet creates only a new **prepared** attempt11 image route. It does not approve the manifest, change the live goal-session adapter/broker, start an Agent, submit GPU work, or make a visual verdict.

## Source and rendered-image identity

| Set | Render manifest SHA256 | Images | PNG bytes | Pixel tiles | Status |
| --- | --- | ---: | ---: | ---: | --- |
| R06/R07 versus frozen dense | `09921001565a907ed6c44b1680bba64713929cce776051eb18d99d911ccd3427` | 20 | 45,103,871 | 384 | rendered, not attached |
| R08 versus frozen dense | `020336e94a679a50caa60ff68ad9070be0481b5c92d649ba1dc872cc640e6d38` | 10 | 23,727,128 | 194 | rendered, not attached |
| Combined, fixed order R06→R07→R08 | two manifests above | 30 | 68,830,999 | 578 | prepared only |

The R08 renderer source is `scripts/phase3_render_r08_visuals.py` SHA256 `168850e4945e024b78be23a96c7724578134837c52af3b28f785d4ca838c07ad`; it pins the 80 candidate frames, frozen dense frames, original LPIPS receipt `e83d0c0b31e4b5d770154f6633b1e7212291c5df23945d2ca7c114d0e30caa19`, and the deterministic selection rule. Every prompt has one 16-frame chronological overview and one original-resolution first/middle/last plus highest-selected-LPIPS detail. Candidate and dense columns/rows are fixed. Each R08 PNG and source tile was reopened and pixel-compared to the hash-locked source. This is image provenance, not an Agent viewing event or video-wide quality conclusion.

The new combined attachment is `artifacts/phase3/cache_image_transport/attachments/sana2ba100-cache_ca-0002-attempt-11-r06-r07-r08.json`, SHA256 `c77781407ae516ca4145af3506f7f6c8ef9cad43308bccfc7e4dc6e89dd0412c`, `status=prepared`. The older R06/R07-only `attempt-11-multi.json`, SHA256 `3fc0d5faf6cfa28854d2fded49555504a9b9d89b660c3c3714051f504cd20ef8`, remains untouched and prepared. Master11 has inspected that **older** path; it must not be silently treated as the combined route.

## Validator and CLI preflight

The new inactive validator `repo/tools/symposium/site_phase3_multi_image_r06_r07_r08.py` SHA256 `05f5c6a99b0b0f06b565ce7f1984a1e07a3eea6618c1c21a9e682f03a0c2e61d` is restricted to `sana2ba100-cache_ca-0002`, attempt11. It pins the prior R06/R07 validator source SHA, both render manifests, R08 renderer and LPIPS receipt, exact 80-frame candidate/dense digests, 30 PNG hashes/paths/order, and 68,830,999 bytes. It fails closed unless the combined attachment status is approved; `allow_prepared=True` is a dry-run-only branch. The older prepared R06/R07 attachment is used solely as a source-check fixture inside the combined validator; it does not independently grant approval.

`repo/tests/test_site_phase3_attempt11_image.py` SHA256 `955deb698c35ad5f7828148065735a90df3893dc17f95e90486f3b12ed94bf95`: 13/13 PASS, including real R08 pixel verifier PASS and actual verifier rejection on mocked bad PNG/source hashes, prepared rejection, wrong session/attempt, bad render/source/receipt/PNG checks, and an approved temporary fixture under `/tmp/lyy-experiments/SolAgent/phase3-image-transport-tests/`. No retained PNG/source was changed. `git diff --check` PASS.

The local Codex CLI advertises `-i, --image <FILE>...`. Static construction of the same `codex exec -s workspace-write -C <Cache worktree> --image <30 exact paths> --json -o <last> -` command produced 30 paths in the expected order and 68,830,999 bytes; **Codex was not invoked with these images**. Its API payload/transport limits and actual pixel consumption remain unproven. If the real attempt11 rejects the 30-image payload, preserve the raw failed attempt and do not claim any round viewed; split only into newly approved exact attempt numbers, without downscaling or silently dropping prompts.

## Activation request, not yet executed

The live goal-session adapter SHA256 `ff0f89cd9faf4610165adcd9cded54f3485ca4f5bcb9197658c7a18f4e3345fc` still supports attempt10 only. Proposed minimal production change: add a fixed path/SHA constant for the new validator and an exact `name==sana2ba100-cache_ca-0002 and attempt==11` branch *before* legacy fallback in `image_inputs()`. The branch checks local source SHA and resolved import path, then calls the validator; failure never falls through. Tests must cover actual adapter `image-dry-run --allow-prepared` 30 paths, real prepared rejection, bad source/hash, and no change to attempt10. Only after coordinator approval: single-field combined manifest `prepared→approved` with pre/post SHA; fresh no-pending/unresolved/no-active-Cache check; scoped goal-broker reload and heartbeat/source SHA; non-override adapter CLI dry-run; then let archived Master retry native `resume_executor --mode continue` with a dated factual correction pointing explicitly to the **new** approved manifest (old file preserved). No outer visual judgement or algorithm suggestion is supplied.

The first real evidence threshold is `telemetry/local_events.jsonl` `agent_api_begin` with 30 `--image` paths and the approved combined-manifest SHA, followed by actual Cache prompt-specific pixel descriptions. Master independent viewing remains a separate future exact-image attempt. No Cache or integrated DELIVERY can be accepted on this preflight alone.
