# R07 visual transport material — rendered, not attached

At 2026-09-14 08:07 UTC, the project-local vetted renderer made a
hash-locked R07-versus-frozen-dense image set for a possible *future* native
Cache resume. This is not a quality verdict and **none of these R07 images
has been input to Cache9 attempt9 or Master11**. Their current R07 pixel
review claims remain externally UNSUPPORTED.

Renderer wrapper `scripts/phase3_render_r07_visuals.py` SHA
`4c2850248838890237b8b26c3a6f033dcdad47db9b0859600dae9be279a377e6`
pins vetted core SHA
`7e5e00a76e7baaf7f67bcfd5a4ac1c1db6df76f389d5f2b88c43ea2456b67231`,
the original R07 LPIPS receipt SHA
`f275f3740a99a2367d1a2d7dae04e6e19b3c9c5408fea196072cfbc33c521356`,
R07 80-frame digest
`46faed6d9430a46aad6c7986cd882c55673d7b07e0fd81c018204e3b91cc4b54`,
and frozen dense 80-frame digest
`728ddacbaa5c9656cf7952e60c835192de70f64a5db8c903f192fd148ed4c964`.
The fixed original-judge maximum selected frame per prompt is
`[16,32,42,49,65]`. R01 appears only as the renderer's already-vetted
second-round anchor and must **not** be attached for R07 assessment.

`preflight` passed source pins. `prepare` wrote one new, non-overwritten
render manifest (retained locally: `../../artifacts/phase3/cache_image_transport/render_r07_v1/render_manifest.json`)
SHA `b230288de5fe000b216c2873660951564747d10b9e42da23d3c727fa5a681e5c`.
Subsequent independent `verify` passed **20 images / 45,066,177 bytes / 384
pixel-checked tiles**; its status is `verified_not_attached`. Only the ten
R07 images (five prompt overview/detail pairs, **22,629,701 bytes**) are
eligible for a later exact attachment manifest. Each overview contains all
16 sampled times per prompt, candidate in the upper two rows and dense in
the lower two. Original-resolution detail shows candidate left/dense right,
with first/middle/last plus the highest selected-LPIPS frame; detail frame
indices by p01–p05 are `[1,8,16]`, `[17,24,32]`, `[33,40,42,48]`,
`[49,56,64]`, `[65,72,80]`. Every rendered tile was reopened and checked
against its pinned source (or deterministic fixed LANCZOS shrink).

This is finite sampled visual coverage, not every original video pixel or
temporal motion. A future exact `codex exec --image` attachment needs a
separately approved attempt/session manifest and observable API event;
mere generation or `montage` of these files is not evidence the Agent saw
their pixels. No active session code or candidate was edited to render them.
