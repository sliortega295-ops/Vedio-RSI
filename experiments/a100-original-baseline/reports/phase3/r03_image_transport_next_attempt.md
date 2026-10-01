# R03 visual input proposal for the next native Cache attempt

This is **PREPARED, NOT APPROVED OR ATTACHED**. Cache9 and Master11 are
still live, and their goal-transport code has not been hot-edited.

The existing vetted renderer core SHA
`7e5e00a76e7baaf7f67bcfd5a4ac1c1db6df76f389d5f2b88c43ea2456b67231`
was called through pinned wrapper `scripts/phase3_render_r03_visuals.py`
SHA `b4eb1d204f9bd631b8f13bd17eb290a7c7c45dd29d00405bdb4a95f9ccae97fa`.
Its dense/R03 inputs were checked against all 80 frame hashes and the
original R03 judge receipt SHA `26709c70…`. The renderer also includes
R01 because its unchanged verifier requires two five-prompt rounds;
future Agent attachment is restricted to R03 only. The render manifest
`artifacts/phase3/cache_image_transport/render_r03_v1/render_manifest.json`
SHA `c4edc006026b2195af60d293f18f8c88ae513f865e2c6e66ee85fc52e15e3f12`
passed prepare→verify with 20 images/386 source-matched tiles/46,075,682
bytes. These include 10 R03 images (five prompt overview and original-size
detail pairs) totaling 23,639,206 bytes. Selection is fixed to all 16
time-ordered frames per prompt at 208×120 overview and first/middle/last
plus highest LPIPS-selected frame at original 832×480 detail. No LPIPS
score or prior visual verdict is overlaid.

The exact next-Cache-attempt attachment manifest is
`artifacts/phase3/cache_image_transport/attachments/sana2ba100-cache_ca-0002-attempt-10.json`
SHA `b038f40d0bd735f223c6a6a7ebf6e2bde200a8de130cbdf2e3bc842c769f692f`.
It is status `prepared`, round `r03`, session
`sana2ba100-cache_ca-0002`, attempt 10, with ten ordered image SHA pins.
A static read-only comparison to the render manifest, image SHA, prompt
order, and byte count passed at 05:52 UTC. The current trusted goal
adapter still accepts only R01/R02 from `render_v1`, so a normal Codex
call cannot use this attachment yet.

After Cache9/Master11 naturally stop and exact next-attempt numbering
is known, proposed site-only adapter change: retain the old R01/R02
source pins; add `r03` as a second immutable render directory/manifest
SHA for exactly this Cache attempt; validate source-frame SHA, ten ordered
image SHA/size, approval flag, and no symlink or path escape before CLI
`--image` injection. Run prepared-mode CLI dry-run and fail-closed
negative tests, have coordinator review the exact diff and manifest,
then change only this manifest `prepared`→`approved` with pre/post SHA.
The archived Master must perform its own separate image-input review
after any genuine Cache DELIVERY. Neither rendering nor attachment
by itself makes Cache visual quality PASS, and sampled images do not
substitute for continuous-video review.
