# Outer acceptance spot-check: R9 FFN temporal-linear candidate

Observed after the 2026-09-13 14:27:56 UTC run exit by the primary acceptance conversation. Three local collected PNGs were opened with the image-view tool from `repo/output/experiments/sana2ba100-kernel_aw-0002/worktree/runs/20260913-140334-kernel_compile_bf16_max_autotune_ffn_temporal_linear/outputs/frames/`:

| Extracted index | SHA256 | Limited visible content |
|---|---|---|
| `f_002.png` | `14af0df9d8e022944001fc445dbe46b0091845f060a8c6db289aa95c16dbe609` | Coherent sunlit forest portrait. |
| `f_040.png` | `a2501e4925ce3dbd9e95ce14d4c556f943e75fc4075a1fd810b1b609874dd1a8` | Coherent sunlit bedroom. |
| `f_080.png` | `284d631c7bc835991988b1770f0bbf0769a8d455450572e7f9ae7150ca616d21` | Coherent panda eating noodles. |

These three indices span different prompts in the five-prompt collection, not first/middle/last temporal frames of one video. This is only a small nonblank/semantic-authenticity spot-check. It does **not** establish video motion, quality parity, all-prompt quality, the Executor's claimed image viewing, or archived Master independent verification. It cannot promote R9 to a validated frontier or replace the Cache LPIPS/visual gate.
