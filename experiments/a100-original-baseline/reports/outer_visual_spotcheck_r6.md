# Outer acceptance spot-check: R6 FFN candidate

Observed 2026-09-13 around 12:55 UTC by the primary acceptance conversation, independently of the Executor's `visual_verdict.json`. Six local PNGs were opened with the image-view tool: extracted-frame indices 002, 040, and 080 from both the frozen dense output and the R6 collected output. These indices span different prompts in the five-prompt collection; they are not a temporal first/middle/last assessment of one prompt.

| Frame | Dense SHA256 | R6 SHA256 | Limited visual observation |
|---|---|---|---|
| `f_002.png` | `5d6b4322ccd202cfe9485f31d908bbccacba75686e1b867e734d393200c5b94c` | `d5883cc84100da0a88886cc54e449cc1466fe7308c5059ce42621355ab8da375` | Coherent portrait in a sunlit forest in both; visible small differences. |
| `f_040.png` | `9d05b7490021bd83e9393f444eb5891820ee830aeab1045d39c7510a8de78d3d` | `7acd8a3c739d1b88a22e61fc882be2e19b4a7c0ac48c5928a1eb460e3bbc18ea` | Coherent sunlit bedroom in both; furniture and wall details differ. |
| `f_080.png` | `70e197eddf5511f325011f4a8224f255138c0aeeb5ad8464f328a00630bf81ca` | `c84dfa953e85a206b0aa569b1490b0518e343d762039bb2c8e4fedcd1df238b6` | Coherent panda eating noodles in both; visible small differences. |

This only establishes that these six specific PNGs contain plausible, nonblank scenes. It does **not** verify video motion, all five prompts and seeds, numerical identity, perceptual quality, the Executor's claim to have visually inspected frames, or the archived Master's independent verification. It cannot promote R6 to a validated frontier or substitute for the Cache LPIPS/visual gate.

Dense PNGs: `repo/runs/20260913-091006-sana_video_2b_a100_dense/outputs/frames/`. R6 PNGs: `repo/output/experiments/sana2ba100-kernel_aw-0002/worktree/runs/20260913-123046-kernel_compile_bf16_default_ffn_inverted_linear/outputs/frames/`.
