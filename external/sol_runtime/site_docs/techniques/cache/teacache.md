# TeaCache

**TeaCache** (Timestep-Embedding-Aware Cache) decides when to skip by watching the
*model input*, not the output: it modulates the noisy latent by the timestep
embedding, accumulates the relative-L1 change of that signal across steps, and
recomputes only when the accumulator crosses a threshold — otherwise it replays the
cached residual. An optional polynomial rescales the per-step distance (calibration).

**In this repo.** Cosmos3's shipped cache line (`thr 1.15 / start 10 / max-continuous 3`,
plus first/last-step guards). Impl: `runtime/cache/cosmos3_teacache.py`,
`runtime/cache/ltx2_teacache.py`. Env: `SGLANG_COSMOS3_TEACACHE_{ENABLED,THRESH,START,MAX_CONTINUOUS_HITS,COEFFICIENTS}`.

**Paper:** [Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model (arXiv:2411.19108)](https://arxiv.org/abs/2411.19108) · [code](https://github.com/ali-vilab/TeaCache)
