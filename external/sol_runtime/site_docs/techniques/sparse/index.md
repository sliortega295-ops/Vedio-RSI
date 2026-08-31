# Sparse attention

Video self-attention over tens of thousands of tokens is quadratic and dominates
latency. Sparse-attention backends restrict each query to a subset of key/value
blocks. They differ in how the important blocks are chosen.

**Scope: training-free + open-source only.** Every method here is a *plug-and-play
backend swap* — it changes only the attention computation, never the model weights, and
has a public implementation. Trainable sparse attention (which fine-tunes or co-trains
the diffusion model — e.g. VSA, VMoBA) is intentionally excluded.

## Backends in this repo

| Backend | Selection idea | Paper |
|---|---|---|
| [PISA](pisa.md) | piecewise: exact selected blocks + centroid remainder | [2602.01077](https://arxiv.org/abs/2602.01077) |
| [Sparse VideoGen / SVG2](svg.md) | spatial/temporal head split; semantic-aware permutation | [2502.01776](https://arxiv.org/abs/2502.01776) / [2505.18875](https://arxiv.org/abs/2505.18875) |
| [STA](sta.md) | hardware-aware 3D sliding tile window | [2502.04507](https://arxiv.org/abs/2502.04507) |
| [Others](others.md) | block-sparse / sparse-linear / RainFusion / LASER | — |

Selected via `--attention-backend` / `--component-attention-backends`.

## Related training-free methods

Open-source, training-free backends from the literature (not wired onto a default line):

| Method | Selection idea | Paper · code |
|---|---|---|
| [SpargeAttn](spargeattn.md) | two-stage online filter, model-agnostic | [2502.18137](https://arxiv.org/abs/2502.18137) · [code](https://github.com/thu-ml/SpargeAttn) |
| [DraftAttention](draftattention.md) | low-res pooled "draft" map guides block selection | [2505.14708](https://arxiv.org/abs/2505.14708) · [code](https://github.com/shawnricecake/draft-attention) |
| [DiTFastAttn](ditfastattn.md) | calibrated spatial/temporal/conditional redundancy | [2406.08552](https://arxiv.org/abs/2406.08552) · [code](https://github.com/thu-nics/DiTFastAttn) |

All are **lossy** (algorithm-level approximation).
