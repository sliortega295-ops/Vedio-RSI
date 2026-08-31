# DiTFastAttn

**DiTFastAttn** is a **training-free** post-training attention-compression method for
diffusion transformers. It identifies three redundancies and caches/sparsifies each:
(1) **spatial** — many heads attend only locally, so use a windowed attention;
(2) **temporal** — neighboring denoise steps have near-identical attention output, so
reuse the previous step's output; (3) **conditional** — the conditional and
unconditional (CFG) passes are very similar, so share the computation. A short
calibration pass picks which compression applies per layer/step — no fine-tuning.

**Why it fits here.** Calibration-only (no training), open-source. Note it also overlaps
the [cache](../cache/index.md) family — its temporal/conditional reuse is step-skip-like,
while its spatial part is genuine sparse attention.

**Paper:** [DiTFastAttn: Attention Compression for Diffusion Transformer Models (arXiv:2406.08552)](https://arxiv.org/abs/2406.08552) · [code](https://github.com/thu-nics/DiTFastAttn)
