# DraftAttention

**DraftAttention** is a **training-free**, plug-and-play sparse attention for video
DiTs. It reshapes the long query/key sequences back into per-frame feature maps and
applies 2D average-pooling to build a low-resolution **draft** attention map. That
cheap draft exposes which blocks matter — spatially within a frame and temporally
across frames — and the full attention is then computed only on the selected blocks.
No profiling and no weight changes; the draft is recomputed each step.

**Why it fits here.** Pure backend-level acceleration, training-free, open-source.

**Paper:** [DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance (arXiv:2505.14708)](https://arxiv.org/abs/2505.14708) · [code](https://github.com/shawnricecake/draft-attention)
