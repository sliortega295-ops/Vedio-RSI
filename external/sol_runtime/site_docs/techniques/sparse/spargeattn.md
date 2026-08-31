# SpargeAttn

**SpargeAttn** is a universal **training-free** sparse attention that accelerates any
model (language / image / video) with no fine-tuning. It uses a two-stage online
filter: a first stage quickly predicts the attention map to skip whole blocks of the
QK^T·V matmuls, and a second stage applies an online softmax-aware filter that skips
further products after the softmax. Because the prediction is computed on the fly, it
adapts per-input without offline profiling or weight changes.

**Why it fits here.** Drop-in and training-free — the same class as PISA/SVG: it only
changes the attention backend, never the weights.

**Paper:** [SpargeAttention: Accurate and Training-free Sparse Attention Accelerating Any Model Inference (arXiv:2502.18137)](https://arxiv.org/abs/2502.18137) · [code](https://github.com/thu-ml/SpargeAttn)
