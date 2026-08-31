# PISA (piecewise sparse attention)

**PISA** chunks Q/K/V into blocks, scores block pairs (top-k routing) augmented by a
K-variance proxy, runs exact attention on the selected blocks, and either
approximates the unselected remainder with block centroids
(`approx_remainder=true`) or drops it. This keeps the dominant interactions exact
while collapsing the long tail.

**In this repo.** LTX `fullopt` uses it on **stage 2 only** (`transformer_2`),
sparsity 0.9, block 64, route `score`, dense layers 0-1. Impl:
`runtime/layers/attention/backends/piecewise_attn.py`. Select via
`--component-attention-backends transformer_2=piecewise_attn` +
`--attention-backend-config piecewise_sparsity=…,piecewise_block_size=…,piecewise_route_mode=…`.

**Paper:** [PISA: Piecewise Sparse Attention Is Wiser for Efficient Diffusion Transformers (arXiv:2602.01077)](https://arxiv.org/abs/2602.01077)
