# SVDQuant (Nunchaku)

**SVDQuant** is a 4-bit post-training quantization that handles outliers with a
low-rank branch: it shifts outliers from activations into weights, absorbs the weight
outliers with an SVD low-rank component (kept high-precision), and quantizes the
residual to 4-bit. The **Nunchaku** engine fuses the low-rank branch into the low-bit
GEMM to avoid extra memory traffic.

**In this repo.** Provided (`runtime/layers/quantization/nunchaku_linear.py`,
`configs/nunchaku_config.py`); not on a default video line.

**Paper:** [SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models (arXiv:2411.05007)](https://arxiv.org/abs/2411.05007) · [code](https://github.com/nunchaku-ai/nunchaku)
