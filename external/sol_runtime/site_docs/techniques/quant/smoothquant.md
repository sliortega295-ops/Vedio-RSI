# SmoothQuant

**SmoothQuant** is a training-free building block, not a standalone diffusion method:
activations have a few large-magnitude channels that are hard to quantize, while
weights are flat and easy. SmoothQuant migrates that difficulty by scaling activations
down per-channel and folding the inverse scale into the preceding weights — the GEMM
is mathematically unchanged, but both operands become quantization-friendly (enabling
W8A8 / W4A8). It originated for LLMs and is now a standard preprocessing step inside
many diffusion / DiT PTQ pipelines (DiTAS, ViDiT-Q and others apply a smoothing pass
of this form before quantizing).

**In this repo.** The smoothing idea is the same one our ModelOpt-based NVFP4/FP8
calibration leans on (per-channel scale absorption before the low-bit GEMM).

**Paper:** [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models (arXiv:2211.10438)](https://arxiv.org/abs/2211.10438) · [code](https://github.com/mit-han-lab/smoothquant)
