# ViDiT-Q

**ViDiT-Q** is a PTQ scheme aimed specifically at **video** diffusion transformers,
where naive W8A8 already degrades quality because activation distributions shift
strongly across both *timesteps* and *tokens*. ViDiT-Q makes the quantization
granularity match that variation: token-wise and timestep-wise dynamic quantization
for activations, plus a metric-decoupled mixed-precision allocation that spends bits
where the perceptual loss is largest. It reaches W8A8 (and W4A8 on the weights) on
text-to-video DiTs with near-lossless quality.

**Relevance here.** This is the closest published recipe to our setting — the same
timestep/token-sensitivity problem is why our NVFP4 line keeps the first/last denoise
steps in BF16 (step-selective) rather than quantizing uniformly.

**Paper:** [ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation (arXiv:2406.02540)](https://arxiv.org/abs/2406.02540) · [code](https://github.com/thu-nics/ViDiT-Q)
